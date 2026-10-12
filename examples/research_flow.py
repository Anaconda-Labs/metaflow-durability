"""Durable web research with agents as function calls inside ordinary steps.

    start      planner agent splits the topic into subtopics; plain code searches and
                dedupes sources for each
    research   (foreach) the researcher agent reads the sources and writes a typed
                Brief, plain code checks its citations
    gather     joins the briefs
    review     plain code routes on the briefs: weak ones get new sources and go to
                `deepen`
    deepen     the same researcher agent reruns per weak brief
    synthesize synthesizer agent writes the report
    publish    renders the report as a card (side effects live in their own step)

Every agent carries `MetaflowDurability()`, so each task's model and tool calls are
journaled and shown live in the task's Agent Journal card.

On the first attempt, the first subtopic's second `fetch_page` call fails on purpose.
`@retry` restarts that task: finished model and tool calls are replayed from the
journal, and only the rest run live. Set SIMULATE_FETCH_FAILURE=0 to turn this off.

Requires TAVILY_API_KEY and a model (MODEL, e.g. in .env):

    pixi run -e examples python examples/research_flow.py run \\
        --topic "solid-state batteries for electric vehicles"
"""

import os
from collections import Counter
from typing import Any
from typing import Literal
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()  # loads .env if present, no-op otherwise

from metaflow import FlowSpec
from metaflow import Parameter
from metaflow import card
from metaflow import current
from metaflow import retry
from metaflow import step
from metaflow.cards import Markdown
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai import RunContext
from pydantic_ai.capabilities import PrepareTools
from pydantic_ai.messages import ModelResponse
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import UsageLimits
from tavily import TavilyClient

from metaflow_durability import MetaflowDurability

MODEL = os.environ.get("MODEL", "test")
# Bedrock ignores the `timeout` model setting; its boto client reads AWS_READ_TIMEOUT
# (default 300s, and botocore retries timed-out reads), so a stalled call can hang
# for many minutes. Fail fast instead and let @retry take over.
os.environ.setdefault("AWS_READ_TIMEOUT", "120")
SYNTHESIS_SETTINGS = ModelSettings(timeout=120, max_tokens=3000)
SIMULATE_FETCH_FAILURE = os.environ.get("SIMULATE_FETCH_FAILURE", "1") == "1"

_tavily = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY", ""))
# Soft budget: once a tool's budget is used up it is no longer offered to the model,
# which then has to write the brief from what it has. UsageLimits is the hard backstop.
TOOL_BUDGET = {"fetch_page": 3}
HARD_LIMITS = UsageLimits(request_limit=20)
MIN_VERIFIED_SOURCES = 2
_fetches = 0
_fail_this_task = False


class Plan(BaseModel):
    subtopics: list[str]


class Brief(BaseModel):
    subtopic: str
    summary: str
    key_points: list[str]
    sources: list[str]
    confidence: Literal["low", "medium", "high"]


# --- plain code: retrieval and checks, no model involved ---


def find_sources(query: str, limit: int = 6) -> list[dict[str, str]]:
    """Search, then keep the best result per domain."""
    results = _tavily.search(query, max_results=10).get("results", [])
    seen: set[str] = set()
    sources = []
    for r in sorted(results, key=lambda r: r.get("score", 0), reverse=True):
        domain = urlparse(r.get("url", "")).netloc.removeprefix("www.")
        if domain and domain not in seen:
            seen.add(domain)
            sources.append({"title": r.get("title", ""), "url": r["url"]})
    return sources[:limit]


def check_citations(brief: Brief, sources: list[dict[str, str]]) -> list[str]:
    """Drop citations the agent was never given; return the dropped URLs."""
    allowed = {s["url"] for s in sources}
    dropped = [url for url in brief.sources if url not in allowed]
    brief.sources = [url for url in brief.sources if url in allowed]
    return dropped


def is_weak(brief: Brief) -> bool:
    return brief.confidence == "low" or len(brief.sources) < MIN_VERIFIED_SOURCES


def research_prompt(topic: str, brief_for: str, sources: list[dict[str, str]]) -> str:
    listing = "\n".join(f"- {s['title']}: {s['url']}" for s in sources)
    return f"Topic: {topic}\nSubtopic: {brief_for}\n\nCandidate sources:\n{listing}"


# --- agent tools and budget ---


def fetch_page(url: str) -> str:
    """Fetch the readable text of a web page (truncated)."""
    global _fetches
    _fetches += 1
    if _fail_this_task and _fetches == 2:
        raise ConnectionError(f"simulated fetch failure for {url}")
    results = _tavily.extract(urls=[url]).get("results", [])
    text = results[0].get("raw_content", "") if results else ""
    return text[:6000] or "no readable content"


async def research_budget(
    ctx: RunContext[Any], tool_defs: list[ToolDefinition]
) -> list[ToolDefinition]:
    """Offer each tool only while it has budget left in this run."""
    used = Counter(
        part.tool_name
        for message in ctx.messages
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ToolCallPart)
    )
    return [t for t in tool_defs if used[t.name] < TOOL_BUDGET.get(t.name, 1_000)]


planner = Agent(
    MODEL,
    name="planner",
    instructions="Split the research topic into 3 focused, non-overlapping subtopics.",
    output_type=Plan,
    capabilities=[MetaflowDurability()],
)
researcher = Agent(
    MODEL,
    name="researcher",
    instructions=(
        "You get a subtopic and candidate sources. Read the most relevant ones with "
        "fetch_page, then write the brief. Cite only candidate URLs you actually read. "
        "Set confidence to low if the sources were thin or contradictory."
    ),
    output_type=Brief,
    tools=[fetch_page],
    capabilities=[MetaflowDurability(), PrepareTools(research_budget)],
)
synthesizer = Agent(
    MODEL,
    name="synthesizer",
    instructions=(
        "Write a concise research report in markdown from the briefs: an overview, "
        "one section per subtopic, open questions, and a sources list."
    ),
    model_settings=SYNTHESIS_SETTINGS,
    capabilities=[MetaflowDurability()],
)


class ResearchFlow(FlowSpec):
    topic = Parameter("topic", default="solid-state batteries for electric vehicles")
    force_deepen = Parameter(
        "force_deepen", default=False, help="Send every brief through deepen."
    )

    @step
    def start(self):
        subtopics = planner.run_sync(str(self.topic)).output.subtopics
        # retrieval happens here, not in `research`: search results change between
        # calls, and a retried agent step must see the same prompt to replay
        self.leads = [
            {"subtopic": s, "sources": find_sources(f"{self.topic}: {s}")}
            for s in subtopics
        ]
        self.next(self.research, foreach="leads")

    @retry(times=1, minutes_between_retries=0)
    @step
    def research(self):
        global _fail_this_task
        _fail_this_task = (
            SIMULATE_FETCH_FAILURE and self.index == 0 and current.retry_count == 0
        )
        lead = self.input
        self.sources = lead["sources"]
        # a failed fetch replays finished calls, then runs the rest live
        brief = researcher.run_sync(
            research_prompt(str(self.topic), lead["subtopic"], self.sources),
            usage_limits=HARD_LIMITS,
        ).output
        self.dropped_citations = check_citations(brief, self.sources)
        self.brief = brief
        self.next(self.gather)

    @step
    def gather(self, inputs):
        self.briefs = [task.brief for task in inputs]
        self.dropped_citations = {
            task.brief.subtopic: task.dropped_citations for task in inputs
        }
        self.next(self.review)

    @step
    def review(self):
        self.weak = [
            i for i, b in enumerate(self.briefs) if self.force_deepen or is_weak(b)
        ]
        self.deepen_sources = {
            i: find_sources(f"{self.briefs[i].subtopic} latest data and analysis")
            for i in self.weak
        }
        self.route = "deepen" if self.weak else "synthesize"
        print(f"weak briefs: {len(self.weak)} of {len(self.briefs)} -> {self.route}")
        self.next(
            {"deepen": self.deepen, "synthesize": self.synthesize}, condition="route"
        )

    @retry(times=1, minutes_between_retries=0)
    @step
    def deepen(self):
        for i in self.weak:
            weak = self.briefs[i]
            sources = self.deepen_sources[i]
            brief = researcher.run_sync(
                research_prompt(str(self.topic), weak.subtopic, sources)
                + f"\n\nA first pass found only this:\n{weak.summary}",
                usage_limits=HARD_LIMITS,
            ).output
            check_citations(brief, sources)
            self.briefs[i] = brief
        self.next(self.synthesize)

    @retry(times=1, minutes_between_retries=0)
    @step
    def synthesize(self):
        briefs = "\n\n".join(b.model_dump_json(indent=2) for b in self.briefs)
        self.report = synthesizer.run_sync(f"Topic: {self.topic}\n\n{briefs}").output
        self.next(self.publish)

    @card(type="blank", id="report")
    @step
    def publish(self):
        current.card["report"].append(Markdown(self.report))
        self.next(self.end)

    @step
    def end(self):
        print(self.report[:500])


if __name__ == "__main__":
    ResearchFlow()
