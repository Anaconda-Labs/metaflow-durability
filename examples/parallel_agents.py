"""Parallel agents with foreach and synthesize.

Fans out a research agent across multiple topics via foreach, gathers
results, then hands off to a synthesizer agent with different tools.

    start → research (foreach) → gather → synthesize → end

Each agent has scoped authority: the researcher only searches, the
synthesizer only compares. Metaflow foreach handles the parallelism.

Requires:
    pip install tavily-python python-dotenv

    # Local dev: put keys in .env
    echo 'TAVILY_API_KEY=tvly-...' >> .env

    # Remote: attach @secrets via --with or in code

Usage:
    pixi run python examples/parallel_agents.py show
    pixi run python examples/parallel_agents.py run
"""

import json
import os

from dotenv import load_dotenv

load_dotenv()  # loads .env if present, no-op otherwise

from metaflow import FlowSpec
from metaflow import Parameter
from metaflow import agent
from metaflow import step

from tavily import TavilyClient

_tavily = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY", ""))


# --- Research tools ---


def search(query: str) -> str:
    """Search the web for information on a topic.

    Args:
        query: The search query.
    """
    response = _tavily.search(query, max_results=5)
    results = response.get("results", [])
    snippets = [
        {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
        for r in results
    ]
    return json.dumps({"query": query, "results": snippets})


# --- Flow ---


class ParallelResearchFlow(FlowSpec):
    """Fan out research across topics, then synthesize a recommendation."""

    topics = Parameter(
        "topics",
        default="solar energy,wind energy,nuclear energy",
        help="Comma-separated list of topics to research.",
    )

    @step
    def start(self):
        self.topic_list = [t.strip() for t in self.topics.split(",")]
        self.next(self.research, foreach="topic_list")

    @agent(
        model="bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0",
        system_prompt=(
            "You are a research assistant. Use the search tool to find "
            "key facts about the assigned topic. Report what you find — "
            "do not analyze or compare."
        ),
        tools=[search],
        max_iterations=15,
    )
    @step
    def research(self):
        """Research one topic. Runs in parallel via foreach."""
        self.topic = self.input
        self.messages = [
            {"role": "user", "content": f"Research: {self.topic}"},
        ]
        self.next(self.gather)

    @step
    def gather(self, inputs):
        """Collect research results from all parallel agents."""
        self.findings = {inp.topic: inp.final_response for inp in inputs}
        self.next(self.synthesize)

    @agent(
        model="bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0",
        system_prompt=(
            "You are a senior energy analyst. You receive research findings "
            "on multiple energy sources. Compare them on cost, reliability, "
            "and environmental impact, then write a concise recommendation."
        ),
        max_iterations=1,
    )
    @step
    def synthesize(self):
        """Compare findings and produce a recommendation."""
        summaries = "\n\n".join(
            f"**{topic}**: {finding}"
            for topic, finding in self.findings.items()
        )
        self.messages = [
            {
                "role": "user",
                "content": (
                    f"Research findings:\n\n{summaries}\n\n"
                    "Compare these energy sources on cost, reliability, and "
                    "environmental impact. Recommend the best option for a "
                    "developing country."
                ),
            },
        ]
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    ParallelResearchFlow()
