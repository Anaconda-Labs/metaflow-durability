# metaflow-durability

> [!WARNING]
> **This project is a proof of concept.** There is no guarantee that it will be maintained, and APIs may change or disappear without notice. You are welcome to discuss it in the [github issue tracker](https://github.com/Anaconda-Labs/metaflow-durability/issues)

Agents in Metaflow steps, made durable. metaflow-durability has two independent parts; use either one, and neither needs the other:

- **`MetaflowDurability`**: a [pydantic-ai](https://ai.pydantic.dev) durable-execution capability. Your agent stays a plain pydantic-ai `Agent` that you call like a function inside any step. Every model call and tool call is journaled to the step's artifacts, and a retry replays the finished ones instead of paying for them again.
- **`@agent`**: a step decorator that runs its own LLM loop (via LiteLLM, an OpenAI-compatible client, or a co-stacked inference decorator) with plain Python tools. See [`@agent`](#agent-step-decorator) below.

## Quick start: MetaflowDurability

```
pip install "metaflow-durability[pydantic-ai]"
```

```python
from metaflow import FlowSpec, Parameter, retry, step
from pydantic import BaseModel
from pydantic_ai import Agent

from metaflow_durability import MetaflowDurability


class Brief(BaseModel):
    summary: str
    sources: list[str]


researcher = Agent(
    "anthropic:claude-sonnet-4-5",
    name="researcher",
    output_type=Brief,
    tools=[web_search, fetch_page],
    capabilities=[MetaflowDurability()],
)


class ResearchFlow(FlowSpec):
    topic = Parameter("topic")

    @retry(times=1)
    @step
    def start(self):
        # a failed fetch replays finished calls, then runs the rest live
        self.brief = researcher.run_sync(self.topic).output
        self.next(self.end)

    @step
    def end(self):
        print(self.brief.summary)


if __name__ == "__main__":
    ResearchFlow()
```

If `fetch_page` fails partway through, `@retry` restarts the step. The searches, fetches, and model calls the first attempt finished are replayed from its journal, and only the rest run live. Nothing else in the flow changes: no extra decorator, and the agent code is ordinary pydantic-ai.

## How it works

- **The agent is a function call inside a step.** Code can run before and after it, a step can call several agents or the same agent several times, and plain code can route on the agent's typed output.
- **Every durable operation is journaled.** pydantic-ai's durable-execution builder hands each model request, tool call, tool discovery, and workspace call to `MetaflowDurability`. Each is recorded in the task's `agent_journal` artifact, keyed by agent name, which run of that agent it was in the task, and its position in that run.
- **Replay needs the name and the inputs to match.** On a retry, an operation is returned from the previous attempt's journal only if its name and a hash of its inputs (prompt, messages, settings, tool arguments) match. The first mismatch ends replay for that run and it continues live, so a changed prompt never gets a stale answer.
- **Nothing to attach.** The metaflow-durability extension adds a small step decorator to every step that hosts the journal and saves it as an artifact, on success or on failure. Outside Metaflow (tests, notebooks) the agent behaves as a plain agent.

## Requirements and constraints

- The agent needs a name: `Agent(name=...)`, or `MetaflowDurability(name=...)`, which takes precedence. The name keys the journal, so renaming it breaks replay of runs recorded under the old name.
- Two different `Agent` objects with the same name in one step raise an error. The same agent called again is fine.
- Tools run one at a time inside a durable run.
- A retry re-runs any tool call whose result was not yet recorded. Keep irreversible side effects (posting, publishing) in their own step.
- A hard kill (OOM, `SIGKILL`, preemption) loses that attempt's journal; exceptions, including `@timeout`, do not.

## Using a local or custom model endpoint

Any OpenAI-compatible endpoint works, including one served inside the task by an inference decorator such as `@llamacpp` or `@vllm`: wrap its client in a pydantic-ai model.

```python
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


def local_model(base_url: str, model_name: str) -> OpenAIChatModel:
    client = AsyncOpenAI(base_url=base_url, api_key="unused")
    return OpenAIChatModel(model_name, provider=OpenAIProvider(openai_client=client))
```

There are two ways to hand that model to a durable agent:

**Build the agent in the step.** Use this when the endpoint only exists once the task is running. The name keeps the journal stable across attempts:

```python
@retry(times=1)
@step
def research(self):
    researcher = Agent(
        local_model(base_url, "granite-4.0-h-tiny"),
        name="researcher",
        capabilities=[MetaflowDurability()],
    )
    self.brief = researcher.run_sync(self.topic).output
```

**Register it, then pick it per call by name.** Use this when the endpoint is known when the agent is defined:

```python
researcher = Agent(
    "anthropic:claude-sonnet-4-5",
    name="researcher",
    capabilities=[MetaflowDurability(models={"local": local_model(base_url, "granite")})],
)

researcher.run_sync(prompt, model="local")  # registered model
researcher.run_sync(prompt)                 # the agent's default model
```

Passing an unregistered model instance at call time (`run_sync(model=...)` or `agent.override(model=...)`) raises an error inside a durable run, because pydantic-ai durable execution only accepts models it can find by name.

## Inspecting a run

The journal is an ordinary artifact, kept for every attempt:

```python
from metaflow import Task

task = Task("ResearchFlow/12/start/3", attempt=1)
for entry in task["agent_journal"].data["researcher"][0]:
    source = "replayed" if entry["replayed"] else "live"
    print(source, entry["name"], f"{entry['duration_ms']:.0f} ms")
```

Every step also gets an **Agent Journal** card:

- It updates live while the task runs: one row per operation, marked live or replayed, with duration, token counts, and a detail column.
- It ends with the agent's final result, rendered as markdown or highlighted JSON.
- A banner shows which attempt you are looking at: yellow while running, green for the final attempt, red for an attempt that failed. A failed attempt's card is re-rendered with its complete journal before the task exits.
- On a retried task, the final attempt's card is listed first (on S3, GCS, and Azure datastores).

```
python myflow.py card view <step_name>
```

## Examples

The examples use pydantic-ai's test model by default and a real model when `MODEL` is set (for example `bedrock:us.anthropic.claude-sonnet-4-6`).

- [`triage_agents.py`](examples/triage_agents.py): three agents chained in one step, with plain code routing on their typed output and the side effect in its own step.
- [`retry_replay.py`](examples/retry_replay.py): a tool fails on the first attempt; the `end` step prints which operations each attempt replayed and which ran live.
- [`foreach_research.py`](examples/foreach_research.py): the same agent across a foreach, then a synthesizer agent after the join.
- [`research_flow.py`](examples/research_flow.py): plain-code retrieval and citation checks around the agent, routing to a `deepen` step on weak briefs, a soft tool budget with pydantic-ai's `PrepareTools`, and a `publish` step that renders the report card. Needs `TAVILY_API_KEY`.

```
pixi run -e examples python examples/research_flow.py run
```

## `@agent` step decorator

`@agent` is the other half of metaflow-durability and does not use pydantic-ai. The decorator runs the whole LLM loop inside the step, before the step body: call the model, dispatch tool calls, append the results, and repeat until a final response or `max_iterations`.

```python
from metaflow import FlowSpec, Parameter, agent, step


class CityComparisonFlow(FlowSpec):
    prompt = Parameter("prompt", default="Compare Tokyo and Paris by population density.")

    @step
    def start(self):
        self.messages = [{"role": "user", "content": self.prompt}]
        self.next(self.compare_cities)

    @agent(
        model="anthropic/claude-sonnet-4",
        system_prompt="Call tools for data, then synthesize.",
        tools=[city_stats, calculator],
        max_iterations=10,
    )
    @step
    def compare_cities(self):
        self.next(self.end)

    @step
    def end(self):
        print(self.final_response)
```

### Providing a model

Resolved in this order:

1. **An OpenAI-compatible client:** `@agent(client=OpenAI(base_url=..., api_key=...), tools=[...])`. Any object with `chat.completions.create()` works.
2. **A co-stacked inference decorator.** On [Anaconda AI Orchestration](https://www.anaconda.com/platform/ai-orchestration), stack `@llamacpp` or `@vllm` under `@agent` and leave out `model=`; the decorator finds the local OpenAI-compatible endpoint. No API keys, and it works air-gapped. Order: `@agent`, then the inference decorator, then `@resources`, closest to `@step`.
3. **A [LiteLLM](https://docs.litellm.ai/docs/providers) model string:** `@agent(model="bedrock/...", litellm_kwargs={"aws_region_name": "us-west-2"}, tools=[...])`. `litellm` is imported only on this path.

### Tools

Tools are plain functions with type hints and Google-style docstrings; schemas are generated with [openai-function-calling](https://github.com/jakecyr/openai-function-calling). The function name is the tool name, the docstring its description. Pass `tool_schemas=` to supply schemas yourself.

### Artifacts and card

| Artifact | Description |
|---|---|
| `messages` | Full conversation history |
| `iteration` | Number of LLM turns completed |
| `dispatched_tools` | Tool name, arguments, result, duration, and errors for every call |
| `final_response` | The final text, or `None` if `max_iterations` was reached |

Each `@agent` step also gets an agent card with summary stats, a tool dispatch table, and the final response. Stack your own `@card` with a different type to replace it.

Examples: [`simple_agent.py`](examples/simple_agent.py), [`parallel_agents.py`](examples/parallel_agents.py) (foreach), [`handoff_agents.py`](examples/handoff_agents.py) (sequential handoff).

## Setup

```
pip install metaflow-durability                 # @agent
pip install "metaflow-durability[pydantic-ai]"  # MetaflowDurability
```

With [pixi](https://pixi.sh/): `pixi install`, then `pixi run test`. The `examples` environment adds Tavily and the pydantic-ai harness: `pixi run -e examples python examples/<name>.py run`. The examples load a `.env` if present (for example `MODEL`, `AWS_PROFILE`, `TAVILY_API_KEY`, and Metaflow service or datastore settings).

## Status

Early development.

- **Built:** `MetaflowDurability` recording and retry replay, and the Agent Journal card. `@agent` with tool dispatch, structured artifacts, and its card.
- **Planned:** resume and fork from a journal; keeping the journal through a hard kill; workspace and filesystem state.
