"""A crash mid-step, and a retry that does not pay for completed work again.

The `quote` agent calls a cheap `list_prices` tool and a rate-limited `fx_rate` tool.
On the first attempt `fx_rate` fails (simulating a provider error after the earlier
model calls and tools already ran). `@retry` restarts the step; operations the first
attempt finished are replayed from its journal, and only the failed tool call and what
follows run live. The `end` step prints both attempts' journals side by side.

    python examples/retry_replay.py run
"""

import os
from dotenv import load_dotenv

load_dotenv()  # loads .env if present, no-op otherwise

from metaflow import FlowSpec
from metaflow import Parameter
from metaflow import Run
from metaflow import Task
from metaflow import current
from metaflow import retry
from metaflow import step
from pydantic_ai import Agent

from metaflow_durability import MetaflowDurability

MODEL = os.environ.get("MODEL", "test")

PRICES_USD = {"laptop": 1299.0, "monitor": 349.0, "dock": 189.0}
FX = {"EUR": 0.92, "GBP": 0.79, "JPY": 149.3}


def list_prices(item: str) -> str:
    """Return the USD list price of a catalog item."""
    return str(PRICES_USD.get(item.lower(), "not in catalog"))


def fx_rate(currency: str) -> str:
    """Return the USD exchange rate for a currency (rate-limited provider)."""
    if current.retry_count == 0:
        raise ConnectionError("fx provider: 429 Too Many Requests")
    return str(FX.get(currency.upper(), "unsupported currency"))


quote = Agent(
    MODEL,
    name="quote",
    instructions="Price the items in USD with list_prices, convert with fx_rate.",
    tools=[list_prices, fx_rate],
    capabilities=[MetaflowDurability()],
)


class RetryReplayFlow(FlowSpec):
    request = Parameter("request", default="Quote a laptop and a dock in EUR.")

    @step
    def start(self):
        self.next(self.price)

    @retry(times=1, minutes_between_retries=0)
    @step
    def price(self):
        self.quote = quote.run_sync(self.request).output
        self.next(self.end)

    @step
    def end(self):
        print(f"quote: {self.quote}")
        price_task = Run(f"{current.flow_name}/{current.run_id}")["price"].task
        for attempt in (0, 1):
            journal = Task(price_task.pathspec, attempt=attempt)["agent_journal"].data
            print(f"attempt {attempt}:")
            for entry in journal["quote"][0]:
                source = "replayed" if entry["replayed"] else "live"
                operation = entry["name"].split("__", 1)[1]
                print(f"  {source:<8} {entry['duration_ms']:7.1f} ms  {operation}")


if __name__ == "__main__":
    RetryReplayFlow()
