from metaflow import FlowSpec
from metaflow import current
from metaflow import retry
from metaflow import step
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from metaflow_durability import MetaflowDurability


def lookup(city: str) -> str:
    """Look up a city."""
    return f"{city}: 14M people"


def publish(text: str) -> str:
    """Publish text. Fails on the first attempt to exercise retry replay."""
    if current.retry_count == 0:
        raise RuntimeError("simulated crash in publish")
    return "published"


researcher = Agent(
    TestModel(), name="researcher", tools=[lookup], capabilities=[MetaflowDurability()]
)
publisher = Agent(
    TestModel(), name="publisher", tools=[publish], capabilities=[MetaflowDurability()]
)


class DurableRetryFlow(FlowSpec):
    @step
    def start(self):
        self.next(self.work)

    @retry(times=1, minutes_between_retries=0)
    @step
    def work(self):
        self.research = researcher.run_sync("Tokyo").output
        self.published = publisher.run_sync(self.research).output
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    DurableRetryFlow()
