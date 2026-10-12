"""Fan out one agent across a foreach, then synthesize the typed results.

The same `researcher` agent runs in every foreach task, each with its own journal and
retry replay. `gather` collects the typed profiles; the `synthesizer` agent compares
them in a separate step.

    python examples/foreach_research.py run
    MODEL=anthropic:claude-sonnet-4-5 python examples/foreach_research.py run \\
        --cities "Tokyo,Lagos,Lima"
"""

import os

from dotenv import load_dotenv

load_dotenv()  # loads .env if present, no-op otherwise

from metaflow import FlowSpec
from metaflow import Parameter
from metaflow import retry
from metaflow import step
from pydantic import BaseModel
from pydantic_ai import Agent

from metaflow_durability import MetaflowDurability

MODEL = os.environ.get("MODEL", "test")

CITY_DATA = {
    "tokyo": {"population_m": 14.0, "area_km2": 2194, "transit_lines": 158},
    "paris": {"population_m": 2.1, "area_km2": 105, "transit_lines": 16},
    "lagos": {"population_m": 15.4, "area_km2": 1171, "transit_lines": 2},
    "lima": {"population_m": 10.1, "area_km2": 2672, "transit_lines": 2},
}


class CityProfile(BaseModel):
    city: str
    density_per_km2: float
    transit_summary: str


class Comparison(BaseModel):
    ranking: list[str]
    takeaway: str


def city_stats(city: str) -> str:
    """Return population (millions), area (km2), and number of transit lines."""
    return str(CITY_DATA.get(city.lower(), "unknown city"))


def divide(numerator: float, denominator: float) -> float:
    """Divide two numbers."""
    return numerator / denominator if denominator else 0.0


researcher = Agent(
    MODEL,
    name="researcher",
    instructions="Profile the city: compute density with the tools, summarize transit.",
    tools=[city_stats, divide],
    output_type=CityProfile,
    capabilities=[MetaflowDurability()],
)
synthesizer = Agent(
    MODEL,
    name="synthesizer",
    instructions="Rank the cities by livability for a car-free resident.",
    output_type=Comparison,
    capabilities=[MetaflowDurability()],
)


class ForeachResearchFlow(FlowSpec):
    cities = Parameter("cities", default="Tokyo,Paris,Lagos")

    @step
    def start(self):
        self.city_list = [c.strip() for c in self.cities.split(",")]
        self.next(self.research, foreach="city_list")

    @retry(times=2)
    @step
    def research(self):
        self.profile = researcher.run_sync(f"Profile {self.input}.").output
        self.next(self.gather)

    @step
    def gather(self, inputs):
        self.profiles = [task.profile for task in inputs]
        self.next(self.synthesize)

    @retry(times=2)
    @step
    def synthesize(self):
        briefs = "\n".join(p.model_dump_json() for p in self.profiles)
        self.comparison = synthesizer.run_sync(briefs).output
        self.next(self.end)

    @step
    def end(self):
        for profile in self.profiles:
            print(f"{profile.city}: {profile.density_per_km2:.0f}/km2")
        print(f"ranking: {self.comparison.ranking}")
        print(f"takeaway: {self.comparison.takeaway}")


if __name__ == "__main__":
    ForeachResearchFlow()
