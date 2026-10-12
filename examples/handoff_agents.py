"""Sequential agent handoff.

One agent researches raw data, a second agent with a different persona
takes that output and produces a polished deliverable. The handoff step
reshapes the first agent's output into the second agent's input.

    start → researcher → handoff → writer → end

The researcher has data-lookup tools. The writer has no tools — it's
pure generation from the research output.

Usage:
    pixi run python examples/handoff_agents.py show
    pixi run python examples/handoff_agents.py run
"""

import json

from dotenv import load_dotenv

load_dotenv()  # loads .env if present, no-op otherwise

from metaflow import FlowSpec
from metaflow import Parameter
from metaflow import agent
from metaflow import step


CITY_DATA = {
    "tokyo": {
        "population": 13960000,
        "area_km2": 2194,
        "gdp_usd_billions": 1920,
        "founded": 1457,
    },
    "paris": {
        "population": 2161000,
        "area_km2": 105,
        "gdp_usd_billions": 765,
        "founded": -250,
    },
    "mumbai": {
        "population": 12478000,
        "area_km2": 603,
        "gdp_usd_billions": 368,
        "founded": 1507,
    },
}


def city_stats(city: str) -> str:
    """Get population, area, GDP, and founding year for a city.

    Args:
        city: City name, e.g. 'Tokyo', 'Paris', 'Mumbai'.
    """
    data = CITY_DATA.get(city.lower().strip())
    if data is None:
        return json.dumps({"error": f"No data for '{city}'."})
    return json.dumps({"city": city, **data})


class HandoffFlow(FlowSpec):
    """Research-to-writing pipeline: one agent gathers data, another writes it up."""

    prompt = Parameter(
        "prompt",
        default="Write a brief city profile comparing Tokyo, Paris, and Mumbai.",
    )

    @step
    def start(self):
        self.messages = [
            {
                "role": "user",
                "content": (
                    f"{self.prompt} "
                    "First, look up key statistics for each city."
                ),
            },
        ]
        self.next(self.researcher)

    @agent(
        model="bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0",
        system_prompt=(
            "You are a data researcher. Use the city_stats tool to gather "
            "facts about each city mentioned. Look up every city before "
            "summarizing. Present raw findings — do not format or write prose."
        ),
        tools=[city_stats],
        max_iterations=5,
    )
    @step
    def researcher(self):
        self.next(self.handoff)

    @step
    def handoff(self):
        research_output = self.final_response
        self.messages = [
            {
                "role": "user",
                "content": (
                    f"Here is raw research data:\n\n{research_output}\n\n"
                    f"Task: {self.prompt}\n"
                    "Produce a polished markdown document with a comparison "
                    "table and key takeaways."
                ),
            },
        ]
        self.next(self.writer)

    @agent(
        model="bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0",
        system_prompt=(
            "You are a technical writer. Take raw research data and produce "
            "polished markdown output. Be concise and professional."
        ),
        max_iterations=1,
    )
    @step
    def writer(self):
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    HandoffFlow()
