"""Simple single-agent flow.

One @agent step with tools. The LLM gathers city stats, computes ratios,
and synthesizes a comparison — all within one Metaflow task.

Usage:
    pixi run python examples/simple_agent.py show
    pixi run python examples/simple_agent.py run
"""

import json
import math

from dotenv import load_dotenv

load_dotenv()  # loads .env if present, no-op otherwise

from metaflow import FlowSpec
from metaflow import Parameter
from metaflow import agent
from metaflow import step


# --- Tools ---

CITY_DATA = {
    "tokyo": {"population": 13960000, "area_km2": 2194, "density_per_km2": 6363},
    "paris": {"population": 2161000, "area_km2": 105, "density_per_km2": 20582},
    "mumbai": {"population": 12478000, "area_km2": 603, "density_per_km2": 20694},
}


def city_stats(city: str) -> str:
    """Get population, area, and density statistics for a city.

    Args:
        city: City name, e.g. 'Tokyo', 'Paris', 'Mumbai'.
    """
    data = CITY_DATA.get(city.lower().strip())
    if data is None:
        return json.dumps(
            {"error": f"No data for '{city}'. Available: {', '.join(CITY_DATA)}"}
        )
    return json.dumps({"city": city, **data})


def calculator(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A math expression to evaluate, e.g. '2**10 + sqrt(144)'.
    """
    allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307
    except Exception as exc:
        return f"Error: {exc}"
    return str(result)


# --- Flow ---


class CityComparisonFlow(FlowSpec):
    """Compare population densities of three cities."""

    prompt = Parameter(
        "prompt",
        default=(
            "Compare Tokyo, Mumbai, and Paris. "
            "For each pair of cities, calculate the ratio of their "
            "population densities. Which pair has the most similar density?"
        ),
    )

    @step
    def start(self):
        """Seed the conversation with the user prompt."""
        self.messages = [{"role": "user", "content": self.prompt}]
        self.next(self.compare_cities)

    @agent(
        model="bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0",
        system_prompt=(
            "You are a helpful assistant with access to tools. "
            "When you need multiple pieces of information, call all "
            "relevant tools simultaneously in a single response rather "
            "than one at a time."
        ),
        tools=[city_stats, calculator],
        max_iterations=10,
    )
    @step
    def compare_cities(self):
        """Look up city stats, compute ratios, synthesize a comparison."""
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    CityComparisonFlow()
