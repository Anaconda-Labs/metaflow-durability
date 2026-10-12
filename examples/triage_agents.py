"""Chained agents in one step: identify -> triage -> extract, then route on the result.

Three named pydantic-ai agents run as plain function calls inside `triage`. Plain code
sits between them and routes on the typed output. The side effect (`notify`) lives in
its own step so a retry of `triage` can never repeat it.

    python examples/triage_agents.py run
    MODEL=anthropic:claude-sonnet-4-5 python examples/triage_agents.py run \\
        --report "Checkout returns 500 for EU customers since the 14:00 deploy."
"""

import os

from dotenv import load_dotenv

load_dotenv()  # loads .env if present, no-op otherwise

from metaflow import FlowSpec
from metaflow import Parameter
from metaflow import Run
from metaflow import current
from metaflow import retry
from metaflow import step
from pydantic import BaseModel
from pydantic_ai import Agent

from metaflow_durability import MetaflowDurability

MODEL = os.environ.get("MODEL", "test")

OWNERS = {"checkout": "payments-team", "search": "discovery-team", "auth": "identity"}
RECENT_DEPLOYS = {"checkout": "v2.31 at 14:00", "search": "v8.2 yesterday"}


class Component(BaseModel):
    name: str
    owner: str


class Triage(BaseModel):
    severity: int
    needs_action: bool
    reason: str


class Action(BaseModel):
    title: str
    assignee: str
    next_step: str


def lookup_owner(component: str) -> str:
    """Return the owning team for a component."""
    return OWNERS.get(component.lower(), "unowned")


def recent_deploy(component: str) -> str:
    """Return the most recent deploy of a component, if any."""
    return RECENT_DEPLOYS.get(component.lower(), "no recent deploy")


identifier = Agent(
    MODEL,
    name="identifier",
    instructions="Identify the affected component and its owner using the tools.",
    tools=[lookup_owner],
    output_type=Component,
    capabilities=[MetaflowDurability()],
)
triager = Agent(
    MODEL,
    name="triager",
    instructions="Rate severity 1-5 and decide whether someone must act now.",
    tools=[recent_deploy],
    output_type=Triage,
    capabilities=[MetaflowDurability()],
)
extractor = Agent(
    MODEL,
    name="extractor",
    instructions="Write a short, actionable ticket for the owning team.",
    output_type=Action,
    capabilities=[MetaflowDurability()],
)


class TriageFlow(FlowSpec):
    report = Parameter(
        "report", default="Search results are empty for queries with accents."
    )

    @step
    def start(self):
        self.next(self.triage)

    @retry(times=2)
    @step
    def triage(self):
        self.component = identifier.run_sync(self.report).output
        context = f"Report: {self.report}\nComponent: {self.component.name}"
        self.triage_result = triager.run_sync(context).output
        self.action = None
        if self.triage_result.needs_action:
            self.action = extractor.run_sync(
                f"{context}\nWhy: {self.triage_result.reason}\n"
                f"Owner: {self.component.owner}"
            ).output
        self.route = "notify" if self.action else "done"
        self.next({"notify": self.notify, "done": self.end}, condition="route")

    @step
    def notify(self):
        print(f"[{self.action.assignee}] {self.action.title}: {self.action.next_step}")
        self.next(self.end)

    @step
    def end(self):
        print(f"severity={self.triage_result.severity} component={self.component.name}")
        journal = (
            Run(f"{current.flow_name}/{current.run_id}")["triage"]
            .task["agent_journal"]
            .data
        )
        for agent_name, runs in journal.items():
            print(f"  {agent_name}: {len(runs[0])} operation(s)")


if __name__ == "__main__":
    TriageFlow()
