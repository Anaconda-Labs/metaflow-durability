"""A coding agent that picks up where it crashed: Coder in a durable workspace.

`code` runs a pydantic-ai-harness `Coder` agent in a workspace. `MetaflowDurability()`
journals every model call, tool call, and workspace call, so a retry replays the
finished ones instead of re-running them. `start` creates the workspace once and
keeps its ref; `code` and `verify` reattach to it, so files survive any retry:

    WORKSPACE=local  a directory on this machine, the same for every attempt
    WORKSPACE=e2b    an E2B sandbox (needs E2B_API_KEY); use this on Kubernetes,
                     Argo, or Outerbounds, where each attempt runs in a new pod

On the first attempt the run crashes on purpose right after its second file-changing
tool call (SIMULATE_CRASH=0 turns this off); the retry continues in the same
workspace. `verify` runs the tests there, and `end` removes an E2B sandbox.
Requires a model (MODEL, e.g. in .env):

    pixi run -e examples python examples/coder_flow.py run
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any

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
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.workspaces import LocalWorkspaceBackend
from pydantic_ai.workspaces import Workspace
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness import Coder

from metaflow_durability import MetaflowDurability

MODEL = os.environ.get("MODEL", "test")
WORKSPACE = os.environ.get("WORKSPACE", "local")
SIMULATE_CRASH = os.environ.get("SIMULATE_CRASH", "1") == "1"
os.environ.setdefault("AWS_READ_TIMEOUT", "120")  # fail fast on a stalled Bedrock call

TASK = (
    "Create a small Python module `textstats.py` with `word_count(text)` and "
    "`top_words(text, n)` (case-insensitive, ignore punctuation), plus "
    "`test_textstats.py` with pytest tests. Run `python -m pytest -q` and fix "
    "anything that fails until the tests pass. Keep it short."
)


class TestReport(BaseModel):
    passed: bool
    summary: str
    files: list[str]


def create_workspace() -> WorkspaceRef:
    """Create the run's workspace once; steps reattach to it by ref."""
    if WORKSPACE == "e2b":
        from pydantic_ai_harness.e2b_sandbox import E2BSandboxBackend

        backend = E2BSandboxBackend()
        asyncio.run(Workspace(backend).run(["pip", "install", "-q", "pytest"]))
        assert backend.ref is not None
        return backend.ref
    path = Path(tempfile.gettempdir()) / "metaflow-durability-workspaces" / current.run_id
    path.mkdir(parents=True, exist_ok=True)
    return LocalWorkspaceBackend(path).ref


def workspace_capability(ref: WorkspaceRef) -> AbstractCapability[Any]:
    """The capability that serves `ref`: the only backend-specific code in the flow."""
    if WORKSPACE == "e2b":
        from pydantic_ai_harness.e2b_sandbox import E2BSandbox

        return E2BSandbox()
    return LocalWorkspace(ref.id)


def destroy_workspace(ref: WorkspaceRef) -> None:
    if WORKSPACE == "e2b":
        from pydantic_ai_harness.e2b_sandbox import E2BSandbox

        asyncio.run(E2BSandbox().destroy(ref))


class CrashOnce(AbstractCapability[Any]):
    """Fail the first attempt right after its second file-changing tool call."""

    def __init__(self) -> None:
        self.changes = 0

    async def after_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: Any,
        result: Any,
    ) -> Any:
        if call.tool_name in {"write_file", "edit_file", "shell"}:
            self.changes += 1
            if SIMULATE_CRASH and current.retry_count == 0 and self.changes == 2:
                raise RuntimeError("simulated crash after a file-changing tool call")
        return result


class CoderFlow(FlowSpec):
    task = Parameter("task", default=TASK)

    @step
    def start(self):
        self.workspace = create_workspace()
        self.next(self.code)

    @retry(times=1, minutes_between_retries=0)
    @step
    def code(self):
        coder = Agent(
            MODEL,
            name="coder",
            capabilities=[
                MetaflowDurability(),
                Coder(sub_agents=False),
                workspace_capability(self.workspace),
                CrashOnce(),
            ],
        )
        self.summary = coder.run_sync(str(self.task), workspace=self.workspace).output
        self.next(self.verify)

    @step
    def verify(self):
        checker = Agent(
            MODEL,
            name="checker",
            output_type=TestReport,
            capabilities=[
                MetaflowDurability(),
                Coder(sub_agents=False),
                workspace_capability(self.workspace),
            ],
        )
        self.report = checker.run_sync(
            "Run `python -m pytest -q` and list the files in the project. "
            "Report whether the tests pass. Do not change any files.",
            workspace=self.workspace,
        ).output
        self.next(self.end)

    @card(type="blank", id="result")
    @step
    def end(self):
        status = "passed" if self.report.passed else "FAILED"
        current.card["result"].append(
            Markdown(
                f"## Tests {status}\n\n**Files:** {', '.join(self.report.files)}\n\n"
                f"{self.report.summary}\n\n### Coder summary\n\n{self.summary}"
            )
        )
        print(f"tests {status}: {self.report.files}")
        destroy_workspace(self.workspace)


if __name__ == "__main__":
    CoderFlow()
