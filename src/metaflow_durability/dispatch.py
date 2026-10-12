"""Dispatch records — structured artifacts for tool execution within agent steps.

Each tool call dispatched by the @agent decorator produces a DispatchRecord.
Records are collected per-iteration as self.dispatched_tools on the agent step's
task, making them queryable via the Client API without adding graph nodes.

The in-process executor runs tools sequentially in the same Python process.
The interface is designed so that swapping in Runner-based dispatch (child flow
runs) or native self.dispatch() (graph sub-tasks) requires changing only the
executor, not the record format.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Protocol


@dataclass(frozen=True)
class DispatchRecord:
    """One tool execution's full record.

    Stored as a list on self.dispatched_tools per iteration. Designed to be
    stable across executor backends: in-process, Runner, or graph-node.
    """

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    tier: str
    result: str
    started_at: float
    finished_at: float
    error: str | None = None

    @property
    def duration_ms(self) -> float:
        return (self.finished_at - self.started_at) * 1000

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["duration_ms"] = self.duration_ms
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DispatchRecord:
        d = dict(d)
        d.pop("duration_ms", None)
        return cls(**d)


@dataclass
class DispatchResult:
    """Aggregated result of dispatching all tool calls in one iteration.

    Contains the individual records plus the tool messages ready to append
    to the conversation history.
    """

    records: list[DispatchRecord] = field(default_factory=list)
    tool_messages: list[dict[str, Any]] = field(default_factory=list)

    def to_dicts(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.records]

    @classmethod
    def from_dicts(cls, dicts: list[dict[str, Any]]) -> DispatchResult:
        records = [DispatchRecord.from_dict(d) for d in dicts]
        return cls(records=records)


class ToolExecutor(Protocol):
    """Protocol for tool execution backends.

    In-process, Runner, and eventual graph-node executors all implement this.
    """

    def execute(
        self,
        tool_calls: list[dict[str, Any]],
    ) -> DispatchResult: ...


class InProcessExecutor:
    """Execute tool calls sequentially in the current process.

    Tools are plain callables. The executor calls each with ``**arguments``
    from the LLM's tool_call and converts the return value to a string
    for the tool message.
    """

    def __init__(self, tool_registry: dict[str, Callable[..., Any]]) -> None:
        self._registry = tool_registry

    def execute(
        self,
        tool_calls: list[dict[str, Any]],
    ) -> DispatchResult:
        records: list[DispatchRecord] = []
        tool_messages: list[dict[str, Any]] = []

        for tc in tool_calls:
            tool_name = tc["name"]
            arguments = tc["arguments"]
            tool_call_id = tc["id"]
            tier = tc.get("tier", "default")

            started_at = time.monotonic()
            error: str | None = None
            result_str: str

            fn = self._registry.get(tool_name)
            if fn is None:
                result_str = f"Unknown tool: {tool_name}"
                error = result_str
            else:
                try:
                    result = fn(**arguments)
                    result_str = (
                        result if isinstance(result, str) else json.dumps(result)
                    )
                except Exception as exc:
                    result_str = f"Error executing {tool_name}: {exc}"
                    error = str(exc)

            finished_at = time.monotonic()

            records.append(
                DispatchRecord(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    arguments=arguments,
                    tier=tier,
                    result=result_str,
                    started_at=started_at,
                    finished_at=finished_at,
                    error=error,
                )
            )

            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result_str,
                }
            )

        return DispatchResult(records=records, tool_messages=tool_messages)
