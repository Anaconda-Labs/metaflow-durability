"""pydantic-ai durable execution backend for Metaflow.

Attach `MetaflowDurability()` to a pydantic-ai `Agent`. Inside a running Metaflow task,
every durable operation (model request, tool call, tool discovery, ...) is journaled to
the task's `agent_journal` artifact. On `@retry`, operations recorded by the previous
attempt are replayed from the journal instead of re-executed. Outside a Metaflow task
the capability is transparent.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable
from collections.abc import Callable
from typing import Any

from pydantic_ai.durable_exec import JSON_CODEC
from pydantic_ai.durable_exec import BaseDurabilityCapability
from pydantic_ai.durable_exec import DurabilityEngineSpec
from pydantic_ai.durable_exec import DurableOperationId
from pydantic_ai.durable_exec import JournalCallableOperationBackend
from pydantic_ai.durable_exec import RoleBasedOperationConfig

from metaflow_durability import journal


class MetaflowSerializationError(Exception):
    """A durable operation's result could not be encoded for the journal."""


class _MetaflowBackend(JournalCallableOperationBackend[None]):
    def __init__(self, agent_name: str, default_model_id: str | None) -> None:
        super().__init__(
            agent_name=agent_name,
            default_model_id=default_model_id,
            config=RoleBasedOperationConfig(
                model=None, event=None, capability=None, tool=None
            ),
        )

    async def execute(
        self,
        *,
        operation_id: DurableOperationId,
        name: str,
        body: Callable[[], Awaitable[object]],
        cache_key: tuple[object, ...],
        config: None,
    ) -> object:
        run = journal.current_run.get()
        if run is None:
            return await body()
        kind = type(operation_id).__name__
        input_hash = journal.input_hash(cache_key)
        prior = run.lookup(name, input_hash)
        if prior is not None:
            run.record(
                journal.JournalEntry(
                    name=name,
                    kind=kind,
                    input_hash=input_hash,
                    payload=prior["payload"],
                    duration_ms=prior["duration_ms"],
                    replayed=True,
                )
            )
            return prior["payload"]
        started = time.monotonic()
        payload = await body()
        run.record(
            journal.JournalEntry(
                name=name,
                kind=kind,
                input_hash=input_hash,
                payload=payload,
                duration_ms=(time.monotonic() - started) * 1000,
                replayed=False,
            )
        )
        return payload


class MetaflowDurability(BaseDurabilityCapability[Any]):
    """Journal and replay a pydantic-ai agent's operations inside a Metaflow step."""

    engine_spec = DurabilityEngineSpec(
        engine_name="Metaflow",
        durable_unit_noun="operation",
        durable_container_noun="task",
        codec=JSON_CODEC,
        serialization_failure=lambda exc: MetaflowSerializationError(str(exc)),
        toolset_lifecycles={
            "function": "enter-always",
            "mcp": "enter-in-durable-unit",
            "dynamic": "enter-never",
        },
        sequential_tools_in_durable_context=True,
    )

    @property
    def in_durable_context(self) -> bool:
        return journal.active() is not None

    def get_durable_operation_backend(self) -> _MetaflowBackend:
        return _MetaflowBackend(self.name, self.default_model_id)

    async def wrap_run(self, ctx: Any, *, handler: Any) -> Any:
        task_journal = journal.active()
        if task_journal is None:
            return await super().wrap_run(ctx, handler=handler)
        token = journal.current_run.set(task_journal.start_run(self.name, self.agent))
        try:
            return await super().wrap_run(ctx, handler=handler)
        finally:
            journal.current_run.reset(token)
