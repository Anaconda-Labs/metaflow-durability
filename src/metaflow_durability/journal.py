"""Task-scoped journal of pydantic-ai durable operations.

Layout of the `agent_journal` artifact::

    {agent_name: [run_0_entries, run_1_entries, ...]}

Each run is the ordered list of durable operations (model requests, tool calls,
discovery, ...) of one `agent.run()` inside the task. Entries are keyed by position:
the k-th `agent.run()` of an agent replays against the k-th recorded run, operation
by operation, until the first operation whose name or inputs differ.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any

from pydantic_ai.exceptions import UserError
from pydantic_core import to_json

JOURNAL_ARTIFACT = "agent_journal"

JournalData = dict[str, list[list[dict[str, Any]]]]


@dataclass
class JournalEntry:
    """One durable operation. `payload` is the JSON-encoded operation result."""

    name: str
    kind: str
    input_hash: str
    payload: Any
    duration_ms: float
    replayed: bool


class AgentRunJournal:
    """Journal of one `agent.run()`: records operations and replays prior ones."""

    def __init__(
        self,
        entries: list[dict[str, Any]],
        prior: list[dict[str, Any]],
        on_change: Callable[[], None] | None = None,
    ):
        self._entries = entries
        self._prior = prior
        self._diverged = False
        self._on_change = on_change

    def lookup(self, name: str, input_hash: str) -> dict[str, Any] | None:
        """Return the prior entry at the next position if name and inputs match."""
        if self._diverged:
            return None
        position = len(self._entries)
        if position < len(self._prior):
            prior = self._prior[position]
            if prior["name"] == name and prior.get("input_hash") == input_hash:
                return prior
        self._diverged = True
        return None

    def record(self, entry: JournalEntry) -> None:
        self._entries.append(asdict(entry))
        if self._on_change is not None:
            self._on_change()


class TaskJournal:
    """All agent runs of one Metaflow task attempt, plus the prior attempt's runs."""

    def __init__(
        self,
        prior: JournalData | None = None,
        on_change: Callable[[JournalData], None] | None = None,
    ):
        self.runs: JournalData = {}
        self._prior: JournalData = prior or {}
        self._owners: dict[str, int] = {}
        self._on_change = on_change

    def _changed(self) -> None:
        if self._on_change is not None:
            self._on_change(self.runs)

    def start_run(self, agent_name: str, agent: object) -> AgentRunJournal:
        owner = self._owners.setdefault(agent_name, id(agent))
        if owner != id(agent):
            raise UserError(
                f"Two different agents named {agent_name!r} ran in the same "
                "Metaflow step. Durable operations are journaled by agent name, "
                "so each agent in a step needs a unique `name`."
            )
        runs = self.runs.setdefault(agent_name, [])
        prior_runs = self._prior.get(agent_name, [])
        prior = prior_runs[len(runs)] if len(runs) < len(prior_runs) else []
        entries: list[dict[str, Any]] = []
        runs.append(entries)
        return AgentRunJournal(entries, prior, self._changed)


_task_journal: TaskJournal | None = None
current_run: ContextVar[AgentRunJournal | None] = ContextVar(
    "metaflow_durability_current_run", default=None
)


def activate(
    prior: JournalData | None,
    on_change: Callable[[JournalData], None] | None = None,
) -> TaskJournal:
    global _task_journal
    _task_journal = TaskJournal(prior, on_change)
    return _task_journal


def deactivate() -> TaskJournal | None:
    global _task_journal
    journal, _task_journal = _task_journal, None
    return journal


def active() -> TaskJournal | None:
    return _task_journal


_VOLATILE_KEYS = frozenset({"timestamp", "run_id", "conversation_id"})


def _strip_volatile(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: _strip_volatile(v) for k, v in value.items() if k not in _VOLATILE_KEYS
        }
    if isinstance(value, list):
        return [_strip_volatile(v) for v in value]
    return value


def input_hash(cache_key: tuple[object, ...]) -> str:
    """Hash an operation's inputs, ignoring values that change between attempts.

    Elements that cannot be serialized (RunContext, tool objects) contribute only
    their type name, so the hash reflects prompts, messages, settings, and tool args.
    """
    parts = []
    for element in cache_key:
        try:
            encoded = to_json(element, fallback=lambda o: type(o).__name__)
        except ValueError:
            encoded = to_json(type(element).__name__)
        parts.append(_strip_volatile(json.loads(encoded)))
    canonical = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def load_prior(pathspec: str, retry_count: int) -> JournalData | None:
    """Load the journal written by the previous attempt of this task, if any."""
    if retry_count < 1:
        return None
    from metaflow import Task
    from metaflow.exception import MetaflowNotFound

    try:
        task = Task(pathspec, attempt=retry_count - 1, _namespace_check=False)
        if JOURNAL_ARTIFACT not in task:
            return None
        data: JournalData = task[JOURNAL_ARTIFACT].data
    except MetaflowNotFound:
        return None
    return data
