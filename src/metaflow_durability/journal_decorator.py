"""Step decorator that hosts the MetaflowDurability journal for a task.

Auto-attached to every step by the metaflow-durability extension (TOGGLE_DECOSPECS). It is a
no-op unless pydantic-ai is installed and a `MetaflowDurability` agent runs in the step.
"""

from __future__ import annotations

import uuid
from typing import Any

from metaflow import current
from metaflow.decorators import StepDecorator

CARD_TYPE = "agent_journal"


def _refresh_card(runs: dict[str, Any]) -> None:
    for card in getattr(current, "card").get(type=CARD_TYPE):
        card.refresh({"journal": runs, "attempt": current.retry_count})


def _render_failed_attempt_card(runs: dict[str, Any]) -> None:
    # Metaflow skips the final card render when a task fails, and live refreshes are
    # rate limited, so the card would freeze mid-run with a "running" banner and miss
    # the last operations. Render once more, synchronously, with the full journal,
    # and write the data too so the UI does not show the card blank.
    for card in getattr(current, "card").get(type=CARD_TYPE):
        card._latest_user_data = {
            "journal": runs,
            "attempt": current.retry_count,
            "failed": True,
        }
        card._card_proc("render_runtime", sync=True)
        card._card_proc("refresh", sync=True)


def _order_card_uuids_by_attempt(retry_count: int) -> None:
    """Give this attempt's journal card a uuid that sorts before earlier attempts'.

    Cards are stored per task, not per attempt (`<type>-<id>-<uuid>.html`), and the UI
    shows every card of a task under its successful attempt, in datastore listing
    order: lexicographic on S3/GCS/Azure. A failed attempt's live card therefore
    lingers next to the final one. Prefixing the uuid with (0xffff - attempt) makes
    the latest attempt's card list first. Local datastores list in filesystem order.
    """
    from metaflow.plugins.cards.component_serializer import CardComponentCollector

    original = CardComponentCollector._add_card
    if getattr(original, "_agent_journal_ordered", False):
        return

    def _add_card(self: Any, card_type: str, *args: Any, **kwargs: Any) -> Any:
        if card_type != CARD_TYPE:
            return original(self, card_type, *args, **kwargs)
        prefix = format(0xFFFF - retry_count, "04x")
        self.create_uuid = lambda: prefix + uuid.uuid4().hex[4:]
        try:
            return original(self, card_type, *args, **kwargs)
        finally:
            del self.create_uuid

    setattr(_add_card, "_agent_journal_ordered", True)
    setattr(CardComponentCollector, "_add_card", _add_card)


class AgentJournalDecorator(StepDecorator):
    name = "agent_journal"
    defaults: dict[str, Any] = {}

    _enabled = False
    _card: Any = None

    def step_init(
        self,
        flow: Any,
        graph: Any,
        step_name: str,
        decorators: list[Any],
        environment: Any,
        flow_datastore: Any,
        logger: Any,
    ) -> None:
        from metaflow.plugins.cards.card_decorator import CardDecorator

        existing = [
            d
            for d in decorators
            if isinstance(d, CardDecorator) and d.attributes.get("type") == CARD_TYPE
        ]
        if existing:
            self._card = existing[0]
            return
        attributes = {**CardDecorator.defaults, "type": CARD_TYPE, "id": CARD_TYPE}
        card = CardDecorator(attributes=attributes)
        decorators.append(card)
        self._card = card
        card.step_init(
            flow, graph, step_name, decorators, environment, flow_datastore, logger
        )

    def task_pre_step(
        self,
        step_name: str,
        task_datastore: Any,
        metadata: Any,
        run_id: str,
        task_id: str,
        flow: Any,
        graph: Any,
        retry_count: int,
        max_user_code_retries: int,
        ubf_context: Any,
        inputs: Any,
    ) -> None:
        try:
            from metaflow_durability import journal
        except ImportError:
            return
        if self._card is not None:
            _order_card_uuids_by_attempt(retry_count)
        pathspec = f"{flow.name}/{run_id}/{step_name}/{task_id}"
        journal.activate(journal.load_prior(pathspec, retry_count), _refresh_card)
        self._enabled = True

    def _flush(self, flow: Any, failed: bool = False) -> None:
        if not self._enabled:
            return
        from metaflow_durability import journal

        task_journal = journal.deactivate()
        if task_journal is not None:
            # Always overwrite: an upstream step's journal would otherwise propagate
            # into this task and be replayed as if it were this task's prior attempt.
            setattr(flow, journal.JOURNAL_ARTIFACT, task_journal.runs)
            if failed:
                _render_failed_attempt_card(task_journal.runs)

    def task_post_step(
        self,
        step_name: str,
        flow: Any,
        graph: Any,
        retry_count: int,
        max_user_code_retries: int,
    ) -> None:
        self._flush(flow)

    def task_exception(
        self,
        exception: Exception,
        step_name: str,
        flow: Any,
        graph: Any,
        retry_count: int,
        max_user_code_retries: int,
    ) -> None:
        self._flush(flow, failed=True)
