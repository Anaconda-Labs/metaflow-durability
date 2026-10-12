import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel

from metaflow_durability import MetaflowDurability
from metaflow_durability import journal

FLOW = Path(__file__).parent / "durable_retry_flow.py"


def _agent(name: str, calls: list[int]) -> Agent:
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        calls.append(1)
        return a + b

    return Agent(
        TestModel(), name=name, tools=[add], capabilities=[MetaflowDurability()]
    )


@pytest.fixture(autouse=True)
def _clean_journal():
    journal.deactivate()
    yield
    journal.deactivate()


def test_transparent_outside_metaflow():
    calls: list[int] = []
    agent = _agent("calc", calls)
    assert agent.run_sync("hi").output == '{"add":0}'
    assert calls == [1]
    assert journal.active() is None


def test_records_every_operation_in_order():
    calls: list[int] = []
    agent = _agent("calc", calls)
    journal.activate(None)
    agent.run_sync("hi")
    runs = journal.deactivate().runs
    names = [e["name"] for e in runs["calc"][0]]
    assert names == [
        "calc__model.request",
        "calc__function_toolset__<agent>.call_tool:add",
        "calc__model.request",
    ]
    assert not any(e["replayed"] for e in runs["calc"][0])


def test_replays_prior_attempt_without_reexecuting():
    calls: list[int] = []
    agent = _agent("calc", calls)
    journal.activate(None)
    agent.run_sync("hi")
    prior = journal.deactivate().runs

    journal.activate(prior)
    assert agent.run_sync("hi").output == '{"add":0}'
    replay = journal.deactivate().runs
    assert calls == [1]
    assert all(e["replayed"] for e in replay["calc"][0])


def test_same_agent_twice_replays_run_by_run():
    calls: list[int] = []
    agent = _agent("calc", calls)
    journal.activate(None)
    agent.run_sync("hi")
    prior = journal.deactivate().runs

    journal.activate(prior)
    agent.run_sync("hi")
    agent.run_sync("hi")
    runs = journal.deactivate().runs["calc"]
    assert len(runs) == 2
    assert all(e["replayed"] for e in runs[0])
    assert not any(e["replayed"] for e in runs[1])
    assert calls == [1, 1]


def test_divergence_stops_replay():
    prior = {
        "calc": [[{"name": "other__model.request", "payload": None, "duration_ms": 0}]]
    }
    calls: list[int] = []
    agent = _agent("calc", calls)
    journal.activate(prior)
    agent.run_sync("hi")
    runs = journal.deactivate().runs
    assert not any(e["replayed"] for e in runs["calc"][0])


def test_changed_prompt_goes_live_instead_of_replaying():
    calls: list[int] = []
    agent = _agent("calc", calls)
    journal.activate(None)
    agent.run_sync("prompt one")
    prior = journal.deactivate().runs

    journal.activate(prior)
    agent.run_sync("prompt two")
    runs = journal.deactivate().runs
    assert not any(e["replayed"] for e in runs["calc"][0])
    assert calls == [1, 1]


def test_distinct_agents_sharing_a_name_error():
    journal.activate(None)
    _agent("dup", []).run_sync("hi")
    with pytest.raises(UserError, match="Two different agents named 'dup'"):
        _agent("dup", []).run_sync("hi")


def test_flow_retry_replays_completed_operations(tmp_path):
    env = {
        **os.environ,
        "PYDANTIC_AI_NO_BANNER": "1",
        "METAFLOW_DATASTORE_SYSROOT_LOCAL": str(tmp_path),
    }
    run_id_file = tmp_path / "run_id"
    subprocess.run(
        [sys.executable, str(FLOW), "run", "--run-id-file", str(run_id_file)],
        check=True,
        cwd=tmp_path,
        env=env,
        capture_output=True,
    )
    code = (
        "from metaflow import Task\n"
        f"p='DurableRetryFlow/{run_id_file.read_text()}/work/2'\n"
        "for a in (0, 1):\n"
        "    j = Task(p, attempt=a)['agent_journal'].data\n"
        "    print(a, [(n, e['replayed']) for n, r in j.items() for e in r[0]])\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    first, second = (eval(line.split(" ", 1)[1]) for line in out)
    assert [r for _, r in first] == [False] * 4
    assert [r for _, r in second] == [True, True, True, True, False, False]


def test_journal_card_renders_operations_and_empty_state():
    from metaflow_durability.journal_card import render_journal_html

    calls: list[int] = []
    agent = _agent("calc", calls)
    journal.activate(None)
    agent.run_sync("hi")
    html = render_journal_html("F/1/s/2", journal.deactivate().runs)
    assert "tool add" in html and "live" in html and "Input tokens" in html
    assert "No durable agents ran" in render_journal_html("F/1/s/2", {})


def test_live_refresh_called_per_operation():
    snapshots: list[int] = []
    agent = _agent("calc", [])
    journal.activate(None, lambda runs: snapshots.append(len(runs["calc"][0])))
    agent.run_sync("hi")
    journal.deactivate()
    assert snapshots == [1, 2, 3]


def test_card_banner_marks_live_and_final_attempts():
    from metaflow_durability.journal_card import render_journal_html

    assert "running, updating live" in render_journal_html("F/1/s/2", {}, attempt=0)
    final = render_journal_html("F/1/s/2", {}, attempt=1, final=True)
    assert "Attempt 1 · final" in final and "after 1 failed attempt" in final


def test_later_attempt_card_uuid_sorts_first():
    from metaflow.plugins.cards.component_serializer import CardComponentCollector

    from metaflow_durability import journal_decorator

    original = CardComponentCollector._add_card
    uuids = []
    try:
        for attempt in (0, 1):
            CardComponentCollector._add_card = original
            journal_decorator._order_card_uuids_by_attempt(attempt)
            collector = CardComponentCollector()
            meta = collector._add_card(
                "agent_journal", "agent_journal", {"type": "agent_journal"}, {}
            )
            uuids.append(meta["uuid"])
            other = collector._add_card("blank", None, {"type": "blank"}, {})
            assert not other["uuid"].startswith("fff")
    finally:
        CardComponentCollector._add_card = original
    assert sorted(uuids) == [uuids[1], uuids[0]]
    assert all(len(u) == 32 for u in uuids)
