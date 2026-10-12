"""Agent card — renders iteration timeline and tool dispatch details.

Registered as card type "agent" via metaflow_extensions. Auto-attached
to every @agent step by the AgentDecorator.
"""

import json
from typing import Any

from metaflow.cards import MetaflowCard


def _render_html(
    pathspec: str,
    iteration: int,
    dispatched: list[dict[str, Any]],
    final_response: str | None,
    messages: list[dict[str, Any]],
) -> str:
    iterations: dict[int, list[dict[str, Any]]] = {}
    for d in dispatched:
        msg_index = _find_iteration_for_dispatch(d, messages)
        iterations.setdefault(msg_index, []).append(d)

    tool_rows = ""
    for it_num in sorted(iterations):
        calls = iterations[it_num]
        for d in calls:
            dur = d.get("duration_ms", 0)
            err = d.get("error") or ""
            args_str = json.dumps(d.get("arguments", {}))
            if len(args_str) > 120:
                args_str = args_str[:117] + "..."
            result_str = d.get("result", "")
            if len(result_str) > 200:
                result_str = result_str[:197] + "..."
            status = (
                '<span style="color:#dc3545">error</span>'
                if err
                else '<span style="color:#28a745">ok</span>'
            )
            tool_rows += (
                f"<tr>"
                f"<td>{it_num}</td>"
                f"<td><code>{d.get('tool_name', '?')}</code></td>"
                f"<td><code>{args_str}</code></td>"
                f"<td>{status}</td>"
                f"<td>{dur:.1f}ms</td>"
                f"<td><code>{result_str}</code></td>"
                f"</tr>\n"
            )

    if not tool_rows:
        tool_rows = '<tr><td colspan="6" style="text-align:center;color:#888">No tool calls dispatched</td></tr>'

    response_html = (
        f"<pre style='white-space:pre-wrap;max-height:400px;overflow:auto'>{final_response}</pre>"
        if final_response
        else '<p style="color:#888">No final response (max iterations reached)</p>'
    )

    total_tools = len(dispatched)
    total_duration = sum(d.get("duration_ms", 0) for d in dispatched)
    n_errors = sum(1 for d in dispatched if d.get("error"))
    msg_count = len(messages)

    return f"""<!DOCTYPE html>
<html>
<head>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; color: #333; }}
h2 {{ border-bottom: 1px solid #ddd; padding-bottom: 8px; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 13px; }}
th {{ background: #f5f5f5; font-weight: 600; }}
tr:hover {{ background: #f9f9f9; }}
.stat {{ display: inline-block; margin-right: 24px; }}
.stat-value {{ font-size: 24px; font-weight: 700; }}
.stat-label {{ font-size: 12px; color: #888; text-transform: uppercase; }}
code {{ font-size: 12px; }}
pre {{ background: #f5f5f5; padding: 12px; border-radius: 4px; font-size: 13px; }}
</style>
</head>
<body>
<h2>Agent Run</h2>
<div>
  <div class="stat"><div class="stat-value">{iteration + 1}</div><div class="stat-label">Iterations</div></div>
  <div class="stat"><div class="stat-value">{total_tools}</div><div class="stat-label">Tool Calls</div></div>
  <div class="stat"><div class="stat-value">{total_duration:.0f}ms</div><div class="stat-label">Tool Time</div></div>
  <div class="stat"><div class="stat-value">{msg_count}</div><div class="stat-label">Messages</div></div>
  <div class="stat"><div class="stat-value">{n_errors}</div><div class="stat-label">Errors</div></div>
</div>

<h2>Tool Dispatch</h2>
<table>
<thead><tr><th>Iter</th><th>Tool</th><th>Arguments</th><th>Status</th><th>Duration</th><th>Result</th></tr></thead>
<tbody>
{tool_rows}
</tbody>
</table>

<h2>Final Response</h2>
{response_html}

<p style="color:#aaa;font-size:11px;margin-top:32px">{pathspec}</p>
</body>
</html>"""


def _find_iteration_for_dispatch(
    dispatch: dict[str, Any], messages: list[dict[str, Any]]
) -> int:
    """Find which iteration a dispatch record belongs to by matching tool_call_id."""
    target_id = dispatch.get("tool_call_id", "")
    iteration = 0
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if tc.get("id") == target_id:
                    return iteration
            iteration += 1
    return iteration


class AgentCard(MetaflowCard):
    type = "agent"
    ALLOW_USER_COMPONENTS = False
    RUNTIME_UPDATABLE = False

    def __init__(self, options: dict[str, Any] | None = None, **kwargs: Any) -> None:
        self._options = options or {}

    def render(self, task: Any) -> str:
        dispatched = task["dispatched_tools"].data if "dispatched_tools" in task else []
        messages = task["messages"].data if "messages" in task else []
        iteration = task["iteration"].data if "iteration" in task else 0
        final_response = (
            task["final_response"].data if "final_response" in task else None
        )
        return _render_html(
            pathspec=task.pathspec,
            iteration=iteration,
            dispatched=dispatched,
            final_response=final_response,
            messages=messages,
        )


class AgentJournalCard(MetaflowCard):
    """Renders the MetaflowDurability `agent_journal`; refreshes live during the task."""

    type = "agent_journal"
    ALLOW_USER_COMPONENTS = False
    RUNTIME_UPDATABLE = True
    RELOAD_POLICY = MetaflowCard.RELOAD_POLICY_ALWAYS

    def __init__(self, options: dict[str, Any] | None = None, **kwargs: Any) -> None:
        self._options = options or {}

    def render(self, task: Any) -> str:
        from metaflow_durability.journal_card import render_journal_html

        journal = task["agent_journal"].data if "agent_journal" in task else None
        return render_journal_html(
            task.pathspec, journal, attempt=task.current_attempt, final=True
        )

    def render_runtime(self, task: Any, data: dict[str, Any]) -> str:
        from metaflow_durability.journal_card import render_journal_html

        user = data.get("user", {})
        return render_journal_html(
            task.pathspec,
            user.get("journal"),
            attempt=user.get("attempt"),
            failed=bool(user.get("failed")),
        )

    def refresh(self, task: Any, data: dict[str, Any]) -> dict[str, Any]:
        return data


CARDS = [AgentCard, AgentJournalCard]
