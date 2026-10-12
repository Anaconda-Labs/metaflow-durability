"""HTML rendering of an `agent_journal` for the Metaflow card."""

from __future__ import annotations

import json
import re
from html import escape
from typing import Any

_MAX_DETAIL = 240

_STYLE = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       margin: 20px; color: #333; }
h2 { border-bottom: 1px solid #ddd; padding-bottom: 8px; }
h3 { margin: 20px 0 4px; font-size: 15px; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 16px; }
th, td { border: 1px solid #ddd; padding: 6px 8px; text-align: left; font-size: 13px;
         vertical-align: top; }
th { background: #f5f5f5; font-weight: 600; }
.stat { display: inline-block; margin-right: 24px; }
.stat-value { font-size: 22px; font-weight: 700; }
.stat-label { font-size: 12px; color: #888; text-transform: uppercase; }
.live { color: #1a7f37; } .replayed { color: #8250df; }
code { font-size: 12px; white-space: pre-wrap; word-break: break-word; }
.muted { color: #888; }
h4 { margin: 4px 0; font-size: 13px; }
pre { background: #f5f5f5; padding: 12px; border-radius: 4px; font-size: 13px;
      white-space: pre-wrap; max-height: 400px; overflow: auto; }
.j-key { color: #0550ae; } .j-str { color: #0a3069; } .j-num { color: #953800; }
.j-lit { color: #8250df; }
.markdown { background: #fafafa; border: 1px solid #eee; border-radius: 4px;
            padding: 4px 16px; font-size: 14px; line-height: 1.5; max-height: 600px;
            overflow: auto; }
.markdown table { width: auto; }
.banner { display: flex; gap: 12px; align-items: center; padding: 10px 14px;
          border-radius: 6px; margin: 0 0 16px; font-size: 14px; }
.banner b { font-size: 15px; }
.banner.final { background: #dafbe1; border: 1px solid #1a7f37; color: #116329; }
.banner.failed { background: #ffebe9; border: 1px solid #cf222e; color: #a40e26; }
.banner.running { background: #fff8c5; border: 1px solid #bf8700; color: #7d4e00; }
.markdown code { background: #f0f0f0; padding: 1px 4px; border-radius: 3px; }
.markdown pre code { background: none; padding: 0; }
.markdown blockquote { color: #666; border-left: 3px solid #ddd; margin: 8px 0;
                       padding-left: 12px; }
"""


# Metaflow's runtime-card protocol: the UI watches METAFLOW_RELOAD_TOKEN (Metaflow
# substitutes the placeholder) and calls metaflow_card_update with refresh data. With
# RELOAD_POLICY_ALWAYS the whole card is re-fetched, so the update hook is a no-op.
_RUNTIME_HOOKS = (
    '<script>var METAFLOW_RELOAD_TOKEN = "[METAFLOW_RELOAD_TOKEN]";'
    "window.metaflow_card_update = function (data) { return true; };</script>"
)


def _truncate(text: str) -> str:
    return text if len(text) <= _MAX_DETAIL else text[: _MAX_DETAIL - 3] + "..."


def _operation_label(name: str) -> str:
    operation = name.split("__", 1)[-1]
    if ".call_tool:" in operation:
        return "tool " + operation.split(".call_tool:", 1)[1]
    if operation.startswith("model.request"):
        return "model"
    return operation


def _detail(entry: dict[str, Any]) -> str:
    payload = entry.get("payload")
    if not isinstance(payload, dict):
        return _truncate(json.dumps(payload, default=str, ensure_ascii=False))
    if payload.get("kind") == "response":
        pieces = []
        for part in payload.get("parts", []):
            if part.get("part_kind") == "tool-call":
                args = json.dumps(part.get("args"), default=str, ensure_ascii=False)
                pieces.append(f"-> {part.get('tool_name')}({args})")
            elif part.get("part_kind") == "text":
                pieces.append(str(part.get("content", "")))
        return _truncate("\n".join(pieces))
    if "result" in payload:
        return _truncate(json.dumps(payload["result"], default=str, ensure_ascii=False))
    return _truncate(json.dumps(payload, default=str, ensure_ascii=False))


def _usage(entry: dict[str, Any]) -> tuple[int, int]:
    payload = entry.get("payload")
    usage = payload.get("usage") if isinstance(payload, dict) else None
    if not isinstance(usage, dict):
        return 0, 0
    return int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)


def _stat(value: str, label: str) -> str:
    return (
        f'<div class="stat"><div class="stat-value">{escape(value)}</div>'
        f'<div class="stat-label">{escape(label)}</div></div>'
    )


_JSON_TOKEN = re.compile(
    r'(?P<key>"(?:\\.|[^"\\])*"(?=\s*:))|(?P<str>"(?:\\.|[^"\\])*")'
    r"|(?P<num>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)|(?P<lit>\b(?:true|false|null)\b)"
)


def _json_html(value: Any) -> str:
    text = json.dumps(value, indent=2, default=str, ensure_ascii=False)
    out, last = [], 0
    for match in _JSON_TOKEN.finditer(text):
        out.append(escape(text[last : match.start()]))
        out.append(f'<span class="j-{match.lastgroup}">{escape(match.group())}</span>')
        last = match.end()
    out.append(escape(text[last:]))
    return f'<pre class="json">{"".join(out)}</pre>'


def _text_html(text: str) -> str:
    stripped = text.strip()
    if stripped[:1] in "{[":
        try:
            return _json_html(json.loads(stripped))
        except ValueError:
            pass
    import markdown

    rendered = markdown.markdown(
        text.replace("&", "&amp;").replace("<", "&lt;"),
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    return f'<div class="markdown">{rendered}</div>'


def _final_result_html(entries: list[dict[str, Any]], failed: bool = False) -> str:
    responses = [
        e["payload"]
        for e in entries
        if isinstance(e.get("payload"), dict) and e["payload"].get("kind") == "response"
    ]
    if not responses:
        return '<p class="muted">No final result (run did not finish).</p>'
    blocks, deferred = [], []
    for part in responses[-1].get("parts", []):
        kind = part.get("part_kind")
        if kind == "text":
            blocks.append(_text_html(str(part.get("content", ""))))
        elif kind == "tool-call":
            name = str(part.get("tool_name", ""))
            if name.startswith("final_result"):
                blocks.append(_json_html(part.get("args")))
            else:
                deferred.append(name)
        elif kind == "file":
            media = part.get("content", {}).get("media_type", "file")
            blocks.append(f'<p class="muted">[{escape(str(media))} output]</p>')
    if deferred:
        names = ", ".join(escape(n) for n in deferred)
        blocks.append(f'<p class="muted">Ended awaiting tool calls: {names}</p>')
    if not blocks:
        return '<p class="muted">No final result (run did not finish).</p>'
    heading = "Last model response before the failure" if failed else "Final result"
    return f"<h4>{heading}</h4>{''.join(blocks)}"


def _run_table(entries: list[dict[str, Any]], failed: bool = False) -> str:
    if not entries:
        return '<p class="muted">No operations recorded.</p>'
    rows = []
    for position, entry in enumerate(entries):
        source = "replayed" if entry.get("replayed") else "live"
        tokens_in, tokens_out = _usage(entry)
        tokens = f"{tokens_in} / {tokens_out}" if tokens_in or tokens_out else ""
        rows.append(
            f"<tr><td>{position}</td>"
            f"<td><code>{escape(_operation_label(entry.get('name', '?')))}</code></td>"
            f'<td class="{source}">{source}</td>'
            f"<td>{float(entry.get('duration_ms', 0)):.1f} ms</td>"
            f"<td>{escape(tokens)}</td>"
            f"<td><code>{escape(_detail(entry))}</code></td></tr>"
        )
    return (
        "<table><thead><tr><th>#</th><th>Operation</th><th>Source</th>"
        "<th>Duration</th><th>Tokens in / out</th><th>Detail</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>{_final_result_html(entries, failed)}"
    )


def _sections(journal: dict[str, Any], failed: bool = False) -> str:
    return "".join(
        f"<h3>{escape(agent_name)} "
        f'<span class="muted">run {index}</span></h3>{_run_table(run, failed)}'
        for agent_name, runs in journal.items()
        for index, run in enumerate(runs)
    )


def _banner(attempt: int | None, final: bool, failed: bool = False) -> str:
    if attempt is None:
        return ""
    if failed:
        return (
            f'<div class="banner failed"><b>Attempt {attempt} · failed</b>'
            "<span>journal up to the failure; a later attempt replays its "
            "completed operations</span></div>"
        )
    if not final:
        return (
            f'<div class="banner running"><b>Attempt {attempt}</b>'
            "<span>running, updating live</span></div>"
        )
    retried = (
        f"<span>after {attempt} failed attempt{'s' if attempt > 1 else ''}; "
        "any card listed below this one is from a failed attempt</span>"
        if attempt
        else ""
    )
    return (
        f'<div class="banner final"><b>Attempt {attempt} · final</b>'
        f"<span>completed</span>{retried}</div>"
    )


def render_journal_html(
    pathspec: str,
    journal: dict[str, Any] | None,
    attempt: int | None = None,
    final: bool = False,
    failed: bool = False,
) -> str:
    journal = journal or {}
    entries = [e for runs in journal.values() for run in runs for e in run]
    if not entries:
        body = '<p class="muted">No durable agents ran in this step.</p>'
    else:
        replayed = [e for e in entries if e.get("replayed")]
        live = [e for e in entries if not e.get("replayed")]
        tokens = [_usage(e) for e in live]
        stats = "".join(
            [
                _stat(str(len(journal)), "Agents"),
                _stat(str(sum(len(runs) for runs in journal.values())), "Runs"),
                _stat(str(len(entries)), "Operations"),
                _stat(str(len(replayed)), "Replayed"),
                _stat(f"{sum(t[0] for t in tokens)}", "Input tokens (live)"),
                _stat(f"{sum(t[1] for t in tokens)}", "Output tokens (live)"),
                _stat(
                    f"{sum(float(e.get('duration_ms', 0)) for e in replayed):.0f} ms",
                    "Time saved by replay",
                ),
            ]
        )
        body = f"<div>{stats}</div>{_sections(journal, failed)}"
    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        f"<style>{_STYLE}</style>{_RUNTIME_HOOKS}</head><body>"
        f"<h2>Agent Journal</h2>{_banner(attempt, final, failed)}{body}"
        f'<p class="muted" style="font-size:11px;margin-top:32px">'
        f"{escape(pathspec)}</p></body></html>"
    )
