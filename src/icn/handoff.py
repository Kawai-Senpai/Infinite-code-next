"""Handoffs: "where I left off", written by one session, claimed once by the next.

Adapted from ai-memory's typed handoff protocol. A handoff is not a memory:
memories are durable facts about code, while a handoff is a baton, meaningful
only until someone picks it up. So it is claimed exactly once, atomically,
and a claimed handoff never reappears as if it were still waiting.

Whoever surfaces it first claims it: the SessionStart hook of the next
session, or workspace(action='open') when no hook is installed. A newer
handoff supersedes any older one still open, so the next session reads the
latest state of the work rather than a pile of them.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from . import ids
from .db import one, rows, write_tx


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

MAX_TEXT = 2000
MAX_ITEMS = 15
MAX_ITEM = 300


def _items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    out = [" ".join(str(v).split())[:MAX_ITEM] for v in value if str(v).strip()]
    return out[:MAX_ITEMS]


def _view(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "handoff_id": row["handoff_id"],
        "summary": row["summary"],
        "open_questions": json.loads(row["open_questions"] or "[]"),
        "next_steps": json.loads(row["next_steps"] or "[]"),
        "files": json.loads(row["files"] or "[]"),
        "from_agent": row["from_agent"],
        "branch": row["branch"],
        "head_commit": row["head_commit"],
        "status": row["status"],
        "created_at": row["created_at"],
        "claimed_at": row["claimed_at"],
        "claimed_by": row["claimed_by"],
    }


def create(conn: sqlite3.Connection, summary: str, open_questions: Any = None,
           next_steps: Any = None, files: Any = None, from_agent: str = "agent",
           branch: str | None = None, head_commit: str | None = None) -> dict[str, Any]:
    summary = (summary or "").strip()
    if not summary:
        return {"ok": False, "error": "a handoff needs summary: where the work stands, in a few "
                                      "sentences the next session can act on"}
    handoff_id = ids.new_id("hof")
    with write_tx(conn):
        superseded = [r["handoff_id"] for r in rows(conn.execute(
            "SELECT handoff_id FROM handoffs WHERE status='open'"))]
        conn.execute("UPDATE handoffs SET status='superseded' WHERE status='open'")
        conn.execute(
            "INSERT INTO handoffs (handoff_id, summary, open_questions, next_steps, files,"
            " from_agent, branch, head_commit, status, created_at) VALUES (?,?,?,?,?,?,?,?,'open',?)",
            (handoff_id, summary[:MAX_TEXT], json.dumps(_items(open_questions)),
             json.dumps(_items(next_steps)), json.dumps(_items(files)), from_agent, branch,
             head_commit, now()))
    return {"ok": True, "handoff_id": handoff_id, "superseded": superseded,
            "note": "the next session in this repository receives it once, at session start or "
                    "workspace(action='open')"}


def claim(conn: sqlite3.Connection, claimed_by: str) -> dict[str, Any] | None:
    """Take the newest open handoff, or None. Safe against two sessions racing."""
    with write_tx(conn):
        row = one(conn.execute("SELECT * FROM handoffs WHERE status='open'"
                               " ORDER BY created_at DESC LIMIT 1"))
        if row is None:
            return None
        updated = conn.execute(
            "UPDATE handoffs SET status='claimed', claimed_at=?, claimed_by=?"
            " WHERE handoff_id=? AND status='open'", (now(), claimed_by, row["handoff_id"]))
        if updated.rowcount != 1:
            return None
    return _view(one(conn.execute("SELECT * FROM handoffs WHERE handoff_id=?", (row["handoff_id"],))))


def get(conn: sqlite3.Connection, handoff_id: str) -> dict[str, Any] | None:
    row = one(conn.execute("SELECT * FROM handoffs WHERE handoff_id=?", (handoff_id,)))
    return _view(row) if row else None


def listing(conn: sqlite3.Connection, limit: int = 10) -> list[dict[str, Any]]:
    return [_view(r) for r in rows(conn.execute(
        "SELECT * FROM handoffs ORDER BY created_at DESC LIMIT ?", (max(1, min(limit, 50)),)))]


def cancel(conn: sqlite3.Connection, handoff_id: str) -> dict[str, Any]:
    with write_tx(conn):
        updated = conn.execute("UPDATE handoffs SET status='cancelled' WHERE handoff_id=?"
                               " AND status='open'", (handoff_id,))
    if updated.rowcount != 1:
        found = get(conn, handoff_id)
        return {"ok": False, "error": (f"no handoff {handoff_id}" if found is None else
                                       f"handoff {handoff_id} is {found['status']}, not open")}
    return {"ok": True, "handoff_id": handoff_id, "status": "cancelled"}


def render(view: dict[str, Any]) -> str:
    """The handoff as a model should read it."""
    lines = [f"Handoff from the previous session ({view['from_agent'] or 'agent'}, "
             f"{(view['created_at'] or '')[:16].replace('T', ' ')} UTC"
             + (f", branch {view['branch']}" if view.get("branch") else "") + "):",
             view["summary"]]
    for label, key in (("Open questions", "open_questions"), ("Next steps", "next_steps"),
                       ("Files", "files")):
        if view.get(key):
            lines.append(f"{label}:")
            lines.extend(f"  - {item}" for item in view[key])
    return "\n".join(lines)
