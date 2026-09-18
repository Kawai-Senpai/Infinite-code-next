"""Rebuild memory bodies that were truncated at write time.

Until the `_compose` fix, a composed body longer than CONTEXT_CHARS was sliced
mid-sentence and stored with a trailing "...". Composition runs *before* the
insert, so the lost text was never in `memories` at all: on this repository 225
of 277 ACTIVE memories (81%) were stored cut, at a median of 903 characters.

The text is recoverable because composition is deterministic and every input
survives. `memories.source_event` points at the `events` row holding the raw
`record()` payload, and the anchors that produced the "Applies to:" list are
still attached. Replaying the composition with the cap removed reproduces the
full body exactly.

Nothing is overwritten destructively: the truncated text is written to
`memory_versions` first, exactly as `correct()` does, so a rebuild can be
inspected or rolled back.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from .compiler import _as_list
from .db import one, rows, write_tx

# A body ending in this was cut by the old cap. Composition never produces a
# trailing ellipsis any other way: every section it appends ends in the text
# the caller supplied.
TRUNCATION_MARK = "..."

# The markers _compose appends after the claim. The claim is whatever precedes
# the first of them, which is how a stored body is split back into its parts.
MARKERS = ("\n\nRecorded while: ", "\n\nWhy: ", "\n\nChanged: ", "\n\nApplies to: ")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _claim_of(body: str) -> str:
    """The claim a stored body starts with, without its context sections."""
    cut = len(body)
    for marker in MARKERS:
        found = body.find(marker)
        if found >= 0:
            cut = min(cut, found)
    return body[:cut]


def _compose_uncapped(claim: str, context: dict[str, Any], is_summary: bool = False) -> str:
    """`compiler._compose` with the length cap removed.

    Deliberately a copy rather than a call with a raised limit: this must keep
    reproducing what the *old* composer produced even if the live one changes
    again, or a rebuild would rewrite bodies it was only meant to restore.
    """
    claim = (claim or "").strip()
    if not claim:
        return claim

    parts = [claim]
    if not is_summary:
        occasion = context.get("occasion") or ""
        if occasion and occasion.lower() not in claim.lower():
            parts.append("Recorded while: " + occasion)

    reasoning = context.get("reasoning") or ""
    if reasoning and reasoning.lower() not in claim.lower():
        parts.append("Why: " + reasoning)

    if context.get("changes"):
        parts.append("Changed: " + "; ".join(context["changes"]))
    if context.get("where"):
        parts.append("Applies to: " + ", ".join(context["where"]))

    return "\n\n".join(parts)


def _context_for(conn: sqlite3.Connection, memory_id: str,
                 payload: dict[str, Any], summary: str) -> dict[str, Any]:
    """Reconstruct the context dict `_event_context` built at record time.

    The "where" list came from the symbols and files the recorder resolved;
    those are exactly the memory's anchors, in insertion order.
    """
    where: list[str] = []
    for anchor in rows(conn.execute(
            "SELECT symbol_path, file_path FROM anchors WHERE memory_id = ?"
            " ORDER BY rowid", (memory_id,))):
        label = anchor["symbol_path"] or anchor["file_path"]
        if label and label not in where:
            where.append(label)

    return {
        "occasion": summary or "",
        "reasoning": str(payload.get("reasoning") or "").strip(),
        "where": where[:6],
        "changes": [c for c in _as_list(payload.get("changes")) if c][:4],
    }


def plan(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Every truncated memory that can be rebuilt, and by how much.

    Read-only: this is what `--dry-run` reports, so a rebuild can be inspected
    before anything is written.
    """
    found: list[dict[str, Any]] = []
    for row in rows(conn.execute(
            "SELECT memory_id, body, source_event FROM memories"
            " WHERE status = 'ACTIVE' AND body LIKE '%...'")):
        if not row["source_event"]:
            continue
        event = one(conn.execute(
            "SELECT payload, summary FROM events WHERE event_id = ?", (row["source_event"],)))
        if not event:
            continue
        try:
            payload = json.loads(event["payload"])
        except (ValueError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue

        context = _context_for(conn, row["memory_id"], payload, event["summary"])
        rebuilt = _compose_uncapped(_claim_of(row["body"]), context)
        # Only a longer body is a recovery. Equal or shorter means the inputs
        # no longer reproduce what was stored, and rewriting would lose text.
        if len(rebuilt) > len(row["body"]):
            found.append({
                "memory_id": row["memory_id"],
                "stored_chars": len(row["body"]),
                "rebuilt_chars": len(rebuilt),
                "recovered": len(rebuilt) - len(row["body"]),
                "body": rebuilt,
            })
    return found


def rebuild(conn: sqlite3.Connection, dry_run: bool = False) -> dict[str, Any]:
    """Restore every recoverable body. Returns what was done."""
    found = plan(conn)
    if dry_run or not found:
        return {"ok": True, "dry_run": dry_run, "rebuilt": 0 if dry_run else 0,
                "candidates": len(found),
                "recovered_chars": sum(f["recovered"] for f in found),
                "memories": [{k: v for k, v in f.items() if k != "body"} for f in found]}

    stamp = now()
    with write_tx(conn):
        for item in found:
            memory = one(conn.execute(
                "SELECT kind, body, severity, status, version FROM memories"
                " WHERE memory_id = ?", (item["memory_id"],)))
            if memory is None:
                continue
            # Keep the truncated text, the same way correct() does. A rebuild
            # that cannot be inspected afterwards is not safe to run.
            conn.execute(
                "INSERT OR IGNORE INTO memory_versions (memory_id, version, kind, body,"
                " severity, status, changed_at, changed_by, reason) VALUES (?,?,?,?,?,?,?,?,?)",
                (item["memory_id"], memory["version"], memory["kind"], memory["body"],
                 memory["severity"], memory["status"], stamp, "recover",
                 "restored text truncated at write time by the old CONTEXT_CHARS cap"))
            # claim is left alone: the claim was never truncated, only the
            # context sections after it, so it is still correct.
            conn.execute(
                "UPDATE memories SET body = ?, version = version + 1, updated_at = ?"
                " WHERE memory_id = ?", (item["body"], stamp, item["memory_id"]))

    return {"ok": True, "dry_run": False, "rebuilt": len(found),
            "candidates": len(found),
            "recovered_chars": sum(f["recovered"] for f in found),
            "memories": [{k: v for k, v in f.items() if k != "body"} for f in found]}
