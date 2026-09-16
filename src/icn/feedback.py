"""Explicit votes on memories: helpful, not helpful, stale, wrong.

Access counts were the only usage signal, and they cannot tell "this answered
the question" from "this wasted a read". Hooks now put memories in front of an
agent unasked, which makes that distinction matter: a memory that is delivered
on every edit of a file and never helps is a tax on every session.

Each signal has one bounded effect, and none of them deletes anything:

    helpful      counts up, nudges confidence up
    not_helpful  counts down; enough of them keeps it out of hook delivery
    stale        the code moved on: anchors go NEEDS_REVIEW until verified
    wrong        the claim is false: NEEDS_REVIEW, and confidence drops below
                 the line hooks and rule promotion will use, until a verify or
                 a correction settles it
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any

from . import ids
from .db import one, write_tx


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

SIGNALS = ("helpful", "not_helpful", "stale", "wrong")

# Below this, a memory is not volunteered by hooks or offered for promotion.
# It stays searchable: an agent asking directly still sees it, labelled.
TRUST_FLOOR = 0.4


def record(conn: sqlite3.Connection, memory_id: str, signal: str, reason: str = "",
           actor: str = "agent") -> dict[str, Any]:
    signal = (signal or "").strip().lower().replace("-", "_").replace(" ", "_")
    if signal == "unhelpful":
        signal = "not_helpful"
    if signal not in SIGNALS:
        return {"ok": False, "error": f"signal must be one of {', '.join(SIGNALS)}"}
    memory = one(conn.execute("SELECT memory_id, status, confidence FROM memories WHERE memory_id=?",
                              (memory_id,)))
    if memory is None:
        return {"ok": False, "error": f"unknown memory {memory_id}"}
    if signal in ("stale", "wrong") and not reason.strip():
        return {"ok": False, "error": f"signal={signal!r} needs a reason: what is out of date or "
                                      "false, so the next agent can verify or correct it"}

    effects: list[str] = []
    with write_tx(conn):
        conn.execute(
            "INSERT INTO memory_feedback (feedback_id, memory_id, signal, reason, actor, created_at)"
            " VALUES (?,?,?,?,?,?)", (ids.new_id("fb"), memory_id, signal, reason.strip() or None,
                                     actor, now()))
        if signal == "helpful":
            conn.execute("UPDATE memories SET helpful_count = COALESCE(helpful_count,0) + 1,"
                         " confidence = MIN(1.0, confidence + 0.05) WHERE memory_id=?", (memory_id,))
            effects.append("ranks higher in search")
        elif signal == "not_helpful":
            conn.execute("UPDATE memories SET unhelpful_count = COALESCE(unhelpful_count,0) + 1"
                         " WHERE memory_id=?", (memory_id,))
            effects.append("ranks lower; repeated votes stop hooks volunteering it")
        else:
            conn.execute("UPDATE anchors SET status='NEEDS_REVIEW' WHERE memory_id=? AND status IN"
                         " ('ACTIVE','DRIFTED')", (memory_id,))
            conn.execute("UPDATE memories SET unhelpful_count = COALESCE(unhelpful_count,0) + 1"
                         " WHERE memory_id=?", (memory_id,))
            effects.append("anchors marked NEEDS_REVIEW until verified")
            if signal == "wrong":
                conn.execute("UPDATE memories SET confidence = MIN(confidence, 0.3) WHERE memory_id=?",
                             (memory_id,))
                effects.append("confidence dropped: hooks and rule promotion skip it until "
                               "memory(action='verify') or a correction")

    counts = one(conn.execute("SELECT helpful_count, unhelpful_count, confidence FROM memories"
                              " WHERE memory_id=?", (memory_id,)))
    return {"ok": True, "memory_id": memory_id, "signal": signal, "effects": effects,
            "helpful": counts["helpful_count"] or 0, "not_helpful": counts["unhelpful_count"] or 0,
            "confidence": round(counts["confidence"], 2),
            "next": ("correct it with memory(action='correct') or supersede it, if you know the "
                     "right claim" if signal == "wrong" else None)}


def withheld_from_delivery(memory: dict[str, Any]) -> bool:
    """Should hooks and promotion skip this memory? Search never does."""
    if (memory.get("confidence") if memory.get("confidence") is not None else 0.8) < TRUST_FLOOR:
        return True
    helpful = memory.get("helpful_count") or 0
    unhelpful = memory.get("unhelpful_count") or 0
    return unhelpful >= 2 and unhelpful > helpful * 2
