"""The orientation briefing returned by `workspace(action="open")`.

This exists because of a measured failure, not a hypothesis.

Building this server, every session began the same way: read the plan, read the
modules, re-derive the architecture, rediscover the constraints. All of that
knowledge already existed. The cost was not that it was missing - it was that
nothing *announced* it. `open` reported symbol counts, which tells an agent
nothing about what it is walking into.

An agent does not know what to ask before it knows what is there. So `open`
answers the question it cannot yet phrase:

  * the rules that govern this codebase, highest severity first
  * what has already been tried and rejected, so it is not retried
  * what is unverified right now, so nothing stale is trusted
  * where the knowledge is concentrated, so it knows which areas are mapped

Everything here is a headline with an id. Bodies stay out: this is a map, not
the territory, and `investigate()` is one call away. The whole briefing is
budgeted to stay a rounding error against the cost of rediscovery.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from .db import one, rows

SEVERITY_ORDER = ("critical", "high", "medium", "low")
_SEVERITY_SQL = ("CASE severity WHEN 'critical' THEN 0 WHEN 'high' THEN 1"
                 " WHEN 'medium' THEN 2 ELSE 3 END")


def _count(conn: sqlite3.Connection, sql: str, args: tuple = ()) -> int:
    row = one(conn.execute(sql, args))
    return int(row["n"]) if row else 0


def build(conn: sqlite3.Connection, limit: int = 5) -> dict[str, Any]:
    """Assemble the briefing. Cheap: a handful of indexed aggregate queries."""
    total = _count(conn, "SELECT COUNT(*) AS n FROM memories WHERE status='ACTIVE'")
    if not total:
        return {
            "memories": 0,
            "note": "no knowledge recorded for this repository yet. As you work, "
                    "record() what you learn - especially what you tried that did "
                    "not work, which nothing else in the toolchain captures.",
        }

    rules = rows(conn.execute(
        "SELECT m.memory_id, m.kind, m.severity, m.title,"
        " MIN(a.status) AS anchor_status"
        " FROM memories m LEFT JOIN anchors a ON a.memory_id = m.memory_id"
        " WHERE m.status='ACTIVE' AND m.kind IN ('invariant','warning','contract','security')"
        " GROUP BY m.memory_id"
        " ORDER BY " + _SEVERITY_SQL + ", m.created_at DESC LIMIT ?", (limit,)))

    # The field nothing else in a toolchain records, and the one that most
    # directly prevents repeated work.
    rejected = rows(conn.execute(
        "SELECT memory_id, title FROM memories"
        " WHERE status='ACTIVE' AND kind='failed_attempt'"
        " ORDER BY created_at DESC LIMIT ?", (limit,)))

    unverified = rows(conn.execute(
        "SELECT m.memory_id, m.kind, m.title, a.status AS anchor_status"
        " FROM anchors a JOIN memories m ON m.memory_id = a.memory_id"
        " WHERE m.status='ACTIVE' AND a.status IN ('NEEDS_REVIEW','DRIFTED','ORPHANED')"
        " ORDER BY CASE a.status WHEN 'NEEDS_REVIEW' THEN 0 WHEN 'DRIFTED' THEN 1"
        " ELSE 2 END LIMIT ?", (limit,)))

    # Where knowledge clusters. An agent heading into a well-mapped area should
    # ask first; heading somewhere unmapped, it is on its own.
    hotspots = rows(conn.execute(
        "SELECT s.symbol_path, s.last_known_path, COUNT(*) AS memories"
        " FROM memory_edges e JOIN symbols s ON s.symbol_id = e.to_id"
        " WHERE e.kind='APPLIES_TO' AND e.status='ACTIVE' AND s.status='ACTIVE'"
        " GROUP BY s.symbol_id HAVING memories >= 2"
        " ORDER BY memories DESC LIMIT ?", (limit,)))

    by_kind = {r["kind"]: r["n"] for r in rows(conn.execute(
        "SELECT kind, COUNT(*) AS n FROM memories WHERE status='ACTIVE' GROUP BY kind"))}

    stale = _count(conn,
                   "SELECT COUNT(DISTINCT a.memory_id) AS n FROM anchors a"
                   " JOIN memories m ON m.memory_id = a.memory_id"
                   " WHERE m.status='ACTIVE' AND a.status IN ('NEEDS_REVIEW','DRIFTED','ORPHANED')")

    has_causal = _count(conn,
                        "SELECT COUNT(*) AS n FROM memory_edges WHERE status='ACTIVE'"
                        " AND kind IN ('CAUSED','LED_TO','ESTABLISHED','REFINES','DEPENDS_ON')")

    briefing: dict[str, Any] = {
        "memories": total,
        "by_kind": by_kind,
        "rules": rules,
        "already_rejected": rejected,
        "needs_verification": unverified,
        "knowledge_hotspots": hotspots,
        "stale_count": stale,
    }

    briefing["read_this_first"] = _headline(total, rules, rejected, stale, hotspots, has_causal)
    return briefing


def _headline(total: int, rules: list[dict[str, Any]], rejected: list[dict[str, Any]],
              stale: int, hotspots: list[dict[str, Any]], has_causal: int) -> str:
    """One paragraph an agent reads before doing anything else."""
    parts = [str(total) + " memories recorded here."]

    if rules:
        top = rules[0]
        parts.append(str(len(rules)) + " active rules govern this code, starting with "
                     + top["severity"] + " '" + (top["title"] or "")[:80] + "'.")
    if rejected:
        parts.append(str(len(rejected)) + " approaches have already been tried and rejected -"
                     " check these before designing anything, they are the most"
                     " expensive knowledge here to rediscover.")
    if stale:
        parts.append(str(stale) + " memories are unverified against the current code;"
                     " they are labelled and must be treated as leads, not facts.")
    if hotspots:
        parts.append("Knowledge is densest around "
                     + ", ".join(h["symbol_path"] for h in hotspots[:3]) + ".")
    if has_causal:
        parts.append("Causal history exists: investigate(action='why', symbol=...)"
                     " explains why a given piece of code exists before you change it.")

    parts.append("Call investigate() with what you are about to do, rather than"
                 " reading files to orient yourself.")
    return " ".join(parts)
