"""Targeted diagnostics over a narrowed subgraph.

PLAN.md "search should actively look for problems", PLAN 2 section 7.

The governing constraint is in the plan's own words: do not run the equivalent
of CodeQL over a 20-million-line workspace on every search. Every check here
takes the already-narrowed candidate set from `investigate()` and asks one
bounded question of it. If the subgraph is 43 symbols, these are 43-symbol
queries, not repository scans.

What separates these from a linter is that they are *knowledge*-aware. A linter
can see that a function has no test. Only this graph knows that the function is
governed by an invariant recorded after a production incident, and that the
caller which bypasses it was added after that invariant was last verified.

Each detector returns findings shaped for an agent to act on: what is wrong,
why it matters, and what to do. Confidence is explicit, and anything inferred
says so - these inform, they do not gate.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any

from .db import rows

# Names that imply "everything should go through me". A caller reaching past
# one of these to the thing it wraps is the classic bypass.
WRAPPER_HINTS = re.compile(
    r"(coordinator|manager|guard|gateway|wrapper|middleware|validator|sanitiz|"
    r"authoriz|authenticat|throttl|limiter|broker|dispatcher|supervisor)", re.I)

DEPRECATION_HINTS = re.compile(
    r"\b(deprecat|obsolete|legacy|do not use|don't use|superseded|to be removed|"
    r"scheduled for removal|use .* instead)\b", re.I)

TEST_PATH = re.compile(r"(^|/)(tests?|spec|__tests__)/|_test\.|\.test\.|\.spec\.|test_", re.I)


def _is_test(symbol: dict[str, Any]) -> bool:
    name = (symbol.get("name") or "").lower()
    path = (symbol.get("last_known_path") or "").lower()
    return bool(name.startswith("test") or name.endswith("test")
                or "spec" in name or TEST_PATH.search(path))


def _chunk(conn: sqlite3.Connection, sql: str, symbol_ids: list[str],
           repeats: int = 1, extra: tuple = ()) -> list[dict[str, Any]]:
    """Run a query whose IN-list is the candidate set, in safe chunks."""
    out: list[dict[str, Any]] = []
    for start in range(0, len(symbol_ids), 300):
        part = symbol_ids[start:start + 300]
        placeholders = ",".join("?" for _ in part)
        args: tuple = ()
        for _ in range(repeats):
            args = args + tuple(part)
        out.extend(rows(conn.execute(sql.replace("{ids}", placeholders), args + extra)))
    return out


# ------------------------------------------------------------------ detectors

def bypassed_wrappers(conn: sqlite3.Connection, symbol_ids: list[str]) -> list[dict[str, Any]]:
    """Callers that reach past a coordinator/guard straight to what it wraps.

    "Is there a caller bypassing the expected wrapper?" - PLAN.md. The graph
    knows the wrapper calls the target, and that a warning says to go through
    it. A caller that calls the target *without* going through the wrapper is
    exactly what the warning was written to prevent.
    """
    findings: list[dict[str, Any]] = []

    guarded = _chunk(conn, """
        SELECT DISTINCT w.symbol_id AS wrapper_id, w.symbol_path AS wrapper,
               t.symbol_id AS target_id, t.symbol_path AS target,
               m.memory_id, m.title, m.severity
        FROM code_edges e
        JOIN symbols w ON w.symbol_id = e.from_id
        JOIN symbols t ON t.symbol_id = e.to_id
        JOIN memory_edges me ON me.to_id = w.symbol_id AND me.kind='APPLIES_TO'
        JOIN memories m ON m.memory_id = me.from_id
        WHERE e.kind='CALLS' AND e.status='ACTIVE'
          AND w.status='ACTIVE' AND t.status='ACTIVE'
          AND m.status='ACTIVE' AND m.kind IN ('warning','invariant','contract')
          AND t.symbol_id IN ({ids})
        LIMIT 60
    """, symbol_ids)

    for row in guarded:
        if not WRAPPER_HINTS.search(row["wrapper"] or ""):
            continue
        others = rows(conn.execute(
            "SELECT s.symbol_path, s.last_known_path FROM code_edges e"
            " JOIN symbols s ON s.symbol_id = e.from_id"
            " WHERE e.to_id=? AND e.kind='CALLS' AND e.status='ACTIVE'"
            " AND s.status='ACTIVE' AND s.symbol_id != ? LIMIT 12",
            (row["target_id"], row["wrapper_id"])))
        bypassers = [o["symbol_path"] for o in others if not _is_test(o)]
        if not bypassers:
            continue
        findings.append({
            "severity": "high" if row["severity"] in ("critical", "high") else "medium",
            "kind": "bypassed_wrapper",
            "detail": ", ".join(bypassers[:4]) + " call " + row["target"]
                      + " directly, bypassing " + row["wrapper"]
                      + ", which is governed by '" + (row["title"] or "") + "'",
            "memory_id": row["memory_id"],
            "confidence": 0.6,
            "recommendation": "confirm these paths are meant to skip "
                              + row["wrapper"] + "; if not, route them through it",
        })
    return findings[:4]


# How far to look for a test. A test almost never calls the governed function
# directly - it drives a public entry point that calls it. Measured on this
# codebase, `Indexer.resolve_calls` is covered by two tests, both two hops
# away, and a one-hop check reported it untested.
TEST_REACH_HOPS = 3


def _test_reaches(conn: sqlite3.Connection, symbol_id: str,
                  hops: int = TEST_REACH_HOPS) -> bool:
    """Is this symbol reachable from any test, within `hops` calls?

    Breadth-first over reversed CALLS edges, bounded in both depth and breadth
    so a hot symbol with hundreds of callers cannot turn one diagnostic into a
    graph walk.
    """
    seen = {symbol_id}
    frontier = [symbol_id]
    for _ in range(max(1, hops)):
        if not frontier:
            return False
        batch = frontier[:60]
        placeholders = ",".join("?" for _ in batch)
        callers = rows(conn.execute(
            "SELECT s.symbol_id, s.name, s.symbol_path, s.last_known_path"
            " FROM code_edges e JOIN symbols s ON s.symbol_id = e.from_id"
            " WHERE e.to_id IN (" + placeholders + ") AND e.kind='CALLS'"
            " AND e.status='ACTIVE' AND s.status='ACTIVE' LIMIT 200", tuple(batch)))
        frontier = []
        for caller in callers:
            if _is_test(caller):
                return True
            if caller["symbol_id"] not in seen:
                seen.add(caller["symbol_id"])
                frontier.append(caller["symbol_id"])
    return False


def untested_callers(conn: sqlite3.Connection, symbol_ids: list[str]) -> list[dict[str, Any]]:
    """Code governed by a memory, with no test anywhere in its caller set.

    "Are there callers without tests?" - PLAN.md. Scoped to symbols that carry
    an invariant, warning or contract: an untested helper is ordinary, but an
    untested *invariant* is a live risk.
    """
    governed = _chunk(conn, """
        SELECT DISTINCT s.symbol_id, s.symbol_path, s.last_known_path,
               m.memory_id, m.title, m.kind AS memory_kind, m.severity
        FROM memory_edges me
        JOIN memories m ON m.memory_id = me.from_id
        JOIN symbols s ON s.symbol_id = me.to_id
        WHERE me.kind='APPLIES_TO' AND me.status='ACTIVE'
          AND m.status='ACTIVE' AND m.kind IN ('invariant','warning','contract','security')
          AND s.status='ACTIVE' AND s.symbol_id IN ({ids})
        LIMIT 40
    """, symbol_ids)

    findings: list[dict[str, Any]] = []
    for row in governed:
        if _is_test(row):
            continue
        has_test = _test_reaches(conn, row["symbol_id"])
        guarded_by = rows(conn.execute(
            "SELECT 1 FROM memory_edges WHERE from_id=? AND kind='GUARDED_BY'"
            " AND status='ACTIVE' LIMIT 1", (row["memory_id"],)))
        if has_test or guarded_by:
            continue
        findings.append({
            "severity": "high" if row["severity"] in ("critical", "high") else "medium",
            "kind": "untested_invariant",
            "detail": row["symbol_path"] + " carries " + row["memory_kind"] + " '"
                      + (row["title"] or "") + "' but no test reaches it",
            "memory_id": row["memory_id"],
            "confidence": 0.7,
            "recommendation": "add a regression test, then record it with"
                              " record(tests=[...]) so it becomes a GUARDED_BY edge",
        })
    return findings[:4]


def deprecated_with_live_callers(conn: sqlite3.Connection,
                                 symbol_ids: list[str]) -> list[dict[str, Any]]:
    """"Does a deprecated symbol still have live callers?" - PLAN.md.

    Deprecation is read from recorded knowledge, not from source comments: a
    memory saying "do not use X" is a stronger, more current signal than a
    decorator someone added years ago.
    """
    marked = _chunk(conn, """
        SELECT DISTINCT s.symbol_id, s.symbol_path, m.memory_id, m.title, m.body
        FROM memory_edges me
        JOIN memories m ON m.memory_id = me.from_id
        JOIN symbols s ON s.symbol_id = me.to_id
        WHERE me.kind='APPLIES_TO' AND me.status='ACTIVE' AND m.status='ACTIVE'
          AND s.status='ACTIVE' AND s.symbol_id IN ({ids})
        LIMIT 60
    """, symbol_ids)

    findings: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in marked:
        blob = (row["title"] or "") + " " + (row["body"] or "")
        if not DEPRECATION_HINTS.search(blob) or row["symbol_id"] in seen:
            continue
        seen.add(row["symbol_id"])
        callers = rows(conn.execute(
            "SELECT s.symbol_path, s.last_known_path FROM code_edges e"
            " JOIN symbols s ON s.symbol_id = e.from_id"
            " WHERE e.to_id=? AND e.kind='CALLS' AND e.status='ACTIVE'"
            " AND s.status='ACTIVE' LIMIT 12", (row["symbol_id"],)))
        live = [c["symbol_path"] for c in callers if not _is_test(c)]
        if not live:
            continue
        findings.append({
            "severity": "medium",
            "kind": "deprecated_with_callers",
            "detail": row["symbol_path"] + " is marked deprecated ('"
                      + (row["title"] or "")[:70] + "') but still called by "
                      + ", ".join(live[:4]),
            "memory_id": row["memory_id"],
            "confidence": 0.65,
            "recommendation": "migrate these callers, or supersede the deprecation"
                              " memory if it no longer holds",
        })
    return findings[:3]


def diverging_implementations(conn: sqlite3.Connection,
                              symbol_ids: list[str]) -> list[dict[str, Any]]:
    """Structurally equivalent code where only one side carries the warning.

    Covers two of the plan's questions at once: "do two implementations violate
    the same invariant differently?" and "is there a warning attached to one
    implementation but another equivalent implementation lacks it?"

    Equivalence uses the skeleton fingerprint - identical structure, different
    identifiers - which is the same signal cascade step 3 uses to tell a rename
    from a rewrite. Two functions with one skeleton are near-certainly
    copy-paste siblings, and a rule that applies to one usually applies to both.
    """
    governed = _chunk(conn, """
        SELECT DISTINCT s.symbol_id, s.symbol_path, s.skeleton_fingerprint,
               m.memory_id, m.title, m.kind AS memory_kind, m.severity
        FROM memory_edges me
        JOIN memories m ON m.memory_id = me.from_id
        JOIN symbols s ON s.symbol_id = me.to_id
        WHERE me.kind='APPLIES_TO' AND me.status='ACTIVE' AND m.status='ACTIVE'
          AND m.kind IN ('invariant','warning','contract','security')
          AND s.status='ACTIVE' AND s.skeleton_fingerprint IS NOT NULL
          AND s.symbol_id IN ({ids})
        LIMIT 40
    """, symbol_ids)

    findings: list[dict[str, Any]] = []
    for row in governed:
        twins = rows(conn.execute(
            "SELECT symbol_id, symbol_path, last_known_path FROM symbols"
            " WHERE skeleton_fingerprint=? AND status='ACTIVE' AND symbol_id != ? LIMIT 6",
            (row["skeleton_fingerprint"], row["symbol_id"])))
        naked = []
        for twin in twins:
            if _is_test(twin):
                continue
            covered = rows(conn.execute(
                "SELECT 1 FROM memory_edges WHERE to_id=? AND from_id=?"
                " AND status='ACTIVE' LIMIT 1", (twin["symbol_id"], row["memory_id"])))
            if not covered:
                naked.append(twin["symbol_path"])
        if not naked:
            continue
        findings.append({
            "severity": "high" if row["severity"] in ("critical", "high") else "medium",
            "kind": "unguarded_equivalent",
            "detail": ", ".join(naked[:3]) + " are structurally identical to "
                      + row["symbol_path"] + ", which carries " + row["memory_kind"]
                      + " '" + (row["title"] or "") + "', but they are not covered by it",
            "memory_id": row["memory_id"],
            "confidence": 0.55,
            "recommendation": "if the rule applies to these too, attach it with"
                              " record(symbols=[...]); if not, narrow the memory's scope",
        })
    return findings[:3]


def drifted_from_decision(conn: sqlite3.Connection,
                          symbol_ids: list[str]) -> list[dict[str, Any]]:
    """"Did the implementation diverge from a documented decision?" - PLAN.md.

    A decision whose anchored code has changed structurally since the decision
    was last verified. Distinct from generic staleness: this is specifically
    "someone decided X, and the code that implemented X is no longer the code
    that was decided upon".
    """
    found = _chunk(conn, """
        SELECT DISTINCT m.memory_id, m.title, s.symbol_path, a.status AS anchor_status,
               a.last_verified_commit, a.anchor_confidence
        FROM anchors a
        JOIN memories m ON m.memory_id = a.memory_id
        JOIN symbols s ON s.symbol_id = a.symbol_id
        WHERE m.kind IN ('decision','contract') AND m.status='ACTIVE'
          AND a.status IN ('NEEDS_REVIEW','DRIFTED')
          AND a.symbol_id IN ({ids})
        LIMIT 20
    """, symbol_ids)

    return [{
        "severity": "high",
        "kind": "implementation_drifted_from_decision",
        "detail": "decision '" + (row["title"] or "") + "' was implemented by "
                  + row["symbol_path"] + ", which has changed since ("
                  + str(row["anchor_status"]) + ")",
        "memory_id": row["memory_id"],
        "confidence": 0.8,
        "recommendation": "confirm the decision still describes the code, then"
                          " memory(action='verify'), or supersede it",
    } for row in found][:3]


ALL_DETECTORS = (
    bypassed_wrappers,
    untested_callers,
    deprecated_with_live_callers,
    diverging_implementations,
    drifted_from_decision,
)


def run_all(conn: sqlite3.Connection, symbol_ids: list[str]) -> list[dict[str, Any]]:
    """Every graph-derived detector over the narrowed subgraph.

    One failing detector must never take the search with it: a diagnostic is an
    enhancement to the answer, not a precondition for it.
    """
    if not symbol_ids:
        return []
    findings: list[dict[str, Any]] = []
    for detector in ALL_DETECTORS:
        try:
            findings.extend(detector(conn, symbol_ids))
        except sqlite3.Error:
            continue
    return findings
