"""Causal chains between memories.

PLAN.md "Memory-to-memory relations are extremely useful". This is the feature
that turns

    "There is a memory mentioning this."

into

    RefreshCoordinator exists because:
      Decision #44 -> production bug #91 -> failed fix #96
                   -> accepted fix #103 -> invariant #117
    Removing it may reintroduce: concurrent token invalidation

The difference is not retrieval quality, it is *shape*. A flat list of five
memories makes an agent reconstruct the story; a chain hands it over. An agent
about to delete a coordinator needs the causal spine, not five paragraphs it
has to order itself.

Two design constraints, both learned the hard way elsewhere in this codebase:

  * These edges are `asserted`, never `inferred`. An agent states the causal
    link explicitly in `record(caused_by=...)`. We do not guess causality from
    timestamps - "B was recorded after A" is not "A caused B", and a wrong
    causal chain is far worse than none, because it reads as authoritative.

  * Traversal is cycle-safe and depth-bounded. Knowledge graphs accrete loops
    (a fix that establishes an invariant that later causes a bug that leads to
    another fix), and a naive walk hangs the server.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from .db import jdump, one, rows

# The narrative vocabulary. Ordered as a story runs, which is also the order
# a chain is rendered in.
CAUSAL_KINDS = {
    "CAUSED":      "led directly to",
    "LED_TO":      "was followed by",
    "ESTABLISHED": "established",
    "REFINES":     "refines",
    "DEPENDS_ON":  "depends on",
}

# Edges that carry the story forward vs. backward when reconstructing "why".
FORWARD = ("CAUSED", "LED_TO", "ESTABLISHED", "REFINES")

MAX_DEPTH = 6
MAX_NODES = 40


def link_causal(conn: sqlite3.Connection, from_memory: str, to_memory: str, kind: str,
                confidence: float = 0.9, evidence: Any = None) -> dict[str, Any]:
    """Assert one narrative edge between two memories."""
    if kind not in CAUSAL_KINDS:
        return {"ok": False, "error": "unknown causal kind " + repr(kind)
                                      + "; expected one of " + str(sorted(CAUSAL_KINDS))}
    if from_memory == to_memory:
        return {"ok": False, "error": "a memory cannot cause itself"}

    for memory_id in (from_memory, to_memory):
        if one(conn.execute("SELECT memory_id FROM memories WHERE memory_id=?", (memory_id,))) is None:
            return {"ok": False, "error": "unknown memory " + repr(memory_id)}

    from . import ids
    from .compiler import now
    conn.execute(
        "INSERT INTO memory_edges (edge_id, from_id, to_id, kind, edge_class, status, confidence,"
        " source, evidence, created_at) VALUES (?,?,?,?,'asserted','ACTIVE',?,'causal',?,?)"
        " ON CONFLICT(from_id, to_id, kind) DO UPDATE SET confidence=excluded.confidence",
        (ids.new_id(ids.EDGE), from_memory, to_memory, kind, confidence, jdump(evidence), now()),
    )
    return {"ok": True, "from": from_memory, "to": to_memory, "kind": kind}


def _neighbours(conn: sqlite3.Connection, memory_id: str,
                direction: str) -> list[dict[str, Any]]:
    """One hop along causal edges, in either direction."""
    placeholders = ",".join("?" for _ in CAUSAL_KINDS)
    if direction == "forward":
        sql = ("SELECT e.kind, e.confidence, m.kind AS memory_kind, m.* FROM memory_edges e"
               " JOIN memories m ON m.memory_id = e.to_id"
               " WHERE e.from_id = ? AND e.status='ACTIVE'"
               " AND e.kind IN (" + placeholders + ")")
    else:
        sql = ("SELECT e.kind, e.confidence, m.kind AS memory_kind, m.* FROM memory_edges e"
               " JOIN memories m ON m.memory_id = e.from_id"
               " WHERE e.to_id = ? AND e.status='ACTIVE'"
               " AND e.kind IN (" + placeholders + ")")
    return rows(conn.execute(sql, (memory_id, *sorted(CAUSAL_KINDS))))


def chain_for(conn: sqlite3.Connection, memory_id: str,
              max_depth: int = MAX_DEPTH) -> dict[str, Any]:
    """Reconstruct the story around one memory: what led here, what followed.

    Walks backward to the root cause and forward to the consequences, so the
    caller gets the whole spine rather than a fragment. Cycle-safe.
    """
    root = one(conn.execute("SELECT * FROM memories WHERE memory_id=?", (memory_id,)))
    if root is None:
        return {"ok": False, "error": "unknown memory " + repr(memory_id)}

    def walk(direction: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        seen = {memory_id}
        frontier = [(memory_id, 0)]
        while frontier and len(out) < MAX_NODES:
            current, depth = frontier.pop(0)
            if depth >= max_depth:
                continue
            for neighbour in _neighbours(conn, current, direction):
                if neighbour["memory_id"] in seen:
                    continue
                seen.add(neighbour["memory_id"])
                out.append({
                    "memory_id": neighbour["memory_id"],
                    # e.kind aliases over m.kind in the join, so the memory's
                    # own type must be read back explicitly. Conflating the two
                    # labelled every node with its neighbour's edge kind.
                    "kind": neighbour["memory_kind"],
                    "relation": neighbour["kind"],
                    "title": neighbour["title"],
                    "severity": neighbour["severity"],
                    "status": neighbour["status"],
                    "depth": depth + 1,
                })
                frontier.append((neighbour["memory_id"], depth + 1))
        return out

    causes = walk("backward")
    effects = walk("forward")
    return {
        "ok": True,
        "memory_id": memory_id,
        "title": root["title"],
        "kind": root["kind"],
        "caused_by": causes,
        "led_to": effects,
        "depth": max((c["depth"] for c in causes + effects), default=0),
    }


def why_does_this_exist(conn: sqlite3.Connection, symbol_id: str,
                        limit: int = 3) -> dict[str, Any] | None:
    """The headline feature: why does this code exist, as a story.

    Finds memories anchored to the symbol, follows each back to its root
    cause, and renders the spine plus what removing it would risk.
    Returns None when there is no causal history - a plain memory list is
    already served elsewhere, and inventing a narrative from one memory would
    be dressing up a fact as a story.
    """
    # Memories that actually carry causal history come first. Taking the first
    # few anchored memories in arbitrary order meant a symbol with one
    # unrelated note plus a real chain reported no history at all.
    anchored = rows(conn.execute(
        "SELECT DISTINCT m.*,"
        " (SELECT COUNT(*) FROM memory_edges ce"
        "    WHERE (ce.from_id = m.memory_id OR ce.to_id = m.memory_id)"
        "      AND ce.status='ACTIVE'"
        "      AND ce.kind IN ('CAUSED','LED_TO','ESTABLISHED','REFINES','DEPENDS_ON')"
        " ) AS causal_links"
        " FROM memory_edges e JOIN memories m ON m.memory_id = e.from_id"
        " WHERE e.to_id = ? AND e.kind IN ('APPLIES_TO','IMPACTS') AND e.status='ACTIVE'"
        " AND m.status='ACTIVE'"
        " ORDER BY causal_links DESC", (symbol_id,)))
    if not anchored or not anchored[0]["causal_links"]:
        return None

    spines: list[dict[str, Any]] = []
    risks: list[str] = []
    guards: list[str] = []

    for memory in anchored[:limit]:
        chain = chain_for(conn, memory["memory_id"])
        if not chain.get("ok"):
            continue
        if not chain["caused_by"] and not chain["led_to"]:
            continue

        ordered = sorted(chain["caused_by"], key=lambda c: -c["depth"])
        spine = [{"memory_id": c["memory_id"], "kind": c["kind"], "title": c["title"],
                  "relation": c["relation"]} for c in ordered]
        spine.append({"memory_id": memory["memory_id"], "kind": memory["kind"],
                      "title": memory["title"], "focus": True,
                      "relation": ordered[-1]["relation"] if ordered else None})
        spine.extend({"memory_id": c["memory_id"], "kind": c["kind"], "title": c["title"],
                      "relation": c["relation"]}
                     for c in sorted(chain["led_to"], key=lambda c: c["depth"]))
        spines.append({"spine": spine})

        # What removing this code may reintroduce is the failure that came
        # BEFORE it - upstream in the chain. A fix is itself stored as
        # bug_history, so scanning the whole chain reports the remedy as the risk.
        for node in chain["caused_by"]:
            # fix_history is deliberately absent: a fix upstream in the chain
            # is what removed the risk, not the risk itself.
            if node.get("kind") in ("bug_history", "incident"):
                risks.append(node["title"])
        for node in chain["caused_by"] + chain["led_to"] + [dict(memory)]:
            if node.get("kind") == "test_evidence":
                guards.append(node["title"])

    if not spines:
        return None

    return {
        "symbol_id": symbol_id,
        "chains": spines,
        "may_reintroduce": sorted(set(risks))[:4],
        "regression_tests": sorted(set(guards))[:4],
        "note": "this code has causal history; read the chain before removing it",
    }


def render(chain: dict[str, Any]) -> str:
    """Human-readable spine, for logs and terminal output."""
    if not chain or not chain.get("chains"):
        return ""
    lines: list[str] = []
    for entry in chain["chains"]:
        rendered = []
        for index, node in enumerate(entry["spine"]):
            label = node["kind"] + ": " + (node["title"] or "")[:60]
            if node.get("focus"):
                label = "* " + label
            if index == 0:
                rendered.append(label)
            else:
                arrow = CAUSAL_KINDS.get(node.get("relation") or "", "led to")
                rendered.append("  --" + arrow + "--> " + label)
        lines.append("\n".join(rendered))
    if chain.get("may_reintroduce"):
        lines.append("may reintroduce: " + "; ".join(chain["may_reintroduce"]))
    return "\n".join(lines)
