"""Structural queries over the code graph: blast radius, paths, import cycles.

These are the questions the memory graph cannot answer on its own. A memory
says why something exists; these say what breaks if you change it, how two
symbols are connected, and where the module graph knots up.

Every answer here carries an epistemic envelope, and that is the point of the
module rather than a decoration on it. Static resolution refuses to guess, so
`impact` returning nothing has always had two very different meanings - nothing
calls this, or the analyzer could not decide who does - and the caller could
not tell them apart. `unresolved_calls` records the second case at index time,
so a result can now say `lower-bound` and name the count it is not showing.
An absent caller is not evidence of no caller, and a tool that implies
otherwise is worse than one that stays silent.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from .db import rows

MAX_DEPTH = 12
DEFAULT_DEPTH = 3
MAX_AFFECTED = 400
MAX_CANDIDATES = 12
MAX_CYCLES = 50
MAX_TRIGGERS_PER_KIND = 15

# Edges walked for reachability. IMPORTS is deliberately excluded: it connects
# files, not symbols, so mixing it in produces paths that do not correspond to
# any call anyone can make.
WALK_KINDS = ("CALLS",)


# --------------------------------------------------------------- symbol lookup

def resolve_target(conn: sqlite3.Connection, name: str,
                   file_hint: str | None = None) -> dict[str, Any]:
    """Find the symbol a caller means.

    Returns either a single match or the candidate list. Picking a winner from
    an ambiguous name is how a blast radius ends up describing the wrong
    function entirely, so ambiguity is reported rather than resolved.
    """
    name = (name or "").strip()
    if not name:
        return {"error": "name is required"}

    # Exact symbol_path first: it is unambiguous by construction when supplied.
    matched = rows(conn.execute(
        "SELECT s.symbol_id, s.name, s.symbol_path, s.kind, s.line_start, s.line_end,"
        " f.path FROM symbols s JOIN files f ON f.file_id = s.file_id"
        " WHERE s.status='ACTIVE' AND s.symbol_path = ?", (name,)))
    if not matched:
        matched = rows(conn.execute(
            "SELECT s.symbol_id, s.name, s.symbol_path, s.kind, s.line_start, s.line_end,"
            " f.path FROM symbols s JOIN files f ON f.file_id = s.file_id"
            " WHERE s.status='ACTIVE' AND s.name = ?", (name,)))

    if file_hint:
        hint = file_hint.replace("\\", "/").lstrip("./")
        narrowed = [m for m in matched if hint in (m["path"] or "")]
        if narrowed:
            matched = narrowed

    if not matched:
        return {"error": f"no active symbol named {name!r}"}
    if len(matched) == 1:
        return {"symbol": matched[0]}
    return {
        "ambiguous": True,
        "total_candidates": len(matched),
        "candidates": matched[:MAX_CANDIDATES],
        "hint": ("pass file_hint (a path fragment, e.g. 'search.py') to disambiguate; "
                 "symbol_path also works for a method, but a module-level function's "
                 "symbol_path is just its name and will not separate these"),
    }


# ------------------------------------------------------------------- envelope

def _dropped_callers(conn: sqlite3.Connection, names: list[str]) -> dict[str, int]:
    """Call sites resolution refused to attribute, keyed by callee name.

    These are exactly the callers an upstream walk cannot reach, because no
    edge was ever written for them.
    """
    if not names:
        return {}
    out: dict[str, int] = {}
    for start in range(0, len(names), 400):
        part = names[start:start + 400]
        placeholders = ",".join("?" for _ in part)
        for row in rows(conn.execute(
                f"SELECT leaf, COUNT(*) n FROM unresolved_calls"
                f" WHERE reason='ambiguous' AND leaf IN ({placeholders})"
                f" GROUP BY leaf", tuple(part))):
            out[row["leaf"]] = row["n"]
    return out


def _envelope(conn: sqlite3.Connection, direction: str, seed_names: list[str],
              inferred_edges: int, external_sites: int) -> dict[str, Any]:
    """Say plainly how complete the answer is, and why it is not more so."""
    dropped = _dropped_callers(conn, seed_names) if direction == "upstream" else {}
    ambiguous = sum(dropped.values())

    boundaries: list[str] = []
    if ambiguous:
        boundaries.append(
            f"{ambiguous} call site(s) name one of these symbols but could not be "
            f"attributed to a single definition, so their callers are missing here.")
    if inferred_edges:
        boundaries.append(
            f"{inferred_edges} of the edges traversed are inferred, not proven - "
            f"resolved by import scope, or by being the only symbol with that name. "
            f"This counts edges walked, so it can exceed counts.inferred, which "
            f"counts distinct symbols reached.")
    if external_sites:
        boundaries.append(
            f"{external_sites} call site(s) left the indexed program (standard "
            f"library, third-party). No in-graph node could have been reached.")

    return {
        "epistemic": "lower-bound" if ambiguous else "exact",
        "boundaries": boundaries,
        "causes": {
            # Every field counts MISSING or UNPROVEN things, never sentences.
            "ambiguous_call_sites": ambiguous,
            "ambiguous_by_name": dropped,
            "inferred_edges_traversed": inferred_edges,
            "external_call_sites": external_sites,
        },
    }


# ---------------------------------------------------------------------- impact

def impact(conn: sqlite3.Connection, target: str, direction: str = "upstream",
           depth: int = DEFAULT_DEPTH, file_hint: str | None = None,
           include_tests: bool = True, min_confidence: float = 0.0) -> dict[str, Any]:
    """Blast radius of changing one symbol.

    upstream   what depends on this - who breaks if the contract changes.
    downstream what this depends on - what it needs in order to work.
    """
    direction = direction if direction in ("upstream", "downstream") else "upstream"
    depth = max(1, min(int(depth or DEFAULT_DEPTH), MAX_DEPTH))

    found = resolve_target(conn, target, file_hint)
    if "symbol" not in found:
        return found
    seed = found["symbol"]

    # upstream walks edges backwards: who points AT me.
    if direction == "upstream":
        sql = ("SELECT e.from_id AS next_id, e.edge_class, e.confidence, e.source"
               " FROM code_edges e WHERE e.to_id = ? AND e.kind='CALLS'"
               " AND e.status='ACTIVE' AND e.confidence >= ?")
    else:
        sql = ("SELECT e.to_id AS next_id, e.edge_class, e.confidence, e.source"
               " FROM code_edges e WHERE e.from_id = ? AND e.kind='CALLS'"
               " AND e.status='ACTIVE' AND e.confidence >= ?")

    seen = {seed["symbol_id"]: 0}
    frontier = [seed["symbol_id"]]
    reached: dict[str, dict[str, Any]] = {}
    inferred_edges = 0

    for hop in range(1, depth + 1):
        next_frontier: list[str] = []
        for symbol_id in frontier:
            for edge in rows(conn.execute(sql, (symbol_id, min_confidence))):
                nxt = edge["next_id"]
                if edge["edge_class"] != "deterministic":
                    inferred_edges += 1
                if nxt in seen:
                    continue
                seen[nxt] = hop
                next_frontier.append(nxt)
                reached[nxt] = {
                    "hops": hop,
                    "edge_class": edge["edge_class"],
                    "confidence": edge["confidence"],
                    "tier": (edge["source"] or "").split(":")[-1],
                }
        frontier = next_frontier
        if not frontier or len(reached) >= MAX_AFFECTED:
            break

    affected: list[dict[str, Any]] = []
    ids_list = list(reached)
    for start in range(0, len(ids_list), 400):
        part = ids_list[start:start + 400]
        placeholders = ",".join("?" for _ in part)
        for row in rows(conn.execute(
                f"SELECT s.symbol_id, s.name, s.symbol_path, s.kind, s.line_start,"
                f" s.line_end, f.path FROM symbols s JOIN files f ON f.file_id=s.file_id"
                f" WHERE s.symbol_id IN ({placeholders}) AND s.status='ACTIVE'",
                tuple(part))):
            row.update(reached[row["symbol_id"]])
            affected.append(row)

    if not include_tests:
        affected = [a for a in affected if not _looks_like_test(a["path"])]

    affected.sort(key=lambda a: (a["hops"], -(a["confidence"] or 0), a["symbol_path"]))
    truncated = len(affected) > MAX_AFFECTED
    affected = affected[:MAX_AFFECTED]

    external = rows(conn.execute(
        "SELECT COUNT(*) n FROM unresolved_calls WHERE from_id=? AND reason='external'",
        (seed["symbol_id"],)))
    external_sites = external[0]["n"] if direction == "downstream" and external else 0

    by_hop: dict[str, int] = {}
    for a in affected:
        by_hop[str(a["hops"])] = by_hop.get(str(a["hops"]), 0) + 1

    result = {
        "target": seed,
        "direction": direction,
        "depth": depth,
        "affected": affected,
        "counts": {
            "total": len(affected),
            "by_hop": by_hop,
            "deterministic": sum(1 for a in affected if a["edge_class"] == "deterministic"),
            "inferred": sum(1 for a in affected if a["edge_class"] != "deterministic"),
            "tests": sum(1 for a in affected if _looks_like_test(a["path"])),
        },
        "truncated": truncated,
    }
    result.update(_envelope(conn, direction, [seed["name"]], inferred_edges,
                            external_sites))
    return result


def _looks_like_test(path: str | None) -> bool:
    p = (path or "").lower()
    return "test" in p or "spec" in p


# ----------------------------------------------------------------------- trace

def trace(conn: sqlite3.Connection, source: str, target: str,
          max_depth: int = 8, file_hint: str | None = None,
          target_file_hint: str | None = None) -> dict[str, Any]:
    """Shortest directed call path from one symbol to another.

    Breadth-first, so the first path found is a shortest one. The weakest edge
    on the path is reported separately: a path is only as trustworthy as its
    least certain hop, and a single `unique_global` link can turn a convincing
    six-step chain into a coincidence of naming.
    """
    max_depth = max(1, min(int(max_depth or 8), MAX_DEPTH))

    start = resolve_target(conn, source, file_hint)
    if "symbol" not in start:
        return {"end": "source", **start}
    goal = resolve_target(conn, target, target_file_hint)
    if "symbol" not in goal:
        return {"end": "target", **goal}

    start_id = start["symbol"]["symbol_id"]
    goal_id = goal["symbol"]["symbol_id"]
    if start_id == goal_id:
        return {"found": True, "path": [start["symbol"]], "hops": 0,
                "epistemic": "exact", "boundaries": [], "note": "source and target are the same symbol"}

    came_from: dict[str, tuple[str, dict[str, Any]]] = {}
    seen = {start_id}
    frontier = [start_id]
    found = False

    for _ in range(max_depth):
        next_frontier: list[str] = []
        for symbol_id in frontier:
            for edge in rows(conn.execute(
                    "SELECT to_id, edge_class, confidence, source FROM code_edges"
                    " WHERE from_id=? AND kind='CALLS' AND status='ACTIVE'",
                    (symbol_id,))):
                nxt = edge["to_id"]
                if nxt in seen:
                    continue
                seen.add(nxt)
                came_from[nxt] = (symbol_id, edge)
                if nxt == goal_id:
                    found = True
                    break
                next_frontier.append(nxt)
            if found:
                break
        if found:
            break
        frontier = next_frontier
        if not frontier:
            break

    if not found:
        envelope = _envelope(conn, "upstream", [goal["symbol"]["name"]], 0, 0)
        return {
            "found": False,
            "source": start["symbol"],
            "target": goal["symbol"],
            "searched_depth": max_depth,
            "note": ("No call path within this depth. A missing path is not proof "
                     "of no path - see boundaries."),
            **envelope,
        }

    # Walk the parent chain back and re-expand each id into a symbol row.
    chain: list[tuple[str, dict[str, Any] | None]] = [(goal_id, None)]
    cursor = goal_id
    while cursor != start_id:
        parent, edge = came_from[cursor]
        chain.append((parent, edge))
        cursor = parent
    chain.reverse()

    ids_list = [c[0] for c in chain]
    placeholders = ",".join("?" for _ in ids_list)
    lookup = {r["symbol_id"]: r for r in rows(conn.execute(
        f"SELECT s.symbol_id, s.name, s.symbol_path, s.kind, s.line_start, s.line_end,"
        f" f.path FROM symbols s JOIN files f ON f.file_id=s.file_id"
        f" WHERE s.symbol_id IN ({placeholders})", tuple(ids_list)))}

    path: list[dict[str, Any]] = []
    weakest = 1.0
    inferred = 0
    for index, (symbol_id, _) in enumerate(chain):
        node = dict(lookup.get(symbol_id, {"symbol_id": symbol_id}))
        if index + 1 < len(chain):
            _, edge = chain[index + 1]
            if edge:
                node["calls_next_via"] = {
                    "edge_class": edge["edge_class"],
                    "confidence": edge["confidence"],
                    "tier": (edge["source"] or "").split(":")[-1],
                }
                weakest = min(weakest, edge["confidence"] or 0.0)
                if edge["edge_class"] != "deterministic":
                    inferred += 1
        path.append(node)

    return {
        "found": True,
        "hops": len(path) - 1,
        "path": path,
        "weakest_link_confidence": weakest,
        "inferred_hops": inferred,
        "epistemic": "exact" if inferred == 0 else "lower-bound",
        "boundaries": ([] if inferred == 0 else [
            f"{inferred} of {len(path) - 1} hop(s) on this path are inferred rather "
            f"than proven; the path is only as good as its weakest link "
            f"({weakest}). A shorter, fully proven path may also exist."]),
    }


# ---------------------------------------------------------------------- cycles

def import_cycles(conn: sqlite3.Connection) -> dict[str, Any]:
    """Directed cycles in the file import graph, via Tarjan's SCC algorithm.

    Reports `component_count` as the number to act on. Elementary cycle counts
    swing wildly - cutting one import can remove thousands at once - so a count
    of cycles is useless for trending, while the number of independent knots is
    stable and is what a fix actually reduces.

    Only resolved file-to-file edges participate, and only eager ones. An
    `extern:` edge leaves the indexed program and cannot close a loop; a
    deferred import - written inside a function body, or a TypeScript
    `import type` - cannot force a module-initialisation order, so it cannot
    make the modules impossible to initialise. Counting deferred imports
    reports the standard fix for a cycle as if it were the cycle.
    """
    edges: dict[str, set[str]] = {}
    nodes: set[str] = set()
    for row in rows(conn.execute(
            "SELECT e.from_id, e.to_id FROM code_edges e"
            " JOIN files f ON f.file_id = e.to_id"
            " WHERE e.kind='IMPORTS' AND e.status='ACTIVE' AND f.status='ACTIVE'"
            "   AND COALESCE(e.source,'') <> 'tree-sitter:deferred'")):
        edges.setdefault(row["from_id"], set()).add(row["to_id"])
        nodes.add(row["from_id"])
        nodes.add(row["to_id"])

    if not nodes:
        return {"status": "no_import_graph", "component_count": 0, "cycles": [],
                "note": "No resolved file-to-file imports. Re-index if this is unexpected."}

    components = _tarjan(nodes, edges)
    knots = [c for c in components
             if len(c) > 1 or (len(c) == 1 and next(iter(c)) in edges.get(next(iter(c)), ()))]

    paths = {r["file_id"]: r["path"] for r in rows(conn.execute(
        "SELECT file_id, path FROM files WHERE status='ACTIVE'"))}

    reported = []
    for component in sorted(knots, key=len, reverse=True)[:MAX_CYCLES]:
        cycle = _representative_cycle(component, edges)
        reported.append({
            "size": len(component),
            "cycle": [paths.get(n, n) for n in cycle],
            "members": sorted(paths.get(n, n) for n in component),
        })

    return {
        "status": "cycles_found" if knots else "clean",
        "component_count": len(knots),
        "components": reported,
        "truncated": len(knots) > MAX_CYCLES,
        "note": ("component_count is the number to act on and to trend: one removed "
                 "import can dissolve a whole component. Each entry shows one "
                 "representative cycle, not every cycle through those files."),
    }


def _tarjan(nodes: set[str], edges: dict[str, set[str]]) -> list[set[str]]:
    """Strongly connected components, iteratively.

    Iterative rather than recursive on purpose: a deep import chain in a large
    repository overruns CPython's recursion limit, and the failure mode is an
    exception in the middle of a health check rather than a wrong answer.
    """
    index_of: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    result: list[set[str]] = []
    counter = 0

    for root in nodes:
        if root in index_of:
            continue
        work: list[tuple[str, list[str]]] = [(root, list(edges.get(root, ())))]
        index_of[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)

        while work:
            node, pending = work[-1]
            if pending:
                child = pending.pop()
                if child not in index_of:
                    index_of[child] = low[child] = counter
                    counter += 1
                    stack.append(child)
                    on_stack.add(child)
                    work.append((child, list(edges.get(child, ()))))
                elif child in on_stack:
                    low[node] = min(low[node], index_of[child])
                continue

            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[node])
            if low[node] == index_of[node]:
                component: set[str] = set()
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.add(member)
                    if member == node:
                        break
                result.append(component)
    return result


def _representative_cycle(component: set[str], edges: dict[str, set[str]]) -> list[str]:
    """One concrete cycle through a component, found by DFS back to the start."""
    start = sorted(component)[0]
    stack = [(start, [start])]
    visited: set[str] = set()
    while stack:
        node, path = stack.pop()
        for nxt in sorted(edges.get(node, ())):
            if nxt not in component:
                continue
            if nxt == start:
                return path + [start]
            if nxt in visited:
                continue
            visited.add(nxt)
            stack.append((nxt, path + [nxt]))
    return sorted(component)


# ----------------------------------------------------------------- entry points

def entry_points(conn: sqlite3.Connection, kind: str = "",
                 limit: int = 200) -> dict[str, Any]:
    """Where control enters this program, grouped by kind."""
    sql = ("SELECT e.kind, e.detail, e.evidence, s.symbol_path, s.line_start, f.path"
           " FROM entry_points e JOIN symbols s ON s.symbol_id = e.symbol_id"
           " JOIN files f ON f.file_id = s.file_id WHERE s.status='ACTIVE'")
    params: tuple[Any, ...] = ()
    if kind:
        sql += " AND e.kind = ?"
        params = (kind,)
    sql += " ORDER BY e.kind, f.path, s.line_start"

    found = rows(conn.execute(sql, params))
    counts: dict[str, int] = {}
    for row in found:
        counts[row["kind"]] = counts.get(row["kind"], 0) + 1

    return {
        "counts": counts,
        "total": len(found),
        "entry_points": found[:limit],
        "truncated": len(found) > limit,
        "note": ("Only declared entry points are listed. A handler registered at "
                 "runtime - added to a dispatch table, wired by a framework - "
                 "leaves no declaration to read and will be missing here."),
    }


def areas(conn: sqlite3.Connection, limit: int = 12,
          include_tests: bool = False) -> dict[str, Any]:
    """Functional areas of the codebase, from the call graph."""
    from . import flows as flows_mod

    described = flows_mod.describe_areas(conn, limit=limit, include_tests=include_tests)
    total = rows(conn.execute(
        "SELECT COUNT(DISTINCT community_id) n FROM communities"))[0]["n"]
    if not described:
        return {"areas": [], "total": 0,
                "note": ("No areas stored. They are computed during indexing; "
                         "run workspace(action='reindex') if this is unexpected.")}
    return {
        "areas": described,
        "total": total,
        "shown": len(described),
        "note": ("Areas come from the call graph, not the directory layout, so an "
                 "area can span directories and a directory can split across "
                 "areas. test_share says how much of an area is test code; with "
                 "include_tests=False a mostly-test area sorts last rather than "
                 "being hidden."),
    }


def reaching_entry_points(conn: sqlite3.Connection, target: str,
                          file_hint: str | None = None,
                          depth: int = 6) -> dict[str, Any]:
    """Which entry points can reach this symbol - what can trigger this code.

    Answers the question a blast radius does not: not "what else changes" but
    "what can a user actually do that runs this". An area with no reaching
    entry point is either a library or dead, and telling those apart is the
    first thing anyone asks about unfamiliar code.
    """
    depth = max(1, min(int(depth or 6), MAX_DEPTH))
    found = resolve_target(conn, target, file_hint)
    if "symbol" not in found:
        return found
    seed = found["symbol"]

    known = {row["symbol_id"]: row for row in rows(conn.execute(
        "SELECT symbol_id, kind, detail FROM entry_points"))}

    seen = {seed["symbol_id"]}
    frontier = [seed["symbol_id"]]
    reached: dict[str, int] = {}
    inferred_edges = 0

    for hop in range(1, depth + 1):
        next_frontier: list[str] = []
        for symbol_id in frontier:
            for edge in rows(conn.execute(
                    "SELECT from_id, edge_class FROM code_edges WHERE to_id=?"
                    " AND kind='CALLS' AND status='ACTIVE'", (symbol_id,))):
                nxt = edge["from_id"]
                if edge["edge_class"] != "deterministic":
                    inferred_edges += 1
                if nxt in seen:
                    continue
                seen.add(nxt)
                next_frontier.append(nxt)
                if nxt in known and nxt not in reached:
                    reached[nxt] = hop
        frontier = next_frontier
        if not frontier:
            break

    paths = {r["symbol_id"]: r for r in rows(conn.execute(
        "SELECT s.symbol_id, s.symbol_path, f.path FROM symbols s"
        " JOIN files f ON f.file_id = s.file_id WHERE s.status='ACTIVE'"))}

    triggers = sorted(
        ({"kind": known[sid]["kind"], "detail": known[sid]["detail"], "hops": hop,
          "symbol_path": paths.get(sid, {}).get("symbol_path"),
          "path": paths.get(sid, {}).get("path")}
         for sid, hop in reached.items()),
        # Tests last. On a widely-used helper they outnumber everything else by
        # an order of magnitude, and "which route reaches this" is drowned by
        # ninety test functions that all reach it transitively.
        key=lambda t: (t["kind"] == "test", t["hops"], t["kind"],
                       t["symbol_path"] or ""))

    by_kind: dict[str, int] = {}
    for trigger in triggers:
        by_kind[trigger["kind"]] = by_kind.get(trigger["kind"], 0) + 1

    # Capped per kind so one noisy kind cannot crowd out the others. The counts
    # above are of everything found, so the total is never understated.
    shown: list[dict[str, Any]] = []
    per_kind: dict[str, int] = {}
    for trigger in triggers:
        seen_of_kind = per_kind.get(trigger["kind"], 0)
        if seen_of_kind >= MAX_TRIGGERS_PER_KIND:
            continue
        per_kind[trigger["kind"]] = seen_of_kind + 1
        shown.append(trigger)

    result = {
        "target": seed,
        "triggered_by": shown,
        "counts": {"total": len(triggers), "by_kind": by_kind},
        "shown": len(shown),
        "truncated": len(shown) < len(triggers),
        "searched_depth": depth,
    }
    if len(shown) < len(triggers):
        result["listing_note"] = (
            f"Showing at most {MAX_TRIGGERS_PER_KIND} per kind. counts.by_kind is "
            f"the full tally; the listing is a sample, ranked with tests last.")
    if not triggers:
        result["note"] = ("No declared entry point reaches this symbol within "
                          "this depth. That makes it a library, dead code, or "
                          "reachable only through wiring static analysis cannot "
                          "see - it is not by itself proof of any of the three.")
    result.update(_envelope(conn, "upstream", [seed["name"]], inferred_edges, 0))
    return result
