"""Entry points and functional areas.

A call graph with no roots is a pile of edges. These two passes give it shape:

    entry points   where control enters the program - HTTP routes, MCP tool
                   handlers, CLI commands, tests. Everything downstream of one
                   is a flow someone can actually trigger.
    communities    which symbols form a functional area, derived from the call
                   graph rather than from directory layout, which is a filing
                   decision and not always a structural one.

Both are deliberately evidence-led. An entry point is recorded only when
something in the source says so - a decorator, a registration, a filename
convention that the language actually enforces. A guess here is worse than a
gap: `route_map` and `entrypoints` present their output as fact, and an
invented route sends the reader somewhere that does not exist.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from typing import Any

from .db import jload, rows, write_tx

# Above this many symbols, community detection is skipped rather than run
# slowly on every index. The result says so instead of silently returning
# nothing, because "no areas" and "too big to compute" are different facts.
MAX_COMMUNITY_SYMBOLS = 60_000
LOUVAIN_ROUNDS = 20
MIN_COMMUNITY_SIZE = 3

# HTTP verbs a decorator can name. `route` and `middleware` carry no verb of
# their own, and are recorded as method-agnostic rather than guessed at.
_VERBS = "get|post|put|patch|delete|head|options|trace"

# @app.get("/x") / @router.post('/x') / @bp.route("/x", methods=["POST"])
_ROUTE_DECORATOR = re.compile(
    r"^@\s*(?P<obj>[\w.]+)\s*\.\s*(?P<verb>" + _VERBS + r"|route|websocket)\s*\("
    r"\s*(?P<quote>['\"])(?P<path>[^'\"]*)(?P=quote)",
    re.IGNORECASE)

# @mcp.tool() / @server.tool(name="x") / @app.tool
_TOOL_DECORATOR = re.compile(r"^@\s*(?P<obj>[\w.]+)\s*\.\s*tool\b", re.IGNORECASE)

# Spring/JAX-RS style: @GetMapping("/x"), @RequestMapping(...), @Path("/x")
_ANNOTATION_ROUTE = re.compile(
    r"^@\s*(?P<verb>Get|Post|Put|Patch|Delete)?(?:Mapping|Path)\s*\("
    r"[^)]*?(?P<quote>['\"])(?P<path>[^'\"]*)(?P=quote)")

# ASP.NET attribute style: [HttpGet("/x")], [HttpPost], [Route("api/x")].
# The path is optional - [HttpGet] on an attribute-routed controller is a
# declaration of a route even though the segment comes from elsewhere.
_ATTRIBUTE_ROUTE = re.compile(
    r"^\[\s*(?:Http(?P<verb>" + _VERBS + r")|(?P<any>Route))\b\s*"
    r"(?:\(\s*(?P<quote>['\"])(?P<path>[^'\"]*)(?P=quote))?",
    re.IGNORECASE)

_METHODS_KWARG = re.compile(r"methods\s*=\s*\[([^\]]*)\]", re.IGNORECASE)

# Leading name of a decorator, annotation or attribute, whichever sigil the
# language spells it with: @Test, #[test], [Fact], @app.post.
_MARKER_NAME = re.compile(r"[\w:.]+")

# Markers a test runner actually collects on. Attribute-declared tests are the
# only kind in Java, Kotlin, C# and Rust: none of them name tests by convention,
# so without reading the annotation there is nothing to read at all.
_TEST_MARKERS = {
    "test", "tests", "testmethod", "parameterizedtest", "repeatedtest",
    "testcase", "testfactory", "fact", "theory", "rstest",
    "tokio::test", "async_std::test", "wasm_bindgen_test", "googletest::test",
}

# Filenames that make a module a command-line surface by convention. A `main`
# in one of these is the program's own entry point, which is as declared as an
# entry point gets - it is the symbol the toolchain itself looks for.
_CLI_FILES = ("cli.py", "__main__.py", "main.py", "manage.py",
              "main.go", "main.rs", "main.c", "main.cpp", "main.cc",
              "main.java", "main.kt", "program.cs", "main.swift")

# How a test declares itself, per language, as (predicate, evidence). The
# evidence string is user-facing and is read as fact, so it names the rule that
# actually applies: saying "pytest naming convention" about a Go file sends the
# reader looking for a pytest config that does not exist.
_TEST_FILE_SUFFIXES = {
    "go": ("_test.go",),
    "rust": ("_test.rs", "tests.rs"),
    "ruby": ("_test.rb", "_spec.rb"),
}
_JS_TEST_MARKERS = (".test.", ".spec.", "__tests__/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _marker_name(text: str) -> str:
    """Leading identifier of a decorator, annotation or attribute, lowercased.

    Strips whichever sigils the language uses, so `@Test`, `#[test]` and
    `[Test]` all reduce to `test` and one marker table serves every language.
    """
    match = _MARKER_NAME.match(text.lstrip("@#[ \t"))
    return match.group(0).lower() if match else ""


def _route_from_decorator(text: str) -> tuple[str, str] | None:
    """(verb, path) for a decorator that declares a route, else None."""
    match = _ATTRIBUTE_ROUTE.match(text)
    if match:
        verb = (match.group("verb") or "ANY").upper()
        return verb, match.group("path") or ""

    match = _ROUTE_DECORATOR.match(text)
    if match:
        verb = match.group("verb").upper()
        path = match.group("path")
        if verb in ("ROUTE", "WEBSOCKET"):
            # Flask carries the verb in a keyword argument; without one, the
            # route really is method-agnostic by declaration.
            methods = _METHODS_KWARG.search(text)
            if methods:
                named = re.findall(r"['\"](\w+)['\"]", methods.group(1))
                if named:
                    return "|".join(v.upper() for v in named), path
            verb = "ANY" if verb == "ROUTE" else "WS"
        return verb, path

    match = _ANNOTATION_ROUTE.match(text)
    if match:
        return (match.group("verb") or "ANY").upper(), match.group("path")
    return None


def _test_evidence(name: str, path: str, lang: str) -> str | None:
    """Why this symbol is a test, named after the convention that makes it one.

    Every branch here quotes a rule some runner actually enforces. Nothing is
    inferred from a name alone: a `test_helpers` module full of functions
    called `test_case_for` is not a suite, and claiming it is puts fictional
    triggers in front of anyone asking what can reach a piece of code.
    """
    lowered = path.lower()
    base = lowered.rsplit("/", 1)[-1]

    for suffix in _TEST_FILE_SUFFIXES.get(lang, ()):
        if lowered.endswith(suffix):
            if lang == "go" and not (name.startswith("Test") or
                                     name.startswith("Benchmark") or
                                     name.startswith("Fuzz") or
                                     name.startswith("Example")):
                return None
            return f"{lang} test-file convention: {base}"

    if lang in ("javascript", "typescript", "tsx"):
        if any(marker in lowered for marker in _JS_TEST_MARKERS):
            return f"test-file naming convention: {base}"
        return None

    if name.startswith("test_") or name.startswith("Test"):
        if "test" in lowered or "spec" in lowered:
            if lang == "python":
                return f"pytest naming convention in {path}"
            return f"test naming convention in {path}"
    return None


def classify(symbol: dict[str, Any], file_path: str) -> tuple[str, str, str] | None:
    """(kind, detail, evidence) if this symbol is an entry point, else None.

    Order matters: a decorated handler is a route or a tool before it is
    anything else, and a test is only a test when nothing stronger applies.
    """
    decorators = jload(symbol.get("decorators"), []) or []
    name = symbol.get("name") or ""
    lang = symbol.get("lang") or ""
    path = (file_path or "").replace("\\", "/")

    declared_test = ""
    for text in decorators:
        route = _route_from_decorator(text)
        if route:
            verb, url = route
            # An ASP.NET `[HttpPost]` declares the verb and takes its segment
            # from the controller, so there is no path to report. Printing "/"
            # for it would name a route that does not exist.
            return "route", f"{verb} {url}".strip(), text
        if _TOOL_DECORATOR.match(text):
            return "tool", name, text
        if not declared_test and _marker_name(text) in _TEST_MARKERS:
            declared_test = text

    # An annotation outranks a naming convention: it is the runner's own
    # declaration, and in Java, Kotlin, C# and Rust it is the only one there is.
    if declared_test:
        return "test", name, f"{declared_test} on {name}"

    evidence = _test_evidence(name, path, lang)
    if evidence:
        return "test", name, evidence

    base = path.rsplit("/", 1)[-1].lower()
    if base in _CLI_FILES and "." not in (symbol.get("symbol_path") or name):
        if name == "main" or name.startswith("cmd_") or name.startswith("command_"):
            return "cli", name, f"module-level {name}() in {base}"

    return None


def detect_entry_points(conn: sqlite3.Connection, commit: str | None) -> dict[str, int]:
    """Rebuild the entry-point table from current symbols.

    Rebuilt wholesale rather than merged: an entry point that stopped being one
    (its decorator was removed) must stop being reported, and there are few
    enough of them that recomputing is cheaper than reconciling.
    """
    found: list[tuple[str, str, str, str, str | None]] = []
    for row in rows(conn.execute(
            "SELECT s.symbol_id, s.name, s.symbol_path, s.lang, s.decorators, f.path"
            " FROM symbols s JOIN files f ON f.file_id = s.file_id"
            " WHERE s.status='ACTIVE' AND f.status='ACTIVE'")):
        verdict = classify(row, row["path"])
        if verdict:
            kind, detail, evidence = verdict
            found.append((row["symbol_id"], kind, detail, evidence[:200], commit))

    with write_tx(conn):
        conn.execute("DELETE FROM entry_points")
        conn.executemany(
            "INSERT OR REPLACE INTO entry_points"
            " (symbol_id, kind, detail, evidence, commit_id) VALUES (?,?,?,?,?)",
            found)

    counts: dict[str, int] = {}
    for _, kind, _, _, _ in found:
        counts[kind] = counts.get(kind, 0) + 1
    counts["total"] = len(found)
    return counts


# ------------------------------------------------------------------ communities

def _louvain(adjacency: dict[str, dict[str, float]]) -> dict[str, int]:
    """Modularity-maximising clustering, one level of Louvain aggregation.

    Louvain rather than label propagation, which was tried first and produced a
    single community holding 40% of this repository. That is label
    propagation's known failure on a graph with hubs: a widely-called helper
    pulls every caller into one label, and an "area" covering half the codebase
    tells a reader nothing.

    Determinism matters more here than the last decimal of modularity. Nodes
    are visited in sorted order and ties break on the lowest community id, so
    the same graph always yields the same areas - otherwise the areas reshuffle
    on every index and nothing downstream can rely on them.
    """
    degree = {node: sum(edges.values()) for node, edges in adjacency.items()}
    total_weight = sum(degree.values()) / 2.0
    if total_weight <= 0:
        return {node: index for index, node in enumerate(sorted(adjacency))}

    community = {node: index for index, node in enumerate(sorted(adjacency))}
    sum_total = {index: degree[node] for node, index in community.items()}
    order = sorted(adjacency)

    for _ in range(LOUVAIN_ROUNDS):
        moved = False
        for node in order:
            own = community[node]
            node_degree = degree[node]

            # Weight from this node into each neighbouring community.
            weights: dict[int, float] = {}
            for peer, weight in adjacency[node].items():
                if peer == node:
                    continue
                weights[community[peer]] = weights.get(community[peer], 0.0) + weight

            # Take the node out before scoring, or it competes with itself.
            sum_total[own] -= node_degree
            best_community, best_gain = own, weights.get(own, 0.0) - \
                sum_total[own] * node_degree / (2.0 * total_weight)
            for candidate, weight in sorted(weights.items()):
                gain = weight - sum_total.get(candidate, 0.0) * node_degree / (2.0 * total_weight)
                if gain > best_gain + 1e-12:
                    best_community, best_gain = candidate, gain

            sum_total[best_community] = sum_total.get(best_community, 0.0) + node_degree
            if best_community != own:
                community[node] = best_community
                moved = True
        if not moved:
            break
    return community


def detect_communities(conn: sqlite3.Connection) -> dict[str, Any]:
    """Cluster the call graph into functional areas.

    Edge confidence is the edge weight, so a call proven from the syntax binds
    two symbols more tightly than one resolved by a name being unique. Without
    that, the weakest tier - which is a name coincidence away from being wrong -
    would shape the areas as strongly as a receiver-bound call.
    """
    total = rows(conn.execute(
        "SELECT COUNT(*) n FROM symbols WHERE status='ACTIVE'"))[0]["n"]
    if total > MAX_COMMUNITY_SYMBOLS:
        return {"skipped": "too_large", "symbols": total,
                "note": f"over {MAX_COMMUNITY_SYMBOLS} active symbols; areas not computed"}

    # Undirected projection: "works with" is symmetric, while "calls" is not.
    adjacency: dict[str, dict[str, float]] = {}
    for row in rows(conn.execute(
            "SELECT from_id, to_id, confidence FROM code_edges"
            " WHERE kind='CALLS' AND status='ACTIVE'")):
        weight = float(row["confidence"] or 1.0)
        a, b = row["from_id"], row["to_id"]
        adjacency.setdefault(a, {})[b] = adjacency.setdefault(a, {}).get(b, 0.0) + weight
        adjacency.setdefault(b, {})[a] = adjacency.setdefault(b, {}).get(a, 0.0) + weight

    if not adjacency:
        return {"communities": 0, "assigned": 0, "note": "no call edges to cluster"}

    label = _louvain(adjacency)

    sizes: dict[int, int] = {}
    for value in label.values():
        sizes[value] = sizes.get(value, 0) + 1
    keep = {value for value, size in sizes.items() if size >= MIN_COMMUNITY_SIZE}

    # Renumber kept communities by size so ids are meaningful and stable.
    ranked = sorted(keep, key=lambda v: (-sizes[v], v))
    renumber = {old: index for index, old in enumerate(ranked)}

    stamp = now()
    payload = [(node, renumber[value], stamp)
               for node, value in label.items() if value in renumber]
    with write_tx(conn):
        conn.execute("DELETE FROM communities")
        conn.executemany(
            "INSERT OR REPLACE INTO communities (symbol_id, community_id, computed_at)"
            " VALUES (?,?,?)", payload)

    return {"communities": len(ranked), "assigned": len(payload),
            "singletons_dropped": len(sizes) - len(ranked)}


def _is_test_path(path: str) -> bool:
    lowered = (path or "").lower()
    return "test" in lowered or "spec" in lowered


def _distinct_name(files: list[str], dirs: list[str], used: dict[str, int]) -> str:
    """A name that separates this area from the ones already named.

    Naming by dominant file was itself the fix for naming by dominant directory,
    which gave six areas called `src/icn` in a flat package. It has the same
    failure one level down: Louvain finds many small communities that share a
    dominant file, so a repository comes back with four areas all called
    `tests` and three called `parsing`, and a list of identical names
    distinguishes nothing at all.

    So the first name that is still free wins, taken from the area's own files
    in order of how much of it they hold. A name only becomes `tests +papers`
    when this area genuinely has no more distinctive file of its own, and a
    bare numeric suffix is the last resort rather than the first.
    """
    candidates = [name for name in files if name]
    base = candidates[0] if candidates else (dirs[0] if dirs else "?")

    if base not in used:
        used[base] = 1
        return base

    for alternative in candidates[1:4]:
        combined = f"{base} +{alternative}"
        if combined not in used:
            used[combined] = 1
            return combined

    for directory in dirs[:2]:
        combined = f"{base} ({directory.rsplit('/', 1)[-1]})"
        if combined not in used:
            used[combined] = 1
            return combined

    used[base] += 1
    return f"{base} {used[base]}"


def describe_areas(conn: sqlite3.Connection, limit: int = 12,
                   include_tests: bool = False) -> list[dict[str, Any]]:
    """Human-readable summary of each functional area.

    An area is named by the directory most of its symbols live in, which is the
    label a reader already recognises, and reported with the entry points that
    reach into it - an area nothing can trigger is usually either dead or a
    library, and that distinction is worth surfacing.

    test_share is reported rather than filtered away, because a test suite does
    form real areas and hiding them would be a lie about the graph. But they
    are not what someone asking "what are the parts of this system" means, so
    by default an area that is mostly tests is ranked below one that is not.
    """
    areas: dict[int, dict[str, Any]] = {}
    for row in rows(conn.execute(
            "SELECT c.community_id, s.symbol_id, s.symbol_path, f.path"
            " FROM communities c JOIN symbols s ON s.symbol_id = c.symbol_id"
            " JOIN files f ON f.file_id = s.file_id"
            " WHERE s.status='ACTIVE'")):
        area = areas.setdefault(row["community_id"], {
            "community_id": row["community_id"], "size": 0,
            "_dirs": {}, "_files": {}, "_test_files": {},
            "members": [], "_ids": set()})
        area["size"] += 1
        area["_ids"].add(row["symbol_id"])
        path = (row["path"] or "").replace("\\", "/")
        if _is_test_path(path):
            area["_tests"] = area.get("_tests", 0) + 1
        directory = path.rsplit("/", 1)[0] or "."
        area["_dirs"][directory] = area["_dirs"].get(directory, 0) + 1
        # Named by dominant file, not dominant directory. In a flat package
        # every area shares one directory, so the directory name distinguishes
        # nothing - "src/icn" six times over says less than nothing.
        #
        # Test files are kept separately and used only when an area has no
        # source file to name it. An area that is entirely tests is real, and
        # `test_papers` tells a reader what it covers; falling through to the
        # directory names every such area `tests`.
        stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        bucket = "_test_files" if _is_test_path(path) else "_files"
        area.setdefault(bucket, {})
        area[bucket][stem] = area[bucket].get(stem, 0) + 1
        if len(area["members"]) < 8:
            area["members"].append(row["symbol_path"])

    entry_rows = rows(conn.execute(
        "SELECT c.community_id, e.kind, e.detail FROM entry_points e"
        " JOIN communities c ON c.symbol_id = e.symbol_id"))
    for row in entry_rows:
        area = areas.get(row["community_id"])
        if area is not None:
            area.setdefault("entry_points", []).append(
                {"kind": row["kind"], "detail": row["detail"]})

    for area in areas.values():
        area["test_share"] = round(area.pop("_tests", 0) / max(1, area["size"]), 2)

    if include_tests:
        ranked = sorted(areas.values(), key=lambda a: -a["size"])
    else:
        # Mostly-test areas sort last rather than vanishing.
        ranked = sorted(areas.values(),
                        key=lambda a: (a["test_share"] > 0.5, -a["size"]))

    out = []
    used: dict[str, int] = {}
    for area in ranked[:limit]:
        dirs = sorted(area.pop("_dirs").items(), key=lambda kv: (-kv[1], kv[0]))
        files = sorted(area.pop("_files").items(), key=lambda kv: (-kv[1], kv[0]))
        tests = sorted(area.pop("_test_files", {}).items(),
                       key=lambda kv: (-kv[1], kv[0]))
        area.pop("_ids", None)
        area["name"] = _distinct_name(
            [f for f, _ in files] + [f for f, _ in tests],
            [d for d, _ in dirs], used)
        area["spans"] = [d for d, _ in dirs[:4]]
        area["files"] = [f for f, _ in files[:4]]
        entries = area.get("entry_points", [])
        area["entry_points"] = entries[:6]
        area["entry_point_count"] = len(entries)
        if not entries:
            area["note"] = ("no entry point reaches this area - it is a library, "
                            "or it is unreachable")
        out.append(area)
    return out
