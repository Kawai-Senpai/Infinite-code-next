"""What git history says that the code alone does not.

The code graph answers "what calls this". History answers three questions it
cannot:

    coupling   which files keep changing together. Two files with no import
               between them and a co-change rate of 90% are coupled by
               something the type system does not record - a shared format, a
               protocol, a duplicated constant - and that is exactly the pair
               that breaks when one is edited alone.
    hotspots   where change concentrates. A file that is central in the call
               graph AND rewritten every other week is where the next bug will
               be; either property alone is unremarkable.
    dead code  symbols nothing reaches. This is the one that has to be said
               carefully, and most of this module's caution lives there.

Everything here is evidence with a stated boundary, in the same shape graph()
uses. History is a lower bound by construction: it only knows the window it
read, only knows commits (not uncommitted work), and knows nothing at all
about a repository that was squash-imported last week. A result that does not
say so invites the reader to treat a shallow clone as proof.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from .db import rows
from .identity import run_git

# Commits read in one pass. Deep enough that coupling means something on an
# active repository, shallow enough to stay one subprocess and well under the
# git timeout on a large history.
DEFAULT_WINDOW = 500
MAX_WINDOW = 5000

# A commit touching more files than this is a bulk edit - a reformat, a licence
# header sweep, a generated-code refresh - and it couples every pair it touches
# to every other. One of them can add more spurious pairs than the entire real
# signal in the window, so they are excluded from coupling and counted openly.
BULK_COMMIT_FILES = 40

DEFAULT_COUPLING_LIMIT = 50
DEFAULT_HOTSPOT_LIMIT = 50
DEFAULT_DEAD_LIMIT = 100

# Commit boundary marker. A byte no path or author name contains, so the parse
# cannot be confused by a filename that happens to look like a header.
_MARK = "\x01"


def read_history(root, window: int = DEFAULT_WINDOW,
                 since: str | None = None) -> dict[str, Any]:
    """Commits and the files each touched, from one git call.

    One `git log --numstat` rather than a call per file: process spawn
    dominates everything else on Windows, and the per-path version made an
    investigation exceed a 90-second client timeout on a few hundred paths
    (the same reason search._recency_map is written this way).

    Merge commits are not diffed by git without an explicit flag, so they
    contribute no files. That is what we want: a merge touches everything on
    the branch and would couple every pair in it.
    """
    window = max(1, min(int(window or DEFAULT_WINDOW), MAX_WINDOW))
    args = ["log", f"-{window}", f"--format={_MARK}%H %ct", "--numstat"]
    if since:
        args.insert(1, f"--since={since}")

    # strip=False: the numstat columns are tab-separated and a leading strip
    # would be harmless here, but this codebase has been bitten once by
    # stripping structured git output and the rule is now unconditional.
    code, out, err = run_git(args, root, strip=False)
    if code != 0:
        return {"available": False, "commits": [], "window": window,
                "reason": (err or "git log failed")[:200]}

    commits: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in out.splitlines():
        if line.startswith(_MARK):
            header = line[1:].split(" ")
            current = {"sha": header[0],
                       "ts": float(header[1]) if len(header) > 1 and header[1].isdigit() else 0.0,
                       "files": {}}
            commits.append(current)
            continue
        if current is None or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        added, deleted, path = parts[0], parts[1], parts[2].strip()
        if not path:
            continue
        # A binary file reports "-" for both counts. It changed, so it counts
        # as a change; it just contributes no line churn.
        current["files"][_normalise(path)] = (
            int(added) if added.isdigit() else 0,
            int(deleted) if deleted.isdigit() else 0,
        )

    return {"available": True, "commits": commits, "window": window,
            "read": len(commits),
            "truncated": len(commits) >= window}


def _normalise(path: str) -> str:
    """A rename shows as `old => new` inside braces; take what the path is now."""
    path = path.replace("\\", "/").strip()
    if "=>" in path:
        # `src/{old => new}/file.py` or `old.py => new.py`
        if "{" in path and "}" in path:
            head, rest = path.split("{", 1)
            middle, tail = rest.split("}", 1)
            path = head + middle.split("=>")[-1].strip() + tail
            path = path.replace("//", "/")
        else:
            path = path.split("=>")[-1].strip()
    return path.strip("/")


def _indexed_paths(conn: sqlite3.Connection) -> dict[str, str]:
    """Active file paths, keyed by their forward-slash form.

    History is filtered to files we actually indexed. A README that changes
    every day is real churn and useless here: it has no symbols, no callers and
    no place in the code graph, so it would crowd out the answer.
    """
    return {(row["path"] or "").replace("\\", "/"): row["file_id"]
            for row in rows(conn.execute(
                "SELECT file_id, path FROM files WHERE status='ACTIVE'"))}


def _coverage(history: dict[str, Any], considered: int, bulk: int) -> dict[str, Any]:
    """The epistemic envelope, in the same shape graph() uses."""
    boundaries: list[str] = []
    if not history.get("available"):
        boundaries.append(
            "No git history is readable here, so nothing below rests on history at "
            "all. " + str(history.get("reason", "")))
    else:
        if history.get("truncated"):
            boundaries.append(
                f"Only the last {history['read']} commits were read. A pair that "
                f"stopped changing together before that window is not shown, and "
                f"one that started just inside it is understated.")
        if bulk:
            boundaries.append(
                f"{bulk} commit(s) touching more than {BULK_COMMIT_FILES} files "
                f"were excluded. A bulk edit couples every pair it touches and "
                f"would drown the real signal.")
        boundaries.append(
            "Only committed work is visible. Uncommitted changes, and history "
            "lost to a squash or a shallow clone, are invisible here rather than "
            "absent from the repository.")
    return {
        "epistemic": "lower-bound",
        "boundaries": boundaries,
        "history": {
            "available": bool(history.get("available")),
            "commits_read": history.get("read", 0),
            "window": history.get("window"),
            "window_exhausted": bool(history.get("truncated")),
            "commits_considered": considered,
            "bulk_commits_excluded": bulk,
        },
    }


# ------------------------------------------------------------------- coupling

def change_coupling(conn: sqlite3.Connection, root, window: int = DEFAULT_WINDOW,
                    since: str | None = None, min_support: int = 2,
                    limit: int = DEFAULT_COUPLING_LIMIT) -> dict[str, Any]:
    """File pairs that keep changing together, ranked by Jaccard overlap.

    Jaccard rather than raw co-change count, because the raw count just ranks
    the busiest files in the repository against each other. `shared / (a + b -
    shared)` asks the useful question instead: of all the times either file
    changed, how often did they change together.

    `also_imports` says whether the code graph already explains the pair. The
    interesting rows are the ones where it does not - those are coupled through
    something static analysis cannot see, and they are what a reader cannot get
    any other way.
    """
    min_support = max(1, int(min_support or 2))
    limit = max(1, min(int(limit or DEFAULT_COUPLING_LIMIT), 500))

    history = read_history(root, window, since)
    indexed = _indexed_paths(conn)

    changes: dict[str, int] = {}
    pairs: dict[tuple[str, str], int] = {}
    considered = bulk = 0
    for commit in history.get("commits", ()):
        touched = sorted(p for p in commit["files"] if p in indexed)
        if len(touched) < 2:
            for path in touched:
                changes[path] = changes.get(path, 0) + 1
            continue
        if len(touched) > BULK_COMMIT_FILES:
            bulk += 1
            continue
        considered += 1
        for path in touched:
            changes[path] = changes.get(path, 0) + 1
        for i, left in enumerate(touched):
            for right in touched[i + 1:]:
                pairs[(left, right)] = pairs.get((left, right), 0) + 1

    imports = _import_pairs(conn, indexed)

    items = []
    for (left, right), shared in pairs.items():
        if shared < min_support:
            continue
        union = changes[left] + changes[right] - shared
        items.append({
            "left": left,
            "right": right,
            "co_changes": shared,
            "left_changes": changes[left],
            "right_changes": changes[right],
            "jaccard": round(shared / union, 3) if union else 0.0,
            "also_imports": (left, right) in imports or (right, left) in imports,
        })

    items.sort(key=lambda item: (-item["jaccard"], -item["co_changes"],
                                 item["left"], item["right"]))
    hidden = [item for item in items[:limit] if not item["also_imports"]]

    result = {
        "pairs": items[:limit],
        "counts": {"total": len(items), "shown": min(len(items), limit),
                   "without_an_import_edge": len(hidden)},
        "truncated": len(items) > limit,
        "min_support": min_support,
        "note": ("A pair with also_imports=false is coupled by something the "
                 "code graph cannot see - a shared format, a protocol, a "
                 "duplicated constant. Those are the rows worth reading."),
    }
    result.update(_coverage(history, considered, bulk))
    return result


def _import_pairs(conn: sqlite3.Connection,
                  indexed: dict[str, str]) -> set[tuple[str, str]]:
    """Unordered file pairs that already have an IMPORTS edge between them."""
    by_id = {file_id: path for path, file_id in indexed.items()}
    found: set[tuple[str, str]] = set()
    for row in rows(conn.execute(
            "SELECT from_id, to_id FROM code_edges WHERE kind='IMPORTS'"
            " AND status='ACTIVE' AND to_id NOT LIKE 'extern:%'")):
        left, right = by_id.get(row["from_id"]), by_id.get(row["to_id"])
        if left and right:
            found.add((left, right))
    return found


# ------------------------------------------------------------------- hotspots

def hotspots(conn: sqlite3.Connection, root, window: int = DEFAULT_WINDOW,
             since: str | None = None, limit: int = DEFAULT_HOTSPOT_LIMIT) -> dict[str, Any]:
    """Files where change and structural weight meet.

    Churn alone ranks the changelog. Centrality alone ranks the utility module
    nobody has touched in a year. The product is the useful one: something both
    heavily depended on and constantly rewritten is where a change is most
    likely to break something far away.

    This deliberately does NOT claim to measure complexity. Cyclomatic
    complexity is not extracted anywhere in this codebase, and a "complexity"
    column derived from file length would be a guess wearing a precise name.
    What is measured is named for what it is: symbols, import degree, call
    degree, commits and line churn.
    """
    limit = max(1, min(int(limit or DEFAULT_HOTSPOT_LIMIT), 500))
    history = read_history(root, window, since)
    indexed = _indexed_paths(conn)

    commits: dict[str, int] = {}
    churn: dict[str, int] = {}
    considered = bulk = 0
    for commit in history.get("commits", ()):
        touched = [p for p in commit["files"] if p in indexed]
        if len(touched) > BULK_COMMIT_FILES:
            bulk += 1
            continue
        considered += 1
        for path in touched:
            added, deleted = commit["files"][path]
            commits[path] = commits.get(path, 0) + 1
            churn[path] = churn.get(path, 0) + added + deleted

    weight = _structural_weight(conn, indexed)

    items = []
    for path, file_id in indexed.items():
        change_count = commits.get(path, 0)
        if not change_count:
            continue
        structure = weight.get(file_id, {"symbols": 0, "import_degree": 0,
                                         "call_degree": 0})
        # Structural weight is the sum of what the graph knows about the file:
        # how much is defined in it, how many files depend on it, and how much
        # traffic its symbols carry. +1 so a file with a real change history
        # never scores zero purely for being a leaf.
        structural = (structure["symbols"] + 2 * structure["import_degree"]
                      + structure["call_degree"] + 1)
        items.append({
            "path": path,
            "commits": change_count,
            "churn": churn.get(path, 0),
            "symbols": structure["symbols"],
            "import_degree": structure["import_degree"],
            "call_degree": structure["call_degree"],
            "structural_weight": structural,
            "score": change_count * structural,
        })

    items.sort(key=lambda item: (-item["score"], -item["commits"], item["path"]))

    result = {
        "files": items[:limit],
        "counts": {"total": len(items), "shown": min(len(items), limit)},
        "truncated": len(items) > limit,
        "scoring": ("score = commits * structural_weight, where structural_weight "
                    "= symbols + 2*import_degree + call_degree + 1. Cyclomatic "
                    "complexity is NOT measured and is not part of this score."),
    }
    result.update(_coverage(history, considered, bulk))
    return result


def _structural_weight(conn: sqlite3.Connection,
                       indexed: dict[str, str]) -> dict[str, dict[str, int]]:
    """Per-file symbol count, inbound import degree and symbol call degree."""
    weight: dict[str, dict[str, int]] = {
        file_id: {"symbols": 0, "import_degree": 0, "call_degree": 0}
        for file_id in indexed.values()}

    for row in rows(conn.execute(
            "SELECT file_id, COUNT(*) n FROM symbols WHERE status='ACTIVE'"
            " GROUP BY file_id")):
        if row["file_id"] in weight:
            weight[row["file_id"]]["symbols"] = row["n"]

    for row in rows(conn.execute(
            "SELECT to_id, COUNT(*) n FROM code_edges WHERE kind='IMPORTS'"
            " AND status='ACTIVE' AND to_id NOT LIKE 'extern:%' GROUP BY to_id")):
        if row["to_id"] in weight:
            weight[row["to_id"]]["import_degree"] = row["n"]

    for row in rows(conn.execute(
            "SELECT s.file_id, COUNT(*) n FROM code_edges e"
            " JOIN symbols s ON s.symbol_id = e.to_id"
            " WHERE e.kind='CALLS' AND e.status='ACTIVE' AND s.status='ACTIVE'"
            " GROUP BY s.file_id")):
        if row["file_id"] in weight:
            weight[row["file_id"]]["call_degree"] = row["n"]

    return weight


# ------------------------------------------------------------------ dead code

def dead_code(conn: sqlite3.Connection, limit: int = DEFAULT_DEAD_LIMIT,
              include_tests: bool = False) -> dict[str, Any]:
    """Symbols nothing in the graph reaches. Candidates, never a verdict.

    This is the most dangerous query in the tool, because its output looks like
    a delete list and is not one. Three separate things produce "no callers",
    and only one of them is dead code:

        nothing calls it            genuinely unreferenced
        the caller was not resolved static resolution refused to guess, so an
                                    edge that exists in the program was never
                                    written. unresolved_calls counts these.
        the caller is not code      reflection, a dispatch table, a framework
                                    convention, an external consumer of a
                                    published API. Nothing static can see it.

    So every candidate carries `confidence`, and a symbol whose NAME appears at
    an unresolved call site is reported as `lower` rather than filtered out -
    filtering would hide the very rows a reader most needs to see, while
    silently promoting the rest to a certainty this cannot support.
    """
    limit = max(1, min(int(limit or DEFAULT_DEAD_LIMIT), 1000))

    called = {row["to_id"] for row in rows(conn.execute(
        "SELECT DISTINCT to_id FROM code_edges WHERE kind='CALLS' AND status='ACTIVE'"))}
    entries = {row["symbol_id"] for row in rows(conn.execute(
        "SELECT symbol_id FROM entry_points"))}
    ambiguous: dict[str, int] = {}
    for row in rows(conn.execute(
            "SELECT leaf, COUNT(*) n FROM unresolved_calls GROUP BY leaf")):
        ambiguous[row["leaf"]] = row["n"]
    imported = _imported_names(conn)

    candidates = []
    runtime_skipped = 0
    for row in rows(conn.execute(
            "SELECT s.symbol_id, s.name, s.symbol_path, s.kind, s.lang,"
            " s.line_start, s.line_end, f.path FROM symbols s"
            " JOIN files f ON f.file_id = s.file_id"
            " WHERE s.status='ACTIVE' AND f.status='ACTIVE'")):
        if row["symbol_id"] in called or row["symbol_id"] in entries:
            continue
        path = (row["path"] or "").replace("\\", "/")
        if not include_tests and _looks_like_test(path):
            continue
        # A container is not called; its methods are. Reporting every class in
        # the repository as dead code would bury the real candidates.
        if row["kind"] in ("class", "interface", "struct", "module", "namespace",
                           "trait", "impl", "object", "enum", "type"):
            continue
        name = row["name"] or ""
        if _is_runtime_invoked(name):
            runtime_skipped += 1
            continue

        # Everything that weakens the claim, gathered before it is graded. A
        # method is reached through a receiver, and receiver-typed resolution
        # is this analyzer's weakest area, so a method is never graded as
        # confidently unreferenced as a free function.
        reasons: list[str] = []
        unattributed = ambiguous.get(name, 0)
        if unattributed:
            reasons.append(f"{unattributed} unresolved call site(s) name it")
        if name in imported:
            reasons.append("imported by name somewhere, possibly under an alias")
        if "." in (row["symbol_path"] or ""):
            reasons.append("a method, reached through a receiver this analyzer "
                           "often cannot type")

        candidates.append({
            "symbol_path": row["symbol_path"],
            "name": name,
            "kind": row["kind"],
            "lang": row["lang"],
            "path": path,
            "line_start": row["line_start"],
            "line_end": row["line_end"],
            "unattributed_call_sites": unattributed,
            "confidence": "lower" if reasons else "higher",
            "weakened_by": reasons,
        })

    # Higher-confidence candidates first: those are the ones worth reading.
    candidates.sort(key=lambda c: (c["confidence"] != "higher",
                                   c["unattributed_call_sites"],
                                   c["path"], c["line_start"] or 0))
    lower = sum(1 for c in candidates if c["confidence"] == "lower")

    boundaries = [
        "A symbol with no inbound CALLS edge is not proven unused. Reflection, "
        "dispatch tables, framework wiring and external consumers of a published "
        "API all reach code without leaving a call site static analysis can read.",
        "A call made from module-level code or from an anonymous callback has no "
        "owning symbol, so no edge is written for it and its callee looks "
        "unreferenced. This is a known gap in call extraction, not a property of "
        "the code: it is why a helper called once at the top of a script appears "
        "here.",
    ]
    if lower:
        boundaries.append(
            f"{lower} candidate(s) carry a reason the claim is weaker - read "
            f"`weakened_by` on each. They are marked confidence='lower' and "
            f"ranked last.")
    if runtime_skipped:
        boundaries.append(
            f"{runtime_skipped} runtime-invoked symbol(s) were excluded entirely "
            f"(dunders, lifecycle hooks, conventional framework overrides). They "
            f"have no written call site in any correct program, so listing them "
            f"would be a false positive rather than a candidate.")

    return {
        "candidates": candidates[:limit],
        "counts": {"total": len(candidates),
                   "higher_confidence": len(candidates) - lower,
                   "lower_confidence": lower,
                   "runtime_invoked_excluded": runtime_skipped,
                   "shown": min(len(candidates), limit)},
        "truncated": len(candidates) > limit,
        "epistemic": "lower-bound",
        "boundaries": boundaries,
        "note": ("Candidates, not a delete list. Confirm each one before acting: "
                 "graph(action='triggers') says whether any entry point reaches "
                 "it, and investigate(action='why') says whether it was "
                 "deliberately left in place."),
    }


def _looks_like_test(path: str) -> bool:
    lowered = (path or "").lower()
    return "test" in lowered or "spec" in lowered


# Names the language or its runtime calls, never a written call site. A dunder
# has no caller anywhere in any correct program, so reporting one as
# unreferenced is not a caveat - it is a false positive by construction, and
# the first run of this query returned __init__, __enter__ and __exit__ at the
# top of the list.
_RUNTIME_INVOKED_PREFIXES = ("__", "test_")
_RUNTIME_INVOKED_NAMES = {
    # Python protocol and lifecycle
    "setUp", "tearDown", "setUpClass", "tearDownClass", "setup_method",
    "teardown_method", "setup_module", "teardown_module",
    # Common framework overrides across languages
    "main", "run", "handle", "toString", "equals", "hashCode", "Dispose",
    "render", "dispose", "finalize", "clone",
}


def _is_runtime_invoked(name: str) -> bool:
    if name.startswith("__") and name.endswith("__"):
        return True
    return name in _RUNTIME_INVOKED_NAMES


def _imported_names(conn: sqlite3.Connection) -> set[str]:
    """Leaf names some file imports by name.

    `from .doctor import main as doctor_main` records a probe specifier ending
    in `main`. The call site then reads `doctor_main(...)`, which resolves to
    nothing, so `main` looks unreferenced while being imported one line above
    the call. The import itself is the evidence that something reaches it.
    """
    from .db import jload

    found: set[str] = set()
    for row in rows(conn.execute(
            "SELECT imports_raw FROM files WHERE status='ACTIVE'"
            " AND imports_raw IS NOT NULL")):
        for spec in jload(row["imports_raw"], []) or []:
            module = (spec.get("module") or "").replace("/", ".")
            leaf = module.rsplit(".", 1)[-1].strip()
            if leaf:
                found.add(leaf)
    return found
