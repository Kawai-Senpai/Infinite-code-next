"""Resolve import specifiers to real file nodes.

An IMPORTS edge is only worth storing if its target is a node the graph can
walk. Writing `to_id = "import:<statement text>"` produces an edge that exists
and answers nothing: it cannot be joined to a file, so it supports no cycle
detection, no module-order reasoning, and - the expensive loss - no way to ask
"which files could this name have come from" when resolving a call.

Resolution runs as a second pass over stored specifiers, for the same reason
call resolution does: during a walk a file is routinely indexed before the file
it imports, so resolving inline drops every edge pointing forward in walk order.

What cannot be resolved is recorded as an edge to `extern:<module>` with
edge_class 'external'. That is a real answer - the import genuinely left the
indexed program - and it is deliberately distinguishable from an import that
should have resolved and did not.
"""

from __future__ import annotations

import posixpath
import sqlite3
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable

from . import ids
from .db import jload, rows, write_tx

# Extensions tried when a specifier omits one, per language family.
JS_EXTENSIONS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".d.ts")
JS_INDEX = tuple(f"index{ext}" for ext in JS_EXTENSIONS)
PY_EXTENSIONS = (".py", ".pyi")

JS_LANGS = {"javascript", "typescript", "tsx"}

# A Go import names a directory, not a file, so it can fan out. Capped: a
# package with 200 files should not produce 200 edges from one statement.
GO_FANOUT = 8

IMPORT_BATCH = 400


def now() -> str:
    """Millisecond UTC stamp, matching indexer.now().

    Defined here rather than imported: the indexer imports this module, so
    borrowing its clock would close an import cycle.
    """
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class FileIndex:
    """Path lookup over the active file set.

    Suffix matching is the load-bearing operation: `icn.db` is stored at
    `src/icn/db.py`, so an exact-path lookup finds nothing. Indexing by
    basename first keeps that a short list scan instead of a scan of every
    path in the repository.
    """

    def __init__(self, file_rows: Iterable[dict[str, Any]]):
        self.by_path: dict[str, str] = {}
        self.by_basename: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self.dirs: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for row in file_rows:
            path = (row["path"] or "").replace("\\", "/").lstrip("./")
            if not path:
                continue
            file_id = row["file_id"]
            self.by_path[path] = file_id
            self.by_basename[posixpath.basename(path)].append((path, file_id))
            self.dirs[posixpath.dirname(path)].append((path, file_id))

    def exact(self, path: str) -> str | None:
        return self.by_path.get(path.replace("\\", "/").lstrip("./"))

    def suffix(self, candidate: str) -> list[str]:
        """file_ids whose path is, or ends with, `candidate`.

        Returns every match rather than the first: an ambiguous import is a
        fact the caller needs, not something to silently pick a winner for.
        """
        candidate = candidate.replace("\\", "/").lstrip("./")
        if not candidate:
            return []
        direct = self.by_path.get(candidate)
        if direct:
            return [direct]
        needle = "/" + candidate
        return [
            file_id
            for path, file_id in self.by_basename.get(posixpath.basename(candidate), ())
            if path.endswith(needle)
        ]

    def in_dir(self, directory: str, exclude_suffix: str = "") -> list[str]:
        directory = directory.replace("\\", "/").lstrip("./")
        return [file_id for path, file_id in self.dirs.get(directory, ())
                if not (exclude_suffix and path.endswith(exclude_suffix))]

    def dirs_ending_with(self, suffix: str) -> list[str]:
        suffix = suffix.replace("\\", "/").strip("/")
        if not suffix:
            return []
        needle = "/" + suffix
        return [d for d in self.dirs if d == suffix or d.endswith(needle)]


def _python_candidates(module: str, level: int, from_path: str) -> list[str]:
    """Candidate paths for a Python import.

    Absolute imports are matched by suffix because the repository root is
    rarely the import root - `icn.db` lives at `src/icn/db.py`. Relative
    imports are exact: the level counts up from the importing file's package,
    so there is nothing to guess.
    """
    parts = [p for p in module.split(".") if p]
    if level:
        base = posixpath.dirname(from_path)
        for _ in range(level - 1):
            base = posixpath.dirname(base)
        stem = posixpath.join(base, *parts) if parts else base
        return [f"{stem}{ext}" for ext in PY_EXTENSIONS] + [
            posixpath.join(stem, f"__init__{ext}") for ext in PY_EXTENSIONS
        ]
    if not parts:
        return []
    stem = "/".join(parts)
    return [f"{stem}{ext}" for ext in PY_EXTENSIONS] + [
        f"{stem}/__init__{ext}" for ext in PY_EXTENSIONS
    ]


def _js_candidates(module: str, from_path: str) -> list[str]:
    """Candidate paths for a JS/TS specifier.

    Only relative specifiers are resolvable without reading tsconfig paths and
    node_modules; a bare `react` is genuinely external and is reported as such.
    """
    if not module.startswith("."):
        return []
    base = posixpath.normpath(posixpath.join(posixpath.dirname(from_path), module))
    if base.startswith("../") or base == "..":
        return []                       # escaped the repository root
    base = base.lstrip("./")
    out = [base] if posixpath.splitext(base)[1] else []
    out += [f"{base}{ext}" for ext in JS_EXTENSIONS]
    out += [posixpath.join(base, name) for name in JS_INDEX]
    return out


def _rust_candidates(module: str) -> list[str]:
    """Candidate paths for a Rust `use` or `mod`, longest prefix first.

    A `use` path ends in an ITEM, not a module: `use crate::store::lookup`
    imports the function `lookup` out of the module `store`. Only trying the
    whole path meant looking for `store/lookup.rs`, which never exists, so
    every intra-crate `use` in the repository was reported as external and the
    Rust module graph was empty.

    Prefixes are tried longest first so the deepest real module wins. The
    residual risk is a repository that happens to contain a file named after an
    external crate - a `src/std.rs` alongside `use std::collections::HashMap` -
    which would resolve to the local file. That is the same suffix-matching
    trade-off `_python_candidates` already makes, and the alternative is
    parsing Cargo.toml to learn which crate names are foreign.
    """
    parts = [p for p in module.replace("::", "/").split("/")
             if p and p not in ("crate", "self", "super")]
    out: list[str] = []
    for end in range(len(parts), 0, -1):
        stem = "/".join(parts[:end])
        out.append(f"{stem}.rs")
        out.append(f"{stem}/mod.rs")
    return out


def _go_package_dirs(module: str, index: FileIndex) -> list[str]:
    """Directories a Go import path could name, deepest match first.

    A Go import path is module-prefixed: `example.com/app/internal/store` names
    the directory `internal/store` in a repository that contains no
    `example.com/app` directory at all. Matching the path whole therefore never
    matched anything, and no Go import in any repository ever resolved.

    Successively shorter suffixes are tried so the module prefix falls away,
    longest first - `internal/store` has to win over a bare `store` when both
    exist, or an import resolves to the wrong package.
    """
    parts = [p for p in module.split("/") if p]
    for start in range(len(parts)):
        suffix = "/".join(parts[start:])
        matches = sorted(index.dirs_ending_with(suffix), key=len, reverse=True)
        if matches:
            return matches
    return []


def resolve_specifier(spec: dict[str, Any], lang: str, from_path: str,
                      index: FileIndex) -> list[str]:
    """file_ids this specifier resolves to. Empty means external."""
    module = (spec.get("module") or "").strip()
    level = int(spec.get("level") or 0)
    if not module and not level:
        return []

    if lang == "python":
        for candidate in _python_candidates(module, level, from_path):
            hit = index.exact(candidate) if level else None
            if hit:
                return [hit]
            if not level:
                found = index.suffix(candidate)
                if found:
                    return found
        return []

    if lang in JS_LANGS:
        for candidate in _js_candidates(module, from_path):
            hit = index.exact(candidate)
            if hit:
                return [hit]
        return []

    if lang == "java":
        return index.suffix("/".join(module.split(".")) + ".java")

    if lang == "kotlin":
        parts = [p for p in module.split(".") if p and p != "*"]
        if not parts:
            return []
        stem = "/".join(parts)
        for candidate in (f"{stem}.kt", f"{stem}.java"):
            found = index.suffix(candidate)
            if found:
                return found
        return []

    if lang == "csharp":
        # A C# namespace is not required to match the directory layout, so this
        # resolves only the conventional case where it does. What it must not
        # do is guess by last segment alone: `using App.Storage` would then
        # attach to any Storage.cs in the repository, and a wrong edge is worse
        # than the honest `extern:` one.
        parts = [p for p in module.split(".") if p]
        if not parts:
            return []
        return index.suffix("/".join(parts) + ".cs")

    if lang in ("c", "cpp"):
        # `#include "net/socket.h"` is already a path; angle includes are system
        # headers and stay external unless the repository happens to vendor one.
        return index.suffix(module)

    if lang == "rust":
        for candidate in _rust_candidates(module):
            found = index.suffix(candidate)
            if found:
                return found
        return []

    if lang == "go":
        # A Go import names a package directory, so it fans out to that
        # directory's files rather than to one file. `_test.go` files are
        # excluded: the toolchain compiles them only for that package's own
        # test binary, so an importer never depends on them and an edge saying
        # so would put test files in production import cycles.
        for directory in _go_package_dirs(module, index):
            found = index.in_dir(directory, exclude_suffix="_test.go")
            if found:
                return found[:GO_FANOUT]
        return []

    return []


def resolve_file_imports(conn: sqlite3.Connection, file_ids: set[str] | None,
                         commit: str | None,
                         deadline: float | None = None) -> dict[str, Any]:
    """Second pass: turn stored import specifiers into IMPORTS edges.

    Rewrites every IMPORTS edge out of each file it touches, so a removed
    import stops being reported and edges written by an older, statement-text
    scheme are replaced rather than accumulating alongside the real ones.

    `deadline` is a wall-clock time (time.time()) after which the pass stops
    between batches and reports `truncated`. A stopped pass resolves fewer
    files; it never leaves one half-resolved, because a file's IMPORTS edges
    are rewritten wholesale inside one transaction.
    """
    if file_ids is not None and not file_ids:
        return {"resolved": 0, "external": 0, "files": 0, "truncated": False}
    if deadline is not None and time.time() > deadline:
        # Before the queries, not after: building the file index and reading
        # every specifier is itself the expensive part on a large repository.
        return {"resolved": 0, "external": 0, "files": 0, "truncated": True}

    index = FileIndex(rows(conn.execute(
        "SELECT file_id, path FROM files WHERE status='ACTIVE'")))

    if file_ids is None:
        targets = rows(conn.execute(
            "SELECT file_id, path, lang, imports_raw FROM files"
            " WHERE status='ACTIVE' AND imports_raw IS NOT NULL"))
    else:
        targets = []
        chunk = list(file_ids)
        for start in range(0, len(chunk), IMPORT_BATCH):
            part = chunk[start:start + IMPORT_BATCH]
            placeholders = ",".join("?" for _ in part)
            targets.extend(rows(conn.execute(
                f"SELECT file_id, path, lang, imports_raw FROM files"
                f" WHERE file_id IN ({placeholders}) AND status='ACTIVE'"
                f" AND imports_raw IS NOT NULL", tuple(part))))

    resolved = external = 0
    truncated = False
    stamp = now()
    for start in range(0, len(targets), IMPORT_BATCH):
        if deadline is not None and time.time() > deadline:
            truncated = True
            targets = targets[:start]
            break
        with write_tx(conn):
            for row in targets[start:start + IMPORT_BATCH]:
                specs = jload(row["imports_raw"], []) or []
                from_path = (row["path"] or "").replace("\\", "/")
                lang = row["lang"] or ""
                conn.execute(
                    "DELETE FROM code_edges WHERE from_id=? AND kind='IMPORTS'",
                    (row["file_id"],))
                written: set[str] = set()
                # Eager imports first. A module that is imported both at module
                # level and again inside a function is eagerly imported, and
                # the dedup below keeps whichever edge is written first.
                specs = sorted(specs, key=lambda s: bool(s.get("deferred")))
                for spec in specs:
                    targets_found = resolve_specifier(spec, lang, from_path, index)
                    if targets_found:
                        pairs = [(t, "deterministic", 1.0) for t in targets_found
                                 if t != row["file_id"]]
                    elif spec.get("probe"):
                        # `from a import b` where b turned out to be a class or
                        # function, not a submodule. Nothing was imported from
                        # outside the program, so recording an external module
                        # here would invent a dependency that does not exist.
                        continue
                    else:
                        module = (spec.get("module") or "").strip()
                        if not module:
                            continue
                        pairs = [(f"extern:{module[:160]}", "external", 1.0)]
                    # A deferred import still creates a dependency, so the edge
                    # is written either way - but it is marked, because it
                    # cannot force a module-initialisation order and so must
                    # not be counted as closing an import cycle.
                    source = "tree-sitter:deferred" if spec.get("deferred") else "tree-sitter"
                    for to_id, edge_class, confidence in pairs:
                        if to_id in written:
                            continue
                        written.add(to_id)
                        conn.execute(
                            "INSERT INTO code_edges (edge_id, from_id, to_id, kind,"
                            " edge_class, status, confidence, source, valid_from_commit,"
                            " created_at, last_verified_at)"
                            " VALUES (?,?,?,'IMPORTS',?,'ACTIVE',?,?,?,?,?)"
                            " ON CONFLICT(from_id, to_id, kind) DO UPDATE SET"
                            " status='ACTIVE', valid_until_commit=NULL,"
                            " edge_class=excluded.edge_class,"
                            " confidence=excluded.confidence,"
                            " source=excluded.source,"
                            " last_verified_at=excluded.last_verified_at",
                            (ids.new_id(ids.EDGE), row["file_id"], to_id, edge_class,
                             confidence, source, commit, stamp, stamp))
                        if edge_class == "external":
                            external += 1
                        else:
                            resolved += 1

    return {"resolved": resolved, "external": external, "files": len(targets),
            "truncated": truncated}
