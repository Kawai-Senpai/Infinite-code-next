"""Incremental code indexing.

Walks the working tree, parses what changed, and maintains files, symbols and
deterministic code edges in the durable repo store.

Two rules shape this module:

1. Deleted things become tombstones, never DELETEs (PLAN.md section 4). A
   symbol row that vanishes takes its decisions, warnings and bug history with
   it, so a symbol that is gone from source is marked DELETED with its last
   known path and the commit that removed it. That record cannot be rebuilt by
   parsing code that no longer contains the symbol, which is exactly why
   symbols live in the durable store and not the cache.

2. Symbol identity is preserved across edits where we can prove it. When a file
   is reparsed, existing rows are matched by fingerprint or symbol path before
   any new ID is minted, so anchors keep pointing at the same symbol_id through
   a rename.
"""

from __future__ import annotations

import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from . import flows, ids, imports as import_resolution, parsing
from .diagnostics import TEST_PATH
from .db import jdump, jload, one, rows, write_tx
from .identity import changed_files, run_git

MAX_FILE_BYTES = 2_000_000

# Largest file a budgeted pass will attempt. Nothing is dropped: a file over
# this size is handed to the unbudgeted background run, which indexes it in
# full. It exists because one file can outlast the whole budget - on
# whatsapp-ghost, minified vendor bundles of 284 KB to 1.1 MB each took 8-12s
# of tree-sitter apiece against an 8s budget, while ordinary source files parse
# in milliseconds. The budget can only be checked between files, so without
# this the first big file overruns it however often the clock is read.
BUDGETED_FILE_BYTES = 128_000
CALL_BATCH = 300

# How much each call-resolution tier is worth, as (edge_class, confidence).
#
# These columns already existed but carried constants, so a bare name that
# happened to be unique in the repository was recorded with the same certainty
# as a call through an explicit receiver. Anything below `same_file` is a
# judgement rather than a reading of the syntax, and is classed as inferred so
# a consumer can choose to trust only what was actually proven.
CALL_TIERS: dict[str, tuple[str, float]] = {
    "receiver_self":   ("deterministic", 1.0),
    "receiver_class":  ("deterministic", 0.95),
    # The receiver is a local or a field whose type was recorded at extraction
    # time, and that type declares the callee. Inferred rather than
    # deterministic: the binding is read off one assignment, so a name
    # reassigned somewhere this pass cannot see would point it elsewhere.
    "receiver_typed":  ("inferred", 0.85),
    # The receiver names a module this file imports, and the callee is defined
    # in it. Deterministic because it rests on a resolved import edge rather
    # than on a name being unusual.
    "imported_module": ("deterministic", 0.92),
    "same_file":       ("deterministic", 0.9),
    "imported":        ("inferred", 0.8),
    "unique_global":   ("inferred", 0.6),
}

DEFAULT_EXCLUDES = {
    ".git", ".agit", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    "env", ".env", "dist", "build", "target", ".next", ".nuxt", ".output",
    "vendor", ".idea", ".vscode", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "coverage", ".tox", "site-packages", ".gradle", "bin", "obj", ".terraform",
    ".cache", "__snapshots__", ".pnpm-store", ".yarn",
}


def now() -> str:
    # Milliseconds, not seconds. Ordering questions ("did this caller
    # appear after that memory was verified?") are decided by comparing
    # these, and second precision made same-second events compare equal,
    # so a genuinely late caller went unreported.
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _excluded(rel: Path, extra: set[str]) -> bool:
    parts = set(rel.parts)
    if parts & DEFAULT_EXCLUDES:
        return True
    for pattern in extra:
        if rel.match(pattern):
            return True
    return False


def iter_source_files(root: Path, extra_excludes: Iterable[str] = ()):
    """Yield parseable files as they are discovered.

    A generator, not a list, because the time budget has to cover discovery as
    well as parsing. Building the full list first meant a large repository
    could burn the entire budget walking directories and then index nothing at
    all - measured at 4621 files: 29s spent, 0 files indexed.

    os.scandir rather than Path.iterdir: its DirEntry carries the file type
    from the directory read, so is_dir/is_file cost no extra stat call. That is
    most of the difference on Windows.
    """
    extra = set(extra_excludes)
    stack = [str(root)]
    root_str = str(root)
    while stack:
        directory = stack.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    name = entry.name
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if name in DEFAULT_EXCLUDES:
                                continue
                            rel = Path(entry.path[len(root_str):].lstrip("\\/"))
                            if extra and _excluded(rel, extra):
                                continue
                            stack.append(entry.path)
                        elif entry.is_file(follow_symlinks=False):
                            if os.path.splitext(name)[1].lower() not in parsing.LANGUAGES:
                                continue
                            rel = Path(entry.path[len(root_str):].lstrip("\\/"))
                            if extra and _excluded(rel, extra):
                                continue
                            yield Path(entry.path)
                    except OSError:
                        continue
        except (OSError, PermissionError):
            continue


def walk_source_files(root: Path, extra_excludes: Iterable[str] = ()) -> list[Path]:
    """Every parseable file in the tree, excludes applied."""
    return list(iter_source_files(root, extra_excludes))


def _defer_under_budget(path: Path) -> bool:
    """Is this file too large to attempt while a budget is running?"""
    try:
        return path.stat().st_size > BUDGETED_FILE_BYTES
    except OSError:
        return False


def _is_test_path(rel_path: str) -> bool:
    """Whether a path is test code, for clone extraction.

    Reuses diagnostics.TEST_PATH rather than restating the pattern: two copies
    of this regex would drift, and one of them would then disagree with the
    other about what counts as a test.
    """
    return bool(TEST_PATH.search(rel_path.lower()))


class Indexer:
    def __init__(self, conn: sqlite3.Connection, root: Path, repo_id: str,
                 excludes: Iterable[str] = ()):
        self.conn = conn
        self.root = root
        self.repo_id = repo_id
        self.excludes = set(excludes)
        self._touched: set[str] = set()
        # Import resolution is a second pass too, so it needs its own touched
        # set: symbol ids and file ids are not interchangeable.
        self._touched_files: set[str] = set()
        # Set by resolve_calls when it stopped at a deadline rather than
        # finishing. Read by full_index, which must report a truncated run so
        # the background pass re-resolves what was left.
        self.resolution_truncated = False

    # ------------------------------------------------------------------ files

    def _file_row(self, rel_path: str) -> dict[str, Any] | None:
        return one(self.conn.execute("SELECT * FROM files WHERE path = ?", (rel_path,)))

    def read_and_parse(self, abs_path: Path) -> dict[str, Any] | None:
        """Read and parse a file with no database involvement.

        Parsing must happen outside the write transaction. Doing it inside
        meant tree-sitter ran while holding SQLite's write lock, so a batch of
        files blocked every other server process for seconds - observed live as
        `database is locked` on a large repository. Reads are cheap and
        parallel-safe; only the write needs the lock.
        """
        rel_path = abs_path.relative_to(self.root).as_posix()
        lang = parsing.language_for(abs_path)
        if lang is None:
            return {"path": rel_path, "skipped": "unsupported"}
        try:
            data = abs_path.read_bytes()
        except OSError as exc:
            return {"path": rel_path, "skipped": f"unreadable: {exc}"}
        if len(data) > MAX_FILE_BYTES:
            return {"path": rel_path, "skipped": "too large"}

        return {
            "path": rel_path,
            "lang": lang,
            "data": data,
            "digest": parsing.content_hash(data),
            "symbols": parsing.parse_file(abs_path, data, lang),
            "imports": parsing.file_import_specs(data, lang),
            # Computed here, with the parse, so it lands outside the write
            # transaction for the same reason parsing does: tree-sitter must
            # never run while holding SQLite's write lock.
            #
            # Test files are skipped outright. Duplicated setup between tests
            # is deliberate - a test that shares a fixture with the code under
            # test stops being an independent check - so reporting it as a
            # clone would train the agent to ignore the whole report.
            "fragments": ([] if _is_test_path(rel_path)
                          else parsing.extract_fragments(data, lang)),
        }

    def index_file(self, abs_path: Path, commit: str | None) -> dict[str, Any]:
        """Parse and store one file. Convenience wrapper for single-file use."""
        commit = normalize_commit(commit)
        parsed = self.read_and_parse(abs_path)
        if parsed is None or parsed.get("skipped"):
            return parsed or {"skipped": "unknown"}
        return self.store_parsed(parsed, commit)

    def store_parsed(self, parsed: dict[str, Any], commit: str | None) -> dict[str, Any]:
        """Write an already-parsed file. Must run inside a write transaction."""
        commit = normalize_commit(commit)
        rel_path = parsed["path"]
        lang = parsed["lang"]
        data = parsed["data"]
        digest = parsed["digest"]
        symbols = parsed["symbols"]
        imports = parsed.get("imports") or []

        existing = self._file_row(rel_path)
        if existing and existing["content_hash"] == digest and existing["status"] == "ACTIVE":
            # Unchanged content is the normal path and must stay cheap - but a
            # content hash only proves the FILE did not move. It cannot detect
            # that the EXTRACTOR moved, and a store written by an older build is
            # missing whatever that build did not know how to read.
            #
            # So the file row carries the extractor version that wrote it. A row
            # behind parsing.EXTRACT_VERSION falls through to a full re-store,
            # which is the only thing that refreshes symbol-level data too:
            # _sync_symbols is skipped entirely by this fast path.
            #
            # An earlier attempt used "imports_raw IS NULL" as the staleness
            # proxy. It survived exactly one version step: once decorators were
            # added, every file a partial run had already touched had a non-NULL
            # imports_raw, looked migrated, and was stranded without decorators -
            # which silently emptied the entry-point table. A stamped version
            # cannot strand a row that way, however many extractions are added.
            #
            # Re-storing marks the file touched, so resolution follows. Doing
            # that for every unchanged file instead would re-resolve the whole
            # repository on every warm open - the regression in mem_0cae7144d9b2.
            if (existing["extract_version"] or 0) >= parsing.EXTRACT_VERSION:
                return {"path": rel_path, "unchanged": True}
            backfilling = True
        else:
            backfilling = False

        if existing:
            file_id = existing["file_id"]
            self.conn.execute(
                "UPDATE files SET lang=?, size=?, content_hash=?, status='ACTIVE',"
                " last_seen_commit=?, deleted_at_commit=NULL, indexed_at=?, imports_raw=?,"
                " extract_version=? WHERE file_id=?",
                (lang, len(data), digest, commit, now(), jdump(imports),
                 parsing.EXTRACT_VERSION, file_id),
            )
        else:
            file_id = ids.new_id(ids.FILE)
            self.conn.execute(
                "INSERT INTO files (file_id, path, lang, size, content_hash, status,"
                " first_seen_commit, last_seen_commit, indexed_at, imports_raw,"
                " extract_version) VALUES (?,?,?,?,?,'ACTIVE',?,?,?,?,?)",
                (file_id, rel_path, lang, len(data), digest, commit, commit, now(),
                 jdump(imports), parsing.EXTRACT_VERSION),
            )

        stats = self._sync_symbols(file_id, rel_path, symbols, commit)
        self._sync_fragments(file_id, parsed.get("fragments") or [])
        self._touched_files.add(file_id)
        result = {"path": rel_path, "symbols": len(symbols), **stats}
        if backfilling:
            result["backfilled"] = True
        return result

    # -------------------------------------------------------------- fragments

    def _sync_fragments(self, file_id: str, fragments: list) -> int:
        """Replace this file's clone fragments wholesale.

        Delete-then-insert rather than the reconciliation _sync_symbols does,
        because a fragment has no identity worth preserving: nothing anchors to
        it, no memory references it, and it carries no history. Matching old
        rows to new ones would buy nothing and cost a comparison per fragment
        on every re-store.
        """
        self.conn.execute("DELETE FROM clone_fragments WHERE file_id=?", (file_id,))
        if not fragments:
            return 0
        stamp = now()
        self.conn.executemany(
            "INSERT INTO clone_fragments (fragment_id, file_id, symbol_path, lang,"
            " kind, start_byte, end_byte, line_start, line_end, token_count,"
            " content_fingerprint, alpha_fingerprint, skeleton_fingerprint,"
            " token_signature, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(ids.new_id(ids.FRAGMENT), file_id, f.symbol_path, f.lang, f.kind,
              f.start_byte, f.end_byte, f.line_start, f.line_end, f.token_count,
              f.content_fingerprint, f.alpha_fingerprint, f.skeleton_fingerprint,
              f.token_signature, stamp) for f in fragments])
        return len(fragments)

    # ---------------------------------------------------------------- symbols

    def _sync_symbols(self, file_id: str, rel_path: str, parsed: list[parsing.ParsedSymbol],
                      commit: str | None) -> dict[str, int]:
        """Reconcile parsed symbols against stored rows, preserving identity."""
        stored = rows(self.conn.execute(
            "SELECT * FROM symbols WHERE file_id = ? AND status = 'ACTIVE'", (file_id,)
        ))
        by_fingerprint = {r["content_fingerprint"]: r for r in stored if r["content_fingerprint"]}
        by_path = {r["symbol_path"]: r for r in stored}
        matched: set[str] = set()
        created = updated = 0

        for sym in parsed:
            # Identity match order: exact content, then same path (edited body),
            # then same structure + same name (moved within the file).
            row = by_fingerprint.get(sym.content_fingerprint)
            if row is None:
                row = by_path.get(sym.symbol_path)
            if row is None:
                for candidate in stored:
                    if (candidate["symbol_id"] not in matched
                            and candidate["skeleton_fingerprint"] == sym.skeleton_fingerprint
                            and candidate["name"] == sym.name):
                        row = candidate
                        break

            if row is not None and row["symbol_id"] not in matched:
                symbol_id = row["symbol_id"]
                matched.add(symbol_id)
                self.conn.execute(
                    "UPDATE symbols SET file_id=?, name=?, symbol_path=?, kind=?, lang=?, signature=?,"
                    " ast_path=?, start_byte=?, end_byte=?, line_start=?, line_end=?,"
                    " content_fingerprint=?, skeleton_fingerprint=?, prev_fingerprint=?,"
                    " next_fingerprint=?, token_signature=?, calls_raw=?, decorators=?,"
                    " receiver_types=?, status='ACTIVE',"
                    " last_known_path=?, last_seen_commit=?, deleted_at_commit=NULL,"
                    " updated_at=? WHERE symbol_id=?",
                    (file_id, sym.name, sym.symbol_path, sym.kind, sym.lang, sym.signature,
                     sym.ast_path, sym.start_byte, sym.end_byte, sym.line_start, sym.line_end,
                     sym.content_fingerprint, sym.skeleton_fingerprint, sym.prev_fingerprint,
                     sym.next_fingerprint, sym.token_signature, jdump(sym.calls),
                     jdump(sym.decorators), jdump(sym.receiver_types), rel_path, commit,
                     now(), symbol_id),
                )
                updated += 1
            else:
                symbol_id = ids.new_id(ids.SYMBOL)
                matched.add(symbol_id)
                self.conn.execute(
                    "INSERT INTO symbols (symbol_id, file_id, name, symbol_path, kind, lang, signature,"
                    " ast_path, start_byte, end_byte, line_start, line_end, content_fingerprint,"
                    " skeleton_fingerprint, prev_fingerprint, next_fingerprint, token_signature,"
                    " calls_raw, decorators, receiver_types, status, last_known_path,"
                    " last_seen_commit, updated_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'ACTIVE',?,?,?)",
                    (symbol_id, file_id, sym.name, sym.symbol_path, sym.kind, sym.lang, sym.signature,
                     sym.ast_path, sym.start_byte, sym.end_byte, sym.line_start, sym.line_end,
                     sym.content_fingerprint, sym.skeleton_fingerprint, sym.prev_fingerprint,
                     sym.next_fingerprint, sym.token_signature, jdump(sym.calls),
                     jdump(sym.decorators), jdump(sym.receiver_types), rel_path, commit, now()),
                )
                created += 1

            self.conn.execute("DELETE FROM fts_symbols WHERE symbol_id = ?", (symbol_id,))
            self.conn.execute(
                "INSERT INTO fts_symbols (symbol_id, symbol_path, name, signature, body)"
                " VALUES (?,?,?,?,?)",
                (symbol_id, sym.symbol_path, sym.name, sym.signature, sym.body[:900]),
            )
            self._touched.add(symbol_id)

        # Anything not matched is gone from source. Tombstone, never delete.
        tombstoned = 0
        for row in stored:
            if row["symbol_id"] not in matched:
                self.conn.execute(
                    "UPDATE symbols SET status='DELETED', deleted_at_commit=?, last_known_path=?,"
                    " updated_at=? WHERE symbol_id=?",
                    (commit, row["last_known_path"] or rel_path, now(), row["symbol_id"]),
                )
                self.conn.execute(
                    "UPDATE code_edges SET status='HISTORICAL', valid_until_commit=?"
                    " WHERE (from_id=? OR to_id=?) AND status='ACTIVE'",
                    (commit, row["symbol_id"], row["symbol_id"]),
                )
                tombstoned += 1

        return {"created": created, "updated": updated, "tombstoned": tombstoned}

    def _resolution_tables(self) -> dict[str, Any]:
        """Lookup tables the call tiers share, built once per resolution pass."""
        by_path: dict[str, list[str]] = {}
        by_name: dict[str, list[tuple[str, str]]] = {}
        for row in rows(self.conn.execute(
                "SELECT symbol_id, file_id, name, symbol_path FROM symbols"
                " WHERE status='ACTIVE'")):
            by_path.setdefault(row["symbol_path"], []).append(row["symbol_id"])
            by_name.setdefault(row["name"], []).append((row["symbol_id"], row["file_id"]))

        # Only real file targets matter here: an `extern:` edge names a module
        # outside the index, so it can never supply a callee.
        imports_of: dict[str, set[str]] = {}
        for row in rows(self.conn.execute(
                "SELECT from_id, to_id FROM code_edges WHERE kind='IMPORTS'"
                " AND status='ACTIVE' AND to_id NOT LIKE 'extern:%'")):
            imports_of.setdefault(row["from_id"], set()).add(row["to_id"])

        # What a receiver could be naming, per imported file. A qualified call
        # writes the MODULE, not a type: Go's `store.Lookup` names the package
        # directory `store`, and Python's `db.jload` names the module file
        # `db.py`. Neither has a symbol called `store` or `db` anywhere, so the
        # receiver tiers all miss and the call is dropped as ambiguous - which
        # is most of what Go leaves unresolved.
        aliases: dict[str, dict[str, set[str]]] = {}
        paths = {row["file_id"]: (row["path"] or "").replace("\\", "/")
                 for row in rows(self.conn.execute(
                     "SELECT file_id, path FROM files WHERE status='ACTIVE'"))}
        for from_id, targets in imports_of.items():
            table = aliases.setdefault(from_id, {})
            for target in targets:
                path = paths.get(target, "")
                if not path:
                    continue
                stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
                package = path.rsplit("/", 2)[-2] if "/" in path else ""
                for alias in (stem, package):
                    if alias:
                        table.setdefault(alias, set()).add(target)

        # Field types, keyed by the container that declares them. A field is
        # assigned in one method - usually the constructor - and read in
        # others, so a per-symbol table alone would resolve `self.store.get()`
        # only inside the method that wrote `self.store`. Types are unioned
        # across the container rather than overwritten: two constructors that
        # assign different types both stay, and resolution then declines
        # instead of picking one.
        field_types: dict[str, dict[str, list[str]]] = {}
        for row in rows(self.conn.execute(
                "SELECT symbol_path, receiver_types FROM symbols"
                " WHERE status='ACTIVE' AND receiver_types IS NOT NULL")):
            path = row["symbol_path"] or ""
            if "." not in path:
                continue
            owner = path.rsplit(".", 1)[0]
            for name, types in (jload(row["receiver_types"], {}) or {}).items():
                if not name.startswith("self."):
                    continue
                known = field_types.setdefault(owner, {}).setdefault(name, [])
                for type_name in types:
                    if type_name not in known:
                        known.append(type_name)

        return {"by_path": by_path, "by_name": by_name, "imports_of": imports_of,
                "aliases": aliases, "field_types": field_types}

    def _resolve_one_call(self, callee: str, caller: dict[str, Any],
                          tables: dict[str, Any]) -> tuple[str | None, str, int]:
        """Resolve one call site. Returns (target_id, tier, candidate_count).

        Tiers run strongest first and stop at the first that answers. Each one
        writes a different confidence, so a caller can tell a receiver-bound
        call from a bare name that merely happened to be unique in the
        repository - the previous scheme recorded both as certainty 1.0.

        A tier of "ambiguous" or "external" means no edge: resolution refuses
        to guess. Those are recorded in unresolved_calls instead, because an
        omitted edge nobody counted is indistinguishable from an absent call.
        """
        by_path = tables["by_path"]
        by_name = tables["by_name"]
        imports_of = tables["imports_of"]

        caller_types = jload(caller["receiver_types"], {}) or {}

        parts = [p for p in callee.strip().split(".") if p]
        if not parts:
            return None, "external", 0
        leaf = parts[-1]
        receiver = parts[-2] if len(parts) > 1 else None

        def pick(candidates: list[str]) -> str | None:
            usable = [c for c in candidates if c != caller["symbol_id"]]
            return usable[0] if len(usable) == 1 else None

        # Tier 1a: `self.x` / `this.x` - the receiver is the caller's own class,
        # which the caller's dotted symbol_path already names.
        if receiver in ("self", "this", "cls"):
            owner = (caller["symbol_path"].rsplit(".", 1)[0]
                     if "." in caller["symbol_path"] else "")
            if owner:
                hit = pick(by_path.get(owner + "." + leaf, []))
                if hit:
                    return hit, "receiver_self", 1

        # Tier 1b: an explicit receiver that names a container we indexed.
        if receiver:
            hit = pick(by_path.get(receiver + "." + leaf, []))
            if hit:
                return hit, "receiver_class", 1

        # Tier 1b2: the receiver is a local or a field with a recorded type.
        # This is what makes `store.get()` resolvable when `store` is neither a
        # class name nor a module - the commonest shape in object-oriented
        # code, and the reason methods dominated the unresolved list.
        #
        # Every candidate type is tried and the results unioned. One type that
        # declares the method gives an answer; two that both do means the
        # receiver's type is genuinely undecided here, and pick() declines
        # rather than choosing the first - a guess would point a real edge at
        # the wrong method, which is worse than no edge.
        if receiver:
            own_types = list(caller_types.get(receiver, []))
            owner_path = (caller["symbol_path"].rsplit(".", 1)[0]
                          if "." in (caller["symbol_path"] or "") else "")
            if owner_path:
                for type_name in tables["field_types"].get(owner_path, {}).get(
                        "self." + receiver, []):
                    if type_name not in own_types:
                        own_types.append(type_name)
            typed: list[str] = []
            for type_name in own_types:
                for candidate in by_path.get(type_name + "." + leaf, []):
                    if candidate not in typed:
                        typed.append(candidate)
            hit = pick(typed)
            if hit:
                return hit, "receiver_typed", len(typed)

        named = by_name.get(leaf, [])
        if not named:
            return None, "external", 0

        # Tier 1c: the receiver names a module this file imports. `store.Lookup`
        # is not a method on a type called `store` - it is the package. This is
        # read off the import graph rather than guessed, so it is as certain as
        # the import edge that supports it.
        if receiver:
            reachable = tables["aliases"].get(caller["file_id"], {}).get(receiver)
            if reachable:
                scoped = [sid for sid, fid in named if fid in reachable]
                hit = pick(scoped)
                if hit:
                    return hit, "imported_module", len(scoped)

        # Tier 2: defined in the calling file.
        same_file = [sid for sid, fid in named if fid == caller["file_id"]]
        hit = pick(same_file)
        if hit:
            return hit, "same_file", len(same_file)

        # Tier 3: defined in a file this one imports.
        reachable = imports_of.get(caller["file_id"], set())
        if reachable:
            imported = [sid for sid, fid in named if fid in reachable]
            hit = pick(imported)
            if hit:
                return hit, "imported", len(imported)

        # Tier 4: unique in the repository. Weakest tier that still emits an
        # edge - it is one name coincidence away from being wrong, which is why
        # it is written as inferred rather than deterministic.
        candidates = [sid for sid, _ in named]
        hit = pick(candidates)
        if hit:
            return hit, "unique_global", 1
        return None, "ambiguous", len(candidates)

    def resolve_calls(self, symbol_ids: set[str] | None, commit: str | None,
                      deadline: float | None = None) -> int:
        """Second pass: turn recorded call names into tiered CALLS edges.

        This has to be a separate pass. Call targets are resolved by name, and
        during the first pass a file is frequently indexed before the file
        defining what it calls - so resolving inline silently drops every edge
        that points "forward" in walk order. That was losing the test-to-code
        edges entirely, which are the ones investigate() most needs.

        Resolution still never guesses, but it no longer throws away what it
        could not decide: an ambiguous or unknown callee lands in
        unresolved_calls so a later query can say how many callers it is
        provably not showing.

        `deadline` is a wall-clock time (time.time()) after which the pass
        stops between batches and sets `resolution_truncated`. It resolves
        fewer symbols; it never leaves one half-resolved, because a symbol's
        outgoing edges are rewritten wholesale inside one transaction.
        """
        self.resolution_truncated = False
        if symbol_ids is not None and not symbol_ids:
            return 0
        if deadline is not None and time.time() > deadline:
            # Reading every symbol row and building the resolution tables is
            # itself minutes of work on a large repository, so the check comes
            # before the query, not after it.
            self.resolution_truncated = True
            return 0
        if symbol_ids is None:
            source_rows = rows(self.conn.execute(
                "SELECT symbol_id, file_id, name, symbol_path, calls_raw,"
                " receiver_types FROM symbols"
                " WHERE status='ACTIVE' AND calls_raw IS NOT NULL"
            ))
        else:
            chunk = list(symbol_ids)
            source_rows = []
            for start in range(0, len(chunk), 400):
                part = chunk[start:start + 400]
                placeholders = ",".join("?" for _ in part)
                source_rows.extend(rows(self.conn.execute(
                    f"SELECT symbol_id, file_id, name, symbol_path, calls_raw,"
                    f" receiver_types FROM symbols"
                    f" WHERE symbol_id IN ({placeholders})"
                    f" AND status='ACTIVE' AND calls_raw IS NOT NULL", tuple(part)
                )))
        if not source_rows:
            return 0

        tables = self._resolution_tables()

        # Commit in batches. A single transaction over every symbol in a large
        # repository held the write lock for minutes and starved every other
        # server process (PLAN 2 section 11: writes stay small and serialised).
        created = 0
        stamp = now()
        for start in range(0, len(source_rows), CALL_BATCH):
            if deadline is not None and time.time() > deadline:
                self.resolution_truncated = True
                break
            with write_tx(self.conn):
                for row in source_rows[start:start + CALL_BATCH]:
                    # Re-resolving replaces this symbol's outgoing call facts
                    # wholesale. Upserting alone left edges for calls the symbol
                    # no longer makes, and pinned them at their old tier.
                    self.conn.execute(
                        "DELETE FROM code_edges WHERE from_id=? AND kind='CALLS'",
                        (row["symbol_id"],))
                    self.conn.execute(
                        "DELETE FROM unresolved_calls WHERE from_id=?", (row["symbol_id"],))

                    for callee in sorted(set(jload(row["calls_raw"], []) or [])):
                        callee = str(callee)
                        leaf = callee.rsplit(".", 1)[-1].strip()
                        if not leaf or leaf == row["name"]:
                            continue
                        target_id, tier, count = self._resolve_one_call(
                            callee, row, tables)
                        if target_id is None:
                            self.conn.execute(
                                "INSERT OR REPLACE INTO unresolved_calls"
                                " (from_id, leaf, callee_raw, reason, candidates, commit_id)"
                                " VALUES (?,?,?,?,?,?)",
                                (row["symbol_id"], leaf, callee[:160], tier, count, commit))
                            continue
                        edge_class, confidence = CALL_TIERS[tier]
                        self.conn.execute(
                            "INSERT INTO code_edges (edge_id, from_id, to_id, kind, edge_class,"
                            " status, confidence, source, valid_from_commit, created_at,"
                            " last_verified_at)"
                            " VALUES (?,?,?,'CALLS',?,'ACTIVE',?,?,?,?,?)"
                            " ON CONFLICT(from_id, to_id, kind) DO UPDATE SET status='ACTIVE',"
                            " valid_until_commit=NULL, edge_class=excluded.edge_class,"
                            " confidence=excluded.confidence, source=excluded.source,"
                            " last_verified_at=excluded.last_verified_at",
                            (ids.new_id(ids.EDGE), row["symbol_id"], target_id, edge_class,
                             confidence, "tree-sitter:" + tier, commit, stamp, stamp),
                        )
                        created += 1
        return created

    # ------------------------------------------------------------------- runs

    def full_index(self, commit: str | None, budget_seconds: float | None = None) -> dict[str, Any]:
        started = time.time()
        # A deadline, not a duration: every pass below has to be able to ask
        # "is there time left" without knowing when the run started. Only None
        # means unbudgeted - budget_seconds=0 means "no time at all".
        deadline = (started + budget_seconds) if budget_seconds is not None else None
        report = {"mode": "full", "files_seen": 0, "files_indexed": 0,
                  "symbols": 0, "skipped": 0, "deferred": 0, "truncated": False}
        seen_paths: set[str] = set()

        def out_of_time() -> bool:
            return deadline is not None and time.time() > deadline

        # Discovery is streamed and batched: one transaction per file on a
        # 4000-file repo is thousands of fsyncs, and the budget must cover
        # walking as well as parsing.
        batch: list[Path] = []

        def flush() -> None:
            if not batch:
                return
            # Parse first, with no lock held.
            parsed_batch = []
            for path in batch:
                # The clock is read per file, not per batch. Asking only
                # between batches let one batch run unchecked for minutes:
                # whatsapp-ghost holds 25,641 symbols in 37 files, so its very
                # first batch was the entire repository and an 8s budget
                # produced a 275s open.
                if out_of_time():
                    report["truncated"] = True
                    break
                if deadline is not None and _defer_under_budget(path):
                    # Deferred, not skipped: the run is marked truncated, so
                    # the background pass that follows indexes this file with
                    # no budget at all. The final index is identical either
                    # way; only the order of the work changes.
                    report["deferred"] += 1
                    report["truncated"] = True
                    continue
                parsed = self.read_and_parse(path)
                if parsed is None or parsed.get("skipped"):
                    report["skipped"] += 1
                else:
                    parsed_batch.append(parsed)
            batch.clear()
            if not parsed_batch:
                return
            # Then write, holding the lock only for the inserts. Everything
            # parsed is stored even when the budget ran out mid-batch: the
            # deadline stops new work, it never throws away work already done.
            with write_tx(self.conn):
                for parsed in parsed_batch:
                    result = self.store_parsed(parsed, commit)
                    if not result.get("unchanged"):
                        report["files_indexed"] += 1
                        report["symbols"] += result.get("symbols", 0)

        for abs_path in iter_source_files(self.root, self.excludes):
            report["files_seen"] += 1
            seen_paths.add(abs_path.relative_to(self.root).as_posix())
            batch.append(abs_path)
            if len(batch) >= 40:
                flush()
                if report["truncated"] or out_of_time():
                    report["truncated"] = True
                    break
        if not report["truncated"]:
            flush()

        if not report["truncated"]:
            with write_tx(self.conn):
                report["files_tombstoned"] = self._tombstone_missing(seen_paths, commit)

        if report["truncated"]:
            # The passes below are whole-repository by nature, and on a large
            # repository each costs more than the walk that precedes them. Over
            # a symbol table that is knowingly incomplete they spend that cost
            # on a result the background full index is about to recompute from
            # nothing, which is how a budgeted first index still answered in
            # 275s. Defer them; the caller sees index_state "partial" until the
            # unbudgeted background run lands.
            report["import_edges"] = {"resolved": 0, "external": 0, "files": 0,
                                      "deferred": "index truncated"}
            report["call_edges"] = 0
            report["entry_points"] = {"deferred": "index truncated"}
            report["areas"] = {"deferred": "index truncated"}
            report["seconds"] = round(time.time() - started, 2)
            return report

        # Second pass. Imports resolve first: tiered call resolution consults
        # the IMPORTS edges to prefer a callee this file can actually reach.
        #
        # Both stop at the same deadline the walk used, and stopping is safe to
        # resume because each rewrites one file's or one symbol's edges
        # wholesale: what was processed is complete, what was not is untouched,
        # and the background run re-resolves the lot with no budget at all.
        report["import_edges"] = import_resolution.resolve_file_imports(
            self.conn, None, commit, deadline=deadline)
        if report["import_edges"].get("truncated"):
            report["truncated"] = True
        report["call_edges"] = self.resolve_calls(None, commit, deadline=deadline)
        if self.resolution_truncated:
            report["truncated"] = True

        # Third pass. Entry points need every symbol present; communities need
        # every call edge, so both have to follow resolution rather than run
        # alongside it - and neither is worth computing over a graph still
        # missing edges.
        if report["truncated"]:
            report["entry_points"] = {"deferred": "index truncated"}
            report["areas"] = {"deferred": "index truncated"}
        else:
            report["entry_points"] = flows.detect_entry_points(self.conn, commit)
            report["areas"] = flows.detect_communities(self.conn)

        report["seconds"] = round(time.time() - started, 2)
        return report

    def incremental_index(self, commit: str | None, since_commit: str | None,
                          budget_seconds: float | None = None) -> dict[str, Any]:
        """Index only what git says changed, plus everything uncommitted.

        `budget_seconds` bounds only the full-index fallback below. The
        incremental path itself is bounded by the size of the edit, but the
        fallback is a whole-repository walk and must not run unbudgeted in the
        foreground just because git could not say what changed.
        """
        started = time.time()
        paths_changed, complete = changed_files(self.root, since_commit)
        if not complete:
            return self.full_index(commit, budget_seconds=budget_seconds)

        report = {"mode": "incremental", "files_seen": len(paths_changed),
                  "files_indexed": 0, "symbols": 0, "skipped": 0, "files_tombstoned": 0}
        for rel in paths_changed:
            abs_path = self.root / rel
            if not abs_path.exists():
                with write_tx(self.conn):
                    report["files_tombstoned"] += self._tombstone_path(rel, commit)
                continue
            if not abs_path.is_file() or parsing.language_for(abs_path) is None:
                continue
            parsed = self.read_and_parse(abs_path)
            if parsed is None or parsed.get("skipped"):
                report["skipped"] += 1
                continue
            with write_tx(self.conn):
                result = self.store_parsed(parsed, commit)
            if not result.get("unchanged"):
                report["files_indexed"] += 1
                report["symbols"] += result.get("symbols", 0)

        # `or None` here meant "nothing changed" became "resolve everything":
        # resolve_calls(None) is the full-index path, so a no-op open
        # re-resolved every symbol in the repository. Measured on a
        # 34,747-symbol repo, that turned a warm open into minutes of work.
        # An empty touched set must resolve nothing.
        #
        # Imports are re-resolved only for the files that changed, for the same
        # reason. The known cost: adding file B does not retroactively upgrade
        # an untouched file A's `extern:b` edge to a real one until A itself is
        # reindexed. Re-resolving every file to catch that would restore exactly
        # the whole-repository warm-open cost this comment exists to prevent;
        # `workspace(action='reindex')` is the deliberate way to force it.
        report["import_edges"] = import_resolution.resolve_file_imports(
            self.conn, self._touched_files, commit)
        report["call_edges"] = self.resolve_calls(self._touched, commit)

        # Only worth recomputing when something actually moved. Both passes are
        # whole-graph by nature, so running them on a no-op warm open would
        # reintroduce exactly the cost mem_0cae7144d9b2 records.
        if report["files_indexed"] or report["files_tombstoned"]:
            report["entry_points"] = flows.detect_entry_points(self.conn, commit)
            report["areas"] = flows.detect_communities(self.conn)

        report["seconds"] = round(time.time() - started, 2)
        return report

    def _tombstone_path(self, rel_path: str, commit: str | None) -> int:
        row = self._file_row(rel_path)
        if row is None or row["status"] == "DELETED":
            return 0
        self.conn.execute(
            "UPDATE files SET status='DELETED', deleted_at_commit=?, indexed_at=? WHERE file_id=?",
            (commit, now(), row["file_id"]),
        )
        self.conn.execute(
            "UPDATE symbols SET status='DELETED', deleted_at_commit=?, updated_at=?"
            " WHERE file_id=? AND status='ACTIVE'",
            (commit, now(), row["file_id"]),
        )
        return 1

    def _tombstone_missing(self, seen_paths: set[str], commit: str | None) -> int:
        count = 0
        for row in rows(self.conn.execute("SELECT path FROM files WHERE status='ACTIVE'")):
            if row["path"] not in seen_paths:
                count += self._tombstone_path(row["path"], commit)
        return count


def index_state(conn: sqlite3.Connection) -> dict[str, Any]:
    def n(sql: str) -> int:
        row = one(conn.execute(sql))
        return int(row["n"]) if row else 0

    return {
        "files_active": n("SELECT COUNT(*) AS n FROM files WHERE status='ACTIVE'"),
        "files_deleted": n("SELECT COUNT(*) AS n FROM files WHERE status='DELETED'"),
        "symbols_active": n("SELECT COUNT(*) AS n FROM symbols WHERE status='ACTIVE'"),
        "symbols_deleted": n("SELECT COUNT(*) AS n FROM symbols WHERE status='DELETED'"),
        "code_edges_active": n("SELECT COUNT(*) AS n FROM code_edges WHERE status='ACTIVE'"),
        "code_edges_historical": n("SELECT COUNT(*) AS n FROM code_edges WHERE status='HISTORICAL'"),
    }


# Ref names that are emphatically not commit ids. Storing one silently
# corrupts every commit comparison downstream: `unreviewed_caller` compares a
# symbol's last_seen_commit against an anchor's last_verified_commit, so a
# store holding "HEAD" on one side and a real SHA on the other reports every
# governed symbol as unreviewed, permanently.
_NOT_A_COMMIT = {"HEAD", "head", "@", "ORIG_HEAD", "FETCH_HEAD", "MERGE_HEAD"}


def normalize_commit(commit: str | None) -> str | None:
    """Accept a commit id, reject a ref name.

    A ref is a moving pointer; a commit id is a fact. Only the latter can be
    compared for equality later, so anything else becomes None - unknown is a
    state the graph handles, a wrong value is not.
    """
    if not commit:
        return None
    text = str(commit).strip()
    if not text or text in _NOT_A_COMMIT:
        return None
    return text


def head_commit(root: Path) -> str | None:
    code, out, _ = run_git(["rev-parse", "HEAD"], root)
    return normalize_commit(out) if code == 0 and out else None
