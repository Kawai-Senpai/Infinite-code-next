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

from . import ids, parsing
from .db import jdump, jload, one, rows, write_tx
from .identity import changed_files, run_git

MAX_FILE_BYTES = 2_000_000
CALL_BATCH = 300

DEFAULT_EXCLUDES = {
    ".git", ".agit", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    "env", ".env", "dist", "build", "target", ".next", ".nuxt", ".output",
    "vendor", ".idea", ".vscode", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "coverage", ".tox", "site-packages", ".gradle", "bin", "obj", ".terraform",
    ".cache", "__snapshots__", ".pnpm-store", ".yarn",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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


class Indexer:
    def __init__(self, conn: sqlite3.Connection, root: Path, repo_id: str,
                 excludes: Iterable[str] = ()):
        self.conn = conn
        self.root = root
        self.repo_id = repo_id
        self.excludes = set(excludes)
        self._touched: set[str] = set()

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
            "imports": parsing.file_imports(data, lang),
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

        existing = self._file_row(rel_path)
        if existing and existing["content_hash"] == digest and existing["status"] == "ACTIVE":
            return {"path": rel_path, "unchanged": True}

        if existing:
            file_id = existing["file_id"]
            self.conn.execute(
                "UPDATE files SET lang=?, size=?, content_hash=?, status='ACTIVE',"
                " last_seen_commit=?, deleted_at_commit=NULL, indexed_at=? WHERE file_id=?",
                (lang, len(data), digest, commit, now(), file_id),
            )
        else:
            file_id = ids.new_id(ids.FILE)
            self.conn.execute(
                "INSERT INTO files (file_id, path, lang, size, content_hash, status,"
                " first_seen_commit, last_seen_commit, indexed_at) VALUES (?,?,?,?,?,'ACTIVE',?,?,?)",
                (file_id, rel_path, lang, len(data), digest, commit, commit, now()),
            )

        stats = self._sync_symbols(file_id, rel_path, symbols, commit)
        self._sync_imports(file_id, parsed.get("imports") or [], commit)
        return {"path": rel_path, "symbols": len(symbols), **stats}

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
                    " next_fingerprint=?, token_signature=?, calls_raw=?, status='ACTIVE',"
                    " last_known_path=?, last_seen_commit=?, deleted_at_commit=NULL,"
                    " updated_at=? WHERE symbol_id=?",
                    (file_id, sym.name, sym.symbol_path, sym.kind, sym.lang, sym.signature,
                     sym.ast_path, sym.start_byte, sym.end_byte, sym.line_start, sym.line_end,
                     sym.content_fingerprint, sym.skeleton_fingerprint, sym.prev_fingerprint,
                     sym.next_fingerprint, sym.token_signature, jdump(sym.calls), rel_path, commit,
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
                    " calls_raw, status, last_known_path, last_seen_commit, updated_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'ACTIVE',?,?,?)",
                    (symbol_id, file_id, sym.name, sym.symbol_path, sym.kind, sym.lang, sym.signature,
                     sym.ast_path, sym.start_byte, sym.end_byte, sym.line_start, sym.line_end,
                     sym.content_fingerprint, sym.skeleton_fingerprint, sym.prev_fingerprint,
                     sym.next_fingerprint, sym.token_signature, jdump(sym.calls), rel_path, commit, now()),
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

    def resolve_calls(self, symbol_ids: set[str] | None, commit: str | None) -> int:
        """Second pass: turn recorded call names into CALLS edges.

        This has to be a separate pass. Call targets are resolved by name, and
        during the first pass a file is frequently indexed before the file
        defining what it calls - so resolving inline silently drops every edge
        that points "forward" in walk order. That was losing the test-to-code
        edges entirely, which are the ones investigate() most needs.

        Name resolution stays shallow on purpose: an ambiguous or unknown
        callee produces no edge rather than a guess. Deterministic edges have
        to stay deterministic or the trust classes mean nothing.
        """
        if symbol_ids is not None and not symbol_ids:
            return 0
        if symbol_ids is None:
            source_rows = rows(self.conn.execute(
                "SELECT symbol_id, name, calls_raw FROM symbols WHERE status='ACTIVE'"
                " AND calls_raw IS NOT NULL"
            ))
        else:
            chunk = list(symbol_ids)
            source_rows = []
            for start in range(0, len(chunk), 400):
                part = chunk[start:start + 400]
                placeholders = ",".join("?" for _ in part)
                source_rows.extend(rows(self.conn.execute(
                    f"SELECT symbol_id, name, calls_raw FROM symbols WHERE symbol_id IN ({placeholders})"
                    f" AND status='ACTIVE' AND calls_raw IS NOT NULL", tuple(part)
                )))

        # Commit in batches. A single transaction over every symbol in a large
        # repository held the write lock for minutes and starved every other
        # server process (PLAN 2 section 11: writes stay small and serialised).
        created = 0
        cache: dict[str, str | None] = {}
        for start in range(0, len(source_rows), CALL_BATCH):
            with write_tx(self.conn):
                for row in source_rows[start:start + CALL_BATCH]:
                    for callee in set(jload(row["calls_raw"], []) or []):
                        leaf = str(callee).rsplit(".", 1)[-1].strip()
                        if not leaf or leaf == row["name"]:
                            continue
                        if leaf not in cache:
                            matches = rows(self.conn.execute(
                                "SELECT symbol_id FROM symbols WHERE name = ? AND status='ACTIVE'"
                                " LIMIT 2", (leaf,),
                            ))
                            cache[leaf] = matches[0]["symbol_id"] if len(matches) == 1 else None
                        target_id = cache[leaf]
                        if not target_id or target_id == row["symbol_id"]:
                            continue
                        self.conn.execute(
                            "INSERT INTO code_edges (edge_id, from_id, to_id, kind, edge_class,"
                            " status, confidence, source, valid_from_commit, created_at,"
                            " last_verified_at)"
                            " VALUES (?,?,?,'CALLS','deterministic','ACTIVE',1.0,'tree-sitter',?,?,?)"
                            " ON CONFLICT(from_id, to_id, kind) DO UPDATE SET status='ACTIVE',"
                            " valid_until_commit=NULL,"
                            " last_verified_at=excluded.last_verified_at",
                            (ids.new_id(ids.EDGE), row["symbol_id"], target_id, commit,
                             now(), now()),
                        )
                        created += 1
        return created

    def _sync_imports(self, file_id: str, statements: list[str], commit: str | None) -> None:
        for statement in statements[:80]:
            self.conn.execute(
                "INSERT INTO code_edges (edge_id, from_id, to_id, kind, edge_class, status,"
                " confidence, source, valid_from_commit, created_at, last_verified_at)"
                " VALUES (?,?,?,'IMPORTS','deterministic','ACTIVE',1.0,'tree-sitter',?,?,?)"
                " ON CONFLICT(from_id, to_id, kind) DO UPDATE SET status='ACTIVE',"
                " valid_until_commit=NULL",
                (ids.new_id(ids.EDGE), file_id, f"import:{statement[:160]}", commit, now(), now()),
            )

    # ------------------------------------------------------------------- runs

    def full_index(self, commit: str | None, budget_seconds: float | None = None) -> dict[str, Any]:
        started = time.time()
        report = {"mode": "full", "files_seen": 0, "files_indexed": 0,
                  "symbols": 0, "skipped": 0, "truncated": False}
        seen_paths: set[str] = set()

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
                parsed = self.read_and_parse(path)
                if parsed is None or parsed.get("skipped"):
                    report["skipped"] += 1
                else:
                    parsed_batch.append(parsed)
            batch.clear()
            if not parsed_batch:
                return
            # Then write, holding the lock only for the inserts.
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
                if budget_seconds and (time.time() - started) > budget_seconds:
                    report["truncated"] = True
                    break
        if not report["truncated"]:
            flush()

        if not report["truncated"]:
            with write_tx(self.conn):
                report["files_tombstoned"] = self._tombstone_missing(seen_paths, commit)

        # Second pass: every symbol now exists, so call names can resolve.
        report["call_edges"] = self.resolve_calls(None, commit)

        report["seconds"] = round(time.time() - started, 2)
        return report

    def incremental_index(self, commit: str | None, since_commit: str | None) -> dict[str, Any]:
        """Index only what git says changed, plus everything uncommitted."""
        started = time.time()
        paths_changed, complete = changed_files(self.root, since_commit)
        if not complete:
            return self.full_index(commit)

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
        report["call_edges"] = self.resolve_calls(self._touched, commit)

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
