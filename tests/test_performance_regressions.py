"""Regressions for bugs that only appeared under live conditions.

Each of these was found by running against a real 4600-file repository or a
real MCP client, not by unit testing. They are cheap to assert and expensive to
rediscover.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from conftest import record_baseline

from icn import imports as import_mod
from icn import parsing, search as search_mod
from icn import workspace as ws_mod
from icn.db import init_repo_store, one, rows
from icn.indexer import Indexer, iter_source_files
from icn import paths


def test_investigate_does_not_spawn_a_git_process_per_symbol(workspace, monkeypatch):
    """`git log -1 -- <path>` per candidate blew past a 90s client timeout.

    Process spawn dominates on Windows, so recency must come from one call.
    """
    record_baseline(workspace)
    search_mod._RECENCY_CACHE.clear()

    calls: list[list[str]] = []
    real = search_mod.run_git

    def counting(args, cwd, *rest, **kwargs):
        calls.append(args)
        return real(args, cwd, *rest, **kwargs)

    monkeypatch.setattr(search_mod, "run_git", counting)
    search_mod.investigate(workspace.store, workspace.catalog, workspace.root,
                           "I want to change refresh token rotation", commit=workspace.commit)

    assert len(calls) <= 1, f"expected at most one git call, got {len(calls)}: {calls[:5]}"


def test_recency_map_is_cached_across_investigations(workspace, monkeypatch):
    record_baseline(workspace)
    search_mod._RECENCY_CACHE.clear()

    calls: list[int] = []
    real = search_mod.run_git

    def counting(args, cwd, *rest, **kwargs):
        calls.append(1)
        return real(args, cwd, *rest, **kwargs)

    monkeypatch.setattr(search_mod, "run_git", counting)
    for _ in range(3):
        search_mod.investigate(workspace.store, workspace.catalog, workspace.root,
                               "change refresh rotation", commit=workspace.commit)
    assert len(calls) <= 1


def test_parsing_happens_outside_the_write_transaction(project):
    """Parsing under the write lock starved other processes.

    read_and_parse must be a pure function of the filesystem: given a closed
    database connection it still works, which it cannot if it touches the DB.
    """
    ws = ws_mod.open_workspace(str(project.root))
    indexer = Indexer(ws.store, ws.root, ws.repo_id)
    ws.store.close()

    parsed = indexer.read_and_parse(project.root / "auth.py")
    assert parsed is not None
    assert parsed["symbols"], "parsing must not depend on the database"
    assert parsed["digest"]
    ws.catalog.close()


def test_two_indexers_on_one_repo_do_not_deadlock(project):
    """Observed live as `database is locked` on a large repository."""
    first = ws_mod.open_workspace(str(project.root))
    db_path = paths.repo_db_path(first.repo_id)
    repo_id, root, commit = first.repo_id, first.root, first.commit
    first.close()

    errors: list[Exception] = []

    def index() -> None:
        conn = init_repo_store(db_path)
        try:
            Indexer(conn, root, repo_id).full_index(commit)
        except Exception as exc:  # noqa: BLE001 - the point is that this stays empty
            errors.append(exc)
        finally:
            conn.close()

    threads = [threading.Thread(target=index) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=90)

    assert errors == [], f"concurrent indexing failed: {errors[:2]}"


def test_directory_walk_is_streamed_not_materialised(project):
    """The budget has to cover discovery, or a big repo indexes nothing."""
    generator = iter_source_files(project.root)
    first = next(generator)
    assert isinstance(first, Path), "walk must yield before the tree is fully scanned"
    generator.close()


def test_token_signature_is_fixed_size_and_still_discriminates():
    """Storing full token sequences drove a real store to 101MB."""
    long_tokens = [f"tok{i}" for i in range(5000)]
    signature = parsing.token_signature(long_tokens)

    assert len(signature) <= 32 * 9 + 8, "signature must not grow with input size"

    same = parsing.token_signature(list(long_tokens))
    assert parsing.token_similarity(signature, same) == 1.0

    nearly = parsing.token_signature(long_tokens[:-50] + ["extra"] * 50)
    unrelated = parsing.token_signature([f"other{i}" for i in range(5000)])

    assert parsing.token_similarity(signature, nearly) > 0.6
    assert parsing.token_similarity(signature, unrelated) < 0.1
    assert parsing.token_similarity("", signature) == 0.0


def test_store_stays_small_relative_to_source(workspace, project):
    """A sanity ceiling on bytes-per-symbol, so the sketch cannot regress."""
    ws_mod.ensure_indexed(workspace, force_full=True)
    symbols = one(workspace.store.execute(
        "SELECT COUNT(*) AS n FROM symbols WHERE status='ACTIVE'"))["n"]
    assert symbols > 0

    signature = one(workspace.store.execute(
        "SELECT token_signature FROM symbols WHERE token_signature IS NOT NULL LIMIT 1"))
    assert signature and len(signature["token_signature"]) < 400


def test_git_children_never_inherit_the_servers_stdin():
    """The single worst bug live testing found.

    This server speaks MCP over its own stdin. A git child that inherits it
    blocks waiting for input that never arrives - every tool call stalled for
    exactly the 20s git timeout - and can consume bytes of the protocol
    stream, which showed up client-side as truncated JSON. Both symptoms, one
    cause: subprocess must be given stdin=DEVNULL.
    """
    import inspect
    import subprocess

    from icn import agit, identity

    for module in (identity, agit):
        source = inspect.getsource(module)
        calls = source.count("subprocess.run(")
        detached = source.count("stdin=subprocess.DEVNULL")
        assert calls > 0
        assert detached == calls, (
            f"{module.__name__}: {calls} subprocess.run calls but {detached} detach stdin"
        )

    # And prove it behaves: git must not read from a pipe we never write to.
    import os
    import tempfile
    from pathlib import Path

    repo = Path(tempfile.mkdtemp())
    proc = subprocess.Popen([os.sys.executable, "-c", "import sys; sys.stdin.read()"],
                            stdin=subprocess.PIPE)
    try:
        code, out, _ = identity.run_git(["--version"], repo, timeout=10)
        assert code == 0 and "git version" in out
    finally:
        proc.kill()


def test_a_no_op_open_does_not_re_resolve_every_symbol(workspace, project, monkeypatch):
    """`self._touched or None` turned "nothing changed" into "resolve all".

    resolve_calls(None) is the full-index path, so an incremental pass with an
    empty touched set re-resolved every symbol in the repository. Measured on a
    34,747-symbol repo, that made a warm open take minutes.
    """
    ws_mod.ensure_indexed(workspace, force_full=True)

    seen: list[object] = []
    real = Indexer.resolve_calls

    def spy(self, symbol_ids, commit, **kwargs):
        seen.append(symbol_ids)
        return real(self, symbol_ids, commit, **kwargs)

    monkeypatch.setattr(Indexer, "resolve_calls", spy)
    report = ws_mod.ensure_indexed(workspace)

    assert report["files_indexed"] == 0, "nothing changed, so nothing should reindex"
    assert seen, "resolve_calls should still be called"
    assert seen[-1] is not None, "an empty touched set must never mean 'resolve everything'"
    assert seen[-1] == set(), "nothing was touched, so nothing should be resolved"


def test_a_ref_name_is_never_stored_as_a_commit_id(workspace, project):
    """A ref is a moving pointer; a commit id is a fact.

    Storing "HEAD" where a commit id belongs breaks equality forever:
    `unreviewed_caller` compares a symbol's last_seen_commit against an
    anchor's last_verified_commit, so a store holding "HEAD" on one side and a
    real SHA on the other reports every governed symbol as unreviewed. Measured
    live: 34 memories and 380 symbols all carrying "HEAD".
    """
    from icn.indexer import normalize_commit

    for ref in ("HEAD", "head", "@", "ORIG_HEAD", "FETCH_HEAD", "MERGE_HEAD", "", "  "):
        assert normalize_commit(ref) is None, f"{ref!r} is a ref, not a commit"
    assert normalize_commit("ebd11e85a825ce9817104075cc72ea9a70d11215") is not None

    # And it holds through the public path, whatever a caller passes.
    from icn import compiler
    compiler.record_event(workspace.store, workspace.catalog, workspace.repo_id,
                          workspace.root, "HEAD",
                          {"kind": "note", "summary": "recorded with a ref name",
                           "warnings": ["careful"], "symbols": ["refresh_session"]})

    stored = rows(workspace.store.execute(
        "SELECT created_commit FROM memories WHERE created_commit IS NOT NULL"))
    assert not any(r["created_commit"] == "HEAD" for r in stored)


def test_an_existing_store_holding_ref_names_is_repaired(workspace):
    """Existing stores must heal, not live with permanent false positives."""
    from icn.db import _repair_ref_commits, write_tx

    with write_tx(workspace.store):
        workspace.store.execute("UPDATE symbols SET last_seen_commit='HEAD'")
    assert rows(workspace.store.execute(
        "SELECT 1 FROM symbols WHERE last_seen_commit='HEAD' LIMIT 1"))

    _repair_ref_commits(workspace.store)

    assert not rows(workspace.store.execute(
        "SELECT 1 FROM symbols WHERE last_seen_commit='HEAD' LIMIT 1")), \
        "a ref name must be nulled out; unknown is honest, wrong is not"


def test_repeated_verification_does_not_grow_anchor_history(workspace, project):
    """A no-op verification is not history.

    Recording every "unchanged" pass filled the 25-entry ring buffer with
    identical rows and evicted the real re-anchors it exists to preserve -
    measured live at 25 consecutive "unchanged" entries on one anchor, shipped
    in every memory(get) response.
    """
    import json

    record_baseline(workspace)
    for _ in range(8):
        ws_mod.ensure_indexed(workspace)

    histories = [json.loads(r["reanchor_history"] or "[]") for r in rows(
        workspace.store.execute("SELECT reanchor_history FROM anchors"))]
    assert histories, "expected anchors"
    worst = max(len(h) for h in histories)
    assert worst <= 2, f"no-op verifications must collapse, got {worst} entries"

    for history in histories:
        unchanged = [e for e in history if e.get("transition") == "unchanged"]
        assert len(unchanged) <= 1, "consecutive no-ops must collapse into one"


def test_symbol_texts_never_joins_the_fts_virtual_table(workspace):
    """A LEFT JOIN onto fts_symbols is quadratic and ran inside the tool call.

    fts_symbols is an FTS5 virtual table and FTS5 has no secondary index, so
    joining it on symbol_id makes SQLite rescan the entire virtual table once
    per symbol row. The plan said it outright:

        SCAN f VIRTUAL TABLE INDEX 0: LEFT-JOIN

    Measured on a real 17k-symbol store, that produced 512 rows in 12s and the
    full set in ~300s, so workspace(open) on a cold repository blew past a 600s
    client timeout. Bodies must be read in one linear pass instead.
    """
    from icn import vectors

    record_baseline(workspace)

    seen: list[str] = []
    workspace.store.set_trace_callback(seen.append)
    try:
        list(vectors._symbol_texts(workspace.store))
    finally:
        workspace.store.set_trace_callback(None)

    offending = []
    for sql in seen:
        if "fts_symbols" not in sql or not sql.strip().upper().startswith("SELECT"):
            continue
        plan = " | ".join(str(tuple(r)) for r in workspace.store.execute(
            "EXPLAIN QUERY PLAN " + sql))
        if "VIRTUAL TABLE" in plan and "JOIN" in plan:
            offending.append(plan)

    assert seen, "no SQL was captured; the trace callback did not fire"
    assert not offending, f"fts_symbols is being joined, which is quadratic: {offending}"


def test_one_batch_cannot_outrun_the_first_index_budget(repo, tmp_path, monkeypatch):
    """The budget was consulted once every 40 files, so a batch ran unchecked.

    whatsapp-ghost holds 25,641 symbols in 37 files: its first batch was the
    whole repository, the 8s budget was never read in time, and the cold open
    took 275s. The clock now gets read per file.
    """
    for n in range(60):
        repo.write(f"mod{n}.py", f"""
def f{n}():
    return {n}
""")
    commit = repo.commit("many files")

    real = Indexer.read_and_parse

    def slow(self, abs_path):
        time.sleep(0.02)
        return real(self, abs_path)

    monkeypatch.setattr(Indexer, "read_and_parse", slow)

    conn = init_repo_store(tmp_path / "store.db")
    try:
        started = time.time()
        report = Indexer(conn, repo.root, "repo_budget").full_index(
            commit, budget_seconds=0.2)
        elapsed = time.time() - started
    finally:
        conn.close()

    assert report["truncated"], "a 0.2s budget over 1.2s of parsing must truncate"
    assert report["files_indexed"] < 40, "the budget must bite inside the first batch"
    assert elapsed < 0.6, f"budget overrun: {elapsed:.2f}s against a 0.2s budget"


def test_a_truncated_first_index_defers_whole_repository_resolution(repo, tmp_path,
                                                                    monkeypatch):
    """Import and call resolution ran unbudgeted even on a truncated index.

    Both are whole-repository passes, so on a big repository they cost more
    than the walk that preceded them - and every edge they wrote was about to
    be recomputed from nothing by the background full index.
    """
    repo.write("a.py", """
from b import b


def a():
    return b()
""")
    repo.write("b.py", """
def b():
    return 1
""")
    commit = repo.commit("two files")

    seen: list[object] = []
    real = Indexer.resolve_calls

    def spy(self, symbol_ids, commit, **kwargs):
        seen.append(symbol_ids)
        return real(self, symbol_ids, commit, **kwargs)

    monkeypatch.setattr(Indexer, "resolve_calls", spy)

    conn = init_repo_store(tmp_path / "store.db")
    try:
        report = Indexer(conn, repo.root, "repo_deferred").full_index(
            commit, budget_seconds=0.0)
        edges = one(conn.execute(
            "SELECT COUNT(*) AS n FROM code_edges WHERE kind='CALLS'"))
    finally:
        conn.close()

    assert report["truncated"], "budget_seconds=0 means no time at all, not no budget"
    assert not seen, "a truncated index must not resolve the whole repository"
    assert report["call_edges"] == 0
    assert report["import_edges"]["deferred"] == "index truncated"
    assert report["entry_points"]["deferred"] == "index truncated"
    assert report["areas"]["deferred"] == "index truncated"
    assert edges["n"] == 0


def test_the_resolution_passes_stop_at_their_deadline(project, tmp_path):
    """Resolution is resumable, so it is allowed to stop: it rewrites one
    file's or one symbol's edges wholesale inside a transaction, never half."""
    conn = init_repo_store(tmp_path / "store.db")
    try:
        idx = Indexer(conn, project.root, "repo_deadline")
        report = idx.full_index(project.head())
        assert not report["truncated"] and report["call_edges"] > 0

        past = time.time() - 1
        assert idx.resolve_calls(None, project.head(), deadline=past) == 0
        assert idx.resolution_truncated

        stopped = import_mod.resolve_file_imports(conn, None, project.head(),
                                                  deadline=past)
        assert stopped["truncated"] and stopped["files"] == 0

        # And the edges the completed run wrote are still there: stopping
        # early resolves fewer symbols, it does not delete what was resolved.
        edges = one(conn.execute(
            "SELECT COUNT(*) AS n FROM code_edges WHERE kind='CALLS'"))
        assert edges["n"] > 0
    finally:
        conn.close()


def test_a_partial_index_is_never_stamped_as_complete(project, monkeypatch):
    """Only a complete run stamps last_indexed_commit, and the run that was
    meant to complete it lives in a daemon thread that dies with the process.

    The next open then saw a non-empty store, asked git what had changed since
    "no commit at all", was told "nothing uncommitted", and stamped the
    commit - freezing a symbol table with no call edges in place as if it were
    whole. An unfinished index has to resume instead.
    """
    ws = ws_mod.open_workspace(str(project.root))
    try:
        monkeypatch.setattr(ws_mod, "_background_finish", lambda *a, **k: None)
        real = Indexer.read_and_parse

        def slow(self, abs_path):
            time.sleep(0.06)
            return real(self, abs_path)

        monkeypatch.setattr(Indexer, "read_and_parse", slow)
        first = ws_mod.ensure_indexed(ws, budget=0.05)
        monkeypatch.setattr(Indexer, "read_and_parse", real)
        ws_mod._BACKGROUND.pop(ws.repo_id, None)

        assert first["truncated"] and first["index_state"] == "partial"
        assert first["files_indexed"] >= 1, "the first file must still be stored"

        second = ws_mod.ensure_indexed(ws)
        assert second["index_state"] == "ready"
        assert second["call_edges"] > 0, "the resumed run must resolve calls"
        stamped = one(ws.catalog.execute(
            "SELECT last_indexed_commit FROM repositories WHERE repo_id=?",
            (ws.repo_id,)))
        assert stamped["last_indexed_commit"] == ws.commit
    finally:
        ws.close()


def test_a_budgeted_pass_hands_oversized_files_to_the_background_run(repo, tmp_path):
    """One file can outlast the whole budget, however often the clock is read.

    The budget can only be checked between files, and whatsapp-ghost's minified
    vendor bundles take 8-12s of tree-sitter each against an 8s budget. A
    budgeted pass therefore leaves them to the unbudgeted run, which is not a
    skip: the run reports truncated, so the background index parses them in
    full and the finished index is the same either way.
    """
    from icn.indexer import BUDGETED_FILE_BYTES

    repo.write("small.py", """
def small():
    return 1
""")
    padding = "# pad" + chr(10)
    repo.write("huge.py", "x = 1" + chr(10) + padding * (BUDGETED_FILE_BYTES // 6))
    commit = repo.commit("one big file")
    assert (repo.root / "huge.py").stat().st_size > BUDGETED_FILE_BYTES

    def index(budget):
        conn = init_repo_store(tmp_path / f"store-{budget}.db")
        try:
            report = Indexer(conn, repo.root, f"repo_{budget}").full_index(
                commit, budget_seconds=budget)
            paths_indexed = [r["path"] for r in rows(conn.execute(
                "SELECT path FROM files WHERE status='ACTIVE'"))]
        finally:
            conn.close()
        return report, paths_indexed

    budgeted, indexed = index(30.0)
    assert budgeted["deferred"] == 1
    assert budgeted["truncated"], "a deferred file must leave the run unfinished"
    assert "small.py" in indexed and "huge.py" not in indexed

    unbudgeted, indexed = index(None)
    assert unbudgeted["deferred"] == 0 and not unbudgeted["truncated"]
    assert "huge.py" in indexed, "the unbudgeted run indexes everything"
