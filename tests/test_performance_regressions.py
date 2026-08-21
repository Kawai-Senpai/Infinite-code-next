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

    def spy(self, symbol_ids, commit):
        seen.append(symbol_ids)
        return real(self, symbol_ids, commit)

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
