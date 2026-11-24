"""Memory durability and discoverability through the real entry points."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from threading import Barrier

from icn import catalog, compiler, paths, search, workspace as ws_mod
from icn.db import connect, init_repo_store
from mcp_client import McpStdioClient


def test_setting_wal_retries_when_another_client_is_initializing(tmp_path, monkeypatch):
    real_connect = sqlite3.connect
    attempts = 0

    class InitiallyLocked(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            nonlocal attempts
            if sql == "PRAGMA journal_mode = WAL":
                attempts += 1
                if attempts == 1:
                    raise sqlite3.OperationalError("database is locked")
            return super().execute(sql, parameters)

    monkeypatch.setattr(sqlite3, "connect",
                        lambda *args, **kwargs: real_connect(*args, factory=InitiallyLocked, **kwargs))
    with closing(connect(tmp_path / "startup.db")) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert attempts == 2


def test_first_opens_share_one_repo_identity(project, monkeypatch):
    before_lock = Barrier(2)
    transaction = catalog.write_tx

    @contextmanager
    def concurrent_start(conn):
        before_lock.wait(timeout=15)
        with transaction(conn):
            yield

    monkeypatch.setattr(catalog, "write_tx", concurrent_start)

    def open_one():
        with closing(catalog.open_catalog()) as conn:
            return catalog.open_workspace(conn, project.root)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = [pool.submit(open_one) for _ in range(2)]
        opened = [first.result(timeout=25), second.result(timeout=25)]
    assert len({entry["repo_id"] for entry in opened}) == 1
    assert len({entry["checkout_id"] for entry in opened}) == 1
    assert sum(entry["created"] for entry in opened) == 1


def test_concurrent_first_opens_share_repository_schema(tmp_path):
    store = tmp_path / "repo.db"
    start = Barrier(5)

    def open_one(_):
        start.wait(timeout=15)
        with closing(init_repo_store(store)) as conn:
            evidence_column = [row["name"] for row in conn.execute("PRAGMA table_info(memories)")]
            assert evidence_column.count("evidence_count") == 1
            return conn.execute("PRAGMA quick_check").fetchone()[0]

    with ThreadPoolExecutor(max_workers=5) as pool:
        assert list(pool.map(open_one, range(5))) == ["ok"] * 5


def test_one_event_reinforces_once_and_opposite_claim_is_preserved(workspace):
    def record(*claims):
        return compiler.record_event(
            workspace.store, workspace.catalog, workspace.repo_id, workspace.root, workspace.commit,
            {"kind": "note", "summary": "Policy review", "symbols": ["refresh_session"],
             "invariants": list(claims)},
        )

    rule = "Refresh sessions must allow concurrent token rotation"
    first = record(rule)["primary_memory"]
    repeated = record(rule, rule)
    assert repeated["memories_reinforced"][0]["evidence_count"] == 2
    assert len(repeated["memories_reinforced"]) == 1
    opposite = record("Refresh sessions must not allow concurrent token rotation")
    assert opposite["primary_memory"] != first
    assert opposite["contradictions"][0]["memory_id"] == first
    assert not opposite["memories_reinforced"]


def test_reopen_repairs_search_without_reinstating_retired_memory(workspace):
    db = workspace.store
    for memory_id, status in (("active", "ACTIVE"), ("legacy", "ACTIVE"),
                              ("forgotten", "FORGOTTEN"), ("superseded", "SUPERSEDED")):
        db.execute(
            "INSERT INTO memories(memory_id,kind,body,status,created_at)"
            " VALUES (?,'warning','example keyword',?,'today')", (memory_id, status),
        )
    db.executemany(
        "INSERT INTO fts_memories(memory_id,title,body,kind) VALUES (?,'','example keyword','warning')",
        [("active",), ("active",), ("forgotten",)],
    )
    assert set(search._seed_memories(db, ["keyword"])) == {"active"}
    db.execute("PRAGMA user_version=10")
    with closing(init_repo_store(paths.repo_db_path(workspace.repo_id))) as reopened:
        assert [row[0] for row in reopened.execute(
            "SELECT memory_id FROM fts_memories ORDER BY memory_id"
        )] == ["active", "legacy"]


def test_an_inflight_index_does_not_report_ready_or_keep_a_lost_worker(workspace, monkeypatch):
    lease = ws_mod._lease_for(workspace.repo_id)
    assert lease.try_acquire()
    try:
        report = ws_mod.ensure_indexed(workspace)
        assert report["files_active"] > 0
        assert report["index_state"] == "partial"
        monkeypatch.setitem(ws_mod._BACKGROUND, workspace.repo_id, None)
        ws_mod._background_finish(workspace.repo_id, workspace.root, workspace.commit, [])
        assert workspace.repo_id not in ws_mod._BACKGROUND
    finally:
        lease.release()


def test_record_restarts_search_and_backups_over_the_wire(project, tmp_path):
    source_dir = str(Path(__file__).resolve().parents[1] / "src")
    env = {"PYTHONPATH": source_dir, "ICN_EMBED_MODEL": "none"}
    summary = "Quartz harbors keep replay ledgers"
    reasoning = "Full stored context " + ("durable evidence " * 100).strip()
    with McpStdioClient(project.root, env) as client:
        result = client.call("record", {"summary": summary, "reasoning": reasoning})
        assert result["ok"] and result["primary_memory"]
        memory_id, event_id = result["primary_memory"], result["event_id"]
    with McpStdioClient(project.root, env) as restarted:
        found = restarted.call("memory", {"action": "get", "memory_id": memory_id})
        assert found["memory"]["source_event"] == event_id
        assert reasoning in found["memory"]["body"]
        result = restarted.call("investigate", {
            "query": "Quartz harbors replay ledgers", "conversations": False,
        })
        assert not result["capsules"]
        assert result["repo_memories"][0]["memory_id"] == memory_id

        output = tmp_path / "snapshot.sqlite"
        process = subprocess.run(
            [sys.executable, "-m", "icn", "backup", str(output), "--root", str(project.root)],
            capture_output=True, text=True, timeout=30,
        )
        assert process.returncode == 0, process.stderr
        assert json.loads(process.stdout)["memories"] == 1

    with closing(sqlite3.connect(output)) as snapshot:
        assert snapshot.execute("PRAGMA quick_check").fetchone() == ("ok",)
        assert snapshot.execute("SELECT event_id FROM events").fetchone() == (event_id,)
        assert snapshot.execute("SELECT body FROM memories").fetchone()[0] == found["memory"]["body"]
        assert snapshot.execute("SELECT memory_id FROM fts_memories WHERE fts_memories MATCH 'quartz'").fetchone() == (memory_id,)
    original = output.read_bytes()
    process = subprocess.run(
        [sys.executable, "-m", "icn", "backup", str(output), "--root", str(project.root)],
        capture_output=True, text=True, timeout=30,
    )
    assert process.returncode != 0
    assert output.read_bytes() == original
