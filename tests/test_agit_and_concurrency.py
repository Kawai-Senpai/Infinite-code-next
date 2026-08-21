"""agit checkpoints, and the multi-process concurrency model."""

from __future__ import annotations

import sqlite3
import threading

import pytest

from icn import agit
from icn import paths
from icn import workspace as ws_mod
from icn.db import init_repo_store, write_tx
from icn.lease import Lease


# ------------------------------------------------------------------------ agit

def test_first_use_initialises_and_gitignores_itself(project):
    result = agit.run(project.root, "status")
    assert result["ok"]
    assert (project.root / ".agit" / "HEAD").exists()
    assert ".agit/" in project.read(".gitignore")


def test_gitignore_entry_is_not_duplicated(project):
    agit.run(project.root, "status")
    agit.init(project.root)
    agit.init(project.root)
    assert project.read(".gitignore").count(".agit/") == 1


def test_checkpoint_and_restore_round_trip(project):
    agit.run(project.root, "commit", message="before risky edit")
    original = project.read("auth.py")

    project.write("auth.py", "# destroyed everything\n")
    assert project.read("auth.py") != original

    restored = agit.run(project.root, "restore", paths_arg=["auth.py"], source="HEAD")
    assert restored["ok"]
    assert project.read("auth.py") == original


def test_agit_history_is_separate_from_real_git(project):
    from conftest import git

    agit.run(project.root, "commit", message="agent checkpoint")
    real_log = git(project.root, "log", "--oneline").stdout
    agent_log = agit.run(project.root, "log", limit=5)

    assert "agent checkpoint" not in real_log
    assert any("agent checkpoint" == e["message"] for e in agent_log["entries"])


def test_paths_cannot_escape_the_work_tree(project):
    with pytest.raises(ValueError):
        agit.ensure_within("../../etc/passwd", project.root)
    with pytest.raises(ValueError):
        agit.ensure_within("C:/Windows/System32/config", project.root)

    refused = agit.run(project.root, "commit", message="x", paths_arg=["../outside.txt"])
    assert refused["ok"] is False
    assert "stay inside" in refused["error"]


def test_commit_with_nothing_staged_is_not_an_error(project):
    agit.run(project.root, "commit", message="first")
    second = agit.run(project.root, "commit", message="second")
    assert second["ok"]
    assert second.get("nothing_to_commit") or second.get("commit")


def test_unknown_action_is_reported(project):
    result = agit.run(project.root, "explode")
    assert result["ok"] is False
    assert "unknown agit action" in result["error"]


# --------------------------------------------------------------- concurrency

def test_only_one_process_holds_the_indexer_lease(tmp_path):
    path = tmp_path / "indexer.lock"
    first = Lease(path)
    second = Lease(path)

    assert first.try_acquire() is True
    assert second.try_acquire() is False, "a live lease must not be stolen"

    first.release()
    assert second.try_acquire() is True
    second.release()


def test_a_stale_lease_is_reclaimed(tmp_path, monkeypatch):
    """The lease holder may be SIGKILLed. Nobody runs cleanup, so a stale
    heartbeat has to be reclaimable or indexing wedges forever."""
    import json
    import os

    path = tmp_path / "indexer.lock"
    path.write_text(json.dumps({"pid": os.getpid() + 99999, "host": "somewhere-else",
                                "owner": "indexer", "heartbeat": 0.0}), encoding="utf-8")

    lease = Lease(path)
    assert lease.holder()["stale"] is True
    assert lease.try_acquire() is True
    lease.release()


def test_second_process_reads_while_first_indexes(project):
    """Concurrent opens must not corrupt or deadlock the store."""
    first = ws_mod.open_workspace(str(project.root))
    ws_mod.ensure_indexed(first)

    second = ws_mod.open_workspace(str(project.root))
    try:
        assert second.repo_id == first.repo_id
        report = ws_mod.ensure_indexed(second)
        # Either it indexed, or it politely stood down because of the lease.
        assert report["indexed"] in (True, False)
        rows = second.store.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
        assert rows > 0
    finally:
        first.close()
        second.close()


def test_concurrent_writes_do_not_lose_data(project):
    """WAL plus BEGIN IMMEDIATE with retry has to survive real contention."""
    ws = ws_mod.open_workspace(str(project.root))
    db_path = paths.repo_db_path(ws.repo_id)
    ws.close()

    errors: list[Exception] = []
    written: list[int] = []

    def writer(tag: int) -> None:
        conn = init_repo_store(db_path)
        try:
            for i in range(15):
                with write_tx(conn):
                    conn.execute(
                        "INSERT INTO events (event_id, kind, payload, created_at)"
                        " VALUES (?,?,?,datetime('now'))",
                        (f"evt_{tag}_{i}", "note", "{}"),
                    )
                written.append(1)
        except Exception as exc:  # noqa: BLE001 - the assertion is that this stays empty
            errors.append(exc)
        finally:
            conn.close()

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert errors == [], f"concurrent writes failed: {errors[:3]}"
    assert len(written) == 60

    conn = sqlite3.connect(str(db_path))
    try:
        count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert count == 60
    finally:
        conn.close()
