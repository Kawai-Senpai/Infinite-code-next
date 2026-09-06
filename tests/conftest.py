"""Shared fixtures.

Every test gets an isolated INFINITE_CODE_HOME so nothing touches the real
store, and a real git repository on disk - the anchoring cascade consults git
directly, so a fake would test the wrong thing.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# Semantic search is on by default for users, but a test suite must not
# download a model or spend seconds encoding every throwaway repository. Tests
# that exercise embeddings opt in explicitly by setting this themselves.
os.environ.setdefault("ICN_EMBED_MODEL", "none")


def git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True)


def rmtree_force(path: Path) -> None:
    """rmtree that survives Windows.

    Git writes its object files read-only, and on Windows that makes unlink
    fail outright rather than deferring to directory permissions.
    """
    import shutil
    import stat

    def on_error(func, target, _exc):
        try:
            os.chmod(target, stat.S_IWRITE)
            func(target)
        except OSError:
            pass

    shutil.rmtree(path, onerror=on_error)


class Repo:
    """A throwaway git repository with helpers for building history."""

    def __init__(self, root: Path):
        self.root = root

    def write(self, rel: str, content: str) -> Path:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def read(self, rel: str) -> str:
        return (self.root / rel).read_text(encoding="utf-8")

    def delete(self, rel: str) -> None:
        (self.root / rel).unlink()

    def commit(self, message: str = "wip") -> str:
        git(self.root, "add", "-A")
        git(self.root, "commit", "-qm", message)
        return git(self.root, "rev-parse", "HEAD").stdout.strip()

    def head(self) -> str:
        return git(self.root, "rev-parse", "HEAD").stdout.strip()


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    """Point the central store at a temp directory for the whole test."""
    home = tmp_path / "icn-home"
    monkeypatch.setenv("INFINITE_CODE_HOME", str(home))
    monkeypatch.delenv("INFINITE_CODE_ROOT", raising=False)
    return home


@pytest.fixture
def repo(tmp_path) -> Repo:
    root = tmp_path / "project"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "config", "user.name", "Test")
    git(root, "config", "commit.gpgsign", "false")
    return Repo(root)


AUTH_SOURCE = '''
class RefreshCoordinator:
    """Serializes refresh operations per session."""

    def acquire(self, session_id):
        lock = self._lock_for(session_id)
        return lock

    def _lock_for(self, session_id):
        return session_id


def refresh_session(session_id):
    coordinator = RefreshCoordinator()
    lock = coordinator.acquire(session_id)
    return rotate_token(lock)


def rotate_token(lock):
    return lock
'''

API_SOURCE = '''
from auth import refresh_session


def post_refresh(request):
    return refresh_session(request["session"])
'''

TEST_SOURCE = '''
from auth import refresh_session


def test_parallel_refresh_regression():
    assert refresh_session("s1") == "s1"
'''


@pytest.fixture
def project(repo: Repo) -> Repo:
    """A small but realistic repository: code, a caller, and a test."""
    repo.write("auth.py", AUTH_SOURCE)
    repo.write("api.py", API_SOURCE)
    repo.write("tests/test_auth.py", TEST_SOURCE)
    repo.commit("initial")
    return repo


@pytest.fixture
def workspace(project: Repo):
    """An opened, fully indexed workspace over `project`."""
    from icn import workspace as ws_mod

    ws = ws_mod.open_workspace(str(project.root))
    ws_mod.ensure_indexed(ws)
    yield ws
    ws.close()


def record_baseline(ws) -> dict:
    """Record the canonical event the anchoring tests build on."""
    from icn import compiler

    return compiler.record_event(
        ws.store, ws.catalog, ws.repo_id, ws.root, ws.commit,
        {
            "kind": "bug_fix",
            "summary": "Fixed concurrent refresh-token invalidation",
            "reasoning": "Parallel requests could rotate the same refresh token.",
            "invariants": ["Only one refresh operation per session may execute at once"],
            "warnings": ["Do not bypass RefreshCoordinator for new refresh entry points"],
            "failed_attempts": ["Redis mutex could deadlock during network failure"],
            "symbols": ["RefreshCoordinator.acquire"],
        },
    )


def anchor_rows(ws, memory_id: str | None = None) -> list[dict]:
    from icn.db import rows

    if memory_id:
        return rows(ws.store.execute("SELECT * FROM anchors WHERE memory_id=?", (memory_id,)))
    return rows(ws.store.execute("SELECT * FROM anchors"))


def acquire_anchor(ws) -> dict:
    """The anchor pointing at RefreshCoordinator.acquire."""
    from icn.db import rows

    found = rows(ws.store.execute(
        "SELECT * FROM anchors WHERE symbol_path LIKE 'RefreshCoordinator.acquire%' LIMIT 1"
    ))
    assert found, "expected an anchor on RefreshCoordinator.acquire"
    return found[0]
