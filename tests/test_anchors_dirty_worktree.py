"""The dirty-worktree fixture harness.

PLAN 2 section 6 calls this non-optional, and it is the only validation that
will exist for this regime. Every re-anchoring result in the literature is
post-hoc mining of committed history; none of it measures behaviour on
uncommitted edits, which is exactly where an MCP server lives - the agent is
mid-edit and nothing has been committed yet.

So each test here edits the working tree WITHOUT committing, re-indexes, and
asserts the cascade's status output. The bias under test is as important as the
matching: an honest ORPHANED or NEEDS_REVIEW beats a confident wrong answer.
"""

from __future__ import annotations

from conftest import acquire_anchor, record_baseline

from icn import anchors as anchor_mod
from icn import workspace as ws_mod
from icn.db import one, rows


def reverify(ws):
    """Re-index the dirty tree, then run the cascade - the real call path."""
    return ws_mod.ensure_indexed(ws)["anchors"]


def test_reformatting_does_not_drift_an_anchor(workspace, project):
    """Whitespace, blank lines and comment rewrites must be invisible.

    If a formatter run marks every memory in the repo NEEDS_REVIEW, the status
    becomes noise and people learn to ignore it.
    """
    record_baseline(workspace)
    before = acquire_anchor(workspace)
    assert before["status"] == anchor_mod.ACTIVE

    source = project.read("auth.py")
    source = source.replace(
        "    def acquire(self, session_id):\n        lock = self._lock_for(session_id)",
        "    def acquire(self, session_id):\n\n        # totally rewritten comment\n"
        "        lock = self._lock_for(session_id)\n",
    )
    project.write("auth.py", source)

    result = reverify(workspace)
    after = acquire_anchor(workspace)
    assert after["status"] == anchor_mod.ACTIVE
    assert after["anchor_confidence"] == 1.0
    assert result["by_status"].get(anchor_mod.NEEDS_REVIEW, 0) == 0


def test_rename_reanchors_and_marks_drifted(workspace, project):
    """A renamed symbol keeps its memory, at reduced confidence."""
    record_baseline(workspace)
    project.write("auth.py", project.read("auth.py")
                  .replace("def acquire(self, session_id)", "def acquire_lock(self, session_id)")
                  .replace("coordinator.acquire(", "coordinator.acquire_lock("))

    reverify(workspace)
    after = acquire_anchor(workspace)

    assert after["symbol_path"] == "RefreshCoordinator.acquire_lock"
    assert after["status"] == anchor_mod.DRIFTED
    assert 0.7 <= after["anchor_confidence"] <= 0.9
    # The previous anchor is retained, so a bad re-anchor stays auditable.
    assert "RefreshCoordinator.acquire" in (after["reanchor_history"] or "")


def test_body_change_marks_needs_review_immediately(workspace, project):
    """The core claim: drift is flagged on the edit that caused it.

    Not by a nightly sweep, not by a decay timer. The measured harm from a
    stale note peaks right after it goes stale, so this must fire now.
    """
    record_baseline(workspace)
    project.write("auth.py", project.read("auth.py").replace(
        "    def acquire(self, session_id):\n        lock = self._lock_for(session_id)\n        return lock",
        "    def acquire(self, session_id):\n        return None  # serialization removed entirely",
    ))

    reverify(workspace)
    after = acquire_anchor(workspace)

    assert after["status"] == anchor_mod.NEEDS_REVIEW
    assert after["anchor_confidence"] < 0.7


def test_move_across_files_follows_the_code(workspace, project):
    """Identical code in a new file is a move, and the memory follows it."""
    record_baseline(workspace)

    source = project.read("auth.py")
    method = ("    def acquire(self, session_id):\n"
              "        lock = self._lock_for(session_id)\n"
              "        return lock\n")
    assert method in source
    project.write("auth.py", source.replace(method, ""))
    project.write("coordinator.py", "class RefreshCoordinator:\n" + method)

    reverify(workspace)
    after = acquire_anchor(workspace)

    assert after["status"] in (anchor_mod.ACTIVE, anchor_mod.DRIFTED)
    assert after["anchor_confidence"] >= 0.7
    assert "coordinator.py" in (after["file_path"] or "")


def test_deletion_orphans_rather_than_destroys(workspace, project):
    """A deleted symbol must never take its knowledge with it."""
    result = record_baseline(workspace)
    memory_id = result["memories_created"][0]["memory_id"]

    source = project.read("auth.py")
    start = source.index("class RefreshCoordinator")
    end = source.index("def refresh_session")
    project.write("auth.py", source[:start] + source[end:])
    project.write("api.py", "def post_refresh(request):\n    return None\n")
    project.write("tests/test_auth.py", "def test_nothing():\n    assert True\n")

    reverify(workspace)
    after = acquire_anchor(workspace)

    assert after["status"] == anchor_mod.ORPHANED
    # The memory itself survives, and so does the tombstone explaining why.
    memory = one(workspace.store.execute("SELECT * FROM memories WHERE memory_id=?", (memory_id,)))
    assert memory is not None and memory["status"] == "ACTIVE"
    tombstone = one(workspace.store.execute(
        "SELECT * FROM symbols WHERE symbol_path='RefreshCoordinator.acquire'"
    ))
    assert tombstone is not None and tombstone["status"] == "DELETED"


def test_weak_migration_is_recorded_but_not_acted_on(workspace, project):
    """PLAN.md section 12: an uncertain migration is a lead, never a fact."""
    record_baseline(workspace)

    source = project.read("auth.py")
    start = source.index("class RefreshCoordinator")
    end = source.index("def refresh_session")
    replacement = (
        "class ChargeProcessor:\n"
        "    def execute(self, invoice_id, retries):\n"
        "        total = 0\n"
        "        for attempt in range(retries):\n"
        "            total += self.charge(invoice_id, attempt)\n"
        "        return total\n\n"
        "    def charge(self, invoice_id, attempt):\n"
        "        return invoice_id\n\n\n"
    )
    project.write("auth.py", source[:start] + replacement + source[end:])
    project.write("api.py", "def post_refresh(request):\n    return None\n")
    project.write("tests/test_auth.py", "def test_nothing():\n    assert True\n")

    reverify(workspace)
    after = acquire_anchor(workspace)

    assert after["status"] == anchor_mod.ORPHANED, "a weak match must not move a memory"
    assert after["symbol_path"] == "RefreshCoordinator.acquire"


def test_verification_survives_repeated_runs(workspace, project):
    """Re-verifying an unchanged tree must be a no-op, not a slow decay."""
    record_baseline(workspace)
    first = reverify(workspace)
    second = reverify(workspace)

    assert first["by_status"].get(anchor_mod.ACTIVE) == second["by_status"].get(anchor_mod.ACTIVE)
    assert second["changed"] == []


def test_committing_the_edit_does_not_change_the_verdict(workspace, project):
    """The dirty-tree verdict and the committed verdict must agree.

    If committing flipped the status, the uncommitted path would be reporting
    something the committed path disagrees with, and neither could be trusted.
    """
    record_baseline(workspace)
    project.write("auth.py", project.read("auth.py")
                  .replace("def acquire(self, session_id)", "def acquire_lock(self, session_id)")
                  .replace("coordinator.acquire(", "coordinator.acquire_lock("))

    reverify(workspace)
    dirty_status = acquire_anchor(workspace)["status"]

    project.commit("rename acquire")
    workspace.commit = project.head()
    reverify(workspace)
    committed_status = acquire_anchor(workspace)["status"]

    assert dirty_status == committed_status == anchor_mod.DRIFTED


def test_verify_action_clears_review_flag(workspace, project):
    """An agent confirming a memory still applies restores ACTIVE."""
    result = record_baseline(workspace)
    memory_id = result["memories_created"][0]["memory_id"]

    project.write("auth.py", project.read("auth.py").replace(
        "        lock = self._lock_for(session_id)\n        return lock",
        "        return None  # changed",
    ))
    reverify(workspace)
    assert acquire_anchor(workspace)["status"] == anchor_mod.NEEDS_REVIEW

    anchor_mod.mark_verified(workspace.store, memory_id, workspace.commit, actor="human")
    refreshed = rows(workspace.store.execute(
        "SELECT * FROM anchors WHERE memory_id=? AND symbol_path LIKE 'RefreshCoordinator.acquire%'",
        (memory_id,),
    ))
    assert refreshed and refreshed[0]["status"] == anchor_mod.ACTIVE
