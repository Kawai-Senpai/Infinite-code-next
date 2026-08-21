"""Identity, rediscovery, lifecycle and the never-throwing resolver.

PLAN.md sections 1-3, 7, 13, 14, 16, 17. The rule under test throughout:
nothing in the knowledge graph may require the referenced filesystem object to
currently exist.
"""

from __future__ import annotations

import shutil

from conftest import git, rmtree_force

from icn import catalog as catalog_mod
from icn import paths, resolver
from icn import workspace as ws_mod
from icn.identity import normalize_remote, probe
from icn.db import one


def test_remote_forms_normalise_to_one_identity():
    assert normalize_remote("git@github.com:acme/backend.git") == "github.com/acme/backend"
    assert normalize_remote("https://github.com/acme/backend.git") == "github.com/acme/backend"
    assert normalize_remote("https://github.com/acme/backend/") == "github.com/acme/backend"
    assert normalize_remote("ssh://git@github.com/acme/backend.git") == "github.com/acme/backend"
    assert normalize_remote("") == ""


def test_root_commit_is_the_strong_identity(project):
    pr = probe(project.root)
    assert pr.is_git
    assert pr.root_commit
    assert pr.identity_strength == "strong"
    kinds = {kind for kind, _ in pr.aliases()}
    assert "root_commit" in kinds and "path" in kinds


def test_moving_a_clone_keeps_its_memories(project, tmp_path):
    """A moved checkout must reattach, not start a fresh memory universe."""
    first = ws_mod.open_workspace(str(project.root))
    original_repo_id = first.repo_id
    first.close()

    moved = tmp_path / "moved-elsewhere"
    shutil.move(str(project.root), str(moved))

    second = ws_mod.open_workspace(str(moved))
    try:
        assert second.repo_id == original_repo_id
        assert second.info["matched_on"] == "root_commit"
        assert second.info["created"] is False
    finally:
        second.close()


def test_changing_the_remote_url_does_not_fork_identity(project):
    git(project.root, "remote", "add", "origin", "git@github.com:acme/backend.git")
    first = ws_mod.open_workspace(str(project.root))
    repo_id = first.repo_id
    first.close()

    git(project.root, "remote", "set-url", "origin", "https://github.com/acme/backend-renamed.git")
    second = ws_mod.open_workspace(str(project.root))
    try:
        assert second.repo_id == repo_id
    finally:
        second.close()


def test_a_non_git_directory_is_supported_but_marked_weak(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "thing.py").write_text("def f():\n    return 1\n", encoding="utf-8")

    ws = ws_mod.open_workspace(str(plain))
    try:
        assert ws.info["identity_strength"] == "weak"
        assert ws.info["vcs"] == "none"
        report = ws_mod.ensure_indexed(ws)
        assert report["symbols_active"] >= 1
    finally:
        ws.close()


def test_vanished_checkout_becomes_missing_not_deleted(project, tmp_path):
    ws = ws_mod.open_workspace(str(project.root))
    repo_id = ws.repo_id
    ws.close()

    rmtree_force(project.root)
    catalog = catalog_mod.open_catalog()
    try:
        result = catalog_mod.reconcile(catalog)
        assert any(c.get("to") in ("MISSING", "OFFLINE") for c in result["changed"])
        repo = one(catalog.execute("SELECT * FROM repositories WHERE repo_id=?", (repo_id,)))
        assert repo["status"] in ("MISSING", "OFFLINE")
        # The knowledge is untouched: nothing was deleted.
        assert paths.repo_db_path(repo_id).exists()
    finally:
        catalog.close()


def test_archive_keeps_everything_searchable(project):
    ws = ws_mod.open_workspace(str(project.root))
    repo_id = ws.repo_id
    ws.close()

    catalog = catalog_mod.open_catalog()
    try:
        result = catalog_mod.set_repo_status(catalog, repo_id, catalog_mod.ARCHIVED, "test")
        assert result["ok"] and result["to"] == "ARCHIVED"
        assert paths.repo_db_path(repo_id).exists()
    finally:
        catalog.close()


def test_purge_is_the_only_destructive_action(project):
    ws = ws_mod.open_workspace(str(project.root))
    repo_id = ws.repo_id
    ws_mod.ensure_indexed(ws)
    ws.close()

    assert paths.repo_db_path(repo_id).exists()
    catalog = catalog_mod.open_catalog()
    try:
        result = catalog_mod.purge(catalog, repo_id)
        assert result["ok"]
        assert not paths.repo_db_path(repo_id).exists()
        repo = one(catalog.execute("SELECT status FROM repositories WHERE repo_id=?", (repo_id,)))
        assert repo["status"] == "PURGED"
    finally:
        catalog.close()


def test_resolver_never_raises_on_anything(project):
    """'Cannot currently resolve' is data, never an exception."""
    ws = ws_mod.open_workspace(str(project.root))
    catalog = ws.catalog
    try:
        assert resolver.resolve(catalog, ws.store, "sym_doesnotexist")["status"] == "UNKNOWN"
        assert resolver.resolve(catalog, None, "mem_nope")["status"] == "UNKNOWN"
        assert resolver.resolve(catalog, ws.store, "")["status"] == "UNKNOWN"
        assert resolver.resolve(catalog, ws.store, "not-an-id-at-all")["status"] == "UNKNOWN"
    finally:
        ws.close()


def test_resolver_reports_details_unavailable_when_store_is_gone(project):
    """A stub outlives its repo store, so the edge stays explainable."""
    ws = ws_mod.open_workspace(str(project.root))
    ws_mod.ensure_indexed(ws)
    repo_id = ws.repo_id

    from conftest import record_baseline
    result = record_baseline(ws)
    memory_id = result["memories_created"][0]["memory_id"]
    ws.close()

    rmtree_force(paths.repo_dir(repo_id))
    catalog = catalog_mod.open_catalog()
    try:
        resolved = resolver.resolve(catalog, None, memory_id)
        assert resolved["status"] in ("DETAILS_UNAVAILABLE", "TARGET_MISSING")
        assert resolved["last_known"]["title"]
    finally:
        catalog.close()


def test_worktrees_share_one_repository(project, tmp_path):
    """Three worktrees must not become three memory universes."""
    linked = tmp_path / "wt-feature"
    result = git(project.root, "worktree", "add", "-b", "feature", str(linked))
    if result.returncode != 0:
        import pytest
        pytest.skip(f"git worktree unavailable: {result.stderr}")

    main_ws = ws_mod.open_workspace(str(project.root))
    linked_ws = ws_mod.open_workspace(str(linked))
    try:
        assert linked_ws.repo_id == main_ws.repo_id
        assert linked_ws.checkout_id != main_ws.checkout_id
        assert linked_ws.info["is_worktree"] is True
    finally:
        main_ws.close()
        linked_ws.close()


def test_a_fork_does_not_silently_inherit_upstream_memories(project, tmp_path):
    """PLAN.md section 14. A fork shares upstream's root commit, which is our
    strongest identity signal, so without explicit handling it would be
    identified as the same repository and adopt its memories as native."""
    import shutil

    git(project.root, "remote", "add", "origin", "git@github.com:acme/backend.git")
    upstream = ws_mod.open_workspace(str(project.root))
    upstream_id = upstream.repo_id
    upstream.close()

    fork_path = tmp_path / "alice-backend"
    shutil.copytree(project.root, fork_path)
    git(fork_path, "remote", "set-url", "origin", "git@github.com:alice/backend.git")

    fork = ws_mod.open_workspace(str(fork_path))
    try:
        assert fork.repo_id != upstream_id, "a fork must not become the same repository"
        assert fork.info["forked_from"] == upstream_id
        catalog = fork.catalog
        link = one(catalog.execute(
            "SELECT * FROM repo_relationships WHERE from_repo=? AND kind='FORKED_FROM'",
            (fork.repo_id,)))
        assert link and link["to_repo"] == upstream_id
    finally:
        fork.close()
