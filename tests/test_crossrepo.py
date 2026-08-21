"""Cross-repository contract edges.

PLAN.md section 8. The governing rule under test: a cross-repo edge must never
depend on the other repository being available. It is written with a snapshot,
it survives that repository going missing, and it reports the breakage instead
of hiding it.
"""

from __future__ import annotations

from conftest import Repo, git, record_baseline, rmtree_force

from icn import catalog as catalog_mod
from icn import compiler, crossrepo
from icn import search as search_mod
from icn import workspace as ws_mod
from icn.db import one


def make_second_repo(tmp_path, name: str = "billing") -> Repo:
    """A second, independent repository to point contracts at."""
    root = tmp_path / name
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "config", "user.name", "Test")
    repo = Repo(root)
    repo.write("billing.py", "def charge_customer(account_id, cents):\n    return cents\n")
    repo.commit("initial")
    return repo


def test_repo_ref_resolves_by_id_remote_and_name(workspace, project):
    git(project.root, "remote", "add", "origin", "git@github.com:acme/backend.git")
    reopened = ws_mod.open_workspace(str(project.root))
    try:
        catalog = reopened.catalog
        by_id = crossrepo.resolve_repo_ref(catalog, reopened.repo_id)
        assert by_id and by_id["repo_id"] == reopened.repo_id

        # ssh and https forms of the same remote must both land on it.
        for form in ("git@github.com:acme/backend.git",
                     "https://github.com/acme/backend",
                     "github.com/acme/backend"):
            found = crossrepo.resolve_repo_ref(catalog, form)
            assert found and found["repo_id"] == reopened.repo_id, form

        assert crossrepo.resolve_repo_ref(catalog, "no-such-repo") is None
        assert crossrepo.resolve_repo_ref(catalog, "") is None
    finally:
        reopened.close()


def test_contract_links_two_known_repositories(workspace, project, tmp_path):
    """The normal case: both repos indexed, the target entity resolves."""
    billing = make_second_repo(tmp_path)
    billing_ws = ws_mod.open_workspace(str(billing.root))
    ws_mod.ensure_indexed(billing_ws)
    compiler.record_event(
        billing_ws.store, billing_ws.catalog, billing_ws.repo_id, billing_ws.root,
        billing_ws.commit,
        {"kind": "decision", "summary": "charging is idempotent",
         "invariants": ["charge_customer must be idempotent"],
         "symbols": ["charge_customer"]},
    )
    billing_repo_id = billing_ws.repo_id
    billing_ws.close()

    record_baseline(workspace)
    result = compiler.record_event(
        workspace.store, workspace.catalog, workspace.repo_id, workspace.root, workspace.commit,
        {"kind": "decision", "summary": "refresh calls out to billing",
         "symbols": ["refresh_session"],
         "contracts_with": [{"repo": billing_repo_id, "entity": "charge_customer",
                             "kind": "CONSUMES_CONTRACT"}]},
    )

    links = result["cross_repo_links"]
    assert len(links) == 1
    assert links[0]["ok"] and links[0]["status"] == "ACTIVE"
    assert links[0]["to_repo"] == billing_repo_id
    assert links[0]["target"]["resolved"] is True


def test_an_unknown_repository_is_recorded_not_rejected(workspace):
    """Asserting a contract before the other repo is ever opened is normal."""
    record_baseline(workspace)
    result = compiler.record_event(
        workspace.store, workspace.catalog, workspace.repo_id, workspace.root, workspace.commit,
        {"kind": "decision", "summary": "depends on a repo nobody has opened",
         "symbols": ["refresh_session"],
         "contracts_with": [{"repo": "github.com/acme/never-seen", "entity": "do_thing",
                             "kind": "CONSUMES_CONTRACT"}]},
    )
    link = result["cross_repo_links"][0]
    assert link["ok"] is True
    assert link["status"] == "UNRESOLVED"
    assert "not known yet" in link["note"]


def test_pending_edges_heal_when_the_other_repo_is_opened(workspace, project, tmp_path):
    billing = make_second_repo(tmp_path)
    git(billing.root, "remote", "add", "origin", "git@github.com:acme/billing.git")

    record_baseline(workspace)
    compiler.record_event(
        workspace.store, workspace.catalog, workspace.repo_id, workspace.root, workspace.commit,
        {"kind": "decision", "summary": "will resolve later", "symbols": ["refresh_session"],
         "contracts_with": [{"repo": "github.com/acme/billing", "entity": "charge_customer",
                             "kind": "CONSUMES_CONTRACT"}]},
    )
    pending = one(workspace.catalog.execute(
        "SELECT COUNT(*) AS n FROM cross_repo_edges WHERE status='UNRESOLVED'"))
    assert pending["n"] == 1

    # Opening billing registers its aliases and entity stubs; the edge heals.
    billing_ws = ws_mod.open_workspace(str(billing.root))
    ws_mod.ensure_indexed(billing_ws)
    compiler.record_event(
        billing_ws.store, billing_ws.catalog, billing_ws.repo_id, billing_ws.root,
        billing_ws.commit,
        {"kind": "note", "summary": "billing entry point", "symbols": ["charge_customer"]},
    )
    billing_ws.close()

    healed = crossrepo.resolve_pending(workspace.catalog, billing_ws.repo_id)
    assert healed == 1
    edge = one(workspace.catalog.execute("SELECT * FROM cross_repo_edges LIMIT 1"))
    assert edge["status"] == "ACTIVE"


def test_a_contract_survives_the_other_repo_disappearing(workspace, project, tmp_path):
    """The whole point. The target repo goes away; the edge stays readable."""
    billing = make_second_repo(tmp_path)
    billing_ws = ws_mod.open_workspace(str(billing.root))
    ws_mod.ensure_indexed(billing_ws)
    compiler.record_event(
        billing_ws.store, billing_ws.catalog, billing_ws.repo_id, billing_ws.root,
        billing_ws.commit,
        {"kind": "note", "summary": "entry point", "symbols": ["charge_customer"]},
    )
    billing_id = billing_ws.repo_id
    billing_ws.close()

    record_baseline(workspace)
    compiler.record_event(
        workspace.store, workspace.catalog, workspace.repo_id, workspace.root, workspace.commit,
        {"kind": "decision", "summary": "refresh depends on billing",
         "symbols": ["refresh_session"],
         "contracts_with": [{"repo": billing_id, "entity": "charge_customer",
                             "kind": "CONSUMES_CONTRACT"}]},
    )

    # Billing vanishes from disk entirely.
    rmtree_force(billing.root)
    catalog_mod.reconcile(workspace.catalog)

    symbol = one(workspace.store.execute(
        "SELECT symbol_id FROM symbols WHERE symbol_path='refresh_session'"))
    edges = crossrepo.edges_for(workspace.catalog, [symbol["symbol_id"]])

    assert len(edges) == 1
    edge = edges[0]
    assert edge["reachable"] is False, "a missing repo must be reported unreachable"
    # And it is still a readable sentence, from the snapshot alone.
    assert edge["kind"] == "CONSUMES_CONTRACT"
    assert edge["target"]["name"] == "charge_customer"
    assert edge["target_repo"]["status"] in ("MISSING", "OFFLINE")


def test_investigate_reports_an_unverifiable_contract(workspace, project, tmp_path):
    billing = make_second_repo(tmp_path)
    billing_ws = ws_mod.open_workspace(str(billing.root))
    ws_mod.ensure_indexed(billing_ws)
    compiler.record_event(
        billing_ws.store, billing_ws.catalog, billing_ws.repo_id, billing_ws.root,
        billing_ws.commit,
        {"kind": "note", "summary": "entry point", "symbols": ["charge_customer"]},
    )
    billing_id = billing_ws.repo_id
    billing_ws.close()

    record_baseline(workspace)
    compiler.record_event(
        workspace.store, workspace.catalog, workspace.repo_id, workspace.root, workspace.commit,
        {"kind": "decision", "summary": "refresh depends on billing",
         "symbols": ["refresh_session"],
         "contracts_with": [{"repo": billing_id, "entity": "charge_customer",
                             "kind": "CONSUMES_CONTRACT"}]},
    )

    rmtree_force(billing.root)
    catalog_mod.reconcile(workspace.catalog)

    result = search_mod.investigate(workspace.store, workspace.catalog, workspace.root,
                                    "refresh session", commit=workspace.commit)
    kinds = {p["kind"] for p in result["problems"]}
    assert "unverifiable_contract" in kinds

    capsule = next(c for c in result["capsules"] if c["symbol"] == "refresh_session")
    assert capsule["cross_repo"]
    assert capsule["cross_repo"][0]["kind"] == "CONSUMES_CONTRACT"
    assert capsule["cross_repo"][0]["reachable"] is False


def test_an_unknown_contract_kind_is_refused(workspace):
    result = crossrepo.link(workspace.catalog, "sym_x", "some-repo", "thing", "NONSENSE")
    assert result["ok"] is False
    assert "unknown cross-repo kind" in result["error"]


def test_incomplete_contract_declarations_are_ignored_not_fatal(workspace):
    record_baseline(workspace)
    result = compiler.record_event(
        workspace.store, workspace.catalog, workspace.repo_id, workspace.root, workspace.commit,
        {"kind": "note", "summary": "sloppy", "symbols": ["refresh_session"],
         "contracts_with": [{"repo": "", "entity": ""}, "not-a-dict"]},
    )
    assert result["ok"] is True
    assert all(link["ok"] is False for link in result["cross_repo_links"])


def test_purge_redacts_contract_snapshots(workspace, project, tmp_path):
    """Purge is a privacy operation: the snapshot of purged code must go."""
    billing = make_second_repo(tmp_path)
    billing_ws = ws_mod.open_workspace(str(billing.root))
    ws_mod.ensure_indexed(billing_ws)
    compiler.record_event(
        billing_ws.store, billing_ws.catalog, billing_ws.repo_id, billing_ws.root,
        billing_ws.commit,
        {"kind": "note", "summary": "entry point", "symbols": ["charge_customer"]},
    )
    billing_id = billing_ws.repo_id
    billing_ws.close()

    record_baseline(workspace)
    compiler.record_event(
        workspace.store, workspace.catalog, workspace.repo_id, workspace.root, workspace.commit,
        {"kind": "decision", "summary": "depends on billing", "symbols": ["refresh_session"],
         "contracts_with": [{"repo": billing_id, "entity": "charge_customer",
                             "kind": "CONSUMES_CONTRACT"}]},
    )

    catalog_mod.purge(workspace.catalog, billing_id, redact_snapshots=True)
    edge = one(workspace.catalog.execute("SELECT * FROM cross_repo_edges LIMIT 1"))
    assert edge["status"] == "TARGET_PURGED"
    assert edge["target_snapshot"] is None


def test_cross_repos_traversal_surfaces_the_other_repos_warnings(workspace, project, tmp_path):
    """The payoff: the repo you depend on has an ACTIVE warning about it.

    Read from the central memory registry, never from the other repository's
    store - which may be missing.
    """
    billing = make_second_repo(tmp_path)
    billing_ws = ws_mod.open_workspace(str(billing.root))
    ws_mod.ensure_indexed(billing_ws)
    compiler.record_event(
        billing_ws.store, billing_ws.catalog, billing_ws.repo_id, billing_ws.root,
        billing_ws.commit,
        {"kind": "decision", "summary": "charging is idempotent",
         "invariants": ["charge_customer must be called at most once per invoice"],
         "symbols": ["charge_customer"]},
    )
    billing_id = billing_ws.repo_id
    billing_ws.close()

    record_baseline(workspace)
    compiler.record_event(
        workspace.store, workspace.catalog, workspace.repo_id, workspace.root, workspace.commit,
        {"kind": "decision", "summary": "refresh calls billing",
         "symbols": ["refresh_session"],
         "contracts_with": [{"repo": billing_id, "entity": "charge_customer",
                             "kind": "CONSUMES_CONTRACT"}]},
    )

    result = search_mod.investigate(
        workspace.store, workspace.catalog, workspace.root, "refresh session",
        commit=workspace.commit, cross_repos=True, repo_id=workspace.repo_id)

    context = result["cross_repo"]
    assert context is not None and context["linked_repos"]
    linked = context["linked_repos"][0]
    assert linked["repo_id"] == billing_id
    assert linked["reachable"] is True
    assert any("at most once per invoice" in (m["summary"] or "")
               for m in linked["memories"]), "the other repo's invariant must surface"


def test_cross_repos_is_off_by_default(workspace):
    record_baseline(workspace)
    result = search_mod.investigate(workspace.store, workspace.catalog, workspace.root,
                                    "refresh session", commit=workspace.commit)
    assert result["cross_repo"] is None


def test_traversal_still_answers_when_the_other_repo_is_gone(workspace, project, tmp_path):
    """The snapshot is what makes this work without the other store."""
    billing = make_second_repo(tmp_path)
    billing_ws = ws_mod.open_workspace(str(billing.root))
    ws_mod.ensure_indexed(billing_ws)
    compiler.record_event(
        billing_ws.store, billing_ws.catalog, billing_ws.repo_id, billing_ws.root,
        billing_ws.commit,
        {"kind": "note", "summary": "entry point", "symbols": ["charge_customer"]},
    )
    billing_id = billing_ws.repo_id
    billing_ws.close()

    record_baseline(workspace)
    compiler.record_event(
        workspace.store, workspace.catalog, workspace.repo_id, workspace.root, workspace.commit,
        {"kind": "decision", "summary": "refresh calls billing",
         "symbols": ["refresh_session"],
         "contracts_with": [{"repo": billing_id, "entity": "charge_customer",
                             "kind": "CONSUMES_CONTRACT"}]},
    )

    rmtree_force(billing.root)
    catalog_mod.reconcile(workspace.catalog)

    result = search_mod.investigate(
        workspace.store, workspace.catalog, workspace.root, "refresh session",
        commit=workspace.commit, cross_repos=True, repo_id=workspace.repo_id)

    context = result["cross_repo"]
    assert context["unreachable"], "a missing repo must be named, not omitted"
    linked = context["linked_repos"][0]
    assert linked["reachable"] is False
    assert linked["contracts"][0]["entity"] == "charge_customer", \
        "the contract stays readable from the snapshot alone"
