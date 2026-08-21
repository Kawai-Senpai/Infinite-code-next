"""Causal chains between memories.

PLAN.md "Memory-to-memory relations are extremely useful". The target output,
verbatim from the plan:

    RefreshCoordinator exists because:
      Decision #44 -> production bug #91 -> failed fix #96
                   -> accepted fix #103 -> invariant #117
    Removing it may reintroduce: concurrent token invalidation
"""

from __future__ import annotations

import pytest

from icn import causal, compiler
from icn.db import one, rows


def rec(ws, **payload):
    return compiler.record_event(ws.store, ws.catalog, ws.repo_id, ws.root, ws.commit, payload)


def memory_of(result, kind):
    """The memory of this kind that the event's causal edge attached to.

    An event can produce several memories of one kind (a summary plus each
    listed fact), so "the first one of this kind" is not necessarily the one a
    causal edge names. record() reports which is primary; prefer it.
    """
    primary = result.get("primary_memory")
    for entry in result["memories_created"]:
        if entry["kind"] == kind and entry["memory_id"] == primary:
            return entry["memory_id"]
    for entry in result["memories_created"]:
        if entry["kind"] == kind:
            return entry["memory_id"]
    raise AssertionError(f"no {kind} memory in {result['memories_created']}")


@pytest.fixture
def story(workspace):
    """The plan's scenario, built through the public record() path."""
    decision = rec(workspace, kind="decision", summary="Use refresh-token rotation",
                   decisions=["Rotate the refresh token on every use"],
                   symbols=["refresh_session"])
    d_id = memory_of(decision, "decision")

    bug = rec(workspace, kind="incident",
              summary="Parallel refresh requests invalidate each other",
              bugs=["Concurrent refreshes rotate the same token and log users out"],
              symbols=["refresh_session"], caused_by=[d_id])
    b_id = memory_of(bug, "bug_history")

    failed = rec(workspace, kind="refactor", summary="Tried a Redis mutex, rejected",
                 failed_attempts=["Redis mutex could deadlock during a network partition"],
                 symbols=["refresh_session"],
                 caused_by=[{"memory": b_id, "kind": "LED_TO"}])
    f_id = memory_of(failed, "failed_attempt")

    fix = rec(workspace, kind="bug_fix", summary="Serialize refresh requests per session",
              invariants=["All refreshes for one session pass through RefreshCoordinator"],
              symbols=["RefreshCoordinator.acquire"],
              tests=["test_parallel_refresh_regression"],
              caused_by=[{"memory": f_id, "kind": "LED_TO"}])
    return {"decision": d_id, "bug": b_id, "failed": f_id,
            "invariant": memory_of(fix, "invariant"),
            "fix": fix["primary_memory"]}


def test_causal_edges_are_written(workspace, story):
    edges = rows(workspace.store.execute(
        "SELECT kind, edge_class FROM memory_edges WHERE kind IN"
        " ('CAUSED','LED_TO','ESTABLISHED','REFINES','DEPENDS_ON')"))
    assert edges, "expected causal edges"
    assert all(e["edge_class"] == "asserted" for e in edges), \
        "causality is asserted by an agent, never inferred from ordering"


def test_the_chain_reconstructs_the_whole_story(workspace, story):
    chain = causal.chain_for(workspace.store, story["invariant"])
    assert chain["ok"]

    titles = [c["title"] for c in chain["caused_by"]]
    assert any("Use refresh-token rotation" in t for t in titles)
    assert any("Parallel refresh requests" in t for t in titles)
    assert any("Redis mutex" in t for t in titles)


def test_each_node_reports_its_own_kind_not_its_edge_kind(workspace, story):
    """e.kind aliases over m.kind in the join; conflating them labelled every
    node with its neighbour's relation."""
    chain = causal.chain_for(workspace.store, story["invariant"])
    kinds = {c["kind"] for c in chain["caused_by"]}
    relations = {c["relation"] for c in chain["caused_by"]}

    assert kinds & {"decision", "bug_history", "failed_attempt"}
    assert relations <= set(causal.CAUSAL_KINDS)
    assert not (kinds & set(causal.CAUSAL_KINDS)), \
        "a memory's kind must never be an edge kind"


def test_why_does_this_exist_names_the_bug_not_the_fix(workspace, story):
    """The fix is itself stored as bug_history, so a naive scan of the chain
    reports the remedy as the risk. Only upstream failures count."""
    result = causal.why_does_this_exist(
        workspace.store,
        one(workspace.store.execute(
            "SELECT symbol_id FROM symbols WHERE symbol_path='RefreshCoordinator.acquire'"
        ))["symbol_id"])

    assert result is not None
    assert any("Parallel refresh requests" in r for r in result["may_reintroduce"])
    assert not any("Serialize refresh requests" in r for r in result["may_reintroduce"]), \
        "the fix must not be reported as the risk it removed"
    assert any("parallel_refresh" in t for t in result["regression_tests"])


def test_render_labels_the_arrows_with_the_relation(workspace, story):
    result = causal.why_does_this_exist(
        workspace.store,
        one(workspace.store.execute(
            "SELECT symbol_id FROM symbols WHERE symbol_path='RefreshCoordinator.acquire'"
        ))["symbol_id"])
    text = causal.render(result)

    assert "decision: Use refresh-token rotation" in text
    assert "-->" in text
    assert any(phrase in text for phrase in causal.CAUSAL_KINDS.values())


def test_a_symbol_with_no_causal_history_returns_none(workspace):
    """A single memory is a fact, not a story. Do not dress it up as one."""
    rec(workspace, kind="note", summary="plain note",
        warnings=["be careful"], symbols=["rotate_token"])
    symbol = one(workspace.store.execute(
        "SELECT symbol_id FROM symbols WHERE symbol_path='rotate_token'"))
    assert causal.why_does_this_exist(workspace.store, symbol["symbol_id"]) is None


def test_cycles_do_not_hang_the_traversal(workspace, story):
    """Knowledge graphs accrete loops; a naive walk would never terminate."""
    causal.link_causal(workspace.store, story["invariant"], story["decision"], "LED_TO")
    workspace.store.commit()

    chain = causal.chain_for(workspace.store, story["decision"])
    assert chain["ok"]
    ids = [c["memory_id"] for c in chain["caused_by"]]
    assert len(ids) == len(set(ids)), "a cycle must not revisit a node"


def test_invalid_causal_assertions_are_refused(workspace, story):
    assert causal.link_causal(workspace.store, story["bug"], story["bug"], "CAUSED")["ok"] is False
    assert causal.link_causal(workspace.store, story["bug"], story["decision"],
                              "NONSENSE")["ok"] is False
    assert causal.link_causal(workspace.store, "mem_missing", story["decision"],
                              "CAUSED")["ok"] is False


def test_causal_history_appears_in_investigate_capsules(workspace, story):
    from icn import search as search_mod

    result = search_mod.investigate(workspace.store, workspace.catalog, workspace.root,
                                    "RefreshCoordinator acquire", commit=workspace.commit)
    capsule = next((c for c in result["capsules"]
                    if c["symbol"] == "RefreshCoordinator.acquire"), None)
    assert capsule is not None
    assert capsule["why_it_exists"] is not None
    assert capsule["why_it_exists"]["may_reintroduce"]


def test_a_causal_edge_names_one_memory_not_all_of_them(workspace):
    """Fanning out to every memory an event produced invents causality.

    Observed live producing "warm-open bug CAUSED MCP annotation invariant",
    which is simply false. A wrong causal chain is worse than none, because it
    reads as authoritative.
    """
    first = rec(workspace, kind="decision", summary="an unrelated earlier decision",
                decisions=["do the earlier thing"], symbols=["rotate_token"])
    prior = first["primary_memory"]

    later = rec(workspace, kind="bug_fix", summary="a fix with several facts",
                invariants=["some invariant"], warnings=["some warning"],
                tests=["test_something"], symbols=["refresh_session"],
                caused_by=[prior])

    from_prior = rows(workspace.store.execute(
        "SELECT to_id FROM memory_edges WHERE from_id=? AND kind='CAUSED'"
        " AND status='ACTIVE'", (prior,)))
    assert len(from_prior) == 1, \
        f"one caused_by must create one causal edge, got {len(from_prior)}"
    assert from_prior[0]["to_id"] == later["primary_memory"]


def test_record_reports_which_memory_is_primary(workspace):
    """Without it a caller passing caused_by next has to guess from an
    unordered list, and naming the wrong memory builds a false chain."""
    result = rec(workspace, kind="bug_fix", summary="a fix",
                 invariants=["inv"], tests=["test_x"], symbols=["refresh_session"])

    primary = result["primary_memory"]
    assert primary, "record must say which memory represents the event"
    flagged = [m for m in result["memories_created"] if m.get("primary")]
    assert len(flagged) == 1 and flagged[0]["memory_id"] == primary


def test_a_fix_is_not_reported_as_the_risk_it_removed(workspace):
    """bug_fix and incident both mapping to bug_history made a remedy
    indistinguishable from a failure, so "why does this exist" reported the
    fix as something removing the code might reintroduce."""
    incident = rec(workspace, kind="incident", summary="the outage",
                   bugs=["tokens were invalidated concurrently"],
                   symbols=["refresh_session"])
    fix = rec(workspace, kind="bug_fix", summary="serialize the refreshes",
              invariants=["refreshes are serialized"],
              symbols=["RefreshCoordinator.acquire"],
              caused_by=[{"memory": incident["primary_memory"], "kind": "LED_TO"}])

    kinds = {m["kind"] for m in fix["memories_created"]}
    assert "fix_history" in kinds, "a fix must not be stored as bug history"
    assert "bug_history" not in kinds

    story = causal.why_does_this_exist(
        workspace.store,
        one(workspace.store.execute(
            "SELECT symbol_id FROM symbols WHERE symbol_path='RefreshCoordinator.acquire'"
        ))["symbol_id"])
    assert story is not None
    # The incident is the risk; the fix that removed it is not.
    assert any("outage" in r for r in story["may_reintroduce"])
    assert not any("serialize" in r.lower() for r in story["may_reintroduce"]),         "the remedy must never be reported as the risk"
