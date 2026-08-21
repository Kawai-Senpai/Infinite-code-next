"""The orientation briefing on workspace open.

Motivated by a measured failure rather than a hypothesis: every session
building this server began by reading files to re-derive knowledge that already
existed. `open` reported symbol counts, which tells an agent nothing about what
it is walking into. An agent cannot ask the right question before it knows what
is on the shelf.
"""

from __future__ import annotations

from conftest import record_baseline

from icn import briefing, compiler
from icn import workspace as ws_mod


def rec(ws, **payload):
    return compiler.record_event(ws.store, ws.catalog, ws.repo_id, ws.root, ws.commit, payload)


def test_an_empty_repository_says_so_and_invites_recording(workspace):
    result = briefing.build(workspace.store)
    assert result["memories"] == 0
    assert "no knowledge recorded" in result["note"]
    assert "did not work" in result["note"], "should point at the highest-value field"


def test_rules_are_surfaced_worst_severity_first(workspace):
    rec(workspace, kind="note", summary="minor", conventions=["prefer f-strings"],
        symbols=["rotate_token"])
    rec(workspace, kind="decision", summary="critical rule",
        security=["never log a refresh token"], symbols=["refresh_session"])

    result = briefing.build(workspace.store)
    assert result["rules"]
    assert result["rules"][0]["severity"] == "critical"
    assert "never log" in result["rules"][0]["title"]


def test_rejected_approaches_are_called_out_separately(workspace):
    """The field nothing else in a toolchain records."""
    record_baseline(workspace)
    result = briefing.build(workspace.store)

    assert result["already_rejected"]
    assert any("Redis mutex" in r["title"] for r in result["already_rejected"])
    assert "already been tried and rejected" in result["read_this_first"]


def test_unverified_memories_are_flagged_in_the_briefing(workspace, project):
    record_baseline(workspace)
    project.write("auth.py", project.read("auth.py").replace(
        "        lock = self._lock_for(session_id)\n        return lock",
        "        return None  # gutted"))
    ws_mod.ensure_indexed(workspace)

    result = briefing.build(workspace.store)
    assert result["stale_count"] >= 1
    assert result["needs_verification"]
    assert "unverified" in result["read_this_first"]


def test_hotspots_show_where_knowledge_is_concentrated(workspace):
    for index in range(3):
        rec(workspace, kind="decision", summary="rule " + str(index),
            invariants=["invariant number " + str(index)],
            symbols=["RefreshCoordinator.acquire"])

    result = briefing.build(workspace.store)
    assert result["knowledge_hotspots"]
    assert result["knowledge_hotspots"][0]["symbol_path"] == "RefreshCoordinator.acquire"
    assert result["knowledge_hotspots"][0]["memories"] >= 2


def test_causal_history_is_advertised_when_present(workspace):
    first = rec(workspace, kind="decision", summary="original decision",
                decisions=["do the thing"], symbols=["refresh_session"])
    rec(workspace, kind="bug_fix", summary="the fix",
        invariants=["the rule"], symbols=["RefreshCoordinator.acquire"],
        caused_by=[first["memories_created"][0]["memory_id"]])

    result = briefing.build(workspace.store)
    assert "action='why'" in result["read_this_first"]


def test_the_briefing_is_a_map_not_the_territory(workspace):
    """Headlines and ids only. Bodies stay out; investigate() is one call away."""
    record_baseline(workspace)
    result = briefing.build(workspace.store)

    for section in ("rules", "already_rejected", "needs_verification"):
        for entry in result[section]:
            assert "body" not in entry, f"{section} must not carry memory bodies"
            assert entry.get("memory_id"), f"{section} entries must be addressable"


def test_open_returns_the_briefing(workspace, project):
    from icn.server import workspace as workspace_tool

    record_baseline(workspace)
    result = workspace_tool(action="open", root=str(project.root))
    assert result["ok"]
    assert "briefing" in result
    assert result["briefing"]["read_this_first"]
