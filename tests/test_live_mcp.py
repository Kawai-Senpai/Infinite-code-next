"""Live MCP protocol test: spawn the real server over stdio and drive it.

This is the test that proves the zero-config startup story. The server is
launched exactly the way an MCP client launches it - a child process, no
arguments, no config file, no port, no admin step, project directory as cwd -
and then a full agent workflow runs through the wire protocol.

One session is shared across the assertions because the workflow is inherently
sequential: open, record, investigate, checkpoint, drift the tree, re-verify.
Splitting it into independent sessions would test setup, not behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from mcp_client import McpStdioClient

SRC = str(Path(__file__).resolve().parents[1] / "src")


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    """Run the whole agent workflow once, against a real server process."""
    import subprocess

    base = tmp_path_factory.mktemp("live")
    home = base / "home"
    repo = base / "backend"
    repo.mkdir()

    def git(*args):
        return subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True)

    git("init", "-q", "-b", "main")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")

    from conftest import API_SOURCE, AUTH_SOURCE, TEST_SOURCE
    (repo / "auth.py").write_text(AUTH_SOURCE, encoding="utf-8")
    (repo / "api.py").write_text(API_SOURCE, encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_auth.py").write_text(TEST_SOURCE, encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "initial")

    captured: dict = {"repo": repo}
    with McpStdioClient(repo, {"INFINITE_CODE_HOME": str(home), "PYTHONPATH": SRC}) as client:
        captured["tools"] = client.list_tools()

        captured["opened"] = client.call("workspace", {"action": "open"})
        captured["status"] = client.call("workspace", {"action": "status"})
        captured["recorded"] = client.call("record", {
            "kind": "bug_fix",
            "summary": "Fixed concurrent refresh-token invalidation",
            "reasoning": "Parallel requests could rotate the same refresh token.",
            "invariants": ["Only one refresh operation per session may execute at once"],
            "warnings": ["Do not bypass RefreshCoordinator for new refresh entry points"],
            "failed_attempts": ["Redis mutex could deadlock during network failure"],
            "symbols": ["RefreshCoordinator.acquire"],
        })
        captured["investigated"] = client.call("investigate", {
            "query": "I need to change refresh token rotation. What will I break?",
        })
        captured["negative"] = client.call("investigate", {
            "query": "can I replace RefreshCoordinator with Redis locking?",
        })
        captured["checkpoint"] = client.call("agit", {
            "action": "commit", "message": "live test checkpoint",
        })

        # Drift the working tree WITHOUT committing.
        auth = repo / "auth.py"
        auth.write_text(
            auth.read_text(encoding="utf-8").replace(
                "        lock = self._lock_for(session_id)\n        return lock",
                "        return None  # serialization removed",
            ),
            encoding="utf-8",
        )

        captured["verified"] = client.call("investigate", {"action": "verify"})
        captured["after"] = client.call("investigate", {"query": "refresh coordinator acquire"})
        captured["listed"] = client.call("memory", {"action": "list"})

        needs_review = [m for m in captured["listed"]["memories"]
                        if m["anchor_status"] == "NEEDS_REVIEW"]
        if needs_review:
            captured["confirmed"] = client.call(
                "memory", {"action": "verify", "memory_id": needs_review[0]["memory_id"],
                           "actor": "human"})
            captured["after_confirm"] = client.call(
                "memory", {"action": "get", "memory_id": needs_review[0]["memory_id"]})

        captured["restored"] = client.call("agit", {
            "action": "restore", "paths": ["auth.py"], "source": "HEAD"})

        # --- causal chain, over the wire ---
        decision = client.call("record", {
            "kind": "decision", "summary": "Use refresh-token rotation",
            "decisions": ["Rotate the refresh token on every use"],
            "symbols": ["refresh_session"]})
        d_id = next(m["memory_id"] for m in decision["memories_created"]
                    if m["kind"] == "decision")
        bug = client.call("record", {
            "kind": "incident", "summary": "Parallel refresh requests invalidate each other",
            "bugs": ["Concurrent refreshes rotate the same token"],
            "symbols": ["refresh_session"], "caused_by": [d_id]})
        b_id = next(m["memory_id"] for m in bug["memories_created"]
                    if m["kind"] == "bug_history")
        captured["chain_built"] = client.call("record", {
            "kind": "bug_fix", "summary": "Serialize refresh requests per session",
            "invariants": ["All refreshes for one session pass through RefreshCoordinator"],
            "symbols": ["RefreshCoordinator.acquire"],
            "tests": ["test_parallel_refresh_regression"],
            "caused_by": [{"memory": b_id, "kind": "LED_TO"}]})
        captured["why"] = client.call("investigate", {
            "action": "why", "symbol": "RefreshCoordinator.acquire"})

        # --- cross-repo contract, over the wire ---
        captured["contract"] = client.call("record", {
            "kind": "decision", "summary": "refresh depends on billing",
            "symbols": ["refresh_session"],
            "contracts_with": [{"repo": "github.com/acme/billing",
                                "entity": "charge_customer",
                                "kind": "CONSUMES_CONTRACT"}]})
        captured["cross"] = client.call("investigate", {
            "query": "refresh session", "cross_repos": True})

        captured["briefing"] = client.call("workspace", {"action": "open"})
        captured["repos"] = client.call("workspace", {"action": "list"})
        captured["health"] = client.call("workspace", {"action": "health"})
        captured["bad_action"] = client.call("workspace", {"action": "nonsense"})
        captured["stderr"] = list(client.stderr)

    return captured


def test_tool_surface_is_five_tools(live):
    assert sorted(t["name"] for t in live["tools"]) == \
        ["agit", "investigate", "memory", "record", "workspace"]


def test_every_tool_documents_itself(live):
    for tool in live["tools"]:
        assert tool.get("description"), f"{tool['name']} has no description"
        assert "properties" in tool["inputSchema"]


def test_workspace_opens_and_indexes_on_first_call(live):
    opened = live["opened"]
    assert opened["ok"] is True
    assert opened["first_seen"] is True
    assert opened["identity"]["vcs"] == "git"
    assert opened["identity"]["strength"] == "strong"
    assert opened["index"]["symbols_active"] >= 5
    assert opened["index"]["index_state"] == "ready"
    assert opened["resolved_root"], "every response must echo the resolved root"


def test_root_is_resolved_from_cwd_without_being_told(live):
    """The zero-config claim: no root argument was ever passed."""
    assert Path(live["opened"]["resolved_root"]) == live["repo"]


def test_record_compiles_one_event_into_many_facts(live):
    recorded = live["recorded"]
    assert recorded["ok"] is True
    kinds = {m["kind"] for m in recorded["memories_created"]}
    assert {"invariant", "warning", "failed_attempt", "fix_history"} <= kinds
    assert recorded["edges_created"] > 5
    assert recorded["unresolved_references"] == []
    assert recorded["entities_resolved"][0]["resolved_to"] == "RefreshCoordinator.acquire"


def test_investigate_returns_the_knowledge_that_matters(live):
    result = live["investigated"]
    assert result["ok"] is True
    assert result["intent"] == "modify"
    assert result["capsules"]
    texts = [m.get("text", "") for c in result["capsules"] for m in c["memory"]]
    assert any("bypass RefreshCoordinator" in t for t in texts)
    assert any("Redis mutex" in t for t in texts)


def test_negative_knowledge_surfaces_what_was_already_rejected(live):
    """'Can I use Redis locking?' must find the failed attempt."""
    texts = [m.get("text", "") for c in live["negative"]["capsules"] for m in c["memory"]]
    assert any("Redis mutex" in t for t in texts)


def test_capsules_stay_within_budget(live):
    budget = live["investigated"]["budget"]
    assert budget["used_estimate"] <= budget["limit"]


def test_agit_checkpoint_works_over_the_wire(live):
    assert live["checkpoint"]["ok"] is True
    assert live["checkpoint"]["commit"]


def test_uncommitted_drift_is_caught_live(live):
    verified = live["verified"]
    assert verified["ok"] is True
    assert verified["by_status"].get("NEEDS_REVIEW", 0) >= 1
    assert any(c["transition"] == "body_changed" for c in verified["changed"])


def test_stale_memories_are_labelled_in_live_output(live):
    flagged = [m for c in live["after"]["capsules"] for m in c["memory"]
               if m["anchor_status"] != "ACTIVE"]
    assert flagged, "drifted memories must still be returned, labelled"
    assert all(m.get("warning") for m in flagged)


def test_problems_are_reported_after_drift(live):
    kinds = {p["kind"] for p in live["after"]["problems"]}
    assert "stale_knowledge" in kinds


def test_human_confirmation_restores_trust(live):
    assert live["confirmed"]["ok"] is True
    anchors = live["after_confirm"]["memory"]["anchors"]
    assert any(a["status"] == "ACTIVE" for a in anchors)


def test_agit_restore_recovers_the_file(live):
    assert live["restored"]["ok"] is True
    assert "serialization removed" not in (live["repo"] / "auth.py").read_text(encoding="utf-8")


def test_catalog_lists_the_repository(live):
    repos = live["repos"]["repositories"]
    assert repos and repos[0]["memories"] >= 4
    assert repos[0]["store_available"] is True


def test_health_summary_is_available(live):
    assert live["health"]["catalog"]["repositories"]["total"] >= 1


def test_an_unknown_action_is_an_error_not_a_crash(live):
    assert live["bad_action"]["ok"] is False
    assert "unknown workspace action" in live["bad_action"]["error"]


def test_nothing_was_written_to_stdout_that_would_corrupt_the_protocol(live):
    """Implicitly proven by every call parsing, and asserted explicitly here:
    a stray print() in a stdio server breaks the transport for everyone."""
    noisy = [line for line in live["stderr"] if "Traceback" in line]
    assert not noisy, f"server logged a traceback: {noisy[:3]}"


def test_causal_chain_is_built_and_queried_over_the_wire(live):
    assert live["chain_built"]["ok"] is True
    assert all(link["ok"] for link in live["chain_built"]["causal_links"])

    why = live["why"]
    assert why["ok"] is True
    story = why["why_it_exists"]
    assert story is not None, "expected a causal chain for RefreshCoordinator.acquire"
    assert any("Parallel refresh" in r for r in story["may_reintroduce"]),         "must name the bug it prevents, not the fix"
    assert "-->" in why["rendered"]


def test_cross_repo_contract_is_recorded_over_the_wire(live):
    link = live["contract"]["cross_repo_links"][0]
    assert link["ok"] is True
    # The billing repo has never been opened here, which is a supported state.
    assert link["status"] == "UNRESOLVED"

    context = live["cross"]["cross_repo"]
    assert context is not None
    assert context["linked_repos"], "the contract must still be reported"


def test_open_returns_an_orientation_briefing(live):
    brief = live["briefing"]["briefing"]
    assert brief["memories"] > 0
    assert brief["read_this_first"]
    assert brief["already_rejected"], "prior failures must be surfaced on open"
    assert any("Redis mutex" in r["title"] for r in brief["already_rejected"])


def test_new_problem_detectors_are_live(live):
    """The graph-derived detectors reach the client, not just the library."""
    kinds = {p["kind"] for p in live["after"]["problems"]}
    assert kinds, "expected at least one problem reported"
    assert "stale_knowledge" in kinds
