"""The memory compiler: entity resolution, edge derivation, and correction rules."""

from __future__ import annotations

from conftest import record_baseline

from icn import compiler
from icn.db import one, rows


def test_one_event_becomes_many_facts(workspace):
    result = record_baseline(workspace)

    assert result["ok"]
    kinds = {m["kind"] for m in result["memories_created"]}
    assert {"invariant", "warning", "failed_attempt", "fix_history"} <= kinds
    assert result["edges_created"] > len(result["memories_created"])
    assert result["unresolved_references"] == []


def test_entity_resolution_handles_the_forms_agents_actually_type(workspace):
    cases = {
        "RefreshCoordinator.acquire": "RefreshCoordinator.acquire",
        "RefreshCoordinator": "RefreshCoordinator",
        "refresh_session": "refresh_session",
        "auth.py": "auth.py",
        "the refresh coordinator": "RefreshCoordinator",
    }
    for reference, expected in cases.items():
        match = compiler.resolve_reference(workspace.store, reference)
        assert match is not None, f"{reference!r} did not resolve"
        actual = match["row"].get("symbol_path") or match["row"].get("path")
        assert actual == expected, f"{reference!r} -> {actual}, expected {expected}"


def test_unknown_reference_is_reported_not_invented(workspace):
    assert compiler.resolve_reference(workspace.store, "TotallyMadeUpThing") is None

    result = compiler.record_event(
        workspace.store, workspace.catalog, workspace.repo_id, workspace.root, workspace.commit,
        {"kind": "note", "summary": "x", "warnings": ["careful"], "symbols": ["NoSuchSymbol"]},
    )
    assert result["unresolved_references"] == ["NoSuchSymbol"]
    # The memory is still kept, honestly scoped to the repo rather than a span.
    memory_id = result["memories_created"][0]["memory_id"]
    anchors = rows(workspace.store.execute("SELECT * FROM anchors WHERE memory_id=?", (memory_id,)))
    assert anchors and anchors[0]["target_kind"] == "repo"


def test_compiler_derives_edges_the_agent_never_mentioned(workspace):
    """One named symbol should pull in its callers and tests automatically."""
    result = record_baseline(workspace)
    memory_id = next(m["memory_id"] for m in result["memories_created"] if m["kind"] == "warning")

    edges = rows(workspace.store.execute(
        "SELECT kind, to_id FROM memory_edges WHERE from_id=?", (memory_id,)))
    kinds = {e["kind"] for e in edges}
    assert "APPLIES_TO" in kinds
    assert "IMPACTS" in kinds or "GUARDED_BY" in kinds
    assert "DERIVED_FROM" in kinds


def test_event_payload_is_stored_verbatim(workspace):
    result = record_baseline(workspace)
    event = one(workspace.store.execute("SELECT * FROM events WHERE event_id=?",
                                        (result["event_id"],)))
    assert "Redis mutex" in event["payload"]
    assert event["kind"] == "bug_fix"


def test_contradiction_is_flagged_not_resolved(workspace):
    compiler.record_event(
        workspace.store, workspace.catalog, workspace.repo_id, workspace.root, workspace.commit,
        {"kind": "decision", "summary": "serialize",
         "invariants": ["All refresh operations must use RefreshCoordinator"],
         "symbols": ["RefreshCoordinator.acquire"]},
    )
    second = compiler.record_event(
        workspace.store, workspace.catalog, workspace.repo_id, workspace.root, workspace.commit,
        {"kind": "decision", "summary": "websocket bypass",
         "invariants": ["WebSocket refresh operations must never use RefreshCoordinator"],
         "symbols": ["RefreshCoordinator.acquire"]},
    )
    assert second["contradictions"], "expected the conflicting invariant to be flagged"
    # Both survive. Nothing was overwritten.
    active = rows(workspace.store.execute(
        "SELECT * FROM memories WHERE kind='invariant' AND status='ACTIVE'"))
    assert len(active) >= 2


def test_agent_cannot_rewrite_a_human_memory(workspace):
    result = compiler.record_event(
        workspace.store, workspace.catalog, workspace.repo_id, workspace.root, workspace.commit,
        {"kind": "decision", "summary": "human rule", "authority": "human",
         "invariants": ["Never retry card_declined"], "symbols": ["refresh_session"]},
    )
    memory_id = next(m["memory_id"] for m in result["memories_created"] if m["kind"] == "invariant")

    refused = compiler.correct(workspace.store, workspace.catalog, workspace.repo_id, memory_id,
                               body="actually retrying is fine", actor="agent")
    assert refused["ok"] is False
    assert "supersede" in refused["error"]

    unchanged = one(workspace.store.execute("SELECT body FROM memories WHERE memory_id=?", (memory_id,)))
    assert "Never retry" in unchanged["body"]


def test_correct_versions_rather_than_overwrites(workspace):
    result = record_baseline(workspace)
    memory_id = result["memories_created"][0]["memory_id"]
    original = one(workspace.store.execute("SELECT body FROM memories WHERE memory_id=?", (memory_id,)))

    compiler.correct(workspace.store, workspace.catalog, workspace.repo_id, memory_id,
                     body="revised text", reason="clarity")
    versions = rows(workspace.store.execute(
        "SELECT * FROM memory_versions WHERE memory_id=?", (memory_id,)))

    assert len(versions) == 1
    assert versions[0]["body"] == original["body"]
    current = one(workspace.store.execute("SELECT * FROM memories WHERE memory_id=?", (memory_id,)))
    assert current["body"] == "revised text" and current["version"] == 2


def test_supersede_keeps_both_and_links_them(workspace):
    result = record_baseline(workspace)
    memory_id = next(m["memory_id"] for m in result["memories_created"] if m["kind"] == "invariant")

    outcome = compiler.supersede(workspace.store, workspace.catalog, workspace.repo_id, memory_id,
                                 body="Refresh serialization now uses Redlock", reason="migrated")
    assert outcome["ok"]

    old = one(workspace.store.execute("SELECT * FROM memories WHERE memory_id=?", (memory_id,)))
    assert old["status"] == "SUPERSEDED"
    edge = one(workspace.store.execute(
        "SELECT * FROM memory_edges WHERE from_id=? AND kind='SUPERSEDES'", (outcome["replacement"],)))
    assert edge["to_id"] == memory_id
    # The replacement inherits the anchors, so it is attached to the same code.
    inherited = rows(workspace.store.execute(
        "SELECT * FROM anchors WHERE memory_id=?", (outcome["replacement"],)))
    assert inherited


def test_listing_reports_the_worst_anchor_not_the_best(workspace, project):
    """A memory is only as trustworthy as its weakest anchor."""
    from icn import workspace as ws_mod

    record_baseline(workspace)
    project.write("auth.py", project.read("auth.py").replace(
        "        lock = self._lock_for(session_id)\n        return lock",
        "        return None  # gutted",
    ))
    ws_mod.ensure_indexed(workspace)

    listed = compiler.list_memories(workspace.store)
    assert listed
    assert any(m["anchor_status"] == "NEEDS_REVIEW" for m in listed)
    assert listed[0]["anchor_status"] != "ACTIVE", "worst anchors must sort first"


def test_a_memory_carries_the_context_it_was_recorded_in(workspace):
    """Measured before this: median body was 150 characters and 35 of 48 were
    under 200 - a headline, not knowledge. "settle must be idempotent" tells a
    future agent nothing about the incident that made it one."""
    result = compiler.record_event(
        workspace.store, workspace.catalog, workspace.repo_id, workspace.root, workspace.commit,
        {"kind": "bug_fix",
         "summary": "Serialised refresh so concurrent requests stop invalidating each other",
         "reasoning": "Two parallel requests each rotated the token, so the second "
                      "invalidated the first and logged the user out.",
         "invariants": ["Every refresh must pass through RefreshCoordinator.acquire"],
         "symbols": ["refresh_session"]})

    invariant = next(m for m in result["memories_created"] if m["kind"] == "invariant")
    body = one(workspace.store.execute(
        "SELECT body FROM memories WHERE memory_id=?", (invariant["memory_id"],)))["body"]

    assert "RefreshCoordinator.acquire" in body, "the claim itself must survive"
    assert "logged the user out" in body, "the reasoning must travel with it"
    assert "refresh_session" in body, "a reader must know what it applies to"
    assert len(body) > 250, f"still a headline at {len(body)} chars"


def test_context_is_not_repeated_back_at_the_reader(workspace):
    """Agents phrase a warning and its summary similarly; echoing the summary
    under a claim that already states it is noise."""
    result = compiler.record_event(
        workspace.store, workspace.catalog, workspace.repo_id, workspace.root, workspace.commit,
        {"kind": "decision", "summary": "Rotate the refresh token on every use",
         "decisions": ["Rotate the refresh token on every use"],
         "symbols": ["refresh_session"]})

    decision = next(m for m in result["memories_created"] if m["kind"] == "decision")
    body = one(workspace.store.execute(
        "SELECT body FROM memories WHERE memory_id=?", (decision["memory_id"],)))["body"]
    assert body.count("Rotate the refresh token on every use") == 1


def test_a_thin_write_is_told_so_while_the_context_is_still_available(workspace):
    """The compiler cannot write the knowledge - only the agent knows why it
    did what it did. It can say the entry will be useless later, while the
    caller can still fix it. Nobody comes back to enrich a memory."""
    thin = compiler.record_event(
        workspace.store, workspace.catalog, workspace.repo_id, workspace.root, workspace.commit,
        {"kind": "bug_fix", "summary": "fixed it",
         "invariants": ["settle must be idempotent"]})

    quality = thin["quality"]
    assert quality["sufficient"] is False
    assert quality["notes"]
    joined = " ".join(quality["notes"])
    assert "reasoning" in joined
    assert "failed_attempts" in joined, "a bug fix with no rejected approach should be queried"
    assert thin["ok"] is True, "poor quality must never block the write"


def test_a_rich_write_passes_without_nagging(workspace):
    rich = compiler.record_event(
        workspace.store, workspace.catalog, workspace.repo_id, workspace.root, workspace.commit,
        {"kind": "bug_fix",
         "summary": "Serialised refresh so concurrent requests stop invalidating each other",
         "reasoning": "Two parallel requests each rotated the token, so whichever "
                      "finished second invalidated the first and logged the user out.",
         "invariants": ["Every refresh for one session must pass through "
                        "RefreshCoordinator.acquire, because two concurrent rotations "
                        "invalidate each other and log the user out"],
         "failed_attempts": ["A Redis mutex was rejected: it deadlocks when the network "
                             "partitions mid-hold, and the TTL that would fix it exceeds "
                             "the request budget"],
         "symbols": ["refresh_session"]})

    assert rich["quality"]["sufficient"] is True, rich["quality"]["notes"]
    assert rich["quality"]["median_body_chars"] > 250
