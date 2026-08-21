"""investigate(): intents, fusion, staleness labelling, budget, problems."""

from __future__ import annotations

from conftest import record_baseline

from icn import search as search_mod
from icn import workspace as ws_mod


def run(ws, query, **kwargs):
    return search_mod.investigate(ws.store, ws.catalog, ws.root, query, commit=ws.commit, **kwargs)


def test_intent_is_inferred_from_the_question():
    assert search_mod.infer_intent("I want to remove RefreshCoordinator") == "modify"
    assert search_mod.infer_intent("why does this queue exist") == "understand"
    assert search_mod.infer_intent("where is UserStatus defined") == "locate"
    assert search_mod.infer_intent("this test is failing with a race") == "debug"
    assert search_mod.infer_intent("audit every caller for coverage") == "audit"


def test_investigation_surfaces_the_relevant_memories(workspace):
    record_baseline(workspace)
    result = run(workspace, "I need to change refresh token rotation. What will I break?")

    assert result["intent"] == "modify"
    assert result["capsules"]
    texts = [m.get("text", "") for c in result["capsules"] for m in c["memory"]]
    assert any("bypass RefreshCoordinator" in t for t in texts)
    assert any("Redis mutex" in t for t in texts)


def test_capsules_carry_structure_not_file_dumps(workspace):
    record_baseline(workspace)
    result = run(workspace, "refresh session")
    capsule = next(c for c in result["capsules"] if c["symbol"] == "refresh_session")

    assert capsule["path"] == "auth.py"
    assert "-" in capsule["lines"]
    assert "post_refresh" in capsule["upstream"]
    assert "test_parallel_refresh_regression" in capsule["tests"]
    assert "score_breakdown" in capsule


def test_graph_proximity_finds_code_the_query_never_named(workspace):
    """A memory two trusted hops away should still surface."""
    record_baseline(workspace)
    result = run(workspace, "RefreshCoordinator", depth=2)
    reached = {c["symbol"] for c in result["capsules"]}
    assert "post_refresh" in reached or "refresh_session" in reached


def test_stale_memories_are_labelled_never_silently_served(workspace, project):
    """The central promise: an unverified memory is visibly unverified."""
    record_baseline(workspace)
    project.write("auth.py", project.read("auth.py").replace(
        "        lock = self._lock_for(session_id)\n        return lock",
        "        return None  # serialization removed",
    ))
    ws_mod.ensure_indexed(workspace)

    result = run(workspace, "refresh coordinator acquire")
    flagged = [m for c in result["capsules"] for m in c["memory"]
               if m["anchor_status"] != "ACTIVE"]
    assert flagged, "expected at least one memory marked unverified"
    assert all("warning" in m for m in flagged)
    assert any("changed since this was recorded" in m["warning"] for m in flagged)


def test_problems_report_stale_knowledge(workspace, project):
    record_baseline(workspace)
    project.write("auth.py", project.read("auth.py").replace(
        "        lock = self._lock_for(session_id)\n        return lock",
        "        return None",
    ))
    ws_mod.ensure_indexed(workspace)

    result = run(workspace, "refresh coordinator")
    kinds = {p["kind"] for p in result["problems"]}
    assert "stale_knowledge" in kinds


def test_problems_report_a_tombstoned_implementation(workspace, project):
    record_baseline(workspace)
    source = project.read("auth.py")
    start = source.index("class RefreshCoordinator")
    end = source.index("def refresh_session")
    project.write("auth.py", source[:start] + source[end:])
    ws_mod.ensure_indexed(workspace)

    result = run(workspace, "refresh session coordinator")
    kinds = {p["kind"] for p in result["problems"]}
    assert kinds & {"historical_implementation", "stale_knowledge"}


def test_budget_is_respected(workspace):
    record_baseline(workspace)
    generous = run(workspace, "refresh", budget=20000)
    tight = run(workspace, "refresh", budget=400)
    assert len(tight["capsules"]) <= len(generous["capsules"])
    assert tight["budget"]["used_estimate"] <= 400 + 2000  # one capsule may exceed a tiny budget


def test_empty_result_explains_itself(workspace):
    result = run(workspace, "zzzz_nonexistent_term_qqqq")
    assert result["capsules"] == []
    assert "note" in result


def test_investigation_is_persisted_and_expandable(workspace):
    record_baseline(workspace)
    result = run(workspace, "refresh token rotation")
    assert result["investigation_id"]

    expanded = search_mod.expand(workspace.store, workspace.catalog,
                                 result["investigation_id"], "refresh_session")
    assert expanded["ok"]
    assert expanded["capsules"]
    assert expanded["capsules"][0]["symbol"] == "refresh_session"


def test_expand_on_unknown_investigation_is_an_error_not_a_crash(workspace):
    outcome = search_mod.expand(workspace.store, workspace.catalog, "inv_nope", "x")
    assert outcome["ok"] is False


def test_a_memory_is_rendered_once_per_result(workspace):
    """A warning anchored to five call sites is one fact, not five.

    Repeating its full text once per capsule spends the token budget on
    duplication instead of coverage - the exact failure investigate() exists
    to prevent.
    """
    record_baseline(workspace)
    result = run(workspace, "refresh coordinator session rotation")

    full_text_counts: dict[str, int] = {}
    back_refs = 0
    for capsule in result["capsules"]:
        for memory in capsule["memory"]:
            if memory.get("also_applies_here"):
                back_refs += 1
                assert "text" not in memory, "a back-reference must not repeat the body"
                assert memory["shown_at"], "a back-reference must say where it was shown"
            elif memory.get("not_shown"):
                # Named rather than quoted: costs a title, not a body.
                assert "text" not in memory
            else:
                full_text_counts[memory["memory_id"]] = \
                    full_text_counts.get(memory["memory_id"], 0) + 1

    assert full_text_counts, "expected at least one memory rendered in full"
    repeated = {k: v for k, v in full_text_counts.items() if v > 1}
    assert not repeated, f"these memories were rendered in full more than once: {repeated}"
    assert back_refs > 0, "expected shared memories to appear as back-references"


def test_memory_bodies_are_searchable_by_their_own_words(workspace):
    """fts_memories has to actually be populated.

    Without it a memory is only reachable through the code it is anchored to,
    so asking about a decision in the words the decision itself uses returns
    nothing - half of retrieval silently dead.
    """
    from icn import compiler

    compiler.record_event(
        workspace.store, workspace.catalog, workspace.repo_id, workspace.root, workspace.commit,
        {"kind": "decision", "summary": "we picked a bottom-k MinHash sketch",
         "invariants": ["Never store the full normalised token text for a symbol"],
         "symbols": ["refresh_session"]},
    )

    result = run(workspace, "store full normalised token text")
    texts = [m.get("text", "") for c in result["capsules"] for m in c["memory"]]
    assert any("Never store the full" in t for t in texts), \
        "a memory must be findable by the words in its own body"


def test_memories_are_ranked_by_relevance_to_this_query(workspace):
    """Sorting by severity alone returned every memory attached to a selected
    symbol, whatever was asked. Measured across four unrelated queries on a
    real store, results overlapped 58-86%."""
    from icn import compiler

    compiler.record_event(
        workspace.store, workspace.catalog, workspace.repo_id, workspace.root, workspace.commit,
        {"kind": "decision", "summary": "token rotation cadence",
         "decisions": ["rotate the refresh token on every single use"],
         "symbols": ["refresh_session"]})
    compiler.record_event(
        workspace.store, workspace.catalog, workspace.repo_id, workspace.root, workspace.commit,
        {"kind": "performance", "summary": "session lookup cost",
         "performance": ["session lookup dominates the p99 latency budget"],
         "symbols": ["refresh_session"]})

    rotation = run(workspace, "how often do we rotate the refresh token")
    quoted = [m for c in rotation["capsules"] for m in c["memory"] if "text" in m]
    assert quoted, "expected at least one memory quoted in full"
    assert any("rotate the refresh token" in m["text"] for m in quoted), \
        "the memory that answers the question must be quoted"


def test_low_relevance_memories_are_named_not_dropped(workspace):
    """Below the floor a memory is still listed with its id, so nothing is
    hidden - it just does not spend the token budget."""
    record_baseline(workspace)
    result = run(workspace, "refresh coordinator acquire")

    named = [m for c in result["capsules"] for m in c["memory"] if m.get("not_shown")]
    for entry in named:
        assert entry["memory_id"], "a named memory must stay addressable"
        assert "text" not in entry, "a named memory must not spend budget on its body"
        assert entry.get("title"), "a named memory must still say what it is"


def test_every_memory_is_searchable_by_its_own_words(workspace):
    """A derived index added after the fact is empty for everything already
    stored, and a half-populated FTS table fails silently. Measured on a real
    store, 25 of 34 memories were invisible to text search."""
    from icn import compiler
    from icn.db import one

    compiler.record_event(
        workspace.store, workspace.catalog, workspace.repo_id, workspace.root, workspace.commit,
        {"kind": "decision", "summary": "indexing strategy",
         "invariants": ["the walk must stream so the budget covers discovery"],
         "symbols": ["refresh_session"]})

    total = one(workspace.store.execute(
        "SELECT COUNT(*) AS n FROM memories WHERE status='ACTIVE'"))["n"]
    indexed = one(workspace.store.execute(
        "SELECT COUNT(*) AS n FROM fts_memories f JOIN memories m"
        " ON m.memory_id = f.memory_id WHERE m.status='ACTIVE'"))["n"]
    assert indexed == total, f"{total - indexed} active memories are invisible to search"
