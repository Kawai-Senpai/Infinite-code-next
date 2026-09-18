"""Derived knowledge: what ICN infers, and the review that settles it.

The safety property under test throughout is that an inference never passes
for a fact. It is written down, it is findable, and it is visibly a proposal
until somebody says otherwise.
"""

from __future__ import annotations

from conftest import record_baseline

from icn import briefing, compiler, infer, search
from icn.db import one, rows


def _propose_one(ws) -> dict:
    """Write a single inference directly, for tests about review rather than detection."""
    from icn import ids
    from icn.db import write_tx

    memory_id = ids.new_id(ids.MEMORY)
    with write_tx(ws.store):
        ws.store.execute(
            "INSERT INTO memories (memory_id, kind, title, body, severity, authority, confidence,"
            " status, scope, version, created_at, updated_at, claim, evidence_count,"
            " is_inference, review_status, parent_count)"
            " VALUES (?,'invariant','proposed','proposed rule','medium','derived',0.5,'ACTIVE',"
            "'repo',1,?,?,'proposed rule',1,1,NULL,3)",
            (memory_id, infer.now(), infer.now()))
        ws.store.execute(
            "INSERT INTO fts_memories (memory_id, title, body, kind)"
            " VALUES (?,'proposed','proposed rule','invariant')", (memory_id,))
    return {"memory_id": memory_id}


# ------------------------------------------------------------------ agreement

def test_agreement_measures_consensus_not_unanimity():
    """The bug that made the first version of this module propose nothing.

    Intersecting every claim's vocabulary requires a word present in all of
    them, which on real data is never true: measured against this repository's
    store, all 203 candidates scored exactly 0.0.
    """
    claims = [
        "the parser must hold the native import lock",
        "the parser must hold the native import lock on windows",
        "native import lock is required before parser startup",
        "something entirely unrelated about billing invoices",
    ]
    score, shared = infer._agreement(claims)

    assert score > 0, "a majority theme must score above zero"
    assert "native" in shared and "import" in shared
    # The outlier contributes vocabulary but does not get to veto the theme.
    assert "billing" not in shared


def test_a_single_claim_cannot_agree_with_itself():
    assert infer._agreement(["only one claim here"]) == (0.0, [])


def test_generic_words_are_not_evidence_of_agreement():
    """Otherwise 'the function must not fail' matches 'the test must not fail'."""
    _, shared = infer._agreement([
        "this function must return a value",
        "that function must return a value",
        "another function must return a value",
    ])
    assert not (set(shared) & infer.GENERIC)


# ------------------------------------------------------------------- proposing

def test_an_inference_is_born_unreviewed_and_marked(workspace):
    record_baseline(workspace)
    infer.propose(workspace.store, workspace.commit)

    for row in rows(workspace.store.execute(
            "SELECT * FROM memories WHERE is_inference=1")):
        assert row["review_status"] is None, "a new proposal is not pre-decided"
        assert row["authority"] == "derived", "provenance must say ICN wrote it"
        assert row["confidence"] <= infer.INFERENCE_CONFIDENCE
        assert row["confidence"] < 0.8, "an inference must not match an asserted memory"


def test_proposing_twice_does_not_raise_the_same_question_twice(workspace):
    record_baseline(workspace)
    first = infer.propose(workspace.store, workspace.commit)
    second = infer.propose(workspace.store, workspace.commit)

    assert second["count"] == 0, "a proposal already made must not be repeated"
    if first["count"]:
        assert infer.queue(workspace.store)["total"] == first["count"]


def test_a_declined_proposal_is_never_raised_again(workspace):
    """A decline is a decision. Re-asking is how a review queue loses its reader."""
    record_baseline(workspace)
    first = infer.propose(workspace.store, workspace.commit)
    if not first["count"]:
        return
    for proposal in first["proposed"]:
        infer.review(workspace.store, proposal["memory_id"], "decline")

    again = infer.propose(workspace.store, workspace.commit)
    assert again["count"] == 0


def test_an_inference_carries_the_memories_it_came_from(workspace):
    """Provenance is the whole basis on which a reviewer can judge a proposal."""
    record_baseline(workspace)
    result = infer.propose(workspace.store, workspace.commit)
    if not result["count"]:
        return

    memory_id = result["proposed"][0]["memory_id"]
    parents = rows(workspace.store.execute(
        "SELECT to_id FROM memory_edges WHERE from_id=? AND kind='DERIVES_FROM'"
        " AND status='ACTIVE'", (memory_id,)))
    assert parents, "an inference with no stated parents cannot be reviewed"
    for parent in parents:
        assert one(workspace.store.execute(
            "SELECT memory_id FROM memories WHERE memory_id=?", (parent["to_id"],)))


def test_inference_edges_are_class_inferred_not_asserted(workspace):
    """Trust class is what stops derived relevance travelling as far as stated."""
    record_baseline(workspace)
    if not infer.propose(workspace.store, workspace.commit)["count"]:
        return
    for edge in rows(workspace.store.execute(
            "SELECT * FROM memory_edges WHERE source='infer'")):
        assert edge["edge_class"] == "inferred"


# ---------------------------------------------------------------------- review

def test_approving_promotes_an_inference_to_a_stated_fact(workspace):
    memory_id = _propose_one(workspace)["memory_id"]
    result = infer.review(workspace.store, memory_id, "approve")

    assert result["ok"] and result["is_inference"] is False
    assert result["review_status"] == "approved"
    row = one(workspace.store.execute(
        "SELECT * FROM memories WHERE memory_id=?", (memory_id,)))
    # Approving says it is true, not that a human wrote it: how it was born is
    # provenance and must survive the promotion.
    assert row["authority"] == "derived"


def test_declining_forgets_it_and_removes_it_from_search(workspace):
    memory_id = _propose_one(workspace)["memory_id"]
    infer.review(workspace.store, memory_id, "decline")

    row = one(workspace.store.execute(
        "SELECT status, review_status FROM memories WHERE memory_id=?", (memory_id,)))
    assert row["status"] == "FORGOTTEN"
    assert row["review_status"] == "declined"
    assert one(workspace.store.execute(
        "SELECT COUNT(*) AS n FROM fts_memories WHERE memory_id=?", (memory_id,)))["n"] == 0


def test_undo_returns_a_declined_memory_exactly_as_it_was(workspace):
    """Why review_status is a separate column: nothing has to be reconstructed."""
    memory_id = _propose_one(workspace)["memory_id"]
    infer.review(workspace.store, memory_id, "decline")
    infer.review(workspace.store, memory_id, "undo")

    row = one(workspace.store.execute(
        "SELECT * FROM memories WHERE memory_id=?", (memory_id,)))
    assert row["status"] == "ACTIVE"
    assert row["review_status"] is None
    assert int(row["is_inference"]) == 1
    assert one(workspace.store.execute(
        "SELECT COUNT(*) AS n FROM fts_memories WHERE memory_id=?", (memory_id,)))["n"] == 1
    assert infer.queue(workspace.store)["total"] == 1


def test_a_stated_memory_cannot_be_reviewed(workspace):
    """Review is for proposals. Approving a fact somebody asserted is meaningless."""
    result = record_baseline(workspace)
    memory_id = result["memories_created"][0]["memory_id"]

    answer = infer.review(workspace.store, memory_id, "approve")
    assert not answer["ok"]
    assert "not an inference" in answer["error"]


def test_undo_with_nothing_to_undo_is_refused(workspace):
    memory_id = _propose_one(workspace)["memory_id"]
    answer = infer.review(workspace.store, memory_id, "undo")
    assert not answer["ok"] and "no review to undo" in answer["error"]


def test_an_unknown_signal_is_refused(workspace):
    memory_id = _propose_one(workspace)["memory_id"]
    assert not infer.review(workspace.store, memory_id, "reject")["ok"]
    assert not infer.review(workspace.store, memory_id, "")["ok"]


def test_review_is_audited(workspace):
    memory_id = _propose_one(workspace)["memory_id"]
    infer.review(workspace.store, memory_id, "approve", reason="checked it", actor="human")

    trail = rows(workspace.store.execute(
        "SELECT * FROM corrections WHERE memory_id=?", (memory_id,)))
    assert any(c["action"] == "review_approve" and c["actor"] == "human" for c in trail)


def test_the_queue_is_ordered_by_strength_of_support(workspace):
    from icn import ids
    from icn.db import write_tx

    for parents in (2, 7, 4):
        with write_tx(workspace.store):
            workspace.store.execute(
                "INSERT INTO memories (memory_id, kind, title, body, severity, authority,"
                " confidence, status, scope, version, created_at, updated_at, claim,"
                " evidence_count, is_inference, review_status, parent_count)"
                " VALUES (?,'invariant',?,?,'medium','derived',0.5,'ACTIVE','repo',1,?,?,?,1,1,"
                " NULL,?)",
                (ids.new_id(ids.MEMORY), f"p{parents}", f"body {parents}", infer.now(),
                 infer.now(), f"claim {parents}", parents))

    pending = infer.queue(workspace.store)["pending"]
    assert [p["parent_count"] for p in pending] == [7, 4, 2]


# ------------------------------------------------------------ search behaviour

def test_an_unreviewed_inference_ranks_below_the_same_stated_memory():
    """The penalty exists so a guess never displaces a fact answering one query."""
    stated = {"memory_id": "mem_a", "kind": "invariant", "severity": "high",
              "is_inference": 0, "title": "x", "body": "x"}
    proposed = {**stated, "memory_id": "mem_b", "is_inference": 1}
    relevance = {"mem_a": 0.5, "mem_b": 0.5}

    for intent in ("modify", "debug", "understand"):
        assert (search._memory_score(proposed, relevance, None, intent)
                < search._memory_score(stated, relevance, None, intent))


def test_approving_an_inference_lifts_the_search_penalty():
    approved = {"memory_id": "mem_a", "kind": "invariant", "severity": "high",
                "is_inference": 0, "review_status": "approved", "title": "x", "body": "x"}
    unreviewed = {**approved, "memory_id": "mem_b", "is_inference": 1, "review_status": None}
    relevance = {"mem_a": 0.5, "mem_b": 0.5}

    gap = (search._memory_score(approved, relevance, None, "modify")
           - search._memory_score(unreviewed, relevance, None, "modify"))
    assert gap == search.INFERENCE_PENALTY


# ---------------------------------------------------------------- the briefing

def test_the_briefing_never_quotes_an_unreviewed_inference_as_a_rule(workspace):
    """The one place read without checking. A proposal must not read as law."""
    from icn import ids
    from icn.db import write_tx

    memory_id = ids.new_id(ids.MEMORY)
    with write_tx(workspace.store):
        workspace.store.execute(
            "INSERT INTO memories (memory_id, kind, title, body, severity, authority, confidence,"
            " status, scope, version, created_at, updated_at, claim, evidence_count,"
            " is_inference, review_status, parent_count)"
            " VALUES (?,'invariant','INFERRED RULE','inferred body','critical','derived',0.5,"
            "'ACTIVE','repo',1,?,?,'inferred body',1,1,NULL,3)",
            (memory_id, infer.now(), infer.now()))

    built = briefing.build(workspace.store)
    assert memory_id not in {r["memory_id"] for r in built["rules"]}
    assert built["inferences_pending_review"] >= 1


def test_the_briefing_counts_what_is_waiting_to_be_reviewed(workspace):
    record_baseline(workspace)
    before = briefing.build(workspace.store)["inferences_pending_review"]
    _propose_one(workspace)
    after = briefing.build(workspace.store)["inferences_pending_review"]
    assert after == before + 1
