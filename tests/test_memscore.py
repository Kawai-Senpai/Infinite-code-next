"""MemScore: three axes, never collapsed into one.

The behaviour under test is mostly a refusal. The benchmark must not declare a
winner when the difference is noise, and must not let an accuracy gain hide
what it cost.
"""

from __future__ import annotations

from conftest import record_baseline

from icn import memscore


def _result(mrr: float, p50: float, tokens: int) -> dict:
    return {"ok": True, "memscore": f"{mrr:.3f} MRR / {p50:.0f}ms / {tokens}tok",
            "accuracy": {"mrr": mrr}, "latency_ms": {"p50": p50},
            "context": {"mean_tokens": tokens}}


def test_a_small_difference_is_reported_as_noise():
    """The recorded warning about this search path, enforced by the tool itself.

    The optimum is flat: the top 15 configurations of the original grid search
    spanned 0.1043 to 0.1060 MRR over 1917 queries. A benchmark that called
    that a win would launder noise into a result.
    """
    verdict = memscore.compare(_result(0.180, 163, 7250),
                               _result(0.1815, 165, 7250))["verdict"]
    assert "noise" in verdict


def test_an_accuracy_gain_that_cost_latency_is_not_called_better():
    verdict = memscore.compare(_result(0.180, 163, 7250),
                               _result(0.210, 400, 14000))["verdict"]
    assert "cost" in verdict and "decision, not a measurement" in verdict


def test_an_accuracy_gain_at_no_cost_is_called_what_it_is():
    verdict = memscore.compare(_result(0.180, 163, 7250),
                               _result(0.210, 150, 7000))["verdict"]
    assert "no cost" in verdict


def test_a_regression_is_reported_plainly():
    verdict = memscore.compare(_result(0.210, 163, 7250),
                               _result(0.150, 163, 7250))["verdict"]
    assert "fell" in verdict


def test_comparing_a_failed_run_is_refused():
    assert not memscore.compare(_result(0.18, 163, 7250), {"ok": False})["ok"]


def test_every_axis_is_reported_separately():
    """The whole point: no single number the reader cannot decompose."""
    compared = memscore.compare(_result(0.180, 163, 7250), _result(0.210, 400, 14000))
    assert set(compared["delta"]) >= {"mrr", "p50_ms", "mean_tokens"}
    assert compared["delta"]["p50_ms"] > 0 and compared["delta"]["mean_tokens"] > 0


def test_too_few_pairs_refuses_to_produce_a_score(workspace):
    """A number from 3 queries would be worse than no number."""
    record_baseline(workspace)
    result = memscore.run(workspace.store, workspace.root, workspace.catalog, sample=120)
    if not result["ok"]:
        assert "meaningless" in result["error"]


def test_inferences_are_never_used_as_ground_truth(workspace):
    """Scoring retrieval against anchors ICN chose itself would mark its own homework."""
    from icn import ids, infer
    from icn.db import write_tx

    record_baseline(workspace)
    memory_id = ids.new_id(ids.MEMORY)
    with write_tx(workspace.store):
        workspace.store.execute(
            "INSERT INTO memories (memory_id, kind, title, body, severity, authority,"
            " confidence, status, scope, version, created_at, updated_at, claim,"
            " evidence_count, is_inference, review_status, parent_count)"
            " VALUES (?,'invariant',?,'b','medium','derived',0.5,'ACTIVE','repo',1,?,?,'c',1,"
            " 1,NULL,3)",
            (memory_id, "a derived title long enough to be selected as a query", infer.now(),
             infer.now()))

    pairs = memscore._gold_pairs(workspace.store, 200)
    assert memory_id not in {p["memory_id"] for p in pairs}


def test_rank_is_one_based_and_matches_on_symbol_path():
    capsules = [{"symbol": "a.b"}, {"symbol": "c.d"}, {"symbol": "e.f"}]
    assert memscore._rank_of(capsules, "a.b") == 1
    assert memscore._rank_of(capsules, "e.f") == 3
    assert memscore._rank_of(capsules, "nope") is None
