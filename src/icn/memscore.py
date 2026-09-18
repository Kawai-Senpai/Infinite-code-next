"""Retrieval quality, measured on three axes and never collapsed into one.

ICN's search weights were tuned once, by a grid search over 1917 query/answer
pairs, and that script was never committed. So the constants in search.py carry
a number nobody can reproduce, and the warnings recorded alongside them say the
optimum is flat and that a 1-2% MRR difference is noise. Both of those are
things a standing benchmark should tell you, not things that should have to be
rediscovered and written down as prose.

This is that benchmark, and it reports **three numbers side by side**:

    MemScore: 0.112 MRR / 38ms / 1240tok
              |         |      |
              |         |      +-- context returned, a proxy for what a call costs
              |         +--------- retrieval latency
              +------------------- rank of the right answer

The refusal to weight them is the point. A configuration that is 2% better on
MRR while returning three times the context is not better, it is a different
trade, and any single score has to pick weights on the user's behalf in order
to hide that. Reported separately, the trade stays visible and the person
reading the number decides.

**Ground truth is the store's own anchors.** A memory's title is a question
somebody really asked about this code, and the symbol it is anchored to is the
answer, established when the memory was recorded rather than labelled after the
fact. That makes the benchmark free and specific to each repository, and it is
also the honest limitation: a memory usually has several equally valid anchor
targets and only one counts as correct, so absolute MRR reads low. The number
is for comparing two configurations of ICN against each other on one
repository. It is not a score to compare against a published retrieval result,
and a run on a different repository is a different benchmark.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

from .db import rows

# Ranks past this contribute nothing. Matches the depth investigate() actually
# returns, because recall the caller never sees is not recall.
CUTOFF = 20

# Enough pairs for the mean to settle without making a run cost minutes. The
# recorded warning about this search path is that the optimum is flat and small
# differences are noise; a sample this size is why.
DEFAULT_SAMPLE = 120


def _gold_pairs(conn: sqlite3.Connection, limit: int) -> list[dict[str, Any]]:
    """(question, correct symbol) drawn from anchored memories.

    Only ACTIVE anchors on ACTIVE symbols, and never an inference: a derived
    memory's anchor was chosen by the machine, so scoring retrieval against it
    would be marking ICN's homework with ICN's own answers.
    """
    return rows(conn.execute(
        "SELECT m.memory_id, m.title, m.kind, a.symbol_id, s.symbol_path"
        " FROM memories m"
        " JOIN anchors a ON a.memory_id = m.memory_id"
        " JOIN symbols s ON s.symbol_id = a.symbol_id"
        " WHERE m.status='ACTIVE' AND COALESCE(m.is_inference, 0) = 0"
        "   AND a.status='ACTIVE' AND s.status='ACTIVE'"
        "   AND m.title IS NOT NULL AND LENGTH(m.title) > 25"
        " GROUP BY m.memory_id"
        " ORDER BY m.memory_id LIMIT ?", (limit,)))


def _rank_of(results: list[dict[str, Any]], gold_path: str) -> int | None:
    """1-based rank of the gold symbol, or None if it never appeared.

    Matched on symbol_path because that is what a capsule carries: capsules are
    built for a reader, so they name the symbol rather than exposing its id.
    A path is unique per symbol within a repository, so this is exact, not a
    fuzzy match.
    """
    for position, capsule in enumerate(results, start=1):
        if capsule.get("symbol") == gold_path:
            return position
    return None


def run(conn: sqlite3.Connection, root: Path, catalog: sqlite3.Connection | None = None,
        sample: int = DEFAULT_SAMPLE, intent: str = "understand") -> dict[str, Any]:
    """Score this repository's retrieval. Reads only; writes nothing."""
    from . import search as search_mod

    pairs = _gold_pairs(conn, sample)
    if len(pairs) < 10:
        return {
            "ok": False,
            "error": f"only {len(pairs)} anchored memories with usable titles; "
                     "a score from fewer than 10 would be meaningless",
            "pairs": len(pairs),
        }

    reciprocal: list[float] = []
    hits_at_1 = hits_at_5 = hits_at_10 = 0
    latencies: list[float] = []
    context_chars: list[int] = []
    failures = 0

    for pair in pairs:
        question = str(pair["title"])
        started = time.perf_counter()
        try:
            # Conversations off: they are leads attached to an answer, not part
            # of ranked retrieval, and recalling them costs time the latency
            # axis would wrongly attribute to search.
            answer = search_mod.investigate(
                conn, catalog, root, question, intent=intent, commit=None,
                conversations=False)
        except Exception:
            # A query that errors is a failure of the run, not a zero score:
            # counting it as a miss would let a crash look like bad ranking.
            failures += 1
            continue
        latencies.append((time.perf_counter() - started) * 1000.0)

        capsules = answer.get("capsules") or []
        context_chars.append(len(str(answer)))
        rank = _rank_of(capsules[:CUTOFF], pair["symbol_path"])
        if rank is None:
            reciprocal.append(0.0)
            continue
        reciprocal.append(1.0 / rank)
        hits_at_1 += rank <= 1
        hits_at_5 += rank <= 5
        hits_at_10 += rank <= 10

    scored = len(reciprocal)
    if not scored:
        return {"ok": False, "error": "every query failed", "failures": failures}

    latencies.sort()
    mrr = sum(reciprocal) / scored
    p50 = latencies[len(latencies) // 2] if latencies else 0.0
    p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else 0.0
    # Roughly four characters to a token. Approximate on purpose: this is a
    # cost proxy for comparing two runs, and a real tokeniser would add a
    # dependency to buy precision the comparison does not need.
    tokens = int((sum(context_chars) / len(context_chars)) / 4) if context_chars else 0

    return {
        "ok": True,
        "memscore": f"{mrr:.3f} MRR / {p50:.0f}ms / {tokens}tok",
        "accuracy": {
            "mrr": round(mrr, 4),
            "hit_at_1": round(hits_at_1 / scored, 3),
            "hit_at_5": round(hits_at_5 / scored, 3),
            "hit_at_10": round(hits_at_10 / scored, 3),
        },
        "latency_ms": {"p50": round(p50, 1), "p95": round(p95, 1),
                       "mean": round(sum(latencies) / len(latencies), 1) if latencies else 0.0},
        "context": {"mean_tokens": tokens,
                    "mean_chars": int(sum(context_chars) / len(context_chars))
                    if context_chars else 0},
        "queries": {"scored": scored, "failed": failures, "intent": intent},
        "how_to_read": (
            "Three axes, deliberately not combined: a configuration that gains MRR while "
            "returning more context is a different trade, not a better one. Compare two runs "
            "on THIS repository; the absolute MRR is not comparable to a published number, "
            "because a memory often has several valid anchors and only one is scored correct."),
        "caution": (
            "The optimum here is flat. Treat a difference under ~2% MRR on one repository as "
            "noise, not as a result."),
    }


def compare(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Put two runs beside each other and refuse to declare a winner.

    Returns the delta on each axis and says plainly when a change bought
    accuracy with latency or context, which is the judgement a single score
    would have quietly made on the reader's behalf.
    """
    if not (before.get("ok") and after.get("ok")):
        return {"ok": False, "error": "both runs must have succeeded"}

    mrr_delta = after["accuracy"]["mrr"] - before["accuracy"]["mrr"]
    latency_delta = after["latency_ms"]["p50"] - before["latency_ms"]["p50"]
    token_delta = after["context"]["mean_tokens"] - before["context"]["mean_tokens"]

    relative = (mrr_delta / before["accuracy"]["mrr"]) if before["accuracy"]["mrr"] else 0.0
    if abs(relative) < 0.02:
        verdict = ("no measurable change in accuracy: under 2% on one repository is inside "
                   "the noise of this benchmark")
    elif mrr_delta > 0 and (latency_delta > 0 or token_delta > 0):
        verdict = ("accuracy improved, but it cost latency or context. Whether that is worth "
                   "it is a decision, not a measurement")
    elif mrr_delta > 0:
        verdict = "accuracy improved at no cost on the other two axes"
    else:
        verdict = "accuracy fell"

    return {
        "ok": True,
        "before": before["memscore"],
        "after": after["memscore"],
        "delta": {"mrr": round(mrr_delta, 4), "mrr_relative": round(relative, 3),
                  "p50_ms": round(latency_delta, 1), "mean_tokens": token_delta},
        "verdict": verdict,
    }
