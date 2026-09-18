"""Truncation at write time, and rebuilding what it destroyed.

The old composer sliced the composed body at CONTEXT_CHARS and stored the
result, so 81% of this repository's memories were cut mid-word before they were
ever inserted. These tests pin both halves of the fix: the composer must never
sever a claim again, and the bodies already cut must be restorable from the
event log.
"""

from __future__ import annotations

import json

from conftest import record_baseline

from icn import compiler, recover
from icn.db import one, rows


# ------------------------------------------------------------- the write path

def test_there_is_no_write_time_length_cap():
    """Storage is not the scarce resource; truncating on write destroyed text."""
    assert compiler.CONTEXT_CHARS is None


def test_a_long_claim_is_never_cut():
    """The claim is the knowledge. It is stored whole, however long."""
    claim = "x" * 20000
    body = compiler._compose(claim, {
        "occasion": "some occasion",
        "reasoning": "some reasoning",
        "where": ["a.py"],
        "changes": ["changed a thing"],
    })

    assert claim in body, "the claim was severed"
    assert not body.endswith("..."), "a claim must never be stored truncated"


def test_every_context_section_survives_in_full():
    """Nothing composed is dropped or sliced: all of it reaches the database."""
    occasion, reasoning, change = "o" * 2000, "r" * 2000, "c" * 2000
    body = compiler._compose("A short claim.", {
        "occasion": occasion, "reasoning": reasoning,
        "where": ["a.py"], "changes": [change],
    })

    assert body.startswith("A short claim.")
    assert not body.endswith("...")
    for expected in (occasion, reasoning, change, "a.py"):
        assert expected in body


def test_a_cap_if_ever_restored_drops_sections_not_sentences(monkeypatch):
    """The guard behind the guard: a future cap must still never sever a claim."""
    monkeypatch.setattr(compiler, "CONTEXT_CHARS", 200)
    claim = "A claim that is itself longer than the cap. " * 10
    body = compiler._compose(claim, {
        "occasion": "o" * 500, "reasoning": "r" * 500, "where": ["a.py"], "changes": [],
    })

    assert body == claim.strip(), "the claim must survive a cap intact and alone"
    assert not body.endswith("...")


def test_a_body_that_fits_is_left_exactly_as_it_was():
    body = compiler._compose("A claim.", {
        "occasion": "An occasion.", "reasoning": "", "where": [], "changes": [],
    })
    assert body == "A claim.\n\nRecorded while: An occasion."


# ---------------------------------------------------------------- the rebuild

def _truncate_in_place(ws, memory_id: str, limit: int = 120) -> str:
    """Store a body cut the way the old composer cut it."""
    memory = one(ws.store.execute(
        "SELECT body FROM memories WHERE memory_id = ?", (memory_id,)))
    cut = memory["body"][:limit].rstrip() + "..."
    ws.store.execute("UPDATE memories SET body = ? WHERE memory_id = ?", (cut, memory_id))
    ws.store.commit()
    return cut


def test_a_truncated_body_is_rebuilt_from_its_event(workspace):
    result = record_baseline(workspace)
    target = result["memories_created"][0]["memory_id"]
    full = one(workspace.store.execute(
        "SELECT body FROM memories WHERE memory_id = ?", (target,)))["body"]
    cut = _truncate_in_place(workspace, target)
    assert cut != full

    report = recover.rebuild(workspace.store)

    assert report["ok"] and report["rebuilt"] >= 1
    restored = one(workspace.store.execute(
        "SELECT body FROM memories WHERE memory_id = ?", (target,)))["body"]
    assert not restored.endswith("...")
    assert len(restored) > len(cut)
    assert restored == full, "the rebuild must reproduce the original composition"


def test_the_truncated_text_is_kept_not_overwritten(workspace):
    """A rebuild that cannot be inspected afterwards is not safe to run."""
    result = record_baseline(workspace)
    target = result["memories_created"][0]["memory_id"]
    cut = _truncate_in_place(workspace, target)

    recover.rebuild(workspace.store)

    versions = rows(workspace.store.execute(
        "SELECT body, changed_by FROM memory_versions WHERE memory_id = ?", (target,)))
    assert any(v["body"] == cut and v["changed_by"] == "recover" for v in versions), \
        "the truncated text must survive in memory_versions"


def test_a_dry_run_reports_without_writing(workspace):
    result = record_baseline(workspace)
    target = result["memories_created"][0]["memory_id"]
    cut = _truncate_in_place(workspace, target)

    report = recover.rebuild(workspace.store, dry_run=True)

    assert report["dry_run"] and report["candidates"] >= 1
    assert report["recovered_chars"] > 0
    unchanged = one(workspace.store.execute(
        "SELECT body FROM memories WHERE memory_id = ?", (target,)))["body"]
    assert unchanged == cut, "a dry run must not write"


def test_an_untruncated_store_is_left_alone(workspace):
    record_baseline(workspace)
    before = rows(workspace.store.execute("SELECT memory_id, body FROM memories"))

    report = recover.rebuild(workspace.store)

    assert report["rebuilt"] == 0
    after = rows(workspace.store.execute("SELECT memory_id, body FROM memories"))
    assert before == after


def test_a_memory_whose_event_is_gone_is_skipped_not_mangled(workspace):
    """Without the event there is no source text, so the body must be left as is."""
    result = record_baseline(workspace)
    target = result["memories_created"][0]["memory_id"]
    cut = _truncate_in_place(workspace, target)
    workspace.store.execute("DELETE FROM events")
    workspace.store.commit()

    report = recover.rebuild(workspace.store)

    assert report["rebuilt"] == 0
    assert one(workspace.store.execute(
        "SELECT body FROM memories WHERE memory_id = ?", (target,)))["body"] == cut


# --------------------------------------------------------------- reading cost

def test_a_stable_anchor_reports_no_history_at_all(workspace):
    """Nothing moved is not news, and saying so per anchor is what cost 12 KB."""
    result = record_baseline(workspace)
    memory = compiler.get_memory(workspace.store, result["memories_created"][0]["memory_id"])

    assert memory["anchors"], "expected anchors to exist"
    for anchor in memory["anchors"]:
        assert "reanchor_history" not in anchor, "the raw log must never ride along"
        assert "reanchor" not in anchor or anchor["reanchor"]


def test_a_moved_anchor_is_summarised_in_one_line(workspace):
    summary = compiler._reanchor_summary(json.dumps([
        {"at": "2026-01-01T00:00:00Z", "transition": "unchanged", "repeats": 9},
        {"at": "2026-01-02T00:00:00Z", "transition": "body_changed"},
    ]))

    assert isinstance(summary, str)
    assert "body_changed" in summary
    assert "unchanged x9" not in summary, "a stable stretch is not the story"


def test_an_empty_history_summarises_to_nothing():
    assert compiler._reanchor_summary(None) is None
    assert compiler._reanchor_summary("[]") is None
    assert compiler._reanchor_summary(json.dumps(
        [{"at": "x", "transition": "unchanged", "repeats": 40}])) is None


def test_rebuilding_twice_changes_nothing_the_second_time(workspace):
    result = record_baseline(workspace)
    target = result["memories_created"][0]["memory_id"]
    _truncate_in_place(workspace, target)

    first = recover.rebuild(workspace.store)
    body_after_first = one(workspace.store.execute(
        "SELECT body FROM memories WHERE memory_id = ?", (target,)))["body"]
    second = recover.rebuild(workspace.store)

    assert first["rebuilt"] >= 1
    assert second["rebuilt"] == 0, "a rebuild must be idempotent"
    assert one(workspace.store.execute(
        "SELECT body FROM memories WHERE memory_id = ?", (target,)))["body"] == body_after_first
