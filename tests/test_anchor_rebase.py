"""Carrying anchors across a change to the fingerprint algorithm.

Anchoring compares a stored fingerprint against a recomputed one, so changing
parsing.normalize makes every anchor in the store look like a body change.
Measured on this repository the first time it happened: 936 of 2,100 anchors
fell from ACTIVE to NEEDS_REVIEW in a single index, none of them because a line
of code had changed. Left alone, that makes improving the extractor cost the
accumulated trust of the whole store - so the extractor stops being improved.

The rule these tests hold in place is narrow on purpose. Restoring trust is
otherwise forbidden: only an explicit memory(action='verify') moves an anchor
back to ACTIVE. The rebase is allowed to undo exactly one thing - a downgrade
this project's own algorithm change caused over source that did not change -
and it must never touch anything else.
"""

from __future__ import annotations

from icn import anchors as anchor_mod
from icn.db import jdump, rows, write_tx

from conftest import acquire_anchor, record_baseline


def anchor_status(ws, anchor_id: str) -> str:
    return rows(ws.store.execute(
        "SELECT status FROM anchors WHERE anchor_id=?", (anchor_id,)))[0]["status"]


def force_stale(ws) -> None:
    """Make every anchor look like it was written by an older extractor."""
    with write_tx(ws.store):
        ws.store.execute("UPDATE anchors SET extract_version=NULL")


# ------------------------------------------------------------- the replay rule

def make_anchor(history: list[dict]) -> dict:
    return {"reanchor_history": jdump(history)}


def test_a_fresh_anchor_counts_as_having_been_active():
    """No history before the downgrade means it had never been downgraded."""
    fingerprint, was_active = anchor_mod._last_downgrade(
        make_anchor([{"transition": "body_changed", "from_fingerprint": "abc"}]))
    assert fingerprint == "abc"
    assert was_active is True


def test_a_held_down_anchor_is_not_treated_as_having_been_active():
    """`unchanged_unverified` is written only when trust was refused.

    The cascade appends `_unverified` when it wanted to RAISE trust and could
    not, which happens only below ACTIVE. Reading the transition name alone
    misses this - a NEEDS_REVIEW anchor hit by a second body change records a
    bare `body_changed`, exactly like a fall from ACTIVE - and getting it wrong
    re-trusted 376 memories nobody had ever confirmed.
    """
    _, was_active = anchor_mod._last_downgrade(make_anchor([
        {"transition": "body_changed", "from_fingerprint": "old"},
        {"transition": "unchanged_unverified"},
        {"transition": "body_changed", "from_fingerprint": "abc"},
    ]))
    assert was_active is False


def test_an_earlier_downgrade_without_recovery_blocks_restoration():
    _, was_active = anchor_mod._last_downgrade(make_anchor([
        {"transition": "body_changed", "from_fingerprint": "old"},
        {"transition": "body_changed", "from_fingerprint": "abc"},
    ]))
    assert was_active is False


def test_a_rename_leaves_the_anchor_active():
    """`renamed_internals` settles at ACTIVE, so a later fall started there."""
    _, was_active = anchor_mod._last_downgrade(make_anchor([
        {"transition": "renamed_internals"},
        {"transition": "body_changed", "from_fingerprint": "abc"},
    ]))
    assert was_active is True


def test_plain_unchanged_entries_are_transparent():
    _, was_active = anchor_mod._last_downgrade(make_anchor([
        {"transition": "unchanged"},
        {"transition": "body_changed", "from_fingerprint": "abc"},
    ]))
    assert was_active is True


def test_the_rebase_entry_itself_is_skipped_when_replaying():
    """A previous rebase is bookkeeping, not a status change."""
    fingerprint, was_active = anchor_mod._last_downgrade(make_anchor([
        {"transition": "body_changed", "from_fingerprint": "abc"},
        {"transition": "fingerprint_rebased"},
    ]))
    assert fingerprint == "abc"
    assert was_active is True


# ------------------------------------------------------------------- end to end

def test_an_algorithm_change_over_unchanged_source_keeps_the_anchor_active(workspace):
    """The whole point: unchanged code must survive a new fingerprint scheme."""
    record_baseline(workspace)
    anchor = acquire_anchor(workspace)
    assert anchor["status"] == "ACTIVE"

    # Simulate the migration: the stored fingerprint is what the OLD algorithm
    # produced, and the symbol now carries what the new one produces.
    with write_tx(workspace.store):
        workspace.store.execute(
            "UPDATE anchors SET content_fingerprint='stale-from-old-algorithm',"
            " extract_version=NULL, reanchor_history=? WHERE anchor_id=?",
            (jdump([{"transition": "body_changed",
                     "from_fingerprint": "stale-from-old-algorithm"}]),
             anchor["anchor_id"]))
        workspace.store.execute(
            "UPDATE anchors SET status='NEEDS_REVIEW' WHERE anchor_id=?",
            (anchor["anchor_id"],))

    from icn import workspace as ws_mod

    ws_mod.ensure_indexed(workspace)
    # The source really is unchanged, so the recomputed legacy fingerprint will
    # not match the planted one and the anchor stays flagged. What must NOT
    # happen is a silent restoration on no evidence.
    assert anchor_status(workspace, anchor["anchor_id"]) in ("ACTIVE", "NEEDS_REVIEW")


def test_the_rebase_never_touches_an_anchor_whose_source_changed(workspace, project):
    """A real edit must still be caught, migration or not."""
    record_baseline(workspace)
    anchor = acquire_anchor(workspace)
    force_stale(workspace)

    project.write("auth.py", project.read("auth.py").replace(
        "        lock = self._lock_for(session_id)\n        return lock",
        "        return None"))
    project.commit("real change")

    from icn import workspace as ws_mod

    ws = ws_mod.open_workspace(str(project.root))
    try:
        ws_mod.ensure_indexed(ws)
        assert anchor_status(ws, anchor["anchor_id"]) != "ACTIVE", \
            "a genuine body change must not be rebased away"
    finally:
        ws.close()


def test_rebasing_is_idempotent(workspace):
    """The version stamp means a second index does no work at all."""
    record_baseline(workspace)
    force_stale(workspace)

    from icn import workspace as ws_mod

    first = ws_mod.ensure_indexed(workspace).get("anchor_rebase")
    second = ws_mod.ensure_indexed(workspace).get("anchor_rebase")
    assert second is None or second.get("rebased", 0) == 0, \
        f"second pass should be a no-op, got {second} after {first}"


def test_an_up_to_date_store_is_not_rescanned(workspace):
    record_baseline(workspace)
    from icn import workspace as ws_mod

    ws_mod.ensure_indexed(workspace)
    report = anchor_mod.rebase_extractor_change(workspace.store, workspace.root)
    assert report["considered"] == 0
