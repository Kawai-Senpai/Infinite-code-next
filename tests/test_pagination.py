"""Paging through stored knowledge, and keeping open() small enough to read.

A repository with thousands of anchors used to put every re-anchored record
inside workspace(action='open'): 448KB of per-anchor detail on a real store,
which overflowed the tool-result limit and buried the briefing the call exists
to deliver. The detail is capped now, and the listing it points at pages.
"""

from __future__ import annotations

import pathlib

import pytest

from icn import compiler, db, ids
from icn.compiler import now


@pytest.fixture()
def store(tmp_path: pathlib.Path):
    conn = db.init_repo_store(tmp_path / "store.db")
    for i in range(25):
        memory_id = ids.new_id("mem")
        conn.execute(
            "INSERT INTO memories (memory_id, kind, title, body, severity, status,"
            " authority, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (memory_id, "invariant", f"rule {i:02d}", "b", "medium", "ACTIVE", "agent", now()))
        conn.execute(
            "INSERT INTO anchors (anchor_id, memory_id, target_kind, status,"
            " anchor_confidence, symbol_path, created_at) VALUES (?,?,?,?,?,?,?)",
            (ids.new_id("anc"), memory_id, "symbol",
             "ACTIVE" if i % 5 else "ORPHANED", 0.9, f"s{i}", now()))
    conn.commit()
    yield conn
    conn.close()


def test_paging_walks_every_memory_exactly_once(store):
    """The property that matters: no row seen twice, none skipped."""
    total = compiler.count_memories(store)
    assert total == 25

    seen: list[str] = []
    offset = 0
    while offset < total:
        page = compiler.list_memories(store, limit=7, offset=offset)
        assert page, "a page before the total must not be empty"
        seen += [row["memory_id"] for row in page]
        offset += len(page)

    assert len(seen) == total
    assert len(set(seen)) == total


def test_paging_past_the_end_is_empty_not_an_error(store):
    assert compiler.list_memories(store, limit=7, offset=25) == []
    assert compiler.list_memories(store, limit=7, offset=10_000) == []


def test_a_negative_offset_is_clamped_rather_than_wrapping(store):
    """SQLite treats a negative OFFSET as no offset; be explicit about it."""
    assert compiler.list_memories(store, limit=5, offset=-3) == \
        compiler.list_memories(store, limit=5, offset=0)


def test_the_count_agrees_with_the_rows_under_every_filter(store):
    for kwargs in ({}, {"kind": "invariant"}, {"kind": "warning"},
                   {"status": "ACTIVE"}, {"anchor_status": "ORPHANED"},
                   {"anchor_status": "ACTIVE"}):
        rows = compiler.list_memories(store, limit=500, **kwargs)
        assert compiler.count_memories(store, **kwargs) == len(rows), kwargs


def test_an_unknown_anchor_status_counts_zero_rather_than_raising(store):
    assert compiler.count_memories(store, anchor_status="NOT_A_STATUS") == 0
    assert compiler.list_memories(store, anchor_status="NOT_A_STATUS") == []


def test_verify_repo_caps_the_detail_it_reports(monkeypatch, tmp_path):
    """open() must stay small no matter how many anchors moved."""
    from icn import anchors as anchor_mod

    moved = [{"anchor_id": f"anc_{i}", "memory_id": f"mem_{i}", "status": "NEEDS_REVIEW",
              "transition": "body_changed", "symbol_path": f"s{i}"} for i in range(1503)]
    monkeypatch.setattr(anchor_mod, "verify_anchor",
                        lambda conn, anchor, root, commit: moved[int(anchor["anchor_id"][4:])])

    conn = db.init_repo_store(tmp_path / "s.db")
    for i in range(1503):
        conn.execute(
            "INSERT INTO anchors (anchor_id, memory_id, target_kind, status, created_at)"
            " VALUES (?,?,?,?,?)",
            (f"anc_{i}", f"mem_{i}", "symbol", "ACTIVE", now()))
    conn.commit()

    report = anchor_mod.verify_repo(conn, tmp_path, "abc123")
    conn.close()

    # The counts survive in full; only the per-anchor rows are trimmed.
    assert report["checked"] == 1503
    assert report["changed_total"] == 1503
    assert len(report["changed"]) == anchor_mod._CHANGED_DETAIL
    assert report["changed_by_transition"] == {"body_changed": 1503}
    assert "offset" in report["changed_note"]

    import json
    assert len(json.dumps(report)) < 20_000, "open() payload must stay readable"
