"""Semantic anchors and the re-anchor cascade.

This is the milestone the product lives or dies on (PLAN 2 sections 5 and 6).

The design premise, from the prior-art pass: notes that drift out of sync with
the code they describe are actively harmful, and the harm peaks immediately
after the drift appears (arXiv 2409.10781), while memory systems routinely
re-serve invalidated memories without noticing (arXiv 2604.20006). So the
status machine is the product, and the matching cascade is allowed to fail
into NEEDS_REVIEW. A confidently wrong re-anchor is worse than an honest
ORPHANED, and every threshold here is biased accordingly.

Verification fires on the edit that touched the span, not on a timer.
"""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import ids
from .db import jdump, jload, one, rows, write_tx
from .identity import blame_move_evidence, run_git
from .parsing import token_similarity

# Anchor status vocabulary.
ACTIVE = "ACTIVE"
DRIFTED = "DRIFTED"
NEEDS_REVIEW = "NEEDS_REVIEW"
ORPHANED = "ORPHANED"
SUPERSEDED = "SUPERSEDED"
RESOLVED = "RESOLVED"

STALE_STATUSES = {DRIFTED, NEEDS_REVIEW, ORPHANED}

# Confidence floors. Below MIGRATION_MIN we refuse to move a memory at all and
# record a candidate instead (PLAN.md section 12).
RENAME_CONFIDENCE = 0.80
MOVE_CONFIDENCE = 0.90
BODY_CHANGED_CONFIDENCE = 0.55
SIMILARITY_ACCEPT = 0.70
MIGRATION_MIN = 0.45
COMMIT_WINDOW = 20


def now() -> str:
    # Milliseconds, not seconds. Ordering questions ("did this caller
    # appear after that memory was verified?") are decided by comparing
    # these, and second precision made same-second events compare equal,
    # so a genuinely late caller went unreported.
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def create_anchor(conn: sqlite3.Connection, memory_id: str, symbol: dict[str, Any] | None,
                  file_row: dict[str, Any] | None, commit: str | None,
                  target_kind: str = "symbol") -> str:
    """Attach a memory to a code target, fingerprinted as of `commit`.

    The extractor version is stamped now, so a later algorithm change can tell
    this anchor's fingerprints apart from ones it has already carried across.
    Without the stamp every index rescans every anchor forever.
    """
    from . import parsing

    anchor_id = ids.new_id(ids.ANCHOR)
    if symbol is not None:
        conn.execute(
            "INSERT INTO anchors (anchor_id, memory_id, target_kind, symbol_id, file_id, file_path,"
            " symbol_path, ast_path, range_start_byte, range_end_byte, line_start, line_end,"
            " content_fingerprint, skeleton_fingerprint, prev_fingerprint, next_fingerprint,"
            " commit_observed, anchor_confidence, status, last_verified_commit, last_verified_at,"
            " reanchor_history, created_at, extract_version)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1.0,?,?,?,?,?,?)",
            (anchor_id, memory_id, "symbol", symbol["symbol_id"], symbol["file_id"],
             symbol.get("last_known_path"), symbol["symbol_path"], symbol.get("ast_path"),
             symbol.get("start_byte"), symbol.get("end_byte"), symbol.get("line_start"),
             symbol.get("line_end"), symbol.get("content_fingerprint"),
             symbol.get("skeleton_fingerprint"), symbol.get("prev_fingerprint"),
             symbol.get("next_fingerprint"), commit, ACTIVE, commit, now(), jdump([]), now(),
             parsing.EXTRACT_VERSION),
        )
    elif file_row is not None:
        conn.execute(
            "INSERT INTO anchors (anchor_id, memory_id, target_kind, file_id, file_path,"
            " commit_observed, anchor_confidence, status, last_verified_commit, last_verified_at,"
            " reanchor_history, created_at) VALUES (?,?,?,?,?,?,1.0,?,?,?,?,?)",
            (anchor_id, memory_id, "file", file_row["file_id"], file_row["path"], commit,
             ACTIVE, commit, now(), jdump([]), now()),
        )
    else:
        conn.execute(
            "INSERT INTO anchors (anchor_id, memory_id, target_kind, commit_observed,"
            " anchor_confidence, status, last_verified_commit, last_verified_at, reanchor_history,"
            " created_at) VALUES (?,?,?,?,1.0,?,?,?,?,?)",
            (anchor_id, memory_id, target_kind, commit, ACTIVE, commit, now(), jdump([]), now()),
        )
    return anchor_id


def _record_transition(anchor: dict[str, Any], kind: str, detail: dict[str, Any]) -> str:
    """Append one transition, keeping only what is worth auditing.

    A no-op verification is not history. Recording every "unchanged" pass fills
    the ring buffer with identical entries and evicts the real re-anchors it
    exists to preserve - measured live at 25 consecutive "unchanged" rows on a
    single anchor, which is also dead weight in every memory(get) response.
    """
    history = jload(anchor.get("reanchor_history"), []) or []

    if kind == "unchanged":
        # Keep one, as evidence the anchor is being checked, and refresh its
        # timestamp rather than growing the list.
        if history and history[-1].get("transition") == "unchanged":
            history[-1]["at"] = now()
            history[-1]["repeats"] = int(history[-1].get("repeats", 1)) + 1
            return jdump(history[-25:])

    history.append({
        "at": now(),
        "transition": kind,
        "from_symbol_path": anchor.get("symbol_path"),
        "from_symbol_id": anchor.get("symbol_id"),
        "from_fingerprint": anchor.get("content_fingerprint"),
        **detail,
    })
    return jdump(history[-25:])


# Trust ordering. The cascade may move an anchor down this list, never up.
_TRUST_RANK = {ACTIVE: 0, DRIFTED: 1, NEEDS_REVIEW: 2, ORPHANED: 3, RESOLVED: 4, SUPERSEDED: 5}


def _apply(conn: sqlite3.Connection, anchor: dict[str, Any], symbol: dict[str, Any] | None,
           status: str, confidence: float, transition: str, detail: dict[str, Any],
           commit: str | None) -> dict[str, Any]:
    """Write a cascade outcome. Never deletes; the previous anchor is retained
    in reanchor_history so a bad re-anchor is auditable (PLAN.md section 11).

    The cascade can only lower trust. Once an anchor is DRIFTED or
    NEEDS_REVIEW, later passes find its freshly re-anchored fingerprint
    matching and would otherwise report "unchanged" and quietly restore ACTIVE
    - re-trusting a memory nobody ever confirmed. Only an explicit
    memory(action='verify') returns an anchor to ACTIVE.
    """
    previous = anchor.get("status") or ACTIVE
    if _TRUST_RANK.get(status, 0) < _TRUST_RANK.get(previous, 0):
        status = previous
        confidence = min(confidence, anchor.get("anchor_confidence") or confidence)
        detail = {**detail, "held_at": previous,
                  "note": "cascade does not restore trust; confirm with memory(action='verify')"}
        transition = f"{transition}_unverified"

    history = _record_transition(anchor, transition, detail)
    if symbol is not None:
        conn.execute(
            "UPDATE anchors SET symbol_id=?, file_id=?, file_path=?, symbol_path=?, ast_path=?,"
            " range_start_byte=?, range_end_byte=?, line_start=?, line_end=?, content_fingerprint=?,"
            " skeleton_fingerprint=?, prev_fingerprint=?, next_fingerprint=?, anchor_confidence=?,"
            " status=?, last_verified_commit=?, last_verified_at=?, reanchor_history=?"
            " WHERE anchor_id=?",
            (symbol["symbol_id"], symbol["file_id"], symbol.get("last_known_path"),
             symbol["symbol_path"], symbol.get("ast_path"), symbol.get("start_byte"),
             symbol.get("end_byte"), symbol.get("line_start"), symbol.get("line_end"),
             symbol.get("content_fingerprint"), symbol.get("skeleton_fingerprint"),
             symbol.get("prev_fingerprint"), symbol.get("next_fingerprint"),
             confidence, status, commit, now(), history, anchor["anchor_id"]),
        )
    else:
        conn.execute(
            "UPDATE anchors SET anchor_confidence=?, status=?, last_verified_commit=?,"
            " last_verified_at=?, reanchor_history=? WHERE anchor_id=?",
            (confidence, status, commit, now(), history, anchor["anchor_id"]),
        )
    return {
        "anchor_id": anchor["anchor_id"],
        "memory_id": anchor["memory_id"],
        "status": status,
        "confidence": round(confidence, 3),
        "transition": transition,
        "symbol_path": symbol["symbol_path"] if symbol else anchor.get("symbol_path"),
        **detail,
    }


def verify_anchor(conn: sqlite3.Connection, anchor: dict[str, Any], root: Path,
                  commit: str | None) -> dict[str, Any]:
    """Run the cascade for one anchor. Cheapest test first, short-circuiting."""
    if anchor["target_kind"] == "file":
        return _verify_file_anchor(conn, anchor, commit)
    if anchor["target_kind"] not in ("symbol",):
        return {"anchor_id": anchor["anchor_id"], "status": anchor["status"],
                "confidence": anchor["anchor_confidence"], "transition": "skipped"}

    fingerprint = anchor["content_fingerprint"]
    symbol_path = anchor["symbol_path"]

    # --- Step 1: exact content, same place. The overwhelmingly common case.
    if fingerprint:
        exact = one(conn.execute(
            "SELECT * FROM symbols WHERE status='ACTIVE' AND content_fingerprint=? AND symbol_path=?",
            (fingerprint, symbol_path),
        ))
        if exact:
            return _apply(conn, anchor, exact, ACTIVE, 1.0, "unchanged", {}, commit)

    # --- Step 2: exact content, somewhere else. A move.
    if fingerprint:
        moved = one(conn.execute(
            "SELECT * FROM symbols WHERE status='ACTIVE' AND content_fingerprint=? AND symbol_path!=?"
            " LIMIT 1",
            (fingerprint, symbol_path),
        ))
        if moved:
            evidence = False
            if moved.get("last_known_path") and moved.get("line_start"):
                evidence = blame_move_evidence(
                    root, moved["last_known_path"], moved["line_start"],
                    min(moved["line_end"] or moved["line_start"], (moved["line_start"] or 1) + 40),
                )
            return _apply(conn, anchor, moved, ACTIVE, MOVE_CONFIDENCE, "moved",
                          {"from": symbol_path, "to": moved["symbol_path"],
                           "git_move_evidence": bool(evidence)}, commit)

    # --- Step 3: same place, different content.
    same_path = one(conn.execute(
        "SELECT * FROM symbols WHERE status='ACTIVE' AND symbol_path=?", (symbol_path,)
    ))
    if same_path:
        if anchor["skeleton_fingerprint"] and same_path["skeleton_fingerprint"] == anchor["skeleton_fingerprint"]:
            # Structure identical, identifiers changed: a rename, which is
            # behavior-preserving, so the memory almost certainly still applies.
            return _apply(conn, anchor, same_path, ACTIVE, RENAME_CONFIDENCE, "renamed_internals",
                          {"structure": "identical"}, commit)
        # The body genuinely changed. This is the moment the research says the
        # harm is highest, so flag it now rather than decaying to it later.
        return _apply(conn, anchor, same_path, NEEDS_REVIEW, BODY_CHANGED_CONFIDENCE,
                      "body_changed", {"structure": "changed"}, commit)

    # --- Step 4: the symbol is gone. Look for where it went.
    candidate, score = _best_candidate(conn, anchor)
    if candidate and score >= SIMILARITY_ACCEPT:
        moved_evidence = _rename_evidence(root, anchor.get("file_path"), candidate.get("last_known_path"))
        return _apply(conn, anchor, candidate, DRIFTED, max(0.7, min(score, 0.85)), "migrated",
                      {"from": symbol_path, "to": candidate["symbol_path"],
                       "similarity": round(score, 3), "git_rename_evidence": moved_evidence}, commit)

    # --- Step 5 would be the embedding fallback. Not available by default.
    # --- Step 6: nothing clears the bar. Keep the memory, mark it honestly.
    detail: dict[str, Any] = {"searched": "fingerprint+skeleton+similarity"}
    if candidate and score >= MIGRATION_MIN:
        # Too weak to move a memory on. Record the lead, do not act on it.
        _record_migration_candidate(conn, anchor, candidate, score)
        detail["possible_migration"] = {
            "to": candidate["symbol_path"], "similarity": round(score, 3),
            "note": "not transferred automatically",
        }
    return _apply(conn, anchor, None, ORPHANED, 0.0, "orphaned", detail, commit)


def _verify_file_anchor(conn: sqlite3.Connection, anchor: dict[str, Any],
                        commit: str | None) -> dict[str, Any]:
    row = one(conn.execute("SELECT * FROM files WHERE file_id=?", (anchor["file_id"],)))
    if row and row["status"] == "ACTIVE":
        return _apply(conn, anchor, None, ACTIVE, 1.0, "unchanged", {}, commit)
    renamed = one(conn.execute(
        "SELECT * FROM files WHERE status='ACTIVE' AND content_hash=(SELECT content_hash FROM files"
        " WHERE file_id=?) AND file_id!=?", (anchor["file_id"], anchor["file_id"])
    ))
    if renamed:
        conn.execute("UPDATE anchors SET file_id=?, file_path=? WHERE anchor_id=?",
                     (renamed["file_id"], renamed["path"], anchor["anchor_id"]))
        return _apply(conn, anchor, None, ACTIVE, MOVE_CONFIDENCE, "file_moved",
                      {"to": renamed["path"]}, commit)
    return _apply(conn, anchor, None, ORPHANED, 0.0, "file_deleted", {}, commit)


def _best_candidate(conn: sqlite3.Connection, anchor: dict[str, Any]) -> tuple[dict[str, Any] | None, float]:
    """Find where a vanished symbol most plausibly went.

    Candidates are drawn from the whole active symbol table, not from the last
    commit's diff. Refactoring-detection work found Move refactorings often
    only become visible across several commits (arXiv 2204.11276), so a
    HEAD~1-only search would systematically miss the very cases that move a
    memory.
    """
    skeleton = anchor.get("skeleton_fingerprint")
    tokens = None
    if anchor.get("symbol_id"):
        prior = one(conn.execute("SELECT token_signature FROM symbols WHERE symbol_id=?",
                                 (anchor["symbol_id"],)))
        tokens = prior["token_signature"] if prior else None

    leaf = (anchor.get("symbol_path") or "").rsplit(".", 1)[-1]
    candidates: list[dict[str, Any]] = []

    if skeleton:
        candidates.extend(rows(conn.execute(
            "SELECT * FROM symbols WHERE status='ACTIVE' AND skeleton_fingerprint=? LIMIT 25",
            (skeleton,),
        )))
    if leaf:
        candidates.extend(rows(conn.execute(
            "SELECT * FROM symbols WHERE status='ACTIVE' AND name=? LIMIT 25", (leaf,),
        )))
    if anchor.get("file_id"):
        candidates.extend(rows(conn.execute(
            "SELECT * FROM symbols WHERE status='ACTIVE' AND file_id=? LIMIT 60",
            (anchor["file_id"],),
        )))

    best: dict[str, Any] | None = None
    best_score = 0.0
    seen: set[str] = set()
    for candidate in candidates:
        if candidate["symbol_id"] in seen:
            continue
        seen.add(candidate["symbol_id"])
        score = 0.0
        if skeleton and candidate["skeleton_fingerprint"] == skeleton:
            score = max(score, 0.85)          # identical structure, renamed
        if leaf and candidate["name"] == leaf:
            score = max(score, 0.72)          # same name, moved
        if tokens and candidate["token_signature"]:
            score = max(score, token_similarity(tokens, candidate["token_signature"]))
        if score > best_score:
            best, best_score = candidate, score
    return best, best_score


def _record_migration_candidate(conn: sqlite3.Connection, anchor: dict[str, Any],
                                candidate: dict[str, Any], score: float) -> None:
    """PLAN.md section 12: an uncertain migration is a lead, not a fact."""
    conn.execute(
        "INSERT INTO memory_edges (edge_id, from_id, to_id, kind, edge_class, status, confidence,"
        " source, evidence, created_at) VALUES (?,?,?,'POSSIBLY_MIGRATED_TO','inferred','ACTIVE',?,"
        "'reanchor-cascade',?,?)"
        " ON CONFLICT(from_id, to_id, kind) DO UPDATE SET confidence=excluded.confidence",
        (ids.new_id(ids.EDGE), anchor.get("symbol_id") or anchor["anchor_id"],
         candidate["symbol_id"], round(score, 3),
         jdump({"anchor_id": anchor["anchor_id"], "from": anchor.get("symbol_path")}), now()),
    )


def _rename_evidence(root: Path, old_path: str | None, new_path: str | None) -> bool:
    """Ask git whether it saw a rename between two paths in a recent window."""
    if not old_path or not new_path or old_path == new_path:
        return False
    code, out, _ = run_git(
        ["log", f"-{COMMIT_WINDOW}", "--diff-filter=R", "--name-status", "--find-renames", "--", new_path],
        root,
    )
    return code == 0 and old_path in out


def rebase_extractor_change(conn: sqlite3.Connection, root: Path,
                            budget_seconds: float | None = None) -> dict[str, Any]:
    """Carry anchors across a change to the FINGERPRINT ALGORITHM, not the code.

    Anchoring compares a stored fingerprint against a recomputed one, so when
    parsing.normalize itself changes, every anchor in the repository looks like
    a body change. Measured on this repository the first time it happened: 936
    of 2,100 anchors dropped from ACTIVE to NEEDS_REVIEW in one index, none of
    them because a single line of code had changed.

    That makes improving the extractor cost the accumulated trust of the whole
    store, which in practice means the extractor stops being improved. This is
    the mechanism that decouples the two.

    The test is exact, not a heuristic: recompute the CURRENT source under the
    PREVIOUS algorithm and compare it with what the anchor recorded. If they
    match, the code the anchor points at is byte-for-byte what it was and only
    the hash moved, so the new fingerprint is written and the status is left
    alone. If they do not match, the code really did change and the cascade is
    left to do its job.

    This is the one place trust may be restored without an explicit
    memory(action='verify'), and only in one direction: undoing a downgrade
    this project's own algorithm change caused. It is not a re-verification,
    and it never touches an anchor whose source has actually changed.
    """
    from . import parsing

    pending = rows(conn.execute(
        "SELECT * FROM anchors WHERE target_kind='symbol' AND status != ?"
        " AND (extract_version IS NULL OR extract_version < ?)",
        (SUPERSEDED, parsing.EXTRACT_VERSION)))
    if not pending:
        return {"considered": 0, "rebased": 0, "restored": 0}

    sources: dict[str, bytes | None] = {}

    def source_of(path: str | None) -> bytes | None:
        if not path:
            return None
        if path not in sources:
            try:
                sources[path] = (root / path).read_bytes()
            except OSError:
                sources[path] = None
        return sources[path]

    rebased = restored = 0
    truncated = False
    deadline = (time.monotonic() + budget_seconds) if budget_seconds else None

    def settle(anchor_id: str) -> None:
        """Stamp the version on an anchor this pass can never rebase.

        Without this the `continue` paths below leave extract_version behind
        forever, so the same unrebasable anchors are re-read and re-parsed on
        every single open. Measured on a 7,152-anchor repo: 1,090 anchors
        permanently stuck below the current version, costing 7.6s per open
        with nothing to show for it. Stamping records "this pass considered
        it and had nothing to do", which is exactly true.
        """
        conn.execute("UPDATE anchors SET extract_version=? WHERE anchor_id=?",
                     (parsing.EXTRACT_VERSION, anchor_id))

    with write_tx(conn):
        for anchor in pending:
            if deadline is not None and time.monotonic() > deadline:
                truncated = True
                break
            symbol = one(conn.execute(
                "SELECT s.*, f.path AS file_path FROM symbols s"
                " JOIN files f ON f.file_id = s.file_id"
                " WHERE s.symbol_id=? AND s.status='ACTIVE'", (anchor["symbol_id"],)))
            if not symbol:
                settle(anchor["anchor_id"])
                continue
            data = source_of(symbol["file_path"])
            if data is None:
                settle(anchor["anchor_id"])
                continue

            legacy = _legacy_fingerprint(parsing, data, symbol)
            if legacy is None:
                settle(anchor["anchor_id"])
                continue

            # The anchor's own fingerprint, or - if a previous index already
            # rewrote it - the one it carried before that transition.
            recorded = anchor.get("content_fingerprint")
            previous, fell_from_active = _last_downgrade(anchor)
            if legacy != recorded and legacy != previous:
                # The source genuinely changed, so the cascade owns this one.
                # Stamp it anyway: re-testing it on every open cannot change
                # the answer, because a later real edit reindexes the symbol
                # and the cascade re-evaluates it there.
                settle(anchor["anchor_id"])
                continue

            status = anchor["status"]
            confidence = anchor["anchor_confidence"]
            if (status != ACTIVE and fell_from_active and legacy == previous):
                # This anchor was ACTIVE, and the only thing that happened to
                # it since is an algorithm change over unchanged source. Undo
                # exactly that, and nothing else.
                status, confidence = ACTIVE, 1.0
                restored += 1
            rebased += 1

            history = _record_transition(
                anchor, "fingerprint_rebased",
                {"reason": "extractor algorithm changed; source is unchanged",
                 "restored_to": status if status != anchor["status"] else None})
            conn.execute(
                "UPDATE anchors SET content_fingerprint=?, skeleton_fingerprint=?,"
                " prev_fingerprint=?, next_fingerprint=?, ast_path=?, status=?,"
                " anchor_confidence=?, extract_version=?, reanchor_history=?"
                " WHERE anchor_id=?",
                (symbol["content_fingerprint"], symbol["skeleton_fingerprint"],
                 symbol["prev_fingerprint"], symbol["next_fingerprint"],
                 symbol["ast_path"], status, confidence, parsing.EXTRACT_VERSION,
                 history, anchor["anchor_id"]))

    return {"considered": len(pending), "rebased": rebased, "restored": restored,
            "truncated": truncated}


def _legacy_fingerprint(parsing, data: bytes, symbol: dict[str, Any]) -> str | None:
    """The symbol's content fingerprint under the pre-v5 algorithm."""
    parser = parsing.get_parser(symbol["lang"] or "")
    if parser is None:
        return None
    try:
        tree = parser.parse(data)
    except Exception:
        return None
    node = _node_at(tree.root_node, symbol["start_byte"], symbol["end_byte"])
    if node is None:
        return None
    return parsing._sha("".join(
        parsing.normalize(node, data, True, include_operators=False)))


def _node_at(root, start: int, end: int):
    """The node occupying exactly this byte range, if one still does."""
    stack = [root]
    while stack:
        node = stack.pop()
        if node.start_byte == start and node.end_byte == end:
            return node
        if node.start_byte <= start and node.end_byte >= end:
            stack.extend(node.named_children)
    return None


# Transitions that leave an anchor ACTIVE. Everything else either lowers trust
# or - with an `_unverified` suffix - records that the cascade wanted to raise
# it and was refused, which only happens when the anchor is already below
# ACTIVE. Between them these reconstruct the status history that the anchor
# row itself does not keep.
_ACTIVE_TRANSITIONS = {"unchanged", "moved", "renamed_internals"}


def _last_downgrade(anchor: dict[str, Any]) -> tuple[str | None, bool]:
    """(fingerprint before the last real transition, was the anchor ACTIVE then).

    The second half is what keeps this honest. Only a downgrade that started at
    ACTIVE may be undone; an anchor already below ACTIVE for a real reason must
    stay there, even though the algorithm change re-flagged it too.

    Inferring that from the transition NAME alone does not work, and getting
    this wrong re-trusted 376 memories nobody had ever confirmed. `_apply`
    appends `_unverified` only when the cascade tried to RAISE trust and was
    refused, so a NEEDS_REVIEW anchor hit by a second body change records a
    bare `body_changed` exactly like a fall from ACTIVE. The status has to be
    replayed from the transitions before it instead.
    """
    history = jload(anchor.get("reanchor_history"), []) or []
    index = None
    for position in range(len(history) - 1, -1, -1):
        transition = history[position].get("transition") or ""
        if transition in ("unchanged", "fingerprint_rebased"):
            continue
        index = position
        break
    if index is None:
        return None, False

    # Replay backwards to the most recent transition that settles the status.
    was_active = True
    for entry in reversed(history[:index]):
        transition = entry.get("transition") or ""
        if transition == "fingerprint_rebased":
            continue
        if transition.endswith("_unverified"):
            was_active = False          # held down: it was already below ACTIVE
            break
        if transition in _ACTIVE_TRANSITIONS:
            was_active = True
            break
        was_active = False              # a real downgrade, never verified back
        break
    return history[index].get("from_fingerprint"), was_active


_CHANGED_DETAIL = 20


def verify_repo(conn: sqlite3.Connection, root: Path, commit: str | None,
                only_memory: str | None = None,
                budget_seconds: float | None = None) -> dict[str, Any]:
    """Re-verify anchors. Called right after indexing, so drift is caught on the
    edit that caused it.

    Each anchor is re-read and re-fingerprinted from disk, so the cost scales
    with the number of anchors, not with the size of the edit. `budget_seconds`
    caps the sweep and reports `truncated`, letting the caller answer now and
    finish the rest in the background. Anchors are swept oldest-verified first,
    so a truncated sweep still makes progress rather than rechecking the same
    prefix every time. A scoped `only_memory` check is never truncated.
    """
    sql = "SELECT * FROM anchors WHERE status != ?"
    args: tuple = (SUPERSEDED,)
    if only_memory:
        sql += " AND memory_id = ?"
        args = args + (only_memory,)
    else:
        # Least-recently-verified first, so consecutive truncated sweeps cover
        # the whole store instead of re-checking the same prefix forever.
        sql += " ORDER BY COALESCE(last_verified_at, '') ASC"

    budget = None if only_memory else budget_seconds
    deadline = (time.monotonic() + budget) if budget else None

    results: list[dict[str, Any]] = []
    truncated = False
    with write_tx(conn):
        for anchor in rows(conn.execute(sql, args)):
            if deadline is not None and time.monotonic() > deadline:
                truncated = True
                break
            results.append(verify_anchor(conn, anchor, root, commit))

    summary: dict[str, int] = {}
    for result in results:
        summary[result["status"]] = summary.get(result["status"], 0) + 1
    changed = [r for r in results if r["transition"] not in ("unchanged", "skipped")]

    # A repository with thousands of anchors produced thousands of these
    # records, and they travelled inside every workspace(action='open')
    # response - hundreds of kilobytes of per-anchor detail that buried the
    # briefing the call exists to deliver. The counts carry the signal; the
    # individual rows are recoverable with memory(action='reanchor'), which
    # is scoped to one memory and so is never truncated.
    detail = changed if only_memory else changed[:_CHANGED_DETAIL]
    report = {"checked": len(results), "by_status": summary, "changed": detail,
              "changed_total": len(changed), "truncated": truncated}
    if len(detail) < len(changed):
        by_transition: dict[str, int] = {}
        for entry in changed:
            key = entry["transition"]
            by_transition[key] = by_transition.get(key, 0) + 1
        report["changed_by_transition"] = by_transition
        report["changed_note"] = (
            f"showing {len(detail)} of {len(changed)} changed anchors; page the rest with "
            "memory(action='list', anchor_status='NEEDS_REVIEW', limit=..., offset=...), "
            "or memory(action='reanchor', memory_id=...) for one memory's detail")
    return report


def anchor_health(conn: sqlite3.Connection) -> dict[str, Any]:
    return {
        r["status"]: r["n"]
        for r in rows(conn.execute("SELECT status, COUNT(*) AS n FROM anchors GROUP BY status"))
    }


def mark_verified(conn: sqlite3.Connection, memory_id: str, commit: str | None,
                  actor: str = "agent") -> dict[str, Any]:
    """An agent or human confirms a memory still holds. Clears review flags."""
    with write_tx(conn):
        conn.execute(
            "UPDATE anchors SET status=?, anchor_confidence=MAX(anchor_confidence, 0.9),"
            " last_verified_commit=?, last_verified_at=? WHERE memory_id=? AND status IN (?,?)",
            (ACTIVE, commit, now(), memory_id, NEEDS_REVIEW, DRIFTED),
        )
        conn.execute(
            "UPDATE memories SET last_verified_commit=?, last_verified_at=? WHERE memory_id=?",
            (commit, now(), memory_id),
        )
        conn.execute(
            "INSERT INTO corrections (correction_id, memory_id, action, reason, actor, created_at)"
            " VALUES (?,?,'verify','confirmed still applies',?,?)",
            (ids.new_id("cor"), memory_id, actor, now()),
        )
    return {"ok": True, "memory_id": memory_id, "verified_at_commit": commit}
