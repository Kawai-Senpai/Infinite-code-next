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
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def create_anchor(conn: sqlite3.Connection, memory_id: str, symbol: dict[str, Any] | None,
                  file_row: dict[str, Any] | None, commit: str | None,
                  target_kind: str = "symbol") -> str:
    """Attach a memory to a code target, fingerprinted as of `commit`."""
    anchor_id = ids.new_id(ids.ANCHOR)
    if symbol is not None:
        conn.execute(
            "INSERT INTO anchors (anchor_id, memory_id, target_kind, symbol_id, file_id, file_path,"
            " symbol_path, ast_path, range_start_byte, range_end_byte, line_start, line_end,"
            " content_fingerprint, skeleton_fingerprint, prev_fingerprint, next_fingerprint,"
            " commit_observed, anchor_confidence, status, last_verified_commit, last_verified_at,"
            " reanchor_history, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1.0,?,?,?,?,?)",
            (anchor_id, memory_id, "symbol", symbol["symbol_id"], symbol["file_id"],
             symbol.get("last_known_path"), symbol["symbol_path"], symbol.get("ast_path"),
             symbol.get("start_byte"), symbol.get("end_byte"), symbol.get("line_start"),
             symbol.get("line_end"), symbol.get("content_fingerprint"),
             symbol.get("skeleton_fingerprint"), symbol.get("prev_fingerprint"),
             symbol.get("next_fingerprint"), commit, ACTIVE, commit, now(), jdump([]), now()),
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


def verify_repo(conn: sqlite3.Connection, root: Path, commit: str | None,
                only_memory: str | None = None) -> dict[str, Any]:
    """Re-verify anchors. Called right after indexing, so drift is caught on the
    edit that caused it."""
    sql = "SELECT * FROM anchors WHERE status != ?"
    args: tuple = (SUPERSEDED,)
    if only_memory:
        sql += " AND memory_id = ?"
        args = args + (only_memory,)

    results: list[dict[str, Any]] = []
    with write_tx(conn):
        for anchor in rows(conn.execute(sql, args)):
            results.append(verify_anchor(conn, anchor, root, commit))

    summary: dict[str, int] = {}
    for result in results:
        summary[result["status"]] = summary.get(result["status"], 0) + 1
    changed = [r for r in results if r["transition"] not in ("unchanged", "skipped")]
    return {"checked": len(results), "by_status": summary, "changed": changed}


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
