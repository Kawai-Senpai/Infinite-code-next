"""Cross-repository contract edges.

PLAN.md sections 8 and 14. This is the piece that lets a memory in one
repository say something about another: "this endpoint is consumed by the
mobile app", "this schema is mirrored in the analytics pipeline".

The governing rule is that a cross-repo edge must never depend on the other
repository being available. Two consequences shape everything here:

  * The target snapshot is captured at write time, not looked up at read time.
    The other repo may be archived, on an unmounted drive, or deleted by the
    time anyone asks, and "CONSUMES_CONTRACT -> acme/billing charge_customer"
    has to stay a readable sentence regardless.

  * Only catalog stubs are consulted. We never open the other repository's
    store, because it may not exist. An edge we cannot fully resolve is
    recorded UNRESOLVED and heals itself later, rather than being rejected.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from . import ids
from .catalog import ACTIVE, PURGED, _log, now
from .db import jdump, jload, one, rows, write_tx
from .identity import normalize_remote

# Contract kinds an agent can assert between repositories. Deliberately small:
# each answers "what breaks over there if this changes here".
CROSS_REPO_KINDS = {
    "PROVIDES_CONTRACT",   # this entity is an API or schema others depend on
    "CONSUMES_CONTRACT",   # this entity depends on something in another repo
    "MIRRORS",             # the same logic is duplicated in another repo
    "RELATED_TO",          # weaker: worth looking at together
}


def resolve_repo_ref(conn: sqlite3.Connection, reference: str) -> dict[str, Any] | None:
    """Find a repository by whatever an agent is likely to type.

    A repo_id, a remote URL in any form, a project id, a name, or a path.
    Everything except repo_id is mutable - which is exactly why those are
    aliases rather than identity - but they are what a person actually knows.
    """
    text = (reference or "").strip()
    if not text:
        return None

    row = one(conn.execute(
        "SELECT * FROM repositories WHERE repo_id = ? AND status != ?", (text, PURGED)))
    if row:
        return row

    candidates = [normalize_remote(text), text]
    try:
        candidates.append(str(Path(text).expanduser().resolve()))
    except (OSError, ValueError):
        pass

    for candidate in candidates:
        if not candidate:
            continue
        alias = one(conn.execute(
            "SELECT r.* FROM repository_aliases a JOIN repositories r ON r.repo_id = a.repo_id"
            " WHERE a.alias = ? AND r.status != ? LIMIT 1", (candidate, PURGED)))
        if alias:
            return alias

    return one(conn.execute(
        "SELECT * FROM repositories WHERE name = ? AND status != ? LIMIT 1", (text, PURGED)))


def _find_remote_entity(conn: sqlite3.Connection, repo_id: str,
                        name: str) -> tuple[str | None, dict[str, Any]]:
    """Look up an entity in another repo through its catalog stubs only."""
    snapshot: dict[str, Any] = {"name": name, "repo_id": repo_id, "resolved": False}
    if not name:
        return None, snapshot

    for row in rows(conn.execute(
        "SELECT entity_id, kind, snapshot FROM entity_refs WHERE repo_id = ? LIMIT 4000", (repo_id,)
    )):
        data = jload(row["snapshot"], {}) or {}
        symbol_path = data.get("symbol_path") or ""
        matches = {symbol_path, data.get("path"), data.get("title")}
        if name in matches or symbol_path.rsplit(".", 1)[-1] == name:
            snapshot.update(data)
            snapshot["kind"] = row["kind"]
            snapshot["resolved"] = True
            return row["entity_id"], snapshot

    return None, snapshot


def link(conn: sqlite3.Connection, from_entity: str, to_repo_ref: str, to_entity_name: str,
         kind: str, confidence: float = 0.7, evidence: Any = None,
         commit: str | None = None) -> dict[str, Any]:
    """Assert an edge from a local entity to something in another repository."""
    if kind not in CROSS_REPO_KINDS:
        return {"ok": False, "error": "unknown cross-repo kind " + repr(kind)
                                      + "; expected one of " + str(sorted(CROSS_REPO_KINDS))}
    if not from_entity or not to_repo_ref or not to_entity_name:
        return {"ok": False, "error": "from_entity, to_repo and to_entity are all required"}

    target_repo = resolve_repo_ref(conn, to_repo_ref)
    edge_id = ids.new_id(ids.EDGE)

    if target_repo is None:
        # Not an error: the other repo may simply not have been opened on this
        # machine yet. Record the intent and let it heal on that repo's first
        # workspace open.
        snapshot = {"repo_ref": to_repo_ref, "name": to_entity_name, "resolved": False}
        with write_tx(conn):
            conn.execute(
                "INSERT INTO cross_repo_edges (edge_id, from_entity, to_entity, kind, edge_class,"
                " status, confidence, evidence, target_snapshot, valid_from_commit, created_at,"
                " last_verified_at) VALUES (?,?,?,?,'asserted','UNRESOLVED',?,?,?,?,?,?)",
                (edge_id, from_entity, "unresolved:" + to_repo_ref + "/" + to_entity_name,
                 kind, confidence, jdump(evidence), jdump(snapshot), commit, now(), now()),
            )
        return {"ok": True, "edge_id": edge_id, "kind": kind, "status": "UNRESOLVED",
                "target": snapshot,
                "note": "repository " + repr(to_repo_ref) + " is not known yet; the edge is"
                        " recorded now and resolves when that repository is opened"}

    target_id, snapshot = _find_remote_entity(conn, target_repo["repo_id"], to_entity_name)
    snapshot["repo_ref"] = to_repo_ref
    status = "ACTIVE" if target_id else "UNRESOLVED"
    fallback = "unresolved:" + target_repo["repo_id"] + "/" + to_entity_name

    with write_tx(conn):
        conn.execute(
            "INSERT INTO cross_repo_edges (edge_id, from_entity, to_entity, kind, edge_class,"
            " status, confidence, evidence, target_snapshot, valid_from_commit, created_at,"
            " last_verified_at) VALUES (?,?,?,?,'asserted',?,?,?,?,?,?,?)",
            (edge_id, from_entity, target_id or fallback, kind, status, confidence,
             jdump(evidence), jdump(snapshot), commit, now(), now()),
        )
        _log(conn, edge_id, "cross_repo_edge", None, status,
             kind + " -> " + target_repo["repo_id"], {"name": to_entity_name})

    return {
        "ok": True, "edge_id": edge_id, "kind": kind, "status": status,
        "to_repo": target_repo["repo_id"], "to_repo_name": target_repo["name"],
        "to_entity": target_id, "target": snapshot,
    }


def edges_for(conn: sqlite3.Connection, entity_ids: list[str]) -> list[dict[str, Any]]:
    """Cross-repo edges touching these entities, with whatever is still known.

    An edge whose target repository is missing is reported with
    reachable=False, never dropped. A broken link is a finding.
    """
    if not entity_ids:
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    for start in range(0, len(entity_ids), 400):
        part = entity_ids[start:start + 400]
        placeholders = ",".join("?" for _ in part)
        for row in rows(conn.execute(
            "SELECT * FROM cross_repo_edges WHERE from_entity IN (" + placeholders + ")"
            " OR to_entity IN (" + placeholders + ") LIMIT 200", (*part, *part)
        )):
            if row["edge_id"] in seen:
                continue
            seen.add(row["edge_id"])
            snapshot = jload(row["target_snapshot"], {}) or {}
            target_repo = None
            if snapshot.get("repo_id"):
                target_repo = one(conn.execute(
                    "SELECT repo_id, name, status FROM repositories WHERE repo_id = ?",
                    (snapshot["repo_id"],)))
            out.append({
                "edge_id": row["edge_id"], "kind": row["kind"], "status": row["status"],
                "confidence": row["confidence"],
                "from_entity": row["from_entity"], "to_entity": row["to_entity"],
                "target": snapshot,
                "target_repo": dict(target_repo) if target_repo else None,
                "reachable": bool(target_repo and target_repo["status"] == ACTIVE),
            })
    return out


def resolve_pending(conn: sqlite3.Connection, repo_id: str) -> int:
    """Upgrade UNRESOLVED edges now that `repo_id` has been opened.

    Called on every workspace open. Asserting an edge before the other
    repository has ever been seen is a normal, supported order of events.
    """
    aliases = {r["alias"] for r in rows(conn.execute(
        "SELECT alias FROM repository_aliases WHERE repo_id = ?", (repo_id,)))}
    aliases.add(repo_id)

    upgraded = 0
    for row in rows(conn.execute(
        "SELECT * FROM cross_repo_edges WHERE status = 'UNRESOLVED' LIMIT 500"
    )):
        snapshot = jload(row["target_snapshot"], {}) or {}
        if snapshot.get("repo_ref") not in aliases:
            continue
        target_id, new_snapshot = _find_remote_entity(conn, repo_id, snapshot.get("name") or "")
        if not target_id:
            continue
        new_snapshot["repo_ref"] = snapshot.get("repo_ref")
        with write_tx(conn):
            conn.execute(
                "UPDATE cross_repo_edges SET to_entity = ?, status = 'ACTIVE',"
                " target_snapshot = ?, last_verified_at = ? WHERE edge_id = ?",
                (target_id, jdump(new_snapshot), now(), row["edge_id"]),
            )
        upgraded += 1
    return upgraded
