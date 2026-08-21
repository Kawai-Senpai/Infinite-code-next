"""Entity resolution that cannot fail.

PLAN.md section 7. Internally we never load a path and assume it worked. Every
lookup goes through resolve(), and "cannot currently resolve" comes back as
data with a status and whatever we last knew, never as an exception.

That single rule is what keeps a missing drive, a deleted clone or a purged
repository from turning an agent request into a stack trace.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from . import ids, paths
from .catalog import entity_snapshot
from .db import one

RESOLVED = "RESOLVED"
DELETED = "DELETED"                      # tombstoned, we know exactly what it was
DETAILS_UNAVAILABLE = "DETAILS_UNAVAILABLE"   # stub survives, repo store does not
TARGET_MISSING = "TARGET_MISSING"        # repository has no live checkout
TARGET_ARCHIVED = "TARGET_ARCHIVED"
TARGET_PURGED = "TARGET_PURGED"
UNKNOWN = "UNKNOWN"


def resolve(catalog: sqlite3.Connection, repo_conn: sqlite3.Connection | None,
            entity_id: str) -> dict[str, Any]:
    """Resolve any entity ID to a status plus the best details available."""
    kind = ids.prefix_of(entity_id)

    if repo_conn is not None:
        local = _resolve_local(repo_conn, entity_id, kind)
        if local is not None:
            return local

    stub = entity_snapshot(catalog, entity_id)
    if stub is None:
        return {"id": entity_id, "status": UNKNOWN, "kind": kind or "unknown"}

    repo = one(catalog.execute("SELECT * FROM repositories WHERE repo_id = ?", (stub["repo_id"],)))
    status = DETAILS_UNAVAILABLE
    if repo is None:
        status = UNKNOWN
    elif repo["status"] == "PURGED":
        status = TARGET_PURGED
    elif repo["status"] == "ARCHIVED":
        status = TARGET_ARCHIVED
    elif repo["status"] in ("MISSING", "OFFLINE"):
        status = TARGET_MISSING
    elif not paths.repo_db_path(stub["repo_id"]).exists():
        status = DETAILS_UNAVAILABLE

    return {
        "id": entity_id,
        "kind": stub["kind"],
        "status": status,
        "repo_id": stub["repo_id"],
        "repo_status": repo["status"] if repo else None,
        "last_known": stub["snapshot"] if status != TARGET_PURGED else None,
    }


def _resolve_local(conn: sqlite3.Connection, entity_id: str, kind: str) -> dict[str, Any] | None:
    if kind == ids.SYMBOL:
        row = one(conn.execute("SELECT * FROM symbols WHERE symbol_id = ?", (entity_id,)))
        if row is None:
            return None
        return {
            "id": entity_id, "kind": "symbol",
            "status": RESOLVED if row["status"] == "ACTIVE" else DELETED,
            "name": row["name"], "symbol_path": row["symbol_path"], "symbol_kind": row["kind"],
            "path": row["last_known_path"], "signature": row["signature"],
            "line_start": row["line_start"], "line_end": row["line_end"],
            "deleted_at_commit": row["deleted_at_commit"],
        }

    if kind == ids.FILE:
        row = one(conn.execute("SELECT * FROM files WHERE file_id = ?", (entity_id,)))
        if row is None:
            return None
        return {
            "id": entity_id, "kind": "file",
            "status": RESOLVED if row["status"] == "ACTIVE" else DELETED,
            "path": row["path"], "lang": row["lang"],
            "deleted_at_commit": row["deleted_at_commit"],
        }

    if kind == ids.MEMORY:
        row = one(conn.execute("SELECT * FROM memories WHERE memory_id = ?", (entity_id,)))
        if row is None:
            return None
        return {
            "id": entity_id, "kind": "memory", "status": RESOLVED,
            "memory_kind": row["kind"], "title": row["title"], "body": row["body"],
            "severity": row["severity"], "authority": row["authority"],
            "memory_status": row["status"],
        }

    if kind == ids.ANCHOR:
        row = one(conn.execute("SELECT * FROM anchors WHERE anchor_id = ?", (entity_id,)))
        if row is None:
            return None
        return {"id": entity_id, "kind": "anchor", "status": RESOLVED,
                "anchor_status": row["status"], "symbol_path": row["symbol_path"]}

    if entity_id.startswith("import:"):
        return {"id": entity_id, "kind": "import", "status": RESOLVED,
                "statement": entity_id[len("import:"):]}
    return None


def describe_edge(catalog: sqlite3.Connection, repo_conn: sqlite3.Connection | None,
                  edge: dict[str, Any]) -> dict[str, Any]:
    """Render an edge with both endpoints resolved.

    An edge whose target cannot be resolved is reported, not dropped. A broken
    link is a diagnostic (PLAN.md section 15), and the caller sees the status
    rather than a silently shorter list.
    """
    target = resolve(catalog, repo_conn, edge["to_id"])
    status = edge.get("status", "ACTIVE")
    if target["status"] in (TARGET_MISSING, DETAILS_UNAVAILABLE, UNKNOWN) and status == "ACTIVE":
        status = "UNRESOLVED"
    elif target["status"] == DELETED and status == "ACTIVE":
        status = "TARGET_DELETED"
    return {
        "kind": edge["kind"],
        "edge_class": edge.get("edge_class"),
        "status": status,
        "confidence": edge.get("confidence"),
        "target": target,
        "valid_until_commit": edge.get("valid_until_commit"),
    }
