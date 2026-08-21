"""The catalog: repository identity, lifecycle, and cross-repo stubs.

PLAN.md sections 2, 3, 9, 13, 14, 16, 17.

The catalog is the part that must keep working when everything else is
unavailable. If a repo store is missing, corrupt, or on an unmounted drive, the
catalog still knows the repository existed, what it was called, and what it
used to point at. That is what makes "cannot currently resolve" a normal
result instead of an exception.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import ids, paths
from .db import init_catalog, jdump, jload, one, rows, write_tx
from .identity import Probe, probe

# Repository lifecycle (PLAN.md section 3). Only PURGED destroys anything.
ACTIVE = "ACTIVE"
MISSING = "MISSING"
MOVED = "MOVED"
OFFLINE = "OFFLINE"
ARCHIVED = "ARCHIVED"
DETACHED = "DETACHED"
PURGED = "PURGED"

# How much a matching alias is worth when identifying a repository.
ALIAS_WEIGHT = {"root_commit": 0.97, "project_id": 0.95, "remote": 0.80, "path": 0.60}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def open_catalog() -> sqlite3.Connection:
    return init_catalog(paths.catalog_path())


def _log(conn: sqlite3.Connection, subject_id: str, kind: str, frm: str | None,
         to: str | None, reason: str, detail: Any = None) -> None:
    conn.execute(
        "INSERT INTO lifecycle_events (subject_id, subject_kind, from_status, to_status, reason, detail, at)"
        " VALUES (?,?,?,?,?,?,?)",
        (subject_id, kind, frm, to, reason, jdump(detail), now()),
    )


def identify(conn: sqlite3.Connection, pr: Probe) -> tuple[str | None, float, str]:
    """Find the repository this checkout belongs to.

    Returns (repo_id, confidence, matched_on). A re-clone at a brand new path
    must attach to its existing knowledge, not start a fresh universe, so the
    strongest alias wins rather than the path.
    """
    best: tuple[str | None, float, str] = (None, 0.0, "none")
    for kind, value in pr.aliases():
        weight = ALIAS_WEIGHT.get(kind, 0.5)
        if weight <= best[1]:
            continue
        row = one(conn.execute(
            "SELECT r.repo_id FROM repository_aliases a JOIN repositories r ON r.repo_id = a.repo_id"
            " WHERE a.alias_kind = ? AND a.alias = ? AND r.status != ?",
            (kind, value, PURGED),
        ))
        if row:
            best = (row["repo_id"], weight, kind)
    return best


def detect_fork(conn: sqlite3.Connection, pr: Probe, candidate_repo: str) -> bool:
    """Is this a fork of the matched repository rather than the same one?

    PLAN.md section 14. A fork shares its root commit with upstream, which is
    the strongest identity signal we have - so without this check
    github.com/alice/backend would be silently identified as
    github.com/acme/backend and inherit all of its memories as if they were
    native. Disjoint, non-empty remote sets are the evidence that these are two
    projects that happen to share history.
    """
    if not pr.remotes:
        return False

    # A repository rename looks exactly like a fork from remotes alone. The
    # difference is the clone: `git remote set-url` reuses this working
    # directory, while a fork is a different checkout. So a path we have
    # already seen under this repository is the same repository, full stop.
    seen_here = one(conn.execute(
        "SELECT checkout_id FROM checkouts WHERE repo_id = ? AND path = ?",
        (candidate_repo, str(pr.root)),
    ))
    if seen_here:
        return False

    known = {
        row["alias"]
        for row in rows(conn.execute(
            "SELECT alias FROM repository_aliases WHERE repo_id = ? AND alias_kind = 'remote'",
            (candidate_repo,),
        ))
    }
    if not known:
        return False
    return known.isdisjoint(set(pr.remotes))


def open_workspace(conn: sqlite3.Connection, root: Path) -> dict[str, Any]:
    """Resolve a directory to a repository, registering or reattaching it.

    This is the entry point every tool call funnels through. It is idempotent
    and cheap; it does not index anything.
    """
    pr = probe(root)
    repo_id, confidence, matched_on = identify(conn, pr)
    created = False
    reattached = False
    forked_from = None

    # A fork shares upstream's root commit, so identity alone would merge the
    # two. Split them, and record the lineage instead of discarding it.
    if repo_id is not None and matched_on == "root_commit" and detect_fork(conn, pr, repo_id):
        forked_from = repo_id
        repo_id = None
        matched_on = "none"
        confidence = 0.0

    with write_tx(conn):
        if repo_id is None:
            repo_id = ids.new_id(ids.REPO)
            created = True
            conn.execute(
                "INSERT INTO repositories (repo_id, name, status, vcs, identity_strength, project_id,"
                " created_at, last_seen_at, last_indexed_commit) VALUES (?,?,?,?,?,?,?,?,NULL)",
                (repo_id, pr.project_name or pr.root.name, ACTIVE,
                 "git" if pr.is_git else "none", pr.identity_strength, pr.project_id, now(), now()),
            )
            _log(conn, repo_id, "repository", None, ACTIVE, "registered", {"root": str(pr.root)})
            if forked_from:
                # Lineage, not equivalence. Upstream's memories stay upstream's
                # until something confirms they still hold here.
                conn.execute(
                    "INSERT OR IGNORE INTO repo_relationships (from_repo, to_repo, kind,"
                    " confidence, evidence, created_at) VALUES (?,?,'FORKED_FROM',?,?,?)",
                    (repo_id, forked_from, 0.9,
                     jdump({"shared_root_commit": pr.root_commit, "remotes": pr.remotes}), now()),
                )
                _log(conn, repo_id, "repository", None, ACTIVE, "fork detected",
                     {"forked_from": forked_from})
        else:
            prev = one(conn.execute("SELECT status FROM repositories WHERE repo_id = ?", (repo_id,)))
            prev_status = prev["status"] if prev else None
            if prev_status in (MISSING, OFFLINE, DETACHED, MOVED):
                # PLAN.md section 13: rediscovery reattaches rather than forking.
                reattached = True
                _log(conn, repo_id, "repository", prev_status, ACTIVE, "rediscovered",
                     {"root": str(pr.root), "matched_on": matched_on, "confidence": confidence})
            if prev_status != ARCHIVED:
                conn.execute("UPDATE repositories SET status = ?, last_seen_at = ? WHERE repo_id = ?",
                             (ACTIVE, now(), repo_id))
            else:
                conn.execute("UPDATE repositories SET last_seen_at = ? WHERE repo_id = ?", (now(), repo_id))

        for kind, value in pr.aliases():
            conn.execute(
                "INSERT OR IGNORE INTO repository_aliases (repo_id, alias_kind, alias, first_seen_at)"
                " VALUES (?,?,?,?)",
                (repo_id, kind, value, now()),
            )

        checkout = one(conn.execute("SELECT * FROM checkouts WHERE path = ?", (str(pr.root),)))
        if checkout is None:
            checkout_id = ids.new_id(ids.CHECKOUT)
            conn.execute(
                "INSERT INTO checkouts (checkout_id, repo_id, path, git_common_dir, is_worktree,"
                " status, last_head, last_seen_at, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (checkout_id, repo_id, str(pr.root), pr.git_common_dir, int(pr.is_worktree),
                 ACTIVE, pr.head, now(), now()),
            )
            _log(conn, checkout_id, "checkout", None, ACTIVE, "attached", {"repo_id": repo_id})
        else:
            checkout_id = checkout["checkout_id"]
            conn.execute(
                "UPDATE checkouts SET status = ?, last_head = ?, last_seen_at = ?, git_common_dir = ?,"
                " is_worktree = ? WHERE checkout_id = ?",
                (ACTIVE, pr.head, now(), pr.git_common_dir, int(pr.is_worktree), checkout_id),
            )

        # A checkout of this repo that no longer exists on disk is MISSING, not
        # deleted. Its knowledge stays exactly where it is.
        for other in rows(conn.execute(
            "SELECT checkout_id, path, status FROM checkouts WHERE repo_id = ? AND checkout_id != ?",
            (repo_id, checkout_id),
        )):
            if other["status"] == ACTIVE and not Path(other["path"]).exists():
                conn.execute("UPDATE checkouts SET status = ? WHERE checkout_id = ?",
                             (MISSING, other["checkout_id"]))
                _log(conn, other["checkout_id"], "checkout", ACTIVE, MISSING, "path not found",
                     {"path": other["path"]})

    repo = one(conn.execute("SELECT * FROM repositories WHERE repo_id = ?", (repo_id,)))
    return {
        "repo_id": repo_id,
        "checkout_id": checkout_id,
        "root": str(pr.root),
        "created": created,
        "reattached": reattached,
        "matched_on": matched_on,
        "match_confidence": confidence,
        "identity_strength": pr.identity_strength,
        "vcs": "git" if pr.is_git else "none",
        "head": pr.head,
        "branch": pr.branch,
        "dirty": pr.dirty,
        "is_worktree": pr.is_worktree,
        "status": repo["status"] if repo else ACTIVE,
        "last_indexed_commit": repo["last_indexed_commit"] if repo else None,
        "forked_from": forked_from,
        "probe": pr,
    }


def list_repositories(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Every repository we know, whether or not it is on disk right now."""
    out: list[dict[str, Any]] = []
    for repo in rows(conn.execute("SELECT * FROM repositories ORDER BY last_seen_at DESC")):
        checkouts = rows(conn.execute(
            "SELECT checkout_id, path, status, is_worktree FROM checkouts WHERE repo_id = ?",
            (repo["repo_id"],),
        ))
        counts = one(conn.execute(
            "SELECT COUNT(*) AS n FROM memory_registry WHERE repo_id = ? AND status = 'ACTIVE'",
            (repo["repo_id"],),
        )) or {"n": 0}
        out.append({
            "repo_id": repo["repo_id"],
            "name": repo["name"],
            "status": repo["status"],
            "identity_strength": repo["identity_strength"],
            "last_seen_at": repo["last_seen_at"],
            "last_indexed_commit": repo["last_indexed_commit"],
            "memories": counts["n"],
            "checkouts": checkouts,
            "store_available": paths.repo_db_path(repo["repo_id"]).exists(),
        })
    return out


def reconcile(conn: sqlite3.Connection) -> dict[str, Any]:
    """Walk every known checkout and update reachability.

    PLAN.md section 16. Cheap enough to run on workspace open. Never destroys
    anything; a vanished path becomes MISSING and stays queryable.
    """
    changed: list[dict[str, str]] = []
    with write_tx(conn):
        for ck in rows(conn.execute("SELECT * FROM checkouts")):
            exists = Path(ck["path"]).exists()
            if exists and ck["status"] in (MISSING, OFFLINE):
                conn.execute("UPDATE checkouts SET status = ?, last_seen_at = ? WHERE checkout_id = ?",
                             (ACTIVE, now(), ck["checkout_id"]))
                _log(conn, ck["checkout_id"], "checkout", ck["status"], ACTIVE, "path reappeared")
                changed.append({"checkout": ck["path"], "from": ck["status"], "to": ACTIVE})
            elif not exists and ck["status"] == ACTIVE:
                # An unmounted drive is OFFLINE, not MISSING: the difference is
                # whether we expect it back.
                anchor = Path(ck["path"]).anchor
                to = OFFLINE if anchor and not Path(anchor).exists() else MISSING
                conn.execute("UPDATE checkouts SET status = ? WHERE checkout_id = ?", (to, ck["checkout_id"]))
                _log(conn, ck["checkout_id"], "checkout", ACTIVE, to, "path not found")
                changed.append({"checkout": ck["path"], "from": ACTIVE, "to": to})

        for repo in rows(conn.execute("SELECT * FROM repositories WHERE status NOT IN (?,?)", (ARCHIVED, PURGED))):
            live = one(conn.execute(
                "SELECT COUNT(*) AS n FROM checkouts WHERE repo_id = ? AND status = ?",
                (repo["repo_id"], ACTIVE),
            ))
            has_live = bool(live and live["n"])
            target = ACTIVE if has_live else MISSING
            if repo["status"] != target:
                conn.execute("UPDATE repositories SET status = ? WHERE repo_id = ?", (target, repo["repo_id"]))
                _log(conn, repo["repo_id"], "repository", repo["status"], target, "reconcile")
                changed.append({"repo": repo["repo_id"], "from": repo["status"], "to": target})

    return {"changed": changed, "checked_at": now()}


def set_repo_status(conn: sqlite3.Connection, repo_id: str, status: str, reason: str) -> dict[str, Any]:
    repo = one(conn.execute("SELECT * FROM repositories WHERE repo_id = ?", (repo_id,)))
    if repo is None:
        return {"ok": False, "error": f"unknown repository {repo_id}"}
    with write_tx(conn):
        conn.execute("UPDATE repositories SET status = ? WHERE repo_id = ?", (status, repo_id))
        _log(conn, repo_id, "repository", repo["status"], status, reason)
    return {"ok": True, "repo_id": repo_id, "from": repo["status"], "to": status}


def forget_checkout(conn: sqlite3.Connection, path: str) -> dict[str, Any]:
    """This local clone is gone. Repository knowledge is untouched."""
    ck = one(conn.execute("SELECT * FROM checkouts WHERE path = ?", (str(Path(path).resolve()),)))
    if ck is None:
        return {"ok": False, "error": f"no checkout registered at {path}"}
    with write_tx(conn):
        conn.execute("UPDATE checkouts SET status = 'FORGOTTEN' WHERE checkout_id = ?", (ck["checkout_id"],))
        _log(conn, ck["checkout_id"], "checkout", ck["status"], "FORGOTTEN", "forget_checkout")
    return {"ok": True, "checkout_id": ck["checkout_id"], "repo_id": ck["repo_id"]}


def purge(conn: sqlite3.Connection, repo_id: str, redact_snapshots: bool = True) -> dict[str, Any]:
    """The only destructive operation. PLAN.md section 17.

    Removes the repo's stores and catalog rows. Cross-repo edges pointing at it
    are redacted rather than left holding identifying snapshots, because purge
    carries privacy intent that MISSING does not.
    """
    repo = one(conn.execute("SELECT * FROM repositories WHERE repo_id = ?", (repo_id,)))
    if repo is None:
        return {"ok": False, "error": f"unknown repository {repo_id}"}

    import shutil

    removed = []
    for directory in (paths.repo_dir(repo_id), paths.cache_dir(repo_id)):
        if directory.exists():
            shutil.rmtree(directory, ignore_errors=True)
            removed.append(str(directory))

    with write_tx(conn):
        if redact_snapshots:
            conn.execute(
                "UPDATE cross_repo_edges SET status = 'TARGET_PURGED', target_snapshot = NULL"
                " WHERE to_entity IN (SELECT entity_id FROM entity_refs WHERE repo_id = ?)",
                (repo_id,),
            )
        conn.execute("DELETE FROM entity_refs WHERE repo_id = ?", (repo_id,))
        conn.execute("DELETE FROM memory_registry WHERE repo_id = ?", (repo_id,))
        conn.execute("DELETE FROM repository_aliases WHERE repo_id = ?", (repo_id,))
        conn.execute("DELETE FROM checkouts WHERE repo_id = ?", (repo_id,))
        conn.execute("UPDATE repositories SET status = ? WHERE repo_id = ?", (PURGED, repo_id))
        _log(conn, repo_id, "repository", repo["status"], PURGED, "purge", {"removed": removed})
    return {"ok": True, "repo_id": repo_id, "removed": removed}


# --------------------------------------------------------------- entity stubs

def upsert_entity_ref(conn: sqlite3.Connection, entity_id: str, repo_id: str, kind: str,
                      snapshot: dict[str, Any], status: str = ACTIVE) -> None:
    """Mirror a durable entity into the catalog so it survives store loss.

    Owns its transaction. Callers are usually inside a write_tx on the *repo
    store*, which is a different connection - relying on that to commit the
    catalog silently dropped every stub, and with them the ability to resolve
    anything cross-repo.
    """
    with write_tx(conn):
        conn.execute(
            "INSERT INTO entity_refs (entity_id, repo_id, kind, snapshot, status, updated_at)"
            " VALUES (?,?,?,?,?,?)"
            " ON CONFLICT(entity_id) DO UPDATE SET snapshot=excluded.snapshot,"
            " status=excluded.status, updated_at=excluded.updated_at",
            (entity_id, repo_id, kind, jdump(snapshot), status, now()),
        )


def register_memory(conn: sqlite3.Connection, memory_id: str, repo_id: str, kind: str,
                    severity: str, status: str, summary: str) -> None:
    """Register a memory in the central catalog. Owns its transaction, for the
    same reason as upsert_entity_ref."""
    with write_tx(conn):
        conn.execute(
            "INSERT INTO memory_registry (memory_id, repo_id, kind, severity, status, summary,"
            " updated_at) VALUES (?,?,?,?,?,?,?)"
            " ON CONFLICT(memory_id) DO UPDATE SET kind=excluded.kind, severity=excluded.severity,"
            " status=excluded.status, summary=excluded.summary, updated_at=excluded.updated_at",
            (memory_id, repo_id, kind, severity, status, summary, now()),
        )


def entity_snapshot(conn: sqlite3.Connection, entity_id: str) -> dict[str, Any] | None:
    row = one(conn.execute("SELECT * FROM entity_refs WHERE entity_id = ?", (entity_id,)))
    if row is None:
        return None
    row["snapshot"] = jload(row["snapshot"], {})
    return row


def health(conn: sqlite3.Connection) -> dict[str, Any]:
    """Graph health summary (PLAN.md section 16)."""
    def count(sql: str, args: tuple = ()) -> int:
        row = one(conn.execute(sql, args))
        return int(row["n"]) if row else 0

    by_status = {
        r["status"]: r["n"]
        for r in rows(conn.execute("SELECT status, COUNT(*) AS n FROM repositories GROUP BY status"))
    }
    return {
        "repositories": {
            "total": count("SELECT COUNT(*) AS n FROM repositories"),
            "by_status": by_status,
        },
        "checkouts": {
            r["status"]: r["n"]
            for r in rows(conn.execute("SELECT status, COUNT(*) AS n FROM checkouts GROUP BY status"))
        },
        "cross_repo_edges": {
            "total": count("SELECT COUNT(*) AS n FROM cross_repo_edges"),
            "by_status": {
                r["status"]: r["n"]
                for r in rows(conn.execute("SELECT status, COUNT(*) AS n FROM cross_repo_edges GROUP BY status"))
            },
        },
        "memories_registered": count("SELECT COUNT(*) AS n FROM memory_registry"),
    }
