"""Workspace context: resolve, open, index, verify.

Every tool call funnels through here. It is the piece that makes the server
zero-config: no picker, no per-repo start, no daemon. Resolution order is
explicit param, then INFINITE_CODE_ROOT, then the process working directory,
and the resolved root is echoed back in every response so a misresolution is
visible immediately rather than silently poisoning the store.

Indexing is lazy and budgeted. The first call on a cold repository returns a
usable answer with index_state "partial" and finishes in the background under
the lease, because a first call that blocks for a minute is the same failure as
requiring a manual start.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import anchors as anchor_mod
from . import catalog as catalog_mod
from . import embed as embed_mod
from . import vectors as vectors_mod
from . import crossrepo
from . import indexer as indexer_mod
from . import paths
from .db import init_repo_store, one, write_tx
from .identity import read_repo_config
from .lease import Lease

FIRST_INDEX_BUDGET = 8.0     # seconds before we hand back a partial index
_BACKGROUND: dict[str, threading.Thread] = {}
_LOCK = threading.Lock()


def resolve_root(explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get(paths.ENV_ROOT)
    if env:
        return Path(env).expanduser().resolve()
    return Path.cwd().resolve()


def named_sibling_roots(root: Path, query: str) -> list[Path]:
    """Return sibling Git repositories explicitly named in a query."""
    lowered = (query or "").lower()
    found: list[Path] = []
    try:
        siblings = root.parent.iterdir()
    except OSError:
        return found
    for sibling in siblings:
        if sibling == root or not sibling.is_dir() or not (sibling / ".git").exists():
            continue
        name = sibling.name.lower()
        if name in lowered or str(sibling).lower() in lowered:
            found.append(sibling.resolve())
    return found


@dataclass
class Workspace:
    root: Path
    repo_id: str
    checkout_id: str
    catalog: sqlite3.Connection
    store: sqlite3.Connection
    commit: str | None
    info: dict[str, Any]

    @property
    def excludes(self) -> list[str]:
        config = read_repo_config(self.root)
        index_config = config.get("index") or {}
        return list(index_config.get("exclude") or [])

    def close(self) -> None:
        for conn in (self.store, self.catalog):
            try:
                conn.close()
            except sqlite3.Error:
                pass


def open_workspace(explicit_root: str | None = None) -> Workspace:
    # Start fetching the embedding model now, on a background thread, so the
    # download overlaps with indexing and with the agent reading the briefing
    # rather than stalling the first question it asks. No-op after the first
    # call, and when semantic search is switched off.
    embed_mod.prewarm()

    root = resolve_root(explicit_root)
    catalog = catalog_mod.open_catalog()
    info = catalog_mod.open_workspace(catalog, root)
    repo_id = info["repo_id"]
    store = init_repo_store(paths.repo_db_path(repo_id))
    paths.cache_dir(repo_id).mkdir(parents=True, exist_ok=True)

    # An edge may have been asserted against this repository before it was ever
    # opened here. Opening it is what lets those resolve.
    try:
        info["cross_repo_resolved"] = crossrepo.resolve_pending(catalog, repo_id)
    except sqlite3.Error:
        info["cross_repo_resolved"] = 0
    return Workspace(
        root=Path(info["root"]),
        repo_id=repo_id,
        checkout_id=info["checkout_id"],
        catalog=catalog,
        store=store,
        # A ref name is not a commit id; storing one corrupts every later
        # commit comparison (see indexer.normalize_commit).
        commit=indexer_mod.normalize_commit(info.get("head")),
        info=info,
    )


def _lease_for(repo_id: str) -> Lease:
    return Lease(paths.cache_dir(repo_id) / "indexer.lock")


def _background_finish(repo_id: str, root: Path, commit: str | None, excludes: list[str]) -> None:
    """Finish a truncated first index without blocking the caller."""
    lease = _lease_for(repo_id)
    if not lease.try_acquire():
        return
    store = None
    try:
        store = init_repo_store(paths.repo_db_path(repo_id))
        idx = indexer_mod.Indexer(store, root, repo_id, excludes)
        idx.full_index(commit)
        catalog = catalog_mod.open_catalog()
        with write_tx(catalog):
            catalog.execute("UPDATE repositories SET last_indexed_commit=? WHERE repo_id=?",
                            (commit, repo_id))
        catalog.close()
        anchor_mod.verify_repo(store, root, commit)
    except Exception:
        # A background index that fails must never take the server with it. The
        # next foreground call sees a partial index and simply resumes.
        pass
    finally:
        if store is not None:
            try:
                store.close()
            except sqlite3.Error:
                pass
        lease.release()
        with _LOCK:
            _BACKGROUND.pop(repo_id, None)


def _build_semantics_later(repo_id: str) -> None:
    """Wait for the model to finish downloading, then encode the repository.

    Without this, a first run would index the code before the 33MB model had
    arrived, find no encoder, and leave the store with no vectors until the
    next time something happened to change a file. Semantic search would look
    broken on exactly the run where a user is deciding whether it works.

    Opens its own connection: sqlite3 connections belong to the thread that
    created them.
    """
    store = None
    try:
        embed_mod.prewarm(blocking=True)
        encoder = embed_mod.load_encoder()
        if encoder is None:
            return
        store = init_repo_store(paths.repo_db_path(repo_id))
        vectors_mod.refresh(store, encoder)
    except Exception:
        # Same rule as the background indexer: an optional enhancement must
        # never take the server down. Lexical search is unaffected.
        pass
    finally:
        if store is not None:
            store.close()


def _refresh_semantics(ws: Workspace) -> dict[str, Any]:
    """Encode new or changed content, or arrange for it once the model lands."""
    encoder = embed_mod.load_encoder()
    if encoder is not None:
        return vectors_mod.refresh(ws.store, encoder)

    state = embed_mod.status()
    if state["state"] == embed_mod.LOADING:
        threading.Thread(target=_build_semantics_later, args=(ws.repo_id,),
                         daemon=True, name="icn-embed-build").start()
    return {"encoded": 0, "skipped": 0, **state}


def ensure_indexed(ws: Workspace, force_full: bool = False,
                   budget: float = FIRST_INDEX_BUDGET) -> dict[str, Any]:
    """Bring the index up to date, then re-verify anchors.

    Anchor verification runs immediately after indexing, on the same call, so
    drift is caught on the edit that caused it rather than by a later sweep.
    """
    repo = one(ws.catalog.execute("SELECT last_indexed_commit FROM repositories WHERE repo_id=?",
                                  (ws.repo_id,)))
    last_indexed = repo["last_indexed_commit"] if repo else None
    state = indexer_mod.index_state(ws.store)
    cold = state["files_active"] == 0 and state["symbols_active"] == 0

    lease = _lease_for(ws.repo_id)
    if not lease.try_acquire():
        holder = lease.holder() or {}
        return {"indexed": False, "reason": "another process holds the indexer lease",
                "lease_holder_pid": holder.get("pid"),
                "index_state": "partial" if cold else "ready", **state}

    try:
        idx = indexer_mod.Indexer(ws.store, ws.root, ws.repo_id, ws.excludes)
        if force_full or cold:
            report = idx.full_index(ws.commit, budget_seconds=budget if cold and not force_full else None)
        else:
            report = idx.incremental_index(ws.commit, last_indexed)

        if report.get("truncated"):
            with _LOCK:
                if ws.repo_id not in _BACKGROUND:
                    thread = threading.Thread(
                        target=_background_finish,
                        args=(ws.repo_id, ws.root, ws.commit, ws.excludes),
                        daemon=True, name=f"icn-index-{ws.repo_id}",
                    )
                    _BACKGROUND[ws.repo_id] = thread
                    lease.release()      # hand the lease to the background run
                    thread.start()
        else:
            with write_tx(ws.catalog):
                ws.catalog.execute("UPDATE repositories SET last_indexed_commit=? WHERE repo_id=?",
                                   (ws.commit, ws.repo_id))
    finally:
        lease.release()

    # Before the cascade, not after: an anchor whose fingerprint moved only
    # because the extractor changed must be carried across first, or the
    # cascade reads the algorithm change as a body change and downgrades an
    # anchor pointing at code nobody touched.
    rebase = anchor_mod.rebase_extractor_change(ws.store, ws.root)
    verification = anchor_mod.verify_repo(ws.store, ws.root, ws.commit)

    # Semantic vectors, refreshed in the same pass that refreshed the symbols
    # they describe. Incremental by content hash, so an unchanged repository
    # costs one query and encodes nothing.
    semantic = _refresh_semantics(ws)

    result = {
        "indexed": True,
        "index_state": "partial" if report.get("truncated") else "ready",
        **report,
        "anchors": verification,
        "semantic": semantic,
        **indexer_mod.index_state(ws.store),
    }
    if rebase.get("rebased"):
        result["anchor_rebase"] = rebase
    if result.get("files_active", 0) and not result.get("symbols_active", 0):
        result["parser_warning"] = (
            "Source files were indexed but no symbols were parsed. Check the Python, "
            "tree-sitter, and tree-sitter-language-pack compatibility with `icn doctor`."
        )
    return result


def status(ws: Workspace) -> dict[str, Any]:
    lease = _lease_for(ws.repo_id)
    return {
        "root": str(ws.root),
        "repo_id": ws.repo_id,
        "checkout_id": ws.checkout_id,
        "repo_status": ws.info.get("status"),
        "identity": {
            "strength": ws.info.get("identity_strength"),
            "matched_on": ws.info.get("matched_on"),
            "confidence": ws.info.get("match_confidence"),
            "vcs": ws.info.get("vcs"),
            "is_worktree": ws.info.get("is_worktree"),
        },
        "head": ws.commit,
        "branch": ws.info.get("branch"),
        "dirty": ws.info.get("dirty"),
        "index": indexer_mod.index_state(ws.store),
        # Both halves: whether the model is loaded, and whether this store has
        # been encoded with it. "loading" is not an error, and a user needs to
        # be able to tell it apart from quietly broken.
        "semantic": {**embed_mod.status(), **vectors_mod.stats(ws.store)},
        "anchors": anchor_mod.anchor_health(ws.store),
        "store": {
            "repo_db": str(paths.repo_db_path(ws.repo_id)),
            "cache": str(paths.cache_dir(ws.repo_id)),
            "catalog": str(paths.catalog_path()),
        },
        "indexer_lease": lease.holder(),
        "agit": {"present": (ws.root / ".agit" / "HEAD").exists()},
    }
