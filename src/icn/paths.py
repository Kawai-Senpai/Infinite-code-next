"""Storage root resolution.

PLAN 2 section 3: one configurable central root, per-repo stores inside it, and
a durable/rebuildable split so clearing the cache is always safe.

Platform defaults:
    Windows   %LOCALAPPDATA%\\InfiniteCode
    Linux     $XDG_DATA_HOME/infinite-code  (or ~/.local/share/infinite-code)
    macOS     ~/Library/Application Support/InfiniteCode

Override with INFINITE_CODE_HOME, which wins over everything.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ENV_HOME = "INFINITE_CODE_HOME"
ENV_ROOT = "INFINITE_CODE_ROOT"


def storage_root() -> Path:
    """The central data root. Created on demand by the callers that write."""
    override = os.environ.get(ENV_HOME)
    if override:
        return Path(override).expanduser().resolve()

    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
        return Path(base).joinpath("InfiniteCode").resolve()

    if sys.platform == "darwin":
        return (Path.home() / "Library" / "Application Support" / "InfiniteCode").resolve()

    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg else (Path.home() / ".local" / "share")
    return base.joinpath("infinite-code").resolve()


def catalog_path() -> Path:
    """catalog.db - repositories, aliases, checkouts, cross-repo edges.

    PLAN.md section 9: the global pieces are one database, not two, because
    SQLite foreign keys cannot cross database boundaries. Anything that must
    stay resolvable when a repo store is unavailable lives here.
    """
    return storage_root() / "catalog.db"


def repo_dir(repo_id: str) -> Path:
    """Durable per-repo directory. Never regenerable from source."""
    return storage_root() / "data" / "repos" / repo_id


def repo_db_path(repo_id: str) -> Path:
    return repo_dir(repo_id) / "repo.db"


def cache_dir(repo_id: str) -> Path:
    """Rebuildable per-repo directory. Safe to delete at any time."""
    return storage_root() / "cache" / "repos" / repo_id


def logs_dir() -> Path:
    return storage_root() / "logs"


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
