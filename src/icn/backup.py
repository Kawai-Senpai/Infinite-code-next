"""Export a consistent SQLite snapshot of an existing repository store."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from contextlib import closing
from pathlib import Path

from . import catalog, paths
from .identity import probe
from .workspace import resolve_root


def backup(root: Path, output: Path) -> dict:
    with closing(sqlite3.connect(paths.catalog_path().as_uri() + "?mode=ro", uri=True)) as registry:
        registry.row_factory = sqlite3.Row
        # A known checkout gives us its exact identity even if it shares a
        # Git root commit with a fork or the remote was renamed.
        checkout = registry.execute(
            "SELECT repo_id FROM checkouts WHERE path = ?", (str(root),),
        ).fetchone()
        if checkout is None:
            identity = probe(root)
            repo_id, _, matched_on = catalog.identify(registry, identity)
            if repo_id and matched_on == "root_commit" and catalog.detect_fork(registry, identity, repo_id):
                repo_id = None
        else:
            repo_id = checkout["repo_id"]
    if not repo_id:
        raise ValueError("This repository has no ICN store yet. Open it in ICN first.")

    store = paths.repo_db_path(repo_id)
    output = output.expanduser().resolve()
    deadline = time.monotonic() + 60

    def check_progress(status: int, remaining: int, total: int) -> None:
        if status != sqlite3.SQLITE_DONE and time.monotonic() >= deadline:
            raise TimeoutError("Timed out while backing up the repository store.")

    with closing(sqlite3.connect(store.as_uri() + "?mode=ro", uri=True)) as source:
        # Reserve the name exclusively. Existing files and the live store are
        # never overwritten, even if another backup targets the same name.
        with output.open("xb"):
            pass
        try:
            with closing(sqlite3.connect(output)) as destination:
                source.backup(destination, pages=256, progress=check_progress)
                check = destination.execute("PRAGMA quick_check").fetchall()
                if check != [("ok",)]:
                    raise sqlite3.DatabaseError(f"Snapshot integrity check failed: {check[0][0]}")
                memories = destination.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        except BaseException:
            output.unlink(missing_ok=True)
            raise
    return {"ok": True, "repo_id": repo_id, "path": str(output), "memories": memories}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="icn backup", description=__doc__)
    parser.add_argument("output", type=Path, help="Path for a new SQLite snapshot")
    parser.add_argument("--root", help="Repository directory, defaults to the current workspace")
    args = parser.parse_args(argv)
    try:
        result = backup(resolve_root(args.root), args.output)
    except (OSError, ValueError, sqlite3.Error) as exc:
        print(f"Backup failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0
