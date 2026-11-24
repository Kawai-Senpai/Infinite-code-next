"""Opt-in, commit-pinned updates for the ICN MCP launcher."""

from __future__ import annotations

import argparse
import re
import shutil
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

from .db import SCHEMA_VERSION
from .paths import storage_root

UPSTREAM = "https://github.com/Kawai-Senpai/Infinite-code-next.git"
BRANCH = "refs/heads/main"
COMMIT = re.compile(r"[0-9a-f]{40}")
# Verify the candidate can import the MCP server without opening a workspace.
PROBE = ("from icn.server import mcp; from icn.db import SCHEMA_VERSION; "
         f"raise SystemExit(SCHEMA_VERSION != {SCHEMA_VERSION})")
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _state_file() -> Path:
    return storage_root() / "cache" / "updates.db"


def _revision(conn: sqlite3.Connection, extras: str) -> str | None:
    row = conn.execute("SELECT revision FROM prepared WHERE extras = ?", (extras,)).fetchone()
    return row[0] if row and isinstance(row[0], str) and COMMIT.fullmatch(row[0]) else None


def _prepared(extras: str) -> str | None:
    try:
        with closing(sqlite3.connect(_state_file().as_uri() + "?mode=ro", uri=True, timeout=0)) as conn:
            return _revision(conn, extras)
    except sqlite3.Error:
        return None


def _command(uv: str, revision: str, extras: str, *, offline: bool) -> list[str]:
    package = f"infinite-code-next[{extras}]" if extras else "infinite-code-next"
    command = [uv, "run", "--quiet", "--no-project", "--no-config", "--no-env-file",
               "--no-python-downloads", "--python", sys.executable,
               "--with", f"{package} @ git+{UPSTREAM}@{revision}"]
    if offline:
        command.append("--offline")
    return [*command, "--", "python", "-I"]


def _probe(command: list[str], timeout: int) -> bool:
    try:
        result = subprocess.run(
            [*command, "-c", PROBE], stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout,
            creationflags=NO_WINDOW,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _latest(git: str) -> str | None:
    output = subprocess.check_output(
        [git, "-c", "credential.interactive=never", "ls-remote", "--exit-code", UPSTREAM, BRANCH],
        stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True, timeout=15,
        cwd=_state_file().parent, creationflags=NO_WINDOW,
    )
    fields = output.split()
    if len(fields) == 2 and fields[1] == BRANCH and COMMIT.fullmatch(fields[0]):
        return fields[0]
    return None


def prepare(uv: str, git: str, extras: str) -> None:
    """Cache one validated upstream revision. Called by a background process."""
    path = _state_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path, timeout=0)) as conn, conn:
            # This lock is only for update workers. MCP sessions read the
            # previous committed revision while this worker downloads.
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("CREATE TABLE IF NOT EXISTS prepared (extras TEXT PRIMARY KEY, revision TEXT NOT NULL)")
            revision = _latest(git)
            if revision is None:
                return
            current = _revision(conn, extras)
            offline = _command(uv, revision, extras, offline=True)
            if current == revision and _probe(offline, 10):
                return
            if _probe(_command(uv, revision, extras, offline=False), 120) and _probe(offline, 10):
                conn.execute(
                    "INSERT INTO prepared(extras, revision) VALUES (?,?)"
                    " ON CONFLICT(extras) DO UPDATE SET revision=excluded.revision",
                    (extras, revision),
                )
    except (OSError, sqlite3.Error, subprocess.SubprocessError):
        # A failed lookup, build, or concurrent update leaves the old pointer usable.
        return


def _start_worker(flags: list[str]) -> None:
    try:
        subprocess.Popen(
            [sys.executable, "-I", "-m", "icn", "--prepare-update", *flags],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=NO_WINDOW,
            start_new_session=True,
        )
    except OSError:
        pass


def main(argv: list[str], *, worker: bool = False) -> int:
    parser = argparse.ArgumentParser(prog="infinite-code-next --auto-update")
    parser.add_argument("--preview", action="store_true", help="Install media renderers in downloaded revisions")
    parser.add_argument("--semantic", action="store_true", help="Install optional semantic search in downloaded revisions")
    args = parser.parse_args(argv)
    enabled = [name for name in ("preview", "semantic") if getattr(args, name)]
    extras = ",".join(enabled)
    uv, git = shutil.which("uv"), shutil.which("git")
    if worker:
        if uv and git:
            prepare(uv, git, extras)
        return 0

    # Read the pointer before starting this launch's download. The candidate
    # prepared here becomes eligible only on a later launch.
    revision = _prepared(extras) if uv else None
    if uv and git:
        _start_worker([f"--{name}" for name in enabled])
    if uv and revision:
        command = _command(uv, revision, extras, offline=True)
        if _probe(command, 10):
            try:
                # Preserve the client's stdin and stdout for MCP. Once any
                # server starts, let the client decide if it wants to restart.
                return subprocess.call([*command, "-m", "icn"], creationflags=NO_WINDOW)
            except OSError:
                pass
    from .server import main as serve
    serve()
    return 0
