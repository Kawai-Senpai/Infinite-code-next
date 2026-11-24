"""Opt-in, commit-pinned updates for the ICN MCP launcher."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from contextlib import closing
from pathlib import Path

from .db import SCHEMA_VERSION
from .paths import storage_root

UPSTREAM = "https://github.com/Kawai-Senpai/Infinite-code-next.git"
BRANCH = "refs/heads/main"
COMMIT = re.compile(r"[0-9a-f]{40}")
SLOT = re.compile(r"[a-z0-9_]{8}")
PROBE = ("from icn.server import mcp; from icn.db import SCHEMA_VERSION; "
         f"raise SystemExit(SCHEMA_VERSION != {SCHEMA_VERSION})")
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _state_file() -> Path:
    return storage_root() / "cache" / "updates.db"


def _environments() -> Path:
    return _state_file().parent / "update-envs"


def _slot(conn: sqlite3.Connection, extras: str) -> tuple[str, str] | None:
    row = conn.execute("SELECT revision, slot FROM installed WHERE extras = ?", (extras,)).fetchone()
    if row is None:
        return None
    revision, slot = row
    if not isinstance(revision, str) or not isinstance(slot, str):
        return None
    return (revision, slot) if COMMIT.fullmatch(revision) and SLOT.fullmatch(slot) else None


def _prepared(extras: str) -> tuple[str, str] | None:
    try:
        with closing(sqlite3.connect(_state_file().as_uri() + "?mode=ro", uri=True, timeout=0)) as conn:
            return _slot(conn, extras)
    except sqlite3.Error:
        return None


def _python(directory: Path) -> Path:
    return directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _windowless(python: Path) -> Path:
    if os.name == "nt":
        gui = python.with_name("pythonw.exe")
        if gui.is_file():
            return gui
    return python


def _run(command: list[str], timeout: int) -> bool:
    try:
        result = subprocess.run(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, timeout=timeout, creationflags=NO_WINDOW,
            cwd=_state_file().parent,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _probe(python: Path) -> bool:
    return _run([str(_windowless(python)), "-I", "-c", PROBE], 15)


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


def _stage(uv: str, revision: str, extras: str) -> str | None:
    base = _environments().resolve()
    base.mkdir(parents=True, exist_ok=True)
    # Keep the environment path short enough for deeply nested wheel files
    # on Windows. The full revision is stored in the database.
    directory = Path(tempfile.mkdtemp(prefix="", dir=base))
    python = _python(directory)
    package = f"infinite-code-next[{extras}]" if extras else "infinite-code-next"
    prepared = False
    try:
        prepared = (
            _run([uv, "venv", "--no-config", "--no-python-downloads", "--python",
                  sys.executable, str(directory)], 30)
            and _run([uv, "pip", "install", "--no-config", "--link-mode", "copy", "--python",
                      str(python), f"{package} @ git+{UPSTREAM}@{revision}"], 120)
            and _probe(python)
        )
        return directory.name if prepared else None
    finally:
        # Only remove the unique, unpublished environment this attempt created.
        if not prepared and directory.resolve().parent == base:
            shutil.rmtree(directory, ignore_errors=True)


def prepare(uv: str, git: str, extras: str) -> None:
    """Download and validate an upstream commit in a separate environment."""
    path = _state_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path, timeout=0)) as conn, conn:
            # Other launchers can still read the last committed pointer.
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS installed (extras TEXT PRIMARY KEY,"
                " revision TEXT NOT NULL, slot TEXT NOT NULL)"
            )
            revision = _latest(git)
            if revision is None:
                return
            current = _slot(conn, extras)
            if current and current[0] == revision and _probe(_python(_environments() / current[1])):
                return
            candidate = _stage(uv, revision, extras)
            if candidate is not None:
                conn.execute(
                    "INSERT INTO installed(extras, revision, slot) VALUES (?,?,?)"
                    " ON CONFLICT(extras) DO UPDATE SET revision=excluded.revision, slot=excluded.slot",
                    (extras, revision, candidate),
                )
    except (OSError, sqlite3.Error, subprocess.SubprocessError):
        # A failed lookup, build, or competing worker leaves the old pointer usable.
        return


def _start_worker(flags: list[str]) -> None:
    try:
        subprocess.Popen(
            [str(_windowless(Path(sys.executable))), "-I", "-m", "icn", "--prepare-update", *flags],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=NO_WINDOW, start_new_session=True,
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
    fetch = uv and git and os.getenv("UV_OFFLINE", "").strip().lower() not in {"1", "true"}
    if worker:
        if fetch:
            prepare(uv, git, extras)
        return 0

    # A pointer published by this launch's worker is for a later launch.
    prepared = _prepared(extras)
    if fetch:
        _start_worker([f"--{name}" for name in enabled])
    if prepared:
        python = _python(_environments() / prepared[1])
        if _probe(python):
            try:
                # Preserve the MCP client's pipes and working directory. The
                # prepared interpreter runs directly without resolving packages.
                return subprocess.call(
                    [str(_windowless(python)), "-I", "-m", "icn"],
                    stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr,
                    creationflags=NO_WINDOW,
                )
            except OSError:
                pass
    from .server import main as serve
    serve()
    return 0
