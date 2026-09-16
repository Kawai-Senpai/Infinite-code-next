"""Experiment lab: a tree of experiments, each one exact code plus a measured run.

Kept apart from agit on purpose. agit is a backup of the working tree; the lab
is a notebook of questions and the code that answered them. They share no git
directory, no branches and no state, so neither can disturb the other.

The idea, adopted from OpenResearch's experiment tree:

    baseline        the starting code and the ONE command that measures it
      child         inherits its parent's code, commits one change, same command
        grandchild  builds on a result its parent already established

Three rules make results comparable. OpenResearch states them in prompts; here
the lab enforces them, because a rule an agent can forget is a suggestion.

    1. One command for every experiment. It is set once and cannot change after
       anything has been measured. Only committed code differs between nodes.
    2. A node freezes once a run answers it. Its code is then permanent; a new
       idea is a child. A run that crashed answered nothing, so the node stays
       editable, but two such runs in a row stop further launches until the
       caller says force=True.
    3. A run never executes an editable checkout. It extracts the node's commit
       into its own directory, so what ran is exactly what the commit says.

Layout, all under <repo>/.icn-lab/ (gitignored, discardable by deleting it):
    repo.git/          bare repository, one branch per experiment: exp/<slug>
    lab.db             tree, runs, verdicts, evidence links
    trees/<slug>/      editable checkout of one experiment
    indexes/<slug>     that checkout's private git index
    runs/<run_id>/     src/ (the extracted commit), log.txt, and labrun.py's files

Measured results reach ICN's memory through conclude(): the verdict becomes a
decision, failed_attempt or rationale memory whose body carries the run id,
commit, command, exit code and metrics, so the evidence survives even if this
folder is deleted.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import labrun
from .db import connect
from .ids import new_id

LAB_DIR = ".icn-lab"
BRANCH_PREFIX = "exp/"
USER_NAME = "Infinite Code Lab"
USER_EMAIL = "lab@infinite-code.local"
GIT_TIMEOUT = 300.0

REPAIR_CAP = 2                 # consecutive unanswered failures before force is needed
SETBACK_STOP = 3               # consecutive losses or failures that suggest stopping
STALE_HEARTBEAT_SECONDS = 45.0
MAX_WAIT_SECONDS = 600.0
DIFF_CHARS = 20_000
DEFAULT_LOG_TAIL = 8_000
FAN_WIDTH = 5                  # children with no grandchildren that make a flat fan
NOODLE_DEPTH = 4               # single-child links in a row that make a noodle

TERMINAL = ("done", "failed", "cancelled", "timed_out", "lost")
UNANSWERED_FAILURES = ("failed", "timed_out", "lost")
VERDICTS = ("win", "loss", "inconclusive", "void")

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS experiments (
    exp_id      TEXT PRIMARY KEY,
    slug        TEXT NOT NULL UNIQUE,
    parent_id   TEXT,
    title       TEXT NOT NULL,
    hypothesis  TEXT,
    branch      TEXT NOT NULL,
    fork_sha    TEXT,
    promoted_at TEXT,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    exp_id      TEXT NOT NULL,
    commit_sha  TEXT NOT NULL,
    command     TEXT NOT NULL,
    status      TEXT NOT NULL,
    exit_code   INTEGER,
    metrics     TEXT,
    answered    INTEGER,          -- NULL while running; 1 answered; 0 answered nothing
    verdict     TEXT,
    note        TEXT,
    runner_pid  INTEGER,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    error       TEXT,
    memory_id   TEXT              -- the primary ICN memory its verdict produced
);
CREATE INDEX IF NOT EXISTS idx_runs_exp ON runs(exp_id, started_at);
CREATE TABLE IF NOT EXISTS evidence (
    memory_id  TEXT NOT NULL,
    run_id     TEXT NOT NULL,
    exp_id     TEXT,
    verdict    TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (memory_id, run_id)
);
"""


class LabError(RuntimeError):
    """A request the lab refuses, with the reason an agent can act on."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _parse_time(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def lab_path(root: Path) -> Path:
    return (Path(root) / LAB_DIR).resolve()


def exists(root: Path) -> bool:
    return (lab_path(root) / "lab.db").exists()


# ------------------------------------------------------------------ plumbing


class Lab:
    """One opened lab. Cheap to open; close it when done."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.dir = lab_path(self.root)
        self.git_dir = self.dir / "repo.git"
        self.conn: sqlite3.Connection = connect(self.dir / "lab.db")
        self.conn.executescript(SCHEMA)
        columns = {row["name"] for row in self.conn.execute("PRAGMA table_info(runs)")}
        if "memory_id" not in columns:
            self.conn.execute("ALTER TABLE runs ADD COLUMN memory_id TEXT")

    def close(self) -> None:
        self.conn.close()

    # -- git ---------------------------------------------------------------

    def git(self, *args: str, work_tree: Path | None = None, index: Path | None = None,
            check: bool = True) -> str:
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
        if index is not None:
            env["GIT_INDEX_FILE"] = str(index)
        command = ["git", f"--git-dir={self.git_dir}"]
        if work_tree is not None:
            command.append(f"--work-tree={work_tree}")
        command.extend(args)
        # stdin detached: a git child that inherits the MCP server's stdin can
        # consume the protocol stream (same reason as identity.run_git).
        proc = subprocess.run(command, capture_output=True, stdin=subprocess.DEVNULL,
                              text=True, encoding="utf-8", errors="replace",
                              timeout=GIT_TIMEOUT, env=env,
                              cwd=str(work_tree or self.root), check=False)
        if check and proc.returncode != 0:
            raise LabError(f"git {' '.join(args[:2])} failed: "
                           f"{(proc.stderr or proc.stdout).strip()[:800]}")
        return (proc.stdout or "").strip()

    def head(self, branch: str) -> str:
        return self.git("rev-parse", "--verify", f"refs/heads/{branch}^{{commit}}")

    def snapshot(self, work_tree: Path, index: Path, branch: str, parent: str | None,
                 message: str) -> tuple[str, bool]:
        """Commit everything in `work_tree` onto `branch`. Returns (sha, changed).

        Plumbing rather than `git commit`, because the bare repository has one
        shared HEAD and several checkouts: write-tree and commit-tree touch only
        the index they are given, and update-ref moves the branch with the old
        value as a guard, so two writers cannot silently overwrite each other.
        """
        self.git("add", "-A", work_tree=work_tree, index=index)
        tree = self.git("write-tree", work_tree=work_tree, index=index)
        if parent and self.git("rev-parse", f"{parent}^{{tree}}") == tree:
            return parent, False
        args = ["commit-tree", tree, "-m", message]
        if parent:
            args[2:2] = ["-p", parent]
        sha = self.git(*args)
        self.git("update-ref", f"refs/heads/{branch}", sha, parent or "0" * 40)
        return sha, True

    def materialize(self, sha: str, target: Path, index: Path) -> None:
        """Write the files of commit `sha` into `target`, tracked by `index`."""
        target.mkdir(parents=True, exist_ok=True)
        index.parent.mkdir(parents=True, exist_ok=True)
        self.git("read-tree", "--reset", "-u", sha, work_tree=target, index=index)

    # -- rows --------------------------------------------------------------

    def meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute("INSERT INTO meta (key, value) VALUES (?, ?) "
                          "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))

    def experiment(self, ref: str) -> dict[str, Any]:
        ref = (ref or "").strip()
        if not ref:
            raise LabError("this action needs exp: an experiment id or slug (see action='tree')")
        row = self.conn.execute("SELECT * FROM experiments WHERE exp_id=? OR slug=?",
                                (ref, ref)).fetchone()
        if row is None:
            raise LabError(f"no experiment {ref!r} in this lab (see action='tree')")
        return dict(row)

    def experiments(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM experiments ORDER BY created_at")]

    def run_row(self, run_id: str) -> dict[str, Any]:
        row = self.conn.execute("SELECT * FROM runs WHERE run_id=?", ((run_id or "").strip(),)).fetchone()
        if row is None:
            raise LabError(f"no run {run_id!r} in this lab (see action='runs')")
        return dict(row)

    def runs_for(self, exp_id: str | None = None, limit: int = 1000) -> list[dict[str, Any]]:
        if exp_id:
            cur = self.conn.execute("SELECT * FROM runs WHERE exp_id=? ORDER BY started_at DESC "
                                    "LIMIT ?", (exp_id, limit))
        else:
            cur = self.conn.execute("SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,))
        return [dict(r) for r in cur]

    def frozen_by(self, exp_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM runs WHERE exp_id=? AND answered=1 "
                                "ORDER BY started_at LIMIT 1", (exp_id,)).fetchone()
        return dict(row) if row else None

    def run_dir(self, run_id: str) -> Path:
        return self.dir / "runs" / run_id

    def tree_dir(self, slug: str) -> Path:
        return self.dir / "trees" / slug

    def index_file(self, slug: str) -> Path:
        return self.dir / "indexes" / slug


def _register_gitignore(root: Path) -> bool:
    ignore_file = root / ".gitignore"
    line = f"{LAB_DIR}/"
    try:
        if ignore_file.exists():
            content = ignore_file.read_text(encoding="utf-8", errors="replace")
            if re.search(rf"^{re.escape(LAB_DIR)}/?\s*$", content, re.MULTILINE):
                return False
            separator = "" if content.endswith("\n") or not content else "\n"
            ignore_file.write_text(f"{content}{separator}{line}\n", encoding="utf-8")
        else:
            ignore_file.write_text(f"{line}\n", encoding="utf-8")
        return True
    except OSError:
        return False


def open_lab(root: Path) -> Lab:
    if not exists(root):
        raise LabError("no experiment lab here yet. Start one with "
                       "experiment(action='init', command='<the one command that measures the code>')")
    return Lab(root)


def _slugify(conn: sqlite3.Connection, title: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:48].strip("-") or "experiment"
    slug, n = base, 2
    while conn.execute("SELECT 1 FROM experiments WHERE slug=?", (slug,)).fetchone():
        slug, n = f"{base}-{n}", n + 1
    return slug


# ------------------------------------------------------------------ run state


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(0x1000, False, int(pid))  # QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5   # access denied: it exists
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == 259                # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _refresh(lab: Lab, run: dict[str, Any]) -> dict[str, Any]:
    """Bring one run row up to date with what its watcher left on disk.

    `lost` is not final: a machine that slept through the heartbeat window can
    still produce exit.json later, and that real outcome replaces the guess.
    """
    if run["status"] in TERMINAL and run["status"] != "lost":
        return run
    directory = lab.run_dir(run["run_id"])
    exit_file = directory / "exit.json"
    if exit_file.exists():
        try:
            final = json.loads(exit_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return run
        status = final.get("outcome") or ("done" if final.get("exit_code") == 0 else "failed")
        answered = 1 if status == "done" else 0
        ended = final.get("ended_at")
        ended_iso = (datetime.fromtimestamp(ended, timezone.utc).isoformat(timespec="milliseconds")
                     if isinstance(ended, (int, float)) else _now())
        # A verdict already given (possible on a run first reported lost) is a
        # judgement and outranks the default derived from the exit code.
        lab.conn.execute(
            "UPDATE runs SET status=?, exit_code=?, metrics=?,"
            " answered=CASE WHEN verdict IS NULL THEN ? ELSE answered END,"
            " ended_at=?, error=? WHERE run_id=?",
            (status, final.get("exit_code"), json.dumps(final.get("metrics") or {}), answered,
             ended_iso, final.get("error"), run["run_id"]),
        )
        return lab.run_row(run["run_id"])

    beat = directory / "heartbeat"
    reference = beat if beat.exists() else directory / "spec.json"
    try:
        age = time.time() - reference.stat().st_mtime
    except OSError:
        age = float("inf")
    if age > STALE_HEARTBEAT_SECONDS and not _pid_alive(run.get("runner_pid")):
        if run["status"] != "lost":
            lab.conn.execute("UPDATE runs SET status='lost', answered=0, ended_at=? WHERE run_id=?",
                             (_now(), run["run_id"]))
            return lab.run_row(run["run_id"])
    return run


def refresh_all(lab: Lab) -> None:
    for row in lab.conn.execute("SELECT * FROM runs WHERE status IN ('running', 'lost')").fetchall():
        _refresh(lab, dict(row))


def _metrics(run: dict[str, Any]) -> dict[str, float]:
    try:
        return json.loads(run["metrics"]) if run.get("metrics") else {}
    except ValueError:
        return {}


def _live_metrics(lab: Lab, run: dict[str, Any]) -> dict[str, float]:
    if run["status"] == "running":
        return labrun.metrics_from_log(lab.run_dir(run["run_id"]) / "log.txt")
    return _metrics(run)


def _duration(run: dict[str, Any]) -> float | None:
    start, end = _parse_time(run.get("started_at")), _parse_time(run.get("ended_at"))
    if start is None:
        return None
    return round((end or time.time()) - start, 2)


def _run_view(lab: Lab, run: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run["run_id"],
        "exp_id": run["exp_id"],
        "status": run["status"],
        "exit_code": run["exit_code"],
        "metrics": _live_metrics(lab, run),
        "answered": None if run["answered"] is None else bool(run["answered"]),
        "verdict": run["verdict"],
        "note": run["note"],
        "commit": run["commit_sha"],
        "command": run["command"],
        "started_at": run["started_at"],
        "ended_at": run["ended_at"],
        "duration_seconds": _duration(run),
        "log": str(lab.run_dir(run["run_id"]) / "log.txt"),
        "error": run["error"],
    }


# ------------------------------------------------------------------ actions


def init(root: Path, command: str, title: str = "baseline", hypothesis: str = "") -> dict[str, Any]:
    """Create the lab and its baseline from the working tree as it is now."""
    root = Path(root).resolve()
    if not (command or "").strip():
        raise LabError("init needs command: the single shell command that runs and measures "
                       "the code. Every experiment will use exactly this command.")
    if exists(root):
        lab = Lab(root)
        try:
            if lab.experiments():
                raise LabError("this lab already has a baseline; use action='create' for a new "
                               "experiment, or delete .icn-lab/ to start over")
        finally:
            lab.close()

    directory = lab_path(root)
    directory.mkdir(parents=True, exist_ok=True)
    git_dir = directory / "repo.git"
    if not (git_dir / "HEAD").exists():
        proc = subprocess.run(["git", "init", "--bare", "-q", str(git_dir)], capture_output=True,
                              stdin=subprocess.DEVNULL, text=True, check=False)
        if proc.returncode != 0:
            raise LabError(f"git init failed: {(proc.stderr or proc.stdout).strip()}")
    gitignored = _register_gitignore(root)

    lab = Lab(root)
    try:
        for key, value in (("user.name", USER_NAME), ("user.email", USER_EMAIL),
                           ("core.autocrlf", "false"), ("core.longpaths", "true"),
                           ("gc.auto", "0")):
            lab.git("config", key, value)
        # The baseline snapshot walks the real working tree, whose .gitignore is
        # honoured, but these two folders are ICN's own and must never become
        # part of an experiment even when nothing ignores them yet.
        exclude = git_dir / "info" / "exclude"
        exclude.parent.mkdir(parents=True, exist_ok=True)
        exclude.write_text(f"/{LAB_DIR}/\n/.agit/\n/.git/\n", encoding="utf-8")

        slug = _slugify(lab.conn, title or "baseline")
        branch = BRANCH_PREFIX + slug
        index = lab.dir / "indexes" / "_baseline_snapshot"
        index.parent.mkdir(parents=True, exist_ok=True)
        try:
            sha, _ = lab.snapshot(root, index, branch, None, f"baseline: {title}")
        finally:
            index.unlink(missing_ok=True)

        exp_id = new_id("exp")
        lab.conn.execute(
            "INSERT INTO experiments (exp_id, slug, parent_id, title, hypothesis, branch, fork_sha,"
            " created_at) VALUES (?,?,?,?,?,?,?,?)",
            (exp_id, slug, None, title or "baseline", hypothesis or "", branch, None, _now()),
        )
        lab.set_meta("command", command.strip())
        files = lab.git("ls-tree", "-r", "--name-only", sha).splitlines()
        return {
            "ok": True,
            "exp_id": exp_id,
            "slug": slug,
            "branch": branch,
            "commit": sha,
            "command": command.strip(),
            "files": len(files),
            "gitignore_updated": gitignored,
            "lab": str(lab.dir),
            "next": (f"experiment(action='run', exp='{slug}') measures the baseline first. "
                     "Print results as lines like `ICN_METRIC loss=0.42` so runs are comparable."),
        }
    finally:
        lab.close()


def set_command(root: Path, command: str) -> dict[str, Any]:
    lab = open_lab(root)
    try:
        if not (command or "").strip():
            raise LabError("set_command needs command")
        answered = lab.conn.execute("SELECT run_id, exp_id FROM runs WHERE answered=1 LIMIT 1").fetchone()
        if answered:
            raise LabError(
                f"the command is a fixed contract once anything has been measured (run "
                f"{answered['run_id']} answered {answered['exp_id']}). Changing it now would make "
                "every earlier result incomparable. Change the code on a child experiment instead.")
        old = lab.meta("command")
        lab.set_meta("command", command.strip())
        return {"ok": True, "command": command.strip(), "previous": old}
    finally:
        lab.close()


def create(root: Path, title: str, hypothesis: str = "", parent: str = "") -> dict[str, Any]:
    lab = open_lab(root)
    try:
        if not (title or "").strip():
            raise LabError("create needs title: the idea in a few words")
        if parent:
            parent_row = lab.experiment(parent)
        else:
            parent_row = _focal(lab)
            if parent_row is None:
                raise LabError("this lab has no baseline; run action='init' first")
        fork = lab.head(parent_row["branch"])
        slug = _slugify(lab.conn, title)
        branch = BRANCH_PREFIX + slug
        lab.git("update-ref", f"refs/heads/{branch}", fork, "0" * 40)
        exp_id = new_id("exp")
        lab.conn.execute(
            "INSERT INTO experiments (exp_id, slug, parent_id, title, hypothesis, branch, fork_sha,"
            " created_at) VALUES (?,?,?,?,?,?,?,?)",
            (exp_id, slug, parent_row["exp_id"], title.strip(), hypothesis or "", branch, fork, _now()),
        )
        result: dict[str, Any] = {
            "ok": True, "exp_id": exp_id, "slug": slug, "branch": branch,
            "parent": {"exp_id": parent_row["exp_id"], "slug": parent_row["slug"],
                       "chosen": "explicit" if parent else "focal"},
            "fork_commit": fork,
            "next": f"experiment(action='checkout', exp='{slug}') gives you a folder to edit",
        }
        refresh_all(lab)
        if lab.frozen_by(parent_row["exp_id"]) is None:
            result["warning"] = (f"parent {parent_row['slug']!r} has not been answered by a run yet, "
                                 "so this child builds on code nobody has measured")
        return result
    finally:
        lab.close()


def _focal(lab: Lab) -> dict[str, Any] | None:
    """Where the next round hangs: the latest promoted winner, else the baseline."""
    row = lab.conn.execute("SELECT * FROM experiments WHERE promoted_at IS NOT NULL "
                           "ORDER BY promoted_at DESC LIMIT 1").fetchone()
    if row is None:
        row = lab.conn.execute("SELECT * FROM experiments WHERE parent_id IS NULL "
                               "ORDER BY created_at LIMIT 1").fetchone()
    return dict(row) if row else None


def checkout(root: Path, exp: str) -> dict[str, Any]:
    lab = open_lab(root)
    try:
        row = lab.experiment(exp)
        refresh_all(lab)
        tree, index = lab.tree_dir(row["slug"]), lab.index_file(row["slug"])
        head = lab.head(row["branch"])
        created = False
        if not (tree.exists() and index.exists()):
            lab.materialize(head, tree, index)
            created = True
        frozen = lab.frozen_by(row["exp_id"])
        result = {
            "ok": True, "exp_id": row["exp_id"], "slug": row["slug"], "path": str(tree),
            "commit": head, "created": created, "frozen": frozen is not None,
            "next": (f"edit files under {tree}, then experiment(action='commit', exp='{row['slug']}', "
                     "message=...)"),
        }
        if frozen:
            result["warning"] = (f"this experiment is frozen: run {frozen['run_id']} answered it. "
                                 "Read it freely, but commit will refuse. Put a new idea on a child.")
            result["next"] = f"experiment(action='create', parent='{row['slug']}', title=...)"
        return result
    finally:
        lab.close()


def _dirty(lab: Lab, row: dict[str, Any]) -> list[str]:
    tree, index = lab.tree_dir(row["slug"]), lab.index_file(row["slug"])
    if not (tree.exists() and index.exists()):
        return []
    lab.git("add", "-A", work_tree=tree, index=index)
    changed = lab.git("diff", "--cached", "--name-only", lab.head(row["branch"]),
                      work_tree=tree, index=index)
    return [line for line in changed.splitlines() if line.strip()]


def commit(root: Path, exp: str, message: str = "") -> dict[str, Any]:
    lab = open_lab(root)
    try:
        row = lab.experiment(exp)
        refresh_all(lab)
        frozen = lab.frozen_by(row["exp_id"])
        if frozen:
            raise LabError(
                f"{row['slug']} is frozen: run {frozen['run_id']} answered it, so its code is the "
                "record of that result and cannot change. Create a child with "
                f"experiment(action='create', parent='{row['slug']}', title=...) and edit that. "
                "If that run did not really answer the question (for example it exited 0 without "
                "doing the work), say so with action='conclude', verdict='void', note=why.")
        tree, index = lab.tree_dir(row["slug"]), lab.index_file(row["slug"])
        if not (tree.exists() and index.exists()):
            raise LabError(f"{row['slug']} has no checkout; call action='checkout' first")
        running = lab.conn.execute("SELECT run_id FROM runs WHERE exp_id=? AND status='running'",
                                   (row["exp_id"],)).fetchone()
        parent = lab.head(row["branch"])
        sha, changed = lab.snapshot(tree, index, row["branch"], parent,
                                    message.strip() or f"{row['title']}")
        if not changed:
            return {"ok": True, "nothing_to_commit": True, "commit": sha}
        base = row["fork_sha"] or parent
        files = lab.git("diff", "--name-status", base, sha).splitlines()
        result = {"ok": True, "exp_id": row["exp_id"], "slug": row["slug"], "commit": sha,
                  "changed_vs_parent": files[:200],
                  "next": f"experiment(action='run', exp='{row['slug']}')"}
        if running:
            result["warning"] = (f"run {running['run_id']} is still measuring the previous commit; "
                                 "its result will not describe this one")
        return result
    finally:
        lab.close()


def diff(root: Path, exp: str, against: str = "") -> dict[str, Any]:
    lab = open_lab(root)
    try:
        row = lab.experiment(exp)
        head = lab.head(row["branch"])
        if against:
            other = lab.experiment(against)
            base, base_label = lab.head(other["branch"]), other["slug"]
        else:
            base, base_label = row["fork_sha"], "parent (fork point)"
        if not base:
            return {"ok": True, "slug": row["slug"], "diff": "", "note": "the baseline has no parent"}
        stat = lab.git("diff", "--stat", base, head)
        patch = lab.git("diff", base, head)
        return {"ok": True, "slug": row["slug"], "against": base_label, "stat": stat,
                "diff": patch[:DIFF_CHARS], "truncated": len(patch) > DIFF_CHARS}
    finally:
        lab.close()


def _detached_kwargs() -> list[dict[str, Any]]:
    """Ways to start the watcher so it outlives this server, most detached first."""
    base = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL, "close_fds": True}
    if sys.platform == "win32":
        detached, group, no_window, breakaway = 0x08, 0x200, 0x08000000, 0x01000000
        # Breaking away from the client's job object is what lets a run survive
        # the editor closing; a job that forbids breakaway refuses the flag with
        # access denied, and then plain detachment is the best available.
        return [{**base, "creationflags": detached | group | no_window | breakaway},
                {**base, "creationflags": detached | group | no_window}]
    return [{**base, "start_new_session": True}]


def start_run(root: Path, exp: str, timeout_seconds: float | None = None,
              force: bool = False) -> dict[str, Any]:
    lab = open_lab(root)
    try:
        row = lab.experiment(exp)
        refresh_all(lab)
        command = lab.meta("command") or ""
        if not command:
            raise LabError("the lab has no command; set one with action='set_command'")

        busy = lab.conn.execute("SELECT run_id FROM runs WHERE exp_id=? AND status='running'",
                                (row["exp_id"],)).fetchone()
        if busy:
            raise LabError(f"run {busy['run_id']} is already measuring {row['slug']}; "
                           "wait for it or cancel it")

        dirty = _dirty(lab, row)
        if dirty:
            raise LabError(f"{row['slug']} has uncommitted edits ({', '.join(dirty[:8])}). A run "
                           "measures the committed code only, so these would silently not be "
                           "tested. Commit them first with action='commit'.")

        streak = 0
        for past in lab.runs_for(row["exp_id"]):
            if past["status"] == "cancelled":
                continue
            if past["status"] in UNANSWERED_FAILURES and not past["answered"]:
                streak += 1
                continue
            break
        if streak >= REPAIR_CAP and not force:
            raise LabError(
                f"{row['slug']} has failed {streak} runs in a row without answering anything. "
                "That is usually a setup problem, not a code problem: stop and ask the user, or "
                "pass force=True if the next attempt is genuinely different.")

        sha = lab.head(row["branch"])
        run_id = new_id("run")
        directory = lab.run_dir(run_id)
        source = directory / "src"
        index = directory / "index"
        lab.materialize(sha, source, index)
        index.unlink(missing_ok=True)

        log = directory / "log.txt"
        log.write_bytes(b"")
        spec = {
            "command": command, "cwd": str(source), "log": str(log),
            "timeout_seconds": timeout_seconds or None,
            # Streaming and encoding defaults, learned the hard way by
            # OpenResearch: block-buffered Python output makes a live log look
            # frozen, and Windows' ANSI codepage crashes on the first non-Latin
            # character. Inert for anything that is not Python.
            "env": {"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
                    "ICN_RUN_ID": run_id, "ICN_EXPERIMENT": row["slug"]},
        }
        (directory / "spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")

        lab.conn.execute(
            "INSERT INTO runs (run_id, exp_id, commit_sha, command, status, started_at)"
            " VALUES (?,?,?,?,'running',?)",
            (run_id, row["exp_id"], sha, command, _now()),
        )
        runner = Path(labrun.__file__).resolve()
        process = None
        failure: OSError | None = None
        for kwargs in _detached_kwargs():
            try:
                process = subprocess.Popen([sys.executable, str(runner), str(directory)],
                                           cwd=str(directory), **kwargs)
                break
            except OSError as exc:
                failure = exc
        if process is None:
            lab.conn.execute("UPDATE runs SET status='failed', answered=0, ended_at=?, error=? "
                             "WHERE run_id=?", (_now(), f"could not start: {failure}", run_id))
            raise LabError(f"could not start the run watcher: {failure}")
        lab.conn.execute("UPDATE runs SET runner_pid=? WHERE run_id=?", (process.pid, run_id))

        result = {
            "ok": True, "run_id": run_id, "exp_id": row["exp_id"], "slug": row["slug"],
            "commit": sha, "command": command, "log": str(log),
            "next": ("experiment(action='wait') returns when this or any other run finishes; "
                     f"experiment(action='log', run='{run_id}') reads output so far"),
        }
        if streak >= REPAIR_CAP:
            result["warning"] = f"forced past {streak} consecutive unanswered failures"
        return result
    finally:
        lab.close()


def status(root: Path, run: str = "", exp: str = "") -> dict[str, Any]:
    lab = open_lab(root)
    try:
        refresh_all(lab)
        if run:
            return {"ok": True, **_run_view(lab, lab.run_row(run))}
        row = lab.experiment(exp)
        runs = lab.runs_for(row["exp_id"], limit=10)
        frozen = lab.frozen_by(row["exp_id"])
        return {
            "ok": True, "exp_id": row["exp_id"], "slug": row["slug"], "title": row["title"],
            "hypothesis": row["hypothesis"], "branch": row["branch"],
            "commit": lab.head(row["branch"]), "command": lab.meta("command"),
            "state": "frozen" if frozen else "provisional",
            "frozen_by": frozen["run_id"] if frozen else None,
            "promoted": row["promoted_at"] is not None,
            "uncommitted": _dirty(lab, row),
            "runs": [_run_view(lab, r) for r in runs],
        }
    finally:
        lab.close()


def read_log(root: Path, run: str, tail: int = DEFAULT_LOG_TAIL, offset: int | None = None) -> dict[str, Any]:
    lab = open_lab(root)
    try:
        refresh_all(lab)
        row = lab.run_row(run)
        log = lab.run_dir(row["run_id"]) / "log.txt"
        try:
            data = log.read_bytes()
        except OSError:
            data = b""
        total = len(data)
        size = max(1, int(tail or DEFAULT_LOG_TAIL))
        start = max(0, total - size) if offset is None else max(0, min(int(offset), total))
        chunk = data[start:start + size]
        return {"ok": True, "run_id": row["run_id"], "status": row["status"],
                "start_byte": start, "end_byte": start + len(chunk), "total_bytes": total,
                "more_above": start > 0, "more_below": start + len(chunk) < total,
                "text": chunk.decode("utf-8", errors="replace")}
    finally:
        lab.close()


def cancel(root: Path, run: str) -> dict[str, Any]:
    lab = open_lab(root)
    try:
        row = _refresh(lab, lab.run_row(run))
        if row["status"] != "running":
            return {"ok": True, "run_id": row["run_id"], "status": row["status"],
                    "note": "not running; nothing to cancel"}
        directory = lab.run_dir(row["run_id"])
        (directory / "cancel").write_text(_now(), encoding="utf-8")
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline and not (directory / "exit.json").exists():
            time.sleep(0.2)
        row = _refresh(lab, lab.run_row(row["run_id"]))
        return {"ok": True, "run_id": row["run_id"], "status": row["status"],
                "note": None if row["status"] != "running" else
                "cancel requested; the watcher has not confirmed yet"}
    finally:
        lab.close()


def wait(root: Path, exp: str = "", run: str = "", timeout: float = 60.0) -> dict[str, Any]:
    """Block until one watched run finishes, all are finished, or time runs out.

    Returns on the FIRST completion, not after the whole batch, so the caller
    can read a result and refill the slot. Like OpenResearch's `exp wait`, the
    return is a wake-up signal, not a ledger: a run that finishes while the
    caller is busy is reported by status/tree, not by the next wait.
    """
    limit = max(0.0, min(float(timeout or 0), MAX_WAIT_SECONDS))
    deadline = time.monotonic() + limit
    lab = open_lab(root)
    try:
        exp_id = lab.experiment(exp)["exp_id"] if exp else None

        def watched() -> list[dict[str, Any]]:
            refresh_all(lab)
            if run:
                return [lab.run_row(run)]
            query = "SELECT * FROM runs WHERE status='running'"
            args: tuple = ()
            if exp_id:
                query += " AND exp_id=?"
                args = (exp_id,)
            return [dict(r) for r in lab.conn.execute(query, args)]

        initial = watched()
        pending = {r["run_id"] for r in initial if r["status"] == "running"}
        if not pending:
            return {"ok": True, "drained": True, "finished": [
                _run_view(lab, r) for r in initial if r["status"] != "running"],
                "note": "no runs in flight"}
        while True:
            finished = [lab.run_row(i) for i in pending]
            finished = [_refresh(lab, r) for r in finished]
            done = [r for r in finished if r["status"] != "running"]
            if done:
                still = [r["run_id"] for r in finished if r["status"] == "running"]
                return {"ok": True, "drained": not still, "finished": [_run_view(lab, r) for r in done],
                        "still_running": still,
                        "next": "read each finished run's log before judging it, then "
                                "action='conclude'"}
            if time.monotonic() >= deadline:
                return {"ok": True, "timed_out": True, "still_running": sorted(pending),
                        "note": "nothing finished yet; call wait again"}
            time.sleep(1.0)
    finally:
        lab.close()


def list_runs(root: Path, exp: str = "", limit: int = 20) -> dict[str, Any]:
    lab = open_lab(root)
    try:
        refresh_all(lab)
        exp_id = lab.experiment(exp)["exp_id"] if exp else None
        runs = lab.runs_for(exp_id, limit=max(1, min(int(limit or 20), 200)))
        return {"ok": True, "runs": [_run_view(lab, r) for r in runs]}
    finally:
        lab.close()


def promote(root: Path, exp: str) -> dict[str, Any]:
    lab = open_lab(root)
    try:
        row = lab.experiment(exp)
        refresh_all(lab)
        if lab.frozen_by(row["exp_id"]) is None:
            raise LabError(f"{row['slug']} has not been answered by a run; only a measured result "
                           "can be the base for the next round")
        lab.conn.execute("UPDATE experiments SET promoted_at=? WHERE exp_id=?", (_now(), row["exp_id"]))
        return {"ok": True, "exp_id": row["exp_id"], "slug": row["slug"], "focal": True,
                "next": "new experiments now default to this one as their parent"}
    finally:
        lab.close()


def conclude(root: Path, verdict: str, exp: str = "", run: str = "", note: str = "") -> dict[str, Any]:
    """Judge a finished run. Returns the lab facts; the caller writes the memory.

    win / loss / inconclusive mark the run as having answered its experiment,
    which freezes the experiment (even a failed exit can be an answer, for a
    hypothesis about memory or time). void marks a run that exited cleanly but
    answered nothing, which unfreezes it; it needs a note, because it is the
    one escape hatch from rule 2 and should be auditable.
    """
    verdict = (verdict or "").lower().strip()
    if verdict not in VERDICTS:
        raise LabError(f"verdict must be one of {', '.join(VERDICTS)}")
    lab = open_lab(root)
    try:
        refresh_all(lab)
        if run:
            run_row = lab.run_row(run)
        else:
            row = lab.experiment(exp)
            # The newest run still awaiting judgement: a voided or cancelled run
            # is not what "conclude this experiment" means.
            finished = [r for r in lab.runs_for(row["exp_id"])
                        if r["status"] in TERMINAL and r["status"] != "cancelled" and not r["verdict"]]
            if not finished:
                raise LabError(f"{row['slug']} has no finished, unjudged run; pass run= to name one")
            run_row = finished[0]
        if run_row["status"] not in TERMINAL:
            raise LabError(f"run {run_row['run_id']} is still {run_row['status']}; wait for it first")
        if run_row["verdict"]:
            raise LabError(f"run {run_row['run_id']} was already judged {run_row['verdict']!r}; "
                           "verdicts are permanent")
        row = lab.experiment(run_row["exp_id"])

        if verdict == "void":
            if not note.strip():
                raise LabError("verdict='void' needs note: why this run did not answer the question")
            lab.conn.execute("UPDATE runs SET answered=0, verdict='void', note=? WHERE run_id=?",
                             (note.strip(), run_row["run_id"]))
            return {"ok": True, "run_id": run_row["run_id"], "slug": row["slug"], "verdict": "void",
                    "frozen": lab.frozen_by(row["exp_id"]) is not None, "memory_payload": None}

        lab.conn.execute("UPDATE runs SET answered=1, verdict=?, note=? WHERE run_id=?",
                         (verdict, note.strip() or None, run_row["run_id"]))
        if verdict == "win":
            lab.conn.execute("UPDATE experiments SET promoted_at=? WHERE exp_id=?",
                             (_now(), row["exp_id"]))
        run_row = lab.run_row(run_row["run_id"])
        payload = _memory_payload(lab, row, run_row, verdict, note.strip())
        return {"ok": True, "run_id": run_row["run_id"], "exp_id": row["exp_id"], "slug": row["slug"],
                "verdict": verdict, "frozen": True, "promoted": verdict == "win",
                "memory_payload": payload}
    finally:
        lab.close()


def _format_metrics(metrics: dict[str, float]) -> str:
    return ", ".join(f"{k}={v:g}" for k, v in sorted(metrics.items())) or "no ICN_METRIC lines"


def evidence_text(lab: Lab, run: dict[str, Any]) -> str:
    exp = lab.experiment(run["exp_id"])
    return (f"Evidence: lab run {run['run_id']} of experiment '{exp['title']}' "
            f"(branch {exp['branch']}, commit {run['commit_sha'][:12]}), command `{run['command']}`, "
            f"status {run['status']}, exit {run['exit_code']}, "
            f"{_duration(run) or 0:g}s, metrics: {_format_metrics(_metrics(run))}.")


def _memory_payload(lab: Lab, exp: dict[str, Any], run: dict[str, Any], verdict: str,
                    note: str) -> dict[str, Any]:
    metrics = _metrics(run)
    comparison = ""
    parent = lab.experiment(exp["parent_id"]) if exp["parent_id"] else None
    if parent:
        answered = [r for r in lab.runs_for(parent["exp_id"]) if r["answered"] and _metrics(r)]
        if answered:
            base = _metrics(answered[0])
            deltas = [f"{k} {base[k]:g} -> {v:g} ({v - base[k]:+g})"
                      for k, v in sorted(metrics.items()) if k in base]
            if deltas:
                comparison = f" Against parent '{parent['title']}' (run {answered[0]['run_id']}): " \
                             + "; ".join(deltas) + "."

    changed: list[str] = []
    touched: dict[str, list[str]] = {}
    if exp["fork_sha"]:
        changed = [p for p in lab.git("diff", "--name-only", exp["fork_sha"],
                                      run["commit_sha"]).splitlines() if p.strip()]
        touched = _touched_identifiers(lab.git("diff", "-U0", "--no-color", exp["fork_sha"],
                                               run["commit_sha"]))

    hypothesis = f" ({exp['hypothesis']})" if exp["hypothesis"] else ""
    outcome = {"win": "won", "loss": "did not help", "inconclusive": "was inconclusive"}[verdict]
    claim = (f"Experiment '{exp['title']}'{hypothesis} {outcome}: {_format_metrics(metrics)}."
             f"{comparison}")
    if note:
        claim += f" {note}"
    claim += f" {evidence_text(lab, run)}"
    reasoning = ("Measured in the ICN experiment lab, not asserted: the run executed the exact "
                 "committed code with the lab's fixed command. "
                 + (f"Files changed against the parent: {', '.join(changed[:20])}." if changed
                    else "This is the baseline, with no parent to compare against."))

    # One complete memory per verdict. An event kind that produces a summary
    # memory would make that thin headline the primary memory, which is what
    # memory(get) returns and what causal links attach to, while the metrics
    # and evidence sat in a sibling. kind='note' produces no summary memory, so
    # the typed claim below is the primary and carries everything.
    payload: dict[str, Any] = {
        "kind": "note",
        "summary": f"Experiment {verdict}: {exp['title']}",
        "reasoning": reasoning,
        "files": changed[:20],
        # Identifiers on the changed lines, per file. The caller maps them to
        # indexed symbols: investigate() only surfaces memories linked to a
        # symbol, so a file-only anchor would store the lesson and never show it.
        "touched_identifiers": touched,
        # The parent's own verdict, when it has one. The child's code was built
        # from the parent's commit, so "parent result LED_TO child result" is a
        # recorded lineage, not causality inferred from timing.
        "lineage_memory": _lineage_memory(lab, parent),
    }
    field = {"win": "decisions", "loss": "failed_attempts", "inconclusive": "rationale_notes"}[verdict]
    payload[field] = [claim]
    if verdict == "win" and metrics and comparison:
        payload["performance"] = [f"'{exp['title']}' measured {_format_metrics(metrics)}.{comparison}"
                                  f" {evidence_text(lab, run)}"]
    return payload


def _lineage_memory(lab: Lab, parent: dict[str, Any] | None) -> str | None:
    if parent is None:
        return None
    row = lab.conn.execute(
        "SELECT memory_id FROM runs WHERE exp_id=? AND memory_id IS NOT NULL AND verdict IN"
        " ('win','loss','inconclusive') ORDER BY started_at DESC LIMIT 1",
        (parent["exp_id"],)).fetchone()
    return row["memory_id"] if row else None


_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def _touched_identifiers(patch: str, per_file: int = 400) -> dict[str, list[str]]:
    """Names appearing on changed lines or hunk headers, grouped by new file path.

    Hunk headers matter as much as the lines: git names the enclosing function
    there (`@@ -3,5 +3,2 @@ def sort_items`), and a body-only change mentions
    nothing but local variables.
    """
    found: dict[str, set[str]] = {}
    current: str | None = None
    for line in patch.splitlines():
        if line.startswith("+++ "):
            target = line[4:].strip()
            current = target[2:] if target.startswith("b/") else None
            if current:
                found.setdefault(current, set())
        elif current is None:
            continue
        elif line.startswith("@@"):
            found[current].update(_IDENTIFIER.findall(line.split("@@", 2)[-1]))
        elif line[:1] in "+-" and not line.startswith(("+++", "---")):
            found[current].update(_IDENTIFIER.findall(line[1:]))
    return {path: sorted(names)[:per_file] for path, names in found.items() if names}


def link_evidence(root: Path, memory_ids: list[str], run_ids: list[str], verdict: str | None,
                  primary: str | None = None) -> None:
    if not exists(root) or not memory_ids:
        return
    lab = Lab(root)
    try:
        for run_id in run_ids:
            run = lab.run_row(run_id)
            for memory_id in memory_ids:
                lab.conn.execute(
                    "INSERT OR IGNORE INTO evidence (memory_id, run_id, exp_id, verdict, created_at)"
                    " VALUES (?,?,?,?,?)", (memory_id, run_id, run["exp_id"], verdict, _now()))
            if verdict and primary:
                lab.conn.execute("UPDATE runs SET memory_id=? WHERE run_id=?", (primary, run_id))
    finally:
        lab.close()


def summary(root: Path) -> dict[str, Any] | None:
    """What a session opening this repository should know about its lab, cheaply.

    The item that matters most is finished runs nobody has judged: a result a
    previous session launched and never concluded is invisible to every other
    tool until someone does.
    """
    if not exists(root):
        return None
    lab = Lab(root)
    try:
        refresh_all(lab)
        counts = lab.conn.execute(
            "SELECT (SELECT COUNT(*) FROM experiments) AS experiments,"
            " (SELECT COUNT(*) FROM runs WHERE status='running') AS running").fetchone()
        unjudged = [
            {"run_id": r["run_id"], "experiment": r["slug"], "status": r["status"],
             "metrics": _metrics(dict(r))}
            for r in lab.conn.execute(
                "SELECT r.*, e.slug FROM runs r JOIN experiments e ON e.exp_id = r.exp_id"
                " WHERE r.status IN ('done','failed','timed_out','lost') AND r.verdict IS NULL"
                " ORDER BY r.started_at DESC LIMIT 5")
        ]
        focal = _focal(lab)
        result: dict[str, Any] = {
            "experiments": counts["experiments"], "running": counts["running"],
            "focal": focal["slug"] if focal else None, "command": lab.meta("command"),
            "unjudged_runs": unjudged, "setbacks_in_a_row": _setbacks(lab),
        }
        if unjudged:
            result["next"] = ("finished runs are waiting for a verdict: read each log, then "
                              "experiment(action='conclude', run=..., verdict=...)")
        elif counts["running"]:
            result["next"] = "runs are in flight: experiment(action='wait')"
        return result
    finally:
        lab.close()


def apply_plan(root: Path, exp: str) -> dict[str, Any]:
    """The patch that brings an experiment's code into the working tree.

    The patch runs from the baseline snapshot to the experiment's head, so it
    carries every step of the lineage. It applies cleanly only while the files
    it touches still match the baseline, which is exactly the condition under
    which applying it is safe.
    """
    lab = open_lab(root)
    try:
        row = lab.experiment(exp)
        refresh_all(lab)
        if lab.frozen_by(row["exp_id"]) is None:
            raise LabError(f"{row['slug']} has not been measured; apply a result, not a guess")
        path, cursor = [], row
        while cursor is not None:
            path.append(cursor)
            cursor = lab.experiment(cursor["parent_id"]) if cursor["parent_id"] else None
        path.reverse()
        baseline = lab.git("rev-list", "--max-parents=0", lab.head(row["branch"])).splitlines()[0]
        head = lab.head(row["branch"])
        patch = lab.git("diff", "--binary", "--full-index", baseline, head)
        lineage = [
            {"slug": node["slug"], "verdict": judged["verdict"], "memory_id": judged["memory_id"]}
            for node in path
            for judged in [lab.conn.execute(
                "SELECT verdict, memory_id FROM runs WHERE exp_id=? AND verdict IN"
                " ('win','loss','inconclusive') ORDER BY started_at DESC LIMIT 1",
                (node["exp_id"],)).fetchone()]
            if judged is not None
        ]
        measured = [r["memory_id"] for r in lab.conn.execute(
            "SELECT DISTINCT e.memory_id FROM evidence e JOIN runs r ON r.run_id = e.run_id"
            " WHERE r.verdict IN ('win','loss','inconclusive')")]
        return {"slug": row["slug"], "commit": head, "baseline": baseline, "patch": patch,
                "files": [p for p in lab.git("diff", "--name-only", baseline, head).splitlines() if p],
                "lineage": lineage, "measured_memories": measured}
    finally:
        lab.close()


def evidence_for(root: Path, run_ids: list[str]) -> list[str]:
    """Evidence sentences for record(evidence=[...]). Only finished runs count."""
    lab = open_lab(root)
    try:
        refresh_all(lab)
        lines = []
        for run_id in run_ids:
            run = lab.run_row(run_id)
            if run["status"] not in TERMINAL:
                raise LabError(f"run {run_id} is still {run['status']}; a result that has not "
                               "happened yet is not evidence")
            lines.append(evidence_text(lab, run))
        return lines
    finally:
        lab.close()


def tree(root: Path) -> dict[str, Any]:
    lab = open_lab(root)
    try:
        refresh_all(lab)
        rows = lab.experiments()
        children: dict[str | None, list[dict[str, Any]]] = {}
        for row in rows:
            children.setdefault(row["parent_id"], []).append(row)
        memories: dict[str, list[str]] = {}
        for link in lab.conn.execute("SELECT exp_id, memory_id FROM evidence"):
            memories.setdefault(link["exp_id"], []).append(link["memory_id"])

        focal = _focal(lab)
        nodes: list[dict[str, Any]] = []

        def visit(row: dict[str, Any], depth: int) -> None:
            runs = lab.runs_for(row["exp_id"])
            frozen = lab.frozen_by(row["exp_id"])
            judged = next((r for r in runs if r["verdict"] and r["verdict"] != "void"), None)
            latest = runs[0] if runs else None
            nodes.append({
                "exp_id": row["exp_id"], "slug": row["slug"], "title": row["title"],
                "hypothesis": row["hypothesis"], "parent": row["parent_id"], "depth": depth,
                "state": "frozen" if frozen else "provisional",
                "verdict": judged["verdict"] if judged else None,
                "promoted": row["promoted_at"] is not None,
                "focal": bool(focal and focal["exp_id"] == row["exp_id"]),
                "children": len(children.get(row["exp_id"], [])),
                "runs": len(runs),
                "latest_run": ({"run_id": latest["run_id"], "status": latest["status"],
                                "metrics": _live_metrics(lab, latest)} if latest else None),
                "memories": memories.get(row["exp_id"], []),
            })
            for child in children.get(row["exp_id"], []):
                visit(child, depth + 1)

        for root_row in children.get(None, []):
            visit(root_row, 0)

        return {"ok": True, "command": lab.meta("command"),
                "focal": focal["slug"] if focal else None,
                "experiments": nodes, "warnings": _shape_warnings(rows, children),
                "setbacks_in_a_row": _setbacks(lab),
                "stop_hint": (f"{SETBACK_STOP} or more losses or failures in a row: consider "
                              "stopping and reporting" if _setbacks(lab) >= SETBACK_STOP else None)}
    finally:
        lab.close()


def _shape_warnings(rows: list[dict[str, Any]],
                    children: dict[str | None, list[dict[str, Any]]]) -> list[str]:
    """The two wrong tree shapes: a flat fan and a noodle."""
    warnings: list[str] = []
    for row in rows:
        kids = children.get(row["exp_id"], [])
        width = FAN_WIDTH - 1 if row["parent_id"] is None else FAN_WIDTH
        if len(kids) >= width and not any(children.get(k["exp_id"]) for k in kids):
            warnings.append(
                f"flat fan under {row['slug']!r}: {len(kids)} children and no grandchildren. Every "
                "result is measured against the same start, so wins never build on each other. "
                "Promote the best one and put the next round under it.")
    for row in rows:
        if row["parent_id"] is not None and len(children.get(row["parent_id"], [])) == 1:
            continue
        length, cursor = 0, row
        while len(children.get(cursor["exp_id"], [])) == 1:
            cursor = children[cursor["exp_id"]][0]
            length += 1
        if length >= NOODLE_DEPTH:
            warnings.append(
                f"noodle from {row['slug']!r}: {length} single-child links in a row. If those "
                "were co-equal options of one decision they should be siblings, not a chain.")
    return warnings


def _setbacks(lab: Lab) -> int:
    count = 0
    for run in lab.runs_for(None):
        if run["status"] == "running" or run["verdict"] == "void" or run["status"] == "cancelled":
            continue
        if run["verdict"] == "loss" or (run["status"] in UNANSWERED_FAILURES and not run["verdict"]):
            count += 1
            continue
        if run["verdict"] in ("win", "inconclusive") or run["status"] == "done":
            break
    return count
