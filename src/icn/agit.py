"""Agent-only git history, stored per codebase in `.agit/`.

The one thing carried over from the old Infinite Code server, and the one
thing that deliberately does not live in the central store.

An agent checkpoint is a snapshot of these exact files on disk right now. It is
meaningless without the working tree it came from, has to survive a wipe of the
central store, and has to be discardable by deleting one folder. So: in the
repo, gitignored, per worktree.

Implementation is the old wrapper's approach and it was already right - a bare
repo driven with `git --git-dir=... --work-tree=...`, no libgit2, no extra
dependency. Changes here: `.agit/` instead of `.infinitecode/agent_vcs/*.git`,
auto-init on first use, and automatic .gitignore registration.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

AGIT_DIR = ".agit"
DEFAULT_BRANCH = "main"
DEFAULT_USER_NAME = "Infinite Code Agent"
DEFAULT_USER_EMAIL = "agent@infinite-code.local"
TIMEOUT = 30.0


def agit_path(work_tree: Path) -> Path:
    return (work_tree / AGIT_DIR).resolve()


def exists(work_tree: Path) -> bool:
    return (agit_path(work_tree) / "HEAD").exists()


def ensure_within(raw_path: str, work_tree: Path) -> str:
    """Keep every path argument inside the working tree.

    Carried over from the old implementation unchanged, because it is correct:
    without it a crafted `paths` entry could stage files from anywhere on disk.
    """
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = (work_tree / candidate).resolve(strict=False)
    else:
        candidate = candidate.resolve(strict=False)
    try:
        relative = candidate.relative_to(work_tree)
    except ValueError as exc:
        raise ValueError(f"path must stay inside the work tree: {raw_path}") from exc
    return relative.as_posix()


def _run(work_tree: Path, args: list[str], timeout: float = TIMEOUT) -> dict[str, Any]:
    git_dir = agit_path(work_tree)
    command = ["git", "-C", str(work_tree), f"--git-dir={git_dir}",
               f"--work-tree={work_tree}", *args]
    try:
        # stdin detached: see the note in identity.run_git. A git child that
        # inherits this server's stdin can eat the MCP protocol stream.
        proc = subprocess.run(command, capture_output=True, stdin=subprocess.DEVNULL,
                              text=True, encoding="utf-8", errors="replace",
                              timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"returncode": 127, "stdout": "", "stderr": str(exc), "command": command}
    return {
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "").strip(),
        "stderr": (proc.stderr or "").strip(),
        "command": command,
    }


def _register_gitignore(work_tree: Path) -> bool:
    """Add .agit/ to the repo's .gitignore, once. Never duplicates the line."""
    ignore_file = work_tree / ".gitignore"
    line = f"{AGIT_DIR}/"
    try:
        if ignore_file.exists():
            content = ignore_file.read_text(encoding="utf-8", errors="replace")
            if re.search(rf"^{re.escape(AGIT_DIR)}/?\s*$", content, re.MULTILINE):
                return False
            separator = "" if content.endswith("\n") or not content else "\n"
            ignore_file.write_text(f"{content}{separator}{line}\n", encoding="utf-8")
        else:
            ignore_file.write_text(f"{line}\n", encoding="utf-8")
        return True
    except OSError:
        return False


def init(work_tree: Path, user_name: str = DEFAULT_USER_NAME,
         user_email: str = DEFAULT_USER_EMAIL) -> dict[str, Any]:
    """Create the isolated repo. Idempotent - safe to call on every operation."""
    git_dir = agit_path(work_tree)
    if exists(work_tree):
        return {"ok": True, "already": True, "git_dir": str(git_dir)}

    git_dir.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["git", "init", "--bare", f"--initial-branch={DEFAULT_BRANCH}", str(git_dir)],
        capture_output=True, stdin=subprocess.DEVNULL, text=True, encoding="utf-8",
        errors="replace", timeout=TIMEOUT, check=False,
    )
    if proc.returncode != 0 and not exists(work_tree):
        return {"ok": False, "error": (proc.stderr or proc.stdout or "git init failed").strip()}

    for key, value in (("user.name", user_name), ("user.email", user_email),
                       ("advice.detachedHead", "false"), ("core.autocrlf", "false")):
        _run(work_tree, ["config", key, value])

    ignored = _register_gitignore(work_tree)
    return {"ok": True, "created": True, "git_dir": str(git_dir), "gitignore_updated": ignored}


def _ensure(work_tree: Path) -> dict[str, Any] | None:
    if exists(work_tree):
        return None
    result = init(work_tree)
    return None if result.get("ok") else result


def run(work_tree: Path, action: str, *, paths_arg: list[str] | None = None,
        message: str | None = None, branch: str | None = None, create: bool = False,
        target: str | None = None, source: str = "HEAD", cached: bool = False,
        hard: bool = False, allow_empty: bool = False, limit: int = 20) -> dict[str, Any]:
    """Dispatch one agit action. Auto-initialises on first use."""
    work_tree = work_tree.resolve()
    failure = _ensure(work_tree)
    if failure is not None:
        return failure

    try:
        rel_paths = [ensure_within(p, work_tree) for p in (paths_arg or [])]
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    act = action.lower().strip()

    if act == "init":
        return init(work_tree)

    if act == "status":
        status = _run(work_tree, ["status", "--porcelain"])
        branch_now = _run(work_tree, ["rev-parse", "--abbrev-ref", "HEAD"])
        entries = [line for line in status["stdout"].splitlines() if line.strip()]
        return {"ok": status["returncode"] == 0, "branch": branch_now["stdout"] or DEFAULT_BRANCH,
                "changed_files": len(entries), "entries": entries[:200],
                "stderr": status["stderr"] or None}

    if act == "diff":
        args = ["diff"]
        if cached:
            args.append("--cached")
        if target:
            args.append(target)
        if rel_paths:
            args += ["--", *rel_paths]
        result = _run(work_tree, args)
        return {"ok": result["returncode"] == 0, "diff": result["stdout"],
                "stderr": result["stderr"] or None}

    if act == "commit":
        add_args = ["add", "--", *rel_paths] if rel_paths else ["add", "-A"]
        added = _run(work_tree, add_args)
        if added["returncode"] != 0:
            return {"ok": False, "error": added["stderr"] or "git add failed"}
        commit_args = ["commit", "-m", message or "agent checkpoint"]
        if allow_empty:
            commit_args.append("--allow-empty")
        result = _run(work_tree, commit_args)
        if result["returncode"] != 0:
            nothing = "nothing to commit" in (result["stdout"] + result["stderr"]).lower()
            return {"ok": nothing, "nothing_to_commit": nothing,
                    "error": None if nothing else (result["stderr"] or result["stdout"])}
        sha = _run(work_tree, ["rev-parse", "HEAD"])["stdout"]
        return {"ok": True, "commit": sha, "message": message or "agent checkpoint",
                "output": result["stdout"]}

    if act == "log":
        args = ["log", f"-{max(1, min(limit, 200))}", "--pretty=format:%H%x1f%an%x1f%ad%x1f%s", "--date=iso"]
        if target:
            args.append(target)
        result = _run(work_tree, args)
        entries = []
        for line in result["stdout"].splitlines():
            parts = line.split("\x1f")
            if len(parts) == 4:
                entries.append({"commit": parts[0], "author": parts[1],
                                "date": parts[2], "message": parts[3]})
        return {"ok": result["returncode"] == 0, "entries": entries,
                "stderr": result["stderr"] or None}

    if act == "branches":
        result = _run(work_tree, ["branch", "--format=%(refname:short)"])
        return {"ok": result["returncode"] == 0,
                "branches": [b for b in result["stdout"].splitlines() if b.strip()]}

    if act == "switch":
        if not branch:
            return {"ok": False, "error": "switch requires `branch`"}
        args = ["switch"] + (["-c"] if create else []) + [branch]
        result = _run(work_tree, args)
        return {"ok": result["returncode"] == 0, "branch": branch,
                "error": result["stderr"] if result["returncode"] != 0 else None}

    if act == "restore":
        if not rel_paths:
            return {"ok": False, "error": "restore requires `paths`"}
        result = _run(work_tree, ["checkout", source, "--", *rel_paths])
        return {"ok": result["returncode"] == 0, "restored": rel_paths, "source": source,
                "error": result["stderr"] if result["returncode"] != 0 else None}

    if act == "reset":
        args = ["reset", "--hard" if hard else "--mixed", target or "HEAD"]
        result = _run(work_tree, args)
        return {"ok": result["returncode"] == 0, "target": target or "HEAD", "hard": hard,
                "output": result["stdout"], "error": result["stderr"] or None}

    if act == "show":
        args = ["show", target or "HEAD"]
        if rel_paths:
            args += ["--", *rel_paths]
        result = _run(work_tree, args)
        return {"ok": result["returncode"] == 0, "output": result["stdout"][:20000],
                "error": result["stderr"] or None}

    return {"ok": False, "error": f"unknown agit action: {action}"}
