"""Repository identity and rediscovery.

PLAN 2 section 4 and PLAN.md sections 1, 2 and 13.

There is no workspace picker. The server works out where it is from the
filesystem plus git, and identity is never the path and never the remote URL -
both are mutable. `git remote set-url` must not create a new memory universe,
and neither must moving a clone from C:\\code to D:\\dev.

Identity signals, strongest first:
    root_commit   the hash of the first commit. Immutable for the life of the
                  repo, survives renames, remote changes and re-clones.
    project_id    an explicit id in a committed .icn.toml, if the team set one.
    remote        normalized, so ssh and https forms of one repo agree.
    path          last resort only, and marks identity as weak.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

GIT_TIMEOUT = 20.0


def run_git(args: list[str], cwd: Path, timeout: float = GIT_TIMEOUT,
            strip: bool = True) -> tuple[int, str, str]:
    """Run git and never raise. A missing git is a degraded mode, not a crash.

    `strip=False` matters for porcelain output: the status field is two
    characters wide and a worktree-only modification renders as " M path", so
    stripping silently eats the first column and corrupts every path.
    """
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            # stdin MUST be detached. This server speaks MCP over its own
            # stdin, so a child that inherits it can block waiting on input
            # that will never come - and worse, can consume bytes of the
            # protocol stream. Observed live as every tool call stalling for
            # exactly the git timeout, and as truncated JSON reaching the
            # client.
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, "", str(exc)
    out = proc.stdout or ""
    err = proc.stderr or ""
    return proc.returncode, (out.strip() if strip else out), err.strip()


def normalize_remote(url: str) -> str:
    """git@github.com:acme/backend.git and https://github.com/acme/backend.git
    are the same repository. Collapse both to github.com/acme/backend."""
    if not url:
        return ""
    text = url.strip()
    text = re.sub(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", "", text)
    text = re.sub(r"^[^/@]+@", "", text)          # strip user@
    text = text.replace(":", "/", 1) if "@" not in text and ":" in text.split("/")[0] else text
    text = re.sub(r"^([^/]+):", r"\1/", text)      # scp-style host:path
    if text.endswith(".git"):
        text = text[:-4]
    return text.rstrip("/").lower()


@dataclass
class Probe:
    """What the filesystem and git can tell us about a directory, right now."""

    root: Path
    is_git: bool = False
    toplevel: Path | None = None
    git_common_dir: str | None = None
    is_worktree: bool = False
    head: str | None = None
    branch: str | None = None
    remotes: list[str] = field(default_factory=list)
    root_commit: str | None = None
    dirty: bool = False
    project_id: str | None = None
    project_name: str | None = None

    @property
    def identity_strength(self) -> str:
        return "strong" if (self.root_commit or self.project_id) else ("weak" if not self.is_git else "medium")

    def aliases(self) -> list[tuple[str, str]]:
        """(kind, value) pairs this checkout can be recognised by later."""
        out: list[tuple[str, str]] = []
        if self.root_commit:
            out.append(("root_commit", self.root_commit))
        if self.project_id:
            out.append(("project_id", self.project_id))
        for remote in self.remotes:
            out.append(("remote", remote))
        out.append(("path", str(self.root)))
        return out


def probe(path: Path) -> Probe:
    """Inspect a directory. Always returns; never raises on a non-repo."""
    root = Path(path).expanduser().resolve()
    result = Probe(root=root)

    code, top, _ = run_git(["rev-parse", "--show-toplevel"], root)
    if code != 0 or not top:
        # Not a git repo. Supported, but every git-derived signal is gone.
        result.project_id, result.project_name = _read_project_file(root)
        return result

    result.is_git = True
    result.toplevel = Path(top).resolve()
    result.root = result.toplevel

    _, common, _ = run_git(["rev-parse", "--git-common-dir"], result.root)
    _, gitdir, _ = run_git(["rev-parse", "--absolute-git-dir"], result.root)
    if common:
        common_abs = Path(common)
        if not common_abs.is_absolute():
            common_abs = (result.root / common).resolve()
        result.git_common_dir = str(common_abs)
        # A linked worktree has its own git dir but shares the common dir.
        result.is_worktree = bool(gitdir) and Path(gitdir).resolve() != common_abs.resolve()

    _, head, _ = run_git(["rev-parse", "HEAD"], result.root)
    result.head = head or None
    _, branch, _ = run_git(["rev-parse", "--abbrev-ref", "HEAD"], result.root)
    result.branch = branch or None

    _, remotes_raw, _ = run_git(["remote", "-v"], result.root)
    seen: set[str] = set()
    for line in remotes_raw.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            norm = normalize_remote(parts[1])
            if norm and norm not in seen:
                seen.add(norm)
                result.remotes.append(norm)

    # The first commit. This is the strongest identity we can get for free.
    code, roots, _ = run_git(["rev-list", "--max-parents=0", "HEAD"], result.root)
    if code == 0 and roots:
        result.root_commit = roots.splitlines()[0].strip() or None

    code, status, _ = run_git(["status", "--porcelain"], result.root)
    result.dirty = bool(status.strip()) if code == 0 else False

    result.project_id, result.project_name = _read_project_file(result.root)
    return result


def _read_project_file(root: Path) -> tuple[str | None, str | None]:
    """Read the optional committed .icn.toml. Absent is the normal case."""
    config = root / ".icn.toml"
    if not config.exists():
        return None, None
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - py<3.11
        return None, None
    try:
        data = tomllib.loads(config.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None
    project = data.get("project") or {}
    pid = project.get("id")
    return (str(pid) if pid else None), (str(project.get("name")) if project.get("name") else None)


def read_repo_config(root: Path) -> dict:
    """Full .icn.toml contents: excludes, related repos, project metadata."""
    config = root / ".icn.toml"
    if not config.exists():
        return {}
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover
        return {}
    try:
        return tomllib.loads(config.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def changed_files(root: Path, since_commit: str | None) -> tuple[list[str], bool]:
    """Files touched since a commit, plus everything currently uncommitted.

    Returns (paths, complete). complete=False means "could not tell, assume
    everything changed" - the caller falls back to a full scan rather than
    silently indexing a stale subset.
    """
    paths: set[str] = set()

    code, out, _ = run_git(["status", "--porcelain", "-z"], root, strip=False)
    if code != 0:
        return [], False

    # -z records are "XY <path>\0", and a rename or copy is followed by a bare
    # "<original path>\0" record. Both sides matter: the old path's symbols need
    # tombstoning and the new path needs parsing.
    entries = [e for e in out.split("\0") if e]
    index = 0
    while index < len(entries):
        entry = entries[index]
        index += 1
        if len(entry) < 4:
            continue
        status, path = entry[:2], entry[3:]
        if path:
            paths.add(path)
        if status[0] in ("R", "C") or status[1] in ("R", "C"):
            if index < len(entries):
                original = entries[index]
                index += 1
                if original:
                    paths.add(original)

    if since_commit:
        code, out, _ = run_git(["diff", "--name-only", f"{since_commit}..HEAD"], root)
        if code != 0:
            return [], False
        paths.update(p for p in out.splitlines() if p.strip())

    return sorted(paths), True


def blame_move_evidence(root: Path, path: str, line_start: int, line_end: int) -> bool:
    """Did git see this range as moved or copied from elsewhere?

    `-M` follows moves within a file, `-C -C` follows copies and moves between
    files. Used to confirm cascade step 2 rather than to find the target.
    """
    code, out, _ = run_git(
        ["blame", "-M", "-C", "-C", "--porcelain", "-L", f"{line_start},{line_end}", "--", path],
        root,
    )
    if code != 0:
        return False
    return "previous " in out or "\nfilename " in out
