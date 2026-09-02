"""Install ICN's agent skill and hooks into a project.

ICN is only useful if the agent actually calls it, and until now that depended
entirely on the user having written the right paragraph into a global
CLAUDE.md by hand. That is a silent failure mode: the server is registered, the
tools list fine, `doctor` says READY, and the agent still never calls any of it
because nothing told it to. Shipping the instruction with the software closes
that gap.

Two things are written, and they do different jobs:

    a skill    the workflow, in full, loaded when the agent needs it.
    a hook     one line at session start, so the agent knows the skill is there.

Everything here is idempotent and additive. Existing settings are merged, never
replaced: a hook installer that overwrites a user's own hooks has done more
damage than the problem it solves.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SKILL_DIR = ".claude/skills/icn-workflow"
SETTINGS_PATH = ".claude/settings.json"

# Marker on the hook we own, so a re-install replaces our entry and leaves
# every other hook in the file untouched.
HOOK_MARKER = "icn-workflow-reminder"

SKILL = """---
name: icn-workflow
description: >-
  Use the Infinite Code Next (ICN) MCP server for any task touching an existing
  codebase: investigating, diagnosing, designing, or modifying code. Covers
  opening a workspace, investigating before editing, checking blast radius,
  checkpointing risky work, and recording what was learned. Use when starting
  work in a repository, before changing a load-bearing symbol, or when asked
  why code is shaped the way it is.
---

# Working with ICN

ICN stores durable knowledge about this codebase - decisions, invariants, past
bugs, and approaches already tried and rejected - anchored to real symbols. It
is not a search index with extra steps: the expensive knowledge it holds is the
kind that cannot be recovered by reading the code, because it is about code
that is no longer there.

## The loop

1. `workspace(action="open", root=<repo>)` once per session, and again on every
   repository switch. Read the briefing it returns before opening any file.
2. `investigate(query=<what you are about to do>, intent=<intent>)` before
   investigating, diagnosing, designing or modifying. It searches code and
   knowledge together, so prefer it over grep for anything but a literal
   string.
3. `graph(action="impact", target=<symbol>)` before changing a symbol other
   things depend on. `graph(action="trace", target=A, to=B)` to prove two
   symbols are actually connected rather than assuming.
4. `agit(action="commit", message=...)` before a risky edit or broad refactor.
   It commits to `.agit/`, never the user's `.git`.
5. `record(...)` after a verified finding or change.

Skip steps 2 and 5 only for purely mechanical work - fixing a typo, running a
command you were explicitly asked to run.

## Reading results honestly

**Anchor status.** Every memory carries one. Anything other than `ACTIVE` has
not been checked against the current code: treat it as a lead, not a fact, and
call `memory(action="verify")` once you have confirmed it still applies.

**The epistemic envelope.** `graph()` results carry `epistemic`, `boundaries`
and `causes`. Read them before the result itself:

- `epistemic: "exact"` - the listing is complete for the edges that exist.
- `epistemic: "lower-bound"` - callers exist that this answer provably does not
  list. An empty `affected` here is **not** evidence that nothing calls the
  symbol. `causes.ambiguous_call_sites` counts the ones being withheld.

**Edge tiers.** Call edges are not equally trustworthy. `receiver_self` (1.0)
and `same_file` (0.9) are read off the syntax; `imported` (0.8) and
`unique_global` (0.6) are judgements. Pass `min_confidence=0.9` to walk only
what was actually proven.

## Recording well

Write as if the next agent has no context, because it does not. State the
failure the rule prevents, the evidence it rests on, and the mechanism that
enforces it - not just the rule.

    Thin:    "settle must be idempotent"
    Useful:  "settle() must be idempotent: a retried webhook must not
              double-charge. From the Feb duplicate-charge incident, where the
              provider retried after a 502 that had already succeeded. The
              guard is the idempotency_key column, unique per invoice."

The highest-value fields are `failed_attempts` and `warnings`. Nothing else in
the toolchain records what was tried and rejected, and that is the most
expensive knowledge for the next agent to rediscover.

## If ICN is unavailable

Say so explicitly and continue with the best evidence you have. Never silently
skip it, and never claim it was used when it was not.
"""

REMINDER = (
    "ICN (infinite-code) is available for this repository. Call "
    "workspace(action='open') before reading files, and investigate() before "
    "changing code. See the icn-workflow skill for the full loop."
)


def _load_settings(path: Path) -> dict[str, Any]:
    """Read settings, tolerating absence but never silently discarding content.

    A malformed settings.json is raised rather than replaced. Overwriting a
    file we could not parse is how an installer eats someone's configuration.
    """
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    try:
        loaded = json.loads(text)
    except ValueError as exc:
        raise ValueError(
            f"{path} is not valid JSON ({exc}). Fix or move it before installing; "
            f"refusing to overwrite a file that may hold your configuration.") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} does not contain a JSON object.")
    return loaded


def _install_hook(settings: dict[str, Any]) -> bool:
    """Merge our SessionStart hook in. Returns True if the file needs writing."""
    hooks = settings.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise ValueError("settings.json 'hooks' is not an object; refusing to replace it.")
    session_start = hooks.setdefault("SessionStart", [])
    if not isinstance(session_start, list):
        raise ValueError("settings.json 'hooks.SessionStart' is not a list.")

    entry = {
        "matcher": "*",
        "hooks": [{
            "type": "command",
            "command": f"echo {json.dumps(REMINDER)}",
            "_source": HOOK_MARKER,
        }],
    }

    for index, existing in enumerate(session_start):
        if not isinstance(existing, dict):
            continue
        inner = existing.get("hooks") or []
        if any(isinstance(h, dict) and h.get("_source") == HOOK_MARKER for h in inner):
            if existing == entry:
                return False            # already exactly right
            session_start[index] = entry
            return True

    session_start.append(entry)
    return True


def install(root: Path, *, with_hook: bool = True) -> dict[str, Any]:
    """Write the skill, and optionally the session hook, into `root`."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        return {"ok": False, "error": f"{root} is not a directory"}

    written: list[str] = []
    unchanged: list[str] = []

    skill_path = root / SKILL_DIR / "SKILL.md"
    skill_path.parent.mkdir(parents=True, exist_ok=True)
    if skill_path.exists() and skill_path.read_text(encoding="utf-8") == SKILL:
        unchanged.append(str(skill_path.relative_to(root)))
    else:
        skill_path.write_text(SKILL, encoding="utf-8")
        written.append(str(skill_path.relative_to(root)))

    settings_path = root / SETTINGS_PATH
    if with_hook:
        settings = _load_settings(settings_path)
        if _install_hook(settings):
            settings_path.parent.mkdir(parents=True, exist_ok=True)
            settings_path.write_text(
                json.dumps(settings, indent=2) + "\n", encoding="utf-8")
            written.append(str(settings_path.relative_to(root)))
        else:
            unchanged.append(str(settings_path.relative_to(root)))

    return {
        "ok": True,
        "root": str(root),
        "written": written,
        "unchanged": unchanged,
        "next": ("Restart the client so the skill and hook load, then call "
                 "workspace(action='open')."),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="icn install",
        description="Install the ICN agent skill and session hook into a project")
    parser.add_argument("--root", default=".", help="project path (default: current directory)")
    parser.add_argument("--no-hook", action="store_true",
                        help="write the skill only, leaving settings.json alone")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        report = install(Path(args.root), with_hook=not args.no_hook)
    except ValueError as exc:
        report = {"ok": False, "error": str(exc)}

    if args.json:
        print(json.dumps(report, indent=2))
    elif not report["ok"]:
        print(f"install failed: {report['error']}")
    else:
        print(f"ICN installed into {report['root']}")
        for path in report["written"]:
            print(f"  wrote     {path}")
        for path in report["unchanged"]:
            print(f"  unchanged {path}")
        print(report["next"])
    return 0 if report.get("ok") else 1
