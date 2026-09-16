"""Install ICN's agent skill and hooks into a project.

ICN is only useful if the agent actually calls it, and until now that depended
entirely on the user having written the right paragraph into a global
CLAUDE.md by hand. That is a silent failure mode: the server is registered, the
tools list fine, `doctor` says READY, and the agent still never calls any of it
because nothing told it to. Shipping the instruction with the software closes
that gap.

What is written, and the job each does:

    a skill    the workflow, in full, loaded when the agent needs it.
    hooks      `icn hook session-start`: the previous session's handoff and the
               highest-standing rules. `icn hook pre-tool-use`: the knowledge
               anchored to a file, just before the agent reads or edits it.
               See hooks.py for the delivery rules.

Claude Code reads hooks from the project's .claude/settings.json. Codex reads
them from $CODEX_HOME/hooks.json (default ~/.codex/hooks.json), one file for
every repository; the hook resolves which repository from the session's cwd
and stays silent anywhere ICN has no knowledge. Codex also asks the user to
trust new hooks once, in its TUI.

Everything here is idempotent and additive. Existing settings are merged, never
replaced: a hook installer that overwrites a user's own hooks has done more
damage than the problem it solves.
"""

from __future__ import annotations

import json
import os
import sys
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
5. `record(...)` after a verified finding or change. Restating a rule that is
   already stored reinforces it (the reply lists `memories_reinforced`), so
   record what you confirmed even when it is not new.
6. Stopping mid-task? `memory(action="handoff", body=<where it stands>,
   next_steps=[...], open_questions=[...], files=[...])`. The next session
   receives it once, automatically.

## Knowledge that arrives on its own

With the hooks installed, the session starts with the previous handoff and
the highest-standing rules, and touching a file shows the rules, warnings and
rejected attempts anchored to it. Treat those lines as you would a briefing.
Then close the loop: `memory(action="feedback", memory_id=..., signal=...)`
with `helpful` when one saved you a mistake, `not_helpful` when it was noise,
`stale` or `wrong` (with a reason) when the code has moved past it. That vote
is what keeps the next agent's context short and correct.

Rules that should bind every task can be promoted into this repository's
CLAUDE.md and AGENTS.md with `memory(action="rules_recommend")` and
`rules_approve`, but only when the user asks: those files are theirs.

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


# Tools whose input names a file. Bash is included because both CLIs read
# files through the shell too; the hook only reacts to real repository files.
PRE_TOOL_MATCHER = "Read|Edit|Write|MultiEdit|NotebookEdit|Bash|apply_patch|shell|exec_command"
HOOK_COMMAND_MARK = "-m icn hook "


def hook_command(event: str, agent: str, python: str | None = None) -> str:
    """The command line for one event, quoted the way each CLI runs it.

    Measured on Windows: Codex runs hook commands through PowerShell, where a
    command starting with a quoted path is a string, not a call, and fails;
    the call operator makes it a call. Claude Code runs the quoted form.
    """
    exe = python or sys.executable
    if agent == "codex" and os.name == "nt":
        return f'& "{exe}" -m icn hook {event} --agent codex'
    return f'"{exe}" -m icn hook {event} --agent {agent}'


def _is_ours(handler: Any) -> bool:
    return isinstance(handler, dict) and (
        handler.get("_source") == HOOK_MARKER or HOOK_COMMAND_MARK in str(handler.get("command", "")))


def _install_hook(settings: dict[str, Any], agent: str = "claude", python: str | None = None) -> bool:
    """Merge our hooks in, replacing any older ICN entry. True if the file needs writing."""
    hooks = settings.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise ValueError("settings.json 'hooks' is not an object; refusing to replace it.")

    wanted = {
        "SessionStart": {"matcher": "*" if agent == "claude" else "", "hooks": [{
            "type": "command", "command": hook_command("session-start", agent, python),
            "timeout": 15}]},
        # Codex tool names differ by version (shell, exec_command, apply_patch,
        # ...); match every tool and let the hook ignore calls without a file.
        "PreToolUse": {"matcher": PRE_TOOL_MATCHER if agent == "claude" else "", "hooks": [{
            "type": "command", "command": hook_command("pre-tool-use", agent, python),
            "timeout": 10}]},
    }
    if agent == "claude":
        for entry in wanted.values():
            entry["hooks"][0]["_source"] = HOOK_MARKER

    changed = False
    for event, entry in wanted.items():
        entries = hooks.setdefault(event, [])
        if not isinstance(entries, list):
            raise ValueError(f"hooks '{event}' is not a list; refusing to replace it.")
        placed = False
        for index, existing in list(enumerate(entries)):
            if not isinstance(existing, dict):
                continue
            if any(_is_ours(h) for h in existing.get("hooks") or []):
                if not placed:
                    if existing != entry:
                        entries[index] = entry
                        changed = True
                    placed = True
                else:
                    entries[index] = None           # a duplicate from an older install
                    changed = True
        hooks[event] = [e for e in entries if e is not None]
        if not placed:
            hooks[event].append(entry)
            changed = True
    return changed


def codex_hooks_path() -> Path:
    home = os.environ.get("CODEX_HOME")
    return (Path(home) if home else Path.home() / ".codex") / "hooks.json"


def install_codex(path: Path | None = None, python: str | None = None) -> dict[str, Any]:
    """Merge ICN's hooks into Codex's user-level hooks.json."""
    path = Path(path) if path else codex_hooks_path()
    settings = _load_settings(path)
    if not _install_hook(settings, agent="codex", python=python):
        return {"ok": True, "path": str(path), "changed": False}
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.icn-tmp")
    temp.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)
    return {"ok": True, "path": str(path), "changed": True,
            "note": "Codex asks you to trust new hooks the next time it starts; accept them once."}


def install(root: Path, *, with_hook: bool = True, codex: bool = False,
            codex_path: Path | None = None) -> dict[str, Any]:
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

    if codex:
        report = install_codex(codex_path)
        (written if report["changed"] else unchanged).append(report["path"])

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
    parser.add_argument("--codex", action="store_true",
                        help="also install the hooks for Codex (~/.codex/hooks.json)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        report = install(Path(args.root), with_hook=not args.no_hook, codex=args.codex)
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
