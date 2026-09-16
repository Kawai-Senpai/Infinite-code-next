"""Deliver knowledge without being asked: Claude Code and Codex lifecycle hooks.

Adapted from base's "delivery is the product": a memory an agent does not know
exists is never queried. ICN's memories are already anchored to exact files and
symbols, so the hook can put the right few in front of the model at the moment
they matter, instead of hoping the agent thinks to ask.

    session-start   the handoff left by the previous session (claimed once),
                    the highest-standing rules, and what needs attention.
    pre-tool-use    before a file is read or edited: the invariants, warnings,
                    contracts and failed attempts anchored to that file.

Measured in both CLIs (Claude Code 2.1.273, Codex 0.154): each reads
`hookSpecificOutput.additionalContext` from stdout JSON and hands it to the
model; plain stdout on tool events is transcript-only.

Rules this module holds to, each from a documented failure elsewhere:

    fail open      any error exits 0 with no output. A broken hook must never
                   block a session (base, ai-memory).
    targeted       only memories anchored to the file being touched. Broad
                   injection burned users' quotas and was turned off by
                   default in agentmemory (#143).
    capped         a hard character budget per injection.
    once           a memory is delivered at most once per session.
    trusted only   memories voted wrong or repeatedly unhelpful, or whose code
                   is gone (ORPHANED), are not volunteered. Search still finds
                   them. Memories on code that changed since they were recorded
                   ARE delivered, labelled unverified: a rule matters most
                   exactly when its code is being changed.
    light          no parser, no index, no embedding model: this runs before
                   every tool call, so it opens two SQLite files and exits.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PRE_TOOL_BUDGET = 1400
SESSION_BUDGET = 2600
PRE_TOOL_MAX = 4
DELIVERED_KINDS = ("invariant", "warning", "contract", "security", "failed_attempt", "bug")
SEVERITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}
KIND_LABEL = {"invariant": "RULE", "warning": "WARNING", "contract": "CONTRACT",
              "security": "SECURITY", "failed_attempt": "TRIED AND REJECTED", "bug": "KNOWN BUG"}
FILE_TOOLS = {"read", "edit", "write", "multiedit", "notebookedit", "apply_patch", "view_file",
              "read_file", "write_file", "edit_file", "create_file"}
PATCH_FILE = re.compile(r"^\*\*\* (?:Update|Add|Delete) File: (.+)$", re.MULTILINE)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# ------------------------------------------------------------------ resolution

def _norm(path: str | Path) -> str:
    text = str(path).replace("\\", "/").rstrip("/")
    return text.lower() if os.name == "nt" else text


def resolve_repo(cwd: str) -> tuple[str, Path] | None:
    """(repo_id, root) for the checkout containing cwd, from the catalog alone."""
    from . import paths
    catalog_file = paths.catalog_path()
    if not catalog_file.exists():
        return None
    conn = sqlite3.connect(f"file:{catalog_file.as_posix()}?mode=ro", uri=True, timeout=2)
    try:
        found = conn.execute("SELECT repo_id, path FROM checkouts WHERE status='ACTIVE'").fetchall()
    finally:
        conn.close()
    here = _norm(Path(cwd).resolve())
    best = None
    for repo_id, path in found:
        root = _norm(path)
        if (here == root or here.startswith(root + "/")) and (best is None or len(root) > len(best[2])):
            best = (repo_id, Path(path), root)
    return (best[0], best[1]) if best else None


def _store(repo_id: str) -> sqlite3.Connection | None:
    from . import db, paths
    path = paths.repo_db_path(repo_id)
    if not path.exists():
        return None
    return db.init_repo_store(path)


def touched_files(payload: dict[str, Any], root: Path) -> list[str]:
    """Repository-relative files a tool call is about to read or change."""
    tool = str(payload.get("tool_name") or "").lower()
    tool_input = payload.get("tool_input") or {}
    if not isinstance(tool_input, dict):
        return []
    raw: list[str] = []
    if tool in FILE_TOOLS:
        for key in ("file_path", "path", "notebook_path", "filePath"):
            if isinstance(tool_input.get(key), str):
                raw.append(tool_input[key])
    command = tool_input.get("command")
    if isinstance(command, list):
        command = " ".join(str(c) for c in command)
    if isinstance(command, str):
        if "*** Begin Patch" in command:
            raw.extend(m.strip() for m in PATCH_FILE.findall(command))
        elif tool in ("bash", "shell", "shell_command", "powershell", "exec_command", "local_shell"):
            # A shell read (`cat src/x.py`, `Get-Content x`) is contact with a
            # file too. Only tokens naming a real file count.
            for token in re.findall(r"[\w./\\:-]+\.[A-Za-z0-9]{1,8}", command)[:30]:
                raw.append(token)

    cwd = Path(payload.get("cwd") or root)
    root_norm = _norm(root.resolve())
    out: list[str] = []
    for item in raw:
        candidate = Path(item)
        if not candidate.is_absolute():
            candidate = cwd / candidate
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        full = _norm(resolved)
        if not full.startswith(root_norm + "/") or not resolved.is_file():
            continue
        relative = str(resolved)[len(str(root.resolve())):].lstrip("\\/").replace("\\", "/")
        if relative.startswith((".git/", ".agit/", ".icn-lab/")) or relative in out:
            continue
        out.append(relative)
    return out[:6]


# ------------------------------------------------------------------ selection

def _trusted(memory: dict[str, Any]) -> bool:
    from .feedback import withheld_from_delivery
    return not withheld_from_delivery(memory)


def _claim_line(memory: dict[str, Any], limit: int = 260) -> str:
    text = memory.get("claim") or memory.get("body") or ""
    for marker in ("\n\nRecorded while: ", "\n\nWhy: ", "\n\nChanged: ", "\n\nApplies to: "):
        text = text.split(marker, 1)[0]
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 3].rstrip() + "..."


def memories_for_files(conn: sqlite3.Connection, files: list[str]) -> list[dict[str, Any]]:
    """Memories anchored to these files, unless their code is gone."""
    if not files:
        return []
    conn.row_factory = sqlite3.Row
    marks = ",".join("?" for _ in files)
    kinds = ",".join("?" for _ in DELIVERED_KINDS)
    found = conn.execute(
        f"SELECT m.memory_id, m.kind, m.severity, m.claim, m.body, m.confidence,"
        f" m.helpful_count, m.unhelpful_count, m.evidence_count, a.file_path, a.symbol_path,"
        f" (SELECT COUNT(*) FROM anchors c WHERE c.memory_id = m.memory_id"
        f"   AND c.status IN ('NEEDS_REVIEW','DRIFTED')) AS unverified"
        f" FROM anchors a JOIN memories m ON m.memory_id = a.memory_id"
        f" WHERE a.file_path IN ({marks}) AND m.status = 'ACTIVE' AND m.kind IN ({kinds})"
        f" AND a.status IN ('ACTIVE','NEEDS_REVIEW','DRIFTED')",
        (*files, *DELIVERED_KINDS)).fetchall()
    by_id: dict[str, dict[str, Any]] = {}
    for row in found:
        memory = dict(row)
        entry = by_id.setdefault(memory["memory_id"], {**memory, "symbols": set()})
        if memory["symbol_path"]:
            entry["symbols"].add(memory["symbol_path"])
    # A memory from a record that touched many files only applies to the
    # files its own claim names (see relevance.py).
    from .relevance import focus
    if by_id:
        marks = ",".join("?" for _ in by_id)
        anchors: dict[str, list[dict[str, Any]]] = {}
        for row in conn.execute(f"SELECT memory_id, file_path, symbol_path FROM anchors"
                                f" WHERE memory_id IN ({marks})", tuple(by_id)):
            anchors.setdefault(row["memory_id"], []).append(dict(row))
        for memory_id in list(by_id):
            memory = by_id[memory_id]
            focused = focus(_claim_line(memory, 10_000), anchors.get(memory_id, []))
            if focused["broad"] and not set(focused["files"]) & set(files):
                del by_id[memory_id]
    ranked = [m for m in by_id.values() if _trusted(m)]
    ranked.sort(key=lambda m: (1 if m["unverified"] else 0, SEVERITY_RANK.get(m["severity"], 2),
                               0 if m["kind"] in ("invariant", "security", "contract") else 1,
                               -(m["evidence_count"] or 1), -(m["helpful_count"] or 0)))
    return ranked


def _undelivered(conn: sqlite3.Connection, session_id: str, memory_ids: list[str]) -> set[str]:
    if not memory_ids:
        return set()
    marks = ",".join("?" for _ in memory_ids)
    seen = {r[0] for r in conn.execute(
        f"SELECT memory_id FROM hook_deliveries WHERE session_id = ? AND memory_id IN ({marks})",
        (session_id, *memory_ids))}
    return set(memory_ids) - seen


def _mark_delivered(conn: sqlite3.Connection, session_id: str, memory_ids: list[str], event: str,
                    agent: str, file_path: str | None = None) -> None:
    if not memory_ids:
        return
    stamp = now()
    conn.executemany(
        "INSERT OR IGNORE INTO hook_deliveries (session_id, memory_id, event, file_path, agent,"
        " delivered_at) VALUES (?,?,?,?,?,?)",
        [(session_id, m, event, file_path, agent, stamp) for m in memory_ids])
    conn.commit()


# ------------------------------------------------------------------ handlers

def pre_tool_use(payload: dict[str, Any], agent: str) -> str:
    located = resolve_repo(payload.get("cwd") or os.getcwd())
    if not located:
        return ""
    repo_id, root = located
    files = touched_files(payload, root)
    if not files:
        return ""
    conn = _store(repo_id)
    if conn is None:
        return ""
    try:
        session = str(payload.get("session_id") or "unknown")
        candidates = memories_for_files(conn, files)
        fresh = _undelivered(conn, session, [m["memory_id"] for m in candidates])
        chosen, lines, used = [], [], 0
        for memory in candidates:
            if memory["memory_id"] not in fresh or len(chosen) >= PRE_TOOL_MAX:
                continue
            where = memory["file_path"] + (f" {', '.join(sorted(memory['symbols'])[:2])}"
                                           if memory["symbols"] else "")
            line = (f"- {KIND_LABEL.get(memory['kind'], memory['kind'].upper())} "
                    f"[{memory['severity']}{', unverified: code changed since recorded' if memory['unverified'] else ''}] "
                    f"{where}: {_claim_line(memory)} ({memory['memory_id']})")
            if used + len(line) > PRE_TOOL_BUDGET:
                break
            lines.append(line)
            used += len(line)
            chosen.append(memory["memory_id"])
        if not chosen:
            return ""
        _mark_delivered(conn, session, chosen, "PreToolUse", agent, files[0])
        more = len([m for m in candidates if m["memory_id"] in fresh]) - len(chosen)
        header = f"ICN knowledge anchored to {', '.join(files)} (recorded by earlier sessions):"
        footer = ("Respect these while working on this file. If one is out of date, say so and call "
                  "memory(action='feedback', signal='stale' or 'wrong', reason=...); if one saved "
                  "you a mistake, signal='helpful'.")
        if more > 0:
            footer += f" {more} more: investigate() on this file."
        return "\n".join([header, *lines, footer])
    finally:
        conn.close()


def session_start(payload: dict[str, Any], agent: str) -> str:
    located = resolve_repo(payload.get("cwd") or os.getcwd())
    if not located:
        return ""
    repo_id, root = located
    conn = _store(repo_id)
    if conn is None:
        return ""
    try:
        from . import handoff as handoff_mod
        conn.row_factory = sqlite3.Row
        session = str(payload.get("session_id") or "unknown")
        parts = [f"ICN holds recorded knowledge for this repository ({root.name}). Call "
                 "workspace(action='open') before reading files and investigate() before changing "
                 "code; knowledge anchored to a file is also shown when you touch that file."]

        claimed = handoff_mod.claim(conn, claimed_by=f"{agent}:{session}")
        if claimed:
            parts.append(handoff_mod.render(claimed))

        kinds = ",".join("?" for _ in ("invariant", "security", "contract"))
        standing = [dict(r) for r in conn.execute(
            f"SELECT m.memory_id, m.kind, m.severity, m.claim, m.body, m.confidence,"
            f" m.helpful_count, m.unhelpful_count, m.evidence_count,"
            f" (SELECT group_concat(DISTINCT a.file_path) FROM anchors a WHERE a.memory_id = m.memory_id)"
            f" AS files FROM memories m WHERE m.status='ACTIVE' AND m.kind IN ({kinds})"
            f" AND m.severity IN ('critical','high')"
            f" AND NOT EXISTS (SELECT 1 FROM anchors b WHERE b.memory_id = m.memory_id AND b.status != 'ACTIVE')"
            f" AND m.memory_id NOT IN (SELECT memory_id FROM promoted_rules)"
            f" ORDER BY CASE m.severity WHEN 'critical' THEN 0 ELSE 1 END,"
            f" COALESCE(m.evidence_count,1) DESC, COALESCE(m.helpful_count,0) DESC LIMIT 12",
            ("invariant", "security", "contract"))]
        standing = [m for m in standing if _trusted(m)][:5]
        if standing:
            lines = ["Highest-standing rules here:"]
            for memory in standing:
                # Same relevance rule as pre-tool-use: a memory recorded with
                # many files is labelled only with the file its claim names.
                from .relevance import focus
                anchors = [{"file_path": f, "symbol_path": None}
                           for f in (memory["files"] or "").split(",") if f]
                named = focus(_claim_line(memory, 10_000), anchors)["files"]
                where = named[0] if named else ""
                lines.append(f"- [{memory['severity']}] {where + ': ' if where else ''}"
                             f"{_claim_line(memory, 200)} ({memory['memory_id']})")
            parts.append("\n".join(lines))
            _mark_delivered(conn, session, [m["memory_id"] for m in standing], "SessionStart", agent)

        review = conn.execute(
            "SELECT COUNT(DISTINCT memory_id) FROM anchors WHERE status IN ('NEEDS_REVIEW','DRIFTED')"
        ).fetchone()[0]
        notes = []
        if review:
            notes.append(f"{review} memories need review against changed code "
                         "(workspace(action='open') lists them)")
        lab = root / ".icn-lab" / "lab.db"
        if lab.exists():
            try:
                labconn = sqlite3.connect(f"file:{lab.as_posix()}?mode=ro", uri=True, timeout=1)
                waiting = labconn.execute(
                    "SELECT COUNT(*) FROM runs WHERE status IN ('done','failed','timed_out','lost') AND verdict IS NULL"
                ).fetchone()[0]
                labconn.close()
                if waiting:
                    notes.append(f"{waiting} experiment runs finished without a verdict "
                                 "(experiment(action='runs'))")
            except sqlite3.Error:
                pass
        if notes:
            parts.append("Needs attention: " + "; ".join(notes) + ".")

        text = "\n\n".join(parts)
        return text if len(text) <= SESSION_BUDGET else text[:SESSION_BUDGET - 3] + "..."
    finally:
        conn.close()


HANDLERS = {"session-start": ("SessionStart", session_start),
            "pre-tool-use": ("PreToolUse", pre_tool_use)}


def _log_failure(event: str, agent: str, err: BaseException) -> None:
    try:
        from . import paths
        log = paths.logs_dir() / "hooks.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"at": now(), "event": event, "agent": agent,
                                     "error": f"{type(err).__name__}: {err}"}) + "\n")
    except Exception:  # noqa: BLE001 - failing to log must not fail the hook either
        pass


def run(argv: list[str], stdin: Any = None, stdout: Any = None) -> int:
    """`icn hook <event> [--agent claude|codex]`. Always exits 0."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    event = argv[0] if argv else ""
    agent = argv[argv.index("--agent") + 1] if "--agent" in argv[:-1] else "claude"
    started = time.perf_counter()
    try:
        if event not in HANDLERS:
            return 0
        raw = stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
        if not isinstance(payload, dict):
            return 0
        name, handler = HANDLERS[event]
        context = handler(payload, agent)
        if context:
            stdout.write(json.dumps({"hookSpecificOutput": {"hookEventName": name,
                                                            "additionalContext": context}}) + "\n")
            stdout.flush()
    except Exception as err:  # noqa: BLE001 - fail open, see module docstring
        _log_failure(event, agent, err)
    finally:
        if os.environ.get("ICN_HOOK_TIMING"):
            sys.stderr.write(f"icn hook {event}: {(time.perf_counter() - started) * 1000:.0f} ms\n")
    return 0
