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
    # The server migrates a store whenever it opens it. Re-running the schema
    # script and backfills here cost ~40 ms of a measured 60-85 ms hook, before
    # every tool call; only a store from an older version needs it.
    conn = db.connect(path)
    if conn.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION:
        return conn
    conn.close()
    return db.init_repo_store(path)


SHELL_TOOLS = ("bash", "shell", "shell_command", "powershell", "exec_command", "local_shell")
# Commands that search or list rather than read. A file named only as their
# argument is not being worked on yet, and delivering its knowledge then spent
# the once-per-session delivery before the agent read or edited the file.
# Measured in a live session: a grep for "def _migrate" in db.py delivered four
# clone-fingerprinting rules, which were then withheld when db.py was edited.
SEARCH_COMMANDS = {"grep", "egrep", "fgrep", "rg", "ag", "ack", "findstr", "select-string", "sls",
                   "find", "fd", "ls", "dir", "gci", "get-childitem", "tree", "wc", "du", "stat",
                   "file", "test-path", "measure-object"}
GIT_SEARCH = {"grep", "log", "status", "ls-files", "shortlog", "rev-list"}
FILE_TOKEN = re.compile(r"[\w./\\:-]+\.[A-Za-z0-9]{1,8}")


def _command_word(segment: str) -> tuple[str, str]:
    words = [w.strip("\"'()") for w in segment.split()]
    words = [w for w in words if w and "=" not in w.split("/")[0]]   # FOO=bar cmd
    if not words:
        return "", ""
    head = words[0].replace("\\", "/").rsplit("/", 1)[-1].lower()
    head = head[:-4] if head.endswith(".exe") else head
    if head in ("powershell", "pwsh", "cmd", "bash", "sh") and len(words) > 1:
        rest = [w for w in words[1:] if not w.startswith(("-", "/"))]
        return _command_word(" ".join(rest)) if rest else (head, "")
    return head, (words[1].lower() if len(words) > 1 else "")


def _segments(command: str) -> list[str]:
    """Split on `|`, `||`, `&&`, `;` and newlines outside quotes.

    A regex split cut `grep -n "def \\|import_from" src/icn/imports.py` at the
    `|` inside the pattern, so the pattern's tail became a "command" and the
    file after it counted as read (measured: it delivered parsing.py rules).
    """
    out, current, quote, i = [], [], "", 0
    while i < len(command):
        char = command[i]
        if quote:
            if char == "\\" and quote == '"' and i + 1 < len(command):
                current.append(command[i:i + 2])
                i += 2
                continue
            if char == quote:
                quote = ""
            current.append(char)
        elif char in "'\"":
            quote = char
            current.append(char)
        elif char in "|;\n&":
            if char == "&" and command[i + 1:i + 2] != "&":
                current.append(char)          # a lone & (background) is not a separator we split on
            else:
                out.append("".join(current))
                current = []
                if command[i + 1:i + 2] in ("|", "&") and char in "|&":
                    i += 1
        else:
            current.append(char)
        i += 1
    out.append("".join(current))
    return [s for s in out if s.strip()]


def _shell_files(command: str) -> list[str]:
    """File tokens in a shell command, skipping those only searched or listed."""
    out: list[str] = []
    for segment in _segments(command):
        head, sub = _command_word(segment)
        if head in SEARCH_COMMANDS or (head == "git" and sub in GIT_SEARCH):
            continue
        out.extend(FILE_TOKEN.findall(segment))
    return out[:30]


def touched_files(payload: dict[str, Any], root: Path) -> list[str]:
    """Repository-relative files a tool call is about to read or change."""
    return list(touched_ranges(payload, root))


def _line_range(text: str, needle: str) -> tuple[int, int] | None:
    at = text.find(needle) if needle else -1
    if at < 0:
        return None
    start = text.count("\n", 0, at) + 1
    # A trailing newline ends the last line; it does not reach the next one.
    return start, start + needle.rstrip("\n").count("\n")


def touched_ranges(payload: dict[str, Any], root: Path) -> dict[str, tuple[int, int] | None]:
    """{repository-relative file: (first line, last line) or None for all of it}."""
    tool = str(payload.get("tool_name") or "").lower()
    tool_input = payload.get("tool_input") or {}
    if not isinstance(tool_input, dict):
        return {}
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
        elif tool in SHELL_TOOLS:
            # A shell read (`cat src/x.py`, `Get-Content x`) is contact with a
            # file too. Only tokens naming a real file count.
            raw.extend(_shell_files(command))

    cwd = Path(payload.get("cwd") or root)
    root_norm = _norm(root.resolve())
    out: dict[str, tuple[int, int] | None] = {}
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
        if relative.startswith((".git/", ".agit/", ".icn-lab/", ".icn-trace/")) or relative in out:
            continue
        out[relative] = _span(tool, tool_input, resolved) if tool in FILE_TOOLS else None
        if len(out) >= 6:
            break
    return out


def _span(tool: str, tool_input: dict[str, Any], path: Path) -> tuple[int, int] | None:
    """The lines a Read or Edit call touches, or None when it is the whole file.

    Any doubt means the whole file: an unlocatable edit or an unreadable file
    must not hide knowledge, only a precisely located one may narrow it.
    """
    try:
        if tool == "read" and (tool_input.get("offset") or tool_input.get("limit")):
            start = max(1, int(tool_input.get("offset") or 1))
            return start, start + max(1, int(tool_input.get("limit") or 2000)) - 1
        edits = []
        if tool in ("edit", "edit_file") and isinstance(tool_input.get("old_string"), str):
            edits = [tool_input]
        elif tool == "multiedit" and isinstance(tool_input.get("edits"), list):
            edits = [e for e in tool_input["edits"] if isinstance(e, dict)]
        if not edits or any(e.get("replace_all") for e in edits):
            return None
        text = path.read_text(encoding="utf-8", errors="replace")
        spans = [_line_range(text, str(e.get("old_string") or "")) for e in edits]
        if any(s is None for s in spans):
            return None
        return min(s[0] for s in spans), max(s[1] for s in spans)
    except (OSError, ValueError, TypeError):
        return None


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


_NOT_CODE = re.compile(r"^\s*(?:(?:#|//|--|;|\*|/\*).*)?$")


def _inside_symbols(conn: sqlite3.Connection, root: Path | None, file_path: str,
                    span: tuple[int, int]) -> bool:
    """Whether every line of code in the span lies inside some indexed symbol.

    Module-level code (constants, imports, top-level statements) belongs to no
    symbol, so knowledge about it is anchored to the file. A span touching it
    is treated as touching the whole file, or INTENT_WEIGHTS in search.py would
    never surface the rule that governs it. Blank and comment lines between
    symbols are not code, or every span crossing two functions would count.
    """
    ranges = conn.execute(
        "SELECT line_start, COALESCE(line_end, line_start) FROM symbols"
        " WHERE last_known_path = ? AND status = 'ACTIVE'"
        " AND line_start IS NOT NULL AND line_start <= ? AND COALESCE(line_end, line_start) >= ?",
        (file_path, span[1], span[0])).fetchall()
    covered: set[int] = set()
    for start, end in ranges:
        covered.update(range(max(start, span[0]), min(end, span[1]) + 1))
    gaps = [n for n in range(span[0], span[1] + 1) if n not in covered]
    if not gaps:
        return True
    if root is None:
        return False
    try:
        with (root / file_path).open(encoding="utf-8", errors="replace") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return False
    return all(n > len(lines) or _NOT_CODE.match(lines[n - 1]) for n in gaps)


def _overlaps(span: tuple[int, int] | None, start: Any, end: Any) -> bool:
    if span is None or start is None:
        return True
    return int(start) <= span[1] and int(end if end is not None else start) >= span[0]


def memories_for_files(conn: sqlite3.Connection, files: list[str] | dict[str, tuple[int, int] | None],
                       root: Path | None = None) -> list[dict[str, Any]]:
    """Memories anchored to these files, unless their code is gone.

    Given {file: (first, last line)}, a memory anchored only to symbols outside
    those lines is left for when that code is touched; a file-level anchor, or
    a symbol with no known lines, always applies.
    """
    spans = files if isinstance(files, dict) else {f: None for f in files}
    if not spans:
        return []
    conn.row_factory = sqlite3.Row
    spans = {f: (s if s is None or _inside_symbols(conn, root, f, s) else None)
             for f, s in spans.items()}
    marks = ",".join("?" for _ in spans)
    kinds = ",".join("?" for _ in DELIVERED_KINDS)
    found = conn.execute(
        f"SELECT m.memory_id, m.kind, m.severity, m.claim, m.body, m.confidence,"
        f" m.helpful_count, m.unhelpful_count, m.evidence_count, a.file_path, a.symbol_path,"
        f" COALESCE(s.line_start, a.line_start) AS line_start,"
        f" COALESCE(s.line_end, a.line_end) AS line_end,"
        f" (SELECT COUNT(*) FROM anchors c WHERE c.memory_id = m.memory_id"
        f"   AND c.status IN ('NEEDS_REVIEW','DRIFTED')) AS unverified"
        f" FROM anchors a JOIN memories m ON m.memory_id = a.memory_id"
        f" LEFT JOIN symbols s ON s.symbol_id = a.symbol_id AND s.status = 'ACTIVE'"
        f" WHERE a.file_path IN ({marks}) AND m.status = 'ACTIVE' AND m.kind IN ({kinds})"
        f" AND a.status IN ('ACTIVE','NEEDS_REVIEW','DRIFTED')",
        (*spans, *DELIVERED_KINDS)).fetchall()
    # A memory from a record that touched many files only applies to the
    # files and symbols its own claim names (see relevance.py).
    from .relevance import applies, narrowed
    # Hooks volunteer knowledge unasked, so a broad memory whose claim names
    # nothing is not volunteered anywhere; investigate() still finds it.
    focused = narrowed(conn, list({row["memory_id"] for row in found}), keep_unnamed=True)
    # (memory, file) -> whether any symbol anchor there overlaps the span, and
    # whether there is any symbol anchor there at all. A memory with symbol
    # anchors in a file is about those symbols: its file-level anchor is just
    # their container and must not make it apply to every line.
    hit: dict[tuple[str, str], bool] = {}
    by_id: dict[str, dict[str, Any]] = {}
    for row in found:
        memory = dict(row)
        focus = focused.get(memory["memory_id"])
        if focus is not None and not applies(focus, memory["file_path"], memory["symbol_path"]):
            continue
        key = (memory["memory_id"], memory["file_path"])
        if memory["symbol_path"]:
            hit[key] = hit.get(key, False) or _overlaps(
                spans.get(memory["file_path"]), memory["line_start"], memory["line_end"])
            if not hit[key]:
                continue
        entry = by_id.setdefault(memory["memory_id"], {**memory, "symbols": set(), "files": set()})
        entry["files"].add(memory["file_path"])
        if memory["symbol_path"]:
            entry["symbols"].add(memory["symbol_path"])
    for memory_id in list(by_id):
        if not any(hit.get((memory_id, f), True) for f in by_id[memory_id]["files"]):
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
    spans = touched_ranges(payload, root)
    if not spans:
        return ""
    files = list(spans)
    conn = _store(repo_id)
    if conn is None:
        return ""
    try:
        session = str(payload.get("session_id") or "unknown")
        candidates = memories_for_files(conn, spans, root)
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
