"""Cross-agent conversation memory: what Codex, Claude Code, Copilot, Cursor
and the others said and did, searchable from any of them.

Every coding agent keeps its own transcripts on disk, in its own private
format, and none of them can read the others'. This module is the read-only
layer over all of them:

    vendor files -> adapter -> normalised session -> SQLite FTS5 -> conversations()

The vendor files are source material, never the API. The normalised shape is
the contract, so when a vendor rearranges its storage (they all do, and none
publish a schema) the tool surface stays identical and the adapter reports the
drift instead of quietly returning nothing.

Sources, by surface:
    codex_cli        $CODEX_HOME/sessions/YYYY/MM/DD/rollout-*.jsonl
                     (plus archived_sessions; titles from session_index.jsonl)
    claude_code      $CLAUDE_CONFIG_DIR/projects/<project>/<session>.jsonl
                     and <session>/subagents/agent-*.jsonl
    copilot_cli      $COPILOT_HOME/session-state/<id>/events.jsonl
    vscode_chat      <VS Code User dir>/workspaceStorage/<ws>/chatSessions/*.jsonl|json
                     for VS Code, Insiders, VSCodium, Cursor, Windsurf, Kiro
    cursor_composer  <Cursor User dir>/globalStorage/state.vscdb (cursorDiskKV)
    cursor_transcript ~/.cursor/projects/<slug>/agent-transcripts/**/*.jsonl
    gemini_cli       ~/.gemini/tmp/<hash>/chats/*.json

Every adapter is defensive in the same way: it counts the record types it saw,
extracts what it recognises, and emits a SCHEMA_DRIFT diagnostic naming the
adapter to fix when a file parses as JSON but yields no transcript. A file that
drifts keeps its last good normalised copy; the index is never emptied by a
vendor update.

Retention defaults to mirror: when the vendor deletes a transcript (Claude Code
does so after 30 days by default), the normalised copy stays searchable.
ICN_TRANSCRIPTS_RETENTION=source makes the index follow the vendor instead.

Privacy: these transcripts carry whatever passed through the agents' tools -
file contents, shell output, pasted secrets. The index lives under the ICN
data root, is never sent anywhere, and the vendor stores are only ever read.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import sys
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from . import paths, transcript_archive
from .db import jdump, jload, one, rows, write_tx

# ----------------------------------------------------------------- constants

ENV_RETENTION = "ICN_TRANSCRIPTS_RETENTION"          # mirror (default) | source
ENV_INCLUDE_REASONING = "ICN_TRANSCRIPTS_INCLUDE_REASONING"
ENV_MAX_TOOL_TEXT = "ICN_TRANSCRIPTS_MAX_TOOL_TEXT"
ENV_EXTRA_ROOTS = "ICN_TRANSCRIPTS_EXTRA_ROOTS"      # os.pathsep-separated dirs

SURFACES = (
    "codex_cli", "claude_code", "copilot_cli", "vscode_chat",
    "cursor_composer", "cursor_transcript", "gemini_cli",
)

# Bump a surface's version whenever its adapter changes shape. Every file for
# that surface is then reparsed on the next refresh, automatically.
PARSER_VERSION = {
    "codex_cli": "codex-cli-v1",
    "claude_code": "claude-code-v1",
    "copilot_cli": "copilot-cli-v1",
    "vscode_chat": "vscode-chat-v1",
    "cursor_composer": "cursor-composer-v1",
    "cursor_transcript": "cursor-transcript-v1",
    "gemini_cli": "gemini-cli-v1",
}

ROLES = ("user", "assistant", "tool", "system", "summary")
KINDS = ("message", "tool_call", "tool_result", "summary", "system", "reasoning")

MAX_DIAGNOSTICS_PER_FILE = 20
MAX_MESSAGE_TEXT = 250_000
DEFAULT_MAX_TOOL_TEXT = 32_000
MAX_HIT_TEXT = 8_000
SCHEMA_SAMPLE_RECORDS = 500

SCHEMA = """
CREATE TABLE IF NOT EXISTS source_files (
    source_path      TEXT PRIMARY KEY,
    provider         TEXT NOT NULL,
    surface          TEXT NOT NULL,
    mtime_ns         INTEGER,
    size_bytes       INTEGER,
    parser_version   TEXT NOT NULL,
    status           TEXT NOT NULL,
    diagnostics_json TEXT NOT NULL DEFAULT '[]',
    indexed_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    session_key    TEXT PRIMARY KEY,
    provider       TEXT NOT NULL,
    surface        TEXT NOT NULL,
    native_id      TEXT NOT NULL,
    source_path    TEXT NOT NULL,
    cwd            TEXT,
    cwd_norm       TEXT,
    repo_root      TEXT,
    repo_root_norm TEXT,
    title          TEXT,
    started_at     TEXT,
    updated_at     TEXT,
    message_count  INTEGER NOT NULL,
    meta_json      TEXT,
    origin         TEXT NOT NULL DEFAULT 'live',
    live_path      TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_provider ON sessions(provider, surface);
CREATE INDEX IF NOT EXISTS idx_sessions_native   ON sessions(native_id);
CREATE INDEX IF NOT EXISTS idx_sessions_cwd      ON sessions(cwd_norm);
CREATE INDEX IF NOT EXISTS idx_sessions_source   ON sessions(source_path);
CREATE INDEX IF NOT EXISTS idx_sessions_updated  ON sessions(updated_at);

CREATE TABLE IF NOT EXISTS messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_key  TEXT NOT NULL,
    ordinal      INTEGER NOT NULL,
    timestamp    TEXT,
    role         TEXT NOT NULL,
    kind         TEXT NOT NULL,
    text         TEXT NOT NULL,
    native_id    TEXT,
    tool_name    TEXT,
    tool_call_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_key, ordinal);

-- External-content FTS: the text is stored once, in messages. Deleting rows
-- therefore needs the old text handed back to the index (see _delete_source).
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    text,
    content='messages',
    content_rowid='id',
    tokenize = "unicode61 remove_diacritics 2"
);
"""


# --------------------------------------------------------------- data model


@dataclass
class Source:
    provider: str
    surface: str
    file: str                       # the bytes actually read: live file, archive copy,
                                    # or "<db>#<key>" for a row in a store
    parser_version: str
    extra: dict[str, Any] = field(default_factory=dict)
    live_path: str | None = None    # the vendor's path, when file is an archive copy
    origin: str = "live"            # live | archive | archive_only | archive_version

    @property
    def identity_path(self) -> str:
        """What the session key is built from. A session keeps its identity when
        the index switches from reading the live file to the archive copy of it;
        an old generation in versions/ is its own session."""
        if self.origin == "archive_version":
            return self.file
        return self.live_path or self.file


@dataclass
class Diagnostic:
    code: str                       # SCHEMA_DRIFT | UNKNOWN_RECORD_TYPES | MALFORMED_JSONL
                                    # | SOURCE_MISSING | IO_ERROR | PARTIAL_PARSE
    severity: str                   # info | warning | error
    provider: str
    surface: str
    source_path: str
    parser_version: str
    message: str
    observed: Any = None
    expected: Any = None
    fix: dict[str, str] | None = None   # {"where": adapter, "action": what to do}

    def as_dict(self) -> dict[str, Any]:
        out = {k: v for k, v in self.__dict__.items() if v is not None}
        return out


@dataclass
class Message:
    ordinal: int
    role: str
    kind: str
    text: str
    timestamp: str | None = None
    native_id: str | None = None
    tool_name: str | None = None
    tool_call_id: str | None = None


@dataclass
class Session:
    provider: str
    surface: str
    native_id: str
    source_path: str
    messages: list[Message]
    diagnostics: list[Diagnostic]
    cwd: str | None = None
    repo_root: str | None = None
    title: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    origin: str = "live"
    live_path: str | None = None
    identity_path: str | None = None

    @property
    def started_at(self) -> str | None:
        stamps = sorted(m.timestamp for m in self.messages if m.timestamp)
        return stamps[0] if stamps else None

    @property
    def updated_at(self) -> str | None:
        stamps = sorted(m.timestamp for m in self.messages if m.timestamp)
        return stamps[-1] if stamps else None


class Transcript:
    """Accumulates normalised messages for one session with the shared rules:
    trim, cap, drop empties, collapse immediate duplicates."""

    def __init__(self, source: Source, include_reasoning: bool):
        self.source = source
        self.include_reasoning = include_reasoning
        self.messages: list[Message] = []
        self.diagnostics: list[Diagnostic] = []
        self.record_types: dict[str, int] = {}
        self.unknown_types: set[str] = set()
        self.max_tool_text = _max_tool_text()
        # Records the adapter believed were conversation. Compared against
        # messages in finish(): found-but-empty is drift, never-found is an
        # empty session. See finish().
        self.content_seen = 0

    def saw(self, record_type: str, known: set[str] | None = None) -> None:
        self.record_types[record_type] = self.record_types.get(record_type, 0) + 1
        if known is not None and record_type not in known:
            self.unknown_types.add(record_type)

    def push(self, role: str, kind: str, text: Any, *, timestamp: Any = None,
             native_id: Any = None, tool_name: Any = None, tool_call_id: Any = None) -> None:
        self.content_seen += 1
        if kind == "reasoning" and not self.include_reasoning:
            return
        body = text if isinstance(text, str) else extract_text(text)
        body = (body or "").strip()
        if not body:
            return
        cap = self.max_tool_text if kind == "tool_result" else MAX_MESSAGE_TEXT
        if len(body) > cap:
            body = body[:cap] + "\n[icn: truncated indexed text]"
        stamp = to_iso(timestamp)
        previous = self.messages[-1] if self.messages else None
        # Some vendors write the same user-facing content under two record
        # types. Identical, adjacent, same-role text is one message.
        if previous and previous.role == role and previous.kind == kind \
                and previous.text == body and previous.timestamp == stamp:
            return
        self.messages.append(Message(
            ordinal=len(self.messages), role=role, kind=kind, text=body, timestamp=stamp,
            native_id=_opt_str(native_id), tool_name=_opt_str(tool_name),
            tool_call_id=_opt_str(tool_call_id),
        ))

    def diagnose(self, code: str, severity: str, message: str, **extra: Any) -> None:
        if len(self.diagnostics) >= MAX_DIAGNOSTICS_PER_FILE:
            return
        self.diagnostics.append(Diagnostic(
            code=code, severity=severity, provider=self.source.provider,
            surface=self.source.surface, source_path=self.source.file,
            parser_version=self.source.parser_version, message=message, **extra,
        ))

    def finish(self, adapter: str, expected: Any, valid_records: int,
               content_records: int | None = None) -> None:
        """The two diagnostics every adapter emits the same way.

        content_records counts only the records the adapter recognised as
        carrying conversation. Without it, a file holding nothing but metadata
        (Claude Code writes sessions that are a single ai-title record, and a
        chat opened but never used is empty by definition) reads as drift, and
        a false SCHEMA_DRIFT is worse than none: it is the signal that the
        vendor changed format, so it has to mean that and nothing else.
        """
        if self.unknown_types:
            self.diagnose(
                "UNKNOWN_RECORD_TYPES", "info",
                f"{adapter} saw record types it does not interpret. Harmless unless "
                "they carry transcript content.",
                observed={"record_types": sorted(self.unknown_types)},
                fix={"where": adapter, "action": "check whether any of these types carry "
                                                 "user, assistant or tool content and add a handler"},
            )
        saw_content = self.content_seen if content_records is None else content_records
        if saw_content > 0 and not self.messages:
            self.diagnose(
                "SCHEMA_DRIFT", "error",
                f"{adapter}: the source parsed as JSON but no transcript content could be "
                "extracted. The vendor format has probably changed.",
                observed={"record_types": self.record_types},
                expected=expected,
                fix={"where": adapter, "action": "run conversations(action='schema', "
                                                 "source_path=...) and update the field mapping"},
            )

    def title_from_first_user(self) -> str | None:
        for m in self.messages:
            if m.role == "user" and m.kind == "message":
                return re.sub(r"\s+", " ", m.text).strip()[:160]
        return None


# ------------------------------------------------------------------ helpers


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _max_tool_text() -> int:
    raw = os.environ.get(ENV_MAX_TOOL_TEXT)
    try:
        return max(1_000, int(raw)) if raw else DEFAULT_MAX_TOOL_TEXT
    except ValueError:
        return DEFAULT_MAX_TOOL_TEXT


def retention_mode() -> str:
    return "source" if os.environ.get(ENV_RETENTION, "").lower() == "source" else "mirror"


def include_reasoning() -> bool:
    return os.environ.get(ENV_INCLUDE_REASONING, "") in ("1", "true", "yes")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def to_iso(value: Any) -> str | None:
    """Vendors write epoch seconds, epoch milliseconds, ISO strings, or nothing."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if value != value:  # NaN
            return None
        seconds = value / 1000.0 if value > 10_000_000_000 else float(value)
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat(
                timespec="milliseconds").replace("+00:00", "Z")
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"\d{10,13}(\.\d+)?", text):
            return to_iso(float(text))
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return text
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat(
            timespec="milliseconds").replace("+00:00", "Z")
    return None


def norm_path(value: str | None) -> str | None:
    """Comparable form of a filesystem path: forward slashes, no trailing
    slash, case-folded on Windows, file:// URIs decoded."""
    if not value:
        return None
    text = str(value).strip()
    if text.startswith("file://"):
        text = urllib.parse.unquote(text[len("file://"):])
        # file:///c%3A/x -> /c:/x -> c:/x
        if re.match(r"^/[A-Za-z]:", text):
            text = text[1:]
    text = text.replace("\\", "/").rstrip("/")
    if sys.platform == "win32" or re.match(r"^[A-Za-z]:/", text):
        text = text.lower()
    return text or None


def file_uri_to_path(value: str | None) -> str | None:
    if not value:
        return None
    text = str(value)
    if not text.startswith("file://"):
        return text
    text = urllib.parse.unquote(text[len("file://"):])
    if re.match(r"^/[A-Za-z]:", text):
        text = text[1:]
    if sys.platform == "win32":
        text = text.replace("/", "\\")
    return text


def safe_json(value: Any, limit: int = 10_000) -> str:
    try:
        out = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        out = str(value)
    return out if len(out) <= limit else out[:limit] + "\n[truncated]"


_TEXT_KEYS = ("text", "value", "content", "message", "output", "result", "summary",
              "detailedContent", "thinking", "reasoning")


def extract_text(value: Any, depth: int = 0) -> str:
    """Best-effort text out of the block/list/dict shapes every vendor uses."""
    if depth > 6 or value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        parts = [extract_text(v, depth + 1) for v in value]
        return "\n".join(p for p in parts if p)
    if isinstance(value, dict):
        # A typed image block is a marker, not text.
        if value.get("type") in ("image", "input_image") and "text" not in value:
            return "[image]"
        parts: list[str] = []
        for key in _TEXT_KEYS:
            if key in value:
                got = extract_text(value[key], depth + 1)
                if got and got not in parts:
                    parts.append(got)
        return "\n".join(parts)
    return ""


def shape_of(value: Any, depth: int = 0) -> Any:
    """Structural skeleton: keys and types, never the values. This is what
    schema inspection returns, so a drift report exposes no conversation."""
    if depth >= 4:
        return "array" if isinstance(value, list) else ("null" if value is None else type(value).__name__)
    if isinstance(value, list):
        return [shape_of(value[0], depth + 1)] if value else []
    if isinstance(value, dict):
        return {k: shape_of(value[k], depth + 1) for k in sorted(value)[:40]}
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    return type(value).__name__


def identity_of(path: str) -> str:
    """The form of a path used for identity, so two spellings of one file are
    one session.

    os.path.realpath resolves Windows 8.3 short names (RANITB~1) and symlinks,
    which matters because the vendor and our archive can record the same file
    differently - TEMP is handed out in short form, while a manifest stores what
    the process resolved. Without this the same conversation indexes twice under
    two keys and search returns it twice.
    """
    try:
        resolved = os.path.realpath(path)
    except (OSError, ValueError):
        resolved = path
    return norm_path(resolved) or resolved


def native_id_from_path(source: Source) -> str:
    """The vendor's own name for a session, from the file that holds it.

    Several vendors put the session id in the filename and nowhere else. Read it
    from identity_path, not from the bytes we happen to be reading: the archive
    copy is always called transcript.jsonl, so using its own name would give the
    archived copy of a session a different id from the live one, and the two
    would index as two separate conversations.
    """
    return os.path.splitext(os.path.basename(source.identity_path))[0]


def session_key(surface: str, native_id: str, source_path: str) -> str:
    # source_path is part of the identity: a restored or copied file with the
    # same native id must not overwrite the original.
    digest = hashlib.sha256(f"{surface}\0{native_id}\0{identity_of(source_path)}".encode("utf-8"))
    return digest.hexdigest()[:24]


def read_jsonl(source: Source, tx: Transcript, on_record: Callable[[dict[str, Any], int], None],
               path: str | None = None) -> int:
    """Stream a JSONL file, one record at a time. Returns the count of valid
    records; malformed lines become diagnostics rather than a failed file."""
    valid = 0
    malformed = 0
    with open(path or source.file, "r", encoding="utf-8", errors="replace") as fh:
        for line_number, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError as exc:
                malformed += 1
                tx.diagnose(
                    "MALFORMED_JSONL", "warning", f"could not parse line {line_number}",
                    observed={"line": line_number, "bytes": len(raw.encode("utf-8", "replace")),
                              "error": str(exc)[:200]},
                    fix={"where": "read_jsonl", "action": "corruption, a concatenated record, "
                                                         "or a new framing; inspect the line"},
                )
                continue
            if not isinstance(record, dict):
                continue
            valid += 1
            on_record(record, line_number)
    if malformed and valid:
        tx.diagnose("PARTIAL_PARSE", "info",
                    f"{malformed} malformed line(s) skipped; {valid} parsed.")
    return valid


# ---------------------------------------------------------- injected context
#
# Both Codex and Claude Code write text the harness injected - environment
# blocks, AGENTS.md, slash-command output, IDE state - as role=user. Indexing
# it as the user's words poisons titles and role filters. These prefixes mark
# such blocks; the content is still indexed, as system.

_INJECTED_PREFIXES = (
    "<environment_context>", "<user_instructions>", "<recommended_plugins>",
    "<permissions instructions", "<skills_instructions>", "<plugins_instructions>",
    "<apps_instructions>", "<collaboration_mode>", "<multi_agent", "<turn_aborted>",
    "<model_switch>", "<image_resize_notice>", "<send_user_message",
    "# AGENTS.md instructions", "# Files mentioned by the",
    "<command-name>", "<command-message>", "<local-command-stdout>", "<local-command-caveat>",
    "<ide_opened_file>", "<ide_selection>", "<task-notification>", "<system-reminder>",
    "<bash-input>", "<bash-stdout>", "<bash-stderr>", "<user-prompt-submit-hook>",
    "<user-memory-input>", "[Request interrupted by user",
)

_COMPACTION_PREFIXES = (
    "This session is being continued from a previous conversation",
)


def classify_user_text(text: str) -> tuple[str, str]:
    """(role, kind) for a vendor 'user' record's text."""
    head = text.lstrip()[:80]
    for prefix in _COMPACTION_PREFIXES:
        if head.startswith(prefix):
            return "summary", "summary"
    for prefix in _INJECTED_PREFIXES:
        if head.startswith(prefix):
            return "system", "system"
    return "user", "message"


# ------------------------------------------------------------------- Codex


_CODEX_KNOWN = {
    "session_meta", "response_item", "event_msg", "compacted", "turn_context",
    "world_state", "token_usage_record", "inter_agent_communication",
    "inter_agent_communication_metadata",
}


def parse_codex(source: Source) -> Session:
    tx = Transcript(source, include_reasoning())
    base = os.path.basename(source.file)
    found = re.search(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", base, re.I)
    native_id = found.group(1) if found else os.path.splitext(base)[0]
    meta: dict[str, Any] = {}
    state = {"cwd": None, "repo_root": None}
    fallback: list[tuple[str, str, Any]] = []

    def on_record(record: dict[str, Any], _line: int) -> None:
        nonlocal native_id
        rtype = str(record.get("type") or "unknown")
        tx.saw(rtype, _CODEX_KNOWN)
        stamp = record.get("timestamp")
        payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}

        if rtype == "session_meta":
            p = payload.get("meta") if isinstance(payload.get("meta"), dict) else payload
            native_id = str(p.get("id") or p.get("session_id") or native_id)
            state["cwd"] = p.get("cwd") or state["cwd"]
            git = p.get("git") if isinstance(p.get("git"), dict) else {}
            state["repo_root"] = git.get("root") or state["repo_root"]
            for key in ("originator", "cli_version", "source", "model_provider"):
                if p.get(key):
                    meta[key] = p[key]
            if git.get("repository_url"):
                meta["repository_url"] = git["repository_url"]
            if git.get("branch"):
                meta["branch"] = git["branch"]
            return

        if rtype == "turn_context":
            if not state["cwd"] and payload.get("cwd"):
                state["cwd"] = payload["cwd"]
            if payload.get("model") and "model" not in meta:
                meta["model"] = payload["model"]
            return

        if rtype == "response_item":
            itype = str(payload.get("type") or "")
            if itype == "message":
                vendor_role = payload.get("role")
                text = extract_text(payload.get("content"))
                if vendor_role == "assistant":
                    role, kind = "assistant", "message"
                elif vendor_role in ("system", "developer"):
                    role, kind = "system", "system"
                else:
                    role, kind = classify_user_text(text)
                tx.push(role, kind, text, timestamp=stamp, native_id=payload.get("id"))
            elif itype in ("function_call", "custom_tool_call", "local_shell_call",
                           "web_search_call", "computer_call"):
                name = payload.get("name") or {"local_shell_call": "shell",
                                               "web_search_call": "web_search"}.get(itype, itype)
                args = payload.get("arguments") or payload.get("input") or \
                    payload.get("action") or payload.get("command") or ""
                tx.push("assistant", "tool_call", f"{name}\n{safe_json(args)}", timestamp=stamp,
                        native_id=payload.get("id"), tool_name=name,
                        tool_call_id=payload.get("call_id") or payload.get("id"))
            elif itype in ("function_call_output", "custom_tool_call_output",
                           "local_shell_call_output", "computer_call_output"):
                out = payload.get("output")
                if out is None:
                    out = payload.get("content") or payload.get("result")
                tx.push("tool", "tool_result", extract_text(out), timestamp=stamp,
                        native_id=payload.get("id"), tool_call_id=payload.get("call_id"))
            elif itype == "reasoning":
                tx.push("assistant", "reasoning", extract_text(payload.get("summary")),
                        timestamp=stamp, native_id=payload.get("id"))
            return

        if rtype == "event_msg":
            etype = str(payload.get("type") or "")
            if etype == "item_completed":
                item = payload.get("item") if isinstance(payload.get("item"), dict) else {}
                kind = item.get("type")
                if kind == "UserMessage":
                    fallback.append(("user", "message", extract_text(item.get("content"))))
                elif kind == "AgentMessage":
                    fallback.append(("assistant", "message", extract_text(item.get("content"))))
            elif etype in ("user_message", "agent_message", "assistant_message"):
                role = "user" if etype == "user_message" else "assistant"
                fallback.append((role, "message", payload.get("message") or extract_text(payload.get("content"))))
            return

        if rtype == "compacted":
            summary = extract_text(payload.get("message") or payload.get("summary"))
            tx.push("summary", "summary", summary, timestamp=stamp)
            return

    valid = read_jsonl(source, tx, on_record)

    # Older and newer rollouts may carry the conversation only as events.
    if not any(m.kind == "message" for m in tx.messages) and fallback:
        for role, kind, text in fallback:
            tx.push(role, kind, text)

    tx.finish("parse_codex", {"transcript_record": "type=response_item",
                              "message_shape": "payload.type=message, payload.role, payload.content[].text"},
              valid)
    titles = source.extra.get("titles") or {}
    return Session(
        provider="codex", surface="codex_cli", native_id=native_id, source_path=source.file,
        messages=tx.messages, diagnostics=tx.diagnostics, cwd=state["cwd"],
        repo_root=state["repo_root"], title=titles.get(native_id) or tx.title_from_first_user(),
        meta=meta,
    )


# ------------------------------------------------------------- Claude Code


_CLAUDE_KNOWN = {
    "user", "assistant", "system", "summary", "ai-title", "custom-title", "progress",
    "attachment", "file-history-snapshot", "file-history-delta", "queue-operation",
    "last-prompt", "atis-latch", "mode", "bridge-session",
}


def parse_claude(source: Source) -> Session:
    tx = Transcript(source, include_reasoning())
    native_id = native_id_from_path(source)
    # Read this from identity_path, not from the file being read: a subagent
    # transcript carries its parent's sessionId in every record, so the only
    # thing that distinguishes it is living in a subagents/ directory. The
    # archive copy is a flat file, so checking its own path would make the
    # archived copy adopt the parent's id and index as a second session.
    identity = source.identity_path
    is_subagent = os.path.basename(os.path.dirname(identity)) == "subagents"
    meta: dict[str, Any] = {}
    state: dict[str, Any] = {"cwd": None, "title": None, "summary_title": None}
    if is_subagent:
        meta["subagent"] = True
        meta["parent_session"] = os.path.basename(os.path.dirname(os.path.dirname(identity)))

    def on_record(record: dict[str, Any], _line: int) -> None:
        nonlocal native_id
        rtype = str(record.get("type") or "unknown")
        tx.saw(rtype, _CLAUDE_KNOWN)
        if record.get("sessionId") and not is_subagent:
            native_id = str(record["sessionId"])
        if record.get("cwd") and not state["cwd"]:
            state["cwd"] = record["cwd"]
        for key, name in (("gitBranch", "branch"), ("version", "cli_version")):
            if record.get(key) and name not in meta:
                meta[name] = record[key]
        stamp = record.get("timestamp")

        if rtype == "ai-title":
            state["title"] = record.get("aiTitle") or state["title"]
            return
        if rtype == "custom-title":
            state["title"] = record.get("customTitle") or state["title"]
            return
        if rtype == "summary":
            state["summary_title"] = record.get("summary") or state["summary_title"]
            return

        if rtype in ("user", "assistant"):
            message = record.get("message") if isinstance(record.get("message"), dict) else {}
            if message.get("model") and "model" not in meta:
                meta["model"] = message["model"]
            base_role = "assistant" if (rtype == "assistant" or message.get("role") == "assistant") else "user"
            content = message.get("content")
            msg_id = message.get("id") or record.get("uuid")
            if isinstance(content, str):
                role, kind = classify_user_text(content) if base_role == "user" else ("assistant", "message")
                tx.push(role, kind, content, timestamp=stamp, native_id=msg_id)
                return
            if not isinstance(content, list):
                tx.push(base_role, "message", extract_text(content), timestamp=stamp, native_id=msg_id)
                return
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text = block.get("text") or ""
                    role, kind = classify_user_text(text) if base_role == "user" else ("assistant", "message")
                    tx.push(role, kind, text, timestamp=stamp, native_id=msg_id)
                elif btype == "thinking":
                    tx.push("assistant", "reasoning", block.get("thinking") or block.get("text"),
                            timestamp=stamp, native_id=msg_id)
                elif btype == "tool_use":
                    name = block.get("name") or "tool"
                    tx.push("assistant", "tool_call", f"{name}\n{safe_json(block.get('input') or {})}",
                            timestamp=stamp, native_id=msg_id, tool_name=name, tool_call_id=block.get("id"))
                elif btype == "tool_result":
                    tx.push("tool", "tool_result", extract_text(block.get("content")), timestamp=stamp,
                            native_id=msg_id, tool_call_id=block.get("tool_use_id"))
                elif btype in ("image", "document"):
                    tx.push(base_role, "message", f"[{btype}]", timestamp=stamp, native_id=msg_id)
                else:
                    tx.push(base_role, "message", extract_text(block), timestamp=stamp, native_id=msg_id)
            return

        if rtype == "system":
            subtype = record.get("subtype")
            text = extract_text(record.get("content") or record.get("message"))
            if subtype == "compact_boundary" and not text:
                text = "[context compacted]"
            if subtype:
                text = f"[{subtype}] {text}"
            tx.push("system", "system", text, timestamp=stamp, native_id=record.get("uuid"))
            return

    valid = read_jsonl(source, tx, on_record)
    tx.finish("parse_claude", {"message_shape": "type=user|assistant, message.content as str "
                                                "or [{type:text|tool_use|tool_result|thinking}]"},
              valid)
    title = state["title"] or state["summary_title"] or tx.title_from_first_user()
    if is_subagent and title:
        title = f"[subagent] {title}"
    return Session(
        provider="claude", surface="claude_code", native_id=native_id, source_path=source.file,
        messages=tx.messages, diagnostics=tx.diagnostics, cwd=state["cwd"], title=title, meta=meta,
    )


# ------------------------------------------------------------- Copilot CLI


_COPILOT_KNOWN = {
    "session.start", "session.resume", "session.shutdown", "session.info", "session.error",
    "session.idle", "session.task_complete", "session.model_change", "session.mode_changed",
    "session.plan_changed", "session.compaction_start", "session.compaction_complete",
    "session.permissions_changed", "session.remote_steerable_changed", "session.title_changed",
    "user.message", "assistant.message", "assistant.turn_start", "assistant.turn_end",
    "tool.execution_start", "tool.execution_complete", "tool.execution_progress",
    "tool.user_requested", "permission.requested", "permission.completed",
    "hook.start", "hook.end", "subagent.started", "subagent.completed", "skill.invoked",
    "system.message",
}


def parse_copilot_cli(source: Source) -> Session:
    tx = Transcript(source, include_reasoning())
    native_id = os.path.basename(os.path.dirname(source.file))
    state: dict[str, Any] = {"cwd": None, "repo_root": None, "title": None}
    meta: dict[str, Any] = {}

    def on_record(record: dict[str, Any], _line: int) -> None:
        nonlocal native_id
        rtype = str(record.get("type") or "unknown")
        tx.saw(rtype, _COPILOT_KNOWN)
        data = record.get("data") if isinstance(record.get("data"), dict) else {}
        stamp = record.get("timestamp")
        rid = record.get("id")

        if rtype in ("session.start", "session.resume"):
            native_id = str(data.get("sessionId") or native_id)
            context = data.get("context") if isinstance(data.get("context"), dict) else {}
            state["cwd"] = context.get("cwd") or data.get("cwd") or state["cwd"]
            state["repo_root"] = context.get("gitRoot") or data.get("gitRoot") or state["repo_root"]
            if context.get("branch"):
                meta["branch"] = context["branch"]
            for key in ("copilotVersion", "producer"):
                if data.get(key):
                    meta[key] = data[key]
            return
        if rtype == "session.title_changed":
            state["title"] = data.get("title") or state["title"]
            return
        if rtype == "session.model_change":
            meta["model"] = data.get("newModel") or data.get("model") or meta.get("model")
            return
        if rtype == "user.message":
            tx.push("user", "message", extract_text(data.get("content") or data.get("message")),
                    timestamp=stamp, native_id=rid)
            return
        if rtype == "assistant.message":
            tx.push("assistant", "reasoning", data.get("reasoningText"), timestamp=stamp, native_id=rid)
            tx.push("assistant", "message", extract_text(data.get("content") or data.get("message")),
                    timestamp=stamp, native_id=data.get("messageId") or rid)
            for req in data.get("toolRequests") or []:
                if isinstance(req, dict):
                    name = req.get("name") or req.get("toolName") or "tool"
                    tx.push("assistant", "tool_call", f"{name}\n{safe_json(req.get('arguments') or {})}",
                            timestamp=stamp, native_id=rid, tool_name=name,
                            tool_call_id=req.get("toolCallId") or req.get("id"))
            return
        if rtype in ("tool.execution_start", "tool.user_requested"):
            name = data.get("toolName") or data.get("name") or "tool"
            tx.push("assistant", "tool_call", f"{name}\n{safe_json(data.get('arguments') or data.get('input') or {})}",
                    timestamp=stamp, native_id=rid, tool_name=name,
                    tool_call_id=data.get("toolCallId") or data.get("id"))
            return
        if rtype == "tool.execution_complete":
            result = data.get("result")
            body: Any
            if isinstance(result, dict):
                body = result.get("detailedContent") or result.get("content") or result.get("contents") or result
            else:
                body = result
            if not body and data.get("error"):
                err = data["error"]
                body = err.get("message") if isinstance(err, dict) else err
            tx.push("tool", "tool_result", extract_text(body), timestamp=stamp, native_id=rid,
                    tool_call_id=data.get("toolCallId"))
            return
        if rtype in ("session.task_complete", "session.compaction_complete"):
            tx.push("summary", "summary", extract_text(data.get("summary") or data.get("compactedSummary")),
                    timestamp=stamp, native_id=rid)
            return
        if rtype in ("system.message", "session.info", "session.error"):
            text = extract_text(data.get("content") or data.get("message"))
            tx.push("system", "system", f"[{rtype}] {text}" if text else "", timestamp=stamp, native_id=rid)
            return

    valid = read_jsonl(source, tx, on_record)
    tx.finish("parse_copilot_cli", {"user": "type=user.message, data.content",
                                    "assistant": "type=assistant.message, data.content",
                                    "tool": "type=tool.execution_start|complete, data.toolName/result"},
              valid)
    return Session(
        provider="copilot", surface="copilot_cli", native_id=native_id, source_path=source.file,
        messages=tx.messages, diagnostics=tx.diagnostics, cwd=state["cwd"],
        repo_root=state["repo_root"], title=state["title"] or tx.title_from_first_user(), meta=meta,
    )


# ------------------------------------------------------------ VS Code chat
#
# VS Code's chatSessions store is a snapshot plus a patch log:
#     {kind: 0, v: <full session>}          first line
#     {kind: 1, k: [path...], v: value}     set at path
#     {kind: 2, k: [path...], v: [items]}   append to the array at path
# Older builds wrote the snapshot at the record's top level, or a whole .json.


def _set_deep(target: Any, keys: list[Any], value: Any) -> None:
    current = target
    for index, key in enumerate(keys[:-1]):
        nxt = keys[index + 1]
        if isinstance(current, list):
            while len(current) <= key:
                current.append(None)
            if current[key] is None:
                current[key] = [] if isinstance(nxt, int) else {}
            current = current[key]
        elif isinstance(current, dict):
            if current.get(key) is None:
                current[key] = [] if isinstance(nxt, int) else {}
            current = current[key]
        else:
            return
    last = keys[-1]
    if isinstance(current, list) and isinstance(last, int):
        while len(current) <= last:
            current.append(None)
        current[last] = value
    elif isinstance(current, dict):
        current[last] = value


def _get_deep(target: Any, keys: list[Any]) -> Any:
    current = target
    for key in keys:
        if isinstance(current, list) and isinstance(key, int) and key < len(current):
            current = current[key]
        elif isinstance(current, dict):
            current = current.get(key)
        else:
            return None
    return current


def _vscode_workspace_folder(session_file: str) -> str | None:
    """workspaceStorage/<ws>/workspace.json names the folder or the
    .code-workspace file; an untitled workspace points at its own json."""
    ws_dir = os.path.dirname(os.path.dirname(session_file))
    meta_file = os.path.join(ws_dir, "workspace.json")
    try:
        with open(meta_file, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
    except (OSError, ValueError):
        return None
    if isinstance(meta.get("folder"), str):
        return file_uri_to_path(meta["folder"])
    ws = meta.get("workspace")
    if not isinstance(ws, str):
        return None
    ws_path = file_uri_to_path(ws) or ""
    if ws_path.endswith(".code-workspace"):
        return os.path.dirname(ws_path)
    # untitled workspace: Workspaces/<id>/workspace.json with folders[]
    try:
        with open(ws_path, "r", encoding="utf-8") as fh:
            spec = json.load(fh)
        folders = spec.get("folders") or []
        first = folders[0] if folders else None
        if isinstance(first, dict):
            return file_uri_to_path(first.get("uri") or first.get("path"))
    except (OSError, ValueError, IndexError, AttributeError):
        pass
    return None


def _vscode_response_text(parts: Any, tx: Transcript, stamp: Any, req_id: Any) -> None:
    """Flatten a request's response[] into assistant/tool messages."""
    if isinstance(parts, str):
        tx.push("assistant", "message", parts, timestamp=stamp, native_id=req_id)
        return
    if not isinstance(parts, list):
        tx.push("assistant", "message", extract_text(parts), timestamp=stamp, native_id=req_id)
        return
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            tx.push("assistant", "message", "".join(buffer), timestamp=stamp, native_id=req_id)
            buffer.clear()

    for part in parts:
        if isinstance(part, str):
            buffer.append(part)
            continue
        if not isinstance(part, dict):
            continue
        kind = part.get("kind")
        if kind in (None, "markdownContent", "markdownVuln"):
            value = part.get("value") or part.get("content")
            text = value if isinstance(value, str) else extract_text(value)
            if text:
                buffer.append(text)
        elif kind == "thinking":
            flush()
            tx.push("assistant", "reasoning", extract_text(part.get("value")), timestamp=stamp, native_id=req_id)
        elif kind in ("toolInvocationSerialized", "toolInvocation", "prepareToolInvocation"):
            flush()
            name = part.get("toolId") or part.get("toolName") or "tool"
            label = extract_text(part.get("invocationMessage") or part.get("pastTenseMessage"))
            detail = part.get("toolSpecificData")
            body = label if not detail else f"{label}\n{safe_json(detail, 4000)}"
            tx.push("assistant", "tool_call", f"{name}\n{body}", timestamp=stamp, native_id=req_id,
                    tool_name=name, tool_call_id=part.get("toolCallId"))
            result = part.get("resultDetails")
            if result:
                tx.push("tool", "tool_result", extract_text(result), timestamp=stamp,
                        native_id=req_id, tool_call_id=part.get("toolCallId"))
        elif kind == "textEditGroup":
            flush()
            uri = part.get("uri")
            target = uri.get("path") if isinstance(uri, dict) else uri
            tx.push("assistant", "tool_call", f"edit\n{target}", timestamp=stamp, native_id=req_id,
                    tool_name="edit")
        elif kind in ("codeblockUri", "inlineReference", "progressMessage", "progressTaskSerialized",
                      "mcpServersStarting", "undoStop", "confirmation", "warning", "command",
                      "reference", "usedContext", "elicitation"):
            continue
        else:
            text = extract_text(part)
            if text:
                buffer.append(text)
    flush()


def parse_vscode_chat(source: Source) -> Session:
    tx = Transcript(source, include_reasoning())
    native_id = native_id_from_path(source)
    cwd = _vscode_workspace_folder(source.file)
    snapshot: Any = None
    loose: list[Any] = []
    patches = 0
    valid = 0

    if source.file.lower().endswith(".json"):
        try:
            with open(source.file, "r", encoding="utf-8", errors="replace") as fh:
                snapshot = json.load(fh)
            valid = 1
        except (OSError, ValueError) as exc:
            tx.diagnose("SCHEMA_DRIFT", "error", "chat session .json could not be parsed",
                        observed={"error": str(exc)[:200]},
                        fix={"where": "parse_vscode_chat", "action": "check whether VS Code changed framing"})
    else:
        def on_record(record: dict[str, Any], _line: int) -> None:
            nonlocal snapshot, patches, native_id
            kind = record.get("kind")
            tx.saw(f"kind:{kind}")
            if kind == 0:
                body = record.get("v") if isinstance(record.get("v"), dict) else record
                if isinstance(body.get("requests"), list):
                    snapshot = json.loads(json.dumps(body))
                elif body.get("request"):
                    loose.append(body["request"])
                return
            if kind in (1, 2) and isinstance(record.get("k"), list) and snapshot is not None:
                patches += 1
                if kind == 1:
                    _set_deep(snapshot, record["k"], record.get("v"))
                else:
                    target = _get_deep(snapshot, record["k"])
                    if isinstance(target, list) and isinstance(record.get("v"), list):
                        target.extend(record["v"])
                    elif target is None:
                        _set_deep(snapshot, record["k"], list(record.get("v") or []))
                return
            if isinstance(record.get("requests"), list):
                snapshot = record

        valid = read_jsonl(source, tx, on_record)

    if isinstance(snapshot, dict):
        native_id = str(snapshot.get("sessionId") or native_id)
    requests = (snapshot or {}).get("requests") if isinstance(snapshot, dict) else None
    if not isinstance(requests, list):
        requests = loose

    meta: dict[str, Any] = {"product": source.extra.get("product")}
    for request in requests:
        if not isinstance(request, dict):
            continue
        stamp = request.get("timestamp") or request.get("requestDate") or request.get("creationDate")
        req_id = request.get("requestId")
        message = request.get("message")
        prompt = message.get("text") if isinstance(message, dict) else message
        if not prompt and isinstance(request.get("request"), dict):
            prompt = extract_text(request["request"].get("message") or request["request"].get("prompt"))
        tx.push("user", "message", extract_text(prompt), timestamp=stamp, native_id=req_id)
        if request.get("modelId") and "model" not in meta:
            meta["model"] = request["modelId"]
        agent = request.get("agent")
        if isinstance(agent, dict) and agent.get("extensionId") and "agent" not in meta:
            ext = agent["extensionId"]
            meta["agent"] = ext.get("value") if isinstance(ext, dict) else ext
        _vscode_response_text(request.get("response") if "response" in request
                              else (request.get("result") or {}).get("response"), tx, stamp, req_id)
        result = request.get("result")
        if isinstance(result, dict):
            err = result.get("errorDetails")
            if isinstance(err, dict) and err.get("message"):
                tx.push("system", "system", f"[error] {err['message']}", timestamp=stamp, native_id=req_id)

    if valid and not tx.messages and (snapshot is not None or patches or loose):
        # An empty session (opened, never used) is not drift. Only report when
        # there were requests we could not decode.
        if requests:
            tx.diagnose("SCHEMA_DRIFT", "error",
                        "chat session has requests but none decoded into messages",
                        observed={"snapshot_shape": shape_of(snapshot) if snapshot else None,
                                  "patches": patches, "loose_requests": len(loose)},
                        expected={"session": "requests[] with message.text and response[] parts"},
                        fix={"where": "parse_vscode_chat", "action": "compare against the current "
                                                                     "chatSessionStore serialisation"})
    title = (snapshot or {}).get("customTitle") if isinstance(snapshot, dict) else None
    return Session(
        provider=source.provider, surface="vscode_chat", native_id=native_id, source_path=source.file,
        messages=tx.messages, diagnostics=tx.diagnostics, cwd=cwd,
        title=title or tx.title_from_first_user(), meta={k: v for k, v in meta.items() if v},
    )


# ------------------------------------------------------------ Cursor store
#
# Cursor keeps its agent chats ("composers") in one SQLite store,
# globalStorage/state.vscdb, table cursorDiskKV:
#     composerData:<composerId>        header: name, createdAt, and the ordered
#                                      message index fullConversationHeadersOnly
#     bubbleId:<composerId>:<bubbleId> one message: type 1=user 2=assistant,
#                                      text, toolFormerData, thinking
# Each composer is one Source whose file is "<state.vscdb>#composerData:<id>",
# and its lastUpdatedAt stands in for mtime.


def _open_ro(db_file: str) -> sqlite3.Connection:
    uri = "file:" + urllib.parse.quote(db_file.replace("\\", "/")) + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    return conn


def _cursor_workspace_map(user_dir: str) -> dict[str, str]:
    """composerId -> folder, from each workspace's own state.vscdb."""
    mapping: dict[str, str] = {}
    ws_root = os.path.join(user_dir, "workspaceStorage")
    if not os.path.isdir(ws_root):
        return mapping
    for entry in os.scandir(ws_root):
        if not entry.is_dir():
            continue
        folder = _vscode_workspace_folder(os.path.join(entry.path, "chatSessions", "x"))
        db_file = os.path.join(entry.path, "state.vscdb")
        if not folder or not os.path.isfile(db_file):
            continue
        try:
            conn = _open_ro(db_file)
            try:
                for key in ("composer.composerData", "composer.composerHeaders"):
                    row = conn.execute("SELECT value FROM ItemTable WHERE key = ?", (key,)).fetchone()
                    if not row:
                        continue
                    data = jload(row[0] if isinstance(row[0], str) else row[0].decode("utf-8", "replace"), {})
                    composers = data.get("allComposers") or data.get("composers") or data.get("headers") or []
                    for composer in composers:
                        cid = composer.get("composerId") if isinstance(composer, dict) else None
                        if cid:
                            mapping[str(cid)] = folder
            finally:
                conn.close()
        except sqlite3.Error:
            continue
    return mapping


def cursor_composer_sources(user_dir: str) -> list[Source]:
    db_file = os.path.join(user_dir, "globalStorage", "state.vscdb")
    if not os.path.isfile(db_file):
        return []
    out: list[Source] = []
    try:
        conn = _open_ro(db_file)
    except sqlite3.Error:
        return out
    try:
        ws_map = _cursor_workspace_map(user_dir)
        for row in conn.execute("SELECT key, value FROM cursorDiskKV WHERE key LIKE 'composerData:%'"):
            raw = row[1]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", "replace")
            header = jload(raw, {})
            cid = row[0].split(":", 1)[1]
            out.append(Source(
                provider="cursor", surface="cursor_composer",
                file=f"{db_file}#{row[0]}", parser_version=PARSER_VERSION["cursor_composer"],
                extra={"db": db_file, "composer_id": cid, "header": header,
                       "updated": header.get("lastUpdatedAt") or header.get("createdAt"),
                       "folder": ws_map.get(cid)},
            ))
    except sqlite3.Error:
        pass
    finally:
        conn.close()
    return out


def parse_cursor_composer(source: Source) -> Session:
    tx = Transcript(source, include_reasoning())
    header = source.extra.get("header") or {}
    cid = source.extra["composer_id"]
    meta: dict[str, Any] = {}

    def push_bubble(bubble: dict[str, Any]) -> None:
        btype = bubble.get("type")
        stamp = bubble.get("createdAt") or bubble.get("timestamp")
        bid = bubble.get("bubbleId")
        if btype == 1:
            tx.push("user", "message", bubble.get("text") or extract_text(bubble.get("richText")),
                    timestamp=stamp, native_id=bid)
            return
        for block in bubble.get("allThinkingBlocks") or ([bubble["thinking"]] if bubble.get("thinking") else []):
            tx.push("assistant", "reasoning", extract_text(block), timestamp=stamp, native_id=bid)
        tool = bubble.get("toolFormerData")
        if isinstance(tool, dict) and (tool.get("name") or tool.get("tool")):
            name = tool.get("name") or tool.get("tool")
            tx.push("assistant", "tool_call", f"{name}\n{safe_json(tool.get('params') or tool.get('rawArgs') or '')}",
                    timestamp=stamp, native_id=bid, tool_name=name, tool_call_id=tool.get("toolCallId") or bid)
            if tool.get("result") is not None:
                tx.push("tool", "tool_result", extract_text(tool.get("result")), timestamp=stamp,
                        native_id=bid, tool_call_id=tool.get("toolCallId") or bid)
        text = bubble.get("text")
        if not text and bubble.get("codeBlocks"):
            text = "\n".join(extract_text(cb.get("content") or cb) for cb in bubble["codeBlocks"]
                             if isinstance(cb, dict))
        tx.push("assistant", "message", text, timestamp=stamp, native_id=bid)

    valid = 0
    try:
        conn = _open_ro(source.extra["db"])
    except sqlite3.Error as exc:
        tx.diagnose("IO_ERROR", "error", f"could not open Cursor store: {exc}")
        conn = None
    if conn is not None:
        try:
            headers = header.get("fullConversationHeadersOnly")
            inline = header.get("conversation")
            if isinstance(headers, list) and headers:
                valid = len(headers)
                for entry in headers:
                    bid = entry.get("bubbleId") if isinstance(entry, dict) else None
                    if not bid:
                        continue
                    row = conn.execute("SELECT value FROM cursorDiskKV WHERE key = ?",
                                       (f"bubbleId:{cid}:{bid}",)).fetchone()
                    if not row:
                        tx.saw("bubble:missing")
                        continue
                    raw = row[0].decode("utf-8", "replace") if isinstance(row[0], bytes) else row[0]
                    bubble = jload(raw, None)
                    if isinstance(bubble, dict):
                        tx.saw(f"bubble:type={bubble.get('type')}")
                        push_bubble(bubble)
            elif isinstance(inline, list) and inline:
                valid = len(inline)
                for bubble in inline:
                    if isinstance(bubble, dict):
                        tx.saw(f"bubble:type={bubble.get('type')}")
                        push_bubble(bubble)
        except sqlite3.Error as exc:
            tx.diagnose("IO_ERROR", "error", f"reading Cursor bubbles failed: {exc}")
        finally:
            conn.close()
    tx.finish("parse_cursor_composer",
              {"header": "composerData:<id>.fullConversationHeadersOnly[].bubbleId",
               "bubble": "bubbleId:<cid>:<bid> with type 1|2, text, toolFormerData"}, valid)
    for key in ("modelConfig", "unifiedMode", "forceMode"):
        if header.get(key):
            meta[key] = header[key] if not isinstance(header[key], dict) else header[key].get("modelName")
    return Session(
        provider="cursor", surface="cursor_composer", native_id=cid, source_path=source.file,
        messages=tx.messages, diagnostics=tx.diagnostics, cwd=source.extra.get("folder"),
        title=header.get("name") or tx.title_from_first_user(),
        meta={k: v for k, v in meta.items() if v},
    )


# -------------------------------------------------- generic JSONL transcripts
#
# Cursor's agent-transcripts and anything we have not seen: try the shapes
# every vendor converges on (role+content, type+message) and report drift
# when none of them hold. Better a generic decode with a diagnostic than
# nothing.


def parse_generic_jsonl(source: Source, provider: str, surface: str) -> Session:
    tx = Transcript(source, include_reasoning())
    native_id = native_id_from_path(source)
    state: dict[str, Any] = {"cwd": None}

    def on_record(record: dict[str, Any], _line: int) -> None:
        rtype = str(record.get("type") or record.get("role") or record.get("kind") or "unknown")
        tx.saw(rtype)
        stamp = record.get("timestamp") or record.get("createdAt") or record.get("ts")
        if record.get("cwd") and not state["cwd"]:
            state["cwd"] = record["cwd"]
        message = record.get("message") if isinstance(record.get("message"), dict) else record
        role = str(message.get("role") or record.get("role") or record.get("type") or "").lower()
        content = message.get("content") if "content" in message else record.get("content", record.get("text"))
        if role in ("user", "human"):
            text = extract_text(content)
            r, k = classify_user_text(text)
            tx.push(r, k, text, timestamp=stamp)
        elif role in ("assistant", "ai", "model", "gemini"):
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        name = block.get("name") or "tool"
                        tx.push("assistant", "tool_call", f"{name}\n{safe_json(block.get('input') or {})}",
                                timestamp=stamp, tool_name=name, tool_call_id=block.get("id"))
                    elif isinstance(block, dict) and block.get("type") == "thinking":
                        tx.push("assistant", "reasoning", extract_text(block), timestamp=stamp)
                    else:
                        tx.push("assistant", "message", extract_text(block), timestamp=stamp)
            else:
                tx.push("assistant", "message", extract_text(content), timestamp=stamp)
        elif role in ("tool", "tool_result", "function"):
            tx.push("tool", "tool_result", extract_text(content), timestamp=stamp,
                    tool_call_id=record.get("tool_call_id") or record.get("tool_use_id"))
        elif role in ("system",):
            tx.push("system", "system", extract_text(content), timestamp=stamp)

    valid = read_jsonl(source, tx, on_record)
    tx.finish(f"parse_generic_jsonl[{surface}]",
              {"record": "role|type in user|assistant|tool with content|text"}, valid)
    return Session(provider=provider, surface=surface, native_id=native_id, source_path=source.file,
                   messages=tx.messages, diagnostics=tx.diagnostics, cwd=state["cwd"],
                   title=tx.title_from_first_user())


# -------------------------------------------------------------- Gemini CLI


def parse_gemini(source: Source) -> Session:
    tx = Transcript(source, include_reasoning())
    native_id = native_id_from_path(source)
    valid = 0
    data: Any = None
    try:
        with open(source.file, "r", encoding="utf-8", errors="replace") as fh:
            data = json.load(fh)
        valid = 1
    except (OSError, ValueError) as exc:
        tx.diagnose("MALFORMED_JSONL", "error", f"could not parse Gemini chat json: {exc}")
    messages = data.get("messages") if isinstance(data, dict) else data
    if isinstance(data, dict) and data.get("sessionId"):
        native_id = str(data["sessionId"])
    for item in messages if isinstance(messages, list) else []:
        if not isinstance(item, dict):
            continue
        itype = str(item.get("type") or item.get("role") or "unknown")
        tx.saw(itype)
        stamp = item.get("timestamp")
        if itype == "user":
            text = extract_text(item.get("content"))
            r, k = classify_user_text(text)
            tx.push(r, k, text, timestamp=stamp, native_id=item.get("id"))
        elif itype in ("gemini", "model", "assistant"):
            tx.push("assistant", "reasoning", extract_text(item.get("thoughts")), timestamp=stamp,
                    native_id=item.get("id"))
            tx.push("assistant", "message", extract_text(item.get("content")), timestamp=stamp,
                    native_id=item.get("id"))
            for call in item.get("toolCalls") or []:
                if not isinstance(call, dict):
                    continue
                name = call.get("name") or "tool"
                tx.push("assistant", "tool_call", f"{name}\n{safe_json(call.get('args') or {})}",
                        timestamp=stamp, tool_name=name, tool_call_id=call.get("id"))
                if call.get("result") is not None:
                    tx.push("tool", "tool_result", extract_text(call.get("result")), timestamp=stamp,
                            tool_call_id=call.get("id"))
    tx.finish("parse_gemini", {"file": "messages[] with type=user|gemini, content, toolCalls[]"}, valid)
    return Session(provider="gemini", surface="gemini_cli", native_id=native_id, source_path=source.file,
                   messages=tx.messages, diagnostics=tx.diagnostics,
                   cwd=source.extra.get("folder"), title=tx.title_from_first_user(),
                   meta={"project_hash": data.get("projectHash")} if isinstance(data, dict) and data.get("projectHash") else {})


# ---------------------------------------------------------------- discovery


def _home() -> Path:
    return Path.home()


def codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME") or (_home() / ".codex"))


def claude_home() -> Path:
    return Path(os.environ.get("CLAUDE_CONFIG_DIR") or (_home() / ".claude"))


def copilot_home() -> Path:
    return Path(os.environ.get("COPILOT_HOME") or (_home() / ".copilot"))


def cursor_home() -> Path:
    return _home() / ".cursor"


def gemini_home() -> Path:
    return _home() / ".gemini"


# VS Code and its forks share the User-data layout. provider says whose chat
# it is, because a Cursor chat is not a Copilot chat.
_EDITOR_PRODUCTS = (
    ("Code", "copilot"), ("Code - Insiders", "copilot"), ("VSCodium", "copilot"),
    ("Cursor", "cursor"), ("Windsurf", "windsurf"), ("Kiro", "kiro"), ("Trae", "trae"),
)


def editor_user_dirs() -> list[tuple[str, str, Path]]:
    """(product, provider, <product>/User) for every editor we know, present or not."""
    out = []
    home = _home()
    for product, provider in _EDITOR_PRODUCTS:
        if sys.platform == "win32":
            base = Path(os.environ.get("APPDATA") or (home / "AppData" / "Roaming"))
            user = base / product / "User"
        elif sys.platform == "darwin":
            user = home / "Library" / "Application Support" / product / "User"
        else:
            base = Path(os.environ.get("XDG_CONFIG_HOME") or (home / ".config"))
            user = base / product / "User"
        out.append((product, provider, user))
    return out


def _walk(root: Path, predicate: Callable[[str], bool]) -> Iterator[str]:
    if not root.is_dir():
        return
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            full = os.path.join(dirpath, name)
            if predicate(full):
                yield full


def _codex_titles(home: Path) -> dict[str, str]:
    index = home / "session_index.jsonl"
    titles: dict[str, str] = {}
    if not index.is_file():
        return titles
    try:
        with open(index, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict) and row.get("id") and row.get("thread_name"):
                    titles[str(row["id"])] = str(row["thread_name"]).strip("* ").strip()
    except OSError:
        pass
    return titles


def discover_sources() -> list[Source]:
    sources: list[Source] = []

    codex = codex_home()
    titles = _codex_titles(codex)
    for sub in ("sessions", "archived_sessions"):
        for file in _walk(codex / sub, lambda f: f.endswith(".jsonl")):
            sources.append(Source("codex", "codex_cli", file, PARSER_VERSION["codex_cli"],
                                  {"titles": titles}))

    for file in _walk(claude_home() / "projects", lambda f: f.endswith(".jsonl")):
        sources.append(Source("claude", "claude_code", file, PARSER_VERSION["claude_code"]))

    for file in _walk(copilot_home() / "session-state", lambda f: os.path.basename(f) == "events.jsonl"):
        sources.append(Source("copilot", "copilot_cli", file, PARSER_VERSION["copilot_cli"]))

    for product, provider, user in editor_user_dirs():
        if not user.is_dir():
            continue

        def is_chat(f: str) -> bool:
            norm = f.replace("\\", "/")
            return ("/chatSessions/" in norm or "/emptyWindowChatSessions/" in norm) \
                and f.endswith((".json", ".jsonl"))

        for sub in ("workspaceStorage", "globalStorage"):
            for file in _walk(user / sub, is_chat):
                sources.append(Source(provider, "vscode_chat", file, PARSER_VERSION["vscode_chat"],
                                      {"product": product}))
        if product == "Cursor":
            sources.extend(cursor_composer_sources(str(user)))

    for file in _walk(cursor_home() / "projects",
                      lambda f: "agent-transcripts" in f.replace("\\", "/") and f.endswith(".jsonl")):
        sources.append(Source("cursor", "cursor_transcript", file, PARSER_VERSION["cursor_transcript"]))

    for file in _walk(gemini_home() / "tmp",
                      lambda f: os.path.basename(os.path.dirname(f)) == "chats" and f.endswith(".json")):
        sources.append(Source("gemini", "gemini_cli", file, PARSER_VERSION["gemini_cli"]))

    extra = os.environ.get(ENV_EXTRA_ROOTS, "")
    for raw in filter(None, extra.split(os.pathsep)):
        for file in _walk(Path(raw).expanduser(), lambda f: f.endswith(".jsonl")):
            sources.append(Source("other", "cursor_transcript", file, PARSER_VERSION["cursor_transcript"]))

    return _apply_tombstones(sources)


def _apply_tombstones(sources: list[Source]) -> list[Source]:
    """exclude_all means the user purged this conversation. Honour it at the
    point of discovery so nothing downstream has to remember to."""
    try:
        store = transcript_archive.ArchiveStore()
    except sqlite3.Error:
        return sources
    try:
        if not store.tombstones(refresh=True):
            return sources
        return [s for s in sources
                if store.tombstone_mode(s.file, s.live_path) != "exclude_all"]
    finally:
        store.close()


def archive_sources() -> list[Source]:
    """Our own copies, as index sources. A session archived while the vendor
    still has it keeps one identity (see Source.identity_path), so the index
    does not double-count it."""
    def factory(provider: str, surface: str, file: str, live_path: str | None, origin: str) -> Source:
        return Source(provider, surface, file,
                      PARSER_VERSION.get(surface, "generic-v1"),
                      live_path=live_path, origin=origin)

    return transcript_archive.archived_sources(factory)


def all_sources(include_archive: bool = True) -> list[Source]:
    """Live vendor files plus archive copies, deduplicated by the bytes read.

    Both are searched because each holds what the other does not: the live file
    has everything written since the last sync, the archive has everything the
    vendor has since deleted.
    """
    sources = discover_sources()
    if not include_archive:
        return sources
    seen = {identity_of(s.file) for s in sources}
    for source in archive_sources():
        marker = identity_of(source.file)
        if marker not in seen:
            seen.add(marker)
            sources.append(source)
    return sources


def parse_source(source: Source) -> Session:
    if source.surface == "codex_cli":
        session = parse_codex(source)
    elif source.surface == "claude_code":
        session = parse_claude(source)
    elif source.surface == "copilot_cli":
        session = parse_copilot_cli(source)
    elif source.surface == "vscode_chat":
        session = parse_vscode_chat(source)
    elif source.surface == "cursor_composer":
        session = parse_cursor_composer(source)
    elif source.surface == "gemini_cli":
        session = parse_gemini(source)
    else:
        session = parse_generic_jsonl(source, source.provider, source.surface)
    session.origin = source.origin
    session.live_path = source.live_path or source.file
    session.identity_path = source.identity_path
    return session


# ------------------------------------------------------------------- store


class Index:
    """The transcripts store. One per machine, under the ICN data root."""

    def __init__(self, db_path: Path | None = None):
        self.path = db_path or paths.transcripts_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=10.0, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.conn.execute("PRAGMA busy_timeout = 10000")
        self.conn.executescript(SCHEMA)
        if sys.platform != "win32":
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass

    def close(self) -> None:
        self.conn.close()

    # -- writes ------------------------------------------------------------

    def _delete_session(self, key: str) -> None:
        # External-content FTS needs the old text to remove its tokens.
        for msg in self.conn.execute("SELECT id, text FROM messages WHERE session_key = ?", (key,)):
            self.conn.execute("INSERT INTO messages_fts(messages_fts, rowid, text) VALUES('delete', ?, ?)",
                              (msg[0], msg[1]))
        self.conn.execute("DELETE FROM messages WHERE session_key = ?", (key,))
        self.conn.execute("DELETE FROM sessions WHERE session_key = ?", (key,))

    def _delete_source(self, source_path: str) -> None:
        for row in self.conn.execute(
                "SELECT session_key FROM sessions WHERE source_path = ? OR live_path = ?",
                (source_path, source_path)):
            self._delete_session(row[0])

    def forget_source(self, source_path: str) -> int:
        """Drop everything indexed from a vendor path or archive path."""
        with write_tx(self.conn):
            before = self.conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE source_path = ? OR live_path = ?",
                (source_path, source_path)).fetchone()[0]
            self._delete_source(source_path)
            self.conn.execute("DELETE FROM source_files WHERE source_path = ?", (source_path,))
        return before

    def save(self, session: Session, source: Source, mtime_ns: int | None, size: int | None) -> None:
        key = session_key(session.surface, session.native_id, session.identity_path or session.source_path)
        with write_tx(self.conn):
            # A session can be readable from two files at once - the live one and
            # the archive copy of it - and both are discovered every run. Only one
            # of them owns the session row (they share an identity by design), but
            # each keeps its own source_files row: that row records "these exact
            # bytes are already indexed", so dropping the loser's row would make
            # that file parse again on every refresh, forever.
            self._delete_session(key)
            self.conn.execute(
                "INSERT OR REPLACE INTO sessions (session_key, provider, surface, native_id, source_path,"
                " cwd, cwd_norm, repo_root, repo_root_norm, title, started_at, updated_at, message_count,"
                " meta_json, origin, live_path) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (key, session.provider, session.surface, session.native_id, session.source_path,
                 session.cwd, norm_path(session.cwd), session.repo_root, norm_path(session.repo_root),
                 session.title, session.started_at, session.updated_at, len(session.messages),
                 jdump(session.meta) if session.meta else None, session.origin,
                 session.live_path or session.source_path),
            )
            for m in session.messages:
                cur = self.conn.execute(
                    "INSERT INTO messages (session_key, ordinal, timestamp, role, kind, text, native_id,"
                    " tool_name, tool_call_id) VALUES (?,?,?,?,?,?,?,?,?)",
                    (key, m.ordinal, m.timestamp, m.role, m.kind, m.text, m.native_id, m.tool_name,
                     m.tool_call_id),
                )
                self.conn.execute("INSERT INTO messages_fts(rowid, text) VALUES (?, ?)",
                                  (cur.lastrowid, m.text))
            self.conn.execute(
                "INSERT OR REPLACE INTO source_files (source_path, provider, surface, mtime_ns, size_bytes,"
                " parser_version, status, diagnostics_json, indexed_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (source.file, source.provider, source.surface, mtime_ns, size, source.parser_version,
                 "ok" if not any(d.severity == "error" for d in session.diagnostics) else "degraded",
                 jdump([d.as_dict() for d in session.diagnostics]) or "[]", now_iso()),
            )

    def record_failure(self, source: Source, mtime_ns: int | None, size: int | None,
                       diagnostics: list[Diagnostic]) -> None:
        """Keep whatever normalised copy exists; only the source row changes."""
        with write_tx(self.conn):
            self.conn.execute(
                "INSERT INTO source_files (source_path, provider, surface, mtime_ns, size_bytes,"
                " parser_version, status, diagnostics_json, indexed_at) VALUES (?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(source_path) DO UPDATE SET mtime_ns=excluded.mtime_ns,"
                " size_bytes=excluded.size_bytes, parser_version=excluded.parser_version,"
                " status='error', diagnostics_json=excluded.diagnostics_json, indexed_at=excluded.indexed_at",
                (source.file, source.provider, source.surface, mtime_ns, size, source.parser_version,
                 "error", jdump([d.as_dict() for d in diagnostics]) or "[]", now_iso()),
            )

    # -- refresh -----------------------------------------------------------

    def refresh(self, sources: list[Source] | None = None, limit_seconds: float | None = None,
                archive: bool | None = None) -> dict[str, Any]:
        started = time.monotonic()
        archive_stats: dict[str, Any] | None = None
        if sources is None:
            live = discover_sources()
            # Copy first, then index: a file archived in this run is indexed in
            # this run, and a vendor deletion between the two loses nothing.
            if archive is not False and transcript_archive.archive_enabled():
                budget = None if limit_seconds is None else max(1.0, limit_seconds * 0.5)
                archive_stats = transcript_archive.sync_all(live, limit_seconds=budget)
            seen = {identity_of(s.file) for s in live}
            sources = live + [s for s in archive_sources()
                              if identity_of(s.file) not in seen]
        seen = {s.file for s in sources}
        counts = {"discovered": len(sources), "indexed": 0, "unchanged": 0, "failed": 0,
                  "skipped_time_budget": 0}
        run_diagnostics: list[dict[str, Any]] = []
        known = {r["source_path"]: r for r in rows(self.conn.execute(
            "SELECT source_path, mtime_ns, size_bytes, parser_version FROM source_files"))}

        for source in sources:
            if limit_seconds is not None and time.monotonic() - started > limit_seconds:
                counts["skipped_time_budget"] += 1
                continue
            mtime_ns: int | None
            size: int | None
            if source.surface == "cursor_composer":
                updated = source.extra.get("updated")
                mtime_ns = int(updated) if isinstance(updated, (int, float)) else None
                size = None
            else:
                try:
                    st = os.stat(source.file)
                    mtime_ns, size = st.st_mtime_ns, st.st_size
                except OSError as exc:
                    diag = Diagnostic("IO_ERROR", "error", source.provider, source.surface, source.file,
                                      source.parser_version, f"could not stat: {exc}")
                    self.record_failure(source, None, None, [diag])
                    run_diagnostics.append(diag.as_dict())
                    counts["failed"] += 1
                    continue
            old = known.get(source.file)
            if old and old["mtime_ns"] == mtime_ns and old["size_bytes"] == size \
                    and old["parser_version"] == source.parser_version:
                counts["unchanged"] += 1
                continue
            try:
                session = parse_source(source)
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
                diag = Diagnostic("IO_ERROR", "error", source.provider, source.surface, source.file,
                                  source.parser_version, f"adapter crashed: {type(exc).__name__}: {exc}",
                                  fix={"where": f"parse for {source.surface}",
                                       "action": "run conversations(action='schema') on this source"})
                self.record_failure(source, mtime_ns, size, [diag])
                run_diagnostics.append(diag.as_dict())
                counts["failed"] += 1
                continue
            fatal = any(d.code == "SCHEMA_DRIFT" and d.severity == "error" for d in session.diagnostics)
            if fatal and old is not None:
                # The vendor changed format under us. Keep the last good copy;
                # never replace searchable history with nothing.
                self.record_failure(source, mtime_ns, size, session.diagnostics)
                run_diagnostics.extend(d.as_dict() for d in session.diagnostics)
                counts["failed"] += 1
                continue
            self.save(session, source, mtime_ns, size)
            run_diagnostics.extend(d.as_dict() for d in session.diagnostics if d.severity != "info")
            counts["indexed"] += 1
            if fatal:
                counts["failed"] += 1

        missing = 0
        for source_path, row in known.items():
            if source_path in seen:
                continue
            real = source_path.split("#", 1)[0]
            if os.path.exists(real) and "#" not in source_path:
                continue  # exists but was not discovered this run (env changed); leave it
            missing += 1
            if retention_mode() == "source":
                with write_tx(self.conn):
                    self._delete_source(source_path)
                    self.conn.execute("DELETE FROM source_files WHERE source_path = ?", (source_path,))
                continue
            with write_tx(self.conn):
                self.conn.execute(
                    "UPDATE source_files SET status = 'source_missing', indexed_at = ? WHERE source_path = ?"
                    " AND status != 'source_missing'", (now_iso(), source_path))
                # The vendor no longer has this conversation; our copy is the
                # only one left. Say so in the origin, because "you can still
                # read this only because it was archived" is the distinction
                # the whole archive exists to make.
                self.conn.execute(
                    "UPDATE sessions SET origin = 'archive_only'"
                    " WHERE live_path = ? AND origin = 'archive'", (source_path,))
        counts["missing_sources"] = missing
        counts["retention"] = retention_mode()
        counts["seconds"] = round(time.monotonic() - started, 2)
        counts["diagnostics"] = run_diagnostics[:50]
        if archive_stats is not None:
            counts["archive"] = archive_stats
        return counts

    # -- reads -------------------------------------------------------------

    def diagnostics(self, limit: int = 50) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in self.conn.execute(
                "SELECT source_path, status, diagnostics_json FROM source_files WHERE status IN ('error','degraded')"
                " ORDER BY indexed_at DESC LIMIT ?", (limit,)):
            for d in jload(row["diagnostics_json"], []) or []:
                if isinstance(d, dict) and d.get("severity") != "info":
                    out.append(d)
        return out[:limit]

    def counts(self) -> list[dict[str, Any]]:
        return rows(self.conn.execute(
            "SELECT provider, surface, origin, COUNT(*) AS sessions,"
            " COALESCE(SUM(message_count),0) AS messages,"
            " MIN(started_at) AS earliest, MAX(updated_at) AS latest FROM sessions"
            " GROUP BY provider, surface, origin ORDER BY provider, surface, origin"))

    def source_status(self) -> list[dict[str, Any]]:
        return rows(self.conn.execute(
            "SELECT provider, surface, status, COUNT(*) AS files FROM source_files"
            " GROUP BY provider, surface, status ORDER BY provider, surface, status"))


# ------------------------------------------------------------------ search


def fts_query(query: str, match: str = "all") -> str:
    """Turn a human query into FTS5 syntax that behaves the way developers
    expect for code: `src/auth/session.ts` and `foo_bar()` become phrases of
    their sub-tokens, a trailing * is a prefix, quoted text is a phrase."""
    if match == "phrase":
        words = re.findall(r"[^\W_]+", query, re.UNICODE)
        if not words:
            raise ValueError("search query contains no searchable words")
        return '"' + " ".join(words) + '"'
    terms: list[str] = []
    for chunk in re.findall(r'"[^"]+"|\S+', query):
        if chunk.startswith('"') and chunk.endswith('"') and len(chunk) > 2:
            words = re.findall(r"[^\W_]+", chunk, re.UNICODE)
            if words:
                terms.append('"' + " ".join(words) + '"')
            continue
        prefix = chunk.endswith("*")
        words = re.findall(r"[^\W_]+", chunk, re.UNICODE)
        if not words:
            continue
        if len(words) == 1:
            terms.append(f'"{words[0]}"' + ("*" if prefix else ""))
        else:
            terms.append('"' + " ".join(words) + '"' + ("*" if prefix else ""))
    if not terms:
        raise ValueError("search query contains no searchable words")
    return (" AND " if match != "any" else " OR ").join(terms)


# Fuzzy matching. FTS5 is exact-token: "tombstones" does not find "tombstone",
# and a typo finds nothing at all. Rather than bolt on a second index, we expand
# each query word against the terms that are actually in the index - the fts5vocab
# table is that list, for free - and OR the near ones in. Expansion is bounded by
# FUZZY_MAX_VARIANTS per word so a common prefix cannot build a thousand-clause
# query, and every variant is a real indexed term, so an expanded query can only
# match documents that exist.

FUZZY_MIN_WORD = 4          # shorter words are mostly noise; a 3-letter typo is a different word
FUZZY_MAX_VARIANTS = 12
FUZZY_VOCAB_SCAN = 60_000

VOCAB_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_vocab
    USING fts5vocab(messages_fts, row);
"""


def _edit_distance_within(a: str, b: str, limit: int) -> int | None:
    """Levenshtein distance, abandoned as soon as it exceeds `limit`.

    The early exit matters: this runs over every candidate term in the vocabulary,
    and the full matrix for a long pair is wasted work when we only care whether
    the distance is 1 or 2.
    """
    if abs(len(a) - len(b)) > limit:
        return None
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        best = i
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            value = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            current.append(value)
            best = min(best, value)
        if best > limit:
            return None
        previous = current
    return previous[-1] if previous[-1] <= limit else None


def _fuzzy_budget(word: str) -> int:
    """How many edits to forgive. Scaled by length: one edit in a 5-letter word is
    a typo, one edit in a 3-letter word is usually a different word entirely."""
    if len(word) >= 8:
        return 2
    if len(word) >= FUZZY_MIN_WORD:
        return 1
    return 0


def fuzzy_variants(index: "Index", word: str, limit: int = FUZZY_MAX_VARIANTS) -> list[str]:
    """Indexed terms within an edit or two of `word`, nearest first.

    Candidates are narrowed in SQL first (same first letter, similar length)
    because scanning the whole vocabulary in Python for every query word is the
    difference between milliseconds and seconds on a large index.
    """
    word = word.lower()
    budget = _fuzzy_budget(word)
    if budget == 0:
        return []
    try:
        index.conn.executescript(VOCAB_SCHEMA)
    except sqlite3.Error:
        return []
    scored: list[tuple[int, int, str]] = []
    try:
        rows_iter = index.conn.execute(
            "SELECT term, cnt FROM messages_vocab WHERE term >= ? AND term < ?"
            " AND length(term) BETWEEN ? AND ? LIMIT ?",
            (word[0], word[0] + "￿", len(word) - budget, len(word) + budget, FUZZY_VOCAB_SCAN))
        for term, cnt in rows_iter:
            if term == word:
                continue
            distance = _edit_distance_within(word, term, budget)
            if distance is not None:
                # Nearer first, then commoner: a frequent term is more likely the
                # word that was meant than a one-off that happens to be close.
                scored.append((distance, -int(cnt or 0), term))
    except sqlite3.Error:
        return []
    scored.sort()
    return [term for _d, _c, term in scored[:limit]]


def expand_query(index: "Index", query: str, match: str = "all") -> tuple[str, dict[str, list[str]]]:
    """Rewrite a query so each word also matches its near-misses in the index.

    Returns the FTS expression and what each word expanded to, so a caller can
    show the user why something matched a word they did not type.
    """
    base = fts_query(query, match)
    if match == "phrase":
        return base, {}
    joiner = " OR " if match == "any" else " AND "
    expansions: dict[str, list[str]] = {}
    parts: list[str] = []
    for chunk in re.findall(r'"[^"]+"|\S+', query):
        quoted = chunk.startswith('"') and chunk.endswith('"')
        words = re.findall(r"[^\W_]+", chunk, re.UNICODE)
        if not words:
            continue
        if quoted or len(words) > 1:
            # A phrase the user asked for stays a phrase; fuzzing inside it
            # would defeat the reason they quoted it.
            parts.append('"' + " ".join(words) + '"')
            continue
        word = words[0]
        variants = fuzzy_variants(index, word)
        if variants:
            expansions[word] = variants
            alts = " OR ".join(f'"{v}"' for v in [word, *variants])
            parts.append(f"({alts})")
        else:
            parts.append(f'"{word}"')
    if not parts:
        return base, {}
    return joiner.join(parts), expansions


def _origin_clause(origin: str, params: list[Any]) -> str:
    """origin filter: live | archive (any archived copy) | archive_only |
    archive_version, or an exact origin value."""
    if origin == "archive":
        return "s.origin IN ('archive', 'archive_only', 'archive_version')"
    params.append(origin)
    return "s.origin = ?"


def _codebase_clause(codebase: str, params: list[Any]) -> str:
    normalized = norm_path(codebase)
    params.extend([normalized, normalized + "/%", normalized, normalized + "/%"])
    return "(s.cwd_norm = ? OR s.cwd_norm LIKE ? OR s.repo_root_norm = ? OR s.repo_root_norm LIKE ?)"


def _session_row(row: Any) -> dict[str, Any]:
    return {
        "session_key": row["session_key"], "provider": row["provider"], "surface": row["surface"],
        "native_id": row["native_id"], "title": row["title"], "cwd": row["cwd"],
        "repo_root": row["repo_root"], "started_at": row["started_at"], "updated_at": row["updated_at"],
        "origin": row["origin"], "live_path": row["live_path"],
    }


def search(index: Index, query: str, *, provider: str | None = None, surface: str | None = None,
           codebase: str | None = None, role: str | None = None, kind: str | None = None,
           tool_name: str | None = None, session: str | None = None, after: str | None = None,
           before: str | None = None, match: str = "all", limit: int = 10, per_session: int = 3,
           context_before: int = 3, context_after: int = 3, sort: str = "relevance",
           origin: str | None = None, fuzzy: str | bool = "auto") -> dict[str, Any]:
    """fuzzy: "auto" (default) retries with near-miss expansion only when the
    exact query found little, so an exact match is never diluted by a fuzzy one;
    True always expands; False never does."""
    expansions: dict[str, list[str]] = {}
    fuzzy_used = False
    if fuzzy is True:
        expression, expansions = expand_query(index, query, match)
        fuzzy_used = bool(expansions)
    else:
        expression = fts_query(query, match)
    where = ["messages_fts MATCH ?"]
    params: list[Any] = [expression]
    if origin:
        where.append(_origin_clause(origin, params))
    if provider:
        where.append("s.provider = ?"); params.append(provider)
    if surface:
        where.append("s.surface = ?"); params.append(surface)
    if codebase:
        where.append(_codebase_clause(codebase, params))
    if role:
        where.append("m.role = ?"); params.append(role)
    if kind:
        where.append("m.kind = ?"); params.append(kind)
    if tool_name:
        where.append("m.tool_name = ?"); params.append(tool_name)
    if session:
        where.append("(s.session_key = ? OR s.native_id = ?)"); params.extend([session, session])
    if after:
        where.append("COALESCE(m.timestamp, s.updated_at, s.started_at) >= ?"); params.append(to_iso(after) or after)
    if before:
        where.append("COALESCE(m.timestamp, s.updated_at, s.started_at) <= ?"); params.append(to_iso(before) or before)
    order = "score ASC" if sort != "recent" else "COALESCE(m.timestamp, s.updated_at) DESC"
    overfetch = max(limit * max(per_session, 1) * 4, 50)
    params.append(overfetch)
    sql = f"""
        SELECT m.id, m.session_key, m.ordinal, m.timestamp, m.role, m.kind, m.text, m.tool_name,
               m.tool_call_id, s.provider, s.surface, s.native_id, s.cwd, s.repo_root, s.title,
               s.started_at, s.updated_at, s.origin, s.live_path, bm25(messages_fts) AS score,
               snippet(messages_fts, 0, '[', ']', ' ... ', 24) AS snippet
        FROM messages_fts
        JOIN messages m ON m.id = messages_fts.rowid
        JOIN sessions s ON s.session_key = m.session_key
        WHERE {" AND ".join(where)}
        ORDER BY {order}
        LIMIT ?
    """
    hits, per, total_seen = _collect_hits(index, sql, params, limit, per_session,
                                          context_before, context_after)

    # Nothing (or almost nothing) matched exactly: the word was probably
    # misspelled, or is a different form of one that is indexed. Retry once with
    # the expanded query rather than telling the user their history is empty.
    if fuzzy == "auto" and len(hits) < max(1, limit // 3):
        expression, expansions = expand_query(index, query, match)
        if expansions and expression != params[0]:
            retry = list(params)
            retry[0] = expression
            fuzzy_hits, _per, fuzzy_seen = _collect_hits(
                index, sql, retry, limit, per_session, context_before, context_after)
            if len(fuzzy_hits) > len(hits):
                hits, total_seen, fuzzy_used = fuzzy_hits, fuzzy_seen, True
                params = retry
            else:
                expansions = {}

    result = {
        "query": query, "fts": params[0], "hits": hits, "returned": len(hits),
        "candidates_scanned": total_seen, "capped": total_seen >= overfetch,
        "note": ("hits are capped per session (per_session) so one long session does not crowd out "
                 "the rest; pass session=<key> to see everything from one session"),
    }
    if fuzzy_used:
        result["fuzzy"] = {
            "applied": True,
            "expanded": expansions,
            "note": "no exact match, so each word was matched against similar terms in the index",
        }
    return result


def _collect_hits(index: Index, sql: str, params: list[Any], limit: int, per_session: int,
                  context_before: int, context_after: int) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    hits: list[dict[str, Any]] = []
    per: dict[str, int] = {}
    total_seen = 0
    for row in index.conn.execute(sql, params):
        total_seen += 1
        key = row["session_key"]
        if per.get(key, 0) >= per_session:
            continue
        per[key] = per.get(key, 0) + 1
        text = row["text"]
        hits.append({
            "score": round(row["score"], 3),
            "snippet": row["snippet"],
            "session": _session_row(row),
            "match": {
                "ordinal": row["ordinal"], "timestamp": row["timestamp"], "role": row["role"],
                "kind": row["kind"], "tool_name": row["tool_name"], "tool_call_id": row["tool_call_id"],
                "text": text if len(text) <= MAX_HIT_TEXT else text[:MAX_HIT_TEXT] + "\n[truncated]",
            },
            "context": context(index, key, row["ordinal"], context_before, context_after),
        })
        if len(hits) >= limit:
            break
    return hits, per, total_seen


def context(index: Index, key: str, ordinal: int, before: int, after: int,
            max_chars: int = 1_500) -> list[dict[str, Any]]:
    out = []
    for row in index.conn.execute(
            "SELECT ordinal, timestamp, role, kind, text, tool_name FROM messages"
            " WHERE session_key = ? AND ordinal BETWEEN ? AND ? ORDER BY ordinal",
            (key, max(0, ordinal - before), ordinal + after)):
        text = row["text"]
        out.append({"ordinal": row["ordinal"], "timestamp": row["timestamp"], "role": row["role"],
                    "kind": row["kind"], "tool_name": row["tool_name"],
                    "text": text if len(text) <= max_chars else text[:max_chars] + " [...]",
                    "is_match": row["ordinal"] == ordinal})
    return out


# What investigate() pulls from here. Conversational turns only: a tool_result
# is a file or shell dump, and a query naming any identifier would otherwise be
# answered by the file that defines it rather than by anyone discussing it.
RECALL_KINDS = ("message", "summary")
RECALL_TEXT_CHARS = 700


def recall(index: Index, terms: list[str], *, codebase: str | None, limit: int = 5,
           per_session: int = 2) -> dict[str, Any]:
    """Compact hits for investigate(): what past sessions, from any agent,
    said about these terms in one codebase.

    Terms are ORed and ranked by bm25, so a turn discussing several of them
    outranks one that mentions a single common word. No fuzzy retry and no
    surrounding context: this is a lead to follow with conversations(action=
    'get', session=...), not the full answer, and it must stay cheap enough
    to run on every investigation."""
    words = [t for t in terms if t.strip()]
    if not words:
        return {"hits": [], "scanned": 0}
    expression = fts_query(" ".join(words), "any")
    kinds = ",".join("?" for _ in RECALL_KINDS)
    where = ["messages_fts MATCH ?", f"m.kind IN ({kinds})"]
    params: list[Any] = [expression, *RECALL_KINDS]
    if codebase:
        where.append(_codebase_clause(codebase, params))
    overfetch = max(limit * max(per_session, 1) * 4, 50)
    params.append(overfetch)
    sql = f"""
        SELECT m.session_key, m.ordinal, m.timestamp, m.role, m.kind, m.text,
               s.provider, s.surface, s.native_id, s.cwd, s.repo_root, s.title,
               s.started_at, s.updated_at, s.origin, s.live_path, bm25(messages_fts) AS score,
               snippet(messages_fts, 0, '[', ']', ' ... ', 24) AS snippet
        FROM messages_fts
        JOIN messages m ON m.id = messages_fts.rowid
        JOIN sessions s ON s.session_key = m.session_key
        WHERE {" AND ".join(where)}
        ORDER BY score ASC
        LIMIT ?
    """
    hits: list[dict[str, Any]] = []
    per: dict[str, int] = {}
    scanned = 0
    for row in index.conn.execute(sql, params):
        scanned += 1
        key = row["session_key"]
        if per.get(key, 0) >= per_session:
            continue
        per[key] = per.get(key, 0) + 1
        text = row["text"]
        hits.append({
            "score": round(row["score"], 3),
            "snippet": row["snippet"],
            "session": _session_row(row),
            "ordinal": row["ordinal"], "timestamp": row["timestamp"], "role": row["role"],
            "text": text if len(text) <= RECALL_TEXT_CHARS else text[:RECALL_TEXT_CHARS] + " [...]",
        })
        if len(hits) >= limit:
            break
    return {"hits": hits, "scanned": scanned}


def list_sessions(index: Index, *, provider: str | None = None, surface: str | None = None,
                  codebase: str | None = None, after: str | None = None, before: str | None = None,
                  title: str | None = None, limit: int = 50, offset: int = 0,
                  origin: str | None = None) -> dict[str, Any]:
    where: list[str] = []
    params: list[Any] = []
    if origin:
        where.append(_origin_clause(origin, params))
    if provider:
        where.append("s.provider = ?"); params.append(provider)
    if surface:
        where.append("s.surface = ?"); params.append(surface)
    if codebase:
        where.append(_codebase_clause(codebase, params))
    if after:
        where.append("COALESCE(s.updated_at, s.started_at) >= ?"); params.append(to_iso(after) or after)
    if before:
        where.append("COALESCE(s.started_at, s.updated_at) <= ?"); params.append(to_iso(before) or before)
    if title:
        where.append("s.title LIKE ?"); params.append(f"%{title}%")
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    total = index.conn.execute(f"SELECT COUNT(*) FROM sessions s {clause}", params).fetchone()[0]
    items = rows(index.conn.execute(
        f"SELECT s.session_key, s.provider, s.surface, s.native_id, s.source_path, s.live_path, s.origin,"
        f" s.cwd, s.repo_root, s.title, s.started_at, s.updated_at, s.message_count, s.meta_json"
        f" FROM sessions s {clause}"
        f" ORDER BY COALESCE(s.updated_at, s.started_at) DESC LIMIT ? OFFSET ?",
        [*params, limit, offset]))
    for item in items:
        item["meta"] = jload(item.pop("meta_json"), None)
    return {"sessions": items, "total": total, "offset": offset,
            "next_offset": offset + len(items) if offset + len(items) < total else None}


def _find_sessions(index: Index, identifier: str) -> list[dict[str, Any]]:
    found = rows(index.conn.execute("SELECT * FROM sessions WHERE session_key = ?", (identifier,)))
    if not found:
        found = rows(index.conn.execute(
            "SELECT * FROM sessions WHERE native_id = ? OR source_path = ? OR live_path = ?"
            " ORDER BY updated_at DESC", (identifier, identifier, identifier)))
    if not found:
        found = rows(index.conn.execute(
            "SELECT * FROM sessions WHERE native_id LIKE ? ORDER BY updated_at DESC LIMIT 5",
            (identifier + "%",)))
    for item in found:
        item["meta"] = jload(item.pop("meta_json"), None)
    return found


def get_session(index: Index, identifier: str, *, start: int = 0, limit: int = 200,
                roles: list[str] | None = None, kinds: list[str] | None = None,
                max_chars: int = 4_000) -> dict[str, Any]:
    found = _find_sessions(index, identifier)
    if not found:
        return {"sessions": [], "error": f"no session matches {identifier!r} as session_key, native_id "
                                         "or source_path"}
    out = []
    for session in found:
        where = ["session_key = ?", "ordinal >= ?"]
        params: list[Any] = [session["session_key"], start]
        if roles:
            where.append(f"role IN ({','.join('?' * len(roles))})"); params.extend(roles)
        if kinds:
            where.append(f"kind IN ({','.join('?' * len(kinds))})"); params.extend(kinds)
        params.append(limit + 1)
        messages = rows(index.conn.execute(
            f"SELECT ordinal, timestamp, role, kind, text, native_id, tool_name, tool_call_id FROM messages"
            f" WHERE {' AND '.join(where)} ORDER BY ordinal LIMIT ?", params))
        more = len(messages) > limit
        messages = messages[:limit]
        for m in messages:
            if len(m["text"]) > max_chars:
                m["text"] = m["text"][:max_chars] + f" [... {len(m['text']) - max_chars} more chars]"
        out.append({"session": session, "messages": messages,
                    "next_ordinal": (messages[-1]["ordinal"] + 1) if more and messages else None})
    return {"sessions": out}


# ------------------------------------------------------------------ digest
#
# A deterministic summary of one session, so an agent can decide whether a
# 2,000-message conversation is worth reading without reading it: what was
# asked, what tools ran, which files were touched, how it ended.

_PATH_RE = re.compile(
    r"(?:[A-Za-z]:[\\/]|/|\./|~/)?(?:[\w.\-@ ]+[\\/])+[\w.\-@]+\.[A-Za-z0-9]{1,8}"
)
_PATH_KEYS = ("file_path", "path", "filePath", "notebook_path", "target_file", "uri", "filename", "file")


def _paths_from_tool_call(text: str) -> set[str]:
    found: set[str] = set()
    _name, _sep, body = text.partition("\n")
    try:
        args = json.loads(body)
    except ValueError:
        args = None
    if isinstance(args, dict):
        for key in _PATH_KEYS:
            value = args.get(key)
            if isinstance(value, str) and 2 < len(value) < 400:
                found.add(value)
        for key in ("changes", "files"):
            value = args.get(key)
            if isinstance(value, dict):
                found.update(k for k in value if isinstance(k, str))
    if not found:
        for m in _PATH_RE.finditer(body[:4000]):
            candidate = m.group(0).strip()
            if 4 < len(candidate) < 300 and " " not in candidate.strip("./"):
                found.add(candidate)
    return found


def digest(index: Index, identifier: str, *, max_user_turns: int = 40, turn_chars: int = 240) -> dict[str, Any]:
    found = _find_sessions(index, identifier)
    if not found:
        return {"error": f"no session matches {identifier!r}"}
    out = []
    for session in found:
        key = session["session_key"]
        by_kind: dict[str, int] = {}
        by_role: dict[str, int] = {}
        tools: dict[str, int] = {}
        files: dict[str, int] = {}
        user_turns: list[dict[str, Any]] = []
        summaries: list[str] = []
        errors: list[str] = []
        last_assistant: str | None = None
        first_ts: str | None = None
        last_ts: str | None = None
        for m in index.conn.execute(
                "SELECT ordinal, timestamp, role, kind, text, tool_name FROM messages WHERE session_key = ?"
                " ORDER BY ordinal", (key,)):
            by_kind[m["kind"]] = by_kind.get(m["kind"], 0) + 1
            by_role[m["role"]] = by_role.get(m["role"], 0) + 1
            if m["timestamp"]:
                first_ts = first_ts or m["timestamp"]
                last_ts = m["timestamp"]
            if m["kind"] == "tool_call":
                name = m["tool_name"] or "tool"
                tools[name] = tools.get(name, 0) + 1
                for p in _paths_from_tool_call(m["text"]):
                    files[p] = files.get(p, 0) + 1
            elif m["kind"] == "message" and m["role"] == "user":
                if len(user_turns) < max_user_turns:
                    text = re.sub(r"\s+", " ", m["text"]).strip()
                    user_turns.append({"ordinal": m["ordinal"], "timestamp": m["timestamp"],
                                       "text": text[:turn_chars] + ("..." if len(text) > turn_chars else "")})
            elif m["kind"] == "message" and m["role"] == "assistant":
                last_assistant = m["text"]
            elif m["kind"] == "summary":
                summaries.append(m["text"][:2000])
            elif m["kind"] == "system" and ("[error]" in m["text"][:12] or "[api_error]" in m["text"][:14]):
                if len(errors) < 10:
                    errors.append(m["text"][:300])
        top_files = sorted(files.items(), key=lambda kv: (-kv[1], kv[0]))[:40]
        out.append({
            "session": session,
            "span": {"first": first_ts, "last": last_ts},
            "counts": {"by_kind": by_kind, "by_role": by_role, "user_turns": by_role.get("user", 0)},
            "tools": dict(sorted(tools.items(), key=lambda kv: -kv[1])),
            "files_touched": [{"path": p, "mentions": n} for p, n in top_files],
            "user_turns": user_turns,
            "user_turns_truncated": by_role.get("user", 0) > len(user_turns),
            "compaction_summaries": summaries,
            "errors": errors,
            "last_assistant_message": (last_assistant[:3000] + ("..." if len(last_assistant) > 3000 else ""))
            if last_assistant else None,
        })
    return {"digests": out,
            "note": "deterministic digest from the normalised transcript, not an LLM summary; "
                    "user_turns is what was asked, files_touched is what tool calls named"}


# ------------------------------------------------------------------ doctor


def doctor(index: Index) -> dict[str, Any]:
    roots: dict[str, Any] = {
        "codex": {"path": str(codex_home() / "sessions"), "exists": (codex_home() / "sessions").is_dir()},
        "claude": {"path": str(claude_home() / "projects"), "exists": (claude_home() / "projects").is_dir()},
        "copilot_cli": {"path": str(copilot_home() / "session-state"),
                        "exists": (copilot_home() / "session-state").is_dir()},
        "cursor_transcripts": {"path": str(cursor_home() / "projects"), "exists": (cursor_home() / "projects").is_dir()},
        "gemini": {"path": str(gemini_home() / "tmp"), "exists": (gemini_home() / "tmp").is_dir()},
        "editors": [{"product": product, "provider": provider, "path": str(user), "exists": user.is_dir()}
                    for product, provider, user in editor_user_dirs()],
    }
    status = index.source_status()
    problems = [row for row in status if row["status"] in ("error", "degraded")]
    return {
        "index": {"path": str(index.path), "retention": retention_mode(),
                  "include_reasoning": include_reasoning(), "max_tool_text": _max_tool_text(),
                  "size_bytes": index.path.stat().st_size if index.path.exists() else 0},
        "roots": roots,
        "counts": index.counts(),
        "source_files": status,
        "adapters": {surface: PARSER_VERSION[surface] for surface in SURFACES},
        "problems": problems,
        "diagnostics": index.diagnostics(),
        "verdict": ("healthy" if not problems else
                    "some sources could not be parsed; read diagnostics and run "
                    "conversations(action='schema', source_path=...) on one of them"),
    }


# ------------------------------------------------------------------ schema


def inspect_schema(index: Index, source_path: str) -> dict[str, Any]:
    """Keys and types only. Never conversation text: this is for fixing an
    adapter, which does not require reading anyone's transcript."""
    row = one(index.conn.execute(
        "SELECT provider, surface, parser_version, status, diagnostics_json FROM source_files"
        " WHERE source_path = ?", (source_path,)))
    surface = row["surface"] if row else _guess_surface(source_path)
    provider = row["provider"] if row else "unknown"
    real = source_path.split("#", 1)[0]
    if not os.path.exists(real):
        return {"ok": False, "error": "SOURCE_MISSING", "source_path": source_path}
    source = Source(provider, surface or "cursor_transcript", source_path,
                    PARSER_VERSION.get(surface or "", "unknown"))
    tx = Transcript(source, False)

    if surface == "cursor_composer":
        try:
            conn = _open_ro(real)
        except sqlite3.Error as exc:
            return {"ok": False, "error": str(exc)}
        try:
            key = source_path.split("#", 1)[1]
            header_row = conn.execute("SELECT value FROM cursorDiskKV WHERE key = ?", (key,)).fetchone()
            header = jload(header_row[0] if header_row and isinstance(header_row[0], str)
                           else (header_row[0].decode("utf-8", "replace") if header_row else None), {})
            cid = key.split(":", 1)[1]
            bubble_row = conn.execute("SELECT value FROM cursorDiskKV WHERE key LIKE ? LIMIT 1",
                                      (f"bubbleId:{cid}:%",)).fetchone()
            bubble = jload(bubble_row[0] if bubble_row and isinstance(bubble_row[0], str)
                           else (bubble_row[0].decode("utf-8", "replace") if bubble_row else None), None)
            return {"ok": True, "source_path": source_path, "surface": surface,
                    "structural_samples": {"composerData": shape_of(header), "bubble": shape_of(bubble)},
                    "stored_status": row["status"] if row else None}
        finally:
            conn.close()

    if real.lower().endswith(".json"):
        try:
            with open(real, "r", encoding="utf-8", errors="replace") as fh:
                data = json.load(fh)
        except (OSError, ValueError) as exc:
            return {"ok": False, "error": str(exc)[:200], "source_path": source_path}
        return {"ok": True, "source_path": source_path, "surface": surface,
                "structural_sample": shape_of(data), "stored_status": row["status"] if row else None}

    types: dict[str, int] = {}
    shapes: dict[str, Any] = {}
    inspected = 0

    def on_record(record: dict[str, Any], _line: int) -> None:
        nonlocal inspected
        if inspected >= SCHEMA_SAMPLE_RECORDS:
            return
        inspected += 1
        rtype = str(record.get("type") or f"kind:{record.get('kind', 'unknown')}")
        payload_type = None
        payload = record.get("payload")
        if isinstance(payload, dict) and payload.get("type"):
            payload_type = str(payload["type"])
        elif isinstance(record.get("data"), dict) and record["data"].get("type"):
            payload_type = str(record["data"]["type"])
        label = f"{rtype}/{payload_type}" if payload_type else rtype
        types[label] = types.get(label, 0) + 1
        shapes.setdefault(label, shape_of(record))

    read_jsonl(source, tx, on_record, path=real)
    return {"ok": True, "source_path": source_path, "surface": surface, "inspected_records": inspected,
            "record_types": types, "structural_samples": shapes,
            "stored_status": row["status"] if row else None,
            "stored_diagnostics": jload(row["diagnostics_json"], []) if row else [],
            "read_diagnostics": [d.as_dict() for d in tx.diagnostics]}


def _guess_surface(source_path: str) -> str | None:
    norm = source_path.replace("\\", "/").lower()
    if "/.codex/" in norm or "/rollout-" in norm:
        return "codex_cli"
    if "/.claude/projects/" in norm:
        return "claude_code"
    if "/session-state/" in norm and norm.endswith("events.jsonl"):
        return "copilot_cli"
    if "/chatsessions/" in norm:
        return "vscode_chat"
    if "#composerdata:" in norm:
        return "cursor_composer"
    if "/chats/" in norm and norm.endswith(".json"):
        return "gemini_cli"
    return None
