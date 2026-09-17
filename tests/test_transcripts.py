"""Transcript adapters, archive and search, on synthetic vendor files.

Synthetic rather than real: the adapters have to be testable on a machine that
has never run Codex, and a fixture is the only way to assert that a *specific*
vendor shape produces a *specific* normalised message. The shapes here were
taken from real files on a machine that has all of them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from icn import transcript_archive as archive_mod
from icn import transcripts as tx


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("INFINITE_CODE_HOME", str(tmp_path / "icn"))
    monkeypatch.delenv("ICN_TRANSCRIPTS_RETENTION", raising=False)
    # Point discovery at the fixtures instead of this machine's real vendor
    # stores: a test that indexed the developer's actual chat history would be
    # slow, non-deterministic, and a privacy problem.
    store = tmp_path / "vendor"
    store.mkdir()
    monkeypatch.setenv("ICN_TRANSCRIPTS_EXTRA_ROOTS", str(store))
    for var in ("CODEX_HOME", "CLAUDE_CONFIG_DIR", "COPILOT_HOME"):
        monkeypatch.setenv(var, str(tmp_path / "absent"))
    monkeypatch.setattr(tx, "_home", lambda: tmp_path / "absent")
    monkeypatch.setattr(tx, "editor_user_dirs", lambda: [])
    return tmp_path


def _jsonl(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


# ------------------------------------------------------------------ adapters


def test_codex_adapter_extracts_turns_and_tools(home):
    file = _jsonl(home / "rollout-x.jsonl", [
        {"type": "session_meta", "payload": {"id": "sess-1", "cwd": str(home / "repo"),
                                             "git": {"branch": "main"}}},
        {"type": "response_item", "payload": {"type": "message", "role": "user",
                                              "content": [{"type": "input_text", "text": "fix the parser"}]}},
        {"type": "response_item", "payload": {"type": "function_call", "name": "shell",
                                              "arguments": "{\"cmd\": \"pytest\"}", "call_id": "c1"}},
        {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "c1",
                                              "output": "3 passed"}},
        {"type": "response_item", "payload": {"type": "message", "role": "assistant",
                                              "content": [{"type": "output_text", "text": "tests pass"}]}},
    ])
    session = tx.parse_source(tx.Source("codex", "codex_cli", str(file), "v"))

    assert session.native_id == "sess-1"
    assert session.cwd == str(home / "repo")
    roles = [(m.role, m.kind) for m in session.messages]
    assert ("user", "message") in roles
    assert ("assistant", "tool_call") in roles
    assert ("tool", "tool_result") in roles
    assert any("fix the parser" in m.text for m in session.messages)
    assert not [d for d in session.diagnostics if d.severity == "error"]


def test_claude_adapter_reads_blocks_and_title(home):
    file = _jsonl(home / "claude" / "s.jsonl", [
        {"type": "ai-title", "aiTitle": "Parser work", "sessionId": "abc"},
        {"type": "user", "sessionId": "abc", "cwd": "/repo", "gitBranch": "main",
         "message": {"role": "user", "content": "why does it fail"}},
        {"type": "assistant", "sessionId": "abc", "message": {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "hidden"},
            {"type": "text", "text": "because of encoding"},
            {"type": "tool_use", "name": "Read", "id": "t1", "input": {"file_path": "a.py"}},
        ]}},
        {"type": "user", "sessionId": "abc", "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "file body"}]}},
    ])
    session = tx.parse_source(tx.Source("claude", "claude_code", str(file), "v"))

    assert session.title == "Parser work"
    assert session.cwd == "/repo"
    assert session.meta.get("branch") == "main"
    kinds = [m.kind for m in session.messages]
    assert "tool_call" in kinds and "tool_result" in kinds
    # Reasoning is dropped unless explicitly enabled: it is the largest and least
    # useful part of a transcript to search.
    assert "reasoning" not in kinds


def test_injected_user_context_is_not_a_user_turn(home):
    file = _jsonl(home / "c.jsonl", [
        {"type": "user", "sessionId": "s", "message": {"role": "user",
         "content": "<ide_opened_file>foo.py</ide_opened_file>"}},
        {"type": "user", "sessionId": "s", "message": {"role": "user", "content": "real question"}},
    ])
    session = tx.parse_source(tx.Source("claude", "claude_code", str(file), "v"))
    by_role = {m.role: m.text for m in session.messages}
    assert by_role["user"] == "real question"
    assert by_role["system"].startswith("<ide_opened_file>")


def test_metadata_only_file_is_empty_not_drift(home):
    """A session holding only an ai-title is a real thing Claude Code writes.
    Reporting it as SCHEMA_DRIFT would cry wolf about a vendor format change."""
    file = _jsonl(home / "t.jsonl", [{"type": "ai-title", "aiTitle": "x", "sessionId": "s"}])
    session = tx.parse_source(tx.Source("claude", "claude_code", str(file), "v"))
    assert session.messages == []
    assert [d for d in session.diagnostics if d.code == "SCHEMA_DRIFT"] == []


def test_unrecognised_shape_reports_drift_with_a_fix(home):
    file = _jsonl(home / "d.jsonl", [
        {"type": "user", "message": {"role": "user", "renamedField": "the text moved"}},
    ])
    session = tx.parse_source(tx.Source("claude", "claude_code", str(file), "v"))
    drift = [d for d in session.diagnostics if d.code == "SCHEMA_DRIFT"]
    assert drift and drift[0].fix and "parse" in drift[0].fix["where"]


# ------------------------------------------------------------------- search


def _index_with(home: Path, messages: list[tuple[str, str]], cwd: str = "/repo") -> tx.Index:
    records = [{"type": "user", "sessionId": "s1", "cwd": cwd,
                "message": {"role": role, "content": text}} for role, text in messages]
    file = _jsonl(home / "s1.jsonl", records)
    index = tx.Index()
    index.refresh(sources=[tx.Source("claude", "claude_code", str(file), "v")])
    return index


def test_search_returns_the_messages_around_a_hit(home):
    index = _index_with(home, [("user", f"turn number {i}") for i in range(10)])
    result = tx.search(index, "number 4", match="phrase", context_before=2, context_after=2)
    assert result["returned"] == 1
    hit = result["hits"][0]
    ordinals = [c["ordinal"] for c in hit["context"]]
    assert hit["match"]["ordinal"] in ordinals
    assert len(ordinals) > 1, "a hit without its surrounding turns is the thing this tool exists to avoid"
    index.close()


def test_search_scopes_to_a_codebase(home):
    a = _jsonl(home / "a.jsonl", [{"type": "user", "sessionId": "a", "cwd": "/repo-a",
                                   "message": {"role": "user", "content": "shared word alpha"}}])
    b = _jsonl(home / "b.jsonl", [{"type": "user", "sessionId": "b", "cwd": "/repo-b",
                                   "message": {"role": "user", "content": "shared word beta"}}])
    index = tx.Index()
    index.refresh(sources=[tx.Source("claude", "claude_code", str(a), "v"),
                           tx.Source("claude", "claude_code", str(b), "v")])
    scoped = tx.search(index, "shared", codebase="/repo-a")
    assert {h["session"]["cwd"] for h in scoped["hits"]} == {"/repo-a"}
    index.close()


def test_fuzzy_finds_a_misspelling_but_only_as_a_fallback(home):
    index = _index_with(home, [("user", "the heartbeat lease went stale")])

    assert tx.search(index, "heartbeat")["returned"] == 1, "exact match must work"
    typo = tx.search(index, "heartbet")
    assert typo["returned"] == 1
    assert typo["fuzzy"]["applied"] is True
    assert "heartbeat" in typo["fuzzy"]["expanded"]["heartbet"]

    # An exact hit must never be diluted by fuzzy variants of the same word.
    assert "fuzzy" not in tx.search(index, "heartbeat")
    assert tx.search(index, "heartbet", fuzzy=False)["returned"] == 0
    index.close()


def test_path_like_queries_are_phrases_not_token_soup(home):
    index = _index_with(home, [("user", "edited src/icn/transcripts.py today"),
                               ("user", "also touched src/other/py")])
    result = tx.search(index, "src/icn/transcripts.py")
    assert result["returned"] == 1
    index.close()


# ------------------------------------------------------------------ archive


def test_archive_appends_then_rotates_on_rewrite(home):
    live = _jsonl(home / "live.jsonl", [{"type": "user", "message": {"content": "one"}}])
    store = archive_mod.ArchiveStore()

    first = archive_mod.sync_file(store, "claude", "claude_code", str(live))
    assert first.action == "created"
    assert archive_mod.sync_file(store, "claude", "claude_code", str(live)).action == "unchanged"

    with open(live, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"type": "user", "message": {"content": "two"}}) + "\n")
    assert archive_mod.sync_file(store, "claude", "claude_code", str(live)).action == "appended"

    archived = archive_mod.archive_dir(first.key) / "transcript.jsonl"
    assert "one" in archived.read_text(encoding="utf-8")
    assert "two" in archived.read_text(encoding="utf-8")

    # The vendor rewrites the file (compaction, id reuse). The bytes we hold are
    # history the live file no longer has, so they must survive.
    _jsonl(live, [{"type": "user", "message": {"content": "replaced"}}])
    assert archive_mod.sync_file(store, "claude", "claude_code", str(live)).action == "rotated"
    versions = list((archive_mod.archive_dir(first.key) / "versions").glob("*.jsonl"))
    assert len(versions) == 1
    assert "one" in versions[0].read_text(encoding="utf-8")
    store.close()


def test_archive_survives_the_vendor_deleting_the_original(home):
    live = _jsonl(home / "vendor" / "gone.jsonl", [
        {"type": "user", "sessionId": "doomed", "cwd": "/repo",
         "message": {"role": "user", "content": "remember this exact sentence"}}])
    index = tx.Index()
    index.refresh()
    os.remove(live)

    index.refresh()
    found = tx.search(index, "remember this exact sentence", match="phrase")
    assert found["returned"] == 1, "the archive exists precisely so this still resolves"
    assert found["hits"][0]["session"]["origin"] == "archive_only"
    index.close()


def test_purge_stays_purged_across_a_later_sync(home):
    live = _jsonl(home / "vendor" / "junk.jsonl", [{"type": "user", "sessionId": "j", "cwd": "/repo",
                                         "message": {"role": "user", "content": "sensitive token xyz"}}])
    store = archive_mod.ArchiveStore()
    archive_mod.sync_file(store, "claude", "claude_code", str(live))

    archive_mod.delete_archive(store, str(live), "exclude_all", "contained a secret")
    # Deleting the copy alone would be pointless: the file is still on disk, so
    # the next sync would put it straight back.
    assert archive_mod.sync_file(store, "claude", "claude_code", str(live)).action == "skipped"
    store.close()

    index = tx.Index()
    index.refresh()
    assert tx.search(index, "sensitive token xyz", match="phrase")["returned"] == 0
    index.close()


def test_retention_is_unlimited_by_default(home):
    store = archive_mod.ArchiveStore()
    assert archive_mod.retention_days() == 0
    assert archive_mod.prune_retention(store)["retention_days"] == "unlimited"
    store.close()


def test_refresh_is_incremental_on_a_second_run(home):
    _jsonl(home / "vendor" / "a.jsonl", [{"type": "user", "sessionId": "a", "cwd": "/r",
                               "message": {"role": "user", "content": "hello"}}])
    index = tx.Index()
    index.refresh()
    again = index.refresh()
    # A background job that reparses everything every tick burns CPU forever.
    assert again["indexed"] == 0
    assert again["unchanged"] > 0
    index.close()
