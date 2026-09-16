"""Reinforcement, feedback, handoffs, rule promotion, and hook delivery.

Hook tests drive `icn hook` the way Claude Code and Codex do: a JSON payload on
stdin, a JSON envelope (or nothing) on stdout. The live runs against the real
CLIs are recorded in the session notes; these pin the behaviour they rely on.
"""

from __future__ import annotations

import io
import json

import pytest

from icn import hooks, install as install_mod, rules as rules_mod, server

BILLING = ('def settle(invoice_id, amount):\n    return {"invoice": invoice_id, "charged": amount}\n\n\n'
           'def refund(invoice_id):\n    return {"invoice": invoice_id}\n')
RULE = "settle() must be idempotent: a retried payment webhook must never charge an invoice twice."


@pytest.fixture
def billing(repo):
    repo.write("billing.py", BILLING)
    repo.write("notes.py", "def slug(text):\n    return text.lower()\n")
    repo.commit("init")
    root = str(repo.root)
    server.workspace(action="open", root=root)
    recorded = server.record(root=root, kind="decision", summary="settle idempotency",
                             invariants=[RULE], symbols=["settle"],
                             failed_attempts=["Deduplicating retries in an in-memory set was tried and "
                                              "rejected: each worker saw the retry as new."])
    by_kind = {m["kind"]: m["memory_id"] for m in recorded["memories_created"]}
    return {"repo": repo, "root": root, "rule": by_kind["invariant"],
            "attempt": by_kind["failed_attempt"]}


def hook(event, payload, agent="claude"):
    out = io.StringIO()
    assert hooks.run([event, "--agent", agent], stdin=io.StringIO(json.dumps(payload)), stdout=out) == 0
    text = out.getvalue().strip()
    return json.loads(text)["hookSpecificOutput"]["additionalContext"] if text else ""


def edit(billing, session, name="billing.py", tool="Edit"):
    return {"session_id": session, "cwd": billing["root"], "tool_name": tool,
            "tool_input": {"file_path": str(billing["repo"].root / name)}}


# ------------------------------------------------------------- reinforcement

def test_restating_a_rule_reinforces_it_instead_of_copying(billing):
    again = server.record(root=billing["root"], kind="note", summary="checked", invariants=[RULE],
                          symbols=["settle"])
    assert again["memories_created"] == []
    assert again["memories_reinforced"][0]["memory_id"] == billing["rule"]
    assert again["memories_reinforced"][0]["evidence_count"] == 2
    assert again["primary_memory"] == billing["rule"]


def test_the_same_claim_about_different_code_is_a_new_memory_with_a_hint(billing):
    other = server.record(root=billing["root"], kind="note", summary="refund too",
                          invariants=[RULE.replace("settle()", "refund()")], symbols=["refund"])
    created = other["memories_created"][0]
    assert created["kind"] == "invariant"
    assert created["similar_to"]["memory_id"] == billing["rule"]


# ------------------------------------------------------------------ feedback

def test_feedback_wrong_needs_a_reason_and_withholds_until_verified(billing):
    root, rule = billing["root"], billing["rule"]
    assert server.memory(action="feedback", memory_id=rule, signal="wrong", root=root)["ok"] is False
    voted = server.memory(action="feedback", memory_id=rule, signal="wrong",
                          reason="settle is now keyed by payment intent", root=root)
    assert voted["ok"] and voted["confidence"] <= 0.3
    assert rule not in hook("pre-tool-use", edit(billing, "s-wrong"))

    server.memory(action="verify", memory_id=rule, root=root)
    assert rule in hook("pre-tool-use", edit(billing, "s-verified"))


def test_helpful_votes_count(billing):
    result = server.memory(action="feedback", memory_id=billing["rule"], signal="helpful",
                           root=billing["root"])
    assert result["helpful"] == 1


# ------------------------------------------------------------------ handoffs

def test_a_handoff_is_claimed_exactly_once_and_newer_supersedes(billing):
    root = billing["root"]
    first = server.memory(action="handoff", body="old state", root=root)
    second = server.memory(action="handoff", body="Adding idempotency_key.", next_steps=["migration"],
                           root=root)
    assert second["superseded"] == [first["handoff_id"]]

    opened = server.workspace(action="open", root=root)
    assert opened["handoff"]["handoff_id"] == second["handoff_id"]
    assert opened["handoff"]["next_steps"] == ["migration"]
    assert "handoff" not in server.workspace(action="open", root=root)
    assert "Adding idempotency_key" not in hook("session-start",
                                                {"session_id": "late", "cwd": root})


def test_session_start_claims_the_handoff_before_open_can(billing):
    root = billing["root"]
    server.memory(action="handoff", body="Resume the migration.", root=root)
    context = hook("session-start", {"session_id": "s1", "cwd": root}, agent="codex")
    assert "Resume the migration." in context and RULE[:40] in context
    assert "handoff" not in server.workspace(action="open", root=root)
    listed = server.memory(action="handoff_list", root=root)["handoffs"][0]
    assert listed["claimed_by"] == "codex:s1"


# ---------------------------------------------------------------------- hooks

def test_pre_tool_use_delivers_file_knowledge_once_per_session(billing):
    first = hook("pre-tool-use", edit(billing, "s1"))
    assert billing["rule"] in first and billing["attempt"] in first
    assert hook("pre-tool-use", edit(billing, "s1", tool="Read")) == ""
    assert billing["rule"] in hook("pre-tool-use", edit(billing, "s2"))


def test_pre_tool_use_is_silent_for_files_without_knowledge(billing):
    assert hook("pre-tool-use", edit(billing, "s1", name="notes.py")) == ""


def test_codex_patch_and_shell_calls_name_their_files(billing):
    patch = {"session_id": "c1", "cwd": billing["root"], "tool_name": "apply_patch",
             "tool_input": {"command": "*** Begin Patch\n*** Update File: billing.py\n@@\n-a\n+b\n"
                                       "*** End Patch"}}
    assert billing["rule"] in hook("pre-tool-use", patch, agent="codex")
    shell = {"session_id": "c2", "cwd": billing["root"], "tool_name": "shell",
             "tool_input": {"command": ["powershell", "-Command", "Get-Content billing.py"]}}
    assert billing["rule"] in hook("pre-tool-use", shell, agent="codex")


def test_knowledge_on_changed_code_is_delivered_labelled_unverified(billing):
    repo = billing["repo"]
    repo.write("billing.py", BILLING.replace(
        'return {"invoice": invoice_id, "charged": amount}',
        'if amount is None:\n        return None\n    total = amount * 2\n'
        '    return {"i": invoice_id, "t": total}'))
    repo.commit("rework settle")
    server.workspace(action="open", root=billing["root"])
    context = hook("pre-tool-use", edit(billing, "s-drift"))
    line = next(l for l in context.splitlines() if billing["rule"] in l)
    assert "unverified" in line


def test_hooks_fail_open(billing, tmp_path):
    out = io.StringIO()
    assert hooks.run(["pre-tool-use"], stdin=io.StringIO("not json"), stdout=out) == 0
    assert out.getvalue() == ""
    outside = {"session_id": "x", "cwd": str(tmp_path), "tool_name": "Edit",
               "tool_input": {"file_path": str(tmp_path / "a.py")}}
    assert hook("pre-tool-use", outside) == ""
    assert hooks.run(["unknown-event"], stdin=io.StringIO("{}"), stdout=out) == 0


def test_pre_tool_use_respects_its_budget(billing):
    root = billing["root"]
    for i in range(8):
        server.record(root=root, kind="note", summary=f"w{i}", symbols=["settle"],
                      warnings=[f"Warning number {i} about settle and the ledger {'x' * 300}"])
    context = hook("pre-tool-use", edit(billing, "s-budget"))
    assert len(context) < hooks.PRE_TOOL_BUDGET + 600
    assert context.count("\n- ") <= hooks.PRE_TOOL_MAX


# --------------------------------------------------------------------- rules

def test_rules_are_codebase_specific_capped_and_leave_the_rest_of_the_file_alone(billing):
    root, repo = billing["root"], billing["repo"]
    repo.write("CLAUDE.md", "# My project\n\nHand-written guidance.\n")

    recommended = server.memory(action="rules_recommend", root=root)
    rule = next(c for c in recommended["candidates"] if c["memory_id"] == billing["rule"])
    assert rule["rule"].startswith("In `billing.py` `settle`:")
    assert billing["attempt"] not in [c["memory_id"] for c in recommended["candidates"]]

    approved = server.memory(action="rules_approve", memory_id=billing["rule"], root=root)
    assert approved["files_changed"] == ["CLAUDE.md"]
    text = repo.read("CLAUDE.md")
    assert text.startswith("# My project\n\nHand-written guidance.\n")
    assert rules_mod.START in text and billing["rule"] in text

    refused = server.memory(action="rules_approve", memory_id=billing["rule"], root=root)
    assert refused["ok"] is False

    removed = server.memory(action="rules_remove", memory_id=billing["rule"], root=root)
    assert removed["ok"] and repo.read("CLAUDE.md").rstrip() == "# My project\n\nHand-written guidance."


def test_rules_refuse_past_the_cap_naming_the_weakest(billing, monkeypatch):
    monkeypatch.setattr(rules_mod, "DEFAULT_CAP", 0)
    from icn import workspace as ws_mod
    current = ws_mod.open_workspace(billing["root"])
    try:
        current.store.execute("INSERT INTO promoted_rules VALUES ('mem_x','x rule',0.1,'t','2026')")
        current.store.commit()
        with pytest.raises(rules_mod.RulesError, match="cap"):
            rules_mod.approve(current.store, current.root, billing["rule"], cap=1)
    finally:
        current.close()


def test_unverified_or_disputed_rules_are_not_eligible(billing):
    root = billing["root"]
    server.memory(action="feedback", memory_id=billing["rule"], signal="stale",
                  reason="migration changed settle", root=root)
    refused = server.memory(action="rules_approve", memory_id=billing["rule"], root=root)
    assert refused["ok"] is False and "not eligible" in refused["error"]


# ------------------------------------------------------------------ installer

def test_installer_writes_both_hooks_for_claude_and_codex(tmp_path):
    project = tmp_path / "p"
    project.mkdir()
    codex_file = tmp_path / "codex" / "hooks.json"
    codex_file.parent.mkdir()
    codex_file.write_text(json.dumps({"hooks": {"Stop": [{"hooks": [{"type": "command",
                                                                    "command": "echo mine"}]}]}}))
    install_mod.install(project, codex=True, codex_path=codex_file)
    install_mod.install(project, codex=True, codex_path=codex_file)

    claude = json.loads((project / install_mod.SETTINGS_PATH).read_text(encoding="utf-8"))["hooks"]
    codex = json.loads(codex_file.read_text(encoding="utf-8"))["hooks"]
    for config, agent in ((claude, "claude"), (codex, "codex")):
        assert len(config["SessionStart"]) == 1 and len(config["PreToolUse"]) == 1
        assert f"hook pre-tool-use --agent {agent}" in config["PreToolUse"][0]["hooks"][0]["command"]
    assert codex["Stop"][0]["hooks"][0]["command"] == "echo mine"


def test_codex_on_windows_uses_the_powershell_call_operator(monkeypatch):
    monkeypatch.setattr(install_mod.os, "name", "nt")
    assert install_mod.hook_command("pre-tool-use", "codex", "C:/py/python.exe").startswith('& "')
    assert install_mod.hook_command("pre-tool-use", "claude", "C:/py/python.exe").startswith('"')


# ------------------------------------------------------------------ relevance

def test_broadly_anchored_memories_apply_only_where_their_claim_points():
    from icn.relevance import focus
    anchors = [{"file_path": f"src/{n}.py", "symbol_path": s}
               for n, s in (("compiler", "_find_duplicate"), ("hooks", "pre_tool_use"),
                            ("rules", "approve"), ("lab", "run"), ("db", "connect"))]
    about_hooks = focus("hooks.run must always exit 0 on any error", anchors)
    assert about_hooks["broad"] and about_hooks["files"] == ["src/hooks.py"]
    plain_words = focus("A run with exit 0 must trust new hooks", anchors)
    assert plain_words["files"] == []
    narrow = focus("anything", anchors[:2])
    assert narrow["files"] == ["src/compiler.py", "src/hooks.py"] and not narrow["broad"]


def test_correcting_a_memory_changes_what_hooks_deliver(billing):
    fixed = "settle() must be idempotent per payment intent id, not per invoice."
    server.memory(action="correct", memory_id=billing["rule"], body=fixed, reason="keyed by intent",
                  root=billing["root"])
    context = hook("pre-tool-use", edit(billing, "s-corrected"))
    assert "per payment intent id" in context and "charge an invoice twice" not in context
