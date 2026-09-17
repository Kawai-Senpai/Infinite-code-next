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


# ------------------------------------------------------------------ targeting

def shell(billing, session, command):
    return {"session_id": session, "cwd": billing["root"], "tool_name": "Bash",
            "tool_input": {"command": command}}


def test_searching_a_file_does_not_spend_its_delivery(billing):
    # A grep is not work on the file. Delivering then used up the once-per-
    # session delivery, and the rule was withheld when the file was edited.
    assert hook("pre-tool-use", shell(billing, "g1", "grep -n settle billing.py | head")) == ""
    assert hook("pre-tool-use", shell(billing, "g1", "git log -- billing.py")) == ""
    assert billing["rule"] in hook("pre-tool-use", edit(billing, "g1"))
    assert billing["rule"] in hook("pre-tool-use", shell(billing, "g2", "cd . && cat billing.py"))


def test_reads_and_edits_get_the_knowledge_of_the_code_they_touch(billing):
    path = str(billing["repo"].root / "billing.py")
    refund_line = BILLING[:BILLING.index("def refund")].count("\n") + 1
    elsewhere = {"session_id": "t1", "cwd": billing["root"], "tool_name": "Read",
                 "tool_input": {"file_path": path, "offset": refund_line, "limit": 2}}
    assert hook("pre-tool-use", elsewhere) == ""
    on_refund = {"session_id": "t1", "cwd": billing["root"], "tool_name": "Edit",
                 "tool_input": {"file_path": path, "old_string": "return {\"invoice\": invoice_id}\n",
                                "new_string": "return None\n"}}
    assert hook("pre-tool-use", on_refund) == ""
    on_settle = {"session_id": "t1", "cwd": billing["root"], "tool_name": "Edit",
                 "tool_input": {"file_path": path, "old_string": "\"charged\": amount",
                                "new_string": "\"charged\": amount or 0"}}
    assert billing["rule"] in hook("pre-tool-use", on_settle)


def test_an_edit_that_cannot_be_located_gets_the_whole_file(billing):
    path = str(billing["repo"].root / "billing.py")
    unlocatable = {"session_id": "u1", "cwd": billing["root"], "tool_name": "Edit",
                   "tool_input": {"file_path": path, "old_string": "not in the file",
                                  "new_string": "x"}}
    assert billing["rule"] in hook("pre-tool-use", unlocatable)


def test_module_level_lines_count_as_the_whole_file(billing):
    repo, root = billing["repo"], billing["root"]
    repo.write("limits.py", "MAX_RETRIES = 3\n\n\ndef retry():\n    return MAX_RETRIES\n")
    repo.commit("limits")
    server.workspace(action="open", root=root)
    recorded = server.record(root=root, kind="decision", summary="retry cap",
                             invariants=["MAX_RETRIES must stay at 3: the provider bans a fourth retry."],
                             files=["limits.py"])
    rule = next(m["memory_id"] for m in recorded["memories_created"] if m["kind"] == "invariant")
    top = {"session_id": "m1", "cwd": root, "tool_name": "Read",
           "tool_input": {"file_path": str(repo.root / "limits.py"), "offset": 1, "limit": 1}}
    assert rule in hook("pre-tool-use", top)


def test_record_accepts_a_bare_string_and_does_not_echo_bodies(billing):
    result = server.record(root=billing["root"], kind="note", summary="refund audit",
                           warnings="refund() must not be called twice for one invoice; the "
                                    "second call double-credits the customer.",
                           symbols="refund")
    assert result["ok"] and result["memories_created"]
    assert all("body" not in m for m in result["memories_created"])
    got = server.memory(action="get", memory_id=result["memories_created"][0]["memory_id"],
                        root=billing["root"])
    assert "double-credits" in got["memory"]["body"]


def test_why_without_a_chain_lists_the_attached_knowledge(billing):
    why = server.investigate(action="why", symbol="settle", root=billing["root"])
    assert why["ok"] and why["why_it_exists"] is None
    assert billing["rule"] in {m["memory_id"] for m in why["attached_memories"]}


def test_investigate_prints_a_broad_memory_only_where_its_claim_points(billing):
    repo, root = billing["repo"], billing["root"]
    for name in ("ledger", "tax", "audit"):
        repo.write(f"{name}.py", f"def {name}_entry(x):\n    return x\n")
    repo.commit("more modules")
    server.workspace(action="open", root=root)
    recorded = server.record(root=root, kind="decision", summary="billing review across modules",
                             invariants=["tax_entry must round half-even: auditors reject banker drift."],
                             files=["billing.py", "ledger.py", "tax.py", "audit.py"],
                             symbols=["tax_entry", "ledger_entry"])
    rule = next(m["memory_id"] for m in recorded["memories_created"] if m["kind"] == "invariant")
    result = server.investigate(query="ledger_entry tax_entry rounding", intent="modify", root=root)
    printed = {c["symbol"]: {m["memory_id"] for m in c["memory"]} for c in result["capsules"]}
    assert rule in printed.get("tax_entry", set())
    assert "ledger_entry" in printed and rule not in printed["ledger_entry"]


def test_file_anchors_are_indexed_for_the_hook_query(billing):
    """The hook asks 'what is anchored to this file' before every tool call.

    Without an index on anchors(file_path) SQLite drove that query from the
    memories table and probed anchors per row: measured 16-24 ms against 11 ms,
    on every tool call.
    """
    from icn import db, paths
    from icn.hooks import resolve_repo

    repo_id, _ = resolve_repo(billing["root"])
    conn = db.init_repo_store(paths.repo_db_path(repo_id))
    try:
        plan = " ".join(str(r[-1]) for r in conn.execute(
            "EXPLAIN QUERY PLAN SELECT m.memory_id FROM anchors a"
            " JOIN memories m ON m.memory_id = a.memory_id"
            " WHERE a.file_path IN ('billing.py') AND m.status='ACTIVE'"))
    finally:
        conn.close()
    assert "idx_anchor_file" in plan, plan
