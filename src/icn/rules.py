"""Promote settled, codebase-specific rules into this repository's CLAUDE.md and AGENTS.md.

Adapted from ai-memory's rules-promotion design, and built around its one hard
lesson: an agent instruction file is a scarce budget, not a filing cabinet.
Models follow a few hundred instructions reliably, the harness spends a share
of those, and every rule added past that point weakens the ones already there.
So promotion is the exception, and the file changes only on command:

    recommend   read-only. Ranked candidates, each with why it qualified.
    approve     one rule into the managed block. Refused at the cap, naming the
                weakest promoted rule, so eviction is always a person's choice.
    edit        reword a promoted rule; human phrasing usually beats extracted.
    remove      demote. The memory is untouched; only the line goes.
    list        what is promoted, and which of it has gone stale.

Codebase-specific by construction. Candidates come only from this repository's
knowledge, are anchored to real code that still matches, and each line names
the file and symbol it governs, so a rule reads "in src/pay/settle.py,
settle(): ..." rather than a platitude that could sit in any repository.

Eligible means all of: a rule-shaped kind; ACTIVE with every anchor ACTIVE; no
unresolved contradiction; not voted wrong or stale since it was last verified;
imperative ("must", "never", ...); and short enough to be one line. Everything
else stays retrieval knowledge, which investigate() and the hooks deliver on
demand at no standing cost.

The managed block sits between markers. Text outside the markers is never read
into the block and never modified.
"""

from __future__ import annotations

import math
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .db import one, rows, write_tx
from .feedback import withheld_from_delivery
from .relevance import focus

START = "<!-- icn:rules:start -->"
END = "<!-- icn:rules:end -->"
TARGET_FILES = ("CLAUDE.md", "AGENTS.md")
DEFAULT_CAP = 12
MAX_RULE_CHARS = 240
RULE_KINDS = ("invariant", "warning", "contract", "security", "convention")
SEVERITY_WEIGHT = {"critical": 1.0, "high": 0.8, "medium": 0.55, "low": 0.3}
IMPERATIVE = re.compile(
    r"\b(must|never|always|do not|don't|does not|should not|shouldn't|only|required|"
    r"cannot|can't|avoid|ensure|keep)\b", re.IGNORECASE)


class RulesError(RuntimeError):
    """A refused rules command, with what to do instead."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _claim(memory: dict[str, Any]) -> str:
    text = memory.get("claim") or ""
    if not text:
        text = memory.get("body") or ""
        for marker in ("\n\nRecorded while: ", "\n\nWhy: ", "\n\nChanged: ", "\n\nApplies to: "):
            text = text.split(marker, 1)[0]
    return " ".join(text.split())


def _first_sentence(text: str) -> str:
    match = re.match(r"(.+?[.!?])(\s|$)", text)
    return (match.group(1) if match else text).strip()


def _scope(conn: sqlite3.Connection, memory_id: str, claim: str = "") -> dict[str, Any]:
    anchors = rows(conn.execute(
        "SELECT a.status, a.file_path, a.symbol_path, a.symbol_id, a.target_kind FROM anchors a"
        " WHERE a.memory_id = ?", (memory_id,)))
    # Only the code the claim is about, not every file its record touched.
    focused = focus(claim, anchors)
    files, symbols = focused["files"], focused["symbols"]
    symbol_ids = [a["symbol_id"] for a in anchors if a["symbol_id"] and a["symbol_path"] in symbols]
    callers = 0
    if symbol_ids:
        marks = ",".join("?" for _ in symbol_ids)
        found = one(conn.execute(
            f"SELECT COUNT(DISTINCT from_id) AS n FROM code_edges WHERE kind='CALLS'"
            f" AND status='ACTIVE' AND to_id IN ({marks})", tuple(symbol_ids)))
        callers = found["n"] if found else 0
    return {"anchors": anchors, "files": files, "symbols": symbols, "callers": callers,
            "all_active": bool(anchors) and all(a["status"] == "ACTIVE" for a in anchors
                                                if not focused["broad"] or a["file_path"] in files),
            "repo_only": all(a["target_kind"] == "repo" for a in anchors)}


def _rule_text(memory: dict[str, Any], scope: dict[str, Any]) -> str:
    sentence = _first_sentence(_claim(memory))
    where = ""
    if scope["files"]:
        where = f"`{scope['files'][0]}`"
        if len(scope["files"]) > 1:
            where += f" (+{len(scope['files']) - 1} more files)"
        if scope["symbols"]:
            where += " " + ", ".join(f"`{s}`" for s in scope["symbols"][:2])
        where = f"In {where}: "
    return f"{where}{sentence}"


def _blockers(conn: sqlite3.Connection, memory: dict[str, Any], scope: dict[str, Any]) -> list[str]:
    reasons = []
    if memory["kind"] not in RULE_KINDS:
        reasons.append(f"kind {memory['kind']} is not a rule")
    if memory["status"] != "ACTIVE":
        reasons.append(f"status is {memory['status']}")
    if not scope["all_active"]:
        reasons.append("an anchor is not ACTIVE: verify it against the current code first")
    if scope["repo_only"]:
        reasons.append("not anchored to any file or symbol, so it is not specific to this codebase")
    if withheld_from_delivery(memory):
        reasons.append("low confidence or voted unhelpful")
    contradicted = one(conn.execute(
        "SELECT 1 AS x FROM memory_edges WHERE kind='CONTRADICTS' AND status='ACTIVE'"
        " AND (from_id=? OR to_id=?) LIMIT 1", (memory["memory_id"], memory["memory_id"])))
    if contradicted:
        reasons.append("has an unresolved contradiction")
    since = memory.get("last_verified_at") or memory.get("created_at") or ""
    disputed = one(conn.execute(
        "SELECT signal FROM memory_feedback WHERE memory_id=? AND signal IN ('stale','wrong')"
        " AND created_at > ? ORDER BY created_at DESC LIMIT 1", (memory["memory_id"], since)))
    if disputed:
        reasons.append(f"voted {disputed['signal']} since it was last verified")
    claim = _claim(memory)
    if not IMPERATIVE.search(_first_sentence(claim)):
        reasons.append("not imperative: a rule says must / never / always, not what happened")
    if len(_first_sentence(claim)) > MAX_RULE_CHARS:
        reasons.append(f"longer than {MAX_RULE_CHARS} characters: it needs its context, so it "
                       "belongs in retrieval, not in an always-loaded file")
    return reasons


def _score(memory: dict[str, Any], scope: dict[str, Any]) -> tuple[float, list[str]]:
    evidence = memory.get("evidence_count") or 1
    helpful = memory.get("helpful_count") or 0
    unhelpful = memory.get("unhelpful_count") or 0
    reach = len(scope["files"]) + scope["callers"]
    used = (memory.get("access_count") or 0) + 0.2 * (memory.get("surfaced_count") or 0)
    score = (SEVERITY_WEIGHT.get(memory["severity"], 0.5)
             * (1 + 0.6 * math.log1p(evidence - 1))
             * (1 + 0.4 * math.log1p(helpful)) / (1 + 0.5 * unhelpful)
             * (1 + 0.25 * math.log1p(reach))
             * (1 + 0.15 * math.log1p(used)))
    why = [f"{memory['severity']} {memory['kind']}"]
    if evidence > 1:
        why.append(f"asserted in {evidence} separate records")
    if helpful:
        why.append(f"voted helpful {helpful}x")
    if scope["callers"]:
        why.append(f"governs code with {scope['callers']} callers")
    if len(scope["files"]) > 1:
        why.append(f"spans {len(scope['files'])} files")
    if used >= 1:
        why.append("opened or surfaced in past work")
    return round(score, 3), why


def _promoted(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return rows(conn.execute("SELECT * FROM promoted_rules ORDER BY approved_at"))


def recommend(conn: sqlite3.Connection, limit: int = 10, cap: int = DEFAULT_CAP) -> dict[str, Any]:
    promoted = {r["memory_id"] for r in _promoted(conn)}
    marks = ",".join("?" for _ in RULE_KINDS)
    memories = rows(conn.execute(
        f"SELECT * FROM memories WHERE status='ACTIVE' AND kind IN ({marks})", RULE_KINDS))
    candidates, near = [], []
    for memory in memories:
        if memory["memory_id"] in promoted:
            continue
        scope = _scope(conn, memory["memory_id"], _claim(memory))
        blockers = _blockers(conn, memory, scope)
        score, why = _score(memory, scope)
        entry = {"memory_id": memory["memory_id"], "rule": _rule_text(memory, scope),
                 "score": score, "why": why}
        if blockers:
            if len(blockers) == 1 and score >= 0.8:
                near.append({**entry, "blocked_by": blockers[0]})
            continue
        candidates.append(entry)
    candidates.sort(key=lambda c: -c["score"])
    near.sort(key=lambda c: -c["score"])
    return {"ok": True, "promoted": len(promoted), "cap": cap,
            "candidates": candidates[:max(1, limit)], "almost": near[:3],
            "note": ("nothing changes until approve. Promote only what would cause a mistake on "
                     "many tasks if forgotten; everything else is delivered on demand already.")}


def _render_block(rules: list[dict[str, Any]], newline: str) -> str:
    lines = [START,
             "## Rules settled in this codebase",
             "",
             "Promoted from verified ICN knowledge, each anchored to the code it governs. "
             "Managed by ICN: change them with `icn rules` or memory(action='rules_*'), "
             "not by editing between these markers.",
             ""]
    lines += [f"- {r['text']} ({r['memory_id']})" for r in rules]
    lines.append(END)
    return newline.join(lines)


def _write(root: Path, rules: list[dict[str, Any]]) -> list[str]:
    """Rewrite the managed block in each target. Returns the files changed."""
    existing = [name for name in TARGET_FILES if (root / name).exists()]
    targets = existing or (list(TARGET_FILES) if rules else [])
    changed = []
    for name in targets:
        path = root / name
        original = path.read_text(encoding="utf-8") if path.exists() else ""
        newline = "\r\n" if "\r\n" in original else "\n"
        pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
        if rules:
            block = _render_block(rules, newline)
            if pattern.search(original):
                updated = pattern.sub(lambda _: block, original, count=1)
            else:
                separator = "" if not original else (newline if original.endswith(newline) else newline * 2)
                updated = f"{original}{separator}{newline if original else ''}{block}{newline}"
        else:
            updated = pattern.sub("", original).rstrip() + (newline if original.strip() else "")
        if updated == original:
            continue
        if not updated.strip():
            path.unlink(missing_ok=True)
        else:
            temp = path.with_suffix(path.suffix + ".icn-tmp")
            temp.write_bytes(updated.encode("utf-8"))
            os.replace(temp, path)
        changed.append(name)
    return changed


def approve(conn: sqlite3.Connection, root: Path, memory_id: str, text: str = "",
            cap: int = DEFAULT_CAP, actor: str = "agent") -> dict[str, Any]:
    memory = one(conn.execute("SELECT * FROM memories WHERE memory_id=?", (memory_id,)))
    if memory is None:
        raise RulesError(f"unknown memory {memory_id}")
    if one(conn.execute("SELECT 1 AS x FROM promoted_rules WHERE memory_id=?", (memory_id,))):
        raise RulesError(f"{memory_id} is already promoted; use rules_edit to reword it")
    scope = _scope(conn, memory_id, _claim(memory))
    blockers = _blockers(conn, memory, scope)
    if blockers:
        raise RulesError("not eligible: " + "; ".join(blockers))
    promoted = _promoted(conn)
    if len(promoted) >= cap:
        weakest = min(promoted, key=lambda r: r["score"] or 0)
        raise RulesError(
            f"the managed block is at its cap ({len(promoted)}/{cap}). Approving this means "
            f"dropping one first: the weakest is {weakest['memory_id']} ({weakest['text'][:90]}). "
            "Remove it with rules_remove, or raise the cap deliberately. Nothing was changed.")
    score, _ = _score(memory, scope)
    rule = " ".join((text or _rule_text(memory, scope)).split())
    if len(rule) > MAX_RULE_CHARS + 80:
        raise RulesError(f"rule text is {len(rule)} characters; keep it to one line")
    with write_tx(conn):
        conn.execute("INSERT INTO promoted_rules (memory_id, text, score, approved_by, approved_at)"
                     " VALUES (?,?,?,?,?)", (memory_id, rule, score, actor, now()))
    changed = _write(root, _promoted(conn))
    return {"ok": True, "memory_id": memory_id, "rule": rule, "files_changed": changed,
            "promoted": len(promoted) + 1, "cap": cap}


def edit(conn: sqlite3.Connection, root: Path, memory_id: str, text: str) -> dict[str, Any]:
    rule = " ".join((text or "").split())
    if not rule:
        raise RulesError("rules_edit needs body: the new wording of the rule")
    with write_tx(conn):
        updated = conn.execute("UPDATE promoted_rules SET text=? WHERE memory_id=?", (rule, memory_id))
    if updated.rowcount != 1:
        raise RulesError(f"{memory_id} is not a promoted rule (see rules_list)")
    return {"ok": True, "memory_id": memory_id, "rule": rule,
            "files_changed": _write(root, _promoted(conn))}


def remove(conn: sqlite3.Connection, root: Path, memory_id: str) -> dict[str, Any]:
    with write_tx(conn):
        deleted = conn.execute("DELETE FROM promoted_rules WHERE memory_id=?", (memory_id,))
    if deleted.rowcount != 1:
        raise RulesError(f"{memory_id} is not a promoted rule (see rules_list)")
    return {"ok": True, "memory_id": memory_id, "files_changed": _write(root, _promoted(conn)),
            "note": "demoted to retrieval knowledge; the memory itself is unchanged"}


def listing(conn: sqlite3.Connection, root: Path, cap: int = DEFAULT_CAP) -> dict[str, Any]:
    out = []
    for rule in _promoted(conn):
        memory = one(conn.execute("SELECT * FROM memories WHERE memory_id=?", (rule["memory_id"],)))
        if memory is None:
            problems = ["the memory no longer exists"]
        else:
            problems = [b for b in _blockers(conn, memory, _scope(conn, rule["memory_id"], _claim(memory)))
                        if not b.startswith("longer than") and not b.startswith("not imperative")]
        out.append({"memory_id": rule["memory_id"], "rule": rule["text"],
                    "approved_at": rule["approved_at"], "stale": bool(problems),
                    "problems": problems})
    in_files = {name: START in (root / name).read_text(encoding="utf-8")
                for name in TARGET_FILES if (root / name).exists()}
    return {"ok": True, "cap": cap, "rules": out, "stale": sum(1 for r in out if r["stale"]),
            "block_present_in": in_files,
            "next": ("a stale rule may now be wrong for this code: verify its memory, or "
                     "rules_remove it" if any(r["stale"] for r in out) else None)}
