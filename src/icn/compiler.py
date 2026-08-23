"""The memory compiler: one agent event becomes many durable facts.

PLAN.md "the agent should only make ONE write" and PLAN 2 section 8.

The agent supplies semantics in plain language and plain names. The server does
entity resolution, anchoring, edge derivation and contradiction detection. The
split matters: the agent is the only thing that knows *why*, and the server is
the only thing that knows the whole graph, so neither should do the other's job.

No LLM is required. Every step here is deterministic, which is what lets the
server run with zero configuration and no API key.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import anchors as anchor_mod
from . import catalog as catalog_mod
from . import causal
from . import crossrepo
from . import ids
from .db import jdump, one, rows, write_tx

# payload field -> (memory kind, default severity)
FIELD_KINDS = {
    "invariants": ("invariant", "high"),
    "warnings": ("warning", "high"),
    "failed_attempts": ("failed_attempt", "medium"),
    "decisions": ("decision", "medium"),
    "rationale_notes": ("rationale", "low"),
    "contracts": ("contract", "high"),
    "migrations": ("migration", "medium"),
    "performance": ("performance", "medium"),
    "security": ("security", "critical"),
    "conventions": ("convention", "low"),
    "bugs": ("bug_history", "medium"),
    "tests": ("test_evidence", "low"),
}

# A fix and a failure are not the same fact. Mapping both to bug_history made
# them indistinguishable, so "why does this exist" reported the remedy as a
# risk the code protects against - the exact inversion the feature exists to
# avoid. `bug_fix` is the remedy; `incident` and the `bugs=[...]` field are
# the failure.
EVENT_KIND_TO_MEMORY = {
    "bug_fix": ("fix_history", "medium"),
    "decision": ("decision", "medium"),
    "refactor": ("rationale", "low"),
    "investigation": ("rationale", "low"),
    "checkpoint": ("rationale", "low"),
    "incident": ("bug_history", "high"),
}

NEGATION = re.compile(r"\b(never|not|no longer|bypass|except|must not|cannot|don't|do not)\b", re.I)
PATH_HINT = re.compile(r"[\\/]|\.\w{1,5}$")
# Words that describe a symbol without being part of its name.
NOISE_WORDS = re.compile(
    r"\b(the|a|an|our|this|that|its|class|function|method|module|struct|interface|"
    r"type|handler|object|instance)\b", re.I
)


def now() -> str:
    # Milliseconds, not seconds. Ordering questions ("did this caller
    # appear after that memory was verified?") are decided by comparing
    # these, and second precision made same-second events compare equal,
    # so a genuinely late caller went unreported.
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# ------------------------------------------------------------ entity resolution

def resolve_reference(conn: sqlite3.Connection, reference: str) -> dict[str, Any] | None:
    """Turn a name an agent typed into the canonical entity it meant.

    Handles the forms PLAN.md calls out: `RefreshCoordinator`, `the refresh
    coordinator`, `src/auth/RefreshCoordinator.ts`, `RefreshCoordinator.acquire`.
    Returns None rather than guessing when nothing matches well enough - an
    unresolved name becomes a weak reference the user can repair, never a
    duplicate concept and never a silent drop.
    """
    text = (reference or "").strip().strip("`'\"")
    if not text:
        return None

    if PATH_HINT.search(text):
        normalized = text.replace("\\", "/").lstrip("./")
        row = one(conn.execute("SELECT * FROM files WHERE path = ?", (normalized,)))
        if row is None:
            row = one(conn.execute("SELECT * FROM files WHERE path LIKE ? ORDER BY status LIMIT 1",
                                   (f"%{normalized}",)))
        if row is not None:
            return {"kind": "file", "row": row, "confidence": 1.0 if row["status"] == "ACTIVE" else 0.6}

    row = one(conn.execute(
        "SELECT * FROM symbols WHERE symbol_path = ? ORDER BY (status='ACTIVE') DESC LIMIT 1", (text,)
    ))
    if row is not None:
        return {"kind": "symbol", "row": row, "confidence": 1.0}

    matches = rows(conn.execute(
        "SELECT * FROM symbols WHERE name = ? AND status='ACTIVE' LIMIT 5", (text,)
    ))
    if len(matches) == 1:
        return {"kind": "symbol", "row": matches[0], "confidence": 0.95}
    if len(matches) > 1:
        # Ambiguous. Prefer a container (class over its method) and say so.
        containers = [m for m in matches if m["kind"] in ("class", "struct", "interface", "module")]
        chosen = containers[0] if containers else matches[0]
        return {"kind": "symbol", "row": chosen, "confidence": 0.6, "ambiguous": len(matches)}

    loose = rows(conn.execute(
        "SELECT * FROM symbols WHERE status='ACTIVE' AND lower(name) = lower(?) LIMIT 3", (text,)
    ))
    if loose:
        return {"kind": "symbol", "row": loose[0], "confidence": 0.55}

    # "the refresh coordinator" -> refreshcoordinator. Agents describe symbols
    # in prose as often as they name them, and PLAN.md calls this form out
    # explicitly, so strip the articles and type nouns before squashing.
    squashed = re.sub(r"[^a-z0-9]", "", NOISE_WORDS.sub(" ", text.lower()))
    if len(squashed) >= 5:
        for candidate in rows(conn.execute(
            "SELECT * FROM symbols WHERE status='ACTIVE' AND length(name) >= 5 LIMIT 4000"
        )):
            if re.sub(r"[^a-z0-9]", "", candidate["name"].lower()) == squashed:
                return {"kind": "symbol", "row": candidate, "confidence": 0.5, "fuzzy": True}
    return None


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    return [str(value)]


# ------------------------------------------------------------------- recording

def record_event(conn: sqlite3.Connection, catalog: sqlite3.Connection, repo_id: str,
                 root: Path, commit: str | None, payload: dict[str, Any]) -> dict[str, Any]:
    """Compile one event into the graph. Returns the delta that was written."""
    from .indexer import normalize_commit
    commit = normalize_commit(commit)
    event_id = ids.new_id(ids.EVENT)
    kind = str(payload.get("kind") or "note")
    summary = str(payload.get("summary") or "").strip()

    references = _as_list(payload.get("files")) + _as_list(payload.get("symbols"))
    resolved: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for reference in references:
        match = resolve_reference(conn, reference)
        if match is None:
            unresolved.append(reference)
        else:
            resolved.append({"reference": reference, **match})

    created_memories: list[dict[str, Any]] = []
    created_edges = 0
    contradictions: list[dict[str, Any]] = []

    # Mirror every resolved entity into the catalog up front. This has to
    # happen per event, not per memory: an event can legitimately name code
    # without producing a memory (kind="note" with no invariants), and those
    # entities must still become resolvable stubs for cross-repo lookups.
    for match in resolved:
        row = match["row"]
        if match["kind"] == "symbol":
            catalog_mod.upsert_entity_ref(
                catalog, row["symbol_id"], repo_id, "symbol",
                {"symbol_path": row["symbol_path"], "path": row["last_known_path"],
                 "kind": row["kind"], "signature": row["signature"], "commit": commit},
            )
        else:
            catalog_mod.upsert_entity_ref(
                catalog, row["file_id"], repo_id, "file",
                {"path": row["path"], "lang": row["lang"], "commit": commit},
            )

    with write_tx(conn):
        conn.execute(
            "INSERT INTO events (event_id, kind, payload, summary, agent, client, commit_sha,"
            " branch, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (event_id, kind, jdump(payload), summary, payload.get("agent"),
             payload.get("client"), commit, payload.get("branch"), now()),
        )

        # Every memory carries the context of the event that produced it, not
        # just its own one-line claim. Measured before this: median body was
        # 150 characters and 35 of 48 were under 200 - a headline, not
        # knowledge. An invariant reading "settle must be idempotent" tells a
        # future agent nothing about the incident that made it one.
        context = _event_context(payload, summary, resolved)

        entries: list[tuple[str, str, str]] = []   # (memory kind, severity, body)
        for field, (memory_kind, severity) in FIELD_KINDS.items():
            for claim in _as_list(payload.get(field)):
                entries.append((memory_kind, severity, _compose(claim, context)))

        # The summary itself is a memory when the event describes a change.
        if summary and kind in EVENT_KIND_TO_MEMORY:
            memory_kind, severity = EVENT_KIND_TO_MEMORY[kind]
            reasoning = str(payload.get("reasoning") or "").strip()
            body = f"{summary}\n\n{reasoning}".strip() if reasoning else summary
            entries.insert(0, (memory_kind, severity,
                               _compose(body, context, is_summary=True)))

        for memory_kind, severity, body in entries:
            memory_id = ids.new_id(ids.MEMORY)
            title = body.strip().split("\n", 1)[0][:200]
            conn.execute(
                "INSERT INTO memories (memory_id, kind, title, body, severity, authority, confidence,"
                " status, scope, source_event, version, created_at, updated_at, created_commit,"
                " last_verified_commit, last_verified_at, valid_from)"
                " VALUES (?,?,?,?,?,?,?,'ACTIVE','repo',?,1,?,?,?,?,?,?)",
                (memory_id, memory_kind, title, body, severity,
                 payload.get("authority", "agent"), float(payload.get("confidence", 0.8)),
                 event_id, now(), now(), commit, commit, now(), now()),
            )
            _index_memory(conn, memory_id, title, body, memory_kind)
            conn.execute(
                "INSERT INTO memory_edges (edge_id, from_id, to_id, kind, edge_class, status,"
                " confidence, source, created_at) VALUES (?,?,?,'DERIVED_FROM','deterministic',"
                "'ACTIVE',1.0,'record',?)",
                (ids.new_id(ids.EDGE), memory_id, event_id, now()),
            )
            created_edges += 1

            anchor_ids: list[str] = []
            for match in resolved:
                row = match["row"]
                if match["kind"] == "symbol":
                    anchor_ids.append(anchor_mod.create_anchor(conn, memory_id, row, None, commit))
                    created_edges += _link(conn, memory_id, row["symbol_id"], "APPLIES_TO",
                                           "asserted", match["confidence"])
                    created_edges += _derive_code_edges(conn, memory_id, row, memory_kind)
                else:
                    anchor_ids.append(anchor_mod.create_anchor(conn, memory_id, None, row, commit))
                    created_edges += _link(conn, memory_id, row["file_id"], "APPLIES_TO",
                                           "asserted", match["confidence"])

            if not anchor_ids:
                # Nothing to attach to. The memory is still worth keeping, but
                # it is honestly repo-scoped rather than pretending to a span.
                anchor_ids.append(anchor_mod.create_anchor(conn, memory_id, None, None, commit,
                                                           target_kind="repo"))

            found = _detect_contradictions(conn, memory_id, memory_kind, body, resolved)
            contradictions.extend(found)
            created_edges += len(found)

            catalog_mod.register_memory(catalog, memory_id, repo_id, memory_kind, severity,
                                        "ACTIVE", title)
            catalog_mod.upsert_entity_ref(catalog, memory_id, repo_id, "memory",
                                          {"title": title, "kind": memory_kind, "commit": commit})
            created_memories.append({"memory_id": memory_id, "kind": memory_kind,
                                     "severity": severity, "title": title,
                                     "body": body, "anchors": len(anchor_ids)})

    _link_test_coverage(conn, created_memories)
    cross_links = _link_contracts(catalog, payload, resolved, created_memories, commit)
    causal_links = _link_causal(conn, payload, created_memories)

    # Say which memory represents this event. A caller passing caused_by next
    # would otherwise have to guess from an unordered list, and a causal edge
    # naming the wrong one silently builds a false chain.
    primary_id = _primary_memory(created_memories)["memory_id"] if created_memories else None
    for entry in created_memories:
        if entry["memory_id"] == primary_id:
            entry["primary"] = True

    return {
        "ok": True,
        "event_id": event_id,
        "memories_created": created_memories,
        "primary_memory": primary_id,
        "edges_created": created_edges,
        "cross_repo_links": cross_links,
        "causal_links": causal_links,
        "entities_resolved": [
            {"reference": m["reference"],
             "resolved_to": m["row"].get("symbol_path") or m["row"].get("path"),
             "kind": m["kind"], "confidence": m["confidence"]}
            for m in resolved
        ],
        "unresolved_references": unresolved,
        "contradictions": contradictions,
        "trust": {
            "authority": payload.get("authority", "agent"),
            "verified": payload.get("authority") == "human",
            "note": ("Human-supplied assertion." if payload.get("authority") == "human" else
                     "Agent-supplied claim; anchoring verifies code location, not semantic truth."),
        },
        "quality": _write_quality(payload, created_memories, resolved, unresolved),
    }



def _link_contracts(catalog: sqlite3.Connection, payload: dict[str, Any],
                    resolved: list[dict[str, Any]], memories: list[dict[str, Any]],
                    commit: str | None) -> list[dict[str, Any]]:
    """Turn declared `contracts_with` entries into cross-repo edges.

    Shape:  {"repo": "acme/billing", "entity": "charge_customer",
             "kind": "CONSUMES_CONTRACT"}

    Anchored to the first resolved symbol if there is one, otherwise to the
    memory itself - a contract that names no local code is still worth
    recording, just at repo scope.
    """
    declared = payload.get("contracts_with")
    if not declared:
        return []
    if isinstance(declared, dict):
        declared = [declared]

    anchor_entity = None
    for match in resolved:
        if match["kind"] == "symbol":
            anchor_entity = match["row"]["symbol_id"]
            break
    if anchor_entity is None and resolved:
        anchor_entity = resolved[0]["row"].get("file_id")
    if anchor_entity is None and memories:
        anchor_entity = memories[0]["memory_id"]
    if anchor_entity is None:
        return []

    out: list[dict[str, Any]] = []
    for item in declared:
        if not isinstance(item, dict):
            continue
        result = crossrepo.link(
            catalog, anchor_entity,
            str(item.get("repo") or ""), str(item.get("entity") or ""),
            str(item.get("kind") or "RELATED_TO"),
            confidence=float(item.get("confidence", 0.7)),
            evidence={"declared_by": "record", "summary": payload.get("summary")},
            commit=commit,
        )
        out.append(result)
    return out






def guard_memory(conn: sqlite3.Connection, memory_id: str, test_memory_id: str) -> dict[str, Any]:
    """Record that an existing test covers an existing rule.

    `record(tests=[...])` only guards rules created in the same call, so a rule
    written before its test had no way to be marked covered - the
    untested-invariant diagnostic kept reporting it, which trains people to
    ignore the finding. This closes that loop after the fact.
    """
    rule = one(conn.execute("SELECT memory_id, kind FROM memories WHERE memory_id=?", (memory_id,)))
    if rule is None:
        return {"ok": False, "error": "unknown memory " + repr(memory_id)}
    test = one(conn.execute("SELECT memory_id, kind, title FROM memories WHERE memory_id=?",
                            (test_memory_id,)))
    if test is None:
        return {"ok": False, "error": "unknown memory " + repr(test_memory_id)}
    if test["kind"] != "test_evidence":
        return {"ok": False, "error": repr(test_memory_id) + " is a " + test["kind"]
                                      + ", not test evidence"}
    if memory_id == test_memory_id:
        return {"ok": False, "error": "a memory cannot guard itself"}

    with write_tx(conn):
        _link(conn, memory_id, test_memory_id, "GUARDED_BY", "asserted", 0.9,
              {"declared_by": "memory(action='guard')"})
    return {"ok": True, "memory_id": memory_id, "guarded_by": test_memory_id,
            "test": test["title"]}

# How much surrounding context a memory carries. Enough that it reads as a
# self-contained note, capped so a capsule does not become a file dump.
CONTEXT_CHARS = 900


def _event_context(payload: dict[str, Any], summary: str,
                   resolved: list[dict[str, Any]]) -> dict[str, Any]:
    """The situation a memory was recorded in.

    A claim without its circumstances is not knowledge. "Redis mutex could
    deadlock" is a sentence; "we hit concurrent token invalidation, tried a
    Redis mutex, and it deadlocks under partition" is something an agent can
    act on a year later.
    """
    where: list[str] = []
    for match in resolved:
        row = match["row"]
        label = row.get("symbol_path") or row.get("path")
        if label and label not in where:
            where.append(label)

    return {
        "occasion": summary,
        "reasoning": str(payload.get("reasoning") or "").strip(),
        "where": where[:6],
        "changes": [c for c in _as_list(payload.get("changes")) if c][:4],
    }


def _compose(claim: str, context: dict[str, Any], is_summary: bool = False) -> str:
    """Attach the event's context to one claim, without repeating it back."""
    claim = (claim or "").strip()
    if not claim:
        return claim

    parts = [claim]
    if not is_summary:
        occasion = context.get("occasion") or ""
        # Skip the occasion when the claim already states it. Repeating the
        # summary under itself is noise, and it happens often because agents
        # phrase a warning and its summary similarly.
        if occasion and occasion.lower() not in claim.lower():
            parts.append("Recorded while: " + occasion)

    reasoning = context.get("reasoning") or ""
    if reasoning and reasoning.lower() not in claim.lower():
        parts.append("Why: " + reasoning)

    if context.get("changes"):
        parts.append("Changed: " + "; ".join(context["changes"]))
    if context.get("where"):
        parts.append("Applies to: " + ", ".join(context["where"]))

    body = "\n\n".join(parts)
    return body if len(body) <= CONTEXT_CHARS else body[:CONTEXT_CHARS].rstrip() + "..."



# A memory shorter than this is a label, not knowledge. Set from measurement:
# before context composition the median body here was 150 characters, and
# those entries read as headlines nobody could act on.
THIN_CLAIM_CHARS = 120


def _write_quality(payload: dict[str, Any], memories: list[dict[str, Any]],
                   resolved: list[dict[str, Any]],
                   unresolved: list[str]) -> dict[str, Any]:
    """Tell the agent, at write time, whether what it stored is usable.

    The compiler cannot write the knowledge for the caller - only the agent
    knows why it did what it did. What it can do is notice that an entry will
    be useless to whoever reads it next, and say so while the context is still
    in the caller's head. Afterwards is too late; nobody comes back to enrich
    a memory they already wrote.

    This never blocks a write. A thin memory still beats no memory.
    """
    notes: list[str] = []

    thin = [m["title"] for m in memories
            if len((m.get("body") or m.get("title") or "")) < THIN_CLAIM_CHARS]
    if thin:
        notes.append(
            f"{len(thin)} of {len(memories)} entries are very short and may not be "
            f"understandable on their own later. Add `reasoning` to give them the "
            f"situation, or restate the claim with the failure it prevents. "
            f"Shortest: {thin[0][:60]!r}")

    if not str(payload.get("reasoning") or "").strip():
        notes.append(
            "No `reasoning` given, so every memory from this call stands alone "
            "without the story behind it. One or two sentences here attaches "
            "context to all of them at once.")

    if not resolved:
        notes.append(
            "Nothing was anchored to code: this knowledge is repo-scoped and "
            "will not surface when someone works on the relevant file. Pass "
            "`symbols` or `files`.")

    if unresolved:
        notes.append(
            f"Could not resolve {', '.join(repr(u) for u in unresolved[:3])}. "
            f"Those memories are stored but not attached to anything.")

    if not payload.get("failed_attempts") and payload.get("kind") in ("bug_fix", "incident"):
        notes.append(
            "No `failed_attempts` recorded. If anything was tried first and "
            "rejected, that is the single most expensive thing for a future "
            "agent to rediscover.")

    return {
        "sufficient": not notes,
        "notes": notes,
        "median_body_chars": _median([len(m.get("body") or "") for m in memories]),
    }


def _median(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[len(ordered) // 2]

def _link_test_coverage(conn: sqlite3.Connection,
                        memories: list[dict[str, Any]]) -> int:
    """Mark rules recorded alongside a test as guarded by it.

    `record(invariants=[...], tests=[...])` means "this rule is covered by that
    test". Without an explicit GUARDED_BY edge the test_evidence memory just
    sits there, and the untested-invariant detector keeps reporting a rule the
    agent already covered - which trains people to ignore the finding.
    """
    tests = [m for m in memories if m["kind"] == "test_evidence"]
    rules = [m for m in memories
             if m["kind"] in ("invariant", "warning", "contract", "security", "decision")]
    if not tests or not rules:
        return 0

    created = 0
    with write_tx(conn):
        for rule in rules:
            for test in tests:
                created += _link(conn, rule["memory_id"], test["memory_id"],
                                 "GUARDED_BY", "asserted", 0.9,
                                 {"declared_by": "record(tests=...)"})
    return created


# Which memory an event is "about", when a causal edge names the event rather
# than a specific fact. Ordered by how much each kind carries the story.
_PRIMARY_ORDER = ("bug_history", "fix_history", "decision", "invariant", "warning",
                  "contract", "security", "failed_attempt", "migration",
                  "performance", "rationale", "convention", "test_evidence")


def _primary_memory(memories: list[dict[str, Any]]) -> dict[str, Any]:
    """The one memory that best represents this event."""
    ranked = sorted(
        memories,
        key=lambda m: _PRIMARY_ORDER.index(m["kind"]) if m["kind"] in _PRIMARY_ORDER
        else len(_PRIMARY_ORDER))
    return ranked[0]

def _link_causal(conn: sqlite3.Connection, payload: dict[str, Any],
                 memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach this event's memories to the story that produced them.

    Shape:  caused_by=["mem_abc123", ...]  or
            caused_by=[{"memory": "mem_abc123", "kind": "CAUSED"}]

    Causality is only ever asserted, never inferred. "B was recorded after A"
    is not "A caused B", and a fabricated causal chain is worse than none
    because it reads as authoritative (see causal.py).
    """
    declared = payload.get("caused_by")
    if not declared or not memories:
        return []
    if isinstance(declared, (str, dict)):
        declared = [declared]

    out: list[dict[str, Any]] = []
    with write_tx(conn):
        for item in declared:
            if isinstance(item, str):
                prior, kind = item, "CAUSED"
            elif isinstance(item, dict):
                prior, kind = str(item.get("memory") or ""), str(item.get("kind") or "CAUSED")
            else:
                continue
            if not prior:
                continue
            # Link to ONE memory - the event's primary claim - not to all of
            # them. Fanning out to every memory the event produced invents
            # causality that was never asserted: recording a bug fix with
            # caused_by=[X] would claim X caused the invariant, the test
            # evidence and the decision alike. Observed live producing a chain
            # reading "warm-open bug CAUSED MCP annotation invariant", which is
            # simply false, and a wrong causal chain is worse than none because
            # it reads as authoritative.
            target = _primary_memory(memories)
            result = causal.link_causal(conn, prior, target["memory_id"], kind)
            out.append(result)

            # The event's other memories hang off its primary one, so the
            # story stays connected without inventing causality. An invariant
            # recorded alongside a fix was ESTABLISHED by that fix; it did not
            # independently follow from whatever caused the fix.
            for memory in memories:
                if memory["memory_id"] == target["memory_id"]:
                    continue
                causal.link_causal(conn, target["memory_id"], memory["memory_id"],
                                   "ESTABLISHED")
    return out

def title_for(body: str) -> str:
    """First line of a memory body, capped. The one place this rule lives."""
    return (body or "").strip().split("\n", 1)[0][:200]

def _index_memory(conn: sqlite3.Connection, memory_id: str, title: str, body: str,
                  kind: str) -> None:
    """Keep fts_memories in step with the memories table.

    Without this, half of investigate()'s retrieval is dead: a memory is only
    findable through the code it is anchored to, so asking about a decision in
    the words the decision itself uses returns nothing. Found by dogfooding -
    "can I store the full token text?" failed to surface the memory that
    says no.
    """
    conn.execute("DELETE FROM fts_memories WHERE memory_id = ?", (memory_id,))
    conn.execute(
        "INSERT INTO fts_memories (memory_id, title, body, kind) VALUES (?,?,?,?)",
        (memory_id, title or "", (body or "")[:4000], kind or ""),
    )

def _link(conn: sqlite3.Connection, from_id: str, to_id: str, kind: str,
          edge_class: str, confidence: float, evidence: Any = None) -> int:
    conn.execute(
        "INSERT INTO memory_edges (edge_id, from_id, to_id, kind, edge_class, status, confidence,"
        " source, evidence, created_at) VALUES (?,?,?,?,?,'ACTIVE',?,'compiler',?,?)"
        " ON CONFLICT(from_id, to_id, kind) DO UPDATE SET confidence=excluded.confidence",
        (ids.new_id(ids.EDGE), from_id, to_id, kind, edge_class, confidence, jdump(evidence), now()),
    )
    return 1


def _derive_code_edges(conn: sqlite3.Connection, memory_id: str, symbol: dict[str, Any],
                       memory_kind: str) -> int:
    """Edges the agent did not mention, read off the deterministic code graph.

    This is where one sentence becomes a dozen useful relations: callers that
    the memory now implicates, and tests that guard it.
    """
    count = 0
    callers = rows(conn.execute(
        "SELECT s.* FROM code_edges e JOIN symbols s ON s.symbol_id = e.from_id"
        " WHERE e.to_id = ? AND e.kind='CALLS' AND e.status='ACTIVE' AND s.status='ACTIVE' LIMIT 12",
        (symbol["symbol_id"],),
    ))
    for caller in callers:
        is_test = _looks_like_test(caller)
        count += _link(
            conn, memory_id, caller["symbol_id"],
            "GUARDED_BY" if is_test else "IMPACTS",
            "inferred", 0.6 if not is_test else 0.75,
            {"via": "CALLS", "target": symbol["symbol_path"]},
        )
    return count


def _looks_like_test(symbol: dict[str, Any]) -> bool:
    name = (symbol.get("name") or "").lower()
    path = (symbol.get("last_known_path") or "").lower()
    return (name.startswith("test") or name.endswith("test") or "spec" in name
            or "/test" in path or "test_" in path or ".spec." in path or ".test." in path)


def _detect_contradictions(conn: sqlite3.Connection, memory_id: str, kind: str, body: str,
                           resolved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flag a possible knowledge conflict rather than overwriting anything.

    Deliberately weak and deliberately `inferred`: this raises the question for
    a human or agent to settle, it never decides. Silently overwriting a prior
    invariant is exactly the failure the trust classes exist to prevent.
    """
    if kind not in ("invariant", "warning", "contract", "decision"):
        return []
    symbol_ids = [m["row"]["symbol_id"] for m in resolved if m["kind"] == "symbol"]
    if not symbol_ids:
        return []

    placeholders = ",".join("?" for _ in symbol_ids)
    existing = rows(conn.execute(
        f"SELECT DISTINCT m.* FROM memories m JOIN memory_edges e ON e.from_id = m.memory_id"
        f" WHERE e.to_id IN ({placeholders}) AND e.kind='APPLIES_TO' AND m.status='ACTIVE'"
        f" AND m.memory_id != ? AND m.kind IN ('invariant','warning','contract','decision')",
        (*symbol_ids, memory_id),
    ))

    new_tokens = set(re.findall(r"[a-z]{4,}", body.lower()))
    new_negated = bool(NEGATION.search(body))
    found: list[dict[str, Any]] = []
    for other in existing:
        other_tokens = set(re.findall(r"[a-z]{4,}", (other["body"] or "").lower()))
        if not other_tokens or not new_tokens:
            continue
        overlap = len(new_tokens & other_tokens) / len(new_tokens | other_tokens)
        if overlap < 0.28:
            continue
        if new_negated == bool(NEGATION.search(other["body"] or "")):
            continue
        _link(conn, memory_id, other["memory_id"], "CONTRADICTS", "inferred", round(overlap, 3),
              {"reason": "similar subject, opposite polarity"})
        shared = sorted(new_tokens & other_tokens)
        found.append({"memory_id": other["memory_id"], "title": other["title"],
                      "overlap": round(overlap, 3),
                      "new_claim": body.split("\n", 1)[0],
                      "existing_claim": (other["body"] or "").split("\n", 1)[0],
                      "shared_terms": shared[:12],
                      "new_polarity": "negative" if new_negated else "positive",
                      "existing_polarity": ("negative" if NEGATION.search(other["body"] or "")
                                            else "positive"),
                      "existing_authority": other["authority"],
                      "note": "flagged for review, nothing was overwritten"})
    return found


# ---------------------------------------------------------------- memory edits

def correct(conn: sqlite3.Connection, catalog: sqlite3.Connection, repo_id: str, memory_id: str,
            body: str | None = None, kind: str | None = None, severity: str | None = None,
            reason: str = "", actor: str = "agent") -> dict[str, Any]:
    """Edit a memory by versioning it. The previous text is never lost.

    A human-authored memory cannot be rewritten by an agent (PLAN.md "separate
    FACTS from MEMORIES"): the agent must supersede it instead, leaving both
    versions and the disagreement visible.
    """
    memory = one(conn.execute("SELECT * FROM memories WHERE memory_id = ?", (memory_id,)))
    if memory is None:
        return {"ok": False, "error": f"unknown memory {memory_id}"}
    if memory["authority"] == "human" and actor != "human":
        return {"ok": False, "error": "human-authored memory cannot be rewritten by an agent; "
                                      "use action='supersede' to record a disagreement",
                "memory_id": memory_id}

    with write_tx(conn):
        conn.execute(
            "INSERT INTO memory_versions (memory_id, version, kind, body, severity, status,"
            " changed_at, changed_by, reason) VALUES (?,?,?,?,?,?,?,?,?)",
            (memory_id, memory["version"], memory["kind"], memory["body"], memory["severity"],
             memory["status"], now(), actor, reason),
        )
        conn.execute(
            "UPDATE memories SET body=?, kind=?, severity=?, version=version+1, updated_at=?"
            " WHERE memory_id=?",
            (body if body is not None else memory["body"], kind or memory["kind"],
             severity or memory["severity"], now(), memory_id),
        )
        _index_memory(conn, memory_id, memory["title"] or "",
                      body if body is not None else memory["body"], kind or memory["kind"])
        conn.execute(
            "INSERT INTO corrections (correction_id, memory_id, action, reason, actor, created_at)"
            " VALUES (?,?,'correct',?,?,?)",
            (ids.new_id("cor"), memory_id, reason, actor, now()),
        )
        updated = one(conn.execute("SELECT * FROM memories WHERE memory_id=?", (memory_id,)))
    catalog_mod.register_memory(catalog, memory_id, repo_id, updated["kind"], updated["severity"],
                                updated["status"], updated["title"] or "")
    return {"ok": True, "memory_id": memory_id, "version": updated["version"]}


def supersede(conn: sqlite3.Connection, catalog: sqlite3.Connection, repo_id: str,
              memory_id: str, body: str, reason: str = "", actor: str = "agent",
              commit: str | None = None) -> dict[str, Any]:
    """Replace a memory with a new one, keeping both and the link between them."""
    old = one(conn.execute("SELECT * FROM memories WHERE memory_id = ?", (memory_id,)))
    if old is None:
        return {"ok": False, "error": f"unknown memory {memory_id}"}

    new_id = ids.new_id(ids.MEMORY)
    with write_tx(conn):
        conn.execute(
            "INSERT INTO memories (memory_id, kind, title, body, severity, authority, confidence,"
            " status, scope, source_event, version, created_at, updated_at, created_commit,"
            " last_verified_commit, last_verified_at, valid_from)"
            " VALUES (?,?,?,?,?,?,?,'ACTIVE',?,?,1,?,?,?,?,?,?)",
            (new_id, old["kind"], body.strip().split("\n", 1)[0][:200], body, old["severity"],
             actor if actor == "human" else "agent", 0.85, old["scope"], old["source_event"],
             now(), now(), commit, commit, now(), now()),
        )
        _index_memory(conn, new_id, title_for(body), body, old["kind"])
        conn.execute("UPDATE memories SET status='SUPERSEDED', valid_until=?, updated_at=?"
                     " WHERE memory_id=?", (now(), now(), memory_id))
        # A superseded memory leaves the search index; it stays fully readable
        # through get/list and the SUPERSEDES edge.
        conn.execute("DELETE FROM fts_memories WHERE memory_id = ?", (memory_id,))
        conn.execute("UPDATE anchors SET status=? WHERE memory_id=?",
                     (anchor_mod.SUPERSEDED, memory_id))
        _link(conn, new_id, memory_id, "SUPERSEDES", "asserted", 1.0, {"reason": reason})
        conn.execute(
            "INSERT INTO corrections (correction_id, memory_id, action, reason, actor, detail,"
            " created_at) VALUES (?,?,'supersede',?,?,?,?)",
            (ids.new_id("cor"), memory_id, reason, actor, jdump({"replacement": new_id}), now()),
        )
        # The replacement inherits the old anchors' targets.
        for anchor in rows(conn.execute("SELECT * FROM anchors WHERE memory_id=?", (memory_id,))):
            if anchor["symbol_id"]:
                symbol = one(conn.execute("SELECT * FROM symbols WHERE symbol_id=?",
                                          (anchor["symbol_id"],)))
                if symbol:
                    anchor_mod.create_anchor(conn, new_id, symbol, None, commit)
                    _link(conn, new_id, symbol["symbol_id"], "APPLIES_TO", "asserted", 0.9)

    catalog_mod.register_memory(catalog, memory_id, repo_id, old["kind"], old["severity"],
                                "SUPERSEDED", old["title"] or "")
    catalog_mod.register_memory(catalog, new_id, repo_id, old["kind"], old["severity"], "ACTIVE",
                                body.strip().split("\n", 1)[0][:200])
    return {"ok": True, "superseded": memory_id, "replacement": new_id}


def resolve_memory(conn: sqlite3.Connection, catalog: sqlite3.Connection, repo_id: str,
                   memory_id: str, reason: str = "", actor: str = "agent") -> dict[str, Any]:
    """Mark a warning or bug memory as no longer live (the fix landed)."""
    memory = one(conn.execute("SELECT * FROM memories WHERE memory_id = ?", (memory_id,)))
    if memory is None:
        return {"ok": False, "error": f"unknown memory {memory_id}"}
    with write_tx(conn):
        conn.execute("UPDATE memories SET status='RESOLVED', updated_at=?, valid_until=?"
                     " WHERE memory_id=?", (now(), now(), memory_id))
        conn.execute("UPDATE anchors SET status=? WHERE memory_id=?",
                     (anchor_mod.RESOLVED, memory_id))
        conn.execute(
            "INSERT INTO corrections (correction_id, memory_id, action, reason, actor, created_at)"
            " VALUES (?,?,'resolve',?,?,?)",
            (ids.new_id("cor"), memory_id, reason, actor, now()),
        )
    catalog_mod.register_memory(catalog, memory_id, repo_id, memory["kind"], memory["severity"],
                                "RESOLVED", memory["title"] or "")
    return {"ok": True, "memory_id": memory_id, "status": "RESOLVED"}


def get_memory(conn: sqlite3.Connection, memory_id: str) -> dict[str, Any] | None:
    memory = one(conn.execute("SELECT * FROM memories WHERE memory_id=?", (memory_id,)))
    if memory is None:
        return None

    # Opening a memory in full is the strongest usage signal there is: it
    # means an agent chose this one out of everything it was shown.
    from .search import note_accessed
    note_accessed(conn, memory_id)

    memory["anchors"] = rows(conn.execute(
        "SELECT anchor_id, target_kind, symbol_path, file_path, status, anchor_confidence,"
        " last_verified_commit, reanchor_history FROM anchors WHERE memory_id=?", (memory_id,)
    ))
    memory["edges"] = rows(conn.execute(
        "SELECT kind, to_id, edge_class, confidence, status FROM memory_edges WHERE from_id=?",
        (memory_id,)
    ))
    memory["versions"] = rows(conn.execute(
        "SELECT version, changed_at, changed_by, reason FROM memory_versions WHERE memory_id=?"
        " ORDER BY version DESC", (memory_id,)
    ))
    return memory


# Worst-first ordering. A memory is only as trustworthy as its weakest anchor,
# so a listing must never report the most optimistic one - alphabetical MIN()
# would pick ACTIVE over DRIFTED and hide exactly what needs attention.
_ANCHOR_RANK = {
    "ACTIVE": 0, "DRIFTED": 1, "NEEDS_REVIEW": 2, "ORPHANED": 3,
    "RESOLVED": 4, "SUPERSEDED": 5,
}
_RANK_TO_STATUS = {v: k for k, v in _ANCHOR_RANK.items()}

_RANK_SQL = ("MAX(CASE a.status WHEN 'ACTIVE' THEN 0 WHEN 'DRIFTED' THEN 1"
             " WHEN 'NEEDS_REVIEW' THEN 2 WHEN 'ORPHANED' THEN 3"
             " WHEN 'RESOLVED' THEN 4 WHEN 'SUPERSEDED' THEN 5 ELSE 0 END)")


def list_memories(conn: sqlite3.Connection, kind: str | None = None, status: str | None = None,
                  anchor_status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    sql = (f"SELECT m.memory_id, m.kind, m.title, m.severity, m.status, m.authority,"
           f" m.last_verified_commit, {_RANK_SQL} AS anchor_rank,"
           f" MIN(a.anchor_confidence) AS anchor_confidence, COUNT(a.anchor_id) AS anchor_count"
           f" FROM memories m LEFT JOIN anchors a ON a.memory_id = m.memory_id WHERE 1=1")
    args: list[Any] = []
    if kind:
        sql += " AND m.kind = ?"
        args.append(kind)
    if status:
        sql += " AND m.status = ?"
        args.append(status)
    sql += " GROUP BY m.memory_id"
    if anchor_status:
        rank = _ANCHOR_RANK.get(anchor_status.upper())
        if rank is None:
            return []
        sql += " HAVING anchor_rank = ?"
        args.append(rank)
    # Worst anchors first: the point of a listing is to surface what needs work.
    sql += " ORDER BY anchor_rank DESC, m.created_at DESC LIMIT ?"
    args.append(max(1, min(limit, 500)))

    out = rows(conn.execute(sql, tuple(args)))
    for row in out:
        rank = row.pop("anchor_rank", 0) or 0
        row["anchor_status"] = _RANK_TO_STATUS.get(rank, "ACTIVE") if row["anchor_count"] else None
        if row["anchor_confidence"] is not None:
            row["anchor_confidence"] = round(row["anchor_confidence"], 2)
    return out
