"""Knowledge ICN derived itself, proposed rather than asserted.

Everything else in this store was written because an agent said it. That makes
the store honest and it makes it small: it can only ever contain what somebody
thought to type. A pattern sitting in plain sight across five memories is
invisible until a human notices it and records it by hand.

This module is the one place ICN reasons from what it already knows to
something it was never told. Two rules govern it.

**It proposes, it never asserts.** A derived memory is written with
`is_inference=1` and no review decision. Search down-ranks it (search.py
INFERENCE_PENALTY), the briefing does not quote it as a rule, and rule
promotion refuses it outright. It has to be approved by somebody before it
carries the weight of a stated fact. This is deliberate and it is the whole
safety argument: the cost of a wrong inference is not that it is wrong, it is
that it becomes indistinguishable from something that was verified.

**It reasons over structure, never over language.** Every rule below fires on
the deterministic code graph - callers, tests, anchors, edges - and on
agreement between existing claims. There is no model in this path and no
paraphrase. That is what makes a proposal explainable: each one carries the
parent memories it came from, and a reviewer can read them and decide. An LLM
asked to "find patterns in these memories" would produce more candidates and
none of them checkable.

The rules are deliberately few and deliberately conservative. A review queue
nobody empties is worse than no queue, so the bar for proposing is high: a
candidate that a reviewer would decline is more expensive than a candidate
never raised. `MIN_PARENTS` and `MIN_AGREEMENT` are set so that a rule fires on
genuine convergence, not coincidence.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any

from . import ids
from .db import jdump, one, rows, write_tx

# A proposal needs this many independent memories behind it. Two memories
# agreeing is a pair; three is a pattern. Set to 3 because at 2 the dominant
# source of candidates was the same claim recorded twice about neighbouring
# symbols, which is a duplicate-detection problem, not an inference.
MIN_PARENTS = 3

# How much of the combined vocabulary must be terms a majority of the claims
# share, before they count as saying the same thing.
#
# Measured, not guessed, and the scale is not intuitive: memory claims are long
# prose, so even claims that clearly converge share only a small fraction of a
# large combined vocabulary. On this repository's own store (203 candidate
# callees, 84 with any shared terms at all) the distribution separates cleanly:
# 4 candidates at >= 0.03 and no change through 0.06, then 22 by 0.02. The
# shoulder is at 0.03, and the 4 above it are real - the strongest is five
# memories converging on the native-import deadlock.
#
# Do not raise this toward the 0.5-1.0 range that looks reasonable for a
# similarity score. The first version of this file used 0.6 and proposed
# literally nothing on a 297-memory store.
MIN_AGREEMENT = 0.03

# Pairwise Jaccard between two individual claims, which is a different and much
# stricter measurement than MIN_AGREEMENT above: that one scores a whole set at
# once and is dragged down by the size of the combined vocabulary, this one
# compares two claims directly. Kept high because a convention proposed from
# claims that merely share topic words would be noise, and this rule fires
# repository-wide rather than on one symbol.
CONVERGENCE_SIMILARITY = 0.5

# Derived memories are born here. Confidence is capped well below an asserted
# memory's default (0.8): an inference that has not been reviewed should never
# outrank a fact somebody stated, whatever its support.
INFERENCE_CONFIDENCE = 0.5

# Words too common in this domain to count as agreement between two claims.
# Without this, "the function must not fail" and "the test must not fail" read
# as the same rule.
GENERIC = {
    "code", "call", "calls", "called", "function", "method", "value", "values",
    "return", "returns", "file", "files", "line", "lines", "test", "tests",
    "must", "should", "this", "that", "with", "from", "when", "then", "than",
    "have", "has", "had", "been", "being", "does", "doing", "done", "make",
    "made", "data", "type", "types", "name", "names", "used", "using",
    "into", "over", "same", "each", "both", "will", "would", "could", "because",
}

# Kinds that can carry a rule. An inference proposes a rule about code, so it
# is only ever derived from memories that were themselves making a claim about
# how the code must behave. A performance note or a test result is evidence,
# not a rule, and generalising from one would be a category error.
RULE_KINDS = ("invariant", "warning", "contract", "decision")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _claim_of(memory: dict[str, Any]) -> str:
    """The claim as stated, falling back for rows written before the column."""
    if memory.get("claim"):
        return str(memory["claim"])
    body = str(memory.get("body") or "")
    for marker in ("\n\nRecorded while: ", "\n\nWhy: ", "\n\nChanged: ", "\n\nApplies to: "):
        body = body.split(marker, 1)[0]
    return body


def _significant(text: str) -> set[str]:
    """Content words of a claim, with domain-generic ones removed."""
    words = {w for w in "".join(c if c.isalnum() else " " for c in text.lower()).split()
             if len(w) >= 4}
    return words - GENERIC


def _agreement(claims: list[str]) -> tuple[float, list[str]]:
    """How much a set of claims actually say the same thing.

    Consensus, not unanimity. The obvious implementation intersects every
    claim's vocabulary, and it is wrong: measured against this repository's own
    store, all 203 candidates scored exactly 0.0, because requiring one word to
    appear in all 84 memories attached to a popular callee is a bar nothing
    real clears. Agreement among many is never total, and a metric that only
    fires on unanimity fires on nothing.

    So a term counts when a MAJORITY of the claims use it, and the score is the
    share of the combined vocabulary such terms make up. That rewards a
    consistent theme across most supporters while still being dragged down by
    a set of claims that are merely adjacent.

    Returns the score and the agreed terms. The terms are what a reviewer reads
    to judge the proposal, so they are part of the result, not a debug aid.
    """
    vocabularies = [v for v in (_significant(c) for c in claims) if v]
    if len(vocabularies) < 2:
        return 0.0, []
    counts: dict[str, int] = {}
    for vocabulary in vocabularies:
        for word in vocabulary:
            counts[word] = counts.get(word, 0) + 1
    needed = max(2, (len(vocabularies) // 2) + 1)
    shared = sorted(w for w, n in counts.items() if n >= needed)
    if not counts:
        return 0.0, []
    return len(shared) / len(counts), shared


# --------------------------------------------------------------------- rules

def _shared_rule_on_callers(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """If everything that calls X obeys the same rule, X probably carries it.

    The strongest structural inference available here, because it uses the one
    thing ICN has that a text-only memory store does not: a real call graph.
    When three separate sessions recorded the same constraint against three
    different callers of the same function, nobody ever wrote it down about the
    function itself - and that is exactly the memory the next agent needs,
    because they will arrive at the callee.

    Requires agreement among the callers AND that the callee has no such rule
    already, so this cannot restate what is already known.
    """
    out: list[dict[str, Any]] = []
    placeholders = ",".join("?" for _ in RULE_KINDS)

    # Callees that memory-bearing callers point at. Grouped in SQL so that the
    # Python below only ever sees candidate callees, not the whole graph.
    candidates = rows(conn.execute(
        f"SELECT e.to_id AS callee, COUNT(DISTINCT m.memory_id) AS supporters"
        f" FROM code_edges e"
        f" JOIN memory_edges me ON me.to_id = e.from_id AND me.kind='APPLIES_TO'"
        f"   AND me.status='ACTIVE'"
        f" JOIN memories m ON m.memory_id = me.from_id"
        f" WHERE e.kind='CALLS' AND e.status='ACTIVE' AND m.status='ACTIVE'"
        f"   AND COALESCE(m.is_inference, 0) = 0"
        f"   AND m.kind IN ({placeholders})"
        f" GROUP BY e.to_id HAVING supporters >= ?",
        (*RULE_KINDS, MIN_PARENTS)))

    for candidate in candidates:
        callee_id = candidate["callee"]
        callee = one(conn.execute(
            "SELECT * FROM symbols WHERE symbol_id=? AND status='ACTIVE'", (callee_id,)))
        if callee is None:
            continue

        # Never propose what is already stated about this symbol.
        existing = one(conn.execute(
            f"SELECT COUNT(*) AS n FROM memory_edges e JOIN memories m ON m.memory_id = e.from_id"
            f" WHERE e.to_id=? AND e.kind='APPLIES_TO' AND e.status='ACTIVE'"
            f" AND m.status='ACTIVE' AND m.kind IN ({placeholders})",
            (callee_id, *RULE_KINDS)))
        if existing and int(existing["n"]):
            continue

        supporters = rows(conn.execute(
            f"SELECT DISTINCT m.memory_id, m.kind, m.title, m.body, m.claim, m.severity"
            f" FROM code_edges e"
            f" JOIN memory_edges me ON me.to_id = e.from_id AND me.kind='APPLIES_TO'"
            f"   AND me.status='ACTIVE'"
            f" JOIN memories m ON m.memory_id = me.from_id"
            f" WHERE e.to_id=? AND e.kind='CALLS' AND e.status='ACTIVE'"
            f"   AND m.status='ACTIVE' AND COALESCE(m.is_inference, 0) = 0"
            f"   AND m.kind IN ({placeholders})",
            (callee_id, *RULE_KINDS)))
        if len(supporters) < MIN_PARENTS:
            continue

        claims = [_claim_of(s) for s in supporters]
        agreement, shared = _agreement(claims)
        if agreement < MIN_AGREEMENT or len(shared) < 2:
            continue

        # The proposal is phrased as a question about the callee, not as a
        # statement of fact. A reviewer approving it is what turns it into one.
        path = callee.get("symbol_path") or callee.get("name") or callee_id
        claim = (f"{path} may carry the rule its callers all observe: "
                 f"{', '.join(shared[:8])}. "
                 f"{len(supporters)} memories on {len(supporters)} of its callers agree, "
                 f"but nothing states this about {path} itself.")
        out.append({
            "rule": "shared_rule_on_callers",
            "claim": claim,
            "kind": "invariant",
            "severity": _dominant_severity(supporters),
            "parents": [s["memory_id"] for s in supporters],
            "anchor_symbol": callee_id,
            "agreement": round(agreement, 3),
            "shared_terms": shared[:12],
        })
    return out


def _unguarded_rule(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """A rule with no test behind it, on code that tests do reach.

    Not a claim about the code: a claim about the knowledge. ICN records
    GUARDED_BY when a test covers a rule, so their absence is legible. This
    proposes the gap as a candidate `warning` so it lands in the review queue
    where somebody decides whether it matters, rather than in a report nobody
    opens.

    Only fires where the symbol IS reached by tests, so it says "this rule is
    untested" rather than "this code is untested", which is a different and far
    less actionable statement.
    """
    out: list[dict[str, Any]] = []
    unguarded = rows(conn.execute(
        "SELECT m.memory_id, m.kind, m.title, m.claim, m.body, m.severity,"
        "       s.symbol_id, s.symbol_path, s.last_known_path"
        " FROM memories m"
        " JOIN memory_edges e ON e.from_id = m.memory_id AND e.kind='APPLIES_TO'"
        "   AND e.status='ACTIVE'"
        " JOIN symbols s ON s.symbol_id = e.to_id AND s.status='ACTIVE'"
        " WHERE m.status='ACTIVE' AND COALESCE(m.is_inference, 0) = 0"
        "   AND m.kind IN ('invariant','contract') AND m.severity IN ('critical','high')"
        "   AND NOT EXISTS (SELECT 1 FROM memory_edges g WHERE g.from_id = m.memory_id"
        "                   AND g.kind='GUARDED_BY' AND g.status='ACTIVE')"
        " LIMIT 200"))

    for memory in unguarded:
        # Does anything test-shaped reach this symbol at all?
        reached = one(conn.execute(
            "SELECT COUNT(*) AS n FROM code_edges e JOIN symbols s ON s.symbol_id = e.from_id"
            " WHERE e.to_id=? AND e.kind='CALLS' AND e.status='ACTIVE' AND s.status='ACTIVE'"
            " AND (LOWER(s.name) LIKE 'test%' OR LOWER(s.last_known_path) LIKE '%test%')",
            (memory["symbol_id"],)))
        if not reached or not int(reached["n"]):
            continue

        path = memory.get("symbol_path") or memory["symbol_id"]
        claim = (f"The {memory['severity']} {memory['kind']} on {path} has no test recorded "
                 f"against it, though {int(reached['n'])} test(s) reach that code. "
                 f"Either a test already covers it and the link is missing "
                 f"(memory(action='guard')), or the rule is unguarded.")
        out.append({
            "rule": "unguarded_rule",
            "claim": claim,
            "kind": "warning",
            "severity": "medium",
            "parents": [memory["memory_id"]],
            "anchor_symbol": memory["symbol_id"],
            "agreement": 1.0,
            "shared_terms": [],
            # One parent, but it is a deterministic structural fact rather than
            # a generalisation, so it does not need MIN_PARENTS behind it.
            "single_parent_ok": True,
        })
    return out


def _convergent_rule(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """The same constraint stated about several unrelated symbols is a convention.

    Duplicate detection already catches the same claim about the SAME code and
    reinforces it. This is the opposite case: the same claim about DIFFERENT
    code, which duplicate detection deliberately leaves alone because two rules
    that read alike about different symbols are two rules. Said about enough
    different places, though, it stops being local and becomes how this
    repository is written - and a convention nobody has stated is exactly what
    a new agent breaks first.
    """
    out: list[dict[str, Any]] = []
    placeholders = ",".join("?" for _ in RULE_KINDS)
    memories = rows(conn.execute(
        f"SELECT m.memory_id, m.kind, m.title, m.claim, m.body, m.severity"
        f" FROM memories m WHERE m.status='ACTIVE' AND COALESCE(m.is_inference, 0) = 0"
        f" AND m.kind IN ({placeholders}) LIMIT 400", RULE_KINDS))

    # Group by shared vocabulary. Small N, so an O(n^2) pass is cheaper than
    # building an index, and this runs on demand rather than per record.
    buckets: list[dict[str, Any]] = []
    for memory in memories:
        vocabulary = _significant(_claim_of(memory))
        if len(vocabulary) < 3:
            continue
        for bucket in buckets:
            shared = bucket["vocabulary"] & vocabulary
            union = bucket["vocabulary"] | vocabulary
            if union and len(shared) / len(union) >= CONVERGENCE_SIMILARITY:
                bucket["members"].append(memory)
                bucket["vocabulary"] = shared
                break
        else:
            buckets.append({"vocabulary": vocabulary, "members": [memory]})

    for bucket in buckets:
        members = bucket["members"]
        if len(members) < MIN_PARENTS or len(bucket["vocabulary"]) < 2:
            continue
        # Must be about different code, or this is duplicate detection's job.
        targets: set[str] = set()
        for member in members:
            targets |= {r["to_id"] for r in conn.execute(
                "SELECT to_id FROM memory_edges WHERE from_id=? AND kind='APPLIES_TO'"
                " AND status='ACTIVE'", (member["memory_id"],))}
        if len(targets) < MIN_PARENTS:
            continue

        shared = sorted(bucket["vocabulary"])
        claim = (f"A repository-wide convention may be unstated: {len(members)} memories across "
                 f"{len(targets)} different symbols all say the same thing "
                 f"({', '.join(shared[:8])}). If it holds generally it belongs in the rules, "
                 f"not repeated per symbol.")
        out.append({
            "rule": "convergent_rule",
            "claim": claim,
            "kind": "convention",
            "severity": _dominant_severity(members),
            "parents": [m["memory_id"] for m in members],
            "anchor_symbol": None,
            "agreement": round(len(bucket["vocabulary"]) / max(1, len(bucket["vocabulary"])), 3),
            "shared_terms": shared[:12],
        })
    return out


def _dominant_severity(memories: list[dict[str, Any]]) -> str:
    """The most serious severity among the parents, never more than they claim."""
    order = ("critical", "high", "medium", "low")
    present = [str(m.get("severity") or "medium") for m in memories]
    for level in order:
        if level in present:
            # An inference is one step less severe than its evidence: it is a
            # proposal, and a proposal that shouts is a proposal that gets
            # approved without being read.
            return {"critical": "high", "high": "medium"}.get(level, level)
    return "medium"


RULES = (_shared_rule_on_callers, _unguarded_rule, _convergent_rule)


# ------------------------------------------------------------------ proposing

def _already_proposed(conn: sqlite3.Connection, rule: str, parents: list[str]) -> bool:
    """Has this exact proposal been made before, whatever its review outcome?

    Keyed on the rule and its parent set, not the claim text, so re-running
    after a declined review does not resurrect what a reviewer rejected. A
    decline is a decision, and repeating the question is how a review queue
    loses its reader.
    """
    signature = jdump({"rule": rule, "parents": sorted(parents)})
    hit = one(conn.execute(
        "SELECT COUNT(*) AS n FROM memory_edges"
        " WHERE kind='DERIVED_FROM' AND source='infer' AND evidence=?", (signature,)))
    return bool(hit and int(hit["n"]))


def propose(conn: sqlite3.Connection, commit: str | None = None,
            limit: int = 25) -> dict[str, Any]:
    """Run every rule and write what they found as unreviewed inferences.

    Returns what was proposed. Writing nothing is the normal and healthy
    outcome on a store whose knowledge is already explicit.
    """
    found: list[dict[str, Any]] = []
    for rule in RULES:
        try:
            found.extend(rule(conn))
        except sqlite3.Error:
            # One malformed rule must not cost the others their output.
            continue

    # Strongest support first, so a truncated run proposes the best candidates
    # rather than whichever rule happened to run first.
    found.sort(key=lambda c: (len(c["parents"]), c.get("agreement", 0)), reverse=True)

    written: list[dict[str, Any]] = []
    for candidate in found:
        if len(written) >= limit:
            break
        parents = candidate["parents"]
        if len(parents) < MIN_PARENTS and not candidate.get("single_parent_ok"):
            continue
        if _already_proposed(conn, candidate["rule"], parents):
            continue

        memory_id = ids.new_id(ids.MEMORY)
        claim = candidate["claim"]
        title = claim.split("\n", 1)[0][:200]
        signature = jdump({"rule": candidate["rule"], "parents": sorted(parents)})

        with write_tx(conn):
            conn.execute(
                "INSERT INTO memories (memory_id, kind, title, body, severity, authority,"
                " confidence, status, scope, version, created_at, updated_at, created_commit,"
                " claim, evidence_count, is_inference, review_status, parent_count)"
                " VALUES (?,?,?,?,?,'derived',?,'ACTIVE','repo',1,?,?,?,?,?,1,NULL,?)",
                (memory_id, candidate["kind"], title, claim, candidate["severity"],
                 INFERENCE_CONFIDENCE, now(), now(), commit, claim, len(parents), len(parents)),
            )
            conn.execute(
                "DELETE FROM fts_memories WHERE memory_id = ?", (memory_id,))
            conn.execute(
                "INSERT INTO fts_memories (memory_id, title, body, kind) VALUES (?,?,?,?)",
                (memory_id, title, claim[:4000], candidate["kind"]))

            # The provenance edge doubles as the de-duplication key.
            conn.execute(
                "INSERT INTO memory_edges (edge_id, from_id, to_id, kind, edge_class, status,"
                " confidence, source, evidence, created_at)"
                " VALUES (?,?,?,'DERIVED_FROM','inferred','ACTIVE',?,'infer',?,?)",
                (ids.new_id(ids.EDGE), memory_id, parents[0], candidate.get("agreement", 0.5),
                 signature, now()))

            # Every parent, so the reviewer can read the evidence.
            for parent in parents:
                conn.execute(
                    "INSERT OR IGNORE INTO memory_edges (edge_id, from_id, to_id, kind,"
                    " edge_class, status, confidence, source, created_at)"
                    " VALUES (?,?,?,'DERIVES_FROM','inferred','ACTIVE',?,'infer',?)",
                    (ids.new_id(ids.EDGE), memory_id, parent,
                     candidate.get("agreement", 0.5), now()))

            if candidate.get("anchor_symbol"):
                conn.execute(
                    "INSERT OR IGNORE INTO memory_edges (edge_id, from_id, to_id, kind,"
                    " edge_class, status, confidence, source, created_at)"
                    " VALUES (?,?,?,'APPLIES_TO','inferred','ACTIVE',?,'infer',?)",
                    (ids.new_id(ids.EDGE), memory_id, candidate["anchor_symbol"],
                     INFERENCE_CONFIDENCE, now()))

        written.append({
            "memory_id": memory_id, "rule": candidate["rule"], "kind": candidate["kind"],
            "severity": candidate["severity"], "title": title,
            "parent_count": len(parents), "parents": parents,
            "shared_terms": candidate.get("shared_terms", []),
        })

    return {
        "ok": True,
        "proposed": written,
        "count": len(written),
        "candidates_considered": len(found),
        "note": ("inferences are unreviewed: they are down-ranked in search and never promoted "
                 "to rules until memory(action='review', signal='approve') settles them"
                 if written else "nothing new to infer; stated knowledge already covers what "
                                 "the rules can see"),
    }


# -------------------------------------------------------------------- review

def queue(conn: sqlite3.Connection, limit: int = 30) -> dict[str, Any]:
    """Inferences awaiting a decision, best-supported first."""
    pending = rows(conn.execute(
        "SELECT memory_id, kind, title, body, severity, parent_count, created_at"
        " FROM memories WHERE status='ACTIVE' AND is_inference=1 AND review_status IS NULL"
        " ORDER BY parent_count DESC, created_at DESC LIMIT ?", (limit,)))
    for item in pending:
        item["parents"] = [r["to_id"] for r in conn.execute(
            "SELECT to_id FROM memory_edges WHERE from_id=? AND kind='DERIVES_FROM'"
            " AND status='ACTIVE'", (item["memory_id"],))]
    total = one(conn.execute(
        "SELECT COUNT(*) AS n FROM memories WHERE status='ACTIVE' AND is_inference=1"
        " AND review_status IS NULL"))
    return {"ok": True, "pending": pending, "count": len(pending),
            "total": int(total["n"]) if total else 0,
            "note": "approve promotes an inference to a stated fact; decline forgets it"}


def review(conn: sqlite3.Connection, memory_id: str, signal: str,
           reason: str = "", actor: str = "agent") -> dict[str, Any]:
    """Settle one inference: approve, decline, or undo a previous decision.

    Approve clears `is_inference`, which is what lifts the search penalty and
    makes the memory eligible for rule promotion. Decline forgets it. Undo
    returns it to the queue exactly as it was, which is why review_status is a
    separate column from status: nothing has to be reconstructed.
    """
    signal = (signal or "").lower().strip()
    if signal not in ("approve", "decline", "undo"):
        return {"ok": False, "error": "signal must be 'approve', 'decline' or 'undo'"}

    memory = one(conn.execute("SELECT * FROM memories WHERE memory_id=?", (memory_id,)))
    if memory is None:
        return {"ok": False, "error": f"unknown memory {memory_id}"}
    if signal != "undo" and not int(memory.get("is_inference") or 0):
        return {"ok": False, "error": f"{memory_id} is not an inference; "
                                      "only derived memories are reviewed",
                "memory_id": memory_id}
    if signal == "undo" and not memory.get("review_status"):
        return {"ok": False, "error": f"{memory_id} has no review to undo", "memory_id": memory_id}

    with write_tx(conn):
        if signal == "approve":
            # A promoted inference becomes an ordinary memory. It keeps
            # authority='derived' forever: how it was born is provenance, and
            # approving it says it is true, not that a human wrote it.
            conn.execute(
                "UPDATE memories SET is_inference=0, review_status='approved', confidence=?,"
                " updated_at=? WHERE memory_id=?", (0.75, now(), memory_id))
        elif signal == "decline":
            conn.execute(
                "UPDATE memories SET status='FORGOTTEN', review_status='declined', updated_at=?"
                " WHERE memory_id=?", (now(), memory_id))
            # A forgotten memory leaves search, exactly as a superseded one does.
            conn.execute("DELETE FROM fts_memories WHERE memory_id=?", (memory_id,))
        else:
            conn.execute(
                "UPDATE memories SET is_inference=1, review_status=NULL, status='ACTIVE',"
                " confidence=?, updated_at=? WHERE memory_id=?",
                (INFERENCE_CONFIDENCE, now(), memory_id))
            conn.execute("DELETE FROM fts_memories WHERE memory_id=?", (memory_id,))
            conn.execute(
                "INSERT INTO fts_memories (memory_id, title, body, kind) VALUES (?,?,?,?)",
                (memory_id, memory["title"] or "", (memory["body"] or "")[:4000],
                 memory["kind"] or ""))

        conn.execute(
            "INSERT INTO corrections (correction_id, memory_id, action, reason, actor, detail,"
            " created_at) VALUES (?,?,?,?,?,?,?)",
            (ids.new_id("cor"), memory_id, f"review_{signal}", reason, actor,
             jdump({"signal": signal}), now()))

    updated = one(conn.execute("SELECT * FROM memories WHERE memory_id=?", (memory_id,)))
    return {"ok": True, "memory_id": memory_id, "signal": signal,
            "is_inference": bool(int(updated["is_inference"] or 0)),
            "review_status": updated["review_status"],
            "status": updated["status"],
            "note": {"approve": "promoted to a stated fact; no longer down-ranked in search",
                     "decline": "forgotten; it leaves search and the queue",
                     "undo": "returned to the review queue, unchanged"}[signal]}
