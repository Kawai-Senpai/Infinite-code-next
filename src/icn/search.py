"""investigate(): the one search that covers everything.

PLAN 2 section 7. Engines fused per call: FTS over symbols and memories, exact
symbol lookup, code-graph traversal, memory-graph traversal, anchor status, git
history and agit checkpoints.

Ranking is a static, inspectable formula rather than a trained reranker. Every
strong result in the retrieval literature buys its gains with labeled relevance
pairs (arXiv 2412.01007, 2608.09650) that a fresh local install does not have.
What we borrow instead needs no labels: per-intent fusion weights in place of
one global blend (arXiv 2605.30237), a specificity discount that penalises
spans matching nearly every query (arXiv 2603.11800), a compact retrieval key
separate from the injected payload (arXiv 2508.15294), and a staleness penalty
with visible status labels (arXiv 2604.20006).

Output is capsules, never file dumps. A memory is never rendered as settled
fact when its anchor says otherwise.
"""

from __future__ import annotations

import math
import re
import sqlite3
from difflib import SequenceMatcher
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import anchors as anchor_mod
from . import causal
from . import crossrepo
from . import diagnostics
from . import ids
from .db import jdump, jload, one, rows, write_tx
from .identity import run_git
from .resolver import describe_edge, resolve

INTENTS = ("locate", "understand", "modify", "debug", "audit")

# Per-intent fusion weights. Hand-tuned constants, not learned.
INTENT_WEIGHTS: dict[str, dict[str, float]] = {
    "locate":     {"lex": 1.0, "sym": 1.5, "graph": 0.3, "sev": 0.2, "time": 0.1, "test": 0.1, "mem": 0.3},
    "understand": {"lex": 0.8, "sym": 1.0, "graph": 0.9, "sev": 0.6, "time": 0.3, "test": 0.3, "mem": 1.0},
    "modify":     {"lex": 0.7, "sym": 1.0, "graph": 1.0, "sev": 1.4, "time": 0.5, "test": 0.8, "mem": 1.3},
    "debug":      {"lex": 0.8, "sym": 0.9, "graph": 0.9, "sev": 1.0, "time": 1.0, "test": 0.6, "mem": 1.1},
    "audit":      {"lex": 0.6, "sym": 0.7, "graph": 1.1, "sev": 1.2, "time": 0.4, "test": 1.0, "mem": 1.2},
}

SEVERITY_SCORE = {"critical": 1.0, "high": 0.75, "medium": 0.45, "low": 0.2}

# Below this a memory is named rather than quoted in full, never dropped.
# Calibrated against measured scores, not guessed: on a real store a memory
# that answers the query lands at 1.8-2.7 and an unrelated one near 1.0, while
# a safety-critical memory on code being modified clears the floor from its
# kind floor alone (0.62 * 1.6 = 0.99, plus severity).
MEMORY_FLOOR = 1.45

# A memory whose anchor cannot vouch for it is down-ranked, never hidden.
STALE_PENALTY = {
    anchor_mod.ACTIVE: 0.0,
    anchor_mod.DRIFTED: 0.35,
    anchor_mod.NEEDS_REVIEW: 0.5,
    anchor_mod.ORPHANED: 0.7,
    anchor_mod.SUPERSEDED: 1.2,
    anchor_mod.RESOLVED: 0.9,
}

# Edge trust decides how far relevance travels along it.
EDGE_DECAY = {"deterministic": 0.75, "asserted": 0.65, "inferred": 0.35}

INTENT_PATTERNS = [
    ("modify", r"\b(change|modify|refactor|rewrite|remove|delete|replace|rename|add|implement|migrate|break)\b"),
    ("debug", r"\b(bug|debug|fail|failing|broken|error|crash|wrong|why is|regression|flaky)\b"),
    ("audit", r"\b(audit|review|coverage|inconsisten|unguarded|everywhere|all (the )?(uses|callers|places))\b"),
    ("understand", r"\b(how|why|explain|understand|architecture|design|rationale|works)\b"),
    ("locate", r"\b(where|find|locate|which file|defined)\b"),
]

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was", "were",
    "i", "it", "this", "that", "what", "how", "why", "where", "which", "do", "does", "did",
    "want", "need", "get", "me", "my", "up", "speed", "tell", "might", "break", "about",
    "can", "should", "would", "there", "then", "with", "from", "into", "have", "has",
    "will", "just", "also", "some", "any", "all", "not", "but", "when", "who", "here",
    "make", "made", "use", "used", "using", "now", "new", "old", "way", "like", "going",
    "based", "observed", "failure", "failures", "must", "show", "hide", "expose", "better",
    "initial", "harden", "fix", "fixed", "issue", "issues",
}


def now() -> str:
    # Milliseconds, not seconds. Ordering questions ("did this caller
    # appear after that memory was verified?") are decided by comparing
    # these, and second precision made same-second events compare equal,
    # so a genuinely late caller went unreported.
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def infer_intent(query: str) -> str:
    text = (query or "").lower()
    for intent, pattern in INTENT_PATTERNS:
        if re.search(pattern, text):
            return intent
    return "understand"


def _terms(query: str) -> list[str]:
    # Hyphens are kept: `re-anchor` must survive as one term so _variants()
    # can also try `reanchor`. Splitting here dropped the `re` as too short and
    # searched for bare `anchor`, which matches far too much.
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_-]*[A-Za-z0-9_]|[A-Za-z_][A-Za-z0-9_]{2,}",
                       query or "")
    out: list[str] = []
    for word in words:
        if word.lower() in STOPWORDS:
            continue
        out.append(word)
        # Split camelCase and snake_case so "RefreshCoordinator" also matches
        # "refresh" and "coordinator".
        for piece in re.findall(r"[A-Z]?[a-z]{3,}|[A-Z]{3,}", word):
            if piece.lower() not in STOPWORDS:
                out.append(piece)
    seen: set[str] = set()
    unique = []
    for term in out:
        key = term.lower()
        if key not in seen:
            seen.add(key)
            unique.append(term)
    return unique[:12]


def _fts_query(terms: list[str]) -> str:
    """One MATCH expression covering every spelling of every term."""
    forms: list[str] = []
    for term in terms:
        for variant in _variants(term):
            if variant not in forms:
                forms.append(variant)
    return " OR ".join(f'"{t}"*' for t in forms) if forms else ""



def _variants(term: str) -> list[str]:
    """Forms of one term that should match the same text.

    `reanchor` and `re-anchor` are the same word to a reader, but FTS5's
    unicode61 tokenizer splits on the hyphen, so neither query finds the other.
    Splitting camelCase and dropping separators covers the cases that actually
    come up in code: reAnchor, re_anchor, re-anchor, reanchor.
    """
    out = [term]
    squashed = re.sub(r"[-_]", "", term)
    if squashed and squashed != term:
        out.append(squashed)
    for piece in re.split(r"[-_]", term):
        if len(piece) > 2 and piece != term:
            out.append(piece)

    # The reverse direction: a query of `reanchor` must reach text that says
    # `re-anchor`. FTS cannot express "ignore separators", so split on common
    # prefixes and search the remainder, which the tokenizer does index as its
    # own token.
    if "-" not in term and "_" not in term and len(term) > 6:
        for prefix in ("re", "un", "de", "pre", "non", "sub", "auto", "multi"):
            if term.lower().startswith(prefix) and len(term) - len(prefix) > 3:
                rest = term[len(prefix):]
                if rest not in out:
                    out.append(rest)
                break
    return out


def _fuzzy_memories(conn: sqlite3.Connection, terms: list[str],
                    limit: int = 20) -> dict[str, float]:
    """Approximate match, for when exact and prefix search both come up empty.

    Deliberately a fallback rather than a default: FTS5 ranking is better than
    anything computed here when it has hits at all, and running fuzzy matching
    over every query would let loose matches outrank exact ones. Scored below
    any real FTS hit so it can only fill an empty result, never displace one.
    """
    if not terms:
        return {}
    needles = [t.lower() for t in terms if len(t) > 3]
    if not needles:
        return {}

    scored: dict[str, float] = {}
    for row in rows(conn.execute(
        "SELECT memory_id, title, body FROM memories WHERE status='ACTIVE' LIMIT 4000"
    )):
        haystack = ((row["title"] or "") + " " + (row["body"] or "")).lower()
        squashed = re.sub(r"[-_\s]", "", haystack)
        best = 0.0
        for needle in needles:
            if needle in haystack:
                best = max(best, 0.5)
                continue
            if re.sub(r"[-_\s]", "", needle) in squashed:
                best = max(best, 0.42)     # matched only across a separator
                continue
            for word in set(re.findall(r"[a-z][a-z0-9]{3,}", haystack)):
                if abs(len(word) - len(needle)) > 3:
                    continue
                score = SequenceMatcher(None, needle, word).ratio()
                if score >= 0.82:
                    best = max(best, score * 0.4)
        if best:
            scored[row["memory_id"]] = best

    # Normalise onto the same 0..1 scale FTS relevance uses. Raw similarity
    # ratios sit around 0.4, which lands every fuzzy hit below MEMORY_FLOOR -
    # so the fallback found the right memories and the capsule then demoted
    # all of them, which reads to a user as "found nothing".
    top = sorted(scored.items(), key=lambda kv: -kv[1])[:limit]
    if not top:
        return {}
    best = top[0][1] or 1.0
    return {mid: min(1.0, score / best) for mid, score in top}


# Usage signal. Two different events, weighted differently on purpose:
#   surfaced  - the memory appeared in a result. Cheap, and mostly says the
#               retrieval matched, not that the memory was useful.
#   accessed  - an agent opened it in full. That is a real vote.
# A memory nobody has ever opened, however severe its author thought it was,
# has not yet proven itself; one agents keep returning to has.
USAGE_HALF_LIFE_DAYS = 45.0


def note_surfaced(conn: sqlite3.Connection, memory_ids: list[str]) -> None:
    """Record that these memories appeared in a result. Never raises: a
    ranking signal must not be able to fail a search."""
    if not memory_ids:
        return
    try:
        with write_tx(conn):
            conn.executemany(
                "UPDATE memories SET surfaced_count = COALESCE(surfaced_count, 0) + 1"
                " WHERE memory_id = ?", [(m,) for m in set(memory_ids)])
    except sqlite3.Error:
        pass


def note_accessed(conn: sqlite3.Connection, memory_id: str) -> None:
    """Record that an agent opened this memory in full."""
    try:
        with write_tx(conn):
            conn.execute(
                "UPDATE memories SET access_count = COALESCE(access_count, 0) + 1,"
                " last_accessed_at = ? WHERE memory_id = ?", (now(), memory_id))
    except sqlite3.Error:
        pass


def usage_boost(memory: dict[str, Any]) -> float:
    """How much a memory's track record should lift it, in 0..~0.5.

    Deliberately bounded and sub-linear. Frequency is evidence, not authority:
    letting it grow without limit would pin whatever was popular last month to
    the top of every result and bury a critical warning recorded yesterday.
    Recency decays it, so a memory that mattered once and never again fades.
    """
    opened = memory.get("access_count") or 0
    surfaced = memory.get("surfaced_count") or 0
    if not opened and not surfaced:
        return 0.0

    # Opens are worth far more than impressions; a memory can be surfaced by
    # a loose lexical match without anyone finding it useful.
    raw = math.log1p(opened * 4 + surfaced * 0.35)

    decay = 1.0
    stamp = memory.get("last_accessed_at")
    if stamp:
        try:
            when = datetime.fromisoformat(str(stamp))
            days = max(0.0, (datetime.now(timezone.utc) - when).total_seconds() / 86400)
            decay = 0.5 ** (days / USAGE_HALF_LIFE_DAYS)
        except (ValueError, TypeError):
            decay = 1.0

    return min(0.5, raw * 0.18 * decay)

# ------------------------------------------------------------------- retrieval

def _seed_symbols(conn: sqlite3.Connection, terms: list[str]) -> dict[str, dict[str, Any]]:
    """Lexical and exact-symbol seeds, with normalised sub-scores."""
    seeds: dict[str, dict[str, Any]] = {}
    if not terms:
        return seeds

    match = _fts_query(terms)
    try:
        hits = rows(conn.execute(
            "SELECT f.symbol_id, bm25(fts_symbols) AS score FROM fts_symbols f"
            " WHERE fts_symbols MATCH ? ORDER BY score LIMIT 120", (match,)
        ))
    except sqlite3.OperationalError:
        hits = []
    if hits:
        # bm25() returns lower-is-better; map onto 0..1 with the best hit at 1.
        best = min(h["score"] for h in hits)
        worst = max(h["score"] for h in hits)
        span = (worst - best) or 1.0
        for hit in hits:
            seeds.setdefault(hit["symbol_id"], {"lex": 0.0, "sym": 0.0})
            seeds[hit["symbol_id"]]["lex"] = 1.0 - ((hit["score"] - best) / span)

    for term in terms:
        for row in rows(conn.execute(
            "SELECT symbol_id, name, symbol_path FROM symbols WHERE status='ACTIVE'"
            " AND (name = ? COLLATE NOCASE OR symbol_path = ? COLLATE NOCASE) LIMIT 10",
            (term, term),
        )):
            seeds.setdefault(row["symbol_id"], {"lex": 0.0, "sym": 0.0})
            seeds[row["symbol_id"]]["sym"] = 1.0
        for row in rows(conn.execute(
            "SELECT symbol_id FROM symbols WHERE status='ACTIVE' AND name LIKE ? LIMIT 10",
            (f"%{term}%",),
        )):
            seeds.setdefault(row["symbol_id"], {"lex": 0.0, "sym": 0.0})
            seeds[row["symbol_id"]]["sym"] = max(seeds[row["symbol_id"]]["sym"], 0.5)
    return seeds


def _boost_explicit_paths(conn: sqlite3.Connection, query: str,
                          seeds: dict[str, dict[str, Any]]) -> None:
    """Give exact filenames and path fragments precedence over generic nouns."""
    candidates = re.findall(
        r"(?:[A-Za-z]:[\\/])?[^\s,;:'\"()]+[\\/][^\s,;:'\"()]+|"
        r"\b[A-Za-z0-9_.-]+\.(?:py|ts|tsx|js|jsx|go|rs|java|kt|cs|md|toml|yaml|yml|json)\b",
        query or "",
    )
    for raw in candidates:
        needle = raw.replace("\\", "/").lower().strip("./")
        basename = needle.rsplit("/", 1)[-1]
        for row in rows(conn.execute(
            "SELECT symbol_id, last_known_path FROM symbols WHERE status='ACTIVE' "
            "AND (LOWER(REPLACE(last_known_path, '\\', '/')) LIKE ? "
            "OR LOWER(REPLACE(last_known_path, '\\', '/')) LIKE ?) LIMIT 80",
            (f"%{needle}%", f"%/{basename}"),
        )):
            seed = seeds.setdefault(row["symbol_id"], {"lex": 0.0, "sym": 0.0})
            seed["lex"] = 1.0
            seed["sym"] = 1.0


def _seed_memories(conn: sqlite3.Connection, terms: list[str]) -> dict[str, float]:
    if not terms:
        return {}
    try:
        hits = rows(conn.execute(
            "SELECT memory_id, bm25(fts_memories) AS score FROM fts_memories"
            " WHERE fts_memories MATCH ? ORDER BY score LIMIT 60", (_fts_query(terms),)
        ))
    except sqlite3.OperationalError:
        hits = []
    if not hits:
        # Nothing matched exactly. A typo or an unfamiliar spelling should
        # still find the memory rather than returning an empty answer.
        return _fuzzy_memories(conn, terms)
    best = min(h["score"] for h in hits)
    worst = max(h["score"] for h in hits)
    span = (worst - best) or 1.0
    return {h["memory_id"]: 1.0 - ((h["score"] - best) / span) for h in hits}


def _expand(conn: sqlite3.Connection, seeds: dict[str, dict[str, Any]], depth: int) -> dict[str, float]:
    """Graph proximity from the seed set, decayed by edge trust class.

    This is the step plain text search cannot do: a memory two deterministic
    hops away often beats a lexically similar paragraph from an unrelated
    corner of the repo.
    """
    proximity: dict[str, float] = {sid: 1.0 for sid in seeds}
    frontier = list(seeds.keys())
    for _ in range(max(0, depth)):
        if not frontier:
            break
        placeholders = ",".join("?" for _ in frontier)
        neighbours = rows(conn.execute(
            f"SELECT from_id, to_id, edge_class FROM code_edges"
            f" WHERE status='ACTIVE' AND (from_id IN ({placeholders}) OR to_id IN ({placeholders}))"
            f" LIMIT 4000",
            (*frontier, *frontier),
        ))
        next_frontier: list[str] = []
        for edge in neighbours:
            decay = EDGE_DECAY.get(edge["edge_class"], 0.4)
            for near, far in ((edge["from_id"], edge["to_id"]), (edge["to_id"], edge["from_id"])):
                if near not in proximity or not far.startswith(ids.SYMBOL):
                    continue
                score = proximity[near] * decay
                if score > proximity.get(far, 0.0):
                    proximity[far] = score
                    next_frontier.append(far)
        frontier = next_frontier[:400]
    return proximity


def _memories_for_symbols(conn: sqlite3.Connection, symbol_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    if not symbol_ids:
        return {}
    placeholders = ",".join("?" for _ in symbol_ids)
    found = rows(conn.execute(
        f"SELECT e.to_id AS symbol_id, m.*, a.status AS anchor_status,"
        f" a.anchor_confidence, a.last_verified_commit AS anchor_verified_commit"
        f" FROM memory_edges e JOIN memories m ON m.memory_id = e.from_id"
        f" LEFT JOIN anchors a ON a.memory_id = m.memory_id AND a.symbol_id = e.to_id"
        f" WHERE e.to_id IN ({placeholders}) AND e.kind IN ('APPLIES_TO','IMPACTS','GUARDED_BY')"
        f" AND e.status='ACTIVE'",
        tuple(symbol_ids),
    ))
    # One memory can reach a symbol by more than one edge kind - APPLIES_TO
    # because the agent named it, IMPACTS because the compiler derived it. That
    # is correct in the graph but must not surface as a duplicate in a capsule.
    grouped: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, str]] = set()
    for row in found:
        key = (row["symbol_id"], row["memory_id"])
        if key in seen:
            continue
        seen.add(key)
        grouped.setdefault(row["symbol_id"], []).append(row)
    return grouped


def _specificity_map(conn: sqlite3.Connection, symbol_ids: list[str]) -> dict[str, float]:
    """Penalise spans that carry so many memories they match everything.

    Borrowed from traceability-link work where penalising over-general
    candidates raised precision (arXiv 2603.11800). Computed as one grouped
    query: the per-symbol version ran twice for every candidate.
    """
    if not symbol_ids:
        return {}
    out: dict[str, float] = {}
    for start in range(0, len(symbol_ids), 400):
        part = symbol_ids[start:start + 400]
        placeholders = ",".join("?" for _ in part)
        for row in rows(conn.execute(
            f"SELECT to_id, COUNT(*) AS n FROM memory_edges WHERE to_id IN ({placeholders})"
            f" AND kind='APPLIES_TO' AND status='ACTIVE' GROUP BY to_id", tuple(part)
        )):
            count = int(row["n"])
            if count > 3:
                out[row["to_id"]] = min(0.45, 0.08 * (count - 3))
    return out


_RECENCY_CACHE: dict[tuple[str, str], dict[str, float]] = {}
RECENCY_COMMITS = 200


def _recency_map(root: Path, commit: str | None) -> dict[str, float]:
    """Last-touched timestamp per path, from ONE git call.

    The obvious implementation - `git log -1 -- <path>` per candidate - spawns
    a subprocess per symbol. Measured live, that took an investigation past a
    90-second client timeout on a few hundred candidates, because process spawn
    dominates everything else on Windows. One `git log --name-only` over a
    window of commits answers the same question for every path at once.
    """
    key = (str(root), commit or "HEAD")
    cached = _RECENCY_CACHE.get(key)
    if cached is not None:
        return cached

    code, out, _ = run_git(
        ["log", f"-{RECENCY_COMMITS}", "--format=%ct", "--name-only"], root, strip=False
    )
    touched: dict[str, float] = {}
    if code == 0:
        stamp = 0.0
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.isdigit() and len(line) >= 9:
                stamp = float(line)
            elif stamp and line not in touched:
                touched[line] = stamp

    if len(_RECENCY_CACHE) > 16:
        _RECENCY_CACHE.clear()
    _RECENCY_CACHE[key] = touched
    return touched


def _recency(touched: dict[str, float], path: str | None) -> float:
    if not path:
        return 0.0
    stamp = touched.get(path)
    if not stamp:
        return 0.0
    age_days = max(0.0, (datetime.now(timezone.utc).timestamp() - stamp) / 86400.0)
    return 1.0 / (1.0 + age_days / 30.0)   # ~0.5 at one month old


# --------------------------------------------------------------------- capsules


def _memory_score(memory: dict[str, Any], relevance: dict[str, float] | None,
                  terms: list[str] | None, intent: str) -> float:
    """How much does this memory answer *this* query?

    Severity says how much a memory matters in general. Relevance says whether
    it is about what was asked. A capsule needs both, weighted by intent: when
    modifying code a critical warning earns its place even on a weak lexical
    match, while an "understand" query wants the memory that is actually on
    topic.
    """
    severity = SEVERITY_SCORE.get(memory["severity"], 0.3)

    lexical = (relevance or {}).get(memory["memory_id"], 0.0)
    if not lexical and terms:
        # FTS covers whole words; fall back to substring overlap so a query for
        # "storage layout" still reaches a body that says "storage layout".
        blob = ((memory.get("title") or "") + " " + (memory.get("body") or "")).lower()
        hits = sum(1 for term in terms if len(term) > 2 and term.lower() in blob)
        lexical = min(1.0, hits / max(1, len(terms))) * 0.7

    # Safety-critical kinds keep a floor, but only when it is earned. While
    # modifying or debugging, a warning on the code in front of you is relevant
    # whether or not you asked about it. While reading, that same floor pinned
    # every warning to an identical score and flattened the ranking - measured
    # as a plateau where eight unrelated memories all scored 1.40.
    # failed_attempt is in this set deliberately. "We already tried that and it
    # did not work" is the knowledge nothing else in a toolchain records, and
    # it is worth most at exactly the moment someone is about to try it again -
    # which is a modify or debug query, not a lexical match on its wording.
    critical = memory["kind"] in ("security", "warning", "invariant", "failed_attempt")
    floor = (0.62 if intent in ("modify", "debug", "audit") else 0.15) if critical else 0.0
    weight = 1.4 if intent in ("modify", "debug") else 0.8

    stale = STALE_PENALTY.get(memory.get("anchor_status") or anchor_mod.ACTIVE, 0.0)
    return (max(floor, lexical) * 1.6 + severity * weight
            + usage_boost(memory) - stale * 0.5)

def _capsule(conn: sqlite3.Connection, catalog: sqlite3.Connection, symbol: dict[str, Any],
             memories: list[dict[str, Any]], intent: str,
             already_shown: dict[str, str] | None = None,
             memory_relevance: dict[str, float] | None = None,
             terms: list[str] | None = None) -> dict[str, Any]:
    """A compact, self-contained briefing for one symbol.

    `already_shown` carries memory_id -> symbol across capsules in one result.
    A warning anchored to five call sites is one fact, and repeating its full
    text five times spends the token budget on duplication instead of
    coverage - which is the exact failure this tool exists to prevent.
    """
    callers = rows(conn.execute(
        "SELECT s.symbol_path, s.last_known_path FROM code_edges e"
        " JOIN symbols s ON s.symbol_id = e.from_id"
        " WHERE e.to_id=? AND e.kind='CALLS' AND e.status='ACTIVE' AND s.status='ACTIVE' LIMIT 6",
        (symbol["symbol_id"],),
    ))
    callees = rows(conn.execute(
        "SELECT s.symbol_path FROM code_edges e JOIN symbols s ON s.symbol_id = e.to_id"
        " WHERE e.from_id=? AND e.kind='CALLS' AND e.status='ACTIVE' AND s.status='ACTIVE' LIMIT 6",
        (symbol["symbol_id"],),
    ))
    broken = [
        describe_edge(catalog, conn, edge)
        for edge in rows(conn.execute(
            "SELECT * FROM code_edges WHERE from_id=? AND status!='ACTIVE' LIMIT 4",
            (symbol["symbol_id"],),
        ))
    ]

    tests = [c["symbol_path"] for c in callers
             if "test" in (c["symbol_path"] or "").lower()
             or "test" in (c["last_known_path"] or "").lower()
             or "spec" in (c["last_known_path"] or "").lower()]

    seen = already_shown if already_shown is not None else {}
    rendered: list[dict[str, Any]] = []

    # Rank by relevance to THIS query, then severity - not severity alone.
    # Sorting on severity only meant every memory attached to a selected symbol
    # was emitted whatever the question was: `run_git` carries seven memories,
    # so any query touching it returned all seven. Measured across four
    # unrelated queries, results overlapped 58-86%.
    scored_memories = sorted(
        ((-_memory_score(m, memory_relevance, terms, intent), m) for m in memories),
        key=lambda pair: pair[0])

    # Ordering alone was not enough. A capsule that emits every memory attached
    # to a symbol returns the same set whatever was asked - `run_git` carries
    # seven, so any query touching it returned all seven. Below the floor a
    # memory is named but not spelled out, so nothing is hidden and the budget
    # goes to what was actually asked about.
    ranked = [m for score, m in scored_memories if -score >= MEMORY_FLOOR][:6]
    demoted = [m for score, m in scored_memories if -score < MEMORY_FLOOR][:4]
    for memory in ranked:
        status = memory.get("anchor_status") or anchor_mod.ACTIVE
        memory_id = memory["memory_id"]

        if memory_id in seen:
            # Say it applies here, but do not pay for the text twice.
            rendered.append({
                "memory_id": memory_id, "kind": memory["kind"],
                "severity": memory["severity"], "anchor_status": status,
                "also_applies_here": True, "shown_at": seen[memory_id],
            })
            continue
        seen[memory_id] = symbol["symbol_path"]

        entry = {
            "memory_id": memory_id,
            "kind": memory["kind"],
            "severity": memory["severity"],
            "authority": memory["authority"],
            "text": (memory["body"] or "")[:600],
            "anchor_status": status,
            "confidence": round(memory.get("anchor_confidence") or memory["confidence"], 2),
        }
        # Never render a memory the anchor cannot vouch for as settled fact.
        if status in anchor_mod.STALE_STATUSES:
            entry["warning"] = _stale_note(conn, memory, status)
        rendered.append(entry)

    # Named, not spelled out. An agent can still ask for any of these by id.
    # Deduped like the quoted ones: a memory already named on another capsule
    # does not need naming again.
    for memory in demoted:
        if memory["memory_id"] in seen:
            continue
        seen[memory["memory_id"]] = symbol["symbol_path"]
        status = memory.get("anchor_status") or anchor_mod.ACTIVE
        entry = {
            "memory_id": memory["memory_id"], "kind": memory["kind"],
            "severity": memory["severity"],
            "anchor_status": status,
            "title": (memory.get("title") or "")[:120],
            "not_shown": "lower relevance to this query; memory(action='get') for the body",
        }
        # Demotion must never strip the staleness label. An unverified memory
        # rendered without its warning is exactly the failure the status
        # machine exists to prevent, and being brief is no excuse for it.
        if status in anchor_mod.STALE_STATUSES:
            entry["warning"] = _stale_note(conn, memory, status)
        rendered.append(entry)

    # Contracts with other repositories. Reported even when the other repo is
    # unreachable - that is a finding, not a reason to hide the dependency.
    contracts = crossrepo.edges_for(catalog, [symbol["symbol_id"]])

    # The causal spine, when there is one. This is what turns "a memory
    # mentions this" into "here is why this code exists".
    history = causal.why_does_this_exist(conn, symbol["symbol_id"])

    return {
        "symbol": symbol["symbol_path"],
        "kind": symbol["kind"],
        "path": symbol["last_known_path"],
        "lines": f"{symbol['line_start']}-{symbol['line_end']}",
        "signature": symbol["signature"],
        "status": symbol["status"],
        "upstream": [c["symbol_path"] for c in callers][:5],
        "downstream": [c["symbol_path"] for c in callees][:5],
        "tests": tests[:4],
        "memory": rendered,
        "broken_edges": [b for b in broken if b["status"] != "ACTIVE"][:3],
        "cross_repo": [
            {"kind": c["kind"], "repo": (c["target_repo"] or {}).get("name")
             or c["target"].get("repo_ref"),
             "entity": c["target"].get("symbol_path") or c["target"].get("name"),
             "status": c["status"], "reachable": c["reachable"]}
            for c in contracts
        ][:4],
        "why_it_exists": history,
    }


def _stale_note(conn: sqlite3.Connection, memory: dict[str, Any], status: str) -> str:
    verified = memory.get("anchor_verified_commit") or memory.get("last_verified_commit")
    if status == anchor_mod.ORPHANED:
        return "the code this was attached to no longer exists; treat as unverified history"
    if status == anchor_mod.NEEDS_REVIEW:
        return f"the anchored code changed since this was recorded (last verified at {verified or 'unknown commit'})"
    if status == anchor_mod.DRIFTED:
        return "re-anchored with reduced confidence after a move or rename; confirm it still applies"
    if status == anchor_mod.SUPERSEDED:
        return "superseded by a newer memory"
    return "not verified against the current code"


def _estimate_tokens(payload: Any) -> int:
    return max(1, len(jdump(payload) or "") // 4)


# ------------------------------------------------------------ problem detection

def detect_problems(conn: sqlite3.Connection, catalog: sqlite3.Connection,
                    symbol_ids: list[str]) -> list[dict[str, Any]]:
    """Targeted checks over the narrowed subgraph, never the whole workspace."""
    if not symbol_ids:
        return []
    placeholders = ",".join("?" for _ in symbol_ids)
    problems: list[dict[str, Any]] = []

    # A memory whose anchor drifted is the highest-value finding we have.
    for row in rows(conn.execute(
        f"SELECT a.*, m.kind, m.title, m.severity FROM anchors a JOIN memories m"
        f" ON m.memory_id = a.memory_id WHERE a.symbol_id IN ({placeholders})"
        f" AND a.status IN ('NEEDS_REVIEW','DRIFTED','ORPHANED') AND m.status='ACTIVE'",
        tuple(symbol_ids),
    )):
        problems.append({
            "severity": "high" if row["severity"] in ("critical", "high") else "medium",
            "kind": "stale_knowledge",
            "detail": f"{row['kind']} '{row['title']}' is {row['status']} at {row['symbol_path']}",
            "memory_id": row["memory_id"],
            "recommendation": "verify it still applies, then memory(action='verify')",
        })

    # Contradictions raise the question; they never decide it.
    for row in rows(conn.execute(
        f"SELECT e.*, m1.title AS a_title, m2.title AS b_title FROM memory_edges e"
        f" JOIN memories m1 ON m1.memory_id = e.from_id"
        f" JOIN memories m2 ON m2.memory_id = e.to_id"
        f" WHERE e.kind='CONTRADICTS' AND e.status='ACTIVE' AND m1.status='ACTIVE'"
        f" AND m2.status='ACTIVE' AND e.from_id IN ("
        f"   SELECT from_id FROM memory_edges WHERE to_id IN ({placeholders}))",
        tuple(symbol_ids),
    )):
        problems.append({
            "severity": "medium", "kind": "knowledge_conflict",
            "detail": f"'{row['a_title']}' may contradict '{row['b_title']}'",
            "confidence": row["confidence"],
            "recommendation": "these cannot both be universally true; narrow one's scope",
        })

    # A caller that appeared after the memory was last verified was never
    # considered by whoever wrote the warning.
    for row in rows(conn.execute(
        # The question is "did this caller appear AFTER the memory was last
        # verified", which is an ordering test. Comparing commit ids for
        # inequality answers a different question - "were these stamped at the
        # same commit" - and indexing only re-stamps files that changed, so on
        # a real store it was true for almost every pair: measured at 10
        # findings per investigation, crowding out every other diagnostic.
        # code_edges.created_at is when the call edge first appeared.
        # Compared against the memory's own last_verified_at, which only an
        # explicit memory(action='verify') updates - never the automatic
        # cascade. The anchor's timestamp cannot work here: ensure_indexed
        # verifies anchors in the same pass that creates the call edge, so the
        # two are always equal to the millisecond and the test can never fire.
        f"SELECT DISTINCT s.symbol_path, m.title, m.memory_id, e.created_at,"
        f" m.last_verified_at FROM code_edges e"
        f" JOIN symbols s ON s.symbol_id = e.from_id"
        f" JOIN anchors a ON a.symbol_id = e.to_id"
        f" JOIN memories m ON m.memory_id = a.memory_id"
        f" WHERE e.to_id IN ({placeholders}) AND e.kind='CALLS' AND e.status='ACTIVE'"
        f" AND s.status='ACTIVE' AND m.kind IN ('invariant','warning','contract')"
        f" AND m.status='ACTIVE' AND e.created_at IS NOT NULL"
        f" AND m.last_verified_at IS NOT NULL AND e.created_at > m.last_verified_at"
        f" LIMIT 6",
        tuple(symbol_ids),
    )):
        problems.append({
            "severity": "low", "kind": "unreviewed_caller",
            "detail": f"{row['symbol_path']} started calling code governed by"
                      f" '{row['title']}' after that memory was last verified",
            "memory_id": row["memory_id"],
        })

    # Active knowledge pointing at a tombstone.
    for row in rows(conn.execute(
        f"SELECT s.symbol_path, s.deleted_at_commit, m.title, m.memory_id FROM anchors a"
        f" JOIN symbols s ON s.symbol_id = a.symbol_id"
        f" JOIN memories m ON m.memory_id = a.memory_id"
        f" WHERE s.status='DELETED' AND m.status='ACTIVE' AND a.symbol_id IN ({placeholders})",
        tuple(symbol_ids),
    )):
        problems.append({
            "severity": "medium", "kind": "historical_implementation",
            "detail": f"'{row['title']}' was implemented by {row['symbol_path']},"
                      f" removed at {row['deleted_at_commit'] or 'an unknown commit'}",
            "memory_id": row["memory_id"],
        })

    # Migration leads recorded by the cascade but deliberately not acted on.
    for row in rows(conn.execute(
        f"SELECT e.*, s.symbol_path FROM memory_edges e LEFT JOIN symbols s ON s.symbol_id = e.to_id"
        f" WHERE e.kind='POSSIBLY_MIGRATED_TO' AND e.status='ACTIVE'"
        f" AND (e.to_id IN ({placeholders}) OR e.from_id IN ({placeholders})) LIMIT 6",
        (*symbol_ids, *symbol_ids),
    )):
        problems.append({
            "severity": "low", "kind": "migration_candidate",
            "detail": f"code may have migrated to {row['symbol_path']}"
                      f" (similarity {row['confidence']}); memories were not transferred",
            "recommendation": "confirm with memory(action='reanchor') if correct",
        })

    # A contract pointing at a repository we can no longer reach is exactly the
    # kind of silent breakage the graph exists to surface (PLAN.md section 15).
    for edge in crossrepo.edges_for(catalog, symbol_ids):
        if edge["reachable"]:
            continue
        target = edge["target"]
        where = (edge["target_repo"] or {}).get("name") or target.get("repo_ref") or "another repo"
        state = (edge["target_repo"] or {}).get("status") or edge["status"]
        problems.append({
            "severity": "medium" if edge["kind"] in ("PROVIDES_CONTRACT", "CONSUMES_CONTRACT")
                        else "low",
            "kind": "unverifiable_contract",
            "detail": edge["kind"] + " with " + str(target.get("name") or "?") + " in " + where
                      + " cannot be checked right now (" + str(state) + ")",
            "recommendation": "the dependency still holds; open that repository to verify it",
        })

    # The graph-derived detectors from PLAN.md: bypassed wrappers, untested
    # invariants, deprecated symbols with live callers, unguarded equivalents,
    # implementations drifted from a decision.
    problems.extend(diagnostics.run_all(conn, symbol_ids))

    order = {"high": 0, "medium": 1, "low": 2}
    problems.sort(key=lambda p: order.get(p["severity"], 3))
    return problems[:14]


# ------------------------------------------------------------------ entry point

def investigate(conn: sqlite3.Connection, catalog: sqlite3.Connection, root: Path,
                query: str, intent: str | None = None, depth: int = 2,
                budget: int = 9000, find_problems: bool = True,
                commit: str | None = None, cross_repos: bool = False,
                repo_id: str | None = None) -> dict[str, Any]:
    """One call, many engines, one budgeted answer."""
    chosen = intent if intent in INTENTS else infer_intent(query)
    weights = INTENT_WEIGHTS[chosen]
    terms = _terms(query)

    seeds = _seed_symbols(conn, terms)
    _boost_explicit_paths(conn, query, seeds)
    memory_hits = _seed_memories(conn, terms)

    # Memories that matched textually pull their anchored symbols in with them,
    # so a warning can surface even when the query never names the code.
    if memory_hits:
        placeholders = ",".join("?" for _ in memory_hits)
        for row in rows(conn.execute(
            f"SELECT symbol_id, memory_id FROM anchors WHERE memory_id IN ({placeholders})"
            f" AND symbol_id IS NOT NULL", tuple(memory_hits)
        )):
            seeds.setdefault(row["symbol_id"], {"lex": 0.0, "sym": 0.0})
            seeds[row["symbol_id"]]["lex"] = max(
                seeds[row["symbol_id"]]["lex"], memory_hits[row["memory_id"]] * 0.8
            )

    proximity = _expand(conn, seeds, depth)
    candidate_ids = list(proximity.keys())[:400]
    if not candidate_ids:
        return _empty_result(query, chosen, terms)

    placeholders = ",".join("?" for _ in candidate_ids)
    symbols = {
        r["symbol_id"]: r
        for r in rows(conn.execute(
            f"SELECT * FROM symbols WHERE symbol_id IN ({placeholders})", tuple(candidate_ids)
        ))
    }
    memories_by_symbol = _memories_for_symbols(conn, candidate_ids)

    touched = _recency_map(root, commit) if weights["time"] > 0.2 else {}

    # Specificity is a per-symbol COUNT query; hoist it into one grouped read
    # rather than issuing it twice per candidate inside the scoring loop.
    specificity = _specificity_map(conn, candidate_ids)

    scored: list[tuple[float, dict[str, Any], list[dict[str, Any]], dict[str, float]]] = []
    for symbol_id, symbol in symbols.items():
        seed = seeds.get(symbol_id, {"lex": 0.0, "sym": 0.0})
        attached = memories_by_symbol.get(symbol_id, [])

        severity = max((SEVERITY_SCORE.get(m["severity"], 0.3) for m in attached), default=0.0)
        memory_signal = min(1.0, 0.35 * len(attached))
        test_signal = 1.0 if any(m["kind"] == "test_evidence" for m in attached) else 0.0
        recency = _recency(touched, symbol["last_known_path"])

        stale = max((STALE_PENALTY.get(m.get("anchor_status") or anchor_mod.ACTIVE, 0.0)
                     for m in attached), default=0.0)

        score = (
            weights["lex"] * seed["lex"]
            + weights["sym"] * seed["sym"]
            + weights["graph"] * proximity.get(symbol_id, 0.0)
            + weights["sev"] * severity
            + weights["time"] * recency
            + weights["test"] * test_signal
            + weights["mem"] * memory_signal
            - specificity.get(symbol_id, 0.0)
            - 0.25 * stale
            - (0.4 if symbol["status"] != "ACTIVE" else 0.0)
        )
        breakdown = {
            "lexical": round(seed["lex"], 3), "symbol": round(seed["sym"], 3),
            "graph": round(proximity.get(symbol_id, 0.0), 3), "severity": round(severity, 3),
            "recency": round(recency, 3), "memory": round(memory_signal, 3),
            "specificity_discount": round(specificity.get(symbol_id, 0.0), 3),
            "stale_penalty": round(0.25 * stale, 3),
        }
        scored.append((score, symbol, attached, breakdown))

    scored.sort(key=lambda item: -item[0])

    capsules: list[dict[str, Any]] = []
    shown: dict[str, str] = {}
    used = 0
    for score, symbol, attached, breakdown in scored:
        if score <= 0.05:
            break
        capsule = _capsule(conn, catalog, symbol, attached, chosen, shown,
                           memory_hits, terms)
        capsule["score"] = round(score, 3)
        capsule["score_breakdown"] = breakdown
        cost = _estimate_tokens(capsule)
        if used + cost > budget and capsules:
            break
        capsules.append(capsule)
        used += cost
        if len(capsules) >= 12:
            break

    top_ids = [
        symbol["symbol_id"]
        for _, symbol, _, _ in scored[:max(len(capsules), 8)]
    ]
    # Record what this answer actually showed, so ranking learns from use.
    note_surfaced(conn, [m["memory_id"] for c in capsules for m in c["memory"]])

    problems = detect_problems(conn, catalog, top_ids) if find_problems else []

    unanchored = rows(conn.execute(
        "SELECT m.memory_id, m.kind, m.title, m.severity FROM memories m"
        " JOIN anchors a ON a.memory_id = m.memory_id"
        " WHERE a.status='ORPHANED' AND m.status='ACTIVE' LIMIT 5"
    ))

    investigation_id = ids.new_id(ids.INVESTIGATION)
    frontier = {"seeds": list(seeds.keys())[:200], "proximity": {k: round(v, 3) for k, v in
                list(proximity.items())[:200]}, "terms": terms}
    with write_tx(conn):
        conn.execute(
            "INSERT INTO investigations (investigation_id, query, intent, commit_sha, frontier,"
            " results, created_at) VALUES (?,?,?,?,?,?,?)",
            (investigation_id, query, chosen, commit, jdump(frontier),
             jdump([c["symbol"] for c in capsules]), now()),
        )

    return {
        "investigation_id": investigation_id,
        "query": query,
        "intent": chosen,
        "terms": terms,
        "capsules": capsules,
        "problems": problems,
        "unanchored_knowledge": unanchored,
        "considered": len(scored),
        "cross_repo": _cross_repo_context(catalog, top_ids, repo_id) if cross_repos else None,
        "budget": {"limit": budget, "used_estimate": used},
        "note": "memories carry an anchor_status; anything not ACTIVE is unverified against current code",
    }



def _cross_repo_context(catalog: sqlite3.Connection, symbol_ids: list[str],
                        repo_id: str | None) -> dict[str, Any]:
    """Follow contract edges into other repositories.

    Reads the *catalog* only - stubs and the memory registry - never another
    repository's store. That store may be missing, on an unmounted drive, or
    purged, and a cross-repo answer must not depend on it (PLAN.md section 8).
    What survives is the snapshot taken when the edge was written, which is
    enough to say what the dependency is and whether it can be checked now.
    """
    edges = crossrepo.edges_for(catalog, symbol_ids)
    if not edges:
        return {"linked_repos": [], "contracts": [], "note": "no cross-repository contracts here"}

    by_repo: dict[str, dict[str, Any]] = {}
    for edge in edges:
        target = edge.get("target_repo") or {}
        key = target.get("repo_id") or (edge["target"] or {}).get("repo_ref") or "unknown"
        entry = by_repo.setdefault(key, {
            "repo_id": target.get("repo_id"), "name": target.get("name") or key,
            "status": target.get("status") or edge["status"],
            "reachable": edge["reachable"], "contracts": [], "memories": [],
        })
        entry["contracts"].append({
            "kind": edge["kind"],
            "entity": (edge["target"] or {}).get("symbol_path")
                      or (edge["target"] or {}).get("name"),
            "status": edge["status"],
        })

    # Knowledge from the other side, via the central memory registry. This is
    # the payoff: "the repo you depend on has an ACTIVE warning about it".
    for key, entry in by_repo.items():
        if not entry["repo_id"]:
            continue
        entry["memories"] = rows(catalog.execute(
            "SELECT memory_id, kind, severity, summary FROM memory_registry"
            " WHERE repo_id = ? AND status = 'ACTIVE'"
            " AND kind IN ('invariant','warning','contract','security')"
            " ORDER BY CASE severity WHEN 'critical' THEN 0 WHEN 'high' THEN 1"
            " WHEN 'medium' THEN 2 ELSE 3 END LIMIT 5", (entry["repo_id"],)))

    return {
        "linked_repos": list(by_repo.values()),
        "unreachable": [r["name"] for r in by_repo.values() if not r["reachable"]],
        "note": "contract targets are rendered from the snapshot taken when the"
                " edge was written, so they stay readable even when that"
                " repository is unavailable",
    }

def _empty_result(query: str, intent: str, terms: list[str]) -> dict[str, Any]:
    return {
        "investigation_id": None, "query": query, "intent": intent, "terms": terms,
        "capsules": [], "problems": [], "unanchored_knowledge": [], "considered": 0,
        "note": "nothing matched; the repository may not be indexed yet "
                "(workspace action='reindex') or the query terms may not appear in this codebase",
    }


def expand(conn: sqlite3.Connection, catalog: sqlite3.Connection, investigation_id: str,
           focus: str, budget: int = 6000) -> dict[str, Any]:
    """Continue an investigation without restarting from zero."""
    saved = one(conn.execute("SELECT * FROM investigations WHERE investigation_id=?",
                             (investigation_id,)))
    if saved is None:
        return {"ok": False, "error": f"unknown investigation {investigation_id}"}

    frontier = jload(saved["frontier"], {}) or {}
    seed_ids = frontier.get("seeds", [])
    focus_terms = _terms(focus)

    # Rank by how specifically the focus names each candidate. A loose
    # substring hit must not outrank an exact symbol match, or "expand on
    # refresh_session" returns whatever the database happened to yield first.
    scored: list[tuple[float, dict[str, Any]]] = []
    if seed_ids:
        placeholders = ",".join("?" for _ in seed_ids)
        lowered = [t.lower() for t in focus_terms]
        focus_lower = focus.strip().lower()
        for symbol in rows(conn.execute(
            f"SELECT * FROM symbols WHERE symbol_id IN ({placeholders})", tuple(seed_ids)
        )):
            path = (symbol["symbol_path"] or "").lower()
            name = (symbol["name"] or "").lower()
            file_path = (symbol["last_known_path"] or "").lower()
            score = 0.0
            if path == focus_lower or name == focus_lower:
                score = 4.0
            elif name in lowered or path in lowered:
                score = 3.0
            elif any(term == name or term == path for term in lowered):
                score = 3.0
            elif any(term in name for term in lowered):
                score = 2.0
            elif any(term in path for term in lowered):
                score = 1.5
            elif any(term in file_path for term in lowered):
                score = 0.5
            if score > 0:
                scored.append((score, symbol))

    scored.sort(key=lambda item: (-item[0], item[1]["symbol_path"]))
    matched = [symbol for _, symbol in scored]

    if not matched:
        # The focus points outside the saved frontier; widen once rather than
        # returning nothing.
        return {
            "ok": True, "investigation_id": investigation_id, "focus": focus,
            "capsules": [], "note": "focus not present in the saved frontier; "
                                    "run investigate() again with the narrower query",
        }

    memories_by_symbol = _memories_for_symbols(conn, [s["symbol_id"] for s in matched])
    capsules = []
    used = 0
    for symbol in matched[:8]:
        capsule = _capsule(conn, catalog, symbol, memories_by_symbol.get(symbol["symbol_id"], []),
                           saved["intent"] or "understand")
        cost = _estimate_tokens(capsule)
        if used + cost > budget and capsules:
            break
        capsules.append(capsule)
        used += cost

    return {"ok": True, "investigation_id": investigation_id, "focus": focus,
            "capsules": capsules, "budget": {"limit": budget, "used_estimate": used}}
