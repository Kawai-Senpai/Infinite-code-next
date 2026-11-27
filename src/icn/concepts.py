"""A repository's reviewed vocabulary, pinned to indexed source symbols."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TypeGuard

FILENAME = ".icn-concepts.json"
MAX_BYTES = 128 * 1024
MAX_CONCEPTS = 128
MAX_ALIASES = 12
MAX_CONTEXT = 8
MAX_TARGETS = 3
MAX_MATCHES = 4
FINGERPRINT = re.compile(r"[0-9a-f]{32}")
WORDS = re.compile(r"[^\W_]+")
QUALIFIED = r"\w+(?:(?:\.|::)\w+)+"
EXPLICIT = re.compile(
    rf"`([^`\s]+)`|(?<![\w.])([\w./\\-]+[/\\][\w./\\-]+|{QUALIFIED}|\w+_\w+|\w+(?=\(\)))(?![\w.])"
)


@dataclass(frozen=True)
class Target:
    path: str
    symbol: str
    fingerprint: str


@dataclass(frozen=True)
class Concept:
    name: str
    phrases: dict[str, frozenset[str]]
    targets: tuple[Target, ...]


def normalized_phrase(text: str) -> str:
    return " ".join(WORDS.findall(text.casefold()))


def qualified_name(query: str) -> str | None:
    candidate = query.strip().strip("`").removesuffix("()")
    return candidate if re.fullmatch(QUALIFIED, candidate) else None


def _bounded_text(value: object, limit: int) -> TypeGuard[str]:
    return isinstance(value, str) and bool(value.strip()) and len(value) <= limit


def _relative_path(value: object) -> TypeGuard[str]:
    return (_bounded_text(value, 240) and "\\" not in value and ":" not in value
            and all(part not in ("", ".", "..") for part in value.split("/")))


def _target(value: object) -> Target:
    if isinstance(value, dict):
        path, symbol, fingerprint = (value.get(field) for field in ("path", "symbol", "fingerprint"))
        if (_relative_path(path) and _bounded_text(symbol, 160)
                and isinstance(fingerprint, str) and FINGERPRINT.fullmatch(fingerprint)):
            return Target(path, symbol, fingerprint)
    raise ValueError("targets need a repository-relative path, exact symbol, and 32-digit fingerprint")


def _alias(value: object) -> tuple[str, frozenset[str]]:
    phrase = value.get("phrase") if isinstance(value, dict) else value
    if not _bounded_text(phrase, 100) or " " not in (key := normalized_phrase(phrase)):
        raise ValueError("aliases must have two or more words and at most 100 characters")
    if not isinstance(value, dict):
        return key, frozenset()

    context = value.get("requires_any")
    if not isinstance(context, list) or not 1 <= len(context) <= MAX_CONTEXT:
        raise ValueError(f"requires_any must contain between 1 and {MAX_CONTEXT} context phrases")
    terms = set()
    for term in context:
        if not _bounded_text(term, 100) or not (normalized := normalized_phrase(term)):
            raise ValueError("context phrases must contain words and have at most 100 characters")
        padded = f" {normalized} "
        if padded in f" {key} ":
            raise ValueError("context phrases must add words outside the alias")
        terms.add(padded)
    return key, frozenset(terms)


def _load(root: Path) -> list[Concept] | None:
    path = root / FILENAME
    try:
        with path.open("rb") as stream:
            source = stream.read(MAX_BYTES + 1)
    except FileNotFoundError:
        return None
    if len(source) > MAX_BYTES:
        raise ValueError(f"{FILENAME} exceeds {MAX_BYTES} bytes")
    data = json.loads(source)
    definitions = data.get("concepts") if isinstance(data, dict) else None
    if not isinstance(definitions, list) or len(definitions) > MAX_CONCEPTS:
        raise ValueError(f"provide a concepts array with at most {MAX_CONCEPTS} entries")

    concepts: list[Concept] = []
    names: set[str] = set()
    for number, entry in enumerate(definitions, 1):
        try:
            if not isinstance(entry, dict):
                raise ValueError("expected a concept object")
            name, aliases, targets = entry.get("name"), entry.get("aliases", []), entry.get("targets")
            if not _bounded_text(name, 100):
                raise ValueError("provide a name of up to 100 characters")
            name_key = normalized_phrase(name)
            if not name_key or name_key in names:
                raise ValueError("concept names must be unique and contain words")
            if not isinstance(aliases, list) or len(aliases) > MAX_ALIASES:
                raise ValueError(f"provide an aliases array with at most {MAX_ALIASES} entries")
            phrases: dict[str, frozenset[str]] = {}
            for alias in aliases:
                phrase, context = _alias(alias)
                if phrases.setdefault(phrase, context) != context:
                    raise ValueError(f"conflicting context for alias {phrase!r}")
            if " " in name_key:
                phrases.setdefault(name_key, frozenset())
            if not phrases:
                raise ValueError("provide a multiword name or alias")
            if not isinstance(targets, list) or not 1 <= len(targets) <= MAX_TARGETS:
                raise ValueError(f"provide between 1 and {MAX_TARGETS} targets")
            concepts.append(Concept(name.strip(), phrases, tuple(dict.fromkeys(map(_target, targets)))))
            names.add(name_key)
        except ValueError as exc:
            raise ValueError(f"concept {number}: {exc}") from exc
    return concepts


def _symbols(conn: sqlite3.Connection, path: str, symbol: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT symbol_id, content_fingerprint, line_start, line_end FROM symbols"
        " WHERE symbol_path=? AND last_known_path=? AND status='ACTIVE' LIMIT 2",
        (symbol, path),
    ).fetchall()


def _target_evidence(conn: sqlite3.Connection, target: Target, index_ready: bool) -> tuple[dict, str | None]:
    evidence = {"path": target.path, "symbol": target.symbol}
    if not index_ready:
        return {**evidence, "status": "index_incomplete"}, None
    candidates = _symbols(conn, target.path, target.symbol)
    if len(candidates) != 1:
        return {**evidence, "status": "ambiguous" if candidates else "unresolved"}, None
    symbol_id, fingerprint, first, last = candidates[0]
    evidence["lines"] = [first, last]
    if fingerprint != target.fingerprint:
        return {**evidence, "status": "needs_review"}, None
    return {**evidence, "status": "grounded"}, symbol_id


def ground(conn: sqlite3.Connection, root: Path, query: str, *,
           index_ready: bool) -> tuple[set[str], dict | None]:
    """Find exact whole-phrase matches and resolve their reviewed targets."""
    try:
        concepts = _load(root)
    except (OSError, ValueError, RecursionError) as exc:
        return set(), {"source": FILENAME, "status": "invalid", "error": str(exc)[:180], "matches": []}
    if concepts is None:
        return set(), None

    text = f" {normalized_phrase(query[:8192])} "
    owners: dict[str, list[Concept]] = defaultdict(list)
    for concept in concepts:
        for phrase, context in concept.phrases.items():
            if f" {phrase} " in text and (not context or any(term in text for term in context)):
                owners[phrase].append(concept)

    matched = []
    accounted_for: set[str] = set()
    grounded_ids: set[str] = set()
    seen: set[str] = set()
    for phrase in sorted(owners, key=lambda value: (-len(value.split()), -len(value), value)):
        if len(matched) == MAX_MATCHES:
            break
        entries = owners[phrase]
        if len(entries) > 1:
            matched.append({"phrase": phrase, "status": "ambiguous_phrase",
                            "concepts": [entry.name for entry in entries[:MAX_MATCHES]]})
            continue
        concept = entries[0]
        if concept.name in seen:
            continue
        seen.add(concept.name)
        targets = []
        for target in concept.targets:
            evidence, symbol_id = _target_evidence(conn, target, index_ready)
            targets.append(evidence)
            if symbol_id is not None:
                grounded_ids.add(symbol_id)
                accounted_for.update(phrase.split())
        matched.append({"concept": concept.name, "phrase": phrase, "targets": targets})

    return grounded_ids, {
        "source": FILENAME,
        "status": "matched" if matched else "no_match",
        "matches": matched,
    } | ({"grounded_words": sorted(accounted_for)} if accounted_for else {})


def explicitly_names_code(conn: sqlite3.Connection, query: str, *, include_files: bool = True) -> bool:
    """Protect explicit symbol/path lookup when a glossary phrase also matches."""
    if qualified_name(query):
        return True
    stripped = query.strip().strip("`")
    names = [a or b for a, b in EXPLICIT.findall(query[:8192])]
    if stripped and not any(char.isspace() for char in stripped):
        names.insert(0, stripped)
    for name in dict.fromkeys(names[:12]):
        name = name.removesuffix("()").replace("\\", "/")
        if conn.execute(
            "SELECT 1 FROM symbols WHERE status='ACTIVE' AND (symbol_path=? OR name=?) LIMIT 1",
            (name, name),
        ).fetchone():
            return True
        if include_files and ("/" in name or "." in name) and conn.execute(
            "SELECT 1 FROM files WHERE path=? AND status='ACTIVE'", (name,),
        ).fetchone():
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    """Print the reviewed target fields for a symbol after source review."""
    from . import workspace

    parser = argparse.ArgumentParser(prog="icn concept-anchor", description=main.__doc__)
    parser.add_argument("path", help="Repository-relative source path using forward slashes")
    parser.add_argument("symbol", help="Exact qualified symbol from the index")
    parser.add_argument("--root", help="Repository containing the symbol (defaults to cwd)")
    args = parser.parse_args(argv)
    if not _relative_path(args.path):
        parser.error("path must be repository-relative and use forward slashes")
    if not _bounded_text(args.symbol, 160):
        parser.error("symbol must be nonempty and at most 160 characters")
    current = workspace.open_workspace(args.root)
    try:
        report = workspace.ensure_indexed(current)
        if report["index_state"] != "ready":
            print("The index is incomplete. Retry when workspace open reports ready.", file=sys.stderr)
            return 1
        symbols = _symbols(current.store, args.path, args.symbol)
        if len(symbols) != 1 or not symbols[0][1]:
            print("Expected one active, fingerprinted symbol at that path.", file=sys.stderr)
            return 1
        print(json.dumps({"path": args.path, "symbol": args.symbol, "fingerprint": symbols[0][1]}))
        return 0
    finally:
        current.close()
