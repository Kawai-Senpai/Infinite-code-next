"""Which of a memory's anchors its claim is actually about.

record() anchors every memory in an event to every file and symbol the event
named, so one record touching ten files yields an invariant about hooks.run
anchored to compiler.py too. Measured on this repository: rule promotion
labelled "hooks.run must always exit 0" as a rule for compiler.py
_find_duplicate. A narrowly anchored memory is taken at its word; a broadly
anchored one only applies where its own text points.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any

BROAD = 3


def _plain_word(name: str) -> bool:
    """`run`, `hooks`, `search`: names that are also ordinary English words."""
    return name.isalpha() and name.islower()


def _mentions(text: str, name: str) -> bool:
    if not name:
        return False
    bounded = r"(?<![A-Za-z0-9_.])" + re.escape(name)
    if _plain_word(name):
        # Measured: "A run with exit 0" matched the symbol run, and "trust new
        # hooks" matched hooks.py. A plain word counts only when written as
        # code: a call, an attribute, or in backticks.
        return re.search(bounded + r"(?=\(|\.[A-Za-z_])|`" + re.escape(name) + r"`", text) is not None
    return re.search(bounded + r"(?![A-Za-z0-9_])", text) is not None


def file_names(path: str) -> list[str]:
    p = PurePosixPath(path)
    return [p.name, p.stem] if p.stem not in ("__init__", "index", "main", "mod") else [path, p.name]


def symbol_names(symbol_path: str) -> list[str]:
    parts = [s for s in re.split(r"[.:#]+", symbol_path or "") if s]
    return [symbol_path, parts[-1]] if parts else []


def claim_text(memory: dict[str, Any]) -> str:
    """The claim alone, without the event context composed into the body."""
    text = memory.get("claim") or memory.get("body") or ""
    for marker in ("\n\nRecorded while: ", "\n\nWhy: ", "\n\nChanged: ", "\n\nApplies to: "):
        text = text.split(marker, 1)[0]
    return text


def narrowed(conn: Any, memory_ids: list[str],
             keep_unnamed: bool = False) -> dict[str, dict[str, Any]]:
    """{memory_id: focus} for broad memories whose claim names specific code.

    Only these need filtering. A narrow memory keeps every anchor. A broad one
    whose claim names nothing it is anchored to also keeps every anchor in
    search, where hiding it would lose knowledge rather than noise; hooks,
    which volunteer knowledge unasked, pass keep_unnamed=True to receive its
    empty focus and so volunteer it nowhere.
    """
    ids = sorted(set(memory_ids))
    if not ids:
        return {}
    anchors: dict[str, list[dict[str, Any]]] = {}
    for start in range(0, len(ids), 400):
        part = ids[start:start + 400]
        marks = ",".join("?" for _ in part)
        for memory_id, file_path, symbol_path in conn.execute(
                f"SELECT memory_id, file_path, symbol_path FROM anchors WHERE memory_id IN ({marks})",
                tuple(part)):
            anchors.setdefault(memory_id, []).append(
                {"file_path": file_path, "symbol_path": symbol_path})
    broad = [m for m, a in anchors.items()
             if len({x["file_path"] for x in a if x["file_path"]}) > BROAD]
    out: dict[str, dict[str, Any]] = {}
    for start in range(0, len(broad), 400):
        part = broad[start:start + 400]
        marks = ",".join("?" for _ in part)
        for memory_id, claim, body in conn.execute(
                f"SELECT memory_id, claim, body FROM memories WHERE memory_id IN ({marks})",
                tuple(part)):
            focused = focus(claim_text({"claim": claim, "body": body}), anchors[memory_id])
            if keep_unnamed or focused["files"] or focused["symbols"]:
                out[memory_id] = focused
    return out


def applies(focused: dict[str, Any], file_path: str | None, symbol_path: str | None) -> bool:
    """Whether a narrowed memory applies to this symbol (or, with no symbol, file)."""
    if symbol_path is None:
        return bool(file_path) and file_path in focused["files"]
    if symbol_path in focused["symbols"]:
        return True
    return bool(file_path) and file_path in focused.get("whole_files", [])


def focus(claim: str, anchors: list[dict[str, Any]]) -> dict[str, Any]:
    """{files, symbols, broad}: the anchors this claim is about.

    Narrow memories (anchored to at most BROAD files) keep every anchor.
    Broad ones keep only what the claim names, and may keep nothing.
    """
    files = sorted({a["file_path"] for a in anchors if a.get("file_path")})
    symbols = sorted({a["symbol_path"] for a in anchors if a.get("symbol_path")})
    if len(files) <= BROAD:
        return {"files": files, "symbols": symbols, "broad": False}
    named_symbols = [s for s in symbols if any(_mentions(claim, n) for n in symbol_names(s))]
    named_files = {f for f in files if any(_mentions(claim, n) for n in file_names(f))}
    whole_files = sorted(named_files)
    for a in anchors:
        if a.get("symbol_path") in named_symbols and a.get("file_path"):
            named_files.add(a["file_path"])
    # `files` includes the files of named symbols, which is right for "does
    # this apply to the file being edited". `whole_files` are the files the
    # claim names outright, which is what makes it apply to every symbol in one.
    return {"files": sorted(named_files), "symbols": named_symbols, "broad": True,
            "whole_files": whole_files}
