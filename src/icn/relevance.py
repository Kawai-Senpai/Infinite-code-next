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
    for a in anchors:
        if a.get("symbol_path") in named_symbols and a.get("file_path"):
            named_files.add(a["file_path"])
    return {"files": sorted(named_files), "symbols": named_symbols, "broad": True}
