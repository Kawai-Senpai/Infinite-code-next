"""Parsing, fingerprints, incremental indexing and tombstones."""

from __future__ import annotations

from pathlib import Path

import pytest

from icn import parsing
from icn import workspace as ws_mod
from icn.db import one, rows

PY_SOURCE = b'''
class Foo:
    def bar(self, x):
        # a comment
        y = helper(x)
        return y + 1


def helper(x):
    return x * 2
'''


def test_reformatting_does_not_change_the_content_fingerprint():
    original = parsing.parse_file(Path("t.py"), PY_SOURCE, "python")
    reformatted = parsing.parse_file(Path("t.py"), PY_SOURCE.replace(
        b"        # a comment\n", b"\n        # an entirely different comment\n\n"
    ), "python")
    assert {s.symbol_path: s.content_fingerprint for s in original} == \
           {s.symbol_path: s.content_fingerprint for s in reformatted}


def test_rename_keeps_the_skeleton_but_changes_the_content():
    original = {s.symbol_path: s for s in parsing.parse_file(Path("t.py"), PY_SOURCE, "python")}
    renamed = {s.symbol_path: s for s in
               parsing.parse_file(Path("t.py"), PY_SOURCE.replace(b"def bar", b"def baz"), "python")}
    assert original["Foo.bar"].skeleton_fingerprint == renamed["Foo.baz"].skeleton_fingerprint
    assert original["Foo.bar"].content_fingerprint != renamed["Foo.baz"].content_fingerprint


def test_logic_change_changes_the_skeleton():
    changed = PY_SOURCE.replace(b"        return y + 1", b"        return None")
    original = {s.symbol_path: s for s in parsing.parse_file(Path("t.py"), PY_SOURCE, "python")}
    edited = {s.symbol_path: s for s in parsing.parse_file(Path("t.py"), changed, "python")}
    assert original["Foo.bar"].skeleton_fingerprint != edited["Foo.bar"].skeleton_fingerprint


def test_nested_symbols_get_dotted_paths():
    found = {s.symbol_path for s in parsing.parse_file(Path("t.py"), PY_SOURCE, "python")}
    assert {"Foo", "Foo.bar", "helper"} <= found


@pytest.mark.parametrize("lang,suffix,source,expected", [
    ("javascript", ".js", b"class A { run(x) { return x; } }\nfunction go(y) { return y; }", "go"),
    ("go", ".go", b"package main\n\nfunc Handle(x int) int {\n\treturn x\n}\n", "Handle"),
    ("rust", ".rs", b"pub fn handle(x: i32) -> i32 { x }\n", "handle"),
])
def test_other_languages_parse(lang, suffix, source, expected):
    if parsing.get_parser(lang) is None:
        pytest.skip(f"grammar for {lang} unavailable")
    found = {s.name for s in parsing.parse_file(Path(f"t{suffix}"), source, lang)}
    assert expected in found


def test_call_edges_resolve_regardless_of_walk_order(workspace):
    """The two-pass resolver must not lose edges that point forward.

    A test file is frequently indexed before the module it imports, and
    resolving call names inline drops every such edge silently.
    """
    target = one(workspace.store.execute(
        "SELECT symbol_id FROM symbols WHERE symbol_path='refresh_session'"))
    callers = rows(workspace.store.execute(
        "SELECT s.name FROM code_edges e JOIN symbols s ON s.symbol_id=e.from_id"
        " WHERE e.to_id=? AND e.kind='CALLS'", (target["symbol_id"],)))
    names = {c["name"] for c in callers}
    assert "post_refresh" in names
    assert "test_parallel_refresh_regression" in names


def test_unchanged_file_is_not_reparsed(workspace, project):
    report = ws_mod.ensure_indexed(workspace)
    assert report["files_indexed"] == 0


def test_deleting_a_file_tombstones_its_symbols(workspace, project):
    project.delete("api.py")
    ws_mod.ensure_indexed(workspace)

    file_row = one(workspace.store.execute("SELECT * FROM files WHERE path='api.py'"))
    assert file_row["status"] == "DELETED"
    symbol = one(workspace.store.execute("SELECT * FROM symbols WHERE symbol_path='post_refresh'"))
    assert symbol["status"] == "DELETED"
    assert symbol["last_known_path"] == "api.py"


def test_excluded_directories_are_skipped(project):
    project.write("node_modules/pkg/index.js", "function nope() { return 1; }")
    project.write("__pycache__/x.py", "def nope2():\n    return 1\n")
    found = {p.name for p in ws_mod.indexer_mod.walk_source_files(project.root)} \
        if hasattr(ws_mod, "indexer_mod") else None
    from icn.indexer import walk_source_files
    found = {p.name for p in walk_source_files(project.root)}
    assert "index.js" not in found
    assert "x.py" not in found
    assert "auth.py" in found


def test_index_state_reports_tombstones_separately(workspace, project):
    project.delete("api.py")
    report = ws_mod.ensure_indexed(workspace)
    assert report["symbols_deleted"] >= 1
    assert report["symbols_active"] >= 1
