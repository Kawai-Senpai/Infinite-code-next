"""Per-language extraction: symbols, imports, decorators, entry points.

Every other graph test in this suite is written in Python, so a language whose
extraction was wrong stayed green forever. These are the shapes that actually
differ between grammars, and each one here failed before it was fixed:

    * JS/TS import statements name the module in a `source` field; sweeping the
      subtree instead reported the imported NAMES as modules.
    * CommonJS states dependencies with a function call no grammar marks.
    * Go declares a method's type in a receiver, and its import paths carry a
      module prefix no directory on disk has.
    * Rust writes `#[test]` as a preceding sibling, `mod x;` as a file
      reference, and `use a::b::item` with an item on the end.
    * Java and Kotlin fold annotations in with access modifiers.
    * Kotlin has no `name` field, so a name search fell through to the type.
    * C++ names a method's class in the declarator of an out-of-line definition.

They are parse-level where a parse proves it, and end-to-end through the
indexer where resolution is the thing under test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from icn import flows, parsing
from icn.db import rows


def parse(lang: str, suffix: str, source: str):
    if parsing.get_parser(lang) is None:
        pytest.skip(f"grammar for {lang} unavailable")
    return parsing.parse_file(Path(f"sample{suffix}"), source.encode(), lang)


def specs(lang: str, source: str) -> list[dict]:
    if parsing.get_parser(lang) is None:
        pytest.skip(f"grammar for {lang} unavailable")
    return parsing.file_import_specs(source.encode(), lang)


def modules(lang: str, source: str) -> set[str]:
    return {s["module"] for s in specs(lang, source)}


def paths_of(symbols) -> set[str]:
    return {s.symbol_path for s in symbols}


def classify(symbol, path: str):
    return flows.classify(
        {"name": symbol.name, "symbol_path": symbol.symbol_path,
         "lang": symbol.lang, "decorators": json.dumps(symbol.decorators)},
        path)


# --------------------------------------------------------------- JS/TS imports

def test_a_named_import_does_not_invent_a_module_per_imported_name():
    """`import { a, b } from "./m"` imports one module, not three.

    The subtree sweep took the import specifiers as modules too, so every named
    import produced `extern:a` and `extern:b` alongside the real edge. Those are
    indistinguishable from genuine third-party dependencies once stored.
    """
    assert modules("typescript",
                   'import { hashToken, other } from "./crypto";\n') == {"./crypto"}


def test_a_type_only_import_is_deferred_and_still_names_its_source():
    found = specs("typescript", 'import type { User } from "./types";\n')
    assert [(s["module"], s.get("deferred")) for s in found] == [("./types", True)]


def test_a_default_and_namespace_import_name_the_module():
    assert modules("typescript",
                   'import express from "express";\nimport * as ns from "./ns";\n') \
        == {"express", "./ns"}


def test_commonjs_require_is_an_import():
    """Nothing in the grammar marks a require, so a require-only file had none."""
    assert modules("javascript", 'const { a } = require("./legacy");\n') == {"./legacy"}


def test_a_require_inside_a_function_is_deferred():
    found = specs("javascript", 'function go() { return require("./lazy"); }\n')
    assert [(s["module"], s.get("deferred")) for s in found] == [("./lazy", True)]


def test_a_computed_require_names_no_module_rather_than_a_fragment():
    assert modules("javascript", 'const m = require(base + "/x");\n') == set()


# ------------------------------------------------------------- JS/TS symbols

def test_an_arrow_function_binding_is_a_symbol():
    """`export const Panel = () => ...` is how React declares a component.

    Only `function` declarations were collected, so a component file written
    this way produced no symbols at all - nothing called it, and nothing it
    called was reachable from it.
    """
    found = paths_of(parse("tsx", ".tsx",
                           "export const Panel = () => <div/>;\n"
                           "const Bar = function () { return 1; };\n"))
    assert {"Panel", "Bar"} <= found


def test_a_plain_value_binding_is_not_a_symbol():
    assert paths_of(parse("typescript", ".ts", "const total = 42;\n")) == set()


# ------------------------------------------------------------------------- Go

def test_a_go_method_is_stored_under_its_receiver_type():
    """Go names the owner in the receiver, not by nesting the declaration."""
    found = paths_of(parse("go", ".go",
                           "package a\n\nfunc (c *Coordinator) Acquire(s string) string "
                           "{ return s }\n\nfunc Acquire(s string) string { return s }\n"))
    assert {"Coordinator.Acquire", "Acquire"} <= found


def test_go_import_paths_are_module_prefixed():
    assert modules("go",
                   'package a\n\nimport (\n\t"fmt"\n\t"example.com/app/internal/store"\n)\n') \
        == {"fmt", "example.com/app/internal/store"}


# ----------------------------------------------------------------------- Rust

def test_a_rust_impl_block_does_not_collide_with_its_struct():
    """Two active symbols sharing one symbol_path make every call to it ambiguous."""
    found = parse("rust", ".rs",
                  "pub struct S;\nimpl S { pub fn go(&self) {} }\n"
                  "impl Display for S { fn fmt(&self) {} }\n")
    paths = paths_of(found)
    assert "S" in paths
    assert "impl S" in paths
    assert "impl Display for S" in paths
    assert "S.go" in paths, "methods stay under the type a caller writes"
    assert len(paths) == len(found), "no two symbols share a path"


def test_a_rust_attribute_is_read_from_the_preceding_sibling():
    found = {s.name: s.decorators for s in parse(
        "rust", ".rs", "#[test]\nfn test_thing() {}\n")}
    assert found["test_thing"] == ["#[test]"]


def test_a_rust_mod_declaration_names_a_file_but_an_inline_mod_does_not():
    assert modules("rust", "mod auth;\npub mod store;\n") == {"auth", "store"}
    assert modules("rust", "mod inline { fn x() {} }\n") == set()


def test_an_inline_rust_mod_is_still_walked_for_its_own_imports():
    assert modules("rust", "mod inline {\n    use crate::store::lookup;\n}\n") \
        == {"crate::store::lookup"}


# --------------------------------------------------- Java, Kotlin, C# markers

def test_java_annotations_are_read_from_the_modifiers_group():
    """Java groups `@Test` with `public`, so direct children show only the group."""
    found = {s.symbol_path: s.decorators for s in parse(
        "java", ".java",
        "public class K {\n  @GetMapping(\"/refresh\")\n  public String m() { return null; }\n"
        "  @Test\n  public void testThing() {}\n}\n")}
    assert found["K.m"] == ['@GetMapping("/refresh")']
    assert found["K.testThing"] == ["@Test"]


def test_a_java_route_annotation_becomes_a_route_entry_point():
    symbols = parse("java", ".java",
                    "public class K {\n  @GetMapping(\"/refresh\")\n"
                    "  public String m() { return null; }\n}\n")
    method = next(s for s in symbols if s.name == "m")
    assert classify(method, "src/K.java") == (
        "route", "GET /refresh", '@GetMapping("/refresh")')


def test_kotlin_names_the_function_not_its_return_type():
    """Kotlin has no `name` field; the search fell through to the return type."""
    found = paths_of(parse(
        "kotlin", ".kt",
        "class C {\n    fun acquire(s: String): String = s\n}\n"
        "fun refreshSession(s: String): String = s\n"))
    assert {"C", "C.acquire", "refreshSession"} <= found
    assert "String" not in found


def test_a_csharp_attribute_declares_a_route_and_a_test():
    symbols = parse("csharp", ".cs",
                    "namespace A {\n  public class C {\n    [HttpGet(\"/users\")]\n"
                    "    public string List() { return \"x\"; }\n    [Fact]\n"
                    "    public void ItWorks() {}\n  }\n}\n")
    by_name = {s.name: s for s in symbols}
    assert classify(by_name["List"], "src/C.cs")[:2] == ("route", "GET /users")
    assert classify(by_name["ItWorks"], "src/C.cs")[:2] == ("test", "ItWorks")


def test_a_verbless_attribute_route_does_not_invent_a_path():
    """`[HttpPost]` takes its segment from the controller, so there is none here."""
    symbols = parse("csharp", ".cs",
                    "public class C {\n  [HttpPost]\n  public string Make() { return \"y\"; }\n}\n")
    assert classify(symbols[-1], "src/C.cs")[1] == "POST"


def test_a_csharp_using_names_one_namespace_not_its_segments():
    assert modules("csharp", "using System;\nusing App.Storage;\n") == {"System", "App.Storage"}


# ------------------------------------------------------------------------ C++

def test_an_out_of_line_cpp_definition_joins_its_class():
    found = paths_of(parse(
        "cpp", ".cpp",
        "namespace app {\nstd::string Coordinator::acquire(const std::string &s)"
        " { return s; }\n}\n"))
    assert "app.Coordinator.acquire" in found


def test_a_fully_qualified_definition_does_not_repeat_its_namespace():
    found = paths_of(parse(
        "cpp", ".cpp", "namespace app {\nvoid app::C::m() {}\n}\n"))
    assert "app.C.m" in found
    assert "app.app.C.m" not in found


# -------------------------------------------------------------- test evidence

@pytest.mark.parametrize("lang,suffix,source,path,expected_kind,expected_in_evidence", [
    ("go", ".go", "package a\nimport \"testing\"\nfunc TestThing(t *testing.T) {}\n",
     "internal/auth/auth_test.go", "test", "go test-file convention"),
    ("rust", ".rs", "#[test]\nfn test_thing() {}\n",
     "src/auth.rs", "test", "#[test]"),
    ("java", ".java", "public class T {\n  @Test\n  public void testThing() {}\n}\n",
     "src/T.java", "test", "@Test"),
    ("python", ".py", "def test_thing():\n    pass\n",
     "tests/test_auth.py", "test", "pytest naming convention"),
])
def test_test_evidence_names_the_convention_that_actually_applies(
        lang, suffix, source, path, expected_kind, expected_in_evidence):
    """The evidence string is read as fact, so it must not name the wrong runner.

    It previously said "pytest naming convention" for every language, which on
    a Go file sent the reader looking for a pytest config that does not exist.
    """
    symbols = parse(lang, suffix, source)
    verdicts = [classify(s, path) for s in symbols]
    matched = [v for v in verdicts if v and v[0] == expected_kind]
    assert matched, f"no {expected_kind} entry point found in {lang}"
    assert expected_in_evidence in matched[0][2]


def test_a_go_helper_in_a_test_file_is_not_a_test():
    """Only Test/Benchmark/Fuzz/Example are collected by the Go toolchain."""
    symbols = parse("go", ".go", "package a\nfunc helper() {}\n")
    assert classify(symbols[0], "internal/auth/auth_test.go") is None


@pytest.mark.parametrize("base", ["main.go", "main.rs", "main.c"])
def test_main_in_a_program_entry_file_is_a_cli_entry_point(base):
    lang = {"go": "go", "rs": "rust", "c": "c"}[base.rsplit(".", 1)[1]]
    source = {"go": "package main\nfunc main() {}\n",
              "rust": "fn main() {}\n",
              "c": "int main(void) { return 0; }\n"}[lang]
    symbols = parse(lang, "." + base.rsplit(".", 1)[1], source)
    main = next(s for s in symbols if s.name == "main")
    assert classify(main, f"cmd/server/{base}")[0] == "cli"


# ----------------------------------------------------- end-to-end resolution

POLYGLOT = {
    "internal/auth/auth.go": (
        'package auth\n\nimport (\n\t"fmt"\n\t"example.com/app/internal/store"\n)\n\n'
        'type Coordinator struct{ prefix string }\n\n'
        'func (c *Coordinator) Acquire(s string) string {\n'
        '\treturn fmt.Sprintf("%s", store.Lookup(s))\n}\n'),
    "internal/store/store.go": "package store\n\nfunc Lookup(s string) string { return s }\n",
    "internal/store/store_test.go": (
        'package store\n\nimport "testing"\n\n'
        'func TestLookup(t *testing.T) { Lookup("x") }\n'),
    "web/auth.ts": (
        'import { hashToken } from "./crypto";\n\n'
        'export function refreshSession(id: string): string {\n'
        '  return hashToken(id);\n}\n'),
    "web/crypto.ts": "export function hashToken(raw: string): string { return raw; }\n",
    "web/legacy.js": 'const { hashToken } = require("./crypto");\n'
                     'function runWorker(j) { return hashToken(j); }\n',
    "rustsrc/main.rs": "mod auth;\n\nfn main() { auth::refresh(); }\n",
    "rustsrc/auth.rs": "use crate::store::lookup;\n\npub fn refresh() -> String { lookup() }\n",
    "rustsrc/store.rs": "pub fn lookup() -> String { String::new() }\n",
}


@pytest.fixture
def polyglot(repo):
    for rel, content in POLYGLOT.items():
        repo.write(rel, content)
    repo.commit("polyglot")

    from icn import workspace as ws_mod

    ws = ws_mod.open_workspace(str(repo.root))
    ws_mod.ensure_indexed(ws)
    yield ws
    ws.close()


def import_targets(ws, from_path: str) -> set[str]:
    """Resolved import targets of one file, as repository paths."""
    return {row["path"] for row in rows(ws.store.execute(
        "SELECT t.path FROM code_edges e"
        " JOIN files f ON f.file_id = e.from_id"
        " JOIN files t ON t.file_id = e.to_id"
        " WHERE e.kind='IMPORTS' AND e.status='ACTIVE' AND f.path = ?",
        (from_path.replace("/", "\\") if "\\" in from_path else from_path,)))}


def all_import_targets(ws, from_path: str) -> set[str]:
    found = import_targets(ws, from_path)
    if found:
        return found
    # Path separators are stored as the platform writes them.
    return {row["path"] for row in rows(ws.store.execute(
        "SELECT t.path FROM code_edges e"
        " JOIN files f ON f.file_id = e.from_id"
        " JOIN files t ON t.file_id = e.to_id"
        " WHERE e.kind='IMPORTS' AND e.status='ACTIVE'"
        " AND REPLACE(f.path, '\\', '/') = ?", (from_path,)))}


def normalised(paths: set[str]) -> set[str]:
    return {p.replace("\\", "/") for p in paths}


def test_a_go_import_resolves_past_its_module_prefix(polyglot):
    """`example.com/app/internal/store` names a directory no path starts with."""
    found = normalised(all_import_targets(polyglot, "internal/auth/auth.go"))
    assert "internal/store/store.go" in found


def test_a_go_package_import_does_not_pull_in_its_test_files(polyglot):
    found = normalised(all_import_targets(polyglot, "internal/auth/auth.go"))
    assert "internal/store/store_test.go" not in found


def test_a_typescript_named_import_resolves_to_one_real_file(polyglot):
    found = normalised(all_import_targets(polyglot, "web/auth.ts"))
    assert found == {"web/crypto.ts"}


def test_a_typescript_import_records_no_external_module_for_a_local_file(polyglot):
    external = [row["to_id"] for row in rows(polyglot.store.execute(
        "SELECT e.to_id FROM code_edges e JOIN files f ON f.file_id = e.from_id"
        " WHERE e.kind='IMPORTS' AND e.status='ACTIVE' AND e.edge_class='external'"
        " AND REPLACE(f.path, '\\', '/') = 'web/auth.ts'"))]
    assert external == [], f"invented external modules: {external}"


def test_a_commonjs_require_resolves(polyglot):
    found = normalised(all_import_targets(polyglot, "web/legacy.js"))
    assert "web/crypto.ts" in found or "web/crypto.js" in found


def test_a_rust_mod_and_use_both_resolve(polyglot):
    assert "rustsrc/auth.rs" in normalised(all_import_targets(polyglot, "rustsrc/main.rs"))
    assert "rustsrc/store.rs" in normalised(all_import_targets(polyglot, "rustsrc/auth.rs"))


def test_a_go_method_call_resolves_through_the_import(polyglot):
    """store.Lookup is reachable only because auth.go's import resolved."""
    found = rows(polyglot.store.execute(
        "SELECT t.symbol_path FROM code_edges e"
        " JOIN symbols s ON s.symbol_id = e.from_id"
        " JOIN symbols t ON t.symbol_id = e.to_id"
        " WHERE e.kind='CALLS' AND e.status='ACTIVE'"
        " AND s.symbol_path = 'Coordinator.Acquire'"))
    assert "Lookup" in {row["symbol_path"] for row in found}


def test_entry_points_are_found_in_more_than_one_language(polyglot):
    found = rows(polyglot.store.execute(
        "SELECT e.kind, e.evidence, s.lang FROM entry_points e"
        " JOIN symbols s ON s.symbol_id = e.symbol_id"))
    langs = {row["lang"] for row in found}
    assert {"go", "rust"} <= langs
    for row in found:
        if row["lang"] != "python":
            assert "pytest" not in (row["evidence"] or "")


# ------------------------------------------------- languages with no hand rules

def test_a_language_without_hand_written_rules_still_yields_symbols():
    """Coverage comes from tree-sitter's own tags.scm, not from guessing.

    A language with no SYMBOL_NODES entry used to return an empty symbol list,
    which is worse than saying support is partial: it looks like a file with no
    code in it. The tags query is upstream's own description of what counts as
    a definition in that grammar.
    """
    found = parse("dart", ".dart",
                  "class Greeter {\n  String greet(String w) { return w; }\n}\n")
    paths = {s.symbol_path for s in found}
    assert {"Greeter", "Greeter.greet"} <= paths


def test_tag_derived_symbols_are_nested_by_containment():
    found = parse("solidity", ".sol",
                  "contract Token {\n  function transfer(address to) public {}\n}\n")
    assert "Token.transfer" in {s.symbol_path for s in found}


def test_tag_derived_calls_name_the_callee_not_the_whole_invocation():
    """`@reference.call` marks the invocation; `@name` marks the callee.

    Reading the invocation's own text recorded an Elixir call as the entire
    "defmodule Auth do\n  def refresh" block.
    """
    found = parse("ocaml", ".ml", "let refresh session = lookup session\n")
    assert found and found[0].calls == ["lookup"]


def test_a_declaration_macro_is_not_recorded_as_a_call():
    """Elixir declares with macros, so a definition is also a call to itself."""
    found = parse("elixir", ".ex",
                  "defmodule Auth do\n  def refresh(s) do\n    lookup(s)\n  end\nend\n")
    for symbol in found:
        assert "defmodule" not in symbol.calls
        assert "def" not in symbol.calls


def test_a_hand_written_language_keeps_its_own_extractor():
    """The hand maps know things a generic query cannot - receivers, impls."""
    assert parsing.support_tier("python") == "full"
    assert parsing.support_tier("go") == "full"
    assert parsing.support_tier("rust") == "full"


def test_support_tier_reports_tags_separately_from_full():
    assert parsing.support_tier("dart") == "tags"


def test_a_package_qualified_call_resolves_through_the_import(polyglot):
    """`store.Lookup` names a package, not a type.

    No symbol called `store` exists anywhere, so every receiver tier missed and
    the call was dropped as ambiguous - which was most of what Go left
    unresolved. The receiver is matched against the modules the file actually
    imports, so the edge rests on a resolved import rather than on a guess.
    """
    found = rows(polyglot.store.execute(
        "SELECT s.symbol_path a, t.symbol_path b, e.source, e.edge_class"
        " FROM code_edges e JOIN symbols s ON s.symbol_id = e.from_id"
        " JOIN symbols t ON t.symbol_id = e.to_id"
        " WHERE e.kind='CALLS' AND e.status='ACTIVE'"
        " AND s.symbol_path='Coordinator.Acquire'"))
    by_target = {row["b"]: row for row in found}
    assert "Lookup" in by_target, f"expected a cross-package edge, got {list(by_target)}"
    assert by_target["Lookup"]["source"] == "tree-sitter:imported_module"
    assert by_target["Lookup"]["edge_class"] == "deterministic"


def test_a_receiver_that_names_no_import_does_not_resolve_by_module(polyglot):
    """The tier must rest on a real import edge, not on any matching name."""
    found = rows(polyglot.store.execute(
        "SELECT COUNT(*) n FROM code_edges e"
        " JOIN symbols s ON s.symbol_id = e.from_id"
        " WHERE e.kind='CALLS' AND e.source='tree-sitter:imported_module'"
        " AND s.symbol_path='Lookup'"))
    assert found[0]["n"] == 0
