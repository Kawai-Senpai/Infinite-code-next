"""Extraction correctness: fingerprints, call scope, import classification.

Every test here corresponds to a defect that was present and is now fixed. They
are grouped by the thing they protect rather than by the function they call,
because the failures were not local: an operator missing from a fingerprint is
an anchoring bug, not a parsing detail.

The fingerprint tests are the load-bearing ones. Anchoring is built on the
promise that a fingerprint changes when the code means something different and
does not change when it merely looks different, and both halves need holding.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from icn import parsing


def parse(lang: str, suffix: str, source: str):
    if parsing.get_parser(lang) is None:
        pytest.skip(f"grammar for {lang} unavailable")
    return parsing.parse_file(Path(f"sample{suffix}"), source.encode(), lang)


def one(lang: str, suffix: str, source: str, name: str):
    found = [s for s in parse(lang, suffix, source) if s.name == name]
    assert found, f"no symbol named {name} in {lang} source"
    return found[0]


def specs(lang: str, source: str) -> list[dict]:
    if parsing.get_parser(lang) is None:
        pytest.skip(f"grammar for {lang} unavailable")
    return parsing.file_import_specs(source.encode(), lang)


def spec_for(lang: str, source: str, module: str) -> dict | None:
    for spec in specs(lang, source):
        if spec["module"] == module:
            return spec
    return None


# ------------------------------------------------------- fingerprint sensitivity

PY_OPERATORS = [
    ("def f(a, b):\n    return a + b\n", "def f(a, b):\n    return a - b\n"),
    ("def f(a, b):\n    return a * b\n", "def f(a, b):\n    return a / b\n"),
    ("def f(a, b):\n    return a == b\n", "def f(a, b):\n    return a != b\n"),
    ("def f(a, b):\n    return a and b\n", "def f(a, b):\n    return a or b\n"),
    ("def f(a, b):\n    return a < b\n", "def f(a, b):\n    return a > b\n"),
    ("def f(a, b):\n    return a in b\n", "def f(a, b):\n    return a is b\n"),
]


@pytest.mark.parametrize("left,right", PY_OPERATORS)
def test_changing_an_operator_changes_the_fingerprint(left, right):
    """The whole point of a content fingerprint.

    Tree-sitter spells operators as anonymous tokens, and the walk used to
    visit only named children - so `a + b` and `a - b` hashed identically and
    swapping an operator did not invalidate a single anchor. That is a logic
    change going unnoticed by the system whose job is to notice logic changes.
    """
    before = one("python", ".py", left, "f")
    after = one("python", ".py", right, "f")
    assert before.content_fingerprint != after.content_fingerprint
    assert before.skeleton_fingerprint != after.skeleton_fingerprint


def test_reformatting_still_does_not_change_the_fingerprint():
    """The other half of the promise: layout is not meaning."""
    tight = one("python", ".py", "def f(a,b):\n    return a+b\n", "f")
    loose = one("python", ".py",
                "def f(\n    a,\n    b,\n):\n    return a + b\n", "f")
    assert tight.content_fingerprint == loose.content_fingerprint


def test_a_comment_change_still_does_not_change_the_fingerprint():
    plain = one("python", ".py", "def f(a):\n    return a\n", "f")
    noted = one("python", ".py",
                "def f(a):\n    # explains why\n    return a\n", "f")
    assert plain.content_fingerprint == noted.content_fingerprint


def test_a_rename_keeps_the_skeleton_and_changes_the_content():
    before = one("python", ".py", "def f(alpha):\n    return alpha + 1\n", "f")
    after = one("python", ".py", "def f(beta):\n    return beta + 1\n", "f")
    assert before.skeleton_fingerprint == after.skeleton_fingerprint
    assert before.content_fingerprint != after.content_fingerprint


def test_punctuation_alone_does_not_change_a_javascript_fingerprint():
    """A semicolon and a trailing comma are style, not meaning."""
    plain = one("javascript", ".js", "function f(a, b) { return a + b }\n", "f")
    fussy = one("javascript", ".js", "function f(a, b,) { return a + b; }\n", "f")
    assert plain.content_fingerprint == fussy.content_fingerprint


def test_a_javascript_operator_change_does_change_the_fingerprint():
    loose = one("javascript", ".js", "function f(a, b) { return a == b; }\n", "f")
    strict = one("javascript", ".js", "function f(a, b) { return a === b; }\n", "f")
    assert loose.content_fingerprint != strict.content_fingerprint


# ------------------------------------------------------------ anchoring stability

def test_inserting_a_comment_does_not_move_a_neighbours_ast_path():
    """ast_path is an anchoring signal, so a comment must not renumber it.

    The ordinal used to count every named sibling, and a comment is a named
    sibling - so adding a note above a function changed that function's
    ast_path and moved its anchor for a change fingerprints ignore.
    """
    before = one("python", ".py", "def a():\n    pass\n\n\ndef b():\n    pass\n", "b")
    after = one("python", ".py",
                "def a():\n    pass\n\n\n# inserted\ndef b():\n    pass\n", "b")
    assert before.ast_path == after.ast_path


def test_inserting_a_comment_does_not_change_context_fingerprints():
    before = one("python", ".py", "def a():\n    pass\n\n\ndef b():\n    pass\n", "b")
    after = one("python", ".py",
                "def a():\n    pass\n\n\n# inserted\ndef b():\n    pass\n", "b")
    assert before.prev_fingerprint == after.prev_fingerprint


def test_the_ordinal_counts_siblings_of_the_same_type():
    """A class between two functions must not renumber the second function."""
    before = one("python", ".py", "def a():\n    pass\n\n\ndef b():\n    pass\n", "b")
    after = one("python", ".py",
                "def a():\n    pass\n\n\nclass Mid:\n    pass\n\n\ndef b():\n    pass\n", "b")
    assert before.ast_path == after.ast_path


# ------------------------------------------------------------------- call scope

def test_a_nested_function_keeps_its_own_calls():
    """An outer function must not be credited with an inner function's calls.

    This is worse than an incomplete graph: it writes a real edge pointing at
    the wrong symbol, so impact() reports a dependency that does not exist.
    """
    outer = one("python", ".py",
                "def outer():\n    first()\n    second()\n\n"
                "    def inner():\n        hidden()\n", "outer")
    assert "hidden" not in outer.calls
    assert "first" in outer.calls and "second" in outer.calls


def test_calls_come_back_in_source_order():
    outer = one("python", ".py",
                "def outer():\n    first()\n    second()\n    third()\n", "outer")
    assert outer.calls == ["first", "second", "third"]


def test_a_function_bound_to_a_name_still_reports_its_calls():
    """The stop-at-a-callable rule must not stop at the symbol's own body."""
    bound = one("javascript", ".js", "const f = () => g();\n", "f")
    assert bound.calls == ["g"]


def test_a_call_inside_an_argument_is_still_this_symbols_call():
    outer = one("python", ".py", "def outer():\n    return f(g())\n", "outer")
    assert outer.calls == ["f", "g"]


def test_a_class_does_not_absorb_its_methods_calls():
    holder = one("python", ".py",
                 "class Holder:\n    def method(self):\n        work()\n", "Holder")
    assert "work" not in holder.calls


# -------------------------------------------------------- import classification

def test_a_javascript_re_export_is_an_import():
    """`export { X } from './m'` is a dependency and produced no edge at all."""
    assert spec_for("javascript", 'export { User } from "./user";\n', "./user")


def test_a_star_re_export_is_an_import():
    assert spec_for("javascript", 'export * from "./user";\n', "./user")


def test_an_export_statement_is_still_walked_for_nested_imports():
    """The re-export node is not a leaf: `export function` holds real code."""
    found = specs("javascript",
                  'export function f() { return import("./lazy"); }\n')
    assert {s["module"] for s in found} == {"./lazy"}


def test_a_dynamic_import_is_deferred_even_at_top_level():
    """import() returns a promise, so it cannot force an initialisation order.

    It is the standard way to break a JavaScript import cycle, so counting it
    as eager reports the fix as the fault.
    """
    spec = spec_for("javascript", 'const e = await import("./editor");\n', "./editor")
    assert spec and spec.get("deferred") is True


def test_a_top_level_require_is_not_deferred():
    spec = spec_for("javascript", 'const a = require("./a");\n', "./a")
    assert spec and not spec.get("deferred")


def test_a_type_checking_import_is_deferred():
    """Python's `import type`, and the standard way to annotate across a cycle."""
    source = ("from typing import TYPE_CHECKING\n\n"
              "if TYPE_CHECKING:\n    from app.models import User\n")
    spec = spec_for("python", source, "app.models")
    assert spec and spec.get("deferred") is True


def test_another_conditional_import_is_not_deferred():
    """`if platform...: import winreg` really does run at import time.

    Treating every guarded import as deferred would hide real cycles, so only
    TYPE_CHECKING is special-cased.
    """
    source = ("import platform\n\n"
              "if platform.system() == 'Windows':\n    import winreg\n")
    spec = spec_for("python", source, "winreg")
    assert spec and not spec.get("deferred")


# -------------------------------------------------------------------------- Go

def test_a_go_import_alias_is_not_recorded_as_a_module():
    found = {s["module"] for s in specs(
        "go", 'package a\n\nimport foo "example.com/x"\n')}
    assert found == {"example.com/x"}


def test_every_type_in_a_grouped_go_declaration_is_extracted():
    """One `type ( ... )` holds many specs, and only the first was found."""
    found = {s.name for s in parse(
        "go", ".go",
        "package a\n\ntype (\n\tA struct{}\n\tB interface{}\n\tC = string\n)\n")}
    assert {"A", "B"} <= found


# ---------------------------------------------------------------- parser cache

def test_a_transient_parser_failure_is_not_cached(monkeypatch):
    """The language pack loads grammars on demand, so a failure can be transient.

    Caching one disabled that language for the life of the process, and the
    whole language then went silently missing from the index.
    """
    import tree_sitter_language_pack as pack

    saved = dict(parsing._PARSERS)
    parsing._PARSERS.pop("python", None)
    real = pack.get_parser
    attempts = {"n": 0}

    def flaky(lang):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("transient download failure")
        return real(lang)

    monkeypatch.setattr(pack, "get_parser", flaky)
    try:
        assert parsing.get_parser("python") is None
        assert parsing.get_parser("python") is not None
        assert attempts["n"] == 2, "the loader must be retried, not cached as dead"
    finally:
        parsing._PARSERS.clear()
        parsing._PARSERS.update(saved)


# ---------------------------------------------------------- language detection

@pytest.mark.parametrize("suffix", [".mts", ".cts", ".pyw", ".kts", ".hxx"])
def test_extensions_beyond_the_explicit_table_are_recognised(suffix):
    """The table is an override; the language pack knows hundreds more."""
    assert parsing.language_for(Path("sample" + suffix)) is not None


def test_the_explicit_table_still_wins():
    """Pinned languages must not drift if the pack renames a grammar."""
    assert parsing.language_for(Path("a.py")) == "python"
    assert parsing.language_for(Path("a.tsx")) == "tsx"


def test_support_tier_distinguishes_parsing_from_extracting():
    """Claiming a language is supported because it parses would be a lie.

    A language with no SYMBOL_NODES entry yields an empty symbol list, which is
    worse than saying support is partial.
    """
    assert parsing.support_tier("python") == "full"
    assert parsing.support_tier("no-such-language") == "none"
