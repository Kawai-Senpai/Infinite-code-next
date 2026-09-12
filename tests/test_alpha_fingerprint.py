"""Alpha normalisation: the middle canonical form, between content and skeleton.

Every test here protects one half of the promise that makes the tier worth
having. The three forms must stay strictly ordered by permissiveness:

    content   identifiers kept        `total` != `result`
    alpha     identifiers renumbered  `total` == `result`, `V1=V2` != `V1=V1`
    skeleton  identifiers erased      `V1=V2` == `V1=V1`

Skeleton can afford to be permissive because the anchor cascade adjudicates
afterwards with token_similarity. Nothing adjudicates a clone report, so alpha
carries its own precision and these tests are what hold it. If alpha ever
becomes as permissive as skeleton, the tier has no reason to exist.

The naming rule is the subtle part and it is measured, not assumed: a CALLED
attribute is API surface and is preserved, a plain data field is not. Getting
that backwards breaks the commonest real case - one validation block copied
between two modules that spell the same field differently.
"""

from __future__ import annotations

import pytest

from icn import parsing


def forms(lang: str, source: str, whole: bool = False) -> tuple[str, str, str]:
    """(content, alpha, skeleton) fingerprints of one snippet."""
    parser = parsing.get_parser(lang)
    if parser is None:
        pytest.skip(f"grammar for {lang} unavailable")
    raw = source.encode()
    root = parser.parse(raw).root_node
    node = root if whole else root.children[0]
    return (
        parsing._sha("".join(parsing.normalize(node, raw, keep_identifiers=True))),
        parsing._sha("".join(parsing.alpha_normalize(node, raw))),
        parsing._sha("".join(parsing.normalize(node, raw, keep_identifiers=False))),
    )


SUM_A = "def f(xs):\n    total = 0\n    for x in xs:\n        total += x\n    return total"
SUM_B = "def g(items):\n    result = 0\n    for i in items:\n        result += i\n    return result"


def test_pure_rename_collapses_under_alpha_but_not_content():
    """A Type-2 clone: same program, every name changed."""
    content_a, alpha_a, _ = forms("python", SUM_A)
    content_b, alpha_b, _ = forms("python", SUM_B)
    assert content_a != content_b, "content must notice a rename"
    assert alpha_a == alpha_b, "alpha must see through a pure rename"


def test_alpha_separates_self_assignment_from_cross_assignment():
    """The reason this tier exists.

    `a = a` and `a = b` are different programs, but skeleton erases every
    identifier and cannot tell them apart. Consistent renaming preserves WHICH
    names coincide, so it can. If this test fails, alpha has degenerated into
    skeleton and should be deleted rather than kept.
    """
    _, alpha_self, skel_self = forms("python", "def f(a, b):\n    a = a\n    return a")
    _, alpha_cross, skel_cross = forms("python", "def f(a, b):\n    a = b\n    return a")
    assert alpha_self != alpha_cross, "alpha must distinguish V1=V1 from V1=V2"
    assert skel_self == skel_cross, "skeleton is expected to collide here"


def test_changed_constant_does_not_split_a_clone_class():
    """`timeout = 30` and `timeout = 60` are the same logic.

    Folding literals to their node type is what makes a "same code, different
    constant" duplicate findable at all.
    """
    content_a, alpha_a, _ = forms("python", "def f():\n    t = 30\n    return t")
    content_b, alpha_b, _ = forms("python", "def f():\n    t = 60\n    return t")
    assert content_a != content_b
    assert alpha_a == alpha_b


def test_called_attribute_is_preserved():
    """A read and a delete must never share a fingerprint.

    Skeleton collides here, which is precisely why a clone report cannot be
    built on skeleton alone.
    """
    _, alpha_get, skel_get = forms("python", "def f(c, k):\n    return c.get(k)")
    _, alpha_del, skel_del = forms("python", "def f(c, k):\n    return c.delete(k)")
    assert alpha_get != alpha_del, "the method name is API surface"
    assert skel_get == skel_del, "skeleton is expected to collide here"


def test_receiver_name_is_renamed_like_any_other_variable():
    """NiCad renames receivers, and so do we.

    `cache.get(key)` and `database.get(user_id)` are a genuine Type-2 clone.
    """
    _, alpha_cache, _ = forms("python", "def f(cache, key):\n    return cache.get(key)")
    _, alpha_db, _ = forms("python", "def f(database, uid):\n    return database.get(uid)")
    assert alpha_cache == alpha_db


def test_plain_data_field_is_renamed_but_called_method_is_not():
    """The measured case that drove the rule.

    Two copies of one validation block reading a differently-named field off a
    differently-named object. Preserving every attribute made these NOT match,
    which defeated the feature. Preserving only CALLED attributes fixes it:
    `req.email` and `data.address` both normalise to placeholders, while
    `.strip` and `.match` stay put and keep this apart from unrelated code.

    The two locals are named differently AND the fields are named differently,
    but each copy uses its names in the same pattern, so the placeholders line
    up. Before the rule was narrowed to called attributes only, the differing
    field names alone were enough to split these apart.
    """
    a = ("addr = req.email.strip().lower()\n"
         "if not EMAIL_RE.match(addr):\n"
         "    raise ValidationError('bad')\n")
    b = ("value = data.address.strip().lower()\n"
         "if not EMAIL_RE.match(value):\n"
         "    raise ValidationError('bad')\n")
    content_a, alpha_a, _ = forms("python", a, whole=True)
    content_b, alpha_b, _ = forms("python", b, whole=True)
    assert content_a != content_b
    assert alpha_a == alpha_b, "a differing data-field name must not split the class"


def test_alpha_stops_where_the_coincidence_pattern_differs():
    """Consistent renaming preserves WHICH names coincide, and that boundary is
    a property rather than a defect.

    Here one copy reuses the field's own name for the local
    (`email = req.email`) and the other does not (`addr = data.address`), so
    the two genuinely differ in which names are shared. Alpha reports them as
    different and NiCad would agree. That residue is a Type-3 clone, and the
    near tier - token_similarity over the alpha token stream - is what catches
    it, with a wide measured margin against unrelated code.

    This test exists so nobody later "fixes" alpha to match these, which would
    mean erasing the coincidence information that makes alpha stronger than
    skeleton in the first place.
    """
    parser = parsing.get_parser("python")
    if parser is None:
        pytest.skip("python grammar unavailable")

    def sketch(source: str):
        raw = source.encode()
        return parsing.token_signature(
            parsing.alpha_normalize(parser.parse(raw).root_node, raw))

    a = ("email = req.email.strip().lower()\n"
         "if not EMAIL_RE.match(email):\n"
         "    raise ValidationError('bad')\n")
    b = ("addr = data.address.strip().lower()\n"
         "if not EMAIL_RE.match(addr):\n"
         "    raise ValidationError('bad')\n")
    unrelated = "rows = db.fetch(query)\nfor r in rows:\n    total += r.amount\n"

    near = parsing.token_similarity(sketch(a), sketch(b))
    far = parsing.token_similarity(sketch(a), sketch(unrelated))
    assert near > 0.4, "the near tier must still recognise a Type-3 clone"
    assert near > far * 5, "and must separate it clearly from unrelated code"


def test_operator_flip_changes_every_form():
    """The defect that motivated including operators in fingerprints at all.

    An operator is meaning, not naming, so no amount of normalisation may erase
    it - alpha included.
    """
    content_a, alpha_a, skel_a = forms("python", "def f(a, b):\n    return a + b")
    content_b, alpha_b, skel_b = forms("python", "def f(a, b):\n    return a - b")
    assert content_a != content_b
    assert alpha_a != alpha_b
    assert skel_a != skel_b


@pytest.mark.parametrize("lang,src_a,src_b", [
    ("javascript",
     "function f(xs){ let t=0; for(const x of xs){ t+=x; } return t; }",
     "function g(items){ let r=0; for(const i of items){ r+=i; } return r; }"),
    ("go",
     "package m\nfunc f(xs []int) int { t := 0; for _, x := range xs { t += x }; return t }",
     "package m\nfunc g(items []int) int { r := 0; for _, i := range items { r += i }; return r }"),
    ("rust",
     "fn f(xs: Vec<i32>) -> i32 { let mut t = 0; for x in xs { t += x; } t }",
     "fn g(items: Vec<i32>) -> i32 { let mut r = 0; for i in items { r += i; } r }"),
    ("java",
     "class A { int f(int[] xs){ int t=0; for(int x: xs){ t+=x; } return t; } }",
     "class A { int g(int[] items){ int r=0; for(int i: items){ r+=i; } return r; } }"),
])
def test_rename_is_seen_through_in_every_supported_language(lang, src_a, src_b):
    """Attribute detection uses per-grammar field names, so it is per-language.

    Relying on the Python spelling alone silently renamed method names in half
    the supported languages, which is why both routes are checked in
    is_attribute and why this is parametrised rather than Python-only.
    """
    content_a, alpha_a, _ = forms(lang, src_a, whole=True)
    content_b, alpha_b, _ = forms(lang, src_b, whole=True)
    assert content_a != content_b, f"{lang}: content must notice the rename"
    assert alpha_a == alpha_b, f"{lang}: alpha must see through the rename"


def test_unrelated_code_does_not_collide():
    """A fingerprint is only worth having if two different programs cannot
    produce the same one."""
    _, alpha_sum, _ = forms("python", SUM_A)
    _, alpha_other, _ = forms(
        "python", "def h(conn):\n    rows = conn.fetch('q')\n    return len(rows)")
    assert alpha_sum != alpha_other
