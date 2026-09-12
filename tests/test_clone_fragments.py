"""Block-level fragment extraction: the piece that makes clone detection see
duplicated logic *inside* functions.

The motivating case, and the reason symbol-level fingerprints are not enough:
the same validation block copied into two unrelated functions. Neither
function matches as a whole, so nothing in the symbol table can represent the
overlap. These tests hold the properties that make the fragment table worth
its row count.
"""

from __future__ import annotations

import pytest

from icn import parsing

pytestmark = pytest.mark.skipif(
    parsing.get_parser("python") is None,
    reason="tree-sitter python grammar not installed",
)


def frags(source: str, **kw):
    return parsing.extract_fragments(source.encode(), "python", **kw)


def test_a_block_inside_a_function_becomes_its_own_fragment():
    """The whole point: sub-function granularity.

    A loop body is a clone candidate in its own right. If extraction only ever
    emitted whole functions this would return one fragment, and duplicated
    logic embedded in differing functions would stay invisible.
    """
    src = """
def outer(items):
    total = 0
    for item in items:
        value = item.amount * item.rate
        adjusted = value + item.offset
        total = total + adjusted
        record(total, adjusted, value, item, items)
    return total
"""
    got = frags(src, min_tokens=0)
    assert len(got) > 1
    spans = {(f.line_start, f.line_end) for f in got}
    # the function itself, plus something strictly inside it
    assert any(end - start < max(e - s for s, e in spans) for start, end in spans)


def test_the_same_block_in_two_functions_shares_an_alpha_fingerprint():
    """Copy-paste with renamed locals is Type 2, and alpha must collapse it."""
    a = """
def handle(payload):
    count = 0
    for entry in payload:
        if entry.valid:
            count = count + entry.weight
            log(count, entry, payload, "seen")
    return count
"""
    b = """
def process(records):
    total = 0
    for row in records:
        if row.valid:
            total = total + row.weight
            log(total, row, records, "seen")
    return total
"""
    fa = {f.alpha_fingerprint for f in frags(a, min_tokens=0)}
    fb = {f.alpha_fingerprint for f in frags(b, min_tokens=0)}
    assert fa & fb, "renamed copy of the same block must share an alpha fingerprint"

    ca = {f.content_fingerprint for f in frags(a, min_tokens=0)}
    cb = {f.content_fingerprint for f in frags(b, min_tokens=0)}
    assert not (ca & cb), "content fingerprints must still tell the copies apart"


def test_unrelated_code_does_not_collide():
    a = "def f(x):\n    return sorted(set(x), key=len, reverse=True)\n"
    b = "def g(conn):\n    conn.execute('DELETE FROM t')\n    conn.commit()\n"
    fa = {f.alpha_fingerprint for f in frags(a, min_tokens=0)}
    fb = {f.alpha_fingerprint for f in frags(b, min_tokens=0)}
    assert not (fa & fb)


def test_a_fragment_is_never_reported_as_a_clone_of_itself():
    """A body spanning the same bytes as its parent must be emitted once.

    Without span de-duplication a function whose body is its only child yields
    two rows with identical fingerprints, and the first clustering pass would
    report the function as a duplicate of itself.
    """
    src = """
def only(values):
    for value in values:
        transform(value, values, value, values, value)
"""
    got = frags(src, min_tokens=0)
    spans = [(f.start_byte, f.end_byte) for f in got]
    assert len(spans) == len(set(spans))


def test_the_token_floor_is_applied_before_returning():
    """The floor must filter at extraction, not at query time, or the table
    stores rows the caller has already decided are worthless."""
    src = "def tiny(a):\n    return a + 1\n"
    assert frags(src, min_tokens=0)
    assert frags(src, min_tokens=60) == []


def test_every_fragment_carries_all_three_tiers_and_a_real_span():
    src = """
def compute(rows, factor):
    out = []
    for row in rows:
        scaled = row.value * factor
        if scaled > row.limit:
            out.append((row.key, scaled, row.limit, factor, rows))
    return out
"""
    for f in frags(src, min_tokens=0):
        assert f.content_fingerprint and f.alpha_fingerprint and f.skeleton_fingerprint
        assert f.token_count > 0
        assert f.line_start <= f.line_end
        assert f.end_byte > f.start_byte
        assert f.lang == "python"


def test_unparseable_or_unknown_language_yields_nothing_rather_than_raising():
    assert parsing.extract_fragments(b"def (((", "python") == []
    assert parsing.extract_fragments(b"whatever", "not-a-language") == []


# ------------------------------------------------------------------ indexing

CLONE_SOURCE = '''
def normalize_user(req):
    email = req.email.strip().lower()
    if not email:
        raise ValueError("missing")
    domain = email.split("@")[1]
    return (email, domain, req, email, domain)


def normalize_invite(data):
    address = data.address.strip().lower()
    if not address:
        raise ValueError("missing")
    host = address.split("@")[1]
    return (address, host, data, address, host)
'''


def test_fragments_are_stored_and_replaced_with_the_file(workspace):
    """Fragments must land in the store, and a re-store must not duplicate
    them: they are derived data, rebuilt wholesale rather than reconciled."""
    from icn import indexer as indexer_mod

    (workspace.root / "clones.py").write_text(CLONE_SOURCE, encoding="utf-8")
    idx = indexer_mod.Indexer(workspace.store, workspace.root,
                              workspace.repo_id, workspace.excludes)
    idx.index_file(workspace.root / "clones.py", None)

    def count():
        return workspace.store.execute(
            "SELECT count(*) FROM clone_fragments cf JOIN files f"
            " ON f.file_id = cf.file_id WHERE f.path = 'clones.py'").fetchone()[0]

    first = count()
    assert first > 0

    # Re-storing the identical file must not accumulate a second copy.
    idx.index_file(workspace.root / "clones.py", None)
    assert count() == first


def test_test_files_contribute_no_fragments(workspace):
    """Duplicated test setup is deliberate. Reporting it teaches the agent to
    ignore the whole clone report."""
    stored = workspace.store.execute(
        "SELECT count(*) FROM clone_fragments cf JOIN files f"
        " ON f.file_id = cf.file_id WHERE f.path LIKE 'tests/%'").fetchone()[0]
    assert stored == 0
