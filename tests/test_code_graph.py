"""Import resolution, tiered call resolution, and the structural queries.

The theme of this file is that a missing edge and an undecidable edge are
different facts. Most of these tests fail loudly if resolution ever goes back
to reporting the second as the first.
"""

from __future__ import annotations

from icn import graph, parsing
from icn import history as history_mod
from icn import workspace as ws_mod
from icn.db import one, rows


# ------------------------------------------------------------ import specifiers

def test_from_package_import_names_the_submodule_not_the_package():
    """`from . import parsing` must reach parsing.py, not __init__.py.

    This is how this codebase imports its own modules. Treating the statement
    as importing only the package linked every such file to the package
    __init__, which put the entire repository one hop from everything.
    """
    specs = parsing.file_import_specs(b"from . import parsing, ids\n", "python")
    modules = {(s["module"], s["level"], s.get("probe", False)) for s in specs}
    assert ("parsing", 1, True) in modules
    assert ("ids", 1, True) in modules


def test_imported_names_are_probes_not_asserted_modules():
    """`from a import SomeClass` must not invent a module named a.SomeClass."""
    specs = parsing.file_import_specs(b"from json import JSONDecoder\n", "python")
    by_module = {s["module"]: s for s in specs}
    assert by_module["json"].get("probe") is not True
    assert by_module["json.JSONDecoder"]["probe"] is True


def test_from_import_does_not_report_imported_names_as_modules():
    """The names in `from a.b import c, d` are not modules a.c and a.d."""
    specs = parsing.file_import_specs(b"from a.b import c, d\n", "python")
    assert {s["module"] for s in specs} == {"a.b", "a.b.c", "a.b.d"}


# ------------------------------------------------------------- resolved imports

def test_imports_resolve_to_real_file_nodes(workspace):
    """An IMPORTS edge must point at a file row, not at statement text.

    The previous scheme wrote to_id='import:<statement>', which is a node that
    exists in no table: the edge could be counted but never walked, so it
    supported no cycle detection and no import-scoped call resolution.
    """
    edges = rows(workspace.store.execute(
        "SELECT e.from_id, e.to_id FROM code_edges e WHERE e.kind='IMPORTS'"
        " AND e.status='ACTIVE' AND e.to_id NOT LIKE 'extern:%'"))
    assert edges, "no resolved file-to-file imports at all"

    for edge in edges:
        target = one(workspace.store.execute(
            "SELECT file_id FROM files WHERE file_id=?", (edge["to_id"],)))
        assert target is not None, f"IMPORTS edge points at non-file {edge['to_id']!r}"

    assert not rows(workspace.store.execute(
        "SELECT 1 FROM code_edges WHERE kind='IMPORTS' AND to_id LIKE 'import:%'")), \
        "legacy statement-text IMPORTS edges are still being written"


def test_api_imports_auth_as_a_walkable_edge(workspace):
    paths = {r["file_id"]: r["path"] for r in rows(workspace.store.execute(
        "SELECT file_id, path FROM files WHERE status='ACTIVE'"))}
    linked = {
        (paths.get(e["from_id"]), paths.get(e["to_id"]))
        for e in rows(workspace.store.execute(
            "SELECT from_id, to_id FROM code_edges WHERE kind='IMPORTS'"
            " AND status='ACTIVE'"))}
    assert ("api.py", "auth.py") in linked


def test_an_unresolvable_import_is_external_not_silently_dropped(repo):
    """A third-party import is a real answer, and must be distinguishable."""
    repo.write("app.py", "import requests\n\ndef go():\n    return requests.get('x')\n")
    repo.commit("external import")
    ws = ws_mod.open_workspace(str(repo.root))
    try:
        ws_mod.ensure_indexed(ws, force_full=True)
        external = rows(ws.store.execute(
            "SELECT to_id, edge_class FROM code_edges WHERE kind='IMPORTS'"
            " AND to_id LIKE 'extern:%'"))
        assert any(e["to_id"] == "extern:requests" for e in external)
        assert all(e["edge_class"] == "external" for e in external)
    finally:
        ws.close()


# -------------------------------------------------------------- resolution tiers

def test_call_edges_carry_the_tier_that_resolved_them(workspace):
    """edge_class and confidence must vary. They used to be constants.

    A bare name that happened to be unique in the repository was recorded with
    the same certainty as a call through an explicit receiver, so a consumer
    had no way to walk only what was actually proven.
    """
    tiers = rows(workspace.store.execute(
        "SELECT source, edge_class, confidence, COUNT(*) n FROM code_edges"
        " WHERE kind='CALLS' AND status='ACTIVE' GROUP BY source"))
    assert tiers
    assert all(t["source"].startswith("tree-sitter:") for t in tiers)
    assert {t["edge_class"] for t in tiers} <= {"deterministic", "inferred"}

    by_tier = {t["source"].split(":")[-1]: t for t in tiers}
    for tier, row in by_tier.items():
        expected_class, expected_conf = ws_mod.indexer_mod.CALL_TIERS[tier]
        assert row["edge_class"] == expected_class
        assert row["confidence"] == expected_conf


def test_self_calls_resolve_through_the_enclosing_class(workspace):
    """RefreshCoordinator.acquire calls self._lock_for - the strongest tier."""
    edge = one(workspace.store.execute(
        "SELECT e.source, e.confidence FROM code_edges e"
        " JOIN symbols a ON a.symbol_id = e.from_id"
        " JOIN symbols b ON b.symbol_id = e.to_id"
        " WHERE a.symbol_path='RefreshCoordinator.acquire'"
        "   AND b.symbol_path='RefreshCoordinator._lock_for' AND e.kind='CALLS'"))
    assert edge is not None, "self.<method> call did not resolve"
    assert edge["source"] == "tree-sitter:receiver_self"
    assert edge["confidence"] == 1.0


def test_a_call_through_a_typed_local_resolves_to_that_type(repo):
    """`store = Cache(); store.flush()` must reach Cache.flush.

    Two classes declare `flush`, and neither lives in the calling file, so the
    same_file and unique_global tiers cannot answer: only the recorded type of
    the local can. Before receiver typing this was the single largest source of
    unresolved calls, and it is why methods used to dominate the dead-code
    listing.
    """
    repo.write("cache.py",
               "class Cache:\n"
               "    def flush(self):\n"
               "        return 1\n"
               "\n"
               "class Buffer:\n"
               "    def flush(self):\n"
               "        return 2\n")
    repo.write("caller.py",
               "from cache import Cache\n"
               "\n"
               "def run():\n"
               "    store = Cache()\n"
               "    return store.flush()\n")
    repo.commit("typed local")
    ws = ws_mod.open_workspace(str(repo.root))
    try:
        ws_mod.ensure_indexed(ws, force_full=True)
        edge = one(ws.store.execute(
            "SELECT e.source, b.symbol_path FROM code_edges e"
            " JOIN symbols a ON a.symbol_id = e.from_id"
            " JOIN symbols b ON b.symbol_id = e.to_id"
            " WHERE a.symbol_path='run' AND e.kind='CALLS' AND e.status='ACTIVE'"
            "   AND b.name='flush'"))
        assert edge is not None, "call through a typed local did not resolve"
        assert edge["symbol_path"] == "Cache.flush"
        assert edge["source"] == "tree-sitter:receiver_typed"
    finally:
        ws.close()


def test_a_call_through_a_typed_field_resolves_across_methods(repo):
    """`self.client` is assigned in __init__ and called in another method.

    Field types have to be unioned over the whole class. Keeping them per
    symbol would resolve `self.client.get()` only inside the constructor that
    wrote it, which is never where the interesting calls are.
    """
    repo.write("store.py",
               "class Redis:\n"
               "    def get(self):\n"
               "        return 1\n"
               "\n"
               "class Memcache:\n"
               "    def get(self):\n"
               "        return 2\n")
    repo.write("service.py",
               "from store import Redis\n"
               "\n"
               "class Service:\n"
               "    def __init__(self):\n"
               "        self.client = Redis()\n"
               "\n"
               "    def lookup(self):\n"
               "        return self.client.get()\n")
    repo.commit("typed field")
    ws = ws_mod.open_workspace(str(repo.root))
    try:
        ws_mod.ensure_indexed(ws, force_full=True)
        edge = one(ws.store.execute(
            "SELECT e.source, b.symbol_path FROM code_edges e"
            " JOIN symbols a ON a.symbol_id = e.from_id"
            " JOIN symbols b ON b.symbol_id = e.to_id"
            " WHERE a.symbol_path='Service.lookup' AND e.kind='CALLS'"
            "   AND e.status='ACTIVE' AND b.name='get'"))
        assert edge is not None, "call through a typed field did not resolve"
        assert edge["symbol_path"] == "Redis.get"
        assert edge["source"] == "tree-sitter:receiver_typed"
    finally:
        ws.close()


def test_a_receiver_with_two_possible_types_is_not_guessed(repo):
    """A name rebound to a second type must leave the call unresolved.

    Recording only the last binding would be a strong update, and a strong
    update over a branch points a real edge at the wrong method - worse than
    the missing edge it replaces.
    """
    repo.write("kinds.py",
               "class Alpha:\n"
               "    def send(self):\n"
               "        return 1\n"
               "\n"
               "class Beta:\n"
               "    def send(self):\n"
               "        return 2\n")
    repo.write("pick.py",
               "from kinds import Alpha, Beta\n"
               "\n"
               "def go(flag):\n"
               "    channel = Alpha()\n"
               "    if flag:\n"
               "        channel = Beta()\n"
               "    return channel.send()\n")
    repo.commit("two types")
    ws = ws_mod.open_workspace(str(repo.root))
    try:
        ws_mod.ensure_indexed(ws, force_full=True)
        assert not rows(ws.store.execute(
            "SELECT 1 FROM code_edges e"
            " JOIN symbols a ON a.symbol_id = e.from_id"
            " JOIN symbols b ON b.symbol_id = e.to_id"
            " WHERE a.symbol_path='go' AND b.name='send'"
            "   AND e.kind='CALLS' AND e.status='ACTIVE'")), \
            "resolution picked one of two possible receiver types"
        assert rows(ws.store.execute(
            "SELECT 1 FROM unresolved_calls WHERE leaf='send'")), \
            "the undecidable call site left no trace"
    finally:
        ws.close()


def test_a_module_level_call_has_an_owner(repo):
    """A call at file scope must produce an edge, from the `<module>` symbol.

    It previously had no owning symbol, so no edge was written and its callee
    looked unreferenced - the false positive that made graph(action='deadcode')
    report functions that module-level code plainly invokes.
    """
    repo.write("main.py",
               "def bootstrap():\n"
               "    return 1\n"
               "\n"
               "bootstrap()\n")
    repo.commit("module level call")
    ws = ws_mod.open_workspace(str(repo.root))
    try:
        ws_mod.ensure_indexed(ws, force_full=True)
        edge = one(ws.store.execute(
            "SELECT a.symbol_path, a.kind FROM code_edges e"
            " JOIN symbols a ON a.symbol_id = e.from_id"
            " JOIN symbols b ON b.symbol_id = e.to_id"
            " WHERE b.symbol_path='bootstrap' AND e.kind='CALLS'"
            "   AND e.status='ACTIVE'"))
        assert edge is not None, "module-level call produced no edge"
        assert edge["symbol_path"] == "<module>"
        assert edge["kind"] == "module"

        # The pseudo-symbol exists to carry calls, not to be one. A `module`
        # kind is already excluded from the dead-code listing, and its name
        # can never match a callee.
        found = history_mod.dead_code(ws.store, include_tests=True)
        assert not [c for c in found["candidates"]
                    if c["name"] in ("bootstrap", "<module>")], \
            "a symbol called at module level was reported as dead code"
    finally:
        ws.close()


def test_an_ambiguous_call_is_recorded_rather_than_dropped(repo):
    """Two definitions of one name must leave evidence, not silence.

    Emitting no edge is correct - the analyzer genuinely cannot tell which is
    called - but emitting nothing at all makes "no callers" and "callers we
    could not attribute" the same answer.
    """
    repo.write("one.py", "def handle():\n    return 1\n")
    repo.write("two.py", "def handle():\n    return 2\n")
    repo.write("caller.py", "def go():\n    return handle()\n")
    repo.commit("ambiguous name")
    ws = ws_mod.open_workspace(str(repo.root))
    try:
        ws_mod.ensure_indexed(ws, force_full=True)
        dropped = rows(ws.store.execute(
            "SELECT leaf, reason, candidates FROM unresolved_calls WHERE leaf='handle'"))
        assert dropped, "ambiguous call site left no trace"
        assert dropped[0]["reason"] == "ambiguous"
        assert dropped[0]["candidates"] == 2

        assert not rows(ws.store.execute(
            "SELECT 1 FROM code_edges e JOIN symbols b ON b.symbol_id=e.to_id"
            " WHERE b.name='handle' AND e.kind='CALLS' AND e.status='ACTIVE'")), \
            "resolution guessed a winner for an ambiguous name"
    finally:
        ws.close()


def test_stale_call_edges_are_removed_when_the_call_goes_away(repo):
    repo.write("m.py", "def a():\n    return b()\n\ndef b():\n    return 1\n")
    repo.commit("with call")
    ws = ws_mod.open_workspace(str(repo.root))
    try:
        ws_mod.ensure_indexed(ws, force_full=True)
        assert rows(ws.store.execute(
            "SELECT 1 FROM code_edges WHERE kind='CALLS' AND status='ACTIVE'"))
    finally:
        ws.close()

    repo.write("m.py", "def a():\n    return 0\n\ndef b():\n    return 1\n")
    repo.commit("call removed")
    ws = ws_mod.open_workspace(str(repo.root))
    try:
        ws_mod.ensure_indexed(ws, force_full=True)
        assert not rows(ws.store.execute(
            "SELECT 1 FROM code_edges e JOIN symbols a ON a.symbol_id=e.from_id"
            " WHERE a.name='a' AND e.kind='CALLS' AND e.status='ACTIVE'")), \
            "an edge survived the call that produced it"
    finally:
        ws.close()


# --------------------------------------------------------------------- impact

def test_impact_upstream_finds_the_caller_and_the_test(workspace):
    result = graph.impact(workspace.store, "refresh_session", "upstream", depth=2)
    reached = {a["symbol_path"] for a in result["affected"]}
    assert "post_refresh" in reached
    assert "test_parallel_refresh_regression" in reached
    assert result["counts"]["total"] == len(result["affected"])


def test_impact_reports_an_ambiguous_name_instead_of_picking_one(repo):
    repo.write("one.py", "def handle():\n    return 1\n")
    repo.write("two.py", "def handle():\n    return 2\n")
    repo.commit("two handles")
    ws = ws_mod.open_workspace(str(repo.root))
    try:
        ws_mod.ensure_indexed(ws, force_full=True)
        result = graph.impact(ws.store, "handle")
        assert result["ambiguous"] is True
        assert result["total_candidates"] == 2

        narrowed = graph.impact(ws.store, "handle", file_hint="one.py")
        assert narrowed["target"]["path"] == "one.py"
    finally:
        ws.close()


def test_impact_says_lower_bound_when_callers_were_dropped(repo):
    """The whole point: an empty result must not read as proof of none."""
    repo.write("one.py", "def handle():\n    return 1\n")
    repo.write("two.py", "def handle():\n    return 2\n")
    repo.write("caller.py", "def go():\n    return handle()\n")
    repo.commit("ambiguous")
    ws = ws_mod.open_workspace(str(repo.root))
    try:
        ws_mod.ensure_indexed(ws, force_full=True)
        result = graph.impact(ws.store, "handle", "upstream", file_hint="one.py")
        assert result["affected"] == []
        assert result["epistemic"] == "lower-bound"
        assert result["causes"]["ambiguous_call_sites"] >= 1
        assert result["boundaries"], "lower-bound with no explanation"
    finally:
        ws.close()


def test_impact_is_exact_when_nothing_was_dropped(workspace):
    result = graph.impact(workspace.store, "rotate_token", "upstream", depth=2)
    assert result["epistemic"] == "exact"
    assert result["causes"]["ambiguous_call_sites"] == 0


def test_min_confidence_excludes_unproven_edges(workspace):
    loose = graph.impact(workspace.store, "refresh_session", "upstream", depth=3)
    strict = graph.impact(workspace.store, "refresh_session", "upstream", depth=3,
                          min_confidence=0.9)
    assert strict["counts"]["total"] <= loose["counts"]["total"]
    assert all(a["confidence"] >= 0.9 for a in strict["affected"])


# ---------------------------------------------------------------------- trace

def test_trace_finds_a_path_and_names_its_weakest_link(workspace):
    result = graph.trace(workspace.store, "post_refresh", "rotate_token", max_depth=4)
    assert result["found"] is True
    assert [n["symbol_path"] for n in result["path"]][0] == "post_refresh"
    assert [n["symbol_path"] for n in result["path"]][-1] == "rotate_token"
    assert 0.0 < result["weakest_link_confidence"] <= 1.0


def test_trace_without_a_path_does_not_claim_there_is_none(workspace):
    result = graph.trace(workspace.store, "rotate_token", "post_refresh", max_depth=3)
    assert result["found"] is False
    assert "not proof" in result["note"]


# --------------------------------------------------------------------- cycles

def test_import_cycles_are_found_and_counted_by_component(repo):
    repo.write("pkg/__init__.py", "")
    repo.write("pkg/a.py", "from pkg import b\n\ndef fa():\n    return b.fb()\n")
    repo.write("pkg/b.py", "from pkg import c\n\ndef fb():\n    return c.fc()\n")
    repo.write("pkg/c.py", "from pkg import a\n\ndef fc():\n    return a.fa()\n")
    repo.write("pkg/d.py", "from pkg import e\n\ndef fd():\n    return e.fe()\n")
    repo.write("pkg/e.py", "def fe():\n    return 1\n")
    repo.commit("circular imports")
    ws = ws_mod.open_workspace(str(repo.root))
    try:
        ws_mod.ensure_indexed(ws, force_full=True)
        result = graph.import_cycles(ws.store)
        assert result["status"] == "cycles_found"
        assert result["component_count"] == 1
        members = set(result["components"][0]["members"])
        assert members == {"pkg/a.py", "pkg/b.py", "pkg/c.py"}
        assert "pkg/d.py" not in members
    finally:
        ws.close()


def test_a_clean_repository_reports_no_cycles(workspace):
    result = graph.import_cycles(workspace.store)
    assert result["status"] in ("clean", "no_import_graph")
    assert result["component_count"] == 0


def test_cycle_detection_survives_a_chain_deeper_than_the_recursion_limit(repo):
    """Tarjan must be iterative.

    A recursive implementation raises RecursionError partway through a health
    check, which is a crash in the middle of an answer rather than a wrong
    answer - and it only shows up on the large repositories that need it most.
    """
    depth = 1200
    repo.write("pkg/__init__.py", "")
    for i in range(depth):
        nxt = f"from pkg import m{i + 1}\n" if i + 1 < depth else ""
        repo.write(f"pkg/m{i}.py", f"{nxt}\ndef f{i}():\n    return {i}\n")
    repo.commit("deep chain")
    ws = ws_mod.open_workspace(str(repo.root))
    try:
        ws_mod.ensure_indexed(ws, force_full=True)
        result = graph.import_cycles(ws.store)
        assert result["component_count"] == 0
    finally:
        ws.close()


def test_a_store_written_before_import_resolution_heals_on_reindex(repo):
    """A full reindex must re-store rows an older extractor wrote.

    A content hash proves the FILE did not move. It cannot prove the EXTRACTOR
    did not, so an existing store short-circuited on the hash and kept its
    legacy statement-text IMPORTS edges forever - a migration that silently did
    nothing, on exactly the repositories that already had knowledge in them.
    """
    repo.write("pkg/__init__.py", "")
    repo.write("pkg/util.py", "def helper():\n    return 1\n")
    repo.write("pkg/app.py", "from pkg import util\n\ndef go():\n    return util.helper()\n")
    repo.commit("initial")

    ws = ws_mod.open_workspace(str(repo.root))
    try:
        ws_mod.ensure_indexed(ws, force_full=True)
        # Simulate a store written by the previous version: imports resolved,
        # but the specifiers never persisted and the edges in the old shape.
        with ws.store:
            ws.store.execute(
                "UPDATE files SET imports_raw = NULL, extract_version = NULL")
            ws.store.execute("DELETE FROM code_edges WHERE kind='IMPORTS'")
            ws.store.execute(
                "INSERT INTO code_edges (edge_id, from_id, to_id, kind, edge_class,"
                " status, confidence, source, created_at, last_verified_at)"
                " SELECT 'edge_legacy'||file_id, file_id, 'import:from pkg import util',"
                " 'IMPORTS','deterministic','ACTIVE',1.0,'tree-sitter','x','x'"
                " FROM files WHERE path LIKE '%app.py'")
        assert rows(ws.store.execute(
            "SELECT 1 FROM code_edges WHERE to_id LIKE 'import:%'"))
    finally:
        ws.close()

    ws = ws_mod.open_workspace(str(repo.root))
    try:
        ws_mod.ensure_indexed(ws, force_full=True)
        assert not rows(ws.store.execute(
            "SELECT 1 FROM code_edges WHERE to_id LIKE 'import:%'")), \
            "legacy IMPORTS edges survived a full reindex"
        assert not rows(ws.store.execute(
            "SELECT 1 FROM files WHERE imports_raw IS NULL AND status='ACTIVE'")), \
            "imports_raw was never backfilled"
        assert not rows(ws.store.execute(
            "SELECT 1 FROM files WHERE COALESCE(extract_version,0) < ?"
            "   AND status='ACTIVE'", (parsing.EXTRACT_VERSION,))), \
            "extract_version was never stamped"

        paths = {r["file_id"]: r["path"] for r in rows(ws.store.execute(
            "SELECT file_id, path FROM files"))}
        linked = {(paths.get(e["from_id"]), paths.get(e["to_id"]))
                  for e in rows(ws.store.execute(
                      "SELECT from_id, to_id FROM code_edges WHERE kind='IMPORTS'"
                      " AND status='ACTIVE'"))}
        assert ("pkg/app.py", "pkg/util.py") in linked
    finally:
        ws.close()


def test_a_deferred_import_does_not_count_as_a_cycle(repo):
    """A function-level import is the standard way to break a cycle.

    Counting it reports the fix as the fault, and a checker that flags
    deliberately-broken cycles is one people learn to ignore.
    """
    repo.write("pkg/__init__.py", "")
    repo.write("pkg/a.py", "from pkg import b\n\ndef fa():\n    return b.fb()\n")
    repo.write("pkg/b.py",
               "def fb():\n    from pkg import a\n    return a.fa()\n")
    repo.commit("cycle broken by a deferred import")
    ws = ws_mod.open_workspace(str(repo.root))
    try:
        ws_mod.ensure_indexed(ws, force_full=True)
        # The dependency is still recorded - it is real - but marked deferred.
        deferred = rows(ws.store.execute(
            "SELECT e.source FROM code_edges e JOIN files f ON f.file_id=e.from_id"
            " WHERE e.kind='IMPORTS' AND f.path LIKE '%b.py'"
            "   AND e.source='tree-sitter:deferred'"))
        assert deferred, "the deferred import was not recorded at all"

        result = graph.import_cycles(ws.store)
        assert result["component_count"] == 0, \
            "a function-level import was counted as closing a cycle"
    finally:
        ws.close()


def test_a_module_level_import_still_counts_when_also_deferred_elsewhere(repo):
    """Eager wins: importing a module both ways is an eager import."""
    repo.write("pkg/__init__.py", "")
    repo.write("pkg/a.py", "from pkg import b\n\ndef fa():\n    return b.fb()\n")
    repo.write("pkg/b.py",
               "from pkg import a\n\ndef fb():\n    from pkg import a as a2\n"
               "    return a.fa() + a2.fa()\n")
    repo.commit("eager and deferred to the same module")
    ws = ws_mod.open_workspace(str(repo.root))
    try:
        ws_mod.ensure_indexed(ws, force_full=True)
        result = graph.import_cycles(ws.store)
        assert result["component_count"] == 1, \
            "an eager import was masked by a deferred one to the same module"
    finally:
        ws.close()


def test_a_partially_migrated_row_is_not_mistaken_for_a_current_one(repo):
    """The failure a proxy staleness detector cannot survive.

    An earlier fix used "imports_raw IS NULL" to mean "this row is stale". It
    worked for exactly one version step. Then decorators were added, and every
    file a partial run had already touched had a non-NULL imports_raw, looked
    migrated, and was stranded without decorators - which silently emptied the
    entry-point table on the live store. Here the row is left in precisely that
    half-way state: imports present, decorators missing, version behind.
    """
    repo.write("app.py", "@mcp.tool()\ndef handler(x):\n    return x\n")
    repo.commit("decorated handler")

    ws = ws_mod.open_workspace(str(repo.root))
    try:
        ws_mod.ensure_indexed(ws, force_full=True)
        assert rows(ws.store.execute("SELECT 1 FROM entry_points WHERE kind='tool'"))
        with ws.store:
            # Exactly what the intermediate build left behind.
            ws.store.execute("UPDATE files SET extract_version = 2")
            ws.store.execute("UPDATE symbols SET decorators = NULL")
            ws.store.execute("DELETE FROM entry_points")
    finally:
        ws.close()

    ws = ws_mod.open_workspace(str(repo.root))
    try:
        ws_mod.ensure_indexed(ws, force_full=True)
        assert rows(ws.store.execute(
            "SELECT 1 FROM entry_points WHERE kind='tool'")), \
            "a row behind the extractor version was treated as current"
    finally:
        ws.close()
