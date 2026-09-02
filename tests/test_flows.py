"""Entry points and functional areas.

An entry point is presented as fact, so the tests here care as much about what
is NOT detected as about what is. An invented route sends a reader somewhere
that does not exist, which is worse than an absent one.
"""

from __future__ import annotations

from icn import flows, graph
from icn import workspace as ws_mod
from icn.db import rows

APP = '''
from fastapi import APIRouter
router = APIRouter()
app = router


@router.get("/users/{uid}")
def get_user(uid):
    return load_user(uid)


@app.post("/users")
def create_user(body):
    return store_user(body)


@router.route("/legacy", methods=["PUT", "DELETE"])
def legacy(body):
    return body


@mcp.tool()
def search_tool(query):
    return load_user(query)


def load_user(uid):
    return uid


def store_user(body):
    return body


def helper_nobody_calls():
    return 1
'''


def _index(repo, sources: dict[str, str]):
    for path, text in sources.items():
        repo.write(path, text)
    repo.commit("fixture")
    ws = ws_mod.open_workspace(str(repo.root))
    ws_mod.ensure_indexed(ws, force_full=True)
    return ws


# ------------------------------------------------------------------ detection

def test_decorated_routes_are_detected_with_verb_and_path(repo):
    ws = _index(repo, {"app.py": APP})
    try:
        found = {r["detail"]: r for r in rows(ws.store.execute(
            "SELECT e.detail, e.kind, s.symbol_path FROM entry_points e"
            " JOIN symbols s ON s.symbol_id = e.symbol_id WHERE e.kind='route'"))}
        assert "GET /users/{uid}" in found
        assert "POST /users" in found
        assert found["GET /users/{uid}"]["symbol_path"] == "get_user"
    finally:
        ws.close()


def test_a_verbless_route_takes_its_methods_from_the_keyword(repo):
    """Flask carries the verb in `methods=`, not in the decorator name."""
    ws = _index(repo, {"app.py": APP})
    try:
        details = {r["detail"] for r in rows(ws.store.execute(
            "SELECT detail FROM entry_points WHERE kind='route'"))}
        assert "PUT|DELETE /legacy" in details
    finally:
        ws.close()


def test_mcp_tool_handlers_are_entry_points(repo):
    ws = _index(repo, {"app.py": APP})
    try:
        tools = rows(ws.store.execute(
            "SELECT detail FROM entry_points WHERE kind='tool'"))
        assert [t["detail"] for t in tools] == ["search_tool"]
    finally:
        ws.close()


def test_an_undecorated_function_is_not_an_entry_point(repo):
    """The point of the pass is that it does not guess."""
    ws = _index(repo, {"app.py": APP})
    try:
        names = {r["symbol_path"] for r in rows(ws.store.execute(
            "SELECT s.symbol_path FROM entry_points e"
            " JOIN symbols s ON s.symbol_id = e.symbol_id"))}
        assert "load_user" not in names
        assert "helper_nobody_calls" not in names
    finally:
        ws.close()


def test_entry_points_disappear_when_the_decorator_does(repo):
    ws = _index(repo, {"app.py": APP})
    try:
        assert rows(ws.store.execute(
            "SELECT 1 FROM entry_points WHERE kind='route'"))
    finally:
        ws.close()

    repo.write("app.py", APP.replace('@router.get("/users/{uid}")\n', ""))
    repo.commit("route removed")
    ws = ws_mod.open_workspace(str(repo.root))
    try:
        ws_mod.ensure_indexed(ws, force_full=True)
        details = {r["detail"] for r in rows(ws.store.execute(
            "SELECT detail FROM entry_points WHERE kind='route'"))}
        assert "GET /users/{uid}" not in details
    finally:
        ws.close()


def test_a_test_function_outside_a_test_file_is_not_a_test_entry_point(repo):
    ws = _index(repo, {"helpers.py": "def test_helper():\n    return 1\n"})
    try:
        assert not rows(ws.store.execute(
            "SELECT 1 FROM entry_points WHERE kind='test'"))
    finally:
        ws.close()


# --------------------------------------------------------------------- triggers

def test_triggers_reports_which_entry_points_reach_a_symbol(repo):
    ws = _index(repo, {"app.py": APP})
    try:
        result = graph.reaching_entry_points(ws.store, "load_user", depth=4)
        reached = {(t["kind"], t["detail"]) for t in result["triggered_by"]}
        assert ("route", "GET /users/{uid}") in reached
        assert ("tool", "search_tool") in reached
        assert ("route", "POST /users") not in reached
    finally:
        ws.close()


def test_an_unreachable_symbol_says_so_without_claiming_it_is_dead(repo):
    ws = _index(repo, {"app.py": APP})
    try:
        result = graph.reaching_entry_points(ws.store, "helper_nobody_calls")
        assert result["triggered_by"] == []
        assert "not by itself proof" in result["note"]
    finally:
        ws.close()


# ----------------------------------------------------------------------- areas

def test_communities_are_deterministic_across_runs(repo):
    """Areas that reshuffle on every index cannot be relied on downstream."""
    ws = _index(repo, {"app.py": APP})
    try:
        first = {r["symbol_id"]: r["community_id"] for r in rows(
            ws.store.execute("SELECT symbol_id, community_id FROM communities"))}
        flows.detect_communities(ws.store)
        second = {r["symbol_id"]: r["community_id"] for r in rows(
            ws.store.execute("SELECT symbol_id, community_id FROM communities"))}
        assert first == second
    finally:
        ws.close()


def test_areas_do_not_collapse_into_one_giant_community(repo):
    """Label propagation put 40% of a real repository in a single area.

    Louvain replaced it. This guards the property that mattered: no single
    area may swallow most of the graph, because an area covering half the
    codebase tells a reader nothing.
    """
    sources = {}
    # Three clusters that only talk to themselves, plus one shared helper.
    for cluster in ("alpha", "beta", "gamma"):
        body = f"def {cluster}_util():\n    return 1\n\n"
        for i in range(6):
            body += (f"def {cluster}_{i}():\n"
                     f"    return {cluster}_util() + {cluster}_{max(0, i - 1)}()\n\n")
        sources[f"{cluster}.py"] = body
    ws = _index(repo, sources)
    try:
        sizes = [r["n"] for r in rows(ws.store.execute(
            "SELECT COUNT(*) n FROM communities GROUP BY community_id"))]
        assert sizes, "no communities computed"
        total = sum(sizes)
        assert max(sizes) < total * 0.8, \
            f"one area swallowed {max(sizes)} of {total} symbols"
    finally:
        ws.close()


def test_areas_report_test_share_rather_than_hiding_test_code(repo):
    ws = _index(repo, {
        "app.py": APP,
        "tests/test_app.py": ("from app import load_user\n\n\n"
                              "def test_load_user():\n    return load_user(1)\n"),
    })
    try:
        result = graph.areas(ws.store)
        assert result["areas"]
        assert all("test_share" in a for a in result["areas"])
    finally:
        ws.close()


def test_entry_point_listing_is_grouped_and_counted(repo):
    ws = _index(repo, {"app.py": APP})
    try:
        result = graph.entry_points(ws.store)
        assert result["counts"]["route"] == 3
        assert result["counts"]["tool"] == 1
        assert result["total"] == sum(result["counts"].values())

        only_tools = graph.entry_points(ws.store, kind="tool")
        assert set(only_tools["counts"]) == {"tool"}
    finally:
        ws.close()


def test_areas_are_named_by_their_dominant_file_not_their_directory(repo):
    """In a flat package every area shares one directory.

    Naming by directory produced six areas all called "src/icn", which
    distinguishes nothing. The file name is what a reader recognises.
    """
    sources = {}
    for cluster in ("alpha", "beta"):
        body = f"def {cluster}_util():\n    return 1\n\n"
        for i in range(5):
            body += (f"def {cluster}_{i}():\n"
                     f"    return {cluster}_util() + {cluster}_{max(0, i - 1)}()\n\n")
        sources[f"pkg/{cluster}.py"] = body
    sources["pkg/__init__.py"] = ""
    ws = _index(repo, sources)
    try:
        names = {a["name"] for a in graph.areas(ws.store)["areas"]}
        assert "pkg" not in names, "areas fell back to the shared directory name"
        assert names & {"alpha", "beta"}, f"unexpected area names: {names}"
    finally:
        ws.close()


def test_triggers_ranks_tests_last_and_caps_each_kind(repo):
    """On a widely-used helper, tests drown the answer.

    Real use on this repository returned 104 triggers, 94 of them tests, which
    buried the six MCP tools and four CLI commands that actually answer "what
    can run this". Counts stay complete; only the listing is a sample.
    """
    body = APP + "\n\n"
    for i in range(30):
        body += f"def spare_{i}():\n    return load_user({i})\n\n"
    sources = {"app.py": body}
    sources["tests/test_many.py"] = (
        "from app import load_user\n\n\n"
        + "\n\n".join(f"def test_case_{i}():\n    return load_user({i})"
                      for i in range(25)) + "\n")
    ws = _index(repo, sources)
    try:
        result = graph.reaching_entry_points(ws.store, "load_user", depth=4)
        kinds = [t["kind"] for t in result["triggered_by"]]
        assert kinds, "no triggers at all"
        # Every non-test kind must appear before the first test.
        first_test = kinds.index("test") if "test" in kinds else len(kinds)
        assert all(k != "test" for k in kinds[:first_test])
        assert kinds.count("test") <= graph.MAX_TRIGGERS_PER_KIND

        # The tally is of everything found, not of what was listed.
        assert result["counts"]["total"] >= len(result["triggered_by"])
        if result["truncated"]:
            assert "counts.by_kind is the full tally" in result["listing_note"]
    finally:
        ws.close()
