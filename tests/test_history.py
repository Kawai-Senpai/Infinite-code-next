"""Change coupling, hotspots and dead-code candidates.

These read git history rather than the AST, so the fixture builds real commits
and the assertions are about what history can and cannot support. The recurring
theme is the boundary: every answer here is a lower bound, and the tests that
matter most are the ones checking it says so.
"""

from __future__ import annotations

import pytest

from icn import history


@pytest.fixture
def coupled(repo):
    """A repository where two files always change together and one never does.

    `alpha` and `beta` share no import, so the code graph cannot explain the
    pair - which is exactly the case coupling exists to surface.
    """
    repo.write("alpha.py", "def alpha():\n    return 1\n")
    repo.write("beta.py", "def beta():\n    return 1\n")
    repo.write("lonely.py", "def lonely():\n    return 1\n")
    repo.commit("initial")

    for n in range(2, 5):
        repo.write("alpha.py", f"def alpha():\n    return {n}\n")
        repo.write("beta.py", f"def beta():\n    return {n}\n")
        repo.commit(f"change both {n}")

    repo.write("lonely.py", "def lonely():\n    return 2\n")
    repo.commit("change lonely alone")

    from icn import workspace as ws_mod

    ws = ws_mod.open_workspace(str(repo.root))
    ws_mod.ensure_indexed(ws)
    yield ws
    ws.close()


def pair_of(result, left: str, right: str):
    for item in result["pairs"]:
        if {item["left"], item["right"]} == {left, right}:
            return item
    return None


# ------------------------------------------------------------------- coupling

def test_files_that_always_change_together_are_coupled(coupled):
    found = history.change_coupling(coupled.store, coupled.root)
    pair = pair_of(found, "alpha.py", "beta.py")
    assert pair is not None
    assert pair["co_changes"] == 4
    assert pair["jaccard"] == 1.0


def test_a_file_that_changes_alone_is_not_coupled_to_it(coupled):
    found = history.change_coupling(coupled.store, coupled.root)
    assert pair_of(found, "alpha.py", "lonely.py") is None


def test_coupling_says_when_the_code_graph_cannot_explain_the_pair(coupled):
    """A pair with no import between them is the whole point of the query."""
    found = history.change_coupling(coupled.store, coupled.root)
    pair = pair_of(found, "alpha.py", "beta.py")
    assert pair["also_imports"] is False
    assert found["counts"]["without_an_import_edge"] >= 1


def test_coupling_never_claims_to_be_exact(coupled):
    found = history.change_coupling(coupled.store, coupled.root)
    assert found["epistemic"] == "lower-bound"
    assert any("committed" in b for b in found["boundaries"])


def test_min_support_excludes_a_one_off_pair(coupled):
    strict = history.change_coupling(coupled.store, coupled.root, min_support=10)
    assert strict["pairs"] == []


def test_a_bulk_commit_does_not_couple_everything_it_touched(repo):
    """One reformat can add more spurious pairs than the real signal."""
    for index in range(history.BULK_COMMIT_FILES + 5):
        repo.write(f"mod{index}.py", f"def f{index}():\n    return 1\n")
    repo.commit("bulk add")
    for index in range(history.BULK_COMMIT_FILES + 5):
        repo.write(f"mod{index}.py", f"def f{index}():\n    return 2\n")
    repo.commit("bulk reformat")

    from icn import workspace as ws_mod

    ws = ws_mod.open_workspace(str(repo.root))
    ws_mod.ensure_indexed(ws)
    try:
        found = history.change_coupling(ws.store, ws.root)
        assert found["pairs"] == []
        assert found["history"]["bulk_commits_excluded"] == 2
        assert any("bulk" in b.lower() for b in found["boundaries"])
    finally:
        ws.close()


def test_a_repository_with_no_git_history_says_so_rather_than_returning_nothing(
        tmp_path, coupled):
    """'no coupling' and 'no history to read' are different facts."""
    found = history.change_coupling(coupled.store, tmp_path / "not-a-repo")
    assert found["history"]["available"] is False
    assert any("no git history" in b.lower() for b in found["boundaries"])


# ------------------------------------------------------------------- hotspots

def test_a_frequently_changed_file_outranks_a_stable_one(coupled):
    found = history.hotspots(coupled.store, coupled.root)
    ranked = [item["path"] for item in found["files"]]
    assert "alpha.py" in ranked
    assert ranked.index("alpha.py") < ranked.index("lonely.py")


def test_hotspots_report_the_parts_of_the_score_not_just_the_score(coupled):
    found = history.hotspots(coupled.store, coupled.root)
    item = found["files"][0]
    assert {"commits", "churn", "symbols", "import_degree", "call_degree",
            "structural_weight", "score"} <= set(item)


def test_hotspots_do_not_claim_to_measure_complexity(coupled):
    """Naming a length-derived number 'complexity' would be a guess in a suit."""
    found = history.hotspots(coupled.store, coupled.root)
    assert "complexity" not in found["files"][0]
    assert "NOT measured" in found["scoring"]


def test_a_file_with_no_commits_in_the_window_is_not_a_hotspot(repo):
    """Indexed but never committed means no change history, so no hotspot.

    A file scores on commits x structural weight, and a file that has never
    been committed has no churn to weigh - listing it at zero would put every
    untracked scratch file in the ranking.
    """
    repo.write("committed.py", "def committed():\n    return 1\n")
    repo.commit("one")
    repo.write("untracked.py", "def untracked():\n    return 1\n")

    from icn import workspace as ws_mod

    ws = ws_mod.open_workspace(str(repo.root))
    ws_mod.ensure_indexed(ws)
    try:
        found = history.hotspots(ws.store, ws.root)
        ranked = [item["path"] for item in found["files"]]
        assert "committed.py" in ranked
        assert "untracked.py" not in ranked
    finally:
        ws.close()


# ------------------------------------------------------------------ dead code

@pytest.fixture
def orchard(repo):
    repo.write("live.py", (
        "def used():\n    return 1\n\n\n"
        "def caller():\n    return used()\n"))
    repo.write("dead.py", "def never_called():\n    return 1\n")
    repo.write("runtime.py", (
        "class Thing:\n"
        "    def __init__(self):\n        self.x = 1\n\n"
        "    def __enter__(self):\n        return self\n"))
    repo.commit("orchard")

    from icn import workspace as ws_mod

    ws = ws_mod.open_workspace(str(repo.root))
    ws_mod.ensure_indexed(ws)
    yield ws
    ws.close()


def names_of(result) -> set[str]:
    return {c["symbol_path"] for c in result["candidates"]}


def test_an_uncalled_function_is_a_candidate(orchard):
    assert "never_called" in names_of(history.dead_code(orchard.store))


def test_a_called_function_is_not_a_candidate(orchard):
    assert "used" not in names_of(history.dead_code(orchard.store))


def test_dunder_methods_are_excluded_entirely(orchard):
    """A dunder has no written call site in any correct program.

    Reporting one is a false positive by construction, not a candidate with a
    caveat - and the first run of this query returned __init__ and __enter__
    at the top of the list.
    """
    found = history.dead_code(orchard.store)
    assert not [c for c in found["candidates"] if c["name"].startswith("__")]
    assert found["counts"]["runtime_invoked_excluded"] >= 2


def test_dead_code_is_labelled_candidates_and_never_a_verdict(orchard):
    found = history.dead_code(orchard.store)
    assert found["epistemic"] == "lower-bound"
    assert "not proven unused" in " ".join(found["boundaries"])
    assert "not a delete list" in found["note"]


def test_every_candidate_carries_a_confidence(orchard):
    found = history.dead_code(orchard.store)
    assert all(c["confidence"] in ("higher", "lower") for c in found["candidates"])


def test_a_name_that_an_unresolved_call_site_mentions_is_downgraded(repo):
    """Two definitions of one name make every call to it unattributable.

    The callee is not dead - resolution simply refused to choose - so it must
    come back weakened rather than filtered away or reported confidently.
    """
    repo.write("one.py", "def duplicated():\n    return 1\n")
    repo.write("two.py", "def duplicated():\n    return 2\n")
    repo.write("caller.py", "def go():\n    return duplicated()\n")
    repo.commit("ambiguous")

    from icn import workspace as ws_mod

    ws = ws_mod.open_workspace(str(repo.root))
    ws_mod.ensure_indexed(ws)
    try:
        found = history.dead_code(ws.store)
        weakened = [c for c in found["candidates"] if c["name"] == "duplicated"]
        assert weakened, "an ambiguous callee must still be reported"
        assert all(c["confidence"] == "lower" for c in weakened)
        assert all(c["unattributed_call_sites"] >= 1 for c in weakened)
        assert all(c["weakened_by"] for c in weakened)
    finally:
        ws.close()


def test_higher_confidence_candidates_are_ranked_first(repo):
    repo.write("plain.py", "def orphan():\n    return 1\n")
    repo.write("one.py", "def shared():\n    return 1\n")
    repo.write("two.py", "def shared():\n    return 2\n")
    repo.write("caller.py", "def go():\n    return shared()\n")
    repo.commit("mixed")

    from icn import workspace as ws_mod

    ws = ws_mod.open_workspace(str(repo.root))
    ws_mod.ensure_indexed(ws)
    try:
        found = history.dead_code(ws.store)
        grades = [c["confidence"] for c in found["candidates"]]
        assert grades == sorted(grades, key=lambda g: g != "higher")
    finally:
        ws.close()
