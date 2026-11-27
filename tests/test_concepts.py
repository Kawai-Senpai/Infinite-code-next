"""Reviewed concepts improve recall only when their source can be verified."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from icn import concepts, search, workspace as ws_mod
from mcp_client import McpStdioClient


def target(ws, symbol="RefreshCoordinator.acquire"):
    row = ws.store.execute(
        "SELECT last_known_path, content_fingerprint FROM symbols WHERE symbol_path=? AND status='ACTIVE'",
        (symbol,),
    ).fetchone()
    return {"path": row[0], "symbol": symbol, "fingerprint": row[1]}


def define(ws, *entries):
    (ws.root / concepts.FILENAME).write_text(json.dumps({"concepts": entries}), encoding="utf-8")


def mapping(ws, symbol="RefreshCoordinator.acquire", name="Session admission"):
    return {"name": name, "aliases": ["request gating"], "targets": [target(ws, symbol)]}


def find(ws, query, **options):
    return search.investigate(ws.store, ws.catalog, ws.root, query, conversations=False,
                              find_problems=False, **options)


def rank_signature(result):
    return [(capsule["symbol"], capsule["score"]) for capsule in result["capsules"]]


def test_reviewed_phrase_routes_a_broad_query_to_source(workspace):
    define(workspace, mapping(workspace))
    query = "Where does request-gating coordinate concurrent sign-ins?"
    baseline = find(workspace, query, use_concepts=False)
    result = find(workspace, query)

    assert not any(c["symbol"] == "RefreshCoordinator.acquire" for c in baseline["capsules"])
    capsule = result["capsules"][0]
    assert capsule["symbol"] == "RefreshCoordinator.acquire"
    assert capsule["score_breakdown"]["reviewed_concept"] > 0
    assert result["grounding"]["matches"] == [{
        "concept": "Session admission", "phrase": "request gating",
        "targets": [{"symbol": "RefreshCoordinator.acquire", "path": "auth.py",
                     "status": "grounded", "lines": [int(line) for line in capsule["lines"].split("-")]}],
    }]
    assert "concurrent" in result["grounding"]["unmapped_terms"]
    assert "gating" not in result["grounding"]["unmapped_terms"]
    assert "request-gating" not in result["grounding"]["unmapped_terms"]


def test_exact_code_and_ordinary_search_keep_existing_ranking(workspace):
    define(workspace, mapping(workspace, "rotate_token"))
    for query in ("RefreshCoordinator.acquire", "Find RefreshCoordinator.acquire during request gating",
                  "Look in api.py for request gating", "Where is request gatekeeping implemented?"):
        result = find(workspace, query)
        baseline = find(workspace, query, use_concepts=False)
        assert rank_signature(result) == rank_signature(baseline)
        assert not result["grounding"]["used_for_ranking"]
    assert find(workspace, "Find RefreshCoordinator.acquire during request gating")["grounding"]["ranking_note"]


def test_named_file_allows_only_its_own_reviewed_targets(workspace):
    define(workspace, mapping(workspace, "rotate_token"))
    result = find(workspace, "Look in auth.py for request gating")
    assert result["capsules"][0]["symbol"] == "rotate_token"
    assert result["grounding"]["used_for_ranking"]


@pytest.mark.parametrize("query", ["`RefreshCoordinator.acquire`", "`RefreshCoordinator.acquire()`"])
def test_unique_qualified_symbol_gets_priority(workspace, query):
    result = find(workspace, query)
    assert result["capsules"][0]["symbol"] == "RefreshCoordinator.acquire"
    assert result["capsules"][0]["score_breakdown"]["exact_qualified_symbol"] > 0
    partial = find(workspace, "RefreshCoordinator.acquire", index_ready=False)
    assert all("exact_qualified_symbol" not in c["score_breakdown"] for c in partial["capsules"])


@pytest.mark.parametrize("query", [
    "RefreshCoordinator.acquire",
    "Find RefreshCoordinator.acquire during request gating",
])
def test_unused_grounding_does_not_displace_code_at_a_tight_budget(workspace, query):
    define(workspace, mapping(workspace, "rotate_token"))
    baseline = find(workspace, query, use_concepts=False)
    assert len(baseline["capsules"]) >= 2
    limit = sum(search._estimate_tokens(c) for c in baseline["capsules"][:2])
    baseline = find(workspace, query, budget=limit, use_concepts=False)
    result = find(workspace, query, budget=limit)
    assert len(baseline["capsules"]) == 2
    assert rank_signature(result) == rank_signature(baseline)
    assert result["budget"] == baseline["budget"]
    assert "grounding" not in result


def test_supported_mappings_stay_visible_without_overriding_code_drift(workspace, project):
    define(workspace, mapping(workspace))
    project.write("auth.py", project.read("auth.py").replace("        return lock", "        return None"))
    assert ws_mod.ensure_indexed(workspace)["index_state"] == "ready"

    result = find(workspace, "request gating", depth=0)
    evidence = result["grounding"]["matches"][0]["targets"][0]
    assert evidence["status"] == "needs_review"
    assert not result["grounding"]["used_for_ranking"]
    assert rank_signature(result) == rank_signature(find(workspace, "request gating", depth=0, use_concepts=False))


@pytest.mark.parametrize(("changes", "index_ready", "expected"), [
    ({"path": "other.py"}, True, "unresolved"),
    ({}, False, "index_incomplete"),
])
def test_wrong_source_path_or_partial_index_does_not_boost(workspace, changes, index_ready, expected):
    entry = mapping(workspace)
    entry["targets"][0].update(changes)
    define(workspace, entry)
    result = find(workspace, "request gating", index_ready=index_ready)
    assert result["grounding"]["matches"][0]["targets"][0]["status"] == expected
    assert result["grounding"]["used_for_ranking"] is False


def test_shared_alias_is_reported_as_ambiguous(workspace):
    define(workspace, mapping(workspace), mapping(workspace, "rotate_token", "Token issuance"))
    response = find(workspace, "request gating")
    assert response["grounding"]["matches"] == [{
        "phrase": "request gating", "status": "ambiguous_phrase",
        "concepts": ["Session admission", "Token issuance"],
    }]
    assert response["grounding"]["used_for_ranking"] is False
    assert find(workspace, "session admission")["grounding"]["used_for_ranking"] is True


def test_alias_context_disambiguates_shared_project_vocabulary(workspace):
    admission = mapping(workspace)
    admission["aliases"] = [{"phrase": "request gating", "requires_any": ["sign in", "session"]}]
    issuance = mapping(workspace, "rotate_token", "Token issuance")
    issuance["aliases"] = [{"phrase": "request gating", "requires_any": ["rotate token"]}]
    define(workspace, admission, issuance)

    for query, expected in [
        ("How does request gating work?", None),
        ("Can anonymous sessions see request gating?", None),
        ("How does request gating work during SIGN-IN?", "RefreshCoordinator.acquire"),
        ("Where does request gating rotate tokens?", None),
        ("Where does request gating rotate token state?", "rotate_token"),
    ]:
        result = find(workspace, query)
        assert result["grounding"]["used_for_ranking"] == (expected is not None)
        if expected:
            assert result["capsules"][0]["symbol"] == expected
        else:
            assert not result["grounding"]["matches"]

    conflict = find(workspace, "Does request gating rotate token state on sign in?")
    assert conflict["grounding"]["matches"][0]["status"] == "ambiguous_phrase"
    assert not conflict["grounding"]["used_for_ranking"]


def test_invalid_alias_context_falls_back_to_search(workspace):
    entry = mapping(workspace)
    entry["aliases"] = [{"phrase": "request gating", "requires_any": ["gating"]}]
    define(workspace, entry)
    result = find(workspace, "request gating")
    assert result["grounding"]["status"] == "invalid"
    assert rank_signature(result) == rank_signature(find(workspace, "request gating", use_concepts=False))


def test_missing_or_invalid_glossary_keeps_search_available(workspace):
    plain = find(workspace, "RefreshCoordinator.acquire")
    assert "grounding" not in plain
    file = workspace.root / concepts.FILENAME
    for content in (b"{no", b"x" * (concepts.MAX_BYTES + 1)):
        file.write_bytes(content)
        result = find(workspace, "RefreshCoordinator.acquire")
        assert result["grounding"]["status"] == "invalid"
        assert rank_signature(result) == rank_signature(plain)
    entry = mapping(workspace)
    entry["targets"][0]["path"] = "../elsewhere.py"
    define(workspace, entry)
    assert find(workspace, "request gating")["grounding"]["status"] == "invalid"


def test_anchor_command_emits_copyable_target(workspace, capsys):
    assert concepts.main(["auth.py", "RefreshCoordinator.acquire",
                          "--root", str(workspace.root)]) == 0
    assert json.loads(capsys.readouterr().out) == target(workspace)


def test_agent_can_read_glossary_evidence_and_opt_out_over_mcp(workspace):
    define(workspace, mapping(workspace))
    other = workspace.root.parent / "other"
    other.mkdir()
    src = str(Path(__file__).resolve().parents[1] / "src")
    with McpStdioClient(workspace.root, {"PYTHONPATH": src, "ICN_EMBED_MODEL": "none"}) as client:
        arguments = {"query": "Where is request gating enforced?", "conversations": False}
        result = client.call("investigate", arguments)
        baseline = client.call("investigate", {**arguments, "use_concepts": False})
        across_repos = client.call("investigate", {**arguments, "roots": [str(workspace.root), str(other)]})
    assert result["index_state"] == "ready"
    assert result["capsules"][0]["symbol"] == "RefreshCoordinator.acquire"
    assert result["grounding"]["used_for_ranking"]
    assert "grounding" not in baseline
    assert across_repos["repositories"][0]["grounding"]["used_for_ranking"]
    assert "grounding" not in across_repos["repositories"][1]
