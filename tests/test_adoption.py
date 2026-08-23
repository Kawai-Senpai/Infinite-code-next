from __future__ import annotations

from pathlib import Path

from conftest import Repo, git


def _repo(root: Path, name: str, function_name: str) -> Repo:
    path = root / name
    path.mkdir()
    git(path, "init", "-q", "-b", "main")
    git(path, "config", "user.email", "test@example.invalid")
    git(path, "config", "user.name", "Test")
    repo = Repo(path)
    repo.write(f"{name}.py", f"def {function_name}():\n    return '{name}'\n")
    repo.commit("initial")
    return repo


def test_named_sibling_repositories_are_reported_as_out_of_scope(tmp_path):
    from icn.workspace import named_sibling_roots

    first = _repo(tmp_path, "ai-service", "run_ai")
    second = _repo(tmp_path, "be-nf-service", "send_notification")

    found = named_sibling_roots(first.root, "audit ai-service and be-nf-service")
    assert found == [second.root.resolve()]


def test_multi_root_investigation_searches_every_explicit_repository(tmp_path):
    from icn.server import investigate

    first = _repo(tmp_path, "accounting-service", "emit_trip_notification")
    second = _repo(tmp_path, "be-nf-service", "send_trip_notification")

    result = investigate(
        query="trip notification",
        roots=[str(first.root), str(second.root)],
        find_problems=False,
        budget=4000,
    )

    assert result["ok"] is True
    assert result["multi_root"] is True
    assert set(result["resolved_roots"]) == {str(first.root.resolve()), str(second.root.resolve())}
    assert len(result["repositories"]) == 2
    symbols = {capsule["symbol"] for capsule in result["capsules"]}
    assert {"emit_trip_notification", "send_trip_notification"} <= symbols


def test_an_explicit_filename_is_ranked_first(workspace):
    from icn import search

    result = search.investigate(
        workspace.store,
        workspace.catalog,
        workspace.root,
        "review tests/test_auth.py before editing",
        find_problems=False,
        commit=workspace.commit,
    )
    assert result["capsules"]
    assert result["capsules"][0]["path"].replace("\\", "/") == "tests/test_auth.py"


def test_doctor_probe_performs_a_real_stdio_handshake(project):
    from icn.doctor import EXPECTED_TOOLS, probe

    result = probe(project.root, timeout=20)
    assert result["ok"] is True
    assert set(result["tools"]) == EXPECTED_TOOLS
    assert result["workspace"]["symbols_active"] > 0


def test_zero_symbol_indexes_explain_parser_compatibility(workspace, monkeypatch):
    from icn import workspace as ws_mod

    monkeypatch.setattr(
        ws_mod.indexer_mod,
        "index_state",
        lambda _store: {
            "files_active": 2,
            "files_deleted": 0,
            "symbols_active": 0,
            "symbols_deleted": 0,
            "code_edges_active": 0,
            "code_edges_historical": 0,
        },
    )
    report = ws_mod.ensure_indexed(workspace, force_full=True)
    assert "tree-sitter" in report["parser_warning"]
