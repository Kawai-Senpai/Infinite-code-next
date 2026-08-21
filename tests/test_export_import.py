"""Export, share and import a repository's knowledge.

The point of these formats is that knowledge outlives the machine it was
recorded on. So the tests care about two things: that a round trip loses
nothing, and that importing someone else's knowledge never lets it
impersonate your own.
"""

from __future__ import annotations

import json

from conftest import Repo, git, record_baseline

from icn import explorer, export
from icn import workspace as ws_mod
from icn.db import one, rows


def graph_of(ws):
    return export.build_graph(ws.store, ws.catalog, ws.repo_id)


def test_the_graph_contains_code_and_knowledge_together(workspace):
    record_baseline(workspace)
    graph = graph_of(workspace)

    kinds = graph["stats"]["by_node_kind"]
    assert kinds.get("symbol", 0) > 0
    assert kinds.get("memory", 0) > 0
    assert kinds.get("file", 0) > 0
    assert graph["stats"]["edges"] > 0

    edge_kinds = set(graph["stats"]["by_edge_kind"])
    assert "CALLS" in edge_kinds, "code structure must be present"
    assert "ANCHORED_TO" in edge_kinds, "knowledge must be attached to code"


def test_every_edge_points_at_a_node_that_exists(workspace):
    """A dangling edge crashes any renderer that trusts the data."""
    record_baseline(workspace)
    graph = graph_of(workspace)
    known = {n["id"] for n in graph["nodes"]}
    for edge in graph["edges"]:
        assert edge["from"] in known and edge["to"] in known


def test_markdown_round_trips_without_loss(workspace):
    record_baseline(workspace)
    graph = graph_of(workspace)

    recovered = export.from_markdown(export.to_markdown(graph))
    assert recovered is not None
    assert len(recovered["nodes"]) == len(graph["nodes"])
    assert len(recovered["edges"]) == len(graph["edges"])


def test_markdown_is_readable_on_its_own(workspace):
    """A knowledge base nobody can read without the tool is one nobody checks."""
    record_baseline(workspace)
    text = export.to_markdown(graph_of(workspace))

    assert text.startswith("# Knowledge:")
    assert "Redis mutex" in text, "the prose must carry the actual knowledge"
    assert "## Invariant" in text or "## Warning" in text
    assert "applies to:" in text, "a reader needs to know what a memory governs"


def test_a_bundle_carries_both_forms(workspace, tmp_path):
    record_baseline(workspace)
    target = export.write_bundle(graph_of(workspace), tmp_path / "k.icn")

    import zipfile
    with zipfile.ZipFile(target) as bundle:
        names = set(bundle.namelist())
        assert {"knowledge.md", "graph.json", "MANIFEST.json"} <= names
        manifest = json.loads(bundle.read("MANIFEST.json"))
        assert manifest["format"] == "infinite-code-next.bundle"

    assert export.read_bundle(target) is not None


def test_import_brings_knowledge_into_another_repository(workspace, project, tmp_path):
    record_baseline(workspace)
    bundle = export.write_bundle(graph_of(workspace), tmp_path / "k.icn")

    other = tmp_path / "other"
    other.mkdir()
    git(other, "init", "-q", "-b", "main")
    git(other, "config", "user.email", "t@t")
    git(other, "config", "user.name", "T")
    receiver = Repo(other)
    receiver.write("auth.py", "def refresh_session(session_id):\n    return session_id\n")
    receiver.commit("initial")

    target = ws_mod.open_workspace(str(other))
    ws_mod.ensure_indexed(target)
    try:
        result = export.import_graph(target.store, target.catalog, target.repo_id,
                                     export.read_bundle(bundle))
        assert result["ok"] and result["imported"] > 0

        imported = rows(target.store.execute(
            "SELECT authority, body FROM memories WHERE status='ACTIVE'"))
        assert imported
        assert all(m["authority"] == "imported" for m in imported), \
            "imported knowledge must not impersonate locally authored knowledge"
        assert any("Redis mutex" in (m["body"] or "") for m in imported)
    finally:
        target.close()


def test_importing_twice_does_not_duplicate(workspace, project, tmp_path):
    record_baseline(workspace)
    bundle = export.write_bundle(graph_of(workspace), tmp_path / "k.icn")

    other = tmp_path / "twice"
    other.mkdir()
    git(other, "init", "-q", "-b", "main")
    git(other, "config", "user.email", "t@t")
    git(other, "config", "user.name", "T")
    Repo(other).write("a.py", "def f():\n    return 1\n")
    Repo(other).commit("initial")

    target = ws_mod.open_workspace(str(other))
    ws_mod.ensure_indexed(target)
    try:
        graph = export.read_bundle(bundle)
        first = export.import_graph(target.store, target.catalog, target.repo_id, graph)
        second = export.import_graph(target.store, target.catalog, target.repo_id, graph)
        assert second["imported"] == 0
        assert second["skipped_duplicates"] == first["imported"]
    finally:
        target.close()


def test_import_does_not_fabricate_test_coverage(workspace, project, tmp_path):
    """Imported memories share one synthetic event.

    Joining rules to tests on that field claimed every imported test covered
    every imported rule - measured at 264 false GUARDED_BY edges from a
    48-memory import, which is worse than none because it silences the
    untested-rule check.
    """
    record_baseline(workspace)
    bundle = export.write_bundle(graph_of(workspace), tmp_path / "k.icn")

    other = tmp_path / "coverage"
    other.mkdir()
    git(other, "init", "-q", "-b", "main")
    git(other, "config", "user.email", "t@t")
    git(other, "config", "user.name", "T")
    Repo(other).write("a.py", "def f():\n    return 1\n")
    Repo(other).commit("initial")

    target = ws_mod.open_workspace(str(other))
    ws_mod.ensure_indexed(target)
    try:
        export.import_graph(target.store, target.catalog, target.repo_id,
                            export.read_bundle(bundle))
        # Reopen: the backfill runs on schema init.
        target.close()
        target = ws_mod.open_workspace(str(other))

        guards = one(target.store.execute(
            "SELECT COUNT(*) AS n FROM memory_edges WHERE kind='GUARDED_BY'"))["n"]
        memories = one(target.store.execute(
            "SELECT COUNT(*) AS n FROM memories WHERE status='ACTIVE'"))["n"]
        assert guards <= memories, \
            f"{guards} coverage edges for {memories} imported memories is fabricated"
    finally:
        target.close()


def test_unreadable_input_is_reported_not_crashed(tmp_path):
    junk = tmp_path / "notes.md"
    junk.write_text("# Just some notes\n\nNothing machine-readable here.\n", encoding="utf-8")
    assert export.read_bundle(junk) is None
    assert export.read_bundle(tmp_path / "missing.icn") is None

    broken = tmp_path / "broken.icn"
    broken.write_bytes(b"not a zip file")
    assert export.read_bundle(broken) is None


def test_the_explorer_page_is_self_contained(workspace, tmp_path):
    """Saving, mailing or committing the file must all work, so nothing may be
    fetched at runtime."""
    record_baseline(workspace)
    page = explorer.render(graph_of(workspace))

    assert "<canvas" in page and "</html>" in page
    for remote in ("http://", "https://", "cdn.", "src=\"//"):
        assert remote not in page.replace("http://127.0.0.1", ""), \
            f"page must not reference {remote}"

    payload = page.split('id="data" type="application/json">', 1)[1].split("</script>", 1)[0]
    assert json.loads(payload)["stats"]["nodes"] > 0


def test_the_page_cannot_be_broken_by_content(workspace, tmp_path):
    """A memory body is arbitrary text; it must not be able to close a tag."""
    from icn import compiler

    compiler.record_event(
        workspace.store, workspace.catalog, workspace.repo_id, workspace.root, workspace.commit,
        {"kind": "note", "summary": "</script><script>window.__pwned=1</script>",
         "warnings": ["</script> and <img onerror=alert(1)> in a body"],
         "symbols": ["refresh_session"]})

    page = explorer.render(graph_of(workspace))
    payload = page.split('id="data" type="application/json">', 1)[1].split("</script>", 1)[0]
    assert json.loads(payload), "an embedded </script> must not truncate the data block"

    # The payload may legitimately contain the text - what must never happen is
    # it escaping the data block and becoming executable markup.
    assert "</script>" not in payload, "a body must not close the data block"
    before_data = page.split('id="data" type="application/json">', 1)[0]
    assert "window.__pwned" not in before_data, "content must not reach the document body"


def test_the_page_offers_every_export_the_cli_does(workspace):
    """Export was CLI-only, which is invisible to anyone using the UI."""
    record_baseline(workspace)
    page = explorer.render(graph_of(workspace))

    for fmt in ('data-fmt="md"', 'data-fmt="json"', 'data-fmt="html"',
                'data-fmt="png"', 'data-fmt="clip"'):
        assert fmt in page, f"export menu is missing {fmt}"
    assert 'id="exportBtn"' in page


def test_browser_and_cli_markdown_agree(workspace):
    """The page rebuilds markdown in JS rather than calling back to the
    server, so the two implementations must not drift: a browser export that
    looks right and silently fails to import is the worst outcome."""
    record_baseline(workspace)
    graph = graph_of(workspace)
    page = explorer.render(graph)

    # The JS mirrors to_markdown(); check the structural contract both sides
    # depend on, not the prose.
    for anchor in ('# Knowledge: ', 'Import with `icn-explore import <this file>`.',
                   '<!-- infinite-code-next:graph -->'):
        assert anchor in export.to_markdown(graph)
        assert anchor in page, f"page markdown builder is missing {anchor!r}"

    # And the memory-kind ordering, which decides section order in both.
    assert "MEM_ORDER" in page
    for kind in ('security', 'invariant', 'warning', 'failed_attempt', 'test_evidence'):
        assert f"'{kind}'" in page


def test_exports_need_no_server(workspace):
    """A saved page must still export. Anything fetching from the origin it
    was served from breaks the moment someone opens it from disk."""
    record_baseline(workspace)
    page = explorer.render(graph_of(workspace))

    js = page.split('id="data"', 1)[1]
    for call in ('fetch(', 'XMLHttpRequest', 'EventSource'):
        assert call not in js, f"export path must not use {call}"
    assert 'URL.createObjectURL' in js, "downloads should come from an in-memory Blob"
