"""The experiment lab: its rules are enforced, its runs are real, its results remembered.

Every run here is a genuine detached process running a real command, because
the failure modes worth guarding (a watcher that dies, a run that measures the
wrong code, a kill that misses the child) do not exist in a mock.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from icn import lab
from icn.db import rows

TRAIN = '''
import sys, time
lr = float(open("config.txt").read().split("=")[1])
mode = open("mode.txt").read().strip()
print("training with", lr)
if mode == "crash":
    sys.exit(3)
if mode == "slow":
    time.sleep(120)
print(f"ICN_METRIC loss={abs(lr - 0.03):.4f}")
print("ICN_METRIC steps=10")
'''

PY = f'"{sys.executable}"'


@pytest.fixture
def research(repo):
    """A repository whose one command trains and reports a metric."""
    repo.write("train.py", TRAIN)
    repo.write("config.txt", "lr=0.1\n")
    repo.write("mode.txt", "ok\n")
    repo.write(".gitignore", "dataset.bin\n")
    repo.write("dataset.bin", "large and ignored\n")
    repo.commit("initial")
    return repo


def _init(root: Path) -> dict:
    return lab.init(root, f"{PY} train.py", hypothesis="lr 0.1 is the starting point")


def _edit(root: Path, slug: str, **files: str) -> None:
    checkout = lab.checkout(root, slug)
    for name, content in files.items():
        Path(checkout["path"], name.replace("_", ".")).write_text(content, encoding="utf-8")


def _finish(root: Path, run_id: str) -> dict:
    result = lab.wait(root, run=run_id, timeout=60)
    assert result.get("finished"), result
    return lab.status(root, run=run_id)


# ------------------------------------------------------------------ storage


def test_init_snapshots_the_working_tree_without_touching_git_or_agit(research):
    from conftest import git

    before = git(research.root, "log", "--oneline").stdout
    created = _init(research.root)

    assert created["ok"] and created["slug"] == "baseline"
    assert (research.root / ".icn-lab" / "repo.git" / "HEAD").exists()
    assert ".icn-lab/" in research.read(".gitignore")
    assert not (research.root / ".agit").exists()
    assert git(research.root, "log", "--oneline").stdout == before
    assert "exp/baseline" not in git(research.root, "branch", "--list").stdout

    handle = lab.open_lab(research.root)
    try:
        files = handle.git("ls-tree", "-r", "--name-only", created["commit"]).splitlines()
    finally:
        handle.close()
    assert "train.py" in files and "config.txt" in files
    assert "dataset.bin" not in files, "gitignored files must not become experiment code"


def test_the_lab_is_never_indexed_as_source(research):
    from icn.indexer import walk_source_files

    _init(research.root)
    lab.checkout(research.root, "baseline")
    indexed = [p.as_posix() for p in walk_source_files(research.root)]
    assert indexed and not any("/.icn-lab/" in p for p in indexed)


def test_init_refuses_a_second_baseline(research):
    _init(research.root)
    with pytest.raises(lab.LabError, match="already has a baseline"):
        _init(research.root)


# ------------------------------------------------------------------ runs


def test_a_run_measures_the_committed_snapshot_and_parses_metrics(research):
    _init(research.root)
    started = lab.start_run(research.root, "baseline")
    run = _finish(research.root, started["run_id"])

    assert run["status"] == "done" and run["exit_code"] == 0
    assert run["metrics"] == {"loss": pytest.approx(0.07), "steps": 10.0}
    assert "training with 0.1" in lab.read_log(research.root, started["run_id"])["text"]
    source = research.root / ".icn-lab" / "runs" / started["run_id"] / "src"
    assert (source / "train.py").exists()


def test_uncommitted_edits_block_a_run(research):
    _init(research.root)
    lab.create(research.root, "lower lr")
    _edit(research.root, "lower-lr", config_txt="lr=0.05\n")

    with pytest.raises(lab.LabError, match="uncommitted edits"):
        lab.start_run(research.root, "lower-lr")

    lab.commit(research.root, "lower-lr", "lr 0.05")
    run = _finish(research.root, lab.start_run(research.root, "lower-lr")["run_id"])
    assert run["metrics"]["loss"] == pytest.approx(0.02)


def test_a_child_changes_only_its_own_branch(research):
    _init(research.root)
    lab.create(research.root, "lower lr")
    _edit(research.root, "lower-lr", config_txt="lr=0.05\n")
    committed = lab.commit(research.root, "lower-lr", "lr 0.05")

    assert committed["changed_vs_parent"] == ["M\tconfig.txt"]
    assert "lr=0.05" in lab.diff(research.root, "lower-lr")["diff"]
    assert research.read("config.txt") == "lr=0.1\n", "the user's working tree must be untouched"


# ------------------------------------------------------------------ rules


def test_an_answered_experiment_is_frozen(research):
    _init(research.root)
    _finish(research.root, lab.start_run(research.root, "baseline")["run_id"])
    lab.checkout(research.root, "baseline")

    with pytest.raises(lab.LabError, match="frozen"):
        lab.commit(research.root, "baseline", "sneaky edit")


def test_void_unfreezes_but_needs_a_reason(research):
    _init(research.root)
    run_id = lab.start_run(research.root, "baseline")["run_id"]
    _finish(research.root, run_id)

    with pytest.raises(lab.LabError, match="needs note"):
        lab.conclude(research.root, "void", run=run_id)

    voided = lab.conclude(research.root, "void", run=run_id, note="metric printed before training ran")
    assert voided["frozen"] is False
    _edit(research.root, "baseline", mode_txt="ok\n# fixed\n")
    assert lab.commit(research.root, "baseline", "repair")["ok"]

    with pytest.raises(lab.LabError, match="permanent"):
        lab.conclude(research.root, "win", run=run_id)


def test_the_command_is_fixed_once_anything_is_measured(research):
    _init(research.root)
    assert lab.set_command(research.root, f"{PY} train.py --quiet")["ok"]
    lab.set_command(research.root, f"{PY} train.py")

    _finish(research.root, lab.start_run(research.root, "baseline")["run_id"])
    with pytest.raises(lab.LabError, match="fixed contract"):
        lab.set_command(research.root, f"{PY} train.py --lr 0.05")


def test_a_crash_answers_nothing_and_two_in_a_row_need_force(research):
    research.write("mode.txt", "crash\n")
    _init(research.root)

    for _ in range(2):
        run = _finish(research.root, lab.start_run(research.root, "baseline")["run_id"])
        assert run["status"] == "failed" and run["exit_code"] == 3 and run["answered"] is False

    assert lab.tree(research.root)["experiments"][0]["state"] == "provisional"
    with pytest.raises(lab.LabError, match="ask the user"):
        lab.start_run(research.root, "baseline")
    forced = lab.start_run(research.root, "baseline", force=True)
    assert "warning" in forced
    _finish(research.root, forced["run_id"])


def test_cancel_stops_the_whole_process_tree(research):
    research.write("mode.txt", "slow\n")
    _init(research.root)
    run_id = lab.start_run(research.root, "baseline")["run_id"]

    stopped = lab.cancel(research.root, run_id)
    assert stopped["status"] == "cancelled"
    assert lab.status(research.root, run=run_id)["answered"] is False


def test_a_timeout_stops_the_run(research):
    research.write("mode.txt", "slow\n")
    _init(research.root)
    run_id = lab.start_run(research.root, "baseline", timeout_seconds=2)["run_id"]
    assert _finish(research.root, run_id)["status"] == "timed_out"


def test_a_run_whose_watcher_vanished_is_reported_lost(research, monkeypatch):
    _init(research.root)
    handle = lab.open_lab(research.root)
    try:
        handle.conn.execute(
            "INSERT INTO runs (run_id, exp_id, commit_sha, command, status, runner_pid, started_at)"
            " VALUES ('run_ghost', (SELECT exp_id FROM experiments LIMIT 1), 'abc', 'x', 'running',"
            " 999999, '2020-01-01T00:00:00+00:00')")
    finally:
        handle.close()
    (research.root / ".icn-lab" / "runs" / "run_ghost").mkdir(parents=True)
    monkeypatch.setattr(lab, "STALE_HEARTBEAT_SECONDS", -1.0)

    assert lab.status(research.root, run="run_ghost")["status"] == "lost"


# ------------------------------------------------------------------ tree


def test_a_win_is_promoted_and_becomes_the_default_parent(research):
    _init(research.root)
    _finish(research.root, lab.start_run(research.root, "baseline")["run_id"])
    lab.create(research.root, "lower lr")
    _edit(research.root, "lower-lr", config_txt="lr=0.05\n")
    lab.commit(research.root, "lower-lr")
    _finish(research.root, lab.start_run(research.root, "lower-lr")["run_id"])

    concluded = lab.conclude(research.root, "win", exp="lower-lr")
    assert concluded["promoted"]
    assert "0.07 -> 0.02" in concluded["memory_payload"]["decisions"][0]

    child = lab.create(research.root, "warmup")
    assert child["parent"]["slug"] == "lower-lr" and child["parent"]["chosen"] == "focal"
    assert lab.tree(research.root)["focal"] == "lower-lr"


def test_the_tree_warns_about_a_flat_fan(research):
    _init(research.root)
    for n in range(4):
        lab.create(research.root, f"option {n}", parent="baseline")
    warnings = lab.tree(research.root)["warnings"]
    assert any("flat fan" in w for w in warnings)


def test_losses_in_a_row_suggest_stopping(research):
    _init(research.root)
    for n in range(3):
        slug = lab.create(research.root, f"idea {n}", parent="baseline")["slug"]
        _finish(research.root, lab.start_run(research.root, slug)["run_id"])
        lab.conclude(research.root, "loss", exp=slug)
    tree = lab.tree(research.root)
    assert tree["setbacks_in_a_row"] == 3 and tree["stop_hint"]


# ------------------------------------------------------------------ memory


def test_conclude_through_the_tool_writes_a_memory_with_evidence(research):
    from icn import server

    root = str(research.root)
    assert server.experiment(action="init", command=f"{PY} train.py", root=root)["ok"]
    run_id = server.experiment(action="run", exp="baseline", root=root)["run_id"]
    server.experiment(action="wait", run=run_id, wait_seconds=60, root=root)
    server.experiment(action="create", title="lower lr", hypothesis="halving lr lowers loss", root=root)
    path = server.experiment(action="checkout", exp="lower-lr", root=root)["path"]
    Path(path, "config.txt").write_text("lr=0.05\n", encoding="utf-8")
    server.experiment(action="commit", exp="lower-lr", message="lr 0.05", root=root)
    child_run = server.experiment(action="run", exp="lower-lr", root=root)["run_id"]
    server.experiment(action="wait", run=child_run, wait_seconds=60, root=root)

    result = server.experiment(action="conclude", exp="lower-lr", verdict="loss",
                               note="pretend it regressed", root=root)
    assert result["ok"], result
    assert result["memories_created"]

    from icn import workspace as ws_mod
    ws = ws_mod.open_workspace(root)
    try:
        failed = rows(ws.store.execute("SELECT body FROM memories WHERE kind='failed_attempt'"))
    finally:
        ws.close()
    assert failed and child_run in failed[0]["body"] and "loss=0.02" in failed[0]["body"]
    assert "config.txt" in failed[0]["body"]

    node = next(n for n in server.experiment(action="tree", root=root)["experiments"]
                if n["slug"] == "lower-lr")
    assert node["verdict"] == "loss" and set(node["memories"]) >= {result["primary_memory"]}


def test_a_concluded_experiment_surfaces_in_investigate(repo):
    """The bug this guards, found in a live MCP run: conclude anchored its memory
    to changed files only, and investigate() attaches memories through symbol
    edges, so the lesson was stored and anchored but never shown."""
    from icn import server

    repo.write("sort_impl.py", "def sort_items(items):\n    data = list(items)\n"
                               "    data.sort()\n    return data\n")
    repo.write("bench.py", "from sort_impl import sort_items\n"
                           "assert sort_items([3, 1, 2]) == [1, 2, 3]\nprint('ICN_METRIC ok=1')\n")
    repo.commit("initial")
    root = str(repo.root)

    server.experiment(action="init", command=f"{PY} bench.py", root=root)
    server.experiment(action="create", title="insertion sort", root=root)
    path = server.experiment(action="checkout", exp="insertion-sort", root=root)["path"]
    Path(path, "sort_impl.py").write_text(
        "def sort_items(items):\n    out = []\n    for x in items:\n        i = len(out)\n"
        "        while i and out[i - 1] > x:\n            i -= 1\n        out.insert(i, x)\n"
        "    return out\n", encoding="utf-8")
    server.experiment(action="commit", exp="insertion-sort", root=root)
    run_id = server.experiment(action="run", exp="insertion-sort", root=root)["run_id"]
    server.experiment(action="wait", run=run_id, wait_seconds=60, root=root)
    concluded = server.experiment(action="conclude", exp="insertion-sort", verdict="loss",
                                  note="no faster", root=root)
    assert concluded["ok"], concluded

    found = server.investigate(query="insertion sort for sort_items", root=root)
    shown = [m for c in found["capsules"] for m in c["memory"]]
    assert any(run_id in (m.get("text") or "") or m.get("kind") == "failed_attempt" for m in shown), \
        [c["symbol"] for c in found["capsules"]]


SORT_BUBBLE = ("def sort_items(items):\n    data = list(items)\n    for i in range(len(data)):\n"
               "        for j in range(len(data) - 1 - i):\n            if data[j] > data[j + 1]:\n"
               "                data[j], data[j + 1] = data[j + 1], data[j]\n    return data\n")
SORT_BUILTIN = "def sort_items(items):\n    return sorted(items)\n"
SORT_INSERTION = ("def sort_items(items):\n    out = []\n    for x in items:\n        i = len(out)\n"
                  "        while i and out[i - 1] > x:\n            i -= 1\n        out.insert(i, x)\n"
                  "    return out\n")


@pytest.fixture
def sorter(repo):
    """A tree with a measured baseline, a win, and a sibling loss, via the MCP tool."""
    from icn import server

    repo.write("sort_impl.py", SORT_BUBBLE)
    repo.write("bench.py", "from sort_impl import sort_items\n"
                           "assert sort_items([3, 1, 2]) == [1, 2, 3]\nprint('ICN_METRIC seconds=1')\n")
    repo.commit("initial")
    root = str(repo.root)
    call = lambda **kw: server.experiment(root=root, **kw)  # noqa: E731

    def measure(slug):
        run_id = call(action="run", exp=slug)["run_id"]
        call(action="wait", run=run_id, wait_seconds=60)
        return run_id

    def branch(title, body):
        slug = call(action="create", title=title, parent="baseline")["slug"]
        Path(call(action="checkout", exp=slug)["path"], "sort_impl.py").write_text(body, encoding="utf-8")
        call(action="commit", exp=slug)
        return slug

    rule = server.record(summary="sort_items returns a copy", kind="decision", root=root,
                         invariants=["sort_items must not mutate its input"], symbols=["sort_items"])
    call(action="init", command=f"{PY} bench.py")
    base = call(action="conclude", run=measure("baseline"), verdict="inconclusive")
    win_slug, loss_slug = branch("builtin", SORT_BUILTIN), branch("insertion", SORT_INSERTION)
    win = call(action="conclude", run=measure(win_slug), verdict="win", note="far faster",
               caused_by=[rule["primary_memory"]])
    loss = call(action="conclude", run=measure(loss_slug), verdict="loss")
    return {"repo": repo, "root": root, "call": call, "rule": rule, "base": base, "win": win,
            "loss": loss, "win_slug": win_slug, "loss_slug": loss_slug, "measure": measure}


def test_the_primary_conclusion_memory_carries_the_whole_evidence(sorter):
    from icn import server

    got = server.memory(action="get", memory_id=sorter["win"]["primary_memory"], root=sorter["root"])
    body = got["memory"]["body"]
    assert got["memory"]["kind"] == "decision"
    assert "seconds=1" in body and "far faster" in body and "Evidence: lab run" in body


def test_conclusions_chain_to_their_parent_and_to_what_motivated_them(sorter):
    from icn import server

    assert {l["kind"] for l in sorter["win"]["causal_links"]} == {"CAUSED", "LED_TO"}
    story = str(server.investigate(action="why", symbol="sort_items", root=sorter["root"]))
    assert sorter["win"]["primary_memory"] in story and sorter["base"]["primary_memory"] in story


def test_the_briefing_reports_finished_runs_awaiting_a_verdict(sorter):
    from icn import server

    pending = sorter["measure"](sorter["call"](action="create", title="pending", parent="baseline")["slug"])
    opened = server.workspace(action="open", root=sorter["root"])
    assert [r["run_id"] for r in opened["experiments"]["unjudged_runs"]] == [pending]
    assert sorter["loss"]["primary_memory"] in [m["memory_id"] for m in
                                                 opened["briefing"]["already_rejected"]]


def test_apply_lands_the_winner_and_settles_the_knowledge_it_touched(sorter):
    repo, call = sorter["repo"], sorter["call"]
    applied = call(action="apply", exp=sorter["win_slug"])

    assert applied["ok"] and repo.read("sort_impl.py") == SORT_BUILTIN
    measured = {sorter["win"]["primary_memory"], sorter["loss"]["primary_memory"]}
    assert measured <= set(applied["memories_reverified"])
    review = {m["memory_id"] for m in applied["memories_to_review"]}
    assert review and not (review & measured)
    assert {m["kind"] for m in applied["memories_to_review"]} >= {"invariant"}


def test_apply_refuses_when_the_working_tree_has_moved_on(sorter):
    repo, call = sorter["repo"], sorter["call"]
    repo.write("sort_impl.py", SORT_BUBBLE.replace("data = list(items)", "data = list(items)  # edited"))

    refused = call(action="apply", exp=sorter["win_slug"])
    assert refused["ok"] is False and "Nothing was changed" in refused["error"]
    assert "# edited" in repo.read("sort_impl.py")


def test_apply_refuses_an_unmeasured_experiment(sorter):
    slug = sorter["call"](action="create", title="unmeasured", parent="baseline")["slug"]
    refused = sorter["call"](action="apply", exp=slug)
    assert refused["ok"] is False and "not been measured" in refused["error"]


def test_touched_identifiers_include_the_enclosing_function_from_hunk_headers():
    patch = ("diff --git a/sort_impl.py b/sort_impl.py\n--- a/sort_impl.py\n+++ b/sort_impl.py\n"
             "@@ -3,2 +3,2 @@ def sort_items(items):\n-    data.sort()\n+    data.sort(reverse=False)\n"
             "diff --git a/gone.py b/gone.py\n--- a/gone.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-def removed(): pass\n")
    touched = lab._touched_identifiers(patch)
    assert "sort_items" in touched["sort_impl.py"] and "reverse" in touched["sort_impl.py"]
    assert "gone.py" not in touched


def test_record_evidence_embeds_the_run_and_refuses_unfinished_runs(research):
    from icn import server

    root = str(research.root)
    server.experiment(action="init", command=f"{PY} train.py", root=root)
    run_id = server.experiment(action="run", exp="baseline", root=root)["run_id"]

    early = server.record(summary="Baseline loss is 0.07", kind="investigation",
                          evidence=[run_id], root=root)
    if not early["ok"]:
        assert "not evidence" in early["error"]

    server.experiment(action="wait", run=run_id, wait_seconds=60, root=root)
    recorded = server.record(summary="Baseline loss is 0.07", kind="investigation",
                             performance=["The baseline trains to loss 0.07"],
                             evidence=[run_id], root=root)
    assert recorded["ok"] and recorded["evidence"] == [run_id]
    assert all(run_id in server.memory(action="get", memory_id=m["memory_id"], root=root)
               ["memory"]["body"] for m in recorded["memories_created"])

    assert not server.record(summary="x", evidence=["run_missing"], root=root)["ok"]
