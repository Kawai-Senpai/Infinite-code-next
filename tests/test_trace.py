"""`icn trace` / graph(action='run'): recording what a command actually did.

These run real interpreters, because the recorder lives inside the traced
process: a unit test of the renderer alone would not catch the event-order
bugs found live (a closed generator popping the wrong frame).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from icn import flowrun, hooks, server

PRICING = '''\
class Order:
    def __init__(self, oid, price, qty):
        self.oid, self.price, self.qty = oid, price, qty


def get_discount(order):
    return 20.0 if order.price * order.qty > 100 else 0.0


def calculate_price(order):
    if order.qty <= 0:
        raise ValueError("quantity must be positive")
    subtotal = order.price * order.qty
    discount = get_discount(order)
    subtotal -= discount
    total = subtotal * 1.08
    return total


def numbers():
    try:
        for i in range(100):
            yield i
    finally:
        pass


def first_three():
    for n in numbers():
        if n == 3:
            break
    return n


def after_close():
    import gc
    gc.collect()
    return sum(double(i) for i in range(3))


def double(i):
    return i * 2


def main():
    prices = []
    for order in (Order(1, 60, 2), Order(2, 10, 0)):
        try:
            prices.append(calculate_price(order))
        except ValueError:
            prices.append(None)
    first_three()
    after_close()
    return prices


if __name__ == "__main__":
    print(main())
'''


def trace(tmp_path: Path, name: str, source: str, *args: str, **kwargs):
    (tmp_path / name).write_text(source, encoding="utf-8")
    result = flowrun.run([sys.executable, name, *args], tmp_path, **kwargs)
    assert result["ok"], result
    return result, Path(result["report"]).read_text(encoding="utf-8")


def test_a_run_explains_its_flow_values_and_errors(tmp_path):
    result, report = trace(tmp_path, "app.py", PRICING)
    assert result["exit_code"] == 0 and result["python_processes"] == 1
    assert "calculate_price(order=Order(oid=1, price=60, qty=2))" in report
    # The values a person would otherwise set watchpoints for.
    assert "`subtotal`: 120 (line 13) → 100.0 (line 15)" in report
    assert "`discount`: 20.0 (line 14)" in report
    # One exception is one story, caught where it was actually caught.
    assert "raised in `calculate_price`" in report and "caught in `main`" in report
    assert (Path(result["trace_dir"]) / "report-full.md").exists()
    assert (tmp_path / ".icn-trace" / ".gitignore").read_text() == "*\n"


def test_a_generator_closed_later_does_not_corrupt_the_call_tree(tmp_path):
    """Measured: gc closing a generator fired PY_THROW + PY_UNWIND while other
    code ran, and the unwind popped that code's frame, so its callees were
    credited to its caller."""
    result, _ = trace(tmp_path, "app.py", PRICING)
    raw = json.loads(next((Path(result["trace_dir"]) / "raw").glob("py-*.json")).read_text())

    def find(node, name):
        if node.get("function") == name:
            return node
        for child in node.get("children", []):
            hit = find(child, name)
            if hit:
                return hit
        return None

    after = find(raw["tree"], "after_close")
    assert after is not None
    assert find(after, "double") is not None, "double() must be under after_close, not beside it"


def test_a_crash_is_reported_with_its_exit_code_and_path(tmp_path):
    result, report = trace(tmp_path, "crash.py",
                           "def parse(x):\n    return int(x)\n\n"
                           "def main():\n    return [parse(v) for v in ['1', 'two']]\n\nmain()\n")
    assert result["exit_code"] != 0
    assert "never caught: the program stopped" in report
    # An inlined comprehension's cleanup handler is not a catch.
    assert "caught in `main`" not in report


def test_a_command_that_outlives_its_timeout_is_stopped(tmp_path):
    result, report = trace(tmp_path, "sleeper.py",
                           "import time\n\ndef wait():\n    while True:\n        time.sleep(0.1)\n\nwait()\n",
                           timeout=2)
    assert result["timed_out"] is True
    assert "timed out" in report


def test_child_python_processes_are_recorded_too(tmp_path):
    source = ("import subprocess, sys\n\n"
              "def work(n):\n    return n * 2\n\n"
              "if len(sys.argv) > 1:\n    print(work(int(sys.argv[1])))\n"
              "else:\n    subprocess.run([sys.executable, __file__, '21'], check=True)\n")
    result, report = trace(tmp_path, "parent.py", source)
    assert result["python_processes"] == 2
    assert "work(n='21')" not in report and "work(n=21)" in report


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_node_processes_get_a_cpu_profile(tmp_path):
    (tmp_path / "work.js").write_text(
        "function fib(n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); }\n"
        "let t = 0; for (let i = 0; i < 30; i++) t += fib(22); console.log(t);\n", encoding="utf-8")
    result = flowrun.run("node work.js", tmp_path)
    assert result["node_profiles"] == 1, Path(result["report"]).read_text(encoding="utf-8")
    assert "## Node.js process" in Path(result["report"]).read_text(encoding="utf-8")


def test_the_mcp_action_returns_a_report_an_agent_can_read(tmp_path):
    (tmp_path / "hello.py").write_text("def greet(name):\n    return 'hi ' + name\n\ngreet('x')\n",
                                       encoding="utf-8")
    result = server.graph(action="run", target=f'"{sys.executable}" hello.py', root=str(tmp_path),
                          timeout_seconds=60)
    assert result["ok"] and result["action"] == "run"
    assert "greet(name='x')" in Path(result["report"]).read_text(encoding="utf-8")
    assert server.graph(action="run", target="", root=str(tmp_path))["ok"] is False
    assert server.graph(action="run", target="x", run_options=["fast"], root=str(tmp_path))["ok"] is False


def test_runtime_calls_are_checked_against_the_static_graph(repo, monkeypatch):
    """Constructors and context managers are implicit, not graph gaps; a call
    through an import alias must be in the graph."""
    from icn import workspace as ws_mod
    monkeypatch.setattr(ws_mod, "_background_finish", lambda *a, **k: None)
    repo.write("pkg/__init__.py", "")
    repo.write("pkg/store.py",
               "class Tx:\n    def __init__(self, name):\n        self.name = name\n"
               "    def __enter__(self):\n        return self\n"
               "    def __exit__(self, *exc):\n        return False\n\n\n"
               "def open_store(name):\n    with Tx(name) as tx:\n        return tx.name\n")
    repo.write("pkg/app.py", "from . import store as store_mod\n\n\n"
                             "def open_store(name='db'):\n    return store_mod.open_store(name)\n")
    repo.write("run.py", "from pkg.app import open_store\n\nopen_store()\n")
    repo.commit("store")
    server.workspace(action="open", root=str(repo.root))
    result = server.graph(action="run", target=f'"{sys.executable}" run.py', root=str(repo.root),
                          timeout_seconds=60)
    static = json.loads((Path(result["trace_dir"]) / "summary.json").read_text())["static_graph"]
    assert static["not_in_static_graph"] == 0, static["missing"]
    assert static["in_static_graph"] >= 2 and static["implicit"] >= 2


def test_cli_passes_the_traced_exit_code_through(tmp_path, capsys):
    from icn import trace_cli
    (tmp_path / "fail.py").write_text("import sys\nsys.exit(3)\n", encoding="utf-8")
    code = trace_cli.main(["--quiet", "--cwd", str(tmp_path), sys.executable, "fail.py"])
    assert code == 3
    assert "Report:" in capsys.readouterr().err


def test_shell_segments_respect_quotes():
    assert hooks._segments('grep -n "a\\|b" x.py | head') == ['grep -n "a\\|b" x.py ', ' head']
    assert hooks._shell_files('cd . && grep -n "def \\|import" src/a.py; grep x src/b.py') == []
    assert hooks._shell_files("sed -n 1,5p src/a.py | grep x") == ["src/a.py"]


PERMISSIONS = '''\
def fetch_user(user_id):
    return {"id": user_id, "role": "admin"}


def load_user(user_id):
    response = fetch_user(user_id)
    role = response.get("role")
    return {"id": response["id"], "role": role}


def validate(user):
    if user["role"] != "admin":
        raise PermissionError("no")
    return True


def refresh(user):
    return load_user(user["id"])


def login(user_id):
    user = load_user(user_id)
    if user["role"] is None:
        user = refresh(user)
    validate(user)
    return "ok"


print(login(7))
'''


def test_a_diff_names_the_first_divergence_and_the_value_that_caused_it(tmp_path):
    (tmp_path / "perms.py").write_text(PERMISSIONS, encoding="utf-8")
    flowrun.run([sys.executable, "perms.py"], tmp_path)
    (tmp_path / "perms.py").write_text(PERMISSIONS.replace('get("role")', 'get("user_role")'),
                                       encoding="utf-8")
    flowrun.run([sys.executable, "perms.py"], tmp_path)

    result = server.graph(action="run_diff", root=str(tmp_path))
    assert result["ok"] and result["diverges"]
    report = Path(result["report"]).read_text(encoding="utf-8")
    assert "- A next called `validate`" in report and "- B next called `refresh`" in report
    values = report.split("## Values that differ", 1)[1]
    # The earliest difference is the cause, and it is listed first.
    first_row = next(line for line in values.splitlines() if line.startswith("- "))
    assert "`role`" in first_row and "`'admin'` (line 7) in A, `None` (line 7) in B" in first_row
    errors = report.split("## Exceptions only in B", 1)[1].split("##", 1)[0]
    assert errors.count("PermissionError") == 1
    assert server.graph(action="run_diff", target="only-one", root=str(tmp_path))["ok"] is False


SHOP = ("def discount(order):\n    return 5 if order > 100 else 0\n\n\n"
        "def total(order):\n    return order - discount(order)\n\n\n"
        "def never_run(order):\n    return order * 2\n\n\nprint(total(120))\n")


def test_a_recording_is_evidence_and_is_wired_into_the_rest_of_icn(repo, monkeypatch):
    """A recording must be usable as proof, visible on open(), and answer the
    question the static graph answers only as a lower bound."""
    # open() can finish indexing on a background thread that outlives this
    # test; left running, it reached a later test's monkeypatched
    # Indexer.resolve_calls and failed that test instead of this one.
    from icn import workspace as ws_mod
    monkeypatch.setattr(ws_mod, "_background_finish", lambda *a, **k: None)
    repo.write("shop.py", SHOP)
    repo.commit("shop")
    root = str(repo.root)
    server.workspace(action="open", root=root)
    traced = server.graph(action="run", target=f'"{sys.executable}" shop.py', root=root,
                          timeout_seconds=60)
    run_name = Path(traced["trace_dir"]).name

    # 1. cited as evidence, the way a lab run is
    recorded = server.record(kind="note", summary="discount threshold",
                             invariants=["discount() returns 5 only above 100"],
                             symbols="discount", evidence=run_name, root=root)
    body = server.memory(action="get", memory_id=recorded["memories_created"][0]["memory_id"],
                         root=root)["memory"]["body"]
    assert f"Evidence: recorded run {run_name}" in body and "exit 0" in body
    assert server.record(kind="note", summary="x", evidence="no-such-run", root=root)["ok"] is False

    # 2. named in the workspace briefing
    latest = server.workspace(action="open", root=root)["recorded_runs"]["latest"][0]
    assert latest["run"] == run_name and latest["exit_code"] == 0

    # 3. why() reports who actually called it, and does not call unrun code dead
    why = server.investigate(action="why", symbol="discount", root=root)["observed_at_runtime"]
    assert why["calls"] == 1 and any(c.startswith("total ") for c in why["called_by"])
    quiet = server.investigate(action="why", symbol="never_run", root=root)["observed_at_runtime"]
    assert quiet["calls"] == 0 and "not proof" in quiet["note"]


def test_a_run_is_found_by_its_exact_name_even_beside_a_retry(tmp_path):
    """Substring matching alone made a run's own name ambiguous against its
    retry suffix: `...-a` matched both `...-a` and `...-a~2`."""
    from icn import flowdiff
    (tmp_path / "a.py").write_text("def f():\n    return 1\n\nf()\n", encoding="utf-8")
    # Both runs must land in the same second to get the `~2` suffix, which is
    # the case under test, so the second is built rather than raced for.
    first = flowrun.run([sys.executable, "a.py"], tmp_path)
    stamp = Path(first["trace_dir"]).name
    shutil.copytree(first["trace_dir"], Path(first["trace_dir"]).with_name(f"{stamp}~2"))
    names = [r.name for r in flowdiff.runs_in(tmp_path)]
    assert names == [stamp, f"{stamp}~2"]
    for name in names:
        assert flowdiff.find_run(tmp_path, name).name == name
        assert f"Evidence: recorded run {name}" in flowrun.evidence_for(tmp_path, [name])[0]
    with pytest.raises(ValueError, match="matches 2 runs"):
        flowrun.evidence_for(tmp_path, ["-a"])


def test_old_recordings_are_pruned_so_the_folder_stays_bounded(tmp_path, monkeypatch):
    """A traced pytest run is about 1.3 MB and nothing else deletes one."""
    from icn import flowdiff
    monkeypatch.setenv("ICN_TRACE_KEEP", "3")
    (tmp_path / "a.py").write_text("def f():\n    return 1\n\nf()\n", encoding="utf-8")
    for _ in range(5):
        flowrun.run([sys.executable, "a.py"], tmp_path)
    kept = flowdiff.runs_in(tmp_path)
    assert len(kept) == 3, [r.name for r in kept]
    # The newest survive, and each is still a complete, readable recording.
    assert all((r / "report.md").exists() and (r / "summary.json").exists() for r in kept)


CALC = '''\
def rate(kind):
    return {"standard": 0.08}.get(kind, 0.0)


def tax(amount, kind="standard"):
    return round(amount * rate(kind), 2)


def checkout(amount):
    t = tax(amount)
    return round(amount + t, 2)


print(checkout(200))
'''


def test_a_diff_puts_the_deepest_cause_above_its_consequences(tmp_path):
    """Two runs that both succeed with identical control flow: the only
    evidence is values, and a differing return must sort before the caller's
    variable it then changes."""
    (tmp_path / "calc.py").write_text(CALC, encoding="utf-8")
    flowrun.run([sys.executable, "calc.py"], tmp_path)
    (tmp_path / "calc.py").write_text(CALC.replace("0.08", "0.10"), encoding="utf-8")
    flowrun.run([sys.executable, "calc.py"], tmp_path)

    result = server.graph(action="run_diff", root=str(tmp_path))
    assert result["ok"] and not result["diverges"], "control flow is identical here"
    section = Path(result["report"]).read_text(encoding="utf-8") \
        .split("## Values that differ", 1)[1].split("\n## ", 1)[0]
    rows = [l for l in section.splitlines() if l.startswith("- ")]
    assert "`rate`" in rows[0] and "0.08" in rows[0], rows
    assert "`t`" in rows[-1], rows
