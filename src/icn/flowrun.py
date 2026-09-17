"""Run any command and explain what it did: `icn trace -- <command>`.

The recorder is injected, not wrapped around one known runtime:

    Python   tracehook/ is put first on PYTHONPATH, so every Python interpreter
             the command starts (python app.py, pytest, a worker process)
             records itself with flowtrace.py.
    Node     --cpu-prof is added to NODE_OPTIONS, so every Node process
             (node, npm test, npx jest) writes a sampled CPU profile.
    Other    the process itself: exit code, wall time, CPU time, peak memory,
             and the tail of its output.

Everything lands in <cwd>/.icn-trace/<run>/, which ignores itself for git.
report.md is the product: written for a person or an agent to read top to
bottom, with the flow first, then where the time went, how values changed,
what failed, and what the recorder could not see. The raw recordings sit
beside it in raw/ for anything the report summarises away.

This module imports only the standard library. The optional check against
ICN's static call graph reads the repository store with sqlite3 directly.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

TRACE_DIR = ".icn-trace"
HOOK_DIR = Path(__file__).resolve().parent / "tracehook"
TAIL_LINES = 60


class Limits:
    """How much of a recording one report spells out.

    report.md is read by agents, so it has a budget: a pytest run on this
    repository rendered to 216 KB, which costs more to read than the question
    was worth. report-full.md keeps everything for when the summary is not
    enough.
    """

    def __init__(self, full: bool) -> None:
        big = 10**6
        self.full = full
        self.narrative_top = big if full else 15
        self.narrative_children = big if full else 8
        self.tree_depth = 40 if full else 10
        self.tree_children = big if full else 12
        self.tree_lines = big if full else 120
        self.changes_per_call = big if full else 10
        self.value_functions = big if full else 25
        self.errors = big if full else 12
        self.static = big if full else 15
        self.output_lines = TAIL_LINES if full else 25
        self.value_chars = 200 if full else 50
        self.used = 0


_LIMITS: Limits = Limits(full=False)
CPU_MIN_MS = 20


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, default)))
    except ValueError:
        return default


# ------------------------------------------------------------------ running

def _slug(command: list[str] | str) -> str:
    words = command.split() if isinstance(command, str) else list(command)
    # Name the run after what it runs, not the interpreter: a full python.exe
    # path made every run folder "Users-user".
    stems = [Path(w.strip("'\"")).stem for w in words if w and not w.startswith("-")]
    stems = [s for s in stems if s.lower() not in ("python", "python3", "pythonw", "py", "node")]
    name = "-".join(stems[:3]) or "run"
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name)[:40].strip("-") or "run"


def _prepare_dir(cwd: Path, command: list[str] | str) -> Path:
    base = cwd / TRACE_DIR
    base.mkdir(parents=True, exist_ok=True)
    ignore = base / ".gitignore"
    if not ignore.exists():
        ignore.write_text("*\n", encoding="utf-8")
    _prune(base)
    run_dir = base / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{_slug(command)}"
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = run_dir.with_name(f"{run_dir.name.split('~')[0]}~{suffix}")
    (run_dir / "raw").mkdir(parents=True)
    return run_dir


def _prune(base: Path) -> None:
    """Keep the most recent runs, delete the rest.

    A traced pytest run is about 1.3 MB, and nothing else ever deletes one, so
    an unbounded .icn-trace/ would grow with every run. Names begin with a
    timestamp, so name order is age order.
    """
    keep = _env_int("ICN_TRACE_KEEP", 20)
    try:
        runs = sorted(p for p in base.iterdir() if p.is_dir())
    except OSError:
        return
    for old in runs[:max(0, len(runs) - keep + 1)]:
        try:
            shutil.rmtree(old)
        except OSError:
            pass            # in use, or gone already: not worth failing a run over


def _script_dirs(command: list[str] | str, cwd: Path) -> list[str]:
    """Directories of scripts named on the command line, so a script outside
    the working directory still counts as project code."""
    words = command if isinstance(command, list) else shlex.split(command, posix=os.name != "nt")
    out = []
    for word in words:
        if word.endswith((".py", ".pyw")):
            path = (cwd / word).resolve() if not Path(word).is_absolute() else Path(word)
            if path.exists() and not str(path).startswith(str(cwd.resolve())):
                out.append(str(path.parent))
    return out


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                           capture_output=True, timeout=15)
        else:
            os.killpg(os.getpgid(proc.pid), 9)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


def _windows_process_stats(proc: subprocess.Popen) -> dict[str, Any]:
    try:
        import ctypes
        from ctypes import wintypes

        handle = getattr(proc, "_handle", None)
        if handle is None:
            return {}

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        stats: dict[str, Any] = {}
        psapi, kernel32 = ctypes.WinDLL("psapi"), ctypes.WinDLL("kernel32")
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.c_void_p] * 4
        if psapi.GetProcessMemoryInfo(int(handle), ctypes.byref(counters), counters.cb):
            stats["peak_rss_bytes"] = int(counters.PeakWorkingSetSize)
        creation, exit_, kernel, user = (wintypes.FILETIME() for _ in range(4))
        if kernel32.GetProcessTimes(int(handle), ctypes.byref(creation), ctypes.byref(exit_),
                                    ctypes.byref(kernel), ctypes.byref(user)):
            def ms(ft: Any) -> float:
                return ((ft.dwHighDateTime << 32) + ft.dwLowDateTime) / 10_000
            stats["cpu_ms"] = round(ms(kernel) + ms(user), 1)
        return stats
    except Exception:  # noqa: BLE001
        return {}


def run(command: list[str] | str, cwd: str | Path | None = None, *, values: bool = True,
        libraries: bool = False, memory: bool = False, timeout: float | None = None,
        max_events: int = 50_000, echo: bool = False,
        static_graph: Callable[[list[dict[str, Any]]], dict[str, Any] | None] | None = None
        ) -> dict[str, Any]:
    """Run `command` under the recorders and write report.md. Never raises for
    a failing command: a crash is exactly what a report is for."""
    cwd = Path(cwd or os.getcwd()).resolve()
    run_dir = _prepare_dir(cwd, command)
    raw = run_dir / "raw"

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(HOOK_DIR), env.get("PYTHONPATH", "")) if p)
    env["ICN_TRACE_DIR"] = str(raw)
    env["ICN_TRACE_ROOTS"] = os.pathsep.join([str(cwd), *_script_dirs(command, cwd)])
    env["ICN_TRACE_VALUES"] = "1" if values else "0"
    env["ICN_TRACE_LIBS"] = "1" if libraries else "0"
    env["ICN_TRACE_MEMORY"] = "1" if memory else "0"
    env["ICN_TRACE_MAX_EVENTS"] = str(max_events)
    # Forward slashes: NODE_OPTIONS treats a backslash inside quotes as an
    # escape, which turned C:\Users\... into C:Users... and no profile at all.
    env["NODE_OPTIONS"] = " ".join(p for p in (env.get("NODE_OPTIONS", ""),
                                                "--cpu-prof", f'--cpu-prof-dir="{raw.as_posix()}"') if p)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    # The recorder runs in the traced process; it must not trace this one.
    env.pop("ICN_TRACE_PARENT", None)

    started_at = datetime.now().isoformat(timespec="seconds")
    started = time.perf_counter()
    stdout_tail: deque[str] = deque(maxlen=TAIL_LINES)
    stderr_tail: deque[str] = deque(maxlen=TAIL_LINES)
    popen_kwargs: dict[str, Any] = {}
    if os.name != "nt":
        popen_kwargs["start_new_session"] = True
    shown_command = command if isinstance(command, str) else " ".join(
        f'"{c}"' if " " in c else c for c in command)
    shell = False
    if isinstance(command, str):
        # A shell only when the command needs one. Through cmd.exe the process
        # stats measured the shell (0 ms CPU, 8 MB) instead of the program.
        if re.search(r"[|&;<>()$`%!^*?]", command):
            shell = True
        else:
            command = shlex.split(command, posix=os.name != "nt")
            command = [w[1:-1] if len(w) > 1 and w[0] == w[-1] and w[0] in "'\"" else w for w in command]
    try:
        proc = subprocess.Popen(
            command, cwd=str(cwd), env=env, shell=shell,
            stdin=None if echo else subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", bufsize=1,
            **popen_kwargs)
    except OSError as err:
        return {"ok": False, "error": f"could not start {command!r}: {err}", "trace_dir": str(run_dir)}

    def pump(stream: Any, tail: deque[str], sink: Any) -> None:
        for line in stream:
            tail.append(line.rstrip("\n"))
            if sink is not None:
                sink.write(line)
                sink.flush()

    pumps = [threading.Thread(target=pump, args=(proc.stdout, stdout_tail, sys.stdout if echo else None),
                              daemon=True),
             threading.Thread(target=pump, args=(proc.stderr, stderr_tail, sys.stderr if echo else None),
                              daemon=True)]
    for thread in pumps:
        thread.start()

    live_stop = threading.Event()
    if echo:
        threading.Thread(target=_live_status, args=(raw, started, live_stop), daemon=True).start()

    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_tree(proc)
        proc.wait()
    except KeyboardInterrupt:
        _kill_tree(proc)
        proc.wait()
    live_stop.set()
    for thread in pumps:
        thread.join(timeout=5)
    wall_ms = round((time.perf_counter() - started) * 1000, 1)
    process = {"exit_code": proc.returncode, "wall_ms": wall_ms, "timed_out": timed_out}
    if os.name == "nt":
        process.update(_windows_process_stats(proc))
    else:
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_CHILDREN)
            process["cpu_ms"] = round((usage.ru_utime + usage.ru_stime) * 1000, 1)
            process["peak_rss_bytes"] = usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        except Exception:  # noqa: BLE001
            pass

    pythons = _load_python(raw)
    nodes = _load_node(raw)
    edges = runtime_edges(pythons, cwd)
    static = None
    if static_graph is not None and edges:
        try:
            static = static_graph(edges)
        except Exception as err:  # noqa: BLE001 - the report must still be written
            static = {"error": f"{type(err).__name__}: {err}"}

    summary = {
        "command": shown_command,
        "cwd": str(cwd), "started_at": started_at, "process": process,
        "python_processes": len(pythons), "node_profiles": len(nodes),
        "stdout_tail": list(stdout_tail), "stderr_tail": list(stderr_tail),
        "options": {"values": values, "libraries": libraries, "memory": memory,
                    "timeout": timeout, "max_events": max_events},
    }
    report_path = run_dir / "report.md"
    report_path.write_text(render(summary, pythons, nodes, edges, static, full=False), encoding="utf-8")
    (run_dir / "report-full.md").write_text(render(summary, pythons, nodes, edges, static, full=True),
                                            encoding="utf-8")
    hot = [f for f in hot_functions(pythons, cwd) if f["kind"] not in ("module", "class")][:5]
    (run_dir / "summary.json").write_text(json.dumps(
        {**summary, "hot_functions": hot, "static_graph": static}, indent=2), encoding="utf-8")
    (cwd / TRACE_DIR / "latest.txt").write_text(str(report_path) + "\n", encoding="utf-8")
    return {
        "ok": True, "report": str(report_path), "trace_dir": str(run_dir),
        "exit_code": proc.returncode, "timed_out": timed_out, "wall_ms": wall_ms,
        "python_processes": len(pythons), "node_profiles": len(nodes),
        "exceptions": sum(len({e.get("exception") or id(e) for e in p.get("errors", [])}) for p in pythons),
        "hot_functions": hot,
        "next": "read the report file; raw/ holds the full recordings",
    }


def evidence_for(cwd: Path, names: list[str]) -> list[str]:
    """Evidence sentences for record(evidence=[...]), from recorded runs.

    A recording proves what happened as a lab run proves a measurement, so it
    travels the same path: the sentence is written into every memory of the
    event, and the claim carries its proof rather than asking to be believed.
    Raises ValueError for a name that matches no run or several.
    """
    from . import flowdiff

    lines = []
    for name in names:
        run = flowdiff.find_run(cwd, name)
        summary = flowdiff.summary_of(run)
        process = summary["process"]
        hot = ", ".join(f"{f['function']} {f['self_ms']:g}ms x{f['calls']}"
                        for f in (summary.get("hot_functions") or [])[:3]) or "none recorded"
        static = summary.get("static_graph") or {}
        unknown = (f", {static['not_in_static_graph']} observed call(s) absent from the static graph"
                   if static.get("not_in_static_graph") else "")
        lines.append(
            f"Evidence: recorded run {run.name} of `{summary['command']}` in {summary['cwd']}, "
            f"exit {process.get('exit_code')}{' (timed out)' if process.get('timed_out') else ''}, "
            f"{process.get('wall_ms', 0) / 1000:g}s, "
            f"{summary.get('python_processes', 0)} Python process(es), hottest: {hot}{unknown}. "
            f"Full report: {run / 'report.md'}.")
    return lines


def _live_status(raw: Path, started: float, stop: threading.Event) -> None:
    """One status line on stderr every two seconds while the program runs."""
    last = ""
    while not stop.wait(2.0):
        where = ""
        for live in raw.glob("py-*.live.json"):
            try:
                data = json.loads(live.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            stacks = [s for s in (data.get("now") or {}).values() if s]
            if stacks:
                where = " > ".join(stacks[0][-3:])
                break
        line = f"[icn trace {time.perf_counter() - started:6.1f}s] {where}"[:160]
        if line != last:
            sys.stderr.write(line + "\n")
            sys.stderr.flush()
            last = line


# ------------------------------------------------------------------ loading

def _load_python(raw: Path) -> list[dict[str, Any]]:
    out = []
    for path in sorted(raw.glob("py-*.json")):
        if path.name.endswith(".live.json"):
            continue
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    # Processes that died without running atexit leave only a live snapshot.
    finished = {p.get("pid") for p in out}
    for path in sorted(raw.glob("py-*.live.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if data.get("pid") not in finished:
            data["killed"] = True
            out.append(data)
    # The busiest process first; wrapper processes (a launcher that only
    # spawns the real program) record almost nothing.
    out.sort(key=lambda p: -_tree_calls(p.get("tree") or {}))
    return out


def _tree_calls(node: dict[str, Any]) -> int:
    return node.get("count", 0) + sum(_tree_calls(c) for c in node.get("children", []))


def _load_node(raw: Path) -> list[dict[str, Any]]:
    profiles = []
    for path in sorted(raw.glob("*.cpuprofile")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        profiles.append(_node_profile(data, path.name))
    return profiles


def _node_profile(data: dict[str, Any], name: str) -> dict[str, Any]:
    """Self and total time per function from a V8 sampled CPU profile."""
    nodes = {n["id"]: n for n in data.get("nodes", [])}
    self_us: dict[int, float] = {}
    samples, deltas = data.get("samples", []), data.get("timeDeltas", [])
    for i, node_id in enumerate(samples):
        # A delta is the time since the previous sample, so it belongs to the
        # sample before it, which is what was running during that interval.
        duration = deltas[i + 1] if i + 1 < len(deltas) else 0
        self_us[node_id] = self_us.get(node_id, 0.0) + max(0, duration)
    total_us: dict[int, float] = {}

    def total(node_id: int) -> float:
        if node_id in total_us:
            return total_us[node_id]
        node = nodes.get(node_id, {})
        value = self_us.get(node_id, 0.0) + sum(total(c) for c in node.get("children", []))
        total_us[node_id] = value
        return value

    for node_id in nodes:
        total(node_id)

    def key_of(node: dict[str, Any]) -> tuple[str, str, int]:
        frame = node.get("callFrame", {})
        return (frame.get("functionName") or "(anonymous)", frame.get("url", ""),
                frame.get("lineNumber", -1) + 1)

    parents = {c: n for n, node in nodes.items() for c in node.get("children", [])}
    functions: dict[tuple[str, str, int], dict[str, float]] = {}
    for node_id, node in nodes.items():
        key = key_of(node)
        entry = functions.setdefault(key, {"self_ms": 0.0, "total_ms": 0.0})
        entry["self_ms"] += self_us.get(node_id, 0.0) / 1000
        # A recursive call's time is already inside its outermost ancestor's.
        ancestor, recursive = parents.get(node_id), False
        while ancestor is not None:
            if key_of(nodes[ancestor]) == key:
                recursive = True
                break
            ancestor = parents.get(ancestor)
        if not recursive:
            entry["total_ms"] += total_us.get(node_id, 0.0) / 1000
    duration_ms = (data.get("endTime", 0) - data.get("startTime", 0)) / 1000
    ranked = sorted(({"function": k[0], "url": k[1], "line": k[2], **{m: round(v, 2) for m, v in e.items()}}
                     for k, e in functions.items()), key=lambda f: -f["self_ms"])
    user = [f for f in ranked if f["url"] and not f["url"].startswith("node:")
            and "node_modules" not in f["url"] and f["function"] not in ("(program)", "(idle)",
                                                                          "(garbage collector)", "(root)")]
    root_id = next(iter(nodes), None)
    return {"file": name, "duration_ms": round(duration_ms, 1), "hot": ranked[:15], "user_hot": user[:15],
            "tree": _node_tree(nodes, root_id, total_us, 0) if root_id is not None else None}


def _node_tree(nodes: dict[int, Any], node_id: int, total_us: dict[int, float], depth: int) -> dict[str, Any]:
    node = nodes[node_id]
    frame = node.get("callFrame", {})
    children = sorted(node.get("children", []), key=lambda c: -total_us.get(c, 0.0))
    return {"function": frame.get("functionName") or "(anonymous)", "url": frame.get("url", ""),
            "line": frame.get("lineNumber", -1) + 1, "total_ms": round(total_us.get(node_id, 0.0) / 1000, 2),
            "children": [_node_tree(nodes, c, total_us, depth + 1) for c in children
                         if total_us.get(c, 0.0) >= 1000] if depth < 30 else []}


# ------------------------------------------------------------------ analysis

def _rel(path: str, cwd: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(cwd)).replace("\\", "/")
    except (ValueError, OSError):
        return path.replace("\\", "/")


def hot_functions(pythons: list[dict[str, Any]], cwd: Path) -> list[dict[str, Any]]:
    flat: dict[tuple[str, str, int], dict[str, float]] = {}

    def walk(node: dict[str, Any], on_stack: set[tuple[str, str, int]]) -> None:
        for child in node.get("children", []):
            key = (child["file"], child["function"], child["line"])
            entry = flat.setdefault(key, {"calls": 0, "total_ms": 0.0, "self_ms": 0.0, "self_cpu_ms": 0.0,
                                          "kind": child.get("kind", "function")})
            entry["calls"] += child["count"]
            entry["self_ms"] += child["self_ms"]
            entry["self_cpu_ms"] += child.get("self_cpu_ms", 0.0)
            if key not in on_stack:             # recursion must not double-count
                entry["total_ms"] += child["total_ms"]
            walk(child, on_stack | {key})

    for process in pythons:
        walk(process.get("tree") or {}, set())
    ranked = [{"function": k[1], "where": f"{_rel(k[0], cwd)}:{k[2]}", "calls": int(v["calls"]),
               "total_ms": round(v["total_ms"], 2), "self_ms": round(v["self_ms"], 2),
               "self_cpu_ms": round(v["self_cpu_ms"], 2), "kind": v["kind"]}
              for k, v in flat.items()]
    ranked.sort(key=lambda f: -f["self_ms"])
    return ranked


_EDGE_CACHE: dict[tuple[str, str], list[dict[str, Any]]] = {}


def edges_of_run(run_dir: Path, cwd: Path) -> list[dict[str, Any]]:
    """Caller -> callee pairs one recorded run observed, by run folder.

    Cached: a finished recording never changes, and parsing one 1 MB
    recording took 225 ms, which investigate(action='why') would otherwise
    pay on every call.
    """
    key = (str(run_dir), str(cwd))
    hit = _EDGE_CACHE.get(key)
    if hit is None:
        if len(_EDGE_CACHE) > 8:
            _EDGE_CACHE.clear()
        hit = _EDGE_CACHE[key] = runtime_edges(_load_python(run_dir / "raw"), cwd)
    return hit


def runtime_edges(pythons: list[dict[str, Any]], cwd: Path) -> list[dict[str, Any]]:
    """Caller -> callee pairs actually observed, for checking the static graph."""
    edges: dict[tuple, int] = {}

    def walk(node: dict[str, Any]) -> None:
        for child in node.get("children", []):
            if node.get("kind") in ("function", "generator", "coroutine") and node.get("file") \
                    and child.get("kind") in ("function", "generator", "coroutine"):
                key = (_rel(node["file"], cwd), node["function"], node["line"],
                       _rel(child["file"], cwd), child["function"], child["line"])
                edges[key] = edges.get(key, 0) + child["count"]
            walk(child)

    for process in pythons:
        walk(process.get("tree") or {})
    return [{"caller_file": k[0], "caller": k[1], "caller_line": k[2], "callee_file": k[3],
             "callee": k[4], "callee_line": k[5], "count": v} for k, v in edges.items()]


def check_static_graph(store_path: Path, edges: list[dict[str, Any]]) -> dict[str, Any]:
    """Which observed calls ICN's static call graph does and does not know.

    A call the static graph lacks is not an error in either: dynamic dispatch,
    callbacks, decorators and framework wiring are invisible to static
    resolution. It is exactly the evidence the graph's `lower-bound` answers
    say is missing.
    """
    conn = sqlite3.connect(f"file:{store_path.as_posix()}?mode=ro", uri=True, timeout=5)
    try:
        def symbol(path: str, qualname: str, line: int) -> str | None:
            if "<" in qualname:                 # <module>, <lambda>, <locals>
                return None
            row = conn.execute(
                "SELECT symbol_id FROM symbols WHERE last_known_path = ? AND status = 'ACTIVE'"
                " AND (symbol_path = ? OR (line_start BETWEEN ? AND ? AND name = ?))"
                " ORDER BY symbol_path = ? DESC LIMIT 1",
                (path, qualname, line - 3, line + 3, qualname.rsplit(".", 1)[-1], qualname)).fetchone()
            return row[0] if row else None

        def edge(a: str, b: str) -> bool:
            return conn.execute("SELECT 1 FROM code_edges WHERE from_id = ? AND to_id = ? AND kind = 'CALLS'"
                                " AND status = 'ACTIVE' LIMIT 1", (a, b)).fetchone() is not None

        known, missing, implicit, unresolved = 0, [], 0, 0
        for observed in edges:
            caller = symbol(observed["caller_file"], observed["caller"], observed["caller_line"])
            callee = symbol(observed["callee_file"], observed["callee"], observed["callee_line"])
            if caller is None or callee is None:
                unresolved += 1
                continue
            if edge(caller, callee):
                known += 1
                continue
            owner, _, method = observed["callee"].rpartition(".")
            if method == "__init__" and owner:
                # `Lease(...)` is recorded statically as a call to the class.
                cls = symbol(observed["callee_file"], owner, 0)
                if cls is not None and edge(caller, cls):
                    known += 1
                    continue
            decorators = (conn.execute("SELECT decorators FROM symbols WHERE symbol_id = ?",
                                       (callee,)).fetchone() or [""])[0] or ""
            if (method.startswith("__") and method.endswith("__")) or "property" in decorators                     or ".setter" in decorators:
                # `with x:`, `for a in x`, `x.attr` for a property: calls the
                # language makes implicitly and no static call graph models.
                implicit += 1
                continue
            missing.append(observed)
        missing.sort(key=lambda e: -e["count"])
        return {"observed": len(edges), "in_static_graph": known, "not_in_static_graph": len(missing),
                "implicit": implicit, "not_indexed": unresolved, "missing": missing[:40]}
    finally:
        conn.close()


# ------------------------------------------------------------------ rendering

def _ms(value: float) -> str:
    if value >= 1000:
        return f"{value / 1000:.2f} s"
    if value >= 10:
        return f"{value:.0f} ms"
    return f"{value:.2f} ms"


def _bytes(value: int | None) -> str:
    if not value:
        return "unknown"
    return f"{value / (1024 * 1024):.1f} MB"


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:max(1, limit - 3)] + "..."


def _args(call: dict[str, Any], limit: int = 110) -> str:
    text = ", ".join(f"{k}={v}" for k, v in (call.get("args") or {}).items()
                     if k not in ("self", "cls"))
    return _clip(text, min(limit, _LIMITS.value_chars * 2))


def _display_command(command: str) -> str:
    """The command as a person typed it: an interpreter by name, not full path."""
    words = command.split(" ", 1)
    head = words[0].strip("'\"")
    if os.sep in head or "/" in head:
        head = Path(head).name
        head = head[:-4] if head.lower().endswith(".exe") else head
    return head + (" " + words[1] if len(words) > 1 else "")


def _label(node: dict[str, Any], cwd: Path, with_args: bool = True, limit: int = 110) -> str:
    kind = node.get("kind", "function")
    if kind == "module":
        return f"import {_rel(node['file'], cwd)}" if node.get("_nested") else Path(node["file"]).name
    if kind == "class":
        return f"class {node['function']} (body)"
    first = (node.get("calls") or [{}])[0]
    prefix = {"generator": "generator ", "coroutine": "async "}.get(kind, "")
    return f"{prefix}{node['function']}({_args(first, limit) if with_args else ''})"


def render(summary: dict[str, Any], pythons: list[dict[str, Any]], nodes: list[dict[str, Any]],
           edges: list[dict[str, Any]], static: dict[str, Any] | None, full: bool = False) -> str:
    global _LIMITS
    _LIMITS = Limits(full)
    cwd = Path(summary["cwd"])
    proc = summary["process"]
    lines = [f"# Execution report: `{_display_command(summary['command'])}`", ""]
    status = ("timed out and was stopped" if proc.get("timed_out")
              else "succeeded" if proc.get("exit_code") == 0 else f"failed with exit code {proc.get('exit_code')}")
    lines += [f"- Ran in `{summary['cwd']}` at {summary['started_at']} and **{status}**.",
              f"- Wall time {_ms(proc['wall_ms'])}"
              + (f", CPU time {_ms(proc['cpu_ms'])}" if proc.get("cpu_ms") is not None else "")
              + f", peak memory {_bytes(proc.get('peak_rss_bytes'))} (top process).",
              f"- Recorded: {summary['python_processes']} Python process(es), "
              f"{summary['node_profiles']} Node.js profile(s).",
              "- Sections: what happened, call flow, where the time went, how values changed, errors, "
              "output, and what this report cannot show."
              + ("" if full else " This is the summary; `report-full.md` beside it has every call "
                 "path, value and error, and `raw/` the recordings.")]
    if not pythons and not nodes:
        lines.append("- No Python or Node.js code ran under the recorder, so this report covers the "
                     "process as a whole: time, memory, exit status and output.")
    lines.append("")

    for index, process in enumerate(pythons):
        lines += _render_python(process, cwd, index, len(pythons))
    for profile in nodes:
        lines += _render_node(profile, cwd)
    if static:
        lines += _render_static(static)
    lines += _render_output(summary)
    lines += _render_limits(summary, pythons)
    return "\n".join(lines).rstrip() + "\n"


def _mark_nested_modules(node: dict[str, Any], depth: int = 0) -> None:
    for child in node.get("children", []):
        if child.get("kind") == "module" and depth > 0:
            child["_nested"] = True
        _mark_nested_modules(child, depth + 1)


def _render_python(process: dict[str, Any], cwd: Path, index: int, count: int) -> list[str]:
    tree = process.get("tree") or {}
    _mark_nested_modules(tree)
    title = f"Python process {process.get('pid')}" + (f" ({index + 1} of {count})" if count > 1 else "")
    argv = " ".join(process.get("argv") or [])
    lines = [f"## {title}", "",
             f"`{argv}` on Python {process.get('python')} ({process.get('engine')}); "
             f"wall {_ms(process.get('wall_ms', 0))}, CPU {_ms(process.get('cpu_ms', 0))}, "
             f"peak memory {_bytes(process.get('peak_rss_bytes'))}"
             + (". **Killed before it could finish; this is its last snapshot.**" if process.get("killed") else "."),
             ""]
    if not tree.get("children"):
        lines += ["No project code ran in this process (only the interpreter, the standard library "
                  "or installed packages).", ""]
        return lines

    lines += ["### What happened", ""]
    lines += _narrative(tree, cwd)
    lines += ["", "### Call flow", "",
              "One line per function per calling context, in the order first called. `x183` = called "
              "183 times from that place; times are totals over those calls; `self` excludes callees. "
              "`uses` lists library functions it called. Values are from the first calls only.",
              "", "```text"]
    _LIMITS.used = 0
    for child in tree.get("children", []):
        _tree_lines(child, cwd, lines, "", True, 0, top=True)
        if _LIMITS.used >= _LIMITS.tree_lines:
            lines.append("... call flow truncated here; report-full.md has all of it")
            break
    lines += ["```", ""]

    hot = [f for f in hot_functions([process], cwd) if f["kind"] not in ("module", "class")]
    if hot:
        total = max(process.get("wall_ms", 0.0), 0.001)
        lines += ["### Where the time went", "",
                  "`Self` is wall time inside the function itself. When `Self CPU` is much lower, the "
                  "function was waiting (I/O, sleep, a lock, a thread join, a network call), not computing. "
                  f"The thread CPU clock is coarse (about 16 ms on Windows), so CPU is shown only from "
                  f"{CPU_MIN_MS} ms of self time.",
                  "", "| Function | Where | Calls | Self | Self CPU | Total | Self % of run |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for fn in hot[:15]:
            cpu = ""
            if fn["self_ms"] >= CPU_MIN_MS:
                waiting = fn["self_cpu_ms"] < 0.3 * fn["self_ms"]
                cpu = _ms(fn["self_cpu_ms"]) + (" (waiting)" if waiting else "")
            lines.append(f"| `{fn['function']}` | {fn['where']} | {fn['calls']} | {_ms(fn['self_ms'])} | "
                         f"{cpu} | {_ms(fn['total_ms'])} | {100 * fn['self_ms'] / total:.1f}% |")
        lines.append("")

    changes = _value_histories(tree, cwd)
    if changes:
        lines += ["### How values changed", "",
                  "Local variables in the first recorded call of each function, in the order they "
                  "changed. `line N` is the line that produced the new value.", ""]
        lines += changes
        lines.append("")

    allocations = process.get("allocations")
    if allocations:
        lines += ["### Memory allocated by line", "", "| Where | KiB | Blocks |", "|---|---:|---:|"]
        lines += [f"| {_rel(a['where'].rsplit(':', 1)[0], cwd)}:{a['where'].rsplit(':', 1)[1]} | "
                  f"{a['kib']} | {a['count']} |" for a in allocations[:12]]
        lines.append("")

    lines += _render_errors(process, cwd)
    return lines


def _render_errors(process: dict[str, Any], cwd: Path) -> list[str]:
    errors = process.get("errors") or []
    if not errors:
        return []
    # One exception produces an event in every frame it passes through; a
    # person wants the story of that exception, not the events.
    chains: dict[Any, list[dict[str, Any]]] = {}
    for err in errors:
        if err["kind"] == "uncaught":
            # The interpreter's last word on an exception already in a chain.
            match = next((c for c in reversed(list(chains.values()))
                          if c[0]["type"] == err["type"] and c[0].get("message") == err.get("message")), None)
            if match is not None:
                match.append(err)
                continue
        chains.setdefault(err.get("exception") or id(err), []).append(err)
    lines = ["### Errors", ""]
    for events in list(chains.values())[:_LIMITS.errors]:
        first = events[0]
        lines.append(f"- {_ms(first.get('at_ms', 0))}: `{first['type']}: {first.get('message')}`: "
                     + " → ".join(_exception_path(events, cwd)) + ".")
    if len(chains) > _LIMITS.errors:
        lines.append(f"- ... {len(chains) - _LIMITS.errors} more exceptions in report-full.md")
    lines.append("")
    return lines


def _exception_path(events: list[dict[str, Any]], cwd: Path) -> list[str]:
    """The path one exception took, from the events every frame reported.

    The interpreter reports more than happened: a RAISE in each frame the
    exception passes through, and a "handled" for the cleanup handler of an
    inlined comprehension or a `finally` that re-raises. Measured on a list
    comprehension that failed: the raw events read "caught in main", although
    nothing caught it. A frame counts as having caught the exception only if
    the exception never left that frame afterwards.
    """
    def where(event: dict[str, Any]) -> str:
        return (f"`{event.get('function') or '?'}` "
                f"({_rel(event.get('file') or '?', cwd)}:{event.get('line') or '?'})")

    steps: list[str] = []
    for i, event in enumerate(events):
        kind, function = event["kind"], event.get("function")
        later = events[i + 1:]
        if kind == "raised":
            if not steps:
                steps.append(f"raised in {where(event)}")
        elif kind == "handled":
            if not any(e["kind"] == "escaped" and e.get("function") == function for e in later):
                steps.append(f"caught in {where(event)}")
        elif kind == "escaped":
            steps.append(f"out of {where(event)}")
        elif kind == "uncaught":
            steps.append("never caught: the program stopped")
    return steps or ["raised"]


def _narrative(tree: dict[str, Any], cwd: Path) -> list[str]:
    lines: list[str] = []
    tops = tree.get("children", [])
    if len(tops) > _LIMITS.narrative_top:
        tops = sorted(tops, key=lambda t: -t["total_ms"])[:_LIMITS.narrative_top]
        lines.append(f"The {len(tops)} longest of {len(tree['children'])} top-level activities, "
                     "longest first:")
    for step, top in enumerate(tops, 1):
        where = f"{_rel(top['file'], cwd)}:{top['line']}"
        if top.get("kind") == "module":
            subject = f"`{_rel(top['file'], cwd)}` ran as the program"
        else:
            subject = f"`{top['function']}()` ({where}) ran"
        if top.get("thread"):
            subject += f" in thread `{top['thread']}`"
        lines.append(f"{step}. {subject} for {_ms(top['total_ms'])}.")
        children = top.get("children", [])
        for child in children[:_LIMITS.narrative_children]:
            first = (child.get("calls") or [{}])[0]
            times = f"{child['count']} times" if child["count"] > 1 else "once"
            if child.get("kind") == "module":
                lines.append(f"   - imported `{_rel(child['file'], cwd)}` ({_ms(child['total_ms'])}).")
                continue
            if child.get("kind") == "class":
                lines.append(f"   - defined class `{child['function']}`.")
                continue
            text = f"   - called `{_label(child, cwd, limit=70)}` {times}, {_ms(child['total_ms'])} in total"
            produced = first.get("return") if child.get("kind") != "generator" else first.get("yield")
            if produced not in (None, "None"):
                verb = "first yielding" if child.get("kind") == "generator" else "returning"
                text += f", {verb} `{produced[:70]}`"
            inner = [g["function"] for g in child.get("children", []) if g.get("kind") != "module"][:4]
            if inner:
                text += "; it called " + ", ".join(f"`{g}`" for g in inner)
            lines.append(text + ".")
        if len(children) > _LIMITS.narrative_children:
            lines.append(f"   - ... and {len(children) - _LIMITS.narrative_children} more distinct calls "
                         "(see the call flow).")
        escaped = [k.split(":", 1)[1] for k in top.get("errors", {}) if k.startswith("escaped")]
        if escaped:
            lines.append(f"   - it ended with an exception: {', '.join(escaped)}.")
    return lines


def _tree_lines(node: dict[str, Any], cwd: Path, out: list[str], prefix: str, last: bool, depth: int,
                top: bool = False) -> None:
    branch = "" if top else ("└─ " if last else "├─ ")
    first = (node.get("calls") or [{}])[0]
    key = (node["file"], node["function"], node["line"])
    levels, calls, children = 1, node["count"], list(node.get("children", []))
    while any((c["file"], c["function"], c["line"]) == key for c in children):
        # Recursion is one fact, not one line per level of depth.
        deeper = next(c for c in children if (c["file"], c["function"], c["line"]) == key)
        children = [c for c in children if c is not deeper] + deeper.get("children", [])
        levels += 1
        calls += deeper["count"]
    node = {**node, "children": children, "count": calls}
    count = f" x{node['count']}" if node["count"] > 1 else ""
    if levels > 1:
        count += f" (recursive, {levels} levels deep)"
    timing = _ms(node["total_ms"])
    if node["children"]:
        timing += f", self {_ms(node['self_ms'])}"
    produced = first.get("yield") if node.get("kind") == "generator" else first.get("return")
    arrow = " yields " if node.get("kind") == "generator" else " -> "
    result = f"{arrow}{_clip(produced, _LIMITS.value_chars)}" if produced not in (None, "None") else ""
    thread = f"  (thread {node['thread']})" if node.get("thread") else ""
    if _LIMITS.used >= _LIMITS.tree_lines:
        return
    _LIMITS.used += 1
    out.append(f"{prefix}{branch}{_label(node, cwd)}{count}  [{_rel(node['file'], cwd)}:{node['line']}]  "
               f"{timing}{result}{thread}")
    child_prefix = prefix + ("" if top else ("   " if last else "│  "))
    details = []
    if node.get("library"):
        top_libs = sorted(node["library"].items(), key=lambda kv: -kv[1])[:6 if _LIMITS.full else 4]
        details.append("uses " + ", ".join(f"{name} x{n}" if n > 1 else name for name, n in top_libs))
    errors = node.get("errors") or {}
    if errors:
        details.append("exceptions: " + ", ".join(f"{k.split(':', 1)[1]} {k.split(':', 1)[0]}"
                                                  + (f" x{v}" if v > 1 else "") for k, v in errors.items()))
    for detail in details:
        if _LIMITS.used >= _LIMITS.tree_lines:
            break
        _LIMITS.used += 1
        out.append(f"{child_prefix}   · {detail}")
    children = node.get("children", [])
    if depth >= _LIMITS.tree_depth and children:
        out.append(f"{child_prefix}└─ ... {len(children)} deeper call path(s), see report-full.md")
        return
    # The costliest paths first when the tree is cut, so what is left is what matters.
    ordered = children if _LIMITS.full or len(children) <= _LIMITS.tree_children else sorted(
        children, key=lambda c: -c["total_ms"])
    shown = ordered[:_LIMITS.tree_children]
    for i, child in enumerate(shown):
        _tree_lines(child, cwd, out, child_prefix,
                    i == len(shown) - 1 and len(children) <= _LIMITS.tree_children, depth + 1)
    if len(children) > _LIMITS.tree_children:
        out.append(f"{child_prefix}└─ ... {len(children) - _LIMITS.tree_children} more call paths "
                   "(the costliest are shown), see report-full.md")


def _value_histories(tree: dict[str, Any], cwd: Path) -> list[str]:
    blocks: list[tuple[float, list[str]]] = []
    seen: set[tuple[str, str, int]] = set()

    def walk(node: dict[str, Any]) -> None:
        key = (node.get("file"), node.get("function"), node.get("line"))
        calls = node.get("calls") or []
        if calls and key not in seen and (calls[0].get("changes") or calls[0].get("return") not in (None, "None")):
            seen.add(key)
            call = calls[0]
            out: list[str] = []
            blocks.append((node.get("total_ms", 0.0), out))
            histories: dict[str, list[str]] = {}
            for change in call.get("changes", []):
                histories.setdefault(change["name"], []).append(
                    f"{_clip(change['value'], _LIMITS.value_chars)} (line {change['line']})")
            head = f"- `{node['function']}` ({_rel(node['file'], cwd)}:{node['line']})"
            if call.get("args"):
                head += f", called with {_args(call, 90)}"
            out.append(head + ":")
            more = call.get("more_changes") or {}
            final = call.get("last") or {}
            for shown, (name, values) in enumerate(histories.items()):
                if shown >= _LIMITS.changes_per_call:
                    out.append(f"  - ... {len(histories) - shown} more variables in raw/")
                    break
                trail = " → ".join(values)
                if name in more:
                    end = final.get(name)
                    trail += f" → ... {more[name] - 1} more changes → " if more[name] > 1 else " → "
                    trail += f"{_clip(end['value'], _LIMITS.value_chars)} (line {end['line']})" if end else "?"
                out.append(f"  - `{name}`: {trail}")
            if call.get("lines_capped"):
                out.append(f"  - (line-by-line values stopped after {call['lines_capped']} lines; "
                           "final values are still shown)")
            produced = call.get("yield") if node.get("kind") == "generator" else call.get("return")
            if produced not in (None, "None"):
                out.append(f"  - {'first yielded' if node.get('kind') == 'generator' else 'returned'} "
                           f"`{_clip(produced, _LIMITS.value_chars * 2)}`")
        for child in node.get("children", []):
            walk(child)

    walk(tree)
    if len(blocks) > _LIMITS.value_functions:
        # The functions the run spent longest in, in the order they ran.
        keep = {id(b) for b in sorted(blocks, key=lambda b: -b[0])[:_LIMITS.value_functions]}
        dropped = len(blocks) - _LIMITS.value_functions
        blocks = [b for b in blocks if id(b) in keep]
        blocks.append((0.0, [f"- ... {dropped} more functions (the costliest are shown); "
                             "report-full.md has them all"]))
    return [line for _, block in blocks for line in block]


def _render_node(profile: dict[str, Any], cwd: Path) -> list[str]:
    lines = [f"## Node.js process ({profile['file']})", "",
             f"Sampled CPU profile over {_ms(profile['duration_ms'])}. Sampling shows where time "
             "went, not every call: functions faster than the sampling interval may not appear.", ""]
    hot = profile["user_hot"] or profile["hot"]
    if hot:
        lines += ["### Where the time went" + (" (your code)" if profile["user_hot"] else ""), "",
                  "| Function | Where | Self | Total |", "|---|---|---:|---:|"]
        for fn in hot[:15]:
            where = f"{_rel(fn['url'].replace('file:///', ''), cwd)}:{fn['line']}" if fn["url"] else "(native)"
            lines.append(f"| `{fn['function']}` | {where} | {_ms(fn['self_ms'])} | {_ms(fn['total_ms'])} |")
        lines.append("")
    if profile.get("tree"):
        lines += ["### Call flow (calls taking at least 1 ms)", "", "```text"]

        def internal(node: dict[str, Any]) -> bool:
            return not node["url"] or node["url"].startswith("node:")

        def walk(node: dict[str, Any], prefix: str, depth: int, folded: int = 0) -> None:
            children = node.get("children", [])
            # Chains of Node's own loader and runtime frames say nothing about
            # the program: fold a single-child internal frame into a count.
            if depth > 0 and internal(node) and len(children) == 1:
                walk(children[0], prefix, depth, folded + 1)
                return
            levels = 1
            same = lambda c: (c["function"], c["url"], c["line"]) == (node["function"], node["url"], node["line"])  # noqa: E731
            while any(same(c) for c in children):
                # A recursion is one fact, not one line per level.
                deeper = next(c for c in children if same(c))
                children = [c for c in children if not same(c)] + deeper.get("children", [])
                levels += 1
            if depth > 0:
                where = f" [{_rel(node['url'].replace('file:///', ''), cwd)}:{node['line']}]" if node["url"] else ""
                skipped = f"({folded} runtime frames) " if folded else ""
                recursion = f" (recursive, {levels} levels deep)" if levels > 1 else ""
                lines.append(f"{prefix}{skipped}{node['function']}{where}  {_ms(node['total_ms'])}{recursion}")
            for child in children[:_LIMITS.tree_children]:
                walk(child, prefix + ("  " if depth > 0 else ""), depth + 1)

        walk(profile["tree"], "", 0)
        lines += ["```", ""]
    return lines


def _render_static(static: dict[str, Any]) -> list[str]:
    if static.get("error"):
        return ["## Static call graph check", "", f"Could not compare: {static['error']}", ""]
    lines = ["## Static call graph check", "",
             f"{static['observed']} distinct calls were observed between project functions: "
             f"{static['in_static_graph']} are in ICN's static call graph, "
             f"{static.get('implicit', 0)} are calls the language makes implicitly (context managers, "
             f"iteration, properties), {static['not_indexed']} involve code ICN has not indexed "
             f"(nested functions, lambdas, module bodies), and **{static['not_in_static_graph']} are "
             "missing from the static graph**. A missing call means dynamic dispatch, a callback, "
             "framework wiring, or a resolver gap: `graph(action='impact')` does not list those "
             "callers.", ""]
    for edge in static.get("missing", [])[:_LIMITS.static]:
        lines.append(f"- `{edge['caller']}` ({edge['caller_file']}:{edge['caller_line']}) → "
                     f"`{edge['callee']}` ({edge['callee_file']}:{edge['callee_line']}), x{edge['count']}")
    lines.append("")
    return lines


def _render_output(summary: dict[str, Any]) -> list[str]:
    lines = []
    for name, key in (("Standard output", "stdout_tail"), ("Standard error", "stderr_tail")):
        tail = summary.get(key) or []
        if tail:
            tail = tail[-_LIMITS.output_lines:]
            lines += [f"## {name} (last {len(tail)} lines)", "", "```text", *tail, "```", ""]
    return lines


def _render_limits(summary: dict[str, Any], pythons: list[dict[str, Any]]) -> list[str]:
    notes = ["Values are shown as truncated representations (about 80 characters) and only for the "
             "first calls of each function in each calling context.",
             "Code inside C extensions and native libraries is timed as part of the Python function "
             "that called it, and appears only by name under `uses`."]
    options = summary["options"]
    if not options["libraries"]:
        notes.append("Standard library and installed packages were not traced; rerun with "
                     "libraries enabled to include them (slower).")
    if not options["values"]:
        notes.append("Value recording was off: flow and timing only.")
    for process in pythons:
        if process.get("values_dropped_after_events"):
            notes.append(f"Process {process['pid']} passed {process['values_dropped_after_events']} recorded "
                         "values, after which only flow and timing were kept.")
        if process.get("node_limit_reached"):
            notes.append(f"Process {process['pid']} reached the call path limit; deeper new paths were "
                         "folded into their callers.")
    notes.append("Recording slows the program down, most in tight loops of small functions, so treat "
                 "absolute times as relative: compare functions with each other, not with production.")
    return ["## What this report cannot show", "", *[f"- {n}" for n in notes], ""]
