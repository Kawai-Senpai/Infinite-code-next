"""Compare two recorded runs: where did their behaviour first diverge?

    icn trace diff                      the two most recent runs here
    icn trace diff <run-a> <run-b>      run folders, or unique parts of their names

A stack trace says where a program stopped. Two recordings of the same program,
one that worked and one that did not, say where it started behaving
differently, which is usually far earlier and is the thing worth reading. The
comparison is structural, over the calling context trees flowtrace.py wrote:

    divergence   walking both trees in call order, the first place the
                 sequence of calls differs.
    calls        call paths present in only one run, and paths whose call
                 count changed.
    values       for functions both runs executed, the first variable whose
                 history differs, with the line that produced each value.
    errors       exceptions raised in only one run.
    time         call paths whose total time moved by more than both 25% and
                 5 ms.

Standard library only, like the rest of the trace tooling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .flowrun import TRACE_DIR, _clip, _display_command, _load_python, _ms, _rel

TIME_RATIO = 1.25
TIME_FLOOR_MS = 5.0
LIST_LIMIT = 20


# ------------------------------------------------------------------ locating runs

def runs_in(cwd: Path) -> list[Path]:
    base = cwd / TRACE_DIR
    if not base.is_dir():
        return []
    return sorted((p for p in base.iterdir() if p.is_dir() and (p / "summary.json").exists()),
                  key=lambda p: p.name)


def find_run(cwd: Path, name: str) -> Path:
    path = Path(name)
    if path.is_dir() and (path / "summary.json").exists():
        return path.resolve()
    recorded = runs_in(cwd)
    # An exact name wins outright. Matching by substring alone made the full
    # name of a run ambiguous against its own retry suffix: `...-a` matched
    # both `...-a` and `...-a~2`, so citing a run by its real name failed.
    exact = [p for p in recorded if p.name == name]
    if exact:
        return exact[0]
    matches = [p for p in recorded if name in p.name]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"no recorded run matches {name!r} in {cwd / TRACE_DIR}")
    raise ValueError(f"{name!r} matches {len(matches)} runs: " + ", ".join(p.name for p in matches[-5:]))


# ------------------------------------------------------------------ comparison

def summary_of(run: Path) -> dict[str, Any]:
    """The stored summary of one recorded run."""
    return json.loads((run / "summary.json").read_text(encoding="utf-8"))


def _main_tree(run: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    pythons = _load_python(run / "raw")
    return summary, (pythons[0] if pythons else None)


def _key(node: dict[str, Any], cwd: Path) -> tuple[str, str]:
    # By file and qualified name, not line: an edit above a function moves its
    # first line without making it a different function.
    return (_rel(node.get("file") or "", cwd), node.get("function") or "")


def _label(key: tuple[str, str]) -> str:
    return f"`{key[1]}` ({key[0]})"


def _children(node: dict[str, Any], cwd: Path) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for child in node.get("children", []):
        out.setdefault(_key(child, cwd), child)
    return out


def _first_divergence(a: dict[str, Any], b: dict[str, Any], cwd_a: Path, cwd_b: Path,
                      path: tuple = ()) -> dict[str, Any] | None:
    """Depth first, in call order: the first node whose calls differ."""
    kids_a, kids_b = _children(a, cwd_a), _children(b, cwd_b)
    order_a, order_b = list(kids_a), list(kids_b)
    for i in range(max(len(order_a), len(order_b))):
        ka = order_a[i] if i < len(order_a) else None
        kb = order_b[i] if i < len(order_b) else None
        if ka != kb:
            return {"path": path, "position": i, "a_next": ka, "b_next": kb,
                    "a_calls": order_a, "b_calls": order_b}
        found = _first_divergence(kids_a[ka], kids_b[kb], cwd_a, cwd_b, path + (ka,))
        if found:
            return found
    return None


def _flatten(node: dict[str, Any], cwd: Path, path: tuple = (),
             out: dict[tuple, dict[str, Any]] | None = None) -> dict[tuple, dict[str, Any]]:
    out = {} if out is None else out
    for child in node.get("children", []):
        child_path = path + (_key(child, cwd),)
        if child_path not in out:
            out[child_path] = child
        _flatten(child, cwd, child_path, out)
    return out


def _history(call: dict[str, Any]) -> dict[str, list[tuple[int, str, float]]]:
    out: dict[str, list[tuple[int, str, float]]] = {}
    for change in call.get("changes", []):
        out.setdefault(change["name"], []).append(
            (change["line"], change["value"], change.get("at_ms", call.get("at_ms", 0.0))))
    return out


def _value_difference(node_a: dict[str, Any], node_b: dict[str, Any]) -> dict[str, Any] | None:
    call_a = (node_a.get("calls") or [None])[0]
    call_b = (node_b.get("calls") or [None])[0]
    if not call_a or not call_b:
        return None
    # `at` is when run B produced the differing value, so differences across
    # the whole tree can be put in the order they happened. Tree order put a
    # caller's variable ahead of the callee value that caused it.
    if call_a.get("args") != call_b.get("args"):
        for name in list(call_a.get("args", {})) + list(call_b.get("args", {})):
            va, vb = call_a.get("args", {}).get(name), call_b.get("args", {}).get(name)
            if va != vb:
                return {"kind": "argument", "name": name, "a": va, "b": vb,
                        "at": call_b.get("at_ms", 0.0)}
    hist_a, hist_b = _history(call_a), _history(call_b)
    earliest = None
    for name in list(hist_a) + [n for n in hist_b if n not in hist_a]:
        seq_a, seq_b = hist_a.get(name, []), hist_b.get(name, [])
        for i in range(max(len(seq_a), len(seq_b))):
            ea = seq_a[i] if i < len(seq_a) else None
            eb = seq_b[i] if i < len(seq_b) else None
            if (ea and ea[1]) != (eb and eb[1]):
                at = eb[2] if eb else (ea[2] if ea else 0.0)
                if earliest is None or at < earliest["at"]:
                    earliest = {"kind": "variable", "name": name, "step": i + 1,
                                "a": ea[1] if ea else None, "a_line": ea[0] if ea else None,
                                "b": eb[1] if eb else None, "b_line": eb[0] if eb else None, "at": at}
                break
    if earliest:
        return earliest
    ra, rb = call_a.get("return"), call_b.get("return")
    if ra != rb:
        # A return happens when its frame exits, which is before the caller
        # assigns it: ordering returns last put the deepest cause below the
        # consequences it produced.
        return {"kind": "return", "a": ra, "b": rb,
                "at": call_b.get("returned_at_ms", call_b.get("at_ms", 0.0))}
    return None


def compare(run_a: Path, run_b: Path) -> dict[str, Any]:
    summary_a, proc_a = _main_tree(run_a)
    summary_b, proc_b = _main_tree(run_b)
    cwd_a, cwd_b = Path(summary_a["cwd"]), Path(summary_b["cwd"])
    result: dict[str, Any] = {
        "a": {"run": str(run_a), "command": summary_a["command"], "exit_code": summary_a["process"]["exit_code"],
              "wall_ms": summary_a["process"]["wall_ms"]},
        "b": {"run": str(run_b), "command": summary_b["command"], "exit_code": summary_b["process"]["exit_code"],
              "wall_ms": summary_b["process"]["wall_ms"]},
    }
    if proc_a is None or proc_b is None:
        result["note"] = ("one of the runs recorded no Python process, so only exit codes, time and "
                          "output can be compared")
        result["output"] = _output_difference(summary_a, summary_b)
        return result

    tree_a, tree_b = proc_a["tree"], proc_b["tree"]
    result["divergence"] = _first_divergence(tree_a, tree_b, cwd_a, cwd_b)
    flat_a, flat_b = _flatten(tree_a, cwd_a), _flatten(tree_b, cwd_b)

    only_a = [p for p in flat_a if p not in flat_b]
    only_b = [p for p in flat_b if p not in flat_a]
    # A path whose parent is itself new is part of that parent's story.
    result["only_a"] = [p for p in only_a if p[:-1] not in only_a]
    result["only_b"] = [p for p in only_b if p[:-1] not in only_b]

    counts, times, values = [], [], []
    for path, node_a in flat_a.items():
        node_b = flat_b.get(path)
        if node_b is None:
            continue
        if node_a["count"] != node_b["count"]:
            counts.append((path, node_a["count"], node_b["count"]))
        ta, tb = node_a["total_ms"], node_b["total_ms"]
        if abs(tb - ta) >= TIME_FLOOR_MS and max(ta, tb) >= TIME_RATIO * max(min(ta, tb), 0.001):
            times.append((path, ta, tb))
        diff = _value_difference(node_a, node_b)
        if diff:
            values.append((path, diff))
    result["count_changes"] = counts
    result["time_changes"] = sorted(times, key=lambda t: -abs(t[2] - t[1]))
    # In the order run B produced them: the earliest difference is usually the cause.
    result["value_changes"] = sorted(values, key=lambda v: v[1].get("at", 0.0))

    def errors(proc: dict[str, Any], cwd: Path) -> dict[tuple, dict[str, Any]]:
        # One exception passes through several frames; it is keyed where it
        # was first raised, so it is one row, not one per frame.
        out, seen = {}, set()
        for err in proc.get("errors", []):
            ident = err.get("exception")
            if err["kind"] not in ("raised", "uncaught") or (ident and ident in seen):
                continue
            if ident:
                seen.add(ident)
            elif any(e["type"] == err["type"] and e.get("message") == err.get("message") for e in out.values()):
                continue
            out.setdefault((err["type"], _rel(err.get("file") or "", cwd), err.get("function")), err)
        return out

    errors_a, errors_b = errors(proc_a, cwd_a), errors(proc_b, cwd_b)
    result["errors_only_a"] = [errors_a[k] for k in errors_a if k not in errors_b]
    result["errors_only_b"] = [errors_b[k] for k in errors_b if k not in errors_a]
    result["output"] = _output_difference(summary_a, summary_b)
    result["cwd_b"] = str(cwd_b)
    return result


def _output_difference(summary_a: dict[str, Any], summary_b: dict[str, Any]) -> dict[str, Any]:
    out_a = summary_a.get("stdout_tail", []) + summary_a.get("stderr_tail", [])
    out_b = summary_b.get("stdout_tail", []) + summary_b.get("stderr_tail", [])
    return {"only_a": [l for l in out_a if l not in out_b][-10:],
            "only_b": [l for l in out_b if l not in out_a][-10:]}


# ------------------------------------------------------------------ rendering

def _path_text(path: tuple) -> str:
    return " → ".join(f"`{p[1]}`" for p in path) if path else "the program's top level"


def render(result: dict[str, Any]) -> str:
    a, b = result["a"], result["b"]
    lines = ["# Execution diff", "",
             f"- **A**: `{_display_command(a['command'])}` → exit {a['exit_code']}, {_ms(a['wall_ms'])} ({Path(a['run']).name})",
             f"- **B**: `{_display_command(b['command'])}` → exit {b['exit_code']}, {_ms(b['wall_ms'])} ({Path(b['run']).name})",
             ""]
    if result.get("note"):
        lines += [result["note"], ""]
    else:
        lines += _render_summary(result)
        lines += _render_divergence(result)
        lines += _render_values(result)
        lines += _render_calls(result)
        lines += _render_errors(result)
        lines += _render_time(result)
    output = result.get("output") or {}
    if output.get("only_a") or output.get("only_b"):
        lines += ["## Output that differs", "", "```text"]
        lines += [f"A: {l}" for l in output.get("only_a", [])]
        lines += [f"B: {l}" for l in output.get("only_b", [])]
        lines += ["```", ""]
    lines += ["## Reading this", "",
              "- Only the busiest Python process of each run is compared.",
              "- Values are compared from the first recorded call of each function in each calling "
              "context, as truncated representations; a difference in a later call is not seen.",
              "- Object addresses and timestamps differ between any two runs; treat such value "
              "differences as noise.", ""]
    return "\n".join(lines).rstrip() + "\n"


def _render_summary(result: dict[str, Any]) -> list[str]:
    div = result.get("divergence")
    parts = []
    if result["a"]["exit_code"] != result["b"]["exit_code"]:
        parts.append(f"exit code changed from {result['a']['exit_code']} to {result['b']['exit_code']}")
    if div:
        parts.append("the call sequence diverges inside " + _path_text(div["path"]))
    if result["value_changes"]:
        path, diff = result["value_changes"][0]
        parts.append(f"the first value difference is `{diff.get('name', 'return')}` in `{path[-1][1]}`")
    if result["errors_only_b"]:
        parts.append(f"{len(result['errors_only_b'])} exception(s) happen only in B")
    if not parts:
        parts.append("no behavioural difference was recorded")
    text = "; ".join(parts)
    return ["## In short", "", text[:1].upper() + text[1:] + ".", ""]


def _render_divergence(result: dict[str, Any]) -> list[str]:
    div = result.get("divergence")
    if not div:
        return ["## First divergence", "", "Both runs made the same calls in the same order.", ""]
    lines = ["## First divergence", "", f"Inside {_path_text(div['path'])}, after "
             f"{div['position']} identical call(s):", ""]
    lines.append(f"- A next called {_label(div['a_next']) if div['a_next'] else 'nothing more'}")
    lines.append(f"- B next called {_label(div['b_next']) if div['b_next'] else 'nothing more'}")
    shared = div["a_calls"][:div["position"]]
    lines += ["", "```text", "A: " + " → ".join(k[1] for k in div["a_calls"][:12]),
              "B: " + " → ".join(k[1] for k in div["b_calls"][:12]), "```", ""]
    if shared:
        lines.append(f"Calls both made first: {', '.join(k[1] for k in shared[-5:])}.")
        lines.append("")
    return lines


def _render_values(result: dict[str, Any]) -> list[str]:
    values = result["value_changes"]
    if not values:
        return []
    lines = ["## Values that differ", "",
             "In functions both runs executed, in the order run B produced them. The first row is usually the cause; "
             "later rows are often its consequences.", ""]
    for path, diff in values[:LIST_LIMIT]:
        where = _path_text(path)
        if diff["kind"] == "argument":
            lines.append(f"- {where}: argument `{diff['name']}` was `{_clip(str(diff['a']), 60)}` in A, "
                         f"`{_clip(str(diff['b']), 60)}` in B")
        elif diff["kind"] == "variable":
            a_text = f"`{_clip(str(diff['a']), 60)}` (line {diff['a_line']})" if diff["a"] is not None else "never set"
            b_text = f"`{_clip(str(diff['b']), 60)}` (line {diff['b_line']})" if diff["b"] is not None else "never set"
            lines.append(f"- {where}: `{diff['name']}`, change #{diff['step']}: {a_text} in A, {b_text} in B")
        else:
            lines.append(f"- {where}: returned `{_clip(str(diff['a']), 60)}` in A, "
                         f"`{_clip(str(diff['b']), 60)}` in B")
    if len(values) > LIST_LIMIT:
        lines.append(f"- ... {len(values) - LIST_LIMIT} more")
    lines.append("")
    return lines


def _render_calls(result: dict[str, Any]) -> list[str]:
    lines = []
    for title, paths in (("Calls only in A", result["only_a"]), ("Calls only in B", result["only_b"])):
        if paths:
            lines += [f"## {title}", ""]
            lines += [f"- {_path_text(p)}" for p in paths[:LIST_LIMIT]]
            if len(paths) > LIST_LIMIT:
                lines.append(f"- ... {len(paths) - LIST_LIMIT} more")
            lines.append("")
    if result["count_changes"]:
        lines += ["## Call counts that changed", ""]
        lines += [f"- {_path_text(p)}: x{a} → x{b}" for p, a, b in result["count_changes"][:LIST_LIMIT]]
        lines.append("")
    return lines


def _render_errors(result: dict[str, Any]) -> list[str]:
    lines = []
    for title, errs in (("Exceptions only in A", result["errors_only_a"]),
                        ("Exceptions only in B", result["errors_only_b"])):
        if errs:
            lines += [f"## {title}", ""]
            lines += [f"- `{e['type']}: {_clip(str(e.get('message')), 100)}` in `{e.get('function')}` "
                      f"(line {e.get('line')})" for e in errs[:LIST_LIMIT]]
            lines.append("")
    return lines


def _render_time(result: dict[str, Any]) -> list[str]:
    times = result["time_changes"]
    if not times:
        return []
    lines = ["## Time that changed", "", "| Call path | A | B |", "|---|---:|---:|"]
    lines += [f"| {_path_text(p)} | {_ms(a)} | {_ms(b)} |" for p, a, b in times[:LIST_LIMIT]]
    lines.append("")
    return lines


def diff(cwd: Path, a: str | None = None, b: str | None = None) -> dict[str, Any]:
    """Compare two runs and write the diff next to run B. Raises ValueError."""
    if a is None and b is None:
        recorded = runs_in(cwd)
        if len(recorded) < 2:
            raise ValueError(f"need two recorded runs in {cwd / TRACE_DIR}; found {len(recorded)}")
        run_a, run_b = recorded[-2], recorded[-1]
    elif a is not None and b is not None:
        run_a, run_b = find_run(cwd, a), find_run(cwd, b)
    else:
        raise ValueError("pass two runs, or none to compare the two most recent")
    result = compare(run_a, run_b)
    out = run_b / f"diff-vs-{run_a.name}.md"
    out.write_text(render(result), encoding="utf-8")
    return {"ok": True, "report": str(out), "a": result["a"], "b": result["b"],
            "diverges": bool(result.get("divergence")),
            "value_differences": len(result.get("value_changes", [])),
            "calls_only_a": len(result.get("only_a", [])), "calls_only_b": len(result.get("only_b", [])),
            "exceptions_only_b": len(result.get("errors_only_b", []))}
