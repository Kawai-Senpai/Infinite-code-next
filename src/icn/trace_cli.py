"""`icn trace`: run any command and get a readable report of what it did.

    icn trace python app.py
    icn trace -- pytest -x tests/test_orders.py
    icn trace --libs --memory -- python -m mypackage.cli --flag
    icn trace npm test

The program's own output streams through as usual, with a one-line status on
stderr every two seconds. At the end the path of report.md is printed; hand
that path to an agent, or open it yourself.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def diff_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="icn trace diff",
        description="Compare two recorded runs: the first divergence, values, calls and errors.")
    parser.add_argument("runs", nargs="*", help="two run folders or name fragments; none = the last two")
    parser.add_argument("--cwd", default=None, help="directory whose .icn-trace holds the runs")
    args = parser.parse_args(argv)
    if len(args.runs) not in (0, 2):
        parser.error("pass two runs, or none to compare the two most recent")
    from . import flowdiff
    try:
        result = flowdiff.diff(Path(args.cwd or ".").resolve(), *(args.runs or [None, None]))
    except ValueError as err:
        sys.stderr.write(f"icn trace diff: {err}\n")
        return 1
    sys.stderr.write(f"A: {Path(result['a']['run']).name} (exit {result['a']['exit_code']})\n"
                     f"B: {Path(result['b']['run']).name} (exit {result['b']['exit_code']})\n"
                     f"diverges: {result['diverges']}, values differing: {result['value_differences']}, "
                     f"calls only in A/B: {result['calls_only_a']}/{result['calls_only_b']}\n"
                     f"Report: {result['report']}\n")
    return 0


def main(argv: list[str]) -> int:
    if argv[:1] == ["diff"]:
        return diff_main(argv[1:])
    parser = argparse.ArgumentParser(
        prog="icn trace",
        description="Run a command and write a report of its call flow, values, time and errors.")
    parser.add_argument("--libs", action="store_true",
                        help="also trace the standard library and installed packages (slower)")
    parser.add_argument("--memory", action="store_true",
                        help="record where memory is allocated (slower)")
    parser.add_argument("--no-values", action="store_true",
                        help="record flow and timing only, not arguments or variables (fastest)")
    parser.add_argument("--timeout", type=float, default=None, metavar="SECONDS",
                        help="stop the command after this long")
    parser.add_argument("--max-events", type=int, default=50_000, metavar="N",
                        help="recorded values per process before only timing is kept")
    parser.add_argument("--cwd", default=None, help="directory to run in (default: here)")
    parser.add_argument("--quiet", action="store_true",
                        help="do not stream the program's output or the status line")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="the command to run")
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.print_help(sys.stderr)
        return 2

    from . import flowrun, hooks, paths
    cwd = Path(args.cwd or ".").resolve()
    static = None
    located = hooks.resolve_repo(str(cwd))
    if located:
        store = paths.repo_db_path(located[0])
        if store.exists():
            static = lambda edges: flowrun.check_static_graph(store, edges)  # noqa: E731

    result = flowrun.run(command, cwd, values=not args.no_values, libraries=args.libs,
                         memory=args.memory, timeout=args.timeout, max_events=args.max_events,
                         echo=not args.quiet, static_graph=static)
    if not result.get("ok"):
        sys.stderr.write(f"icn trace: {result.get('error')}\n")
        return 1
    out = sys.stderr
    out.write("\n" + "-" * 60 + "\n")
    status = "timed out" if result["timed_out"] else f"exit code {result['exit_code']}"
    out.write(f"icn trace: {status}, {result['wall_ms'] / 1000:.2f} s, "
              f"{result['python_processes']} Python process(es), {result['node_profiles']} Node profile(s)\n")
    for fn in result.get("hot_functions", [])[:3]:
        out.write(f"  hot: {fn['function']} ({fn['where']}) self {fn['self_ms']:.1f} ms x{fn['calls']}\n")
    if result.get("exceptions"):
        out.write(f"  exceptions raised: {result['exceptions']} (see the Errors section)\n")
    out.write(f"Report: {result['report']}\n")
    # The traced command's exit code, so `icn trace` composes in scripts and CI.
    return result["exit_code"] if isinstance(result["exit_code"], int) else 1
