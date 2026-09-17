# Recording what a program actually did

`icn trace` runs a command and writes a report of what happened while it ran:
which functions ran and in what order, what they were called with and what
they returned, how local variables changed, where the time went, and which
exceptions were raised and where they were caught. `icn trace diff` compares
two such recordings and says where their behaviour first diverged.

## Contents

1. [Run it](#1-run-it)
2. [What gets recorded](#2-what-gets-recorded)
3. [Reading the report](#3-reading-the-report)
4. [Comparing two runs](#4-comparing-two-runs)
5. [From an agent (MCP)](#5-from-an-agent-mcp)
6. [Options](#6-options)
7. [How it works](#7-how-it-works)
8. [Limits](#8-limits)
9. [Troubleshooting](#9-troubleshooting)

## 1. Run it

Put `icn trace` in front of the command you would normally run:

```sh
icn trace python app.py
icn trace python -m mypackage.cli --flag value
icn trace -- pytest -x tests/test_orders.py
icn trace npm test
icn trace --timeout 60 -- uvicorn app:app
```

The program's output streams through as usual, with one status line on stderr
every two seconds showing where the program is. At the end:

```text
icn trace: exit code 0, 0.18 s, 1 Python process(es), 0 Node profile(s)
  hot: slow_shipping (app.py:13) self 91.3 ms x3
Report: C:\work\shop\.icn-trace\20260917-155150-app\report.md
```

`icn trace` exits with the traced command's exit code, so it works in scripts
and CI. `.icn-trace/` ignores itself for git, and `.icn-trace/latest.txt` holds
the path of the newest report.

## 2. What gets recorded

| Runtime | How | What you get |
|---|---|---|
| Python (any process the command starts: the script, pytest, workers) | a recorder injected at interpreter start | call flow, arguments, return values, variable changes, wall and CPU time per function, library calls, exceptions, peak memory |
| Node.js (node, npm, npx, jest) | `--cpu-prof` through `NODE_OPTIONS` | sampled CPU profile: where time went, call tree of calls taking at least 1 ms |
| Anything else | the process itself | exit code, wall time, CPU time, peak memory, last lines of output |

By default only project code is recorded in detail: files under the directory
the command runs in (and the directory of a script named on the command line),
not the standard library or installed packages. Calls project code makes into
libraries are still listed by name under `uses`.

## 3. Reading the report

`report.md` is sized for an agent to read; `report-full.md` beside it has every
call path, value and error. Sections, in order:

- **What happened**: a numbered story of the top-level activity and what each
  part called, with call counts and returned values.
- **Call flow**: a calling context tree. One line per function per call path;
  `x183` means 183 calls from that place; times are totals over those calls,
  and `self` excludes the functions it called. Recursion is folded into one
  line with its depth.
- **Where the time went**: functions by self time. When `Self CPU` is much
  lower than `Self`, the function was waiting (I/O, a lock, a sleep, a thread
  join), not computing.
- **How values changed**: for the first recorded call of each function, every
  local variable in the order it changed, with the line that produced each
  value. Mutations count: `items.append(x)` shows as a new value of `items`.
  Plain objects show their fields, `Order(id=1, price=60)`, not an address.
- **Errors**: one line per exception, following its path, for example
  `raised in parse (crash.py:2) → out of parse → out of main → never caught`.
- **Static call graph check** (indexed repositories): calls that ran but are
  missing from ICN's static graph. Constructors, context managers, iteration
  and properties are counted separately as implicit calls.
- **Output** and **What this report cannot show**.

## 4. Comparing two runs

Record a run that works and one that does not, then:

```sh
icn trace diff                                  # the two most recent runs
icn trace diff 20260917-160602 20260917-160603  # by folder name fragments
```

The diff is written next to the second run as `diff-vs-<first run>.md`:

```text
## First divergence

Inside `<module>` → `login`, after 1 identical call(s):

- A next called `validate_permissions` (perms.py)
- B next called `refresh_permissions` (perms.py)

## Values that differ

- `<module>` → `login` → `load_user`: `role`, change #1: `'admin'` (line 7) in A, `None` (line 7) in B
- `<module>` → `login`: `user`, change #1: `{'id': 7, 'role': 'admin'}` (line 26) in A, ...
```

Value differences are listed in the order the second run produced them, so the
first row is usually the cause and the rest its consequences.

## 5. From an agent (MCP)

Two actions on the existing `graph` tool:

```text
graph(action="run", target="pytest -x tests/test_orders.py", timeout_seconds=300)
graph(action="run_diff")                              # two most recent runs in root
graph(action="run_diff", target="<run a>", to="<run b>")
```

Both return the path of the report. The agent reads that file. `run_options`
takes `libraries`, `memory` or `no-values`. A command without shell syntax is
run directly rather than through a shell, so the process statistics are the
program's own.

### A recording is evidence

Recordings are part of the knowledge graph, not a side artifact:

```text
record(evidence=["20260917-160603-perms"], invariants=[...], symbols=[...])
```

The run's command, exit code, timing and hottest functions are written into
every memory of that event, with the path of its report, exactly as a lab run
id is. `workspace(action="open")` lists the most recent recordings, and
`investigate(action="why", symbol=...)` reports whether a recorded run actually
executed that symbol and which functions called it on that run. That is the
complement of `graph(action="impact")`, which answers who *could* call
something and labels the answer a lower bound. A symbol the recordings never
reached is reported as not reached, never as unreachable.

## 6. Options

| CLI | MCP `run_options` | Effect |
|---|---|---|
| `--libs` | `libraries` | also record the standard library and installed packages (slower) |
| `--memory` | `memory` | record allocation sites with tracemalloc (slower) |
| `--no-values` | `no-values` | flow and timing only (fastest) |
| `--timeout S` | `timeout_seconds` | stop the command, and every process it started, after S seconds |
| `--max-events N` | | recorded values per process before only flow and timing are kept (default 50,000) |
| `--cwd DIR` | `root` | directory to run in |
| `--quiet` | | do not stream output or the status line |

`.icn-trace/` keeps the 20 most recent runs and deletes older ones, so it stays
bounded however often you record (a traced pytest run is about 1.3 MB). Set
`ICN_TRACE_KEEP` to change that.

## 7. How it works

`icn trace` puts a small `sitecustomize.py` first on `PYTHONPATH`. Every Python
interpreter the command starts imports it at startup; it loads the recorder by
file path (never the `icn` package, which would pull native extensions into
your process) and then hands over to any `sitecustomize` your environment
already had.

On Python 3.12 and later the recorder uses `sys.monitoring` (PEP 669). Code
outside the project is switched off per location, so the standard library runs
at close to full speed. Calls build a calling context tree, so a loop of a
million calls stays one line with a count. Values are recorded by comparing
truncated representations of the locals line by line, for the first three calls
of each function in each calling context; loop variables keep their first six
values and their final one. At exit the tree is written as JSON, and `icn trace`
renders the reports from it.

## 8. Limits

- Recording slows the program down, most in tight loops of small functions.
  Compare times with each other, not with production.
- Values are truncated representations, recorded for the first calls only.
- Code inside C extensions is timed as part of the Python function that called
  it and appears only by name.
- Node.js profiles are sampled: functions faster than the sampling interval may
  not appear, and values are not recorded.
- The diff compares the busiest Python process of each run, and the first
  recorded call of each function.
- On Windows the thread CPU clock ticks about every 16 ms, so CPU time is shown
  only for functions with at least 20 ms of self time. It is also measured only
  for frames within 12 levels of the top, because reading that clock on every
  call of a deep recursion costs more than the answer is worth.

## 9. Troubleshooting

**"No Python or Node.js code ran under the recorder."** The command ran
something else, or the interpreter was started with `-I`, `-E` or `-S`, which
ignore `PYTHONPATH` and `sitecustomize`.

**Only `<module>` appears, nothing inside.** The code lives outside the
directory the command ran in. Run from the project root, or pass `--cwd`.

**"Killed before it could finish; this is its last snapshot."** The process was
stopped (timeout or a crash that skipped exit handlers). The report uses the
snapshot written every second while it ran.

**`report.md` is too long.** Use `--no-values`, or read `What happened` and
`Where the time went` first; the full detail is in `report-full.md`.
