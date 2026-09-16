"""The process that watches one experiment run.

Launched detached by lab.start_run as `python labrun.py <run_dir>`, by file path
rather than `-m icn.labrun`, so it needs nothing on sys.path and imports nothing
but the standard library. That matters twice: the MCP server that launched it
may exit long before the run does, and importing the icn package would drag in
the native extensions whose late import is known to wedge on Windows.

Contract with lab.py, all inside <run_dir>:
    spec.json     written by the launcher: command, cwd, log, timeout, env
    state.json    written here once the command has started: pids, start time
    heartbeat     rewritten every HEARTBEAT_SECONDS while the command runs
    cancel        created by lab.cancel_run; this process kills the tree
    exit.json     written here last, atomically: exit code, metrics, outcome

exit.json is the only authoritative end state. A run with no exit.json and a
stale heartbeat is one whose watcher died, which lab.py reports as `lost`.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

HEARTBEAT_SECONDS = 2.0
POLL_SECONDS = 0.25
KILL_GRACE_SECONDS = 5.0

# One metric per line, printed by the experiment itself:  ICN_METRIC loss=0.4312
# A fixed, language-agnostic line format rather than a results file, so any
# command that can print can report, and the log stays the single evidence.
METRIC_LINE = re.compile(
    r"^[ \t]*ICN_METRIC[ \t]+([A-Za-z0-9_.:/-]+)[ \t]*=[ \t]*"
    r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[-+]?inf|nan)[ \t]*\r?$",
    re.MULTILINE | re.IGNORECASE,
)


def parse_metrics(text: str) -> dict[str, float]:
    """Every ICN_METRIC line in `text`. The last value printed for a name wins."""
    found: dict[str, float] = {}
    for name, value in METRIC_LINE.findall(text):
        try:
            found[name] = float(value)
        except ValueError:
            continue
    return found


def metrics_from_log(log_path: Path) -> dict[str, float]:
    try:
        text = log_path.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return {}
    return parse_metrics(text)


def _write_json_atomic(path: Path, payload: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temp, path)


def _kill_tree(child: subprocess.Popen) -> None:
    """Stop the command and everything it started.

    shell=True means the pid we hold is the shell, not the training script, so
    killing only that pid would orphan the real work and leave it running.
    """
    if child.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, creationflags=0x08000000, check=False)
    else:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return
        deadline = time.monotonic() + KILL_GRACE_SECONDS
        while child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    try:
        child.wait(timeout=KILL_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass


def run(run_dir: Path) -> int:
    spec = json.loads((run_dir / "spec.json").read_text(encoding="utf-8"))
    log_path = Path(spec["log"])
    exit_path = run_dir / "exit.json"
    cancel_path = run_dir / "cancel"
    heartbeat_path = run_dir / "heartbeat"
    timeout = spec.get("timeout_seconds")

    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in (spec.get("env") or {}).items()})

    kwargs: dict = {}
    if sys.platform == "win32":
        # CREATE_NO_WINDOW: this watcher has no console, so without it every
        # cmd.exe it starts would open a visible window. New process group so
        # the kill below reaches the whole tree.
        kwargs["creationflags"] = 0x08000000 | 0x00000200
    else:
        kwargs["start_new_session"] = True

    started = time.time()
    with open(log_path, "ab", buffering=0) as log:
        try:
            child = subprocess.Popen(spec["command"], shell=True, cwd=spec["cwd"],
                                     stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, env=env, **kwargs)
        except OSError as exc:
            log.write(f"\n[icn] could not start the command: {exc}\n".encode("utf-8"))
            _write_json_atomic(exit_path, {
                "exit_code": 127, "outcome": "failed", "error": str(exc),
                "started_at": started, "ended_at": time.time(),
                "metrics": {},
            })
            return 127

        _write_json_atomic(run_dir / "state.json", {
            "runner_pid": os.getpid(), "child_pid": child.pid, "started_at": started,
        })

        outcome = None
        last_beat = 0.0
        while child.poll() is None:
            now = time.monotonic()
            if now - last_beat >= HEARTBEAT_SECONDS:
                try:
                    heartbeat_path.write_text(str(time.time()), encoding="utf-8")
                except OSError:
                    pass
                last_beat = now
            if cancel_path.exists():
                outcome = "cancelled"
                _kill_tree(child)
                break
            if timeout and time.time() - started > float(timeout):
                outcome = "timed_out"
                _kill_tree(child)
                break
            time.sleep(POLL_SECONDS)

        code = child.wait()
        if outcome == "cancelled":
            log.write(b"\n[icn] run cancelled\n")
        elif outcome == "timed_out":
            log.write(f"\n[icn] run exceeded its {timeout}s timeout and was stopped\n".encode("utf-8"))

    if outcome is None:
        outcome = "done" if code == 0 else "failed"
    _write_json_atomic(exit_path, {
        "exit_code": code, "outcome": outcome, "started_at": started,
        "ended_at": time.time(), "metrics": metrics_from_log(log_path),
    })
    return code


if __name__ == "__main__":
    sys.exit(run(Path(sys.argv[1])) if len(sys.argv) > 1 else 2)
