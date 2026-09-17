"""Background transcript sync, running inside the MCP server.

The archive only helps if it is current when the vendor deletes the original,
and nobody remembers to run a sync script. So the server does it: a daemon
thread started at boot, copying new vendor bytes into the archive and folding
them into the index, on a timer.

The complication is that there is never one server. Every editor window and
every terminal starts its own `python -m icn`, so N processes wake up wanting
to write the same SQLite files. Three things keep that safe:

    1. A Lease (the same heartbeat file the code indexer uses, different name).
       Exactly one process syncs; the rest skip their tick and try again later.
       The holder can be SIGKILLed and the lease goes stale in 60s.
    2. A time budget per tick, so a first run over a 459MB store yields the
       lease instead of holding it for minutes. The next tick continues where
       this one stopped - refresh() is incremental by mtime and size.
    3. Daemon threads and no stdout. Anything written to stdout would be framed
       as an MCP message and corrupt the protocol; failures go to the log file.

A tick is cheap when there is nothing to do: discovery stats the vendor dirs,
finds every mtime unchanged, and returns.
"""

from __future__ import annotations

import os
import threading
import time
import traceback
from typing import Any

from . import paths
from .lease import Lease

ENV_AUTOSYNC = "ICN_TRANSCRIPTS_AUTOSYNC"          # 1 (default) | 0
ENV_INTERVAL = "ICN_TRANSCRIPTS_SYNC_INTERVAL"     # seconds between ticks
ENV_BUDGET = "ICN_TRANSCRIPTS_SYNC_BUDGET"         # seconds of work per tick

DEFAULT_INTERVAL = 900.0        # 15 minutes: transcripts are not latency critical
DEFAULT_BUDGET = 60.0
FIRST_DELAY = 20.0              # let the server answer its first calls before any IO

_thread: threading.Thread | None = None
_stop = threading.Event()
_last: dict[str, Any] = {"state": "not started"}
_lock = threading.Lock()


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default
    return value if value > 0 else default


def autosync_enabled() -> bool:
    return os.environ.get(ENV_AUTOSYNC, "1").strip().lower() not in ("0", "false", "no", "off")


def lease_path():
    return paths.storage_root() / "transcripts-sync.lease"


def _log(message: str) -> None:
    """Never stdout: this process speaks MCP framing on it."""
    try:
        directory = paths.logs_dir()
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "transcript-sync.log", "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [{os.getpid()}] {message}\n")
    except OSError:
        pass


def run_once(budget_seconds: float | None = None, wait_for_lease: bool = False) -> dict[str, Any]:
    """One sync tick. Returns the refresh counts, or why it did nothing.

    Imports transcripts lazily: the module opens SQLite and pulls in the
    adapters, and a server that never touches conversations should not pay for
    that at import time.
    """
    from . import transcripts

    budget = budget_seconds if budget_seconds is not None else _env_float(ENV_BUDGET, DEFAULT_BUDGET)
    lease = Lease(lease_path(), owner="transcript-sync")
    deadline = time.monotonic() + budget
    while not lease.try_acquire():
        if not wait_for_lease or time.monotonic() > deadline:
            holder = lease.holder() or {}
            return {"skipped": "another process is syncing", "holder_pid": holder.get("pid"),
                    "holder_age_seconds": holder.get("age_seconds")}
        if _stop.wait(1.0):
            return {"skipped": "stopping"}
    index = None
    try:
        index = transcripts.Index()
        result = index.refresh(limit_seconds=budget)
        result["lease"] = "held"
        return result
    finally:
        if index is not None:
            index.close()
        lease.release()


def _loop(interval: float, budget: float) -> None:
    if _stop.wait(FIRST_DELAY):
        return
    while not _stop.is_set():
        started = time.time()
        try:
            result = run_once(budget_seconds=budget)
            with _lock:
                _last.clear()
                _last.update({"state": "ok", "at": time.strftime("%Y-%m-%dT%H:%M:%S"), **result})
            if result.get("indexed"):
                _log(f"indexed={result['indexed']} archive={result.get('archive', {}).get('bytes', 0)}B "
                     f"in {result.get('seconds')}s")
        except Exception as exc:  # noqa: BLE001 - a sync failure must not kill the server
            with _lock:
                _last.clear()
                _last.update({"state": "error", "error": f"{type(exc).__name__}: {exc}",
                              "at": time.strftime("%Y-%m-%dT%H:%M:%S")})
            _log(f"tick failed: {traceback.format_exc()}")
        # Sleep from the end of the tick, so a slow tick does not stack.
        elapsed = time.time() - started
        if _stop.wait(max(5.0, interval - elapsed)):
            return


def start(force: bool = False) -> dict[str, Any]:
    """Start the daemon thread. Idempotent: a second call is a no-op."""
    global _thread
    if not force and not autosync_enabled():
        with _lock:
            _last.clear()
            _last.update({"state": "disabled", "reason": f"{ENV_AUTOSYNC}=0"})
        return dict(_last)
    with _lock:
        if _thread is not None and _thread.is_alive():
            return {"state": "already running"}
        interval = _env_float(ENV_INTERVAL, DEFAULT_INTERVAL)
        budget = _env_float(ENV_BUDGET, DEFAULT_BUDGET)
        _stop.clear()
        _thread = threading.Thread(target=_loop, args=(interval, budget),
                                   name="icn-transcript-sync", daemon=True)
        _thread.start()
        _last.clear()
        _last.update({"state": "starting", "interval_seconds": interval, "budget_seconds": budget})
        return dict(_last)


def stop(timeout: float = 5.0) -> None:
    global _thread
    _stop.set()
    thread = _thread
    if thread is not None and thread.is_alive():
        thread.join(timeout)
    _thread = None


def status() -> dict[str, Any]:
    thread = _thread
    with _lock:
        last = dict(_last)
    holder = Lease(lease_path(), owner="transcript-sync").holder()
    return {
        "enabled": autosync_enabled(),
        "running": bool(thread and thread.is_alive()),
        "interval_seconds": _env_float(ENV_INTERVAL, DEFAULT_INTERVAL),
        "budget_seconds": _env_float(ENV_BUDGET, DEFAULT_BUDGET),
        "lease_holder": holder,
        "last_tick": last,
    }
