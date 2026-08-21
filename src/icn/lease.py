"""Cross-process indexer lease.

PLAN 2 section 11. Auto-start means N server processes per repository - one per
editor window, one per terminal. Exactly one of them may index at a time, and
the winner may be SIGKILLed at any moment, so the lease has to be reclaimable
without anybody running cleanup.

A heartbeat file, not an OS lock: an OS lock dies with the process that held
it, which sounds convenient until you want to know *who* holds it and for how
long. The heartbeat gives a stale-lease timeout and a debuggable state on disk.
"""

from __future__ import annotations

import json
import os
import socket
import time
import uuid
from pathlib import Path
from typing import Any

STALE_AFTER = 60.0     # seconds without a heartbeat before a lease is reclaimable
HEARTBEAT_EVERY = 10.0


class Lease:
    """Best-effort mutual exclusion. `acquired` is False if someone else holds it.

    Ownership is per Lease instance, not per process. Two holders inside one
    process are normal here - the foreground indexer and the background thread
    that finishes a truncated first index - and identifying by pid alone would
    let them trample each other while both believed they held it.
    """

    def __init__(self, path: Path, owner: str = "indexer"):
        self.path = path
        self.owner = owner
        self.acquired = False
        self.token = uuid.uuid4().hex
        self._last_beat = 0.0

    def _read(self) -> dict[str, Any] | None:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _write(self) -> None:
        payload = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "owner": self.owner,
            "token": self.token,
            "heartbeat": time.time(),
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, self.path)

    def try_acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        current = self._read()
        if current is not None:
            fresh = (time.time() - float(current.get("heartbeat", 0))) < STALE_AFTER
            mine = current.get("token") == self.token
            if fresh and not mine:
                return False
            # Stale lease: the holder died mid-index. Reclaim it.
        try:
            self._write()
        except OSError:
            return False
        self.acquired = True
        self._last_beat = time.time()
        return True

    def beat(self) -> None:
        """Call periodically during long work so the lease is not judged stale."""
        if self.acquired and (time.time() - self._last_beat) > HEARTBEAT_EVERY:
            try:
                self._write()
                self._last_beat = time.time()
            except OSError:
                pass

    def release(self) -> None:
        if not self.acquired:
            return
        current = self._read()
        if current and current.get("token") == self.token:
            try:
                self.path.unlink()
            except OSError:
                pass
        self.acquired = False

    def holder(self) -> dict[str, Any] | None:
        current = self._read()
        if not current:
            return None
        current["age_seconds"] = round(time.time() - float(current.get("heartbeat", 0)), 1)
        current["stale"] = current["age_seconds"] > STALE_AFTER
        return current

    def __enter__(self) -> "Lease":
        self.try_acquire()
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.release()
        return False
