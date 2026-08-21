"""A minimal synchronous MCP stdio client for the live tests.

Deliberately not the SDK's async `stdio_client`: on Windows its asyncio
subprocess transport hangs under pytest, and using it would mean the live test
exercises the client library's event loop as much as the server. Speaking the
wire protocol directly keeps the test about the bytes the server actually
emits, which is the thing worth asserting.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "2025-06-18"


class McpStdioClient:
    """Launches the server the way an MCP client does and speaks JSON-RPC to it."""

    def __init__(self, cwd: Path, env_overrides: dict[str, str] | None = None,
                 command: list[str] | None = None, timeout: float = 90.0):
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
        env.pop("INFINITE_CODE_ROOT", None)
        env.update(env_overrides or {})

        # No arguments, no config file, no port. cwd is the entire configuration.
        self.proc = subprocess.Popen(
            command or [sys.executable, "-m", "icn"],
            cwd=str(cwd), env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        self.timeout = timeout
        self._next_id = 0
        self.stderr: list[str] = []
        # stderr must be drained or a chatty server deadlocks on a full pipe.
        self._drain = threading.Thread(target=self._read_stderr, daemon=True)
        self._drain.start()

    def _read_stderr(self) -> None:
        for line in self.proc.stderr:
            self.stderr.append(line.rstrip())

    def _send(self, message: dict[str, Any]) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()

    def _read_response(self, request_id: int) -> dict[str, Any]:
        """Read until the matching id arrives, skipping notifications."""
        assert self.proc.stdout is not None
        deadline = threading.Event()
        timer = threading.Timer(self.timeout, deadline.set)
        timer.start()
        try:
            while True:
                line = self.proc.stdout.readline()
                if not line:
                    raise RuntimeError(
                        f"server closed stdout; stderr:\n" + "\n".join(self.stderr[-25:])
                    )
                if deadline.is_set():
                    raise TimeoutError(f"no response to request {request_id} within {self.timeout}s")
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AssertionError(
                        f"server wrote non-JSON to stdout, which corrupts the protocol: {line[:200]}"
                    ) from exc
                if message.get("id") == request_id:
                    return message
        finally:
            timer.cancel()

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._next_id += 1
        request_id = self._next_id
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}})
        message = self._read_response(request_id)
        if "error" in message:
            raise RuntimeError(f"{method} failed: {message['error']}")
        return message["result"]

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def initialize(self) -> dict[str, Any]:
        result = self.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "icn-live-test", "version": "1"},
        })
        self.notify("notifications/initialized")
        return result

    def list_tools(self) -> list[dict[str, Any]]:
        return self.request("tools/list").get("tools", [])

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        result = self.request("tools/call", {"name": name, "arguments": arguments or {}})
        structured = result.get("structuredContent")
        if isinstance(structured, dict):
            return structured.get("result", structured)
        content = result.get("content") or []
        assert content, f"{name} returned no content"
        return json.loads(content[0]["text"])

    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.terminate()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()

    def __enter__(self) -> "McpStdioClient":
        self.initialize()
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.close()
        return False
