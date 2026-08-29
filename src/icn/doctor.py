"""Human-facing installation diagnostics for MCP clients."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "2025-06-18"
EXPECTED_TOOLS = {"agit", "investigate", "memory", "paper", "record", "workspace"}


def _read_response(proc: subprocess.Popen[str], request_id: int, timeout: float) -> dict[str, Any]:
    expired = threading.Event()
    timer = threading.Timer(timeout, expired.set)
    timer.start()
    try:
        while True:
            if expired.is_set():
                raise TimeoutError(f"MCP handshake exceeded {timeout:g}s")
            line = proc.stdout.readline() if proc.stdout else ""
            if not line:
                detail = proc.stderr.read() if proc.stderr else ""
                raise RuntimeError(f"MCP server exited during handshake: {detail[-1000:]}")
            message = json.loads(line)
            if message.get("id") == request_id:
                return message
    finally:
        timer.cancel()


def _send(proc: subprocess.Popen[str], message: dict[str, Any]) -> None:
    if proc.stdin is None:
        raise RuntimeError("MCP server stdin is unavailable")
    proc.stdin.write(json.dumps(message) + "\n")
    proc.stdin.flush()


def probe(root: Path, timeout: float = 15.0) -> dict[str, Any]:
    """Launch ICN exactly as an MCP client does and verify tools/list."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    env.pop("INFINITE_CODE_ROOT", None)
    proc = subprocess.Popen(
        [sys.executable, "-m", "icn"], cwd=str(root), env=env,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    try:
        _send(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "icn-doctor", "version": "1"},
        }})
        initialized = _read_response(proc, 1, timeout)
        if "error" in initialized:
            raise RuntimeError(f"initialize failed: {initialized['error']}")
        _send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        _send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        listed = _read_response(proc, 2, timeout)
        if "error" in listed:
            raise RuntimeError(f"tools/list failed: {listed['error']}")
        tools = sorted(tool["name"] for tool in listed.get("result", {}).get("tools", []))
        _send(proc, {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
            "name": "workspace", "arguments": {"action": "open", "root": str(root)},
        }})
        opened_message = _read_response(proc, 3, timeout)
        if "error" in opened_message:
            raise RuntimeError(f"workspace(open) failed: {opened_message['error']}")
        tool_result = opened_message.get("result", {})
        structured = tool_result.get("structuredContent") or {}
        opened = structured.get("result", structured)
        if not opened:
            content = tool_result.get("content") or []
            opened = json.loads(content[0]["text"]) if content else {}
        index = opened.get("index") or {}
        parser_warning = index.get("parser_warning")
        return {
            "ok": set(tools) == EXPECTED_TOOLS and not parser_warning,
            "python": sys.executable,
            "python_version": sys.version.split()[0],
            "root": str(root),
            "tools": tools,
            "missing_tools": sorted(EXPECTED_TOOLS - set(tools)),
            "workspace": {
                "repo_id": opened.get("repo_id"),
                "files_active": index.get("files_active", 0),
                "symbols_active": index.get("symbols_active", 0),
                "index_state": index.get("index_state"),
            },
            "parser_warning": parser_warning,
        }
    finally:
        try:
            if proc.stdin:
                proc.stdin.close()
            proc.terminate()
            proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            proc.kill()


def _codex_config() -> dict[str, Any]:
    path = Path.home() / ".codex" / "config.toml"
    if not path.exists():
        return {"path": str(path), "exists": False, "icn_registered": False}
    text = path.read_text(encoding="utf-8", errors="replace")
    registered = "[mcp_servers.icn]" in text
    explicitly_disabled = False
    if registered:
        section = text.split("[mcp_servers.icn]", 1)[1].split("[", 1)[0]
        explicitly_disabled = "enabled = false" in section.lower()
    return {
        "path": str(path),
        "exists": True,
        "icn_registered": registered,
        "explicitly_disabled": explicitly_disabled,
        "legacy_registration_present": "[mcp_servers.infinite-code]" in text,
    }


def build_report(root: Path, client: str, timeout: float) -> dict[str, Any]:
    report = {"client": client, "probe": probe(root, timeout)}
    if client == "codex":
        report["configuration"] = _codex_config()
    config = report.get("configuration", {})
    report["ready"] = bool(
        report["probe"]["ok"]
        and (client != "codex" or (config.get("icn_registered") and not config.get("explicitly_disabled")))
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="icn doctor", description="Verify ICN MCP installation")
    parser.add_argument("--client", choices=("codex", "generic"), default="codex")
    parser.add_argument("--root", default=".", help="repository used for the MCP probe")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = build_report(Path(args.root).expanduser().resolve(), args.client, args.timeout)
    except Exception as exc:  # noqa: BLE001 - doctor must explain, not traceback
        report = {"client": args.client, "ready": False, "error": f"{type(exc).__name__}: {exc}"}
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"ICN MCP doctor ({args.client})")
        if report.get("probe"):
            print(f"  root: {report['probe']['root']}")
            print(f"  python: {report['probe']['python']}")
            print(f"  python version: {report['probe']['python_version']}")
            print(f"  tools: {', '.join(report['probe']['tools']) or 'none'}")
            workspace = report["probe"].get("workspace") or {}
            print(f"  index: {workspace.get('files_active', 0)} files, "
                  f"{workspace.get('symbols_active', 0)} symbols")
            if report["probe"].get("parser_warning"):
                print(f"  parser warning: {report['probe']['parser_warning']}")
        if report.get("configuration"):
            cfg = report["configuration"]
            print(f"  config: {cfg['path']}")
            print(f"  registered: {cfg['icn_registered']}")
            print(f"  explicitly disabled: {cfg['explicitly_disabled']}")
            if cfg.get("legacy_registration_present"):
                print("  note: legacy [mcp_servers.infinite-code] registration is also present")
        if report.get("error"):
            print(f"  error: {report['error']}")
        print("READY: restart the MCP client and call workspace(open)." if report.get("ready")
              else "NOT READY: fix the reported configuration or handshake error.")
    return 0 if report.get("ready") else 1
