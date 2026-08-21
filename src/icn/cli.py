"""`icn-explore` - open, export and import a repository's knowledge.

Deliberately separate from the MCP server. The server speaks JSON-RPC on stdin
and must never print to stdout; this is the human-facing side, and mixing the
two would mean one of them writing where the other is parsing.

    icn-explore                     open the explorer for this repository
    icn-explore export -o k.md      write shareable markdown
    icn-explore export -o k.icn     write a bundle (markdown + graph)
    icn-explore import k.icn        bring another codebase's knowledge in
    icn-explore stats               what is stored here
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from . import explorer, export
from . import workspace as ws_mod


def _open(root: str | None, index: bool = True):
    ws = ws_mod.open_workspace(root)
    if index:
        ws_mod.ensure_indexed(ws)
    return ws


def cmd_explore(args: argparse.Namespace) -> int:
    ws = _open(args.root)
    try:
        graph = export.build_graph(ws.store, ws.catalog, ws.repo_id,
                                   include_deleted=args.include_deleted)
    finally:
        ws.close()

    stats = graph["stats"]
    if not stats["nodes"]:
        print("  Nothing to show yet: this repository has no indexed code or memories.")
        print("  Run the MCP server against it, or `icn-explore stats` to check.")
        return 1

    target = Path(args.out) if args.out else Path(tempfile.gettempdir()) / "icn-explorer.html"
    explorer.write_html(graph, target)
    print(f"  {graph['repo_name']}: {stats['nodes']} nodes, {stats['edges']} edges")
    print(f"  Wrote {target}")
    if args.no_serve:
        return 0
    explorer.serve(target, port=args.port, open_browser=not args.no_browser)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    ws = _open(args.root)
    try:
        graph = export.build_graph(ws.store, ws.catalog, ws.repo_id,
                                   include_deleted=args.include_deleted)
    finally:
        ws.close()

    target = Path(args.out)
    suffix = target.suffix.lower()
    if suffix in (".icn", ".zip"):
        export.write_bundle(graph, target)
    elif suffix == ".json":
        target.write_text(json.dumps(graph, indent=1, ensure_ascii=False), encoding="utf-8")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(export.to_markdown(graph), encoding="utf-8")

    memories = graph["stats"]["by_node_kind"].get("memory", 0)
    print(f"  Exported {memories} memories, {graph['stats']['nodes']} nodes -> {target}")
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    paths: list[Path] = []
    for raw in args.sources:
        path = Path(raw)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.md")))
            paths.extend(sorted(path.glob("*.icn")))
        else:
            paths.append(path)

    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"  Not found: {', '.join(str(p) for p in missing)}")
        return 1

    graph = export.read_many(paths)
    if graph is None:
        print("  Could not read a knowledge graph from those files.")
        print("  Expected .md exported by `icn-explore export`, .icn, .zip or .json.")
        return 1

    if args.preview:
        target = Path(tempfile.gettempdir()) / "icn-imported.html"
        explorer.write_html(graph, target)
        print(f"  Preview only, nothing written to this repository.")
        explorer.serve(target, port=args.port, open_browser=not args.no_browser)
        return 0

    ws = _open(args.root)
    try:
        result = export.import_graph(ws.store, ws.catalog, ws.repo_id, graph,
                                     source=str(paths[0]))
    finally:
        ws.close()

    if not result.get("ok"):
        print(f"  {result.get('error')}")
        return 1
    print(f"  Imported {result['imported']} memories from {result['origin']}"
          f" ({result['skipped_duplicates']} already known,"
          f" {result['anchored_to_local_code']} anchored to local code)")
    print(f"  {result['note']}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    ws = _open(args.root)
    try:
        graph = export.build_graph(ws.store, ws.catalog, ws.repo_id, include_deleted=True)
        print(f"  {graph['repo_name']}  ({graph['repo_id']})")
        print(f"  root: {ws.root}")
        print()
        for kind, count in sorted(graph["stats"]["by_node_kind"].items(),
                                  key=lambda kv: -kv[1]):
            print(f"    {count:6}  {kind}")
        print()
        for kind, count in sorted(graph["stats"]["by_edge_kind"].items(),
                                  key=lambda kv: -kv[1])[:12]:
            print(f"    {count:6}  {kind} edges")
    finally:
        ws.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="icn-explore",
        description="Explore, export and import Infinite Code knowledge.")
    parser.add_argument("--root", help="repository path (default: current directory)")
    sub = parser.add_subparsers(dest="command")

    def shared(p, browser=True):
        p.add_argument("--root", help="repository path (default: current directory)")
        if browser:
            p.add_argument("--port", type=int, default=0, help="port (default: auto)")
            p.add_argument("--no-browser", action="store_true", help="do not open a browser")

    explore = sub.add_parser("explore", help="open the interactive graph (default)")
    shared(explore)
    explore.add_argument("-o", "--out", help="also write the HTML here")
    explore.add_argument("--no-serve", action="store_true", help="write the file and exit")
    explore.add_argument("--include-deleted", action="store_true",
                         help="include tombstoned files and symbols")
    explore.set_defaults(func=cmd_explore)

    exp = sub.add_parser("export", help="write knowledge to .md, .icn, .zip or .json")
    shared(exp, browser=False)
    exp.add_argument("-o", "--out", required=True, help="output path; suffix picks the format")
    exp.add_argument("--include-deleted", action="store_true")
    exp.set_defaults(func=cmd_export)

    imp = sub.add_parser("import", help="import knowledge exported from another codebase")
    shared(imp)
    imp.add_argument("sources", nargs="+", help=".md, .icn, .zip, .json, or a directory")
    imp.add_argument("--preview", action="store_true",
                     help="open it in the explorer without importing")
    imp.set_defaults(func=cmd_import)

    stats = sub.add_parser("stats", help="what is stored for this repository")
    shared(stats, browser=False)
    stats.set_defaults(func=cmd_stats)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    # Bare `icn-explore` is the common case; default it to `explore`.
    if not argv or (argv[0].startswith("-") and "--help" not in argv and "-h" not in argv):
        argv = ["explore", *argv]
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:                       # noqa: BLE001 - a CLI must not traceback
        print(f"  {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
