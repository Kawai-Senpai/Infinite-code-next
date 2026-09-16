"""`icn rules` - promote settled rules into CLAUDE.md / AGENTS.md, by hand.

    icn rules recommend            ranked candidates, read-only
    icn rules approve <memory_id>  [--text "wording"]
    icn rules edit <memory_id> --text "wording"
    icn rules remove <memory_id>
    icn rules list
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import rules as rules_mod
from . import workspace as ws_mod


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="icn rules", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("recommend", "approve", "edit", "remove", "list"))
    parser.add_argument("memory_id", nargs="?")
    parser.add_argument("--text", default="")
    parser.add_argument("--root", default=None)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    current = ws_mod.open_workspace(args.root)
    try:
        if args.command in ("approve", "edit", "remove") and not args.memory_id:
            parser.error(f"{args.command} needs a memory_id")
        if args.command == "recommend":
            result = rules_mod.recommend(current.store, limit=args.limit)
        elif args.command == "list":
            result = rules_mod.listing(current.store, Path(current.root))
        elif args.command == "approve":
            result = rules_mod.approve(current.store, Path(current.root), args.memory_id, args.text,
                                       actor="human")
        elif args.command == "edit":
            result = rules_mod.edit(current.store, Path(current.root), args.memory_id, args.text)
        else:
            result = rules_mod.remove(current.store, Path(current.root), args.memory_id)
    except rules_mod.RulesError as err:
        result = {"ok": False, "error": str(err)}
    finally:
        current.close()

    if args.json or args.command not in ("recommend", "list"):
        print(json.dumps(result, indent=2))
    elif args.command == "recommend":
        print(f"{result['promoted']}/{result['cap']} rules promoted. Candidates:")
        for c in result["candidates"]:
            print(f"  {c['score']:>5}  {c['memory_id']}  {c['rule']}\n         why: {', '.join(c['why'])}")
        for c in result["almost"]:
            print(f"  almost {c['memory_id']}: {c['rule'][:90]}  (blocked: {c['blocked_by']})")
    else:
        for r in result["rules"]:
            print(f"  {'STALE ' if r['stale'] else ''}{r['memory_id']}  {r['rule']}")
            for problem in r["problems"]:
                print(f"         {problem}")
    return 0 if result.get("ok") else 1
