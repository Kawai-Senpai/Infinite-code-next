"""MCP surface: seven tools.

PLAN 2 section 10. Agents waste turns choosing between near-identical tools, so
the surface is deliberately small and grouped by action:

    workspace    open, status, list, reindex, health, reconcile, archive,
                 detach, forget_checkout, purge
    investigate  search (default), expand, verify
    graph        impact, trace, cycles, entrypoints, areas, triggers,
                 coupling, hotspots, deadcode
    record       write one event, compiled into many facts
    memory       get, list, correct, supersede, verify, reanchor, resolve
    agit         status, diff, commit, log, branches, switch, restore, reset, show
    paper        search, fetch, read, grep, render, figures, download, list,
                 forget, remember

Every response carries the resolved root, so a wrong workspace is visible at
once instead of quietly poisoning the store.

paper() is annotated `-> Any` rather than `-> dict[str, Any]` like its five
siblings, and that difference is load-bearing: FastMCP builds an output model
from the return annotation and validates against it, so a dict annotation
rejects the mixed [summary, Image, Image] list that action='render' returns.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.types import Image

from . import agit as agit_mod
from . import anchors as anchor_mod
from . import briefing as briefing_mod
from . import catalog as catalog_mod
from . import causal
from . import compiler
from . import embed as embed_mod
from . import graph as graph_mod
from . import history as history_mod
from . import papers as papers_mod
from . import search as search_mod
from . import workspace as ws_mod

INSTRUCTIONS = """Persistent code knowledge for this workspace.

Start with workspace(action='open'). It returns a briefing: the rules that
govern this code, what has already been tried and rejected, and what is
currently unverified. Read it before opening files - it exists so you do not
have to rediscover by reading.

Then investigate() with what you are about to do, in plain language. One call
returns code structure, the rationale behind it, prior failures, invariants,
tests and blast radius, as compact capsules rather than file dumps. Prefer it
over grepping: it searches code and knowledge together.

Before changing or deleting something load-bearing, investigate(action='why',
symbol=...) reconstructs why it exists - the decision, the bug that followed,
the fix that was rejected, the invariant that resulted, and what removing it
may reintroduce.

graph() answers the structural question investigate() does not: what else moves
if you touch this. action='impact' is the blast radius, 'trace' proves two
symbols are actually connected, 'cycles' finds knots in the import graph. Read
its `epistemic` field before its result: 'lower-bound' means callers exist that
the answer provably does not list, and `causes` counts them. An absent caller
is never by itself evidence of no caller.

Finish with record(). One call becomes many anchored facts. The highest-value
fields are failed_attempts and warnings: nothing else in your toolchain
captures what was tried and rejected, and that is what future agents
find most expensive to rediscover. caused_by=[memory_id] links your work into the causal
chain.

Memories carry an anchor_status. Anything other than ACTIVE has not been
verified against the current code: treat it as a lead, not a fact, and call
memory(action='verify') once you have confirmed it still applies.

paper() is prior art on the same footing. Before building a non-trivial
mechanism, search arXiv, then actually read the paper rather than its abstract:
fetch() caches the full text, read() serves it by section or page range, grep()
answers one question without loading the rest, and render() returns page images
for the figures and tables the text layer drops. paper(action='remember') writes
what a paper settled into this repository's memory graph, so the next agent
asking why the code is shaped this way finds the citation instead of guessing.
"""

mcp = FastMCP("infinite-code-next", instructions=INSTRUCTIONS)


def _fail(error: str, root: str | None = None) -> dict[str, Any]:
    return {"ok": False, "error": error, "resolved_root": root}


# What a paper can establish, mapped to the payload field record_event compiles
# memories from. Memories come from these typed fields; an event carrying only
# a summary produces nothing.
_REMEMBER_FIELDS = {
    "rationale": "rationale_notes",
    "note": "rationale_notes",
    "investigation": "rationale_notes",
    "decision": "decisions",
    "warning": "warnings",
    "invariant": "invariants",
    "convention": "conventions",
    "performance": "performance",
    "security": "security",
    "failed_attempt": "failed_attempts",
}


def _text_is_empty(result: dict[str, Any]) -> bool:
    """Did this read actually return anything to read?

    The page markers read() emits are not content: "[page 1]\\n\\n[page 2]\\n"
    is what a three-page scan looks like, and it is not an empty string.
    """
    if not isinstance(result, dict) or "text" not in result:
        return False
    stripped = re.sub(r"\[page \d+\]", "", result.get("text") or "")
    return not stripped.strip()


def _read_as_images(result: dict[str, Any], paper_id: str, url: str, path: str,
                    pages: str, dpi: int, save_to: str) -> list[Any]:
    """Answer a text read that came back empty with the pages themselves.

    Only the pages the caller actually asked for, capped by the per-call render
    limit: a 40-page scan cannot come back as 40 images, so the summary says
    which pages these are and how to ask for the rest.
    """
    wanted = pages or result.get("selected_pages") or ""
    if isinstance(wanted, list):
        wanted = ",".join(str(n) for n in wanted)
    if not wanted:
        total = int(result.get("pages") or 1)
        wanted = f"1-{min(total, papers_mod.MAX_RENDER_PAGES)}"

    try:
        rendered = papers_mod.render(paper_id=paper_id, url=url, path=path,
                                     pages=wanted, dpi=dpi, save_to=save_to)
    except papers_mod.PaperError as err:
        return [{**result, "vision_error": str(err)}]

    shown = [item["page"] for item in rendered["rendered"]]
    images = [Image(data=item.pop("png"), format="png") for item in rendered["rendered"]]
    summary = {
        **{k: v for k, v in result.items() if k != "text"},
        "mode": "rendered",
        "served_as": "images",
        "why": ("this paper has no extractable text layer, so the pages are "
                "returned as images instead of as an empty string"),
        "rendered_pages": shown,
        "dpi": rendered.get("dpi", dpi),
    }
    remaining = int(result.get("pages") or 0) - max(shown or [0])
    if remaining > 0:
        summary["more"] = (
            f"{remaining} further pages exist; ask again with "
            f"pages='{max(shown) + 1}-{min(int(result['pages']), max(shown) + papers_mod.MAX_RENDER_PAGES)}'"
        )
    return [summary, *images]


@mcp.tool()
def workspace(
    action: str = "open",
    root: str | None = None,
    force_full: bool = False,
    repo_id: str | None = None,
    reason: str = "",
    confirm: bool = False,
) -> dict[str, Any]:
    """Open, inspect, and manage repositories known to this server.

    WHEN TO USE: first call of any session, and again whenever you switch
    repository. action='open' returns the briefing - the rules that govern this
    code, what has already been tried and rejected, and what is unverified.
    AFTER THIS: investigate() with what you are about to do. Do not start by
    reading files; the briefing exists so you do not have to.


    Actions:
      open              resolve this directory to a repository, index what
                        changed, re-verify anchors. Call this first.
      status            current identity, index and anchor health.
      list              every repository known, including ones not on disk.
      reindex           full re-index (force_full is implied).
      health            graph health summary across all repositories.
      reconcile         re-check which checkouts are still reachable.
      archive           stop indexing, keep all knowledge searchable.
      detach            stop watching, keep repository knowledge.
      forget_checkout   this local clone is gone; repository knowledge stays.
      purge             permanently delete stored data. Requires confirm=True.

    Args:
        action: one of the actions above.
        root: repository path. Defaults to the server's working directory.
        force_full: re-index everything rather than only what changed.
        repo_id: target repository for archive/detach/purge.
        reason: recorded in the lifecycle log for state changes.
        confirm: required for purge, which is the only destructive action.
    """
    act = (action or "open").lower().strip()

    if act in ("list", "health", "reconcile"):
        catalog = catalog_mod.open_catalog()
        try:
            if act == "list":
                return {"ok": True, "repositories": catalog_mod.list_repositories(catalog)}
            if act == "health":
                return {"ok": True, "catalog": catalog_mod.health(catalog)}
            return {"ok": True, **catalog_mod.reconcile(catalog)}
        finally:
            catalog.close()

    if act in ("archive", "detach", "purge"):
        catalog = catalog_mod.open_catalog()
        try:
            if not repo_id:
                current = ws_mod.open_workspace(root)
                repo_id = current.repo_id
                current.close()
            if act == "purge":
                if not confirm:
                    return _fail("purge permanently deletes this repository's stored knowledge; "
                                 "call again with confirm=True")
                return catalog_mod.purge(catalog, repo_id)
            target = catalog_mod.ARCHIVED if act == "archive" else catalog_mod.DETACHED
            return catalog_mod.set_repo_status(catalog, repo_id, target, reason or act)
        finally:
            catalog.close()

    if act == "forget_checkout":
        catalog = catalog_mod.open_catalog()
        try:
            return catalog_mod.forget_checkout(catalog, root or str(ws_mod.resolve_root(None)))
        finally:
            catalog.close()

    current = ws_mod.open_workspace(root)
    try:
        if act == "status":
            return {"ok": True, **ws_mod.status(current)}
        if act in ("open", "reindex"):
            report = ws_mod.ensure_indexed(current, force_full=force_full or act == "reindex")
            state = ws_mod.status(current)
            urgent = compiler.list_memories(current.store, anchor_status=anchor_mod.NEEDS_REVIEW,
                                            status="ACTIVE", limit=5)
            # What already exists here. An agent cannot ask the right question
            # before it knows what is on the shelf, so open() volunteers it.
            brief = briefing_mod.build(current.store)
            return {
                "ok": True,
                "resolved_root": str(current.root),
                "repo_id": current.repo_id,
                # "registered" read as "is this repository known", and was
                # False on every open after the first - the opposite of the
                # truth for a repository with knowledge already stored.
                "first_seen": bool(current.info.get("created")),
                "reattached": bool(current.info.get("reattached")),
                "identity": state["identity"],
                "head": current.commit,
                "index": report,
                "anchors": state["anchors"],
                "needs_review": urgent,
                "briefing": brief,
                "next": "investigate('what you are about to change')",
            }
        return _fail(f"unknown workspace action: {action}", str(current.root))
    finally:
        current.close()


@mcp.tool()
def investigate(
    query: str = "",
    intent: str | None = None,
    root: str | None = None,
    roots: list[str] | None = None,
    depth: int = 2,
    budget: int = 9000,
    find_problems: bool = True,
    cross_repos: bool = False,
    action: str = "search",
    investigation_id: str | None = None,
    focus: str = "",
    symbol: str = "",
) -> dict[str, Any]:
    """Get up to speed on code in one call: structure, rationale, risks.

    WHEN TO USE: before investigating, diagnosing, designing or modifying
    anything. Prefer it over grep: it searches code and knowledge together, so
    it returns the reason a thing is shaped the way it is, not just where it is.
    AFTER THIS: graph(action='impact') if you are about to change a symbol;
    investigate(action='why', symbol=...) before deleting something
    load-bearing; record() once you have verified something.


    Searches code and knowledge together - lexical, symbol, code graph, memory
    graph, anchor status and git history - and returns compact capsules under a
    token budget rather than file dumps.

    Actions:
      search  (default) run an investigation.
      expand  drill into a prior investigation without restarting it.
      verify  re-verify anchors for this repository right now.
      why     reconstruct why one symbol exists, as a causal chain:
              decision -> bug -> failed fix -> accepted fix -> invariant -> test.

    Args:
        query: what you are trying to do, in plain language.
        intent: locate | understand | modify | debug | audit. Inferred if omitted.
        root: primary repository path. Defaults to the server's working directory.
        roots: explicit repository paths for one bounded multi-repository search.
        depth: graph expansion hops from the seed set.
        budget: approximate token ceiling for the returned capsules.
        find_problems: run targeted checks over the narrowed subgraph.
        cross_repos: also follow contract edges into other repositories and
            report their active warnings. Reads the central catalog only, so
            it works even when those repositories are missing.
        action: search, expand, or verify.
        investigation_id: required for expand.
        focus: what to drill into, for expand.
        symbol: the symbol to explain, for action="why".
    """
    requested_roots = [value for value in (roots or []) if str(value).strip()]
    if requested_roots and (action or "search").lower().strip() != "search":
        return _fail("roots is supported only for investigate action='search'", root)
    if requested_roots:
        ordered = [root, *requested_roots] if root else requested_roots
        unique: list[str] = []
        seen: set[str] = set()
        for value in ordered:
            resolved = str(Path(value).expanduser().resolve())
            if resolved.lower() not in seen:
                seen.add(resolved.lower())
                unique.append(resolved)
        per_repo_budget = max(500, budget // max(1, len(unique)))
        repository_results: list[dict[str, Any]] = []
        for repository_root in unique:
            current = ws_mod.open_workspace(repository_root)
            try:
                index_report = ws_mod.ensure_indexed(current)
                result = search_mod.investigate(
                    current.store, current.catalog, current.root, query,
                    intent=intent, depth=depth, budget=per_repo_budget,
                    find_problems=find_problems, commit=current.commit,
                    cross_repos=cross_repos, repo_id=current.repo_id,
                )
                repository_results.append({
                    "root": str(current.root), "repo_id": current.repo_id,
                    "index_state": index_report.get("index_state", "ready"), **result,
                })
            finally:
                current.close()
        capsules = [
            {"repository_root": item["root"], "repository_id": item["repo_id"], **capsule}
            for item in repository_results for capsule in item.get("capsules", [])
        ]
        capsules.sort(key=lambda value: -float(value.get("score", 0)))
        problems = [
            {"repository_root": item["root"], **problem}
            for item in repository_results for problem in item.get("problems", [])
        ]
        return {
            "ok": True,
            "multi_root": True,
            "resolved_roots": unique,
            "query": query,
            "repositories": repository_results,
            "capsules": capsules[:12],
            "problems": problems,
            "budget": {"limit": budget, "per_repository": per_repo_budget},
        }

    current = ws_mod.open_workspace(root)
    try:
        act = (action or "search").lower().strip()

        if act == "verify":
            # Index first. Verifying against a stale index reports every anchor
            # as ACTIVE no matter what changed on disk, which is precisely the
            # silent-staleness failure this whole subsystem exists to prevent.
            index_report = ws_mod.ensure_indexed(current)
            return {"ok": True, "resolved_root": str(current.root),
                    "index_state": index_report.get("index_state", "ready"),
                    **index_report.get("anchors", {})}

        if act == "why":
            if not symbol.strip():
                return _fail("why requires `symbol`", str(current.root))
            # `open_workspace` opens the store but deliberately does not index
            # it. `why` must still resolve methods when it is the first call
            # after an agent switches repositories.
            ws_mod.ensure_indexed(current)
            from . import compiler as _compiler
            match = _compiler.resolve_reference(current.store, symbol)
            if match is None or match["kind"] != "symbol":
                return _fail("could not resolve symbol " + repr(symbol), str(current.root))
            story = causal.why_does_this_exist(current.store, match["row"]["symbol_id"])
            if story is None:
                return {"ok": True, "resolved_root": str(current.root),
                        "symbol": match["row"]["symbol_path"], "why_it_exists": None,
                        "note": "no causal history recorded for this symbol; use"
                                " investigate() for the memories attached to it"}
            return {"ok": True, "resolved_root": str(current.root),
                    "symbol": match["row"]["symbol_path"], "why_it_exists": story,
                    "rendered": causal.render(story)}

        if act == "expand":
            if not investigation_id:
                return _fail("expand requires investigation_id", str(current.root))
            return {"resolved_root": str(current.root),
                    **search_mod.expand(current.store, current.catalog, investigation_id, focus, budget)}

        if not query.strip():
            return _fail("investigate requires a query", str(current.root))

        # Index and re-verify before searching, so results reflect the tree as
        # it is right now, including uncommitted edits.
        index_report = ws_mod.ensure_indexed(current)
        result = search_mod.investigate(
            current.store, current.catalog, current.root, query,
            intent=intent, depth=depth, budget=budget,
            find_problems=find_problems, commit=current.commit,
            cross_repos=cross_repos, repo_id=current.repo_id,
        )
        result["ok"] = True
        result["resolved_root"] = str(current.root)
        result["index_state"] = index_report.get("index_state", "ready")
        known = catalog_mod.list_repositories(current.catalog)
        result["repository_scope"] = {
            "selected": {"repo_id": current.repo_id, "root": str(current.root)},
            "alternatives": [
                {"repo_id": repo["repo_id"], "name": repo["name"],
                 "checkouts": [checkout["path"] for checkout in repo["checkouts"]]}
                for repo in known if repo["repo_id"] != current.repo_id
            ],
            "note": "Results come only from the selected repository unless roots=[...] is passed.",
        }
        result["validation_boundary"] = (
            "Static indexing and stored memories are leads, not runtime proof. Verify semantic "
            "claims with direct source inspection plus focused tests or live execution."
        )
        scope = ws_mod.named_sibling_roots(current.root, query)
        if scope:
            result["scope_warning"] = (
                "The query names sibling repositories outside this investigation scope. "
                "Pass roots=[...] to search them explicitly."
            )
            result["suggested_roots"] = [str(path) for path in scope]
        return result
    finally:
        current.close()


@mcp.tool()
def graph(
    action: str = "impact",
    target: str = "",
    to: str = "",
    direction: str = "upstream",
    depth: int = 3,
    file_hint: str | None = None,
    to_file_hint: str | None = None,
    include_tests: bool = True,
    min_confidence: float = 0.0,
    window: int = 500,
    since: str = "",
    root: str | None = None,
) -> dict[str, Any]:
    """Structural questions about the code graph: blast radius, paths, cycles.

    WHEN TO USE: after investigate() has told you what a symbol is for, and you
    need to know what changing it costs. investigate() answers "what is this and
    why"; graph() answers "what else moves if I touch it".
    AFTER THIS: investigate(action='why', symbol=...) on anything surprising in
    the blast radius, then record() what you decided.

    Actions:
      impact  blast radius of one symbol. direction='upstream' is who depends on
              it (who breaks if the contract changes); 'downstream' is what it
              depends on.
      trace   shortest directed call path from `target` to `to`. Use it to prove
              two symbols are actually connected rather than assuming it.
      cycles  directed cycles in the file import graph. Act on component_count,
              not on the number of cycles: one removed import can dissolve a
              whole component, so the cycle count swings wildly and the
              component count is what a fix actually reduces. A deferred import
              (inside a function, or a TypeScript `import type`) is excluded: it
              cannot force an initialisation order, and it is the usual fix for
              a cycle rather than a cause of one.
      entrypoints
              where control enters the program - routes, MCP tool handlers, CLI
              commands, tests. Pass a kind in `target` to filter. Start here on
              an unfamiliar codebase.
      areas   functional areas from the call graph, not the directory layout, so
              an area can span directories. Each carries test_share and the
              entry points that reach it; an area nothing reaches is a library
              or is dead.
      triggers
              which entry points can reach `target` - what a user can actually
              do that runs this code. The complement of impact: not "what else
              changes" but "what can set this off".
      coupling
              file pairs that keep changing together, from git history. The
              rows worth reading are the ones with also_imports=false: those
              are coupled by something no static analysis can see.
      hotspots
              where change and structural weight meet. Neither alone is
              interesting; a file both heavily depended on and constantly
              rewritten is where a change is most likely to break something
              far away.
      deadcode
              symbols nothing in the graph reaches. CANDIDATES, never a
              verdict - read `confidence` on every row and the boundaries
              before acting on one.

    READ THE ENVELOPE BEFORE THE RESULT. Every answer carries:
      epistemic   'exact' or 'lower-bound'. 'lower-bound' means callers exist
                  that this result provably does not list. An empty `affected`
                  with epistemic='lower-bound' is NOT proof that nothing calls
                  the symbol.
      boundaries  one plain sentence per reason, for humans.
      causes      the machine-readable why. Every field counts MISSING or
                  UNPROVEN things, never sentences:
                    ambiguous_call_sites - call sites naming this symbol that
                        could not be attributed to a single definition. These
                        are the callers you are not being shown.
                    inferred_edges_traversed - edges resolved by import scope or
                        by uniqueness of the name, not proven from the syntax.
                        Counts edges walked, so it can exceed counts.inferred,
                        which counts distinct symbols reached.
                    external_call_sites - calls that left the indexed program.
                        Not a defect: no in-graph node could have been reached.

    Edges are tiered, and the tier is on every result. receiver_self (1.0) and
    same_file (0.9) are read off the syntax; imported (0.8) and unique_global
    (0.6) are judgements. Pass min_confidence=0.9 to walk only what was proven.

    The three history actions carry their own envelope. It is always
    'lower-bound': history only knows the commits it read, only knows committed
    work, and knows nothing that a squash or a shallow clone removed. `history`
    in the result says how much was actually read.

    Args:
        action: impact (default), trace, cycles, entrypoints, areas, triggers,
            coupling, hotspots, deadcode.
        target: symbol to analyse, the path source for trace, or - for
            action='entrypoints' - a kind to filter by (route, tool, cli, test).
            A bare name is fine; if it is ambiguous the candidates come back
            for you to choose.
        to: destination symbol, for trace.
        direction: upstream or downstream, for impact.
        depth: hops to walk. Clamped to 12.
        file_hint: path fragment disambiguating `target`, e.g. 'search.py'.
        to_file_hint: the same, for `to`.
        include_tests: include test files in the blast radius. For
            action='areas', rank mostly-test areas alongside the rest instead of
            sorting them last. For action='deadcode', report unreferenced test
            symbols too.
        min_confidence: drop edges below this confidence before walking.
        window: commits to read, for the history actions. Default 500, max 5000.
        since: a git date ('3 months ago', '2026-01-01') bounding that window.
        root: repository path. Defaults to the server's working directory.
    """
    action = (action or "impact").lower().strip()
    known = ("impact", "trace", "cycles", "entrypoints", "areas", "triggers",
             "coupling", "hotspots", "deadcode")
    if action not in known:
        return _fail(f"unknown action {action!r}; use one of {', '.join(known)}", root)
    if action in ("impact", "trace", "triggers") and not target.strip():
        return _fail(f"target is required for action={action!r}", root)
    if action == "trace" and not to.strip():
        return _fail("to is required for action='trace'", root)

    current = ws_mod.open_workspace(root)
    try:
        index_report = ws_mod.ensure_indexed(current)
        if action == "impact":
            result = graph_mod.impact(
                current.store, target, direction=direction, depth=depth,
                file_hint=file_hint, include_tests=include_tests,
                min_confidence=min_confidence)
        elif action == "trace":
            result = graph_mod.trace(
                current.store, target, to, max_depth=depth,
                file_hint=file_hint, target_file_hint=to_file_hint)
        elif action == "entrypoints":
            result = graph_mod.entry_points(current.store, kind=target.strip())
        elif action == "areas":
            result = graph_mod.areas(current.store, include_tests=include_tests)
        elif action == "triggers":
            result = graph_mod.reaching_entry_points(
                current.store, target, file_hint=file_hint, depth=depth)
        elif action == "coupling":
            result = history_mod.change_coupling(
                current.store, current.root, window=window, since=since or None)
        elif action == "hotspots":
            result = history_mod.hotspots(
                current.store, current.root, window=window, since=since or None)
        elif action == "deadcode":
            result = history_mod.dead_code(
                current.store, include_tests=include_tests)
        else:
            result = graph_mod.import_cycles(current.store)

        result["ok"] = "error" not in result
        result["action"] = action
        result["resolved_root"] = str(current.root)
        result["index_state"] = index_report.get("index_state", "ready")
        result["validation_boundary"] = (
            "Static resolution only. An edge is evidence, not proof that the call "
            "happens at runtime; dynamic dispatch, reflection and framework wiring "
            "are invisible here. Confirm with a focused test before relying on it."
        )
        return result
    finally:
        current.close()


@mcp.tool()
def record(
    summary: str,
    kind: str = "note",
    reasoning: str = "",
    files: list[str] | None = None,
    symbols: list[str] | None = None,
    changes: list[str] | None = None,
    invariants: list[str] | None = None,
    warnings: list[str] | None = None,
    failed_attempts: list[str] | None = None,
    decisions: list[str] | None = None,
    contracts: list[str] | None = None,
    performance: list[str] | None = None,
    security: list[str] | None = None,
    conventions: list[str] | None = None,
    rationale: list[str] | None = None,
    bugs: list[str] | None = None,
    migrations: list[str] | None = None,
    tests: list[str] | None = None,
    contracts_with: list[dict] | None = None,
    caused_by: list[str | dict] | None = None,
    authority: str = "agent",
    root: str | None = None,
) -> dict[str, Any]:
    """Write what you learned. One call becomes many durable, anchored facts.

    WHEN TO USE: after any verified finding or change - not at the end of the
    session, when the detail has already gone. Skip it only for purely
    mechanical work such as fixing a typo.
    AFTER THIS: nothing. This is the call that makes the next session cheaper,
    and the fields that pay off most are failed_attempts and warnings, because
    nothing else in the toolchain records what was tried and rejected.


    You supply the semantics; the server resolves names to real symbols,
    anchors each memory to the code, derives the edges you did not mention
    (callers, tests, blast radius), and flags contradictions with what it
    already knows. Nothing you write is ever silently overwritten.

    Use kind='checkpoint' at the end of a task to distil the session.

    WRITE LIKE THE NEXT AGENT HAS NO CONTEXT, because it does not. Every entry
    should survive being read alone, months later, by someone who was not here.

        Thin, nearly useless:  "settle must be idempotent"
        Actually useful:       "settle() must be idempotent: a retried webhook
                                must not double-charge. We learned this from
                                the Feb duplicate-charge incident, where the
                                payment provider retried after a 502 that had
                                already succeeded. The guard is the
                                idempotency_key column, unique per invoice."

    The second one costs you a few extra seconds now and saves the next agent
    an afternoon. State the failure it prevents, the evidence it rests on, and
    the mechanism that enforces it - not just the rule.

    Args:
        summary: what happened, in one full sentence. Required.
        kind: bug_fix | decision | refactor | investigation | checkpoint | incident | note.
        reasoning: why this happened and why it was done this way. Two or three
            sentences. This is attached to every memory the call produces, so
            it is the cheapest way to make all of them self-contained.
        files: file paths this touches.
        symbols: symbol names this touches, e.g. RefreshCoordinator.acquire.
        changes: what actually changed.
        invariants: things that must remain true, and what breaks if they are
            not. "X must hold, because otherwise Y" beats "X must hold".
        warnings: things a future agent must not do, and the symptom it causes.
            A warning nobody can recognise in the wild will not be heeded.
        failed_attempts: what was tried, why it was rejected, and how the
            failure showed itself. This is the highest-value field in the
            system - nothing else in your toolchain records it, and it is what
            stops the next agent spending a day on a dead end you already
            walked down.
        decisions: choices made and the alternatives rejected.
        contracts: assumptions other code or repos rely on.
        performance: performance-relevant facts.
        security: security-relevant facts.
        conventions: local conventions worth following.
        rationale: why the code is shaped the way it is.
        bugs: bugs this code has caused before.
        migrations: migration steps or ordering constraints.
        tests: tests that cover this.
        contracts_with: cross-repository dependencies, as
            [{"repo": "acme/billing", "entity": "charge_customer",
              "kind": "CONSUMES_CONTRACT"}]. kind is one of
            PROVIDES_CONTRACT, CONSUMES_CONTRACT, MIRRORS, RELATED_TO.
            The other repository does not need to be indexed, or even
            present, for this to be recorded.
        caused_by: memory ids this event follows from, building the causal
            chain an agent sees as "why does this code exist":
            decision -> bug -> failed fix -> accepted fix -> invariant -> test.
            Pass ids, or [{"memory": "mem_x", "kind": "CAUSED"}] where kind is
            CAUSED, LED_TO, ESTABLISHED, REFINES or DEPENDS_ON.
        authority: 'agent' or 'human'. Human memories cannot be rewritten by an agent.
        root: repository path. Defaults to the server's working directory.
    """
    if not summary.strip():
        return _fail("record requires a summary")

    current = ws_mod.open_workspace(root)
    try:
        ws_mod.ensure_indexed(current)
        payload = {
            "kind": kind, "summary": summary, "reasoning": reasoning,
            "files": files or [], "symbols": symbols or [], "changes": changes or [],
            "invariants": invariants or [], "warnings": warnings or [],
            "failed_attempts": failed_attempts or [], "decisions": decisions or [],
            "contracts": contracts or [], "performance": performance or [],
            "security": security or [], "conventions": conventions or [],
            "rationale_notes": rationale or [], "bugs": bugs or [],
            "migrations": migrations or [],
            "tests": tests or [], "contracts_with": contracts_with or [],
            "caused_by": caused_by or [], "authority": authority,
        }
        result = compiler.record_event(current.store, current.catalog, current.repo_id,
                                       current.root, current.commit, payload)
        result["resolved_root"] = str(current.root)
        if result.get("unresolved_references"):
            result["hint"] = ("some names did not resolve to indexed code; they were kept as "
                              "repo-scoped knowledge. Check spelling or run "
                              "workspace(action='reindex').")
        return result
    finally:
        current.close()


@mcp.tool()
def memory(
    action: str = "list",
    memory_id: str | None = None,
    body: str = "",
    kind: str | None = None,
    severity: str | None = None,
    status: str | None = None,
    anchor_status: str | None = None,
    reason: str = "",
    actor: str = "agent",
    limit: int = 30,
    offset: int = 0,
    root: str | None = None,
) -> dict[str, Any]:
    """Inspect and correct stored knowledge.

    WHEN TO USE: when a memory surfaced with anchor_status other than ACTIVE and
    you have just confirmed whether it still holds, or when something stored is
    wrong. A memory that is merely out of date should be corrected or
    superseded, never left to rot - an unverified fact costs the next agent
    more than no fact at all.
    AFTER THIS: continue the task. Use action='verify' the moment you have the
    evidence, while you still have it.


    Actions:
      list        browse memories, filterable by kind, status, anchor_status.
      get         one memory with its anchors, edges and version history.
      verify      confirm a memory still applies. Clears NEEDS_REVIEW/DRIFTED.
      correct     edit a memory in place. Previous text is versioned, not lost.
      supersede   replace a memory, keeping both and the link between them.
      resolve     mark a warning or bug as no longer live.
      guard       record that an existing test covers an existing rule, when
                  the two were written in separate calls. Pass the rule as
                  memory_id and the test_evidence memory id as body.
      reanchor    re-run the anchoring cascade for one memory.

    Args:
        action: one of the actions above.
        memory_id: target memory.
        body: new text for correct and supersede.
        kind: new kind for correct, or a filter for list.
        severity: new severity for correct.
        status: filter for list (ACTIVE, SUPERSEDED, RESOLVED).
        anchor_status: filter for list (ACTIVE, NEEDS_REVIEW, DRIFTED, ORPHANED).
        reason: why the change is being made. Recorded.
        actor: 'agent' or 'human'.
        limit: maximum rows for list.
        offset: rows to skip for list. With `total` and `next_offset` in the
            reply, this pages through a filter larger than one call can carry.
        root: repository path. Defaults to the server's working directory.
    """
    current = ws_mod.open_workspace(root)
    try:
        act = (action or "list").lower().strip()

        if act == "list":
            found = compiler.list_memories(current.store, kind=kind, status=status,
                                           anchor_status=anchor_status, limit=limit,
                                           offset=offset)
            total = compiler.count_memories(current.store, kind=kind, status=status,
                                            anchor_status=anchor_status)
            page: dict[str, Any] = {"ok": True, "resolved_root": str(current.root),
                                    "memories": found, "total": total,
                                    "offset": max(0, offset), "count": len(found)}
            # Only when there is more: an absent next_offset is the end of the
            # listing, so a reader never has to compare numbers to know it.
            if max(0, offset) + len(found) < total:
                page["next_offset"] = max(0, offset) + len(found)
            return page
        if not memory_id:
            return _fail(f"{act} requires memory_id", str(current.root))

        if act == "get":
            found = compiler.get_memory(current.store, memory_id)
            return ({"ok": True, "resolved_root": str(current.root), "memory": found}
                    if found else _fail(f"unknown memory {memory_id}", str(current.root)))
        if act == "verify":
            return {"resolved_root": str(current.root),
                    **anchor_mod.mark_verified(current.store, memory_id, current.commit, actor)}
        if act == "correct":
            return {"resolved_root": str(current.root),
                    **compiler.correct(current.store, current.catalog, current.repo_id, memory_id,
                                       body or None, kind, severity, reason, actor)}
        if act == "supersede":
            if not body.strip():
                return _fail("supersede requires body", str(current.root))
            return {"resolved_root": str(current.root),
                    **compiler.supersede(current.store, current.catalog, current.repo_id,
                                         memory_id, body, reason, actor, current.commit)}
        if act == "resolve":
            return {"resolved_root": str(current.root),
                    **compiler.resolve_memory(current.store, current.catalog, current.repo_id,
                                              memory_id, reason, actor)}
        if act == "guard":
            if not memory_id or not body.strip():
                return _fail("guard requires `memory_id` and `body` set to the"
                             " test_evidence memory id", str(current.root))
            return {"resolved_root": str(current.root),
                    **compiler.guard_memory(current.store, memory_id, body.strip())}

        if act == "reanchor":
            ws_mod.ensure_indexed(current)
            return {"ok": True, "resolved_root": str(current.root),
                    **anchor_mod.verify_repo(current.store, current.root, current.commit,
                                             only_memory=memory_id)}
        return _fail(f"unknown memory action: {action}", str(current.root))
    finally:
        current.close()


@mcp.tool()
def agit(
    action: str = "status",
    message: str = "",
    paths: list[str] | None = None,
    branch: str | None = None,
    create: bool = False,
    target: str | None = None,
    source: str = "HEAD",
    cached: bool = False,
    hard: bool = False,
    allow_empty: bool = False,
    limit: int = 20,
    root: str | None = None,
) -> dict[str, Any]:
    """Agent-only git history in `.agit/`, separate from the user's real repo.

    WHEN TO USE: action='commit' before any risky edit or broad refactor, so
    there is something to roll back to. It writes to `.agit/`, never the user's
    `.git`, so it cannot touch their history, staging or branches.
    AFTER THIS: make the risky change. If it goes wrong,
    agit(action='restore', paths=[...]) - which does rewrite real working-tree
    files, so check agit(action='diff') first.


    Checkpoint risky work without touching `.git`. Auto-initialises on first
    use and adds itself to .gitignore. Per working tree, never central: a
    checkpoint only means anything against the tree it snapshotted.

    Actions: status, diff, commit, log, branches, switch, restore, reset, show.

    Args:
        action: one of the actions above.
        message: commit message.
        paths: file paths, relative to the repository root.
        branch: branch name for switch.
        create: create the branch if switching to a new one.
        target: commit-ish for diff, log, reset, show.
        source: source ref for restore.
        cached: diff the staged index.
        hard: use --hard for reset.
        allow_empty: allow an empty commit.
        limit: maximum log entries.
        root: repository path. Defaults to the server's working directory.
    """
    current = ws_mod.open_workspace(root)
    try:
        result = agit_mod.run(
            current.root, action, paths_arg=paths, message=message or None, branch=branch,
            create=create, target=target, source=source, cached=cached, hard=hard,
            allow_empty=allow_empty, limit=limit,
        )
        result["resolved_root"] = str(current.root)

        # A checkpoint is history too. Index it so investigate() can surface
        # "the agent checkpointed here right before this change".
        if action.lower().strip() == "commit" and result.get("commit"):
            from .db import jdump, write_tx
            from .ids import new_id
            with write_tx(current.store):
                current.store.execute(
                    "INSERT OR IGNORE INTO agit_checkpoints (checkpoint_id, agit_commit, message,"
                    " files, repo_commit, created_at) VALUES (?,?,?,?,?,?)",
                    (new_id("cp"), result["commit"], result.get("message"), jdump(paths or []),
                     current.commit, anchor_mod.now()),
                )
        return result
    finally:
        current.close()


@mcp.tool()
def paper(
    action: str = "search",
    query: str = "",
    paper_id: str = "",
    url: str = "",
    path: str = "",
    category: str = "",
    max_results: int = 10,
    sort: str = "relevance",
    start: int = 0,
    mode: str = "outline",
    pages: str = "",
    section: str = "",
    pattern: str = "",
    ignore_case: bool = True,
    context: int = 320,
    max_chars: int = 24000,
    offset: int = 0,
    dpi: int = 140,
    dest: str = "",
    filename: str = "",
    save_to: str = "",
    refresh: bool = False,
    with_html: bool = False,
    with_latex: bool = False,
    vision: bool = True,
    with_text: bool = False,
    limit: int = 50,
    note: str = "",
    symbols: list[str] | None = None,
    files: list[str] | None = None,
    kind: str = "rationale",
    root: str | None = None,
) -> Any:
    """Prior art you can actually read: arXiv search, full papers, page images.

    WHEN TO USE: before building a non-trivial mechanism - caching, consensus,
    retry and idempotency, ranking, scheduling, rate limiting, a wire format.
    Anything where the naive version breaks at scale and a wrong choice costs a
    rebuild rather than a typo.
    AFTER THIS: read the paper rather than its abstract - read() by section,
    grep() for one question, render() for figures the text layer drops - then
    paper(action='remember') so the next agent finds the citation instead of
    re-deriving the decision.


    The abstract is not the paper. Everything here is built to get past it: the
    whole text, cached once and served in slices, plus rendered pages for the
    figures and tables no text extractor recovers.

    Sources are interchangeable. paper_id takes an arXiv id in any form
    (2401.12345v2, arXiv:2401.12345, an abs or pdf URL); url takes any http(s)
    PDF or a landing page that names one; path takes a local .pdf.

    Actions:
      search    query arXiv. Returns real metadata, abstracts truncated,
                because a search is for choosing what to read.
      fetch     download, extract, and cache one paper. Returns the outline and
                page count so you know what to ask for next. Idempotent.
      read      the cached text. mode='outline' (default) lists sections;
                mode='full' is the entire paper; mode='latex' reads the
                submission's original TeX, where section boundaries are
                declared rather than guessed and formulas are intact;
                mode='html' is arXiv's HTML rendering;
                mode='abstract' if you only want the pointer. pages='7-12' or
                section='4'/'method' narrow it. Long reads page via offset.
                A paper with no text layer (a scan) comes back as page images
                rather than as an empty string, so a read always answers.
      grep      regex search inside the full text, with page-anchored context.
                Answers one question without reading the other thirty pages.
                Says so when the paper has no text layer, because zero hits
                there means 'not searchable', not 'not present'.
      render    rasterise pages to PNG and return them as images. This is how
                you see a figure, a table, an architecture diagram, or a
                scanned PDF whose text layer is empty. Max 8 pages per call.
      figures   extract embedded raster figures at their own resolution.
                Vector figures are not rasters; use render for those.
      download  copy the PDF into any directory you name.
      list      what is already cached.
      forget    drop one paper from the cache.
      remember  write what a paper settled into this repository's memory graph,
                anchored to the symbols and files it informed, so the next
                agent finds the citation instead of rediscovering it.

    Args:
        action: one of the actions above.
        query: search text. Fielded arXiv syntax (ti:, au:, abs:, AND/OR) is
            passed through untouched; plain text is wrapped in all:"...".
        paper_id: an arXiv id, in any of the forms above.
        url: any http(s) PDF URL, or a landing page that names one.
        path: a local .pdf file.
        category: arXiv categories to restrict to, e.g. "cs.DC cs.DB".
        max_results: search hits to return, 1-100.
        sort: relevance | recent | updated.
        start: offset into the search results, for paging.
        mode: for read - outline | full | latex | html | abstract.
        pages: page selection like "3" or "7-12" or "1,4,9-11". Used by read,
            render, and figures.
        section: section number ("4", "4.2") or name ("method") for read.
        pattern: regex for grep.
        ignore_case: case-insensitive grep. Default true.
        context: characters of context around each grep hit.
        max_chars: cap on returned text per read call. 0 means no cap.
        offset: character offset into a read, for continuing a long one.
        dpi: render resolution, 50-400. Higher is sharper and much larger.
        dest: directory to download the PDF into.
        filename: override the generated filename for download.
        save_to: directory for rendered pages or extracted figures.
        refresh: re-download and re-extract even if cached.
        with_html: on fetch, also pull arXiv's HTML rendering if it exists.
        vision: on read, when the paper has no extractable text layer, return
            the requested pages as images instead of an empty string. On by
            default - a scanned paper is readable, just not as text. Set false
            if you want the empty result rather than the pixels.
        with_latex: on fetch, also pull the original TeX source. Pulled
            automatically when the PDF extracts poorly, which is the same set
            of old papers whose metadata the arXiv API tends to refuse.
        with_text: on download, write the extracted text alongside the PDF.
        limit: maximum rows for list.
        note: for remember - what this paper settled, in your own words. This
            is the memory's substance; without it only the citation is stored.
        symbols: for remember - symbols this paper informed, so the memory
            anchors to real code.
        files: for remember - files this paper informed.
        kind: for remember - rationale | decision | note | investigation.
        root: repository path, for remember. Defaults to the working directory.
    """
    verb = (action or "search").lower().strip()

    try:
        if verb == "search":
            return papers_mod.search(query=query, category=category, max_results=max_results,
                                     sort=sort, start=start)

        if verb == "fetch":
            return papers_mod.fetch(paper_id=paper_id, url=url, path=path,
                                    refresh=refresh, with_html=with_html,
                                    with_latex=with_latex)

        if verb == "read":
            result = papers_mod.read(paper_id=paper_id, url=url, path=path, mode=mode,
                                     pages=pages, section=section, max_chars=max_chars,
                                     offset=offset)
            # A scanned paper has no text to return. Handing back an empty
            # string and a warning makes the caller do a second round trip to
            # find out the paper is pixels; rendering the same pages it just
            # asked for answers the original question instead. This is the
            # whole point of a vision-capable reader having the PDF already.
            if vision and _text_is_empty(result):
                return _read_as_images(result, paper_id=paper_id, url=url, path=path,
                                       pages=pages, dpi=dpi, save_to=save_to)
            return result

        if verb == "grep":
            return papers_mod.grep(paper_id=paper_id, url=url, path=path, pattern=pattern,
                                   ignore_case=ignore_case, context=context)

        if verb == "render":
            result = papers_mod.render(paper_id=paper_id, url=url, path=path,
                                       pages=pages or "1", dpi=dpi, save_to=save_to)
            images = [Image(data=item.pop("png"), format="png") for item in result["rendered"]]
            # A mixed list is why this tool returns Any: the summary tells the
            # caller which page each image is, and the images are the point.
            return [result, *images]

        if verb == "figures":
            return papers_mod.figures(paper_id=paper_id, url=url, path=path, pages=pages,
                                      save_to=save_to)

        if verb == "download":
            return papers_mod.download(paper_id=paper_id, url=url, path=path, dest=dest,
                                       filename=filename, with_text=with_text)

        if verb == "list":
            return papers_mod.cached(limit=limit)

        if verb == "forget":
            return papers_mod.forget(paper_id=paper_id, url=url, path=path)

        if verb == "remember":
            if not note.strip():
                return _fail("remember needs note: what this paper settled, in your own words. "
                             "A citation with no claim attached helps nobody.")
            field = _REMEMBER_FIELDS.get((kind or "rationale").lower())
            if field is None:
                return _fail(f"unknown kind {kind!r} for remember. Valid: "
                             + ", ".join(sorted(_REMEMBER_FIELDS)))

            meta = papers_mod.fetch(paper_id=paper_id, url=url, path=path)
            cite = papers_mod.citation(meta)
            source = meta.get("abs_url") or meta.get("source_url") or meta.get("source_path")
            current = ws_mod.open_workspace(root)
            try:
                ws_mod.ensure_indexed(current)
                # The claim goes in a typed field, not just the summary: the
                # compiler builds memories from those fields, and an event whose
                # kind is absent from EVENT_KIND_TO_MEMORY with every field
                # empty compiles to nothing at all, silently.
                payload = {
                    "kind": "note",
                    "summary": f"{note.strip()} [prior art: {cite}]",
                    "reasoning": (
                        f"Established from {cite}, read in full rather than from its abstract. "
                        f"Source: {source}. Cached at {meta.get('cache_dir')} "
                        f"({meta.get('pages')} pages)."
                    ),
                    "files": files or [], "symbols": symbols or [], "changes": [],
                    "invariants": [], "warnings": [], "failed_attempts": [],
                    "decisions": [], "contracts": [], "performance": [], "security": [],
                    "conventions": [], "rationale_notes": [], "bugs": [], "migrations": [],
                    "tests": [], "contracts_with": [], "caused_by": [], "authority": "agent",
                }
                payload[field] = [f"{note.strip()} (prior art: {cite}; {source})"]

                result = compiler.record_event(current.store, current.catalog, current.repo_id,
                                               current.root, current.commit, payload)
                result["resolved_root"] = str(current.root)
                result["citation"] = cite
                result["recorded_as"] = {"memory_kind": kind, "payload_field": field}
                if not result.get("memories_created"):
                    result["warning"] = ("no memory was created; the note may have been empty "
                                         "after trimming")
                return result
            finally:
                current.close()

        return _fail(f"unknown paper action {action!r}. Valid: search, fetch, read, grep, "
                     "render, figures, download, list, forget, remember")

    except papers_mod.PaperError as err:
        return _fail(str(err))


def main() -> None:
    # Before mcp.run(), never after: loading numpy's native extensions once the
    # stdio server owns the process wedges in the Windows loader and the first
    # workspace(action='open') never returns. See embed.preload_native.
    embed_mod.preload_native()
    mcp.run("stdio")
