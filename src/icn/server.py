"""MCP surface: nine tools.

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
    experiment   init, tree, create, checkout, commit, diff, run, status, log,
                 wait, cancel, runs, conclude, promote, apply, set_command
    paper        search, fetch, read, grep, render, figures, download, list,
                 forget, remember
    conversations search, list, get, digest, refresh, status, doctor, schema,
                 purge, restore - past chats from every agent on this machine

Every response carries the resolved root, so a wrong workspace is visible at
once instead of quietly poisoning the store.

paper() is annotated `-> Any` rather than `-> dict[str, Any]` like its
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
from . import feedback as feedback_mod
from . import handoff as handoff_mod
from . import history as history_mod
from . import rules as rules_mod
from . import lab as lab_mod
from . import papers as papers_mod
from . import search as search_mod
from . import workspace as ws_mod
from .db import rows

INSTRUCTIONS = """Persistent code knowledge for this workspace.

Start with workspace(action='open'). It returns a briefing: the rules that
govern this code, what has already been tried and rejected, and what is
currently unverified. Read it before opening files - it exists so you do not
have to rediscover by reading.

Then investigate() with what you are about to do, in plain language. One call
returns code structure, the rationale behind it, prior failures, invariants,
tests and blast radius, as compact capsules rather than file dumps, plus what
earlier sessions from any coding agent said while working in this repository.
Prefer it over grepping: it searches code, knowledge and past conversations
together.

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

experiment() is for questions only a measurement can settle: is this change
faster, more accurate, cheaper? Each experiment is exact code on its own branch
in .icn-lab/ (separate from agit and from .git), run with one fixed command.
conclude() turns a measured result into a memory that carries its evidence, so
a later agent meets "we tried X, run R measured Y" rather than an opinion.

conversations() reads the transcripts every coding agent on this machine keeps
(Codex, Claude Code, Copilot, Cursor, Gemini, ...) plus ICN's own archive of
the ones the vendors have since deleted. investigate() already attaches the
hits for this repository; use conversations() to read a whole session
(action='get'), see what it did (action='digest'), or search across every
codebase. A chat turn is anchored to nothing, so it is a lead, never a fact.
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
            # Titles only: this is a five-row pointer inside the open briefing,
            # not a reading surface. memory(action='list') carries the bodies.
            urgent = compiler.list_memories(current.store, anchor_status=anchor_mod.NEEDS_REVIEW,
                                            status="ACTIVE", limit=5, bodies=False)
            # What already exists here. An agent cannot ask the right question
            # before it knows what is on the shelf, so open() volunteers it.
            brief = briefing_mod.build(current.store)
            # A handoff the session-start hook has not already claimed. Claimed
            # here, so it is shown exactly once whichever path reaches it first.
            waiting = handoff_mod.claim(current.store, claimed_by="workspace-open")
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
                **_lab_briefing(current.root),
                **_trace_briefing(current.root),
                **({"handoff": waiting} if waiting else {}),
                "next": ("read the handoff, then investigate() the next step it names" if waiting
                         else "investigate('what you are about to change')"),
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
    conversations: bool = True,
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
    token budget rather than file dumps. The `conversations` section is what
    past sessions from any coding agent (Codex, Claude Code, Copilot, Cursor,
    ...) said about the same terms while working in this repository: leads to
    follow with conversations(action='get'), not anchored facts.

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
        conversations: also search past agent transcripts scoped to this
            repository. Off skips the transcript index entirely.
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
                    conversations=conversations,
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
        recalled = [
            {"repository_root": item["root"], **hit}
            for item in repository_results
            for hit in ((item.get("conversations") or {}).get("hits") or [])
        ]
        recalled.sort(key=lambda value: float(value.get("score", 0)))
        return {
            "ok": True,
            "multi_root": True,
            "resolved_roots": unique,
            "query": query,
            "repositories": repository_results,
            "capsules": capsules[:12],
            "problems": problems,
            "conversations": {"hits": recalled[:search_mod.CONVERSATION_HITS]} if conversations else None,
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
                # No chain is not no knowledge: a symbol governed by a
                # high-severity invariant used to come back empty-handed here.
                attached = rows(current.store.execute(
                    "SELECT DISTINCT m.memory_id, m.kind, m.severity, m.title,"
                    " (SELECT a.status FROM anchors a WHERE a.memory_id = m.memory_id"
                    "   AND a.symbol_id = e.to_id LIMIT 1) AS anchor_status"
                    " FROM memory_edges e JOIN memories m ON m.memory_id = e.from_id"
                    " WHERE e.to_id = ? AND e.kind = 'APPLIES_TO' AND e.status = 'ACTIVE'"
                    " AND m.status = 'ACTIVE'"
                    " ORDER BY CASE m.severity WHEN 'critical' THEN 0 WHEN 'high' THEN 1"
                    " WHEN 'medium' THEN 2 ELSE 3 END LIMIT 8",
                    (match["row"]["symbol_id"],)))
                return {"ok": True, "resolved_root": str(current.root),
                        "symbol": match["row"]["symbol_path"], "why_it_exists": None,
                        "attached_memories": attached,
                        **_observed(current.root, match["row"]),
                        "note": "no causal chain recorded for this symbol; the memories"
                                " attached to it are listed instead"}
            return {"ok": True, "resolved_root": str(current.root),
                    "symbol": match["row"]["symbol_path"], "why_it_exists": story,
                    **_observed(current.root, match["row"]),
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
            conversations=conversations,
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
    timeout_seconds: int = 600,
    run_options: list[str] | str | None = None,
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
      run     the runtime counterpart of all of the above: execute `target`
              (any shell command: 'python app.py', 'pytest -x tests/test_a.py',
              'npm test') and record what actually happened. Python gets the
              full call flow, arguments, return values, how local variables
              changed line by line, time and CPU per function, exceptions and
              library calls; Node.js gets a sampled CPU profile; anything else
              gets time, memory, exit code and output. Calls the static graph
              does not know (dynamic dispatch, callbacks) are listed. Returns
              the path of report.md: READ THAT FILE, it is written for you.
      run_diff
              compare two recorded runs, e.g. before and after a change, or a
              passing and a failing input: the first point where the call
              sequence diverges, the first variables whose values differ, calls
              and exceptions present in only one run, and time that moved.
              `target` and `to` name the two runs (folder or name fragment);
              omit both to compare the two most recent runs in `root`.

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
            For action='run', the directory the command runs in.
        timeout_seconds: for action='run', stop the command after this long.
        run_options: for action='run', any of 'libraries' (also trace the
            standard library and installed packages; slower), 'memory'
            (allocation sites; slower), 'no-values' (flow and timing only;
            fastest).
    """
    action = (action or "impact").lower().strip()
    known = ("impact", "trace", "cycles", "entrypoints", "areas", "triggers",
             "coupling", "hotspots", "deadcode", "run", "run_diff")
    if action not in known:
        return _fail(f"unknown action {action!r}; use one of {', '.join(known)}", root)
    if action == "run":
        return _run_traced(target, root, timeout_seconds, run_options)
    if action == "run_diff":
        from . import flowdiff
        cwd = Path(root).resolve() if root else Path.cwd()
        try:
            if bool(target.strip()) != bool(to.strip()):
                return _fail("run_diff takes both `target` and `to`, or neither", root)
            result = flowdiff.diff(cwd, target.strip() or None, to.strip() or None)
        except ValueError as err:
            return _fail(str(err), root)
        return {**result, "action": "run_diff", "resolved_root": str(cwd),
                "next": "read the report file: first divergence, differing values, calls, errors"}
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


def _run_traced(command: str, root: str | None, timeout_seconds: int,
                options: list[str] | str | None) -> dict[str, Any]:
    from . import flowrun
    if not command.strip():
        return _fail("target is required for action='run': the command to execute", root)
    chosen = {o.strip().lower() for o in ([options] if isinstance(options, str) else options or [])}
    unknown = chosen - {"libraries", "memory", "no-values"}
    if unknown:
        return _fail(f"unknown run_options {sorted(unknown)}; use libraries, memory, no-values", root)
    cwd = Path(root).resolve() if root else Path.cwd()
    if not cwd.is_dir():
        return _fail(f"{cwd} is not a directory", root)
    store_path = None
    try:
        # Only a repository ICN already knows is compared, found through the
        # catalog alone: opening a workspace on an arbitrary directory would
        # register it as a new repository. It is indexed first so the runtime
        # calls meet the current code, then released, since the command may
        # run for minutes.
        from . import hooks as hooks_mod, paths
        located = hooks_mod.resolve_repo(str(cwd))
        if located:
            current = ws_mod.open_workspace(str(located[1]))
            try:
                ws_mod.ensure_indexed(current)
                store_path = paths.repo_db_path(current.repo_id)
            finally:
                current.close()
    except Exception:  # noqa: BLE001 - tracing works outside a repository too
        store_path = None
    result = flowrun.run(
        command, cwd, values="no-values" not in chosen, libraries="libraries" in chosen,
        memory="memory" in chosen, timeout=max(1, timeout_seconds), echo=False,
        static_graph=(lambda edges: flowrun.check_static_graph(store_path, edges))
        if store_path and store_path.exists() else None)
    result["action"] = "run"
    result["resolved_root"] = str(cwd)
    result["validation_boundary"] = (
        "Runtime evidence from one execution. It proves these calls and values happened on this "
        "run with these inputs; it says nothing about paths this run did not take.")
    return result


@mcp.tool()
def record(
    summary: str,
    kind: str = "note",
    reasoning: str = "",
    files: list[str] | str | None = None,
    symbols: list[str] | str | None = None,
    changes: list[str] | str | None = None,
    invariants: list[str] | str | None = None,
    warnings: list[str] | str | None = None,
    failed_attempts: list[str] | str | None = None,
    decisions: list[str] | str | None = None,
    contracts: list[str] | str | None = None,
    performance: list[str] | str | None = None,
    security: list[str] | str | None = None,
    conventions: list[str] | str | None = None,
    rationale: list[str] | str | None = None,
    bugs: list[str] | str | None = None,
    migrations: list[str] | str | None = None,
    tests: list[str] | str | None = None,
    contracts_with: list[dict] | None = None,
    caused_by: list[str | dict] | None = None,
    evidence: list[str] | str | None = None,
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
        evidence: proof that what this records actually happened. Either
            experiment run ids (from experiment()), whose commit, command,
            exit code and metrics are written into every memory, or recorded
            runs (from graph(action='run'), named by their folder such as
            '20260917-160603-perms'), whose command, exit code, timing and
            hottest functions are written in the same way, with the path of
            the full report. The claim then carries its proof instead of
            asking to be believed.
        authority: 'agent' or 'human'. Human memories cannot be rewritten by an agent.
        root: repository path. Defaults to the server's working directory.
    """
    if not summary.strip():
        return _fail("record requires a summary")
    # Agents routinely pass one claim as a bare string. Rejecting that with a
    # schema error cost a retry of the whole call; one string is a one-item list.
    (files, symbols, changes, invariants, warnings, failed_attempts, decisions, contracts,
     performance, security, conventions, rationale, bugs, migrations, tests, evidence) = (
        [v] if isinstance(v, str) else v
        for v in (files, symbols, changes, invariants, warnings, failed_attempts, decisions,
                  contracts, performance, security, conventions, rationale, bugs, migrations,
                  tests, evidence))

    current = ws_mod.open_workspace(root)
    try:
        run_ids = [r.strip() for r in (evidence or []) if r and r.strip()]
        if run_ids:
            # Two kinds of proof, one field: a lab run measured something, a
            # recorded run observed something. Recorded runs are named by their
            # folder (`20260917-160603-perms`), lab runs by a `run_` id.
            traces = [r for r in run_ids if not r.startswith("run_")]
            lab_runs = [r for r in run_ids if r.startswith("run_")]
            proof = []
            try:
                if lab_runs:
                    proof += lab_mod.evidence_for(current.root, lab_runs)
                if traces:
                    from . import flowrun
                    proof += flowrun.evidence_for(current.root, traces)
            except (lab_mod.LabError, ValueError) as err:
                return _fail(str(err), str(current.root))
            reasoning = "\n".join([reasoning.strip(), *proof]).strip()
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
        if run_ids:
            lab_mod.link_evidence(current.root, [m["memory_id"] for m in result["memories_created"]],
                                  run_ids, None)
            result["evidence"] = run_ids
        if result.get("unresolved_references"):
            result["hint"] = ("some names did not resolve to indexed code; they were kept as "
                              "repo-scoped knowledge. Check spelling or run "
                              "workspace(action='reindex').")
        # The caller wrote these bodies moments ago. Echoing each composed body
        # (claim plus the shared reasoning, ~900 characters apiece) made a
        # seven-memory record answer with ~5k tokens of its own input.
        for memory in result.get("memories_created", []):
            memory.pop("body", None)
        result["memories_note"] = ("bodies omitted here; memory(action='list') returns them for "
                                   "many at once, memory(action='get', memory_id=...) for one")
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
    signal: str = "",
    open_questions: list[str] | None = None,
    next_steps: list[str] | None = None,
    files: list[str] | None = None,
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
                  Bodies are included, so this reads many at once rather than
                  costing one get() per memory.
      get         one memory with its anchors, edges and version history.
                  Re-anchoring logs are summarised; action='history' has them raw.
      history     the full re-anchoring audit trail for one memory.
      verify      confirm a memory still applies. Clears NEEDS_REVIEW/DRIFTED.
      correct     edit a memory in place. Previous text is versioned, not lost.
      supersede   replace a memory, keeping both and the link between them.
      resolve     mark a warning or bug as no longer live.
      guard       record that an existing test covers an existing rule, when
                  the two were written in separate calls. Pass the rule as
                  memory_id and the test_evidence memory id as body.
      reanchor    re-run the anchoring cascade for one memory.
      feedback    vote on a memory you were shown: signal='helpful',
                  'not_helpful', 'stale' or 'wrong' (stale and wrong need a
                  reason). Wrong or repeatedly unhelpful memories stop being
                  volunteered by hooks; nothing is deleted.

    Handoffs, "where I left off" for the next session (claimed exactly once,
    by its session-start hook or workspace(action='open')):
      handoff          write one before stopping mid-task: body=summary, plus
                       open_questions, next_steps, files.
      handoff_list     recent handoffs and their status, without claiming.
      handoff_cancel   withdraw an open one (memory_id=handoff id).

    Rules promoted into this repository's CLAUDE.md and AGENTS.md, a small
    capped block changed only by these commands:
      rules_recommend  ranked, codebase-specific candidates. Read-only.
      rules_approve    promote memory_id (body= optional wording). Refused at
                       the cap, naming the weakest rule to remove first.
      rules_edit       reword a promoted rule (body=).
      rules_remove     demote; the memory itself is unchanged.
      rules_list       what is promoted, and which rules have gone stale.
      Promote only on the user's say-so: these edit files they own.

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
        signal: for feedback.
        open_questions: for handoff.
        next_steps: for handoff.
        files: for handoff.
        root: repository path. Defaults to the server's working directory.
    """
    current = ws_mod.open_workspace(root)
    try:
        act = (action or "list").lower().strip()
        where = str(current.root)

        if act == "handoff":
            identity = ws_mod.status(current).get("identity") or {}
            return {"resolved_root": where,
                    **handoff_mod.create(current.store, body, open_questions, next_steps, files,
                                         from_agent=actor, branch=identity.get("branch"),
                                         head_commit=current.commit)}
        if act == "handoff_list":
            return {"ok": True, "resolved_root": where,
                    "handoffs": handoff_mod.listing(current.store, limit)}
        if act == "handoff_cancel":
            if not memory_id:
                return _fail("handoff_cancel requires memory_id set to the handoff id", where)
            return {"resolved_root": where, **handoff_mod.cancel(current.store, memory_id)}
        if act.startswith("rules_"):
            try:
                if act == "rules_recommend":
                    return {"resolved_root": where, **rules_mod.recommend(current.store, limit=min(limit, 20))}
                if act == "rules_list":
                    return {"resolved_root": where, **rules_mod.listing(current.store, current.root)}
                if not memory_id:
                    return _fail(f"{act} requires memory_id", where)
                if act == "rules_approve":
                    return {"resolved_root": where,
                            **rules_mod.approve(current.store, current.root, memory_id, body, actor=actor)}
                if act == "rules_edit":
                    return {"resolved_root": where,
                            **rules_mod.edit(current.store, current.root, memory_id, body)}
                if act == "rules_remove":
                    return {"resolved_root": where,
                            **rules_mod.remove(current.store, current.root, memory_id)}
            except rules_mod.RulesError as err:
                return _fail(str(err), where)
            return _fail(f"unknown rules action: {action}", where)

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
        if act == "history":
            # The raw re-anchoring log, which get() summarises. Separate because
            # it is an audit trail: large, repetitive, and rarely what a reader
            # of the knowledge itself is after.
            return {"ok": True, "resolved_root": str(current.root), "memory_id": memory_id,
                    "anchors": rows(current.store.execute(
                        "SELECT anchor_id, symbol_path, file_path, status, reanchor_history"
                        " FROM anchors WHERE memory_id=?", (memory_id,)))}
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

        if act == "feedback":
            return {"resolved_root": str(current.root),
                    **feedback_mod.record(current.store, memory_id, signal, reason, actor)}

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


def _observed(root: Path, symbol: dict[str, Any]) -> dict[str, Any]:
    """Whether a recorded run actually ran this symbol, and who called it.

    The static graph answers 'who could call this' and says so as a lower
    bound. A recording answers 'who did call this, on that run', which is the
    evidence the bound is missing. Never allowed to break why().
    """
    try:
        from . import flowdiff, flowrun

        recorded = flowdiff.runs_in(root)
        if not recorded:
            return {}
        wanted = symbol.get("symbol_path")
        for run in recorded[-3:][::-1]:
            edges = flowrun.edges_of_run(run, root)
            callers = sorted({f"{e['caller']} ({e['caller_file']}:{e['caller_line']})"
                              for e in edges if e["callee"] == wanted})
            calls = sum(e["count"] for e in edges if e["callee"] == wanted)
            if calls:
                return {"observed_at_runtime": {
                    "run": run.name, "calls": calls, "called_by": callers[:8],
                    "report": str(run / "report.md"),
                    "note": "observed on that run only; it says nothing about paths that run "
                            "did not take"}}
        return {"observed_at_runtime": {
            "run": recorded[-1].name, "calls": 0,
            "note": "the most recent recorded runs never reached this symbol; that is not proof "
                    "it is unreachable, only that those runs did not take it"}}
    except Exception:  # noqa: BLE001 - see docstring
        return {}


def _trace_briefing(root: Path) -> dict[str, Any]:
    """Recorded runs waiting to be read, for open().

    A recording nobody knows about is worth as little as a memory nobody
    queries, so open() names the most recent ones. Like the lab briefing, it
    is never allowed to break open().
    """
    try:
        from . import flowdiff

        recorded = flowdiff.runs_in(root)
        if not recorded:
            return {}
        latest = []
        for run in recorded[-3:][::-1]:
            summary = flowdiff.summary_of(run)
            latest.append({"run": run.name, "command": summary["command"],
                           "exit_code": summary["process"].get("exit_code"),
                           "report": str(run / "report.md")})
        return {"recorded_runs": {
            "total": len(recorded), "latest": latest,
            "note": "graph(action='run') records a new one; graph(action='run_diff') compares two; "
                    "record(evidence=['<run>']) cites one as proof"}}
    except Exception as err:          # noqa: BLE001 - see docstring
        return {"recorded_runs": {"error": f"recordings unreadable: {err}"}}


def _lab_briefing(root: Path) -> dict[str, Any]:
    """The experiment lab's state for open(), or nothing when there is no lab.

    Never allowed to break open(): the briefing is a convenience, and a damaged
    lab must not stop an agent from reaching the rest of its knowledge.
    """
    try:
        found = lab_mod.summary(root)
    except Exception as err:          # noqa: BLE001 - see docstring
        return {"experiments": {"error": f"lab unreadable: {err}"}}
    return {"experiments": found} if found else {}


def _apply_experiment(current: Any, exp: str) -> dict[str, Any]:
    """Write a measured experiment's code into the working tree and settle its memories.

    Applying the winner changes exactly the code its conclusion memories
    describe, so ICN's anchor cascade flags them NEEDS_REVIEW as if they had
    gone stale. They have not: this is the change they recorded. They are
    re-verified here; every other memory the change put under review is
    reported, because those genuinely need a look.
    """
    import subprocess
    import tempfile

    plan = lab_mod.apply_plan(current.root, exp)
    if not plan["patch"].strip():
        return {"ok": True, "slug": plan["slug"], "applied": False,
                "note": "this experiment's code is identical to the baseline"}

    with tempfile.NamedTemporaryFile("wb", suffix=".patch", delete=False) as handle:
        handle.write(plan["patch"].encode("utf-8") + b"\n")
        patch_file = handle.name
    try:
        def git_apply(*extra: str) -> subprocess.CompletedProcess:
            # stdin detached for the same reason as every other git child here.
            return subprocess.run(["git", "apply", "--whitespace=nowarn", *extra, patch_file],
                                  cwd=str(current.root), capture_output=True, text=True,
                                  stdin=subprocess.DEVNULL, check=False)

        checked = git_apply("--check")
        if checked.returncode != 0:
            raise lab_mod.LabError(
                "the working tree no longer matches the baseline in the files this experiment "
                f"changes ({', '.join(plan['files'][:10])}), so applying it could overwrite other "
                "work. Nothing was changed. git said: "
                f"{(checked.stderr or checked.stdout).strip()[:600]}")
        applied = git_apply()
        if applied.returncode != 0:
            raise lab_mod.LabError(f"git apply failed: {(applied.stderr or applied.stdout).strip()[:600]}")
    finally:
        Path(patch_file).unlink(missing_ok=True)

    ws_mod.ensure_indexed(current)
    # A conclusion is a measurement of a recorded lab commit, so changing the
    # working tree cannot make it untrue: "insertion sort was slower" still
    # holds after the winner lands. Every lab-measured memory the cascade just
    # flagged is re-verified; flagging them would bury the knowledge that does
    # need a look under a pile that does not.
    measured = set(plan["measured_memories"])
    placeholders = ",".join("?" for _ in plan["files"]) or "''"
    flagged = [dict(r) for r in current.store.execute(
        "SELECT DISTINCT m.memory_id, m.kind, m.title FROM anchors a"
        " JOIN memories m ON m.memory_id = a.memory_id"
        f" WHERE a.status = ? AND m.status = 'ACTIVE' AND a.file_path IN ({placeholders})",
        (anchor_mod.NEEDS_REVIEW, *plan["files"]))]
    reverified = []
    for memory in flagged:
        if memory["memory_id"] in measured:
            anchor_mod.mark_verified(current.store, memory["memory_id"], current.commit,
                                     actor="experiment-apply")
            reverified.append(memory["memory_id"])
    review = [m for m in flagged if m["memory_id"] not in measured]
    return {
        "ok": True, "slug": plan["slug"], "applied": True, "files": plan["files"],
        "from_lab_commit": plan["commit"], "lineage": plan["lineage"],
        "memories_reverified": reverified,
        "memories_to_review": [{k: m.get(k) for k in ("memory_id", "kind", "title")} for m in review],
        "next": ("the code is in the working tree but not committed to git; run the project's "
                 "tests, then commit it as you normally would"
                 + ("; memories_to_review lists knowledge this change may have outdated"
                    if review else "")),
    }


def _touched_symbols(store: Any, touched: dict[str, list[str]], cap: int = 12) -> list[str]:
    """Indexed symbols in each changed file whose name the experiment's diff mentions.

    investigate() attaches memories to capsules through symbol edges only, so a
    conclusion anchored to files alone is stored, anchored, and never surfaced.
    Measured live: a concluded loss on sort_impl.py did not appear for a query
    naming it. Matching is by name within the same file, against the index of
    the working tree, which is what a later agent will be investigating.
    """
    picked: list[str] = []
    for path, names in touched.items():
        wanted = set(names)
        for row in store.execute(
            "SELECT symbol_path, name FROM symbols WHERE status='ACTIVE' AND last_known_path=?"
            " ORDER BY line_start", (path,)
        ):
            if row["name"] in wanted and row["symbol_path"] not in picked:
                picked.append(row["symbol_path"])
                if len(picked) >= cap:
                    return picked
    return picked


@mcp.tool()
def experiment(
    action: str = "tree",
    exp: str = "",
    run: str = "",
    title: str = "",
    hypothesis: str = "",
    parent: str = "",
    command: str = "",
    message: str = "",
    against: str = "",
    verdict: str = "",
    note: str = "",
    timeout_seconds: float = 0,
    wait_seconds: float = 60,
    tail: int = 8000,
    offset: int | None = None,
    force: bool = False,
    limit: int = 20,
    caused_by: list[str | dict] | None = None,
    root: str | None = None,
) -> dict[str, Any]:
    """Measure ideas instead of arguing them: a tree of experiments with real runs.

    WHEN TO USE: when a question can only be settled by running code - is this
    faster, more accurate, smaller, more stable - and the answer should outlive
    the session. Not for ordinary edits; use agit to checkpoint those.
    AFTER THIS: conclude() every result you judged, so it becomes a memory with
    its evidence attached.


    Think of recipes. The baseline is the recipe you have and the one way you
    taste it (the command). Each experiment copies a recipe and changes one
    thing. You always taste the same way, you never scribble on a recipe you
    already tasted, and the next round starts from the winner.

    The lab lives in .icn-lab/ (gitignored), fully separate from agit and from
    the user's .git. It enforces the rules rather than suggesting them:
      - One command for every experiment, fixed once anything is measured.
        Vary code and config on a child, never the command or env vars.
      - An experiment freezes once a run answers it. commit then refuses; put
        the next idea on a child. A crash answered nothing, so the node stays
        editable, but two unanswered failures in a row need force=True.
      - A run executes the committed snapshot in its own folder, never the
        editable checkout, and refuses while the checkout has uncommitted edits.
    Report results by printing lines like `ICN_METRIC loss=0.4312`.

    Grow the tree downward: siblings are the co-equal options of ONE decision;
    the next decision goes under that round's winner. tree() warns about a flat
    fan (everything under the root) and a noodle (a chain of one-child links).

    Actions:
      init         create the lab. Snapshots the working tree as the baseline.
                   Needs command. title defaults to 'baseline'.
      tree         every experiment with its state, verdict, latest run and
                   metrics, the focal node, and tree-shape warnings.
      create       new experiment. Needs title; hypothesis is what you expect.
                   parent defaults to the focal node (latest winner, else the
                   baseline).
      checkout     a folder with the experiment's code, to edit. Returns path.
      commit       commit that folder onto the experiment. Refused when frozen.
      diff         the experiment's change against its parent, or `against`.
      run          launch a detached run of the committed code. Survives this
                   server exiting. timeout_seconds bounds it.
      status       one run (run=) or one experiment (exp=) in detail.
      log          a run's output. tail bytes from the end, or from offset.
      wait         block up to wait_seconds (max 600) until a run finishes.
                   Returns on the FIRST finish so you can judge it and refill.
      cancel       stop a run and everything it started.
      runs         recent runs, all or for one experiment.
      conclude     judge a finished run: verdict win | loss | inconclusive |
                   void. win, loss and inconclusive freeze the experiment and
                   write an ICN memory (decision, failed_attempt, rationale)
                   carrying the run's commit, command, exit code, metrics and
                   the change against the parent. win also promotes it. void
                   says the run answered nothing (needs note) and unfreezes.
      promote      make a measured experiment the parent for the next round.
      apply        bring a measured experiment's code into the working tree:
                   the whole lineage from the baseline, applied only if those
                   files still match the baseline. The memories describing
                   that lineage are re-verified against the new code, and any
                   other memory the change put under review is listed.
      set_command  change the command. Only before anything is measured.

    It ties into the rest of ICN: a conclusion links to its parent's
    conclusion (LED_TO), so investigate(action='why') tells the story of how
    the code got here; caused_by= links it to what motivated it, such as a
    paper(action='remember') memory; workspace(action='open') lists finished
    runs still waiting for a verdict.

    Read a run's log before concluding: status alone is not evidence.

    Args:
        action: one of the actions above.
        exp: experiment id or slug.
        run: run id.
        title: for init and create.
        hypothesis: for init and create - the expected effect, in a sentence.
        parent: for create - experiment id or slug to branch from.
        command: for init and set_command.
        message: for commit.
        against: for diff - compare with this experiment instead of the parent.
        verdict: for conclude.
        note: for conclude - what the result means, in your own words.
        timeout_seconds: for run. 0 means no limit.
        wait_seconds: for wait.
        tail: for log - bytes to return.
        offset: for log - start byte instead of the tail.
        force: for run - launch past the consecutive-failure cap.
        limit: for runs.
        caused_by: for conclude - memory ids that motivated this experiment,
            e.g. the paper memory that suggested it. Same shape as record().
        root: repository path. Defaults to the server's working directory.
    """
    verb = (action or "tree").lower().strip()
    current = ws_mod.open_workspace(root)
    work = current.root
    try:
        if verb == "init":
            result = lab_mod.init(work, command, title=title or "baseline", hypothesis=hypothesis)
        elif verb == "tree":
            result = lab_mod.tree(work)
        elif verb == "create":
            result = lab_mod.create(work, title, hypothesis=hypothesis, parent=parent)
        elif verb == "checkout":
            result = lab_mod.checkout(work, exp)
        elif verb == "commit":
            result = lab_mod.commit(work, exp, message)
        elif verb == "diff":
            result = lab_mod.diff(work, exp, against)
        elif verb == "run":
            result = lab_mod.start_run(work, exp, timeout_seconds=timeout_seconds or None,
                                       force=force)
        elif verb == "status":
            result = lab_mod.status(work, run=run, exp=exp)
        elif verb == "log":
            result = lab_mod.read_log(work, run, tail=tail, offset=offset)
        elif verb == "wait":
            result = lab_mod.wait(work, exp=exp, run=run, timeout=wait_seconds)
        elif verb == "cancel":
            result = lab_mod.cancel(work, run)
        elif verb == "runs":
            result = lab_mod.list_runs(work, exp, limit=limit)
        elif verb == "promote":
            result = lab_mod.promote(work, exp)
        elif verb == "set_command":
            result = lab_mod.set_command(work, command)
        elif verb == "conclude":
            result = lab_mod.conclude(work, verdict, exp=exp, run=run, note=note)
            payload = result.pop("memory_payload", None)
            if payload:
                # Memories come from typed fields; fill the ones record_event
                # expects so an absent list is never read as a missing key.
                for field in ("symbols", "changes", "invariants", "warnings", "failed_attempts",
                              "decisions", "contracts", "performance", "security", "conventions",
                              "rationale_notes", "bugs", "migrations", "tests", "contracts_with",
                              "caused_by"):
                    payload.setdefault(field, [])
                payload["authority"] = "agent"
                ws_mod.ensure_indexed(current)
                payload["symbols"] = _touched_symbols(current.store,
                                                      payload.pop("touched_identifiers", {}))
                lineage = payload.pop("lineage_memory", None)
                payload["caused_by"] = list(caused_by or [])
                if lineage:
                    payload["caused_by"].append({"memory": lineage, "kind": "LED_TO"})
                recorded = compiler.record_event(current.store, current.catalog, current.repo_id,
                                                 current.root, current.commit, payload)
                memory_ids = [m["memory_id"] for m in recorded.get("memories_created", [])]
                lab_mod.link_evidence(work, memory_ids, [result["run_id"]], result["verdict"],
                                      primary=recorded.get("primary_memory"))
                result["causal_links"] = recorded.get("causal_links", [])
                result["memories_created"] = [
                    {k: m[k] for k in ("memory_id", "kind", "title") if k in m}
                    for m in recorded.get("memories_created", [])
                ]
                result["primary_memory"] = recorded.get("primary_memory")
        elif verb == "apply":
            result = _apply_experiment(current, exp)
        else:
            return _fail(f"unknown experiment action {action!r}. Valid: init, tree, create, "
                         "checkout, commit, diff, run, status, log, wait, cancel, runs, "
                         "conclude, promote, apply, set_command", str(work))
        result["resolved_root"] = str(work)
        return result
    except lab_mod.LabError as err:
        return _fail(str(err), str(work))
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
    source: str = "arxiv",
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
      search    query a literature index. Returns real metadata, abstracts
                truncated, because a search is for choosing what to read.
                source='arxiv' (default) is the arXiv API. 'alphaxiv' searches
                the full text of arXiv papers and returns the matching snippets;
                'alphaxiv_semantic' is the same corpus by meaning rather than
                words. 'openalex' reaches journals and every discipline, and
                'biorxiv' is OpenAlex limited to bioRxiv preprints. Each result
                says how to fetch it, or that no open PDF is known.
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
        category: arXiv categories to restrict to, e.g. "cs.DC cs.DB". arXiv only.
        source: for search - arxiv | alphaxiv | alphaxiv_semantic | openalex | biorxiv.
        max_results: search hits to return, 1-100.
        sort: relevance | recent | updated, plus popular | historical for the
            non-arXiv sources.
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
                                     sort=sort, start=start, source=source)

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


@mcp.tool()
def conversations(
    action: str = "search",
    query: str | None = None,
    session: str | None = None,
    codebase: str | None = None,
    provider: str | None = None,
    surface: str | None = None,
    role: str | None = None,
    kind: str | None = None,
    tool_name: str | None = None,
    origin: str | None = None,
    after: str | None = None,
    before: str | None = None,
    match: str = "all",
    limit: int = 10,
    offset: int = 0,
    per_session: int = 3,
    context_before: int = 3,
    context_after: int = 3,
    start: int = 0,
    sort: str = "relevance",
    source_path: str | None = None,
    target: str | None = None,
    mode: str = "archive_only",
    reason: str | None = None,
    include_archive: bool = True,
    fuzzy: bool | str = "auto",
) -> dict[str, Any]:
    """Search past conversations from every coding agent on this machine.

    Codex, Claude Code, Copilot (CLI and VS Code), Cursor and the rest each keep
    transcripts in their own private format, and none can read the others'. This
    reads all of them, plus our own archive of transcripts the vendors have since
    deleted, and answers across the lot.

    Actions:
      search   full-text across every agent, with the messages around each hit.
               Filter by codebase (a directory), provider, surface, role, kind,
               tool_name, origin (live | archive), and a time window.
               Misspellings and other word forms still match: when the exact
               query finds little, each word is matched against similar terms
               that are actually in the index, and the reply says what it
               expanded to. match="any" ORs the words, match="phrase" requires
               them adjacent, a trailing * is a prefix, "quoted text" is exact.
      list     browse sessions rather than messages.
      get      one session's messages, paginated from `start`.
      digest   what a session did: tools used, files touched, user turns, errors.
      refresh  sync the archive and reindex now (the server also does this on a
               timer; you do not have to call this).
      status   is the background sync running, when did it last tick, what is
               in the archive.
      doctor   index health and any vendor schema drift.
      schema   the record shapes in one source file, for fixing an adapter.
      purge    delete an archived copy and tombstone it so a later sync does not
               copy it straight back. mode=archive_only keeps the live file
               searchable; mode=exclude_all hides it everywhere.
      restore  remove a tombstone so `target` can be archived and indexed again.

    Vendor stores are only ever read. The index and archive stay on this machine
    and carry whatever went through the agents' tools, so treat them as sensitive.
    """
    from . import transcripts as tx_mod
    from . import transcript_archive as archive_mod
    from . import transcript_sync as sync_mod

    action = (action or "search").strip().lower()
    index = None
    try:
        if action in ("status", "purge", "restore", "archives", "tombstones"):
            store = archive_mod.ArchiveStore()
            try:
                if action == "status":
                    return {"ok": True, "sync": sync_mod.status(), "archive": store.stats()}
                if action == "archives":
                    return {"ok": True, "archives": store.list_archives(provider=provider,
                                                                       limit=limit, offset=offset)}
                if action == "tombstones":
                    return {"ok": True, "tombstones": store.list_tombstones()}
                if action == "purge":
                    what = target or source_path or session
                    if not what:
                        return _fail("purge needs target=<live path, archive path or archive key>")
                    if mode not in archive_mod.TOMBSTONE_MODES:
                        return _fail(f"mode must be one of {archive_mod.TOMBSTONE_MODES}")
                    result = archive_mod.delete_archive(store, what, mode, reason)
                    # The index would otherwise keep serving what was purged.
                    idx = tx_mod.Index()
                    try:
                        result["index_rows_removed"] = idx.forget_source(what)
                    finally:
                        idx.close()
                    return {"ok": True, **result}
                if action == "restore":
                    if not target:
                        return _fail("restore needs target=<the tombstoned identifier>")
                    return {"ok": True, "removed": store.remove_tombstone(target), "target": target}
            finally:
                store.close()

        index = tx_mod.Index()

        if action == "search":
            if not query:
                return _fail("search needs a query")
            result = tx_mod.search(
                index, query, provider=provider, surface=surface, codebase=codebase, role=role,
                kind=kind, tool_name=tool_name, session=session, after=after, before=before,
                match=match, limit=limit, per_session=per_session, context_before=context_before,
                context_after=context_after, sort=sort, origin=origin, fuzzy=fuzzy)
            return {"ok": True, **result}

        if action == "list":
            return {"ok": True, **tx_mod.list_sessions(
                index, provider=provider, surface=surface, codebase=codebase, after=after,
                before=before, title=query, limit=limit, offset=offset, origin=origin)}

        if action == "get":
            if not session:
                return _fail("get needs session=<session_key, native id or source path>")
            return {"ok": True, **tx_mod.get_session(
                index, session, start=start, limit=limit, roles=role, kinds=kind)}

        if action == "digest":
            if not session:
                return _fail("digest needs session=<session_key, native id or source path>")
            return {"ok": True, **tx_mod.digest(index, session)}

        if action == "refresh":
            sources = None if include_archive else tx_mod.discover_sources()
            return {"ok": True, **index.refresh(sources=sources)}

        if action == "doctor":
            return {"ok": True, **tx_mod.doctor(index)}

        if action == "schema":
            if not source_path:
                return _fail("schema needs source_path=<a vendor transcript file>")
            return {"ok": True, **tx_mod.inspect_schema(index, source_path)}

        return _fail(f"unknown conversations action {action!r}. Valid: search, list, get, digest, "
                     "refresh, status, doctor, schema, purge, restore, archives, tombstones")
    finally:
        if index is not None:
            index.close()


def main() -> None:
    # Before mcp.run(), never after: loading numpy's native extensions once the
    # stdio server owns the process wedges in the Windows loader and the first
    # workspace(action='open') never returns. See embed.preload_native.
    embed_mod.preload_native()
    # Transcript sync runs itself from here on: the vendors delete history on
    # their own schedule, so waiting for someone to run a sync command loses it.
    # One lease-holding process does the work however many servers are running,
    # and a failure here must never stop the server coming up.
    try:
        from . import transcript_sync
        transcript_sync.start()
    except Exception:  # noqa: BLE001
        pass
    mcp.run("stdio")
