"""MCP surface: five tools.

PLAN 2 section 10. Agents waste turns choosing between near-identical tools, so
the surface is deliberately small and grouped by action:

    workspace    open, status, list, reindex, health, reconcile, archive,
                 detach, forget_checkout, purge
    investigate  search (default), expand, verify
    record       write one event, compiled into many facts
    memory       get, list, correct, supersede, verify, reanchor, resolve
    agit         status, diff, commit, log, branches, switch, restore, reset, show

Every response carries the resolved root, so a wrong workspace is visible at
once instead of quietly poisoning the store.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from . import agit as agit_mod
from . import anchors as anchor_mod
from . import briefing as briefing_mod
from . import catalog as catalog_mod
from . import causal
from . import compiler
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

Finish with record(). One call becomes many anchored facts. The highest-value
fields are failed_attempts and warnings: nothing else in your toolchain
captures what was tried and rejected, and that is what future agents
find most expensive to rediscover. caused_by=[memory_id] links your work into the causal
chain.

Memories carry an anchor_status. Anything other than ACTIVE has not been
verified against the current code: treat it as a lead, not a fact, and call
memory(action='verify') once you have confirmed it still applies.
"""

mcp = FastMCP("infinite-code-next", instructions=INSTRUCTIONS)


def _fail(error: str, root: str | None = None) -> dict[str, Any]:
    return {"ok": False, "error": error, "resolved_root": root}


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
    root: str | None = None,
) -> dict[str, Any]:
    """Inspect and correct stored knowledge.

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
        root: repository path. Defaults to the server's working directory.
    """
    current = ws_mod.open_workspace(root)
    try:
        act = (action or "list").lower().strip()

        if act == "list":
            return {"ok": True, "resolved_root": str(current.root),
                    "memories": compiler.list_memories(current.store, kind=kind, status=status,
                                                       anchor_status=anchor_status, limit=limit)}
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


def main() -> None:
    mcp.run("stdio")
