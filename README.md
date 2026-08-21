# Infinite Code Next

A persistent code-intelligence and provenance layer for AI coding agents,
delivered as a single zero-config MCP server.

Git blame tells you who changed a line. This tells an agent *why the code
exists*, *what was already tried and rejected*, *what must stay true*, and it
carries that knowledge with the code when the code moves.

## Install

```bash
pip install -e .
```

Register it once with your MCP client. There is no second step: no admin panel,
no port, no daemon, no per-repository setup.

```json
{
  "mcpServers": {
    "icn": { "command": "infinite-code-next" }
  }
}
```

Claude Code:

```bash
claude mcp add icn -- infinite-code-next
```

The server works out which repository it is in from its working directory, and
every response echoes the root it resolved so a wrong workspace is obvious
immediately.

## The workflow

Two calls to become productive, one to leave the next agent smarter.

```
workspace(action="open")
investigate("I need to change subscription cancellation. What will I break?")
   ... do the work ...
record(summary="...", warnings=[...], invariants=[...], failed_attempts=[...])
```

## Tools

| Tool | Actions |
|---|---|
| `workspace` | `open`, `status`, `list`, `reindex`, `health`, `reconcile`, `archive`, `detach`, `forget_checkout`, `purge` |
| `investigate` | search (default), `why`, `expand`, `verify` |
| `record` | write one event, compiled into many anchored facts |
| `memory` | `get`, `list`, `verify`, `correct`, `supersede`, `resolve`, `reanchor` |
| `agit` | `status`, `diff`, `commit`, `log`, `branches`, `switch`, `restore`, `reset`, `show` |

### `workspace(action="open")`

Returns an orientation **briefing**: the rules that govern this code, what has
already been tried and rejected, what is currently unverified, and where
knowledge is concentrated. Headlines and ids only, never bodies.

This exists because of a measured failure. Every session building this server
began by reading files to re-derive knowledge that already existed - `open`
reported symbol counts, which tells an agent nothing about what it is walking
into. An agent cannot ask the right question before it knows what is on the
shelf.

### `investigate()`

One call fuses lexical search (FTS5), symbol lookup, code-graph traversal,
memory-graph traversal, anchor status, git history and agit checkpoints, then
returns compact capsules under a token budget instead of file dumps.

Ranking is a static, inspectable formula with per-intent weights. There is no
trained reranker and no embedding model to download, because a fresh local
install has no labeled relevance data to train one on.

### `investigate(action="why", symbol=...)`

Reconstructs why a piece of code exists, as a causal chain rather than a list:

```
decision: Use refresh-token rotation
  --was followed by--> bug_history: Parallel refresh requests invalidate each other
  --was followed by--> failed_attempt: Redis mutex could deadlock during a partition
  --was followed by--> * invariant: All refreshes pass through RefreshCoordinator

may reintroduce: Parallel refresh requests invalidate each other
regression tests: test_parallel_refresh_regression
```

The difference is shape, not retrieval quality. A flat list of five memories
makes an agent reconstruct the story; a chain hands it over. Causal edges are
always **asserted** by an agent via `record(caused_by=[...])`, never inferred
from timestamps - "B was recorded after A" is not "A caused B", and a wrong
causal chain reads as authoritative.

### `record()`

You supply the semantics. The server resolves plain names like
`RefreshCoordinator.acquire`, `auth.py`, or even `the refresh coordinator` to
canonical symbols, anchors each memory to the code, derives the edges you did
not mention (callers, tests, blast radius), and flags contradictions with what
it already knows. One call typically writes 5 memories and 25+ edges.

`contracts_with=[...]` records cross-repository dependencies; the target is
snapshotted at write time, so the contract stays readable even if that
repository is archived or deleted. `caused_by=[...]` links this work into the
causal chain above.

No LLM and no API key are involved. The whole pipeline is deterministic.

### `agit`

Agent-only git history in `.agit/`, separate from the user's real `.git`.
Checkpoint risky work, restore it, and never touch the user's history. Auto-
initialises and adds itself to `.gitignore` on first use.

## What makes it different: anchors that know when they are stale

A memory is not stored at `src/auth/oauth.ts:193`. Line numbers are a rendering
detail. Each memory is attached to a **semantic anchor** holding the symbol
path, an AST path, a content fingerprint (structure + identifiers) and a
skeleton fingerprint (structure only), plus its surrounding context.

When the code changes, a cascade tries to relocate the anchor, cheapest test
first:

| Step | Test | Result |
|---|---|---|
| 1 | Same fingerprint, same place | `ACTIVE` 1.0 |
| 2 | Same fingerprint elsewhere, confirmed by `git blame -C -M` | `ACTIVE` 0.9, moved |
| 3a | Same place, skeleton identical (a rename) | `ACTIVE` 0.8 |
| 3b | Same place, structure changed | **`NEEDS_REVIEW`** |
| 4 | Symbol gone, strong similarity match | `DRIFTED`, re-anchored |
| 5 | Nothing clears the bar | `ORPHANED` - memory kept, never deleted |

Two rules make this trustworthy:

- **Verification fires on the edit that caused the drift**, not on a timer.
- **The cascade can only lower trust, never raise it.** Once an anchor is
  `DRIFTED` or `NEEDS_REVIEW`, only an explicit `memory(action="verify")`
  returns it to `ACTIVE`. Otherwise the next pass would find its freshly
  re-anchored fingerprint matching, report "unchanged", and quietly re-trust a
  memory nobody ever confirmed.

Every memory in `investigate()` output carries its `anchor_status`, and
anything that is not `ACTIVE` comes with an explicit warning. A stale memory is
never rendered as settled fact.

## Problem detection

`investigate()` narrows to a subgraph first, then asks targeted questions of
it - never a workspace-wide scan. What separates these from a linter is that
they are knowledge-aware: a linter sees that a function has no test; only this
graph knows the function is governed by an invariant recorded after a
production incident.

| Finding | Question it answers |
|---|---|
| `stale_knowledge` | which memories drifted from the code they describe |
| `bypassed_wrapper` | is a caller reaching past a coordinator or guard |
| `untested_invariant` | is a governed rule reachable by no test |
| `deprecated_with_callers` | does a deprecated symbol still have live callers |
| `unguarded_equivalent` | does a structurally identical sibling lack the rule |
| `implementation_drifted_from_decision` | did the code diverge from what was decided |
| `knowledge_conflict` | do two memories contradict each other |
| `historical_implementation` | is active knowledge pointing at deleted code |
| `unverifiable_contract` | is a cross-repo dependency currently uncheckable |
| `unreviewed_caller` | did a caller appear after the memory was verified |
| `migration_candidate` | did code plausibly move somewhere the cascade would not follow |

A failing detector never breaks the search: a diagnostic enhances the answer,
it is not a precondition for one.

## Storage

Central, configurable, with one logical store per repository. The split is by
what the data *is*: knowledge that must outlive the checkout goes central,
state that only means something relative to this working tree stays in the repo.

```
%LOCALAPPDATA%\InfiniteCode\          (Windows)
$XDG_DATA_HOME/infinite-code/         (Linux)
~/Library/Application Support/InfiniteCode/  (macOS)

  catalog.db                repositories, aliases, checkouts, cross-repo edges
  data/repos/<id>/repo.db   DURABLE   code graph, memories, anchors, events
  cache/repos/<id>/         REBUILDABLE  safe to delete at any time

<repo>/.agit/               agent git, gitignored
<repo>/.icn.toml            optional, committed, tiny
```

Override the root with `INFINITE_CODE_HOME`.

Repository identity is never the path and never the remote URL - both are
mutable. It is derived from the root commit hash, an optional committed project
id, and normalised remotes, so moving a clone or running `git remote set-url`
reattaches to existing knowledge instead of starting a fresh memory universe.
Three worktrees of one repository share one set of memories.

## Nothing is ever destroyed

- Deleted symbols become **tombstones** with their last known path and the
  commit that removed them, so "this decision was implemented by something that
  no longer exists" stays answerable.
- Edges carry `valid_from_commit` / `valid_until_commit` and become
  `HISTORICAL` rather than disappearing.
- A vanished checkout is `MISSING`, an unmounted drive is `OFFLINE`, and
  neither deletes anything.
- Corrections version the previous text; supersession keeps both memories and
  the link between them.
- An agent cannot rewrite a human-authored memory - it must supersede it,
  leaving the disagreement visible.
- `purge` is the only destructive operation and requires `confirm=True`.

Lookups go through a resolver that never raises: "cannot currently resolve" is
returned as data with whatever was last known.

## Concurrency

Several server processes per repository is the normal case - one per editor
window, one per terminal. WAL mode, `BEGIN IMMEDIATE` with retry, short write
transactions, and a heartbeat lease so exactly one process indexes at a time.
The lease is reclaimable, because the holder can be killed at any moment and
nobody runs cleanup.

## Degradation

Every optional capability degrades rather than blocking.

| Missing | Behaviour |
|---|---|
| Embeddings | Cascade step 5 skipped. Retrieval is FTS + symbol + graph. |
| LLM / API key | Never needed. Compilation is deterministic. |
| Git | Path-hash identity, marked `weak`. `agit` unavailable. |
| Cold index | Immediate response, `index_state: "partial"`, background completion. |

## Tests

```bash
python -m pytest
```

Includes a live MCP test that spawns the real server over stdio and drives a
full agent workflow through the wire protocol, and a dirty-worktree fixture
harness that asserts cascade behaviour on **uncommitted** edits - reformat,
rename, body change, cross-file move, delete, and weak migration. That regime
is unvalidated by the published literature, which only ever measures post-hoc
commit-history mining, so it is measured here directly.

The live test earns its keep. It found a bug that in-process testing cannot
see: subprocess calls were inheriting the server's stdin, which *is* the MCP
protocol pipe. Git blocked on it for its full 20-second timeout on every tool
call, and could swallow protocol bytes. Fixing it took tool latency from 20s to
0.2s.

## Design

The reasoning behind each decision lives next to the code it governs: every
module's docstring states what it does and, more importantly, which failure it
exists to prevent. `anchors.py` explains why the cascade may only lower trust,
`briefing.py` why `open` volunteers a summary, `causal.py` why causality is
asserted and never inferred.

Planning notes are kept locally and are not part of the shipped artifact.
