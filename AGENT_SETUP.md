# ICN agent setup

This guide makes Infinite Code Next visible and routinely used by coding agents. It covers both parts of the setup: registering the MCP server with the client, and instructing the agent to use it.

## 1. Install and register ICN

```bash
pip install -e .
```

For Codex, add this to `~/.codex/config.toml`:

```toml
[mcp_servers.icn]
command = "infinite-code-next"
args = []
```

For Claude Code:

```bash
claude mcp add icn -- infinite-code-next
```

Restart the client, then verify the real MCP handshake, tool catalogue, repository resolution, and parser health:

```bash
icn doctor --client codex --root /path/to/repository
```

Do not consider setup complete until the final line says `READY`.

## 2. Install the skill and hook (Claude Code)

The MCP registration makes ICN callable. It does not make it called: the server can be registered, `doctor` can say `READY`, and the agent can still never touch it because nothing told it to.

```bash
icn install --root /path/to/repository
```

That writes two things into the project:

- `.claude/skills/icn-workflow/SKILL.md` - the full workflow, loaded when the agent needs it.
- a `SessionStart` hook in `.claude/settings.json` - one line so the agent knows the skill exists.

It is idempotent, and it merges into an existing `settings.json` rather than replacing it. A `settings.json` it cannot parse is refused, not overwritten. Use `--no-hook` to write only the skill.

Restart the client afterwards so both load.

## 3. Approve the read-only tools once

ICN is designed to be called constantly: on every session start, before every
investigation, and before every change. A client that prompts for approval on
each of those calls defeats that design. The agent either waits on a human for
every question it asks, or it learns to avoid the tool and falls back to
grepping, which is the behaviour ICN exists to replace.

Pre-approve the read-only tools. In Claude Code, `.claude/settings.json`:

```json
{
  "permissions": {
    "allow": [
      "mcp__icn__workspace",
      "mcp__icn__investigate",
      "mcp__icn__graph",
      "mcp__icn__memory",
      "mcp__icn__paper",
      "mcp__icn__record"
    ]
  }
}
```

Other clients express the same idea differently. Codex uses trusted-tool
configuration in `~/.codex/config.toml`; Cursor and Windsurf expose per-server
auto-approval in their MCP settings panels. The rule is the same everywhere:
approve the tools that only read, and leave anything that writes to your files
prompting.

`record` is on the list because it writes only to ICN's own knowledge store,
never to your source. `agit` is deliberately **not** on the list: it commits to
`.agit/`, so it changes files on disk and should stay behind a prompt.

A rejected or unapproved call is not a slow call. It is a call that never ran,
and it will sit in the prompt for as long as nobody answers it. If ICN appears
to hang, check for a pending approval before suspecting the indexer.

## 4. Add the global agent instruction

Steps 2 and 3 cover Claude Code per project. This covers every client, and is what you want if you would rather configure ICN once globally than per repository.

### Codex: `~/.codex/AGENTS.md`

```markdown
## Infinite Code Next (ICN)
- For any task involving an existing codebase, use ICN. At the beginning of a session and whenever you switch repositories, discover deferred tools if necessary and call `mcp__icn__workspace(action="open", root=<repo>)`.
- Before investigating, diagnosing, designing, or modifying code, call `mcp__icn__investigate(query=<task>, intent=<intent>, root=<repo>)`. Treat memories whose anchor status is not `ACTIVE` as unverified leads.
- After verified findings or changes, call `mcp__icn__record` with the decision, failure prevented, affected files/symbols, invariants, warnings, failed attempts, contracts, and tests. Skip investigation and recording only for purely mechanical actions such as correcting a typo or running an explicitly requested command.
- Checkpoint your own work with `mcp__icn__agit(action="commit", message=...)` before risky edits or broad refactors, and roll back with `mcp__icn__agit(action="restore", paths=[...])`. It commits to `.agit/`, never the user's `.git`, so it is not a substitute for asking before a real commit.
- If `mcp__icn__*` is not visible, search the available/deferred tool catalogue and load it. If it still cannot be loaded, explicitly report that ICN is unavailable and continue with the best evidence. Never silently skip ICN or claim it was used when it was not.
```

### Claude Code: `~/.claude/CLAUDE.md`

```markdown
## Infinite Code Next (ICN)
- For any task involving an existing codebase, use ICN. At the beginning of a session and whenever you switch repositories, ensure the `icn` MCP server and its tools are loaded and call `workspace(action="open", root=<repo>)`.
- Before investigating, diagnosing, designing, or modifying code, call `investigate(query=<task>, intent=<intent>, root=<repo>)`. Treat memories whose anchor status is not `ACTIVE` as unverified leads.
- After verified findings or changes, call `record` with the decision, failure prevented, affected files/symbols, invariants, warnings, failed attempts, contracts, and tests. Skip investigation and recording only for purely mechanical actions such as correcting a typo or running an explicitly requested command.
- Checkpoint your own work with `agit(action="commit", message=...)` before risky edits or broad refactors, and roll back with `agit(action="restore", paths=[...])`. It commits to `.agit/`, never the user's `.git`, so it is not a substitute for asking before a real commit.
- If ICN is not loaded, try to reconnect or load the configured `icn` MCP server. If it remains unavailable, explicitly report that fact and continue with the best evidence. Never silently skip ICN or claim it was used when it was not.
```

Use the same block in a repository-local `AGENTS.md` or `CLAUDE.md` when global configuration is not permitted.

## 5. Expected agent workflow

1. Open the current repository with `workspace(action="open")`.
2. Investigate the task before broad file exploration or significant changes.
3. Verify memories against current code, especially when their anchor status is not `ACTIVE`.
4. Check the blast radius with `graph(action="impact", target=<symbol>)` before changing anything other code depends on. Read the result's `epistemic` field first: `lower-bound` means callers exist that the answer does not list, and `causes.ambiguous_call_sites` counts them. An empty result is not proof that nothing calls the symbol.
5. Checkpoint with `agit(action="commit")` before risky edits or broad refactors; `agit(action="restore")` rolls back without touching the user's `.git`.
6. Perform and test the work.
7. Record durable decisions, warnings, failed attempts, contracts, and test evidence.

For a bounded task spanning several repositories, pass explicit `roots=[...]` to `investigate`. Use `cross_repos=True` when following previously recorded contracts.

## What the prompt cannot do

An instruction cannot expose a server that the MCP client failed to register or start. `icn doctor`, a client restart, and a valid client configuration provide visibility. The global prompt ensures the agent discovers deferred tools and does not silently bypass ICN once the client makes it available.

