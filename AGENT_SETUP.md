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

## 2. Add the global agent instruction

The MCP registration makes ICN callable. This instruction makes agents proactively discover and use it.

### Codex: `~/.codex/AGENTS.md`

```markdown
## Infinite Code Next (ICN)
- For any task involving an existing codebase, use ICN. At the beginning of a session and whenever you switch repositories, discover deferred tools if necessary and call `mcp__icn__workspace(action="open", root=<repo>)`.
- Before investigating, diagnosing, designing, or modifying code, call `mcp__icn__investigate(query=<task>, intent=<intent>, root=<repo>)`. Treat memories whose anchor status is not `ACTIVE` as unverified leads.
- After verified findings or changes, call `mcp__icn__record` with the decision, failure prevented, affected files/symbols, invariants, warnings, failed attempts, contracts, and tests. Skip investigation and recording only for purely mechanical actions such as correcting a typo or running an explicitly requested command.
- If `mcp__icn__*` is not visible, search the available/deferred tool catalogue and load it. If it still cannot be loaded, explicitly report that ICN is unavailable and continue with the best evidence. Never silently skip ICN or claim it was used when it was not.
```

### Claude Code: `~/.claude/CLAUDE.md`

```markdown
## Infinite Code Next (ICN)
- For any task involving an existing codebase, use ICN. At the beginning of a session and whenever you switch repositories, ensure the `icn` MCP server and its tools are loaded and call `workspace(action="open", root=<repo>)`.
- Before investigating, diagnosing, designing, or modifying code, call `investigate(query=<task>, intent=<intent>, root=<repo>)`. Treat memories whose anchor status is not `ACTIVE` as unverified leads.
- After verified findings or changes, call `record` with the decision, failure prevented, affected files/symbols, invariants, warnings, failed attempts, contracts, and tests. Skip investigation and recording only for purely mechanical actions such as correcting a typo or running an explicitly requested command.
- If ICN is not loaded, try to reconnect or load the configured `icn` MCP server. If it remains unavailable, explicitly report that fact and continue with the best evidence. Never silently skip ICN or claim it was used when it was not.
```

Use the same block in a repository-local `AGENTS.md` or `CLAUDE.md` when global configuration is not permitted.

## 3. Expected agent workflow

1. Open the current repository with `workspace(action="open")`.
2. Investigate the task before broad file exploration or significant changes.
3. Verify memories against current code, especially when their anchor status is not `ACTIVE`.
4. Perform and test the work.
5. Record durable decisions, warnings, failed attempts, contracts, and test evidence.

For a bounded task spanning several repositories, pass explicit `roots=[...]` to `investigate`. Use `cross_repos=True` when following previously recorded contracts.

## What the prompt cannot do

An instruction cannot expose a server that the MCP client failed to register or start. `icn doctor`, a client restart, and a valid client configuration provide visibility. The global prompt ensures the agent discovers deferred tools and does not silently bypass ICN once the client makes it available.

