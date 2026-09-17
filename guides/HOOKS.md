# Hooks, handoffs, feedback and promoted rules

ICN stores knowledge anchored to exact files and symbols: rules, warnings,
rejected approaches. Tools like `investigate()` only help when an agent thinks
to ask. Hooks remove that dependency: Claude Code and Codex call ICN at fixed
moments, and ICN puts the few relevant facts in front of the model before it
acts.

This guide covers setup, what each piece does, how to check it works, and how
to fix it when it does not.

## Contents

1. [What you get](#1-what-you-get)
2. [Requirements](#2-requirements)
3. [Install](#3-install)
4. [Check that it works](#4-check-that-it-works)
5. [What the hooks deliver](#5-what-the-hooks-deliver)
6. [Handoffs](#6-handoffs)
7. [Feedback on memories](#7-feedback-on-memories)
8. [Reinforcement instead of duplicates](#8-reinforcement-instead-of-duplicates)
9. [Promoting rules into CLAUDE.md and AGENTS.md](#9-promoting-rules-into-claudemd-and-agentsmd)
10. [Design rules the hooks follow](#10-design-rules-the-hooks-follow)
11. [Troubleshooting](#11-troubleshooting)
12. [Uninstall](#12-uninstall)
13. [Reference](#13-reference)

## 1. What you get

| Moment | What the model receives |
|---|---|
| Session starts | The handoff the previous session left (once), up to five of the highest-standing rules, and counts of what needs attention (memories to review, experiment runs without a verdict). |
| Before a tool reads or edits a file | The rules, warnings, contracts, security notes, known bugs and rejected attempts anchored to that file, at most four, never repeated in the same session. |

And three things an agent does with ICN's `memory` tool:

| Action | Purpose |
|---|---|
| `handoff` | Leave "where I left off" for the next session, in any client. |
| `feedback` | Vote a delivered memory helpful, not helpful, stale or wrong. |
| `rules_*` | Promote a settled rule into this repository's CLAUDE.md and AGENTS.md. |

## 2. Requirements

- ICN installed for the Python the hooks will run (`py -3.12 -c "import icn"` must work).
- The repository opened in ICN at least once (`workspace(action="open")` from
  any client). Hooks look repositories up in ICN's catalog and stay silent for
  directories ICN has never opened.
- Claude Code 2.1 or later, and/or Codex CLI 0.154 or later (hooks feature
  enabled, which is the default: `codex features list` shows `hooks stable true`).

## 3. Install

From the repository root:

```sh
py -3.12 -m icn install --codex
```

Drop `--codex` if you only use Claude Code. The command is idempotent: run it
again after upgrading ICN or moving Python, and it replaces its own entries
without duplicating them or touching anyone else's.

What it writes:

| File | Scope | Contents |
|---|---|---|
| `.claude/skills/icn-workflow/SKILL.md` | this repository | The ICN workflow for the agent, including handoff and feedback. |
| `.claude/settings.json` | this repository | `SessionStart` and `PreToolUse` hooks for Claude Code. Existing settings and hooks are merged, never replaced; a malformed file is refused rather than overwritten. |
| `~/.codex/hooks.json` (or `$CODEX_HOME/hooks.json`) | every repository Codex opens | The same two hooks for Codex. Safe to be global: the hook resolves the repository from the session's working directory and prints nothing where ICN has no knowledge. |

The installed commands use the absolute path of the Python that ran the
installer, for example:

```text
Claude Code:  "C:\...\Python312\python.exe" -m icn hook pre-tool-use --agent claude
Codex:        & "C:\...\Python312\python.exe" -m icn hook pre-tool-use --agent codex
```

The `&` for Codex on Windows is deliberate (see [Troubleshooting](#11-troubleshooting)).

Then:

1. **Claude Code:** start a new session. Hooks in `.claude/settings.json` are
   picked up without further steps.
2. **Codex:** start `codex` once in a terminal. It lists the new hooks under
   "Hooks need review"; choose to trust them. Until then Codex does not run them.
   (For scripted runs, `codex exec --dangerously-bypass-hook-trust` skips the
   prompt; only use it for hooks you have read.)

## 4. Check that it works

### 4.1 Without any client

Pipe the same JSON a client sends into the hook:

```sh
echo '{"session_id":"check","cwd":"C:/path/to/repo"}' | py -3.12 -m icn hook session-start
echo '{"session_id":"check","cwd":"C:/path/to/repo","tool_name":"Read","tool_input":{"file_path":"C:/path/to/repo/src/app.py"}}' | py -3.12 -m icn hook pre-tool-use
```

Expected: one JSON line, `{"hookSpecificOutput": {"hookEventName": ..., "additionalContext": "..."}}`,
or no output at all when there is nothing to say. Add `ICN_HOOK_TIMING=1` to
print the time taken to stderr; typical is 30-80 ms.

Running the same `pre-tool-use` line twice with the same `session_id` prints
nothing the second time. That is the once-per-session rule working.

### 4.2 In Claude Code

Start a session in the repository and ask the agent to read a file you know
has knowledge, then ask it to quote any ICN context it received. In the
transcript the context appears as `PreToolUse:Read hook additional context`.

A verified run (Claude Code 2.1.273): the session started with a handoff and
the rule "settle() must be idempotent"; reading `billing.py` delivered "an
in-memory set was tried and rejected"; asked to deduplicate with an in-memory
set, the agent declined and cited the rejected attempt.

### 4.3 In Codex

```sh
codex exec -C C:/path/to/repo "Read src/app.py and quote any ICN context you were given"
```

The run log shows `hook: SessionStart Completed` and `hook: PreToolUse Completed`.
`Failed` means the command did not run (see Troubleshooting).

Verified on Codex 0.154: the installed `SessionStart` hook runs and claims the
handoff under the Codex session id. Model-side delivery of `PreToolUse`
context in Codex has not been observed live yet; the payload formats (the
`apply_patch` patch text, shell command lists) are covered by tests.

### 4.4 See what was delivered

Every delivery is logged in the repository's store, in the `hook_deliveries`
table (session, memory, event, file, agent, time), so you can compare what was
shown against the feedback those memories later received.

## 5. What the hooks deliver

### Session start

```text
ICN holds recorded knowledge for this repository (billing). Call workspace(action='open') ...

Handoff from the previous session (codex, 2026-09-16 16:33 UTC, branch main):
Half-way through adding an idempotency_key column to settle().
Next steps:
  - write the migration
  - add a unique index

Highest-standing rules here:
- [high] billing.py: settle() must be idempotent: a retried payment webhook must never charge an invoice twice. (mem_370495d41f33)

Needs attention: 3 memories need review against changed code (workspace(action='open') lists them).
```

- Rules shown: invariants, security notes and contracts of high or critical
  severity, whose anchors all still match the code, ordered by severity, then
  evidence count, then helpful votes. Rules already promoted into CLAUDE.md are
  skipped, since the client loads that file anyway.
- Budget: 2,600 characters.

### Before a tool touches a file

```text
ICN knowledge anchored to billing.py (recorded by earlier sessions):
- RULE [high] billing.py settle: settle() must be idempotent: ... (mem_370495d41f33)
- TRIED AND REJECTED [medium] billing.py settle: Deduplicating retries in memory was tried and rejected: ... (mem_041d469ecb7a)
Respect these while working on this file. If one is out of date, say so and call memory(action='feedback', ...)
```

Which tool calls count as touching a file:

| Client | Tool | File taken from |
|---|---|---|
| Claude Code | Read, Edit, Write, MultiEdit, NotebookEdit | `file_path` / `notebook_path` |
| Claude Code | Bash | tokens in the command that name a real file in the repository (`cat src/x.py`) |
| Codex | `apply_patch` | `*** Update File:` / `*** Add File:` / `*** Delete File:` lines |
| Codex | shell (`shell`, `exec_command`, `local_shell`, ...) | tokens naming a real file, including list-form commands |

Selection rules:

- Only kinds worth interrupting for: invariant, warning, contract, security,
  bug, failed_attempt. Decisions and rationale stay in `investigate()`.
- At most four memories and 1,400 characters, most severe first.
- Never a memory already delivered in this session (at start or on an earlier file).
- A memory whose code changed since it was recorded is still delivered,
  labelled `unverified: code changed since recorded`. A rule matters most
  exactly when its code is being changed, so hiding it would be backwards.
- A memory whose code is gone (ORPHANED), voted wrong, or voted unhelpful
  repeatedly is not delivered. Search still finds it.
- A memory recorded together with many files (more than three) is only
  delivered for the files its own text names. `record()` anchors every memory
  in a call to every file in that call, so without this a note about
  `hooks.run` would appear whenever `compiler.py` is read. Ordinary words only
  count when written as code: `run(`, `hooks.run`, `` `run` ``.

## 6. Handoffs

A handoff is "where I left off", written when a session stops mid-task and
received once by the next session in the repository, in any client.

Write one (the agent does this through MCP):

```python
memory(action="handoff",
       body="Half-way through adding an idempotency_key column to settle(). Migration not written.",
       next_steps=["write the migration", "add a unique index on idempotency_key"],
       open_questions=["backfill old invoices?"],
       files=["billing.py"],
       actor="claude")
```

How it is received:

- The next session's `SessionStart` hook claims it and shows it.
- Without hooks, the next `workspace(action="open")` claims it and returns it
  under `handoff`, with `next` pointing at it.
- Claiming is atomic: whichever comes first gets it, and it is never shown again.
- Writing a new handoff supersedes any older one still waiting, so the next
  session sees the latest state rather than a pile.

Inspect and withdraw:

```python
memory(action="handoff_list")                          # status, claimed_by, without claiming
memory(action="handoff_cancel", memory_id="hof_...")   # withdraw one still open
```

## 7. Feedback on memories

Access counts cannot tell "this answered the question" from "this wasted a
read". Delivered memories ask for a vote:

```python
memory(action="feedback", memory_id="mem_...", signal="helpful")
memory(action="feedback", memory_id="mem_...", signal="not_helpful")
memory(action="feedback", memory_id="mem_...", signal="stale", reason="settle moved to payments/")
memory(action="feedback", memory_id="mem_...", signal="wrong", reason="keyed by payment intent, not invoice")
```

| Signal | Effect |
|---|---|
| `helpful` | Counted; confidence +0.05; ranks higher in search and rule promotion. |
| `not_helpful` | Counted; ranks lower. Two or more, outnumbering helpful votes two to one, stop hook delivery. |
| `stale` (reason required) | Anchors set to NEEDS_REVIEW; blocks rule promotion until verified. |
| `wrong` (reason required) | As stale, and confidence drops to 0.3, below the delivery floor. |

Nothing is deleted. `memory(action="verify")` restores a memory confirmed to
still hold; `memory(action="correct")` or `supersede` fixes one that is wrong.
Correcting a memory changes what hooks deliver immediately.

## 8. Reinforcement instead of duplicates

`record()` compares each claim with existing memories of the same kind:

- **Same claim about the same code** (80% word overlap or more, a shared
  anchor): no copy is made. The existing memory's `evidence_count` goes up, its
  review flags clear, and the reply lists it under `memories_reinforced`.
- **Same claim about different code**, or a close match (50% or more): a new
  memory is made, carrying `similar_to` with the match, so you can check it is
  not the same rule.
- A summary and a decision in the same `record()` call never reinforce each other.

Record what you confirmed even when it is not new: that is what makes a rule settled.

## 9. Promoting rules into CLAUDE.md and AGENTS.md

A small number of rules deserve to be loaded on every task, not only when a
file is touched. Promotion writes them into a managed block in this
repository's `CLAUDE.md` and `AGENTS.md`. The file is a scarce budget: models
follow a limited number of instructions reliably, and every extra rule weakens
the others. So promotion is manual and capped.

```sh
icn rules recommend               # ranked candidates, read-only
icn rules approve mem_...         # promote one; --text "better wording" to reword
icn rules edit mem_... --text "..."
icn rules remove mem_...          # demote; the memory is unchanged
icn rules list                    # what is promoted, and which rules went stale
```

(The same through MCP: `memory(action="rules_recommend" | "rules_approve" | ...)`.
Agents should only approve on the user's say-so.)

A memory is eligible only if all of these hold:

- kind is invariant, warning, contract, security or convention;
- it is anchored to real files or symbols in this repository, and those anchors still match the code;
- no unresolved contradiction, and no stale or wrong vote since it was last verified;
- its first sentence is imperative (must, never, always, only, ...) and at most 240 characters.

Candidates are scored by severity, evidence count, helpful votes, how much code
depends on the anchored symbols (callers, files), and past use. `recommend`
also lists high scorers blocked by one thing, usually "verify it first".

Each promoted line names the code it governs:

```markdown
<!-- icn:rules:start -->
## Rules settled in this codebase

Promoted from verified ICN knowledge, each anchored to the code it governs. ...

- In `src/icn/workspace.py` `ensure_indexed`: Deferring a file under budget must always set report['truncated']. (mem_dae11dbd0f9b)
<!-- icn:rules:end -->
```

Guarantees:

- Text outside the markers is never read into the block or modified.
- Both files are updated when both exist; if neither exists, both are created.
- Removing the last rule removes the block (and a file that only held the block).
- At the cap (12), `approve` refuses and names the weakest promoted rule to remove first.
- `list` marks a promoted rule `STALE` when its code changed or it was disputed, so it can be re-verified or removed.

## 10. Design rules the hooks follow

Each comes from a failure documented in another memory system:

| Rule | Why |
|---|---|
| Fail open: any error exits 0 with no output, logged to `<ICN home>/logs/hooks.log`. | A broken hook must never block a session. |
| Targeted: only knowledge anchored to the file being touched. | Broad context injection at every session start used up users' quotas in agentmemory and was turned off by default. |
| Capped per injection, once per session. | Repeated context is paid for on every turn after it. |
| Light: no parser, index or embedding model is imported. | The hook runs before every tool call. |
| No LLM calls. | Zero cost, and deterministic output. |

## 11. Troubleshooting

**Nothing is ever delivered.**

1. Pipe a payload in by hand (section 4.1). If that prints nothing:
   - Was the repository opened in ICN from this path? `icn --where` shows the
     store; the hook matches the session's `cwd` against checkouts in its catalog.
   - Does the file have knowledge? `investigate(query="<file name>")`.
   - Was it already delivered in this session? Try a new `session_id`.
   - Different store? The hook uses `INFINITE_CODE_HOME` if set; the client
     must launch hooks with the same value as the MCP server.
2. If it prints context by hand but not in the client, the client is not running
   the command: see the next two entries.

**Codex shows `hook: SessionStart Failed`.** On Windows, Codex runs hook commands
through PowerShell. A command starting with a quoted path (`"C:\...\python.exe" -m ...`)
is a string expression there, not a call, and fails. The installer writes
`& "C:\...\python.exe" ...` for Codex. If you edited the file by hand, restore
the `&` or rerun `py -3.12 -m icn install --codex`.

**Codex never runs the hooks.** It has not been told to trust them. Start
`codex` interactively once and accept them, and check `codex features list`
shows `hooks` enabled.

**Claude Code does not run them.** Hooks in `.claude/settings.json` only load
when the session's project is that repository, and not when it was started
with `--setting-sources` excluding `project`. Check the file contains the
`icn hook` entries; `/hooks` in Claude Code lists what is active.

**Python moved or was upgraded.** The hooks hold an absolute interpreter path.
Rerun the installer with the new Python.

**Something went wrong silently.** Look at `<ICN home>/logs/hooks.log`: one
JSON line per failure with the event, client and error.

**Too much or wrong context.** Vote: `not_helpful` or `wrong` with a reason.
A memory that names the wrong file usually came from a `record()` listing many
files; `correct` it to name the right symbol.

## 12. Uninstall

- Claude Code: delete the `SessionStart` and `PreToolUse` entries whose command
  contains `-m icn hook` from `.claude/settings.json`.
- Codex: delete the same entries from `~/.codex/hooks.json`.
- Promoted rules: `icn rules remove <id>` for each, or delete the block between
  the `icn:rules` markers.

Stored knowledge, handoffs and feedback are unaffected.

## 13. Reference

Command line:

```text
icn hook session-start [--agent claude|codex]   reads the client's JSON on stdin
icn hook pre-tool-use  [--agent claude|codex]
icn install [--root PATH] [--no-hook] [--codex] [--json]
icn rules recommend|approve|edit|remove|list [memory_id] [--text T] [--root PATH] [--json]
```

Environment:

| Variable | Effect |
|---|---|
| `INFINITE_CODE_HOME` | ICN store location; must match the MCP server's. |
| `CODEX_HOME` | Where the installer writes Codex's `hooks.json`. |
| `ICN_HOOK_TIMING=1` | Hook prints its duration to stderr. |

Limits (in `src/icn/hooks.py` and `src/icn/rules.py`):

| Constant | Value |
|---|---|
| `PRE_TOOL_BUDGET` | 1,400 characters |
| `PRE_TOOL_MAX` | 4 memories |
| `SESSION_BUDGET` | 2,600 characters |
| `relevance.BROAD` | more than 3 files counts as broadly anchored |
| `rules.DEFAULT_CAP` | 12 promoted rules |
| `rules.MAX_RULE_CHARS` | 240 characters |
| `feedback.TRUST_FLOOR` | confidence below 0.4 is not delivered |

Storage (per-repository `repo.db`, schema version 10): `memory_evidence`,
`memory_feedback`, `handoffs`, `promoted_rules`, `hook_deliveries`, and the
memory columns `claim`, `evidence_count`, `helpful_count`, `unhelpful_count`.
