"""Targeted diagnostics over a narrowed subgraph.

PLAN.md "search should actively look for problems". Each detector answers one
of the plan's questions, and each is knowledge-aware: a linter can see that a
function has no test, but only this graph knows the function is governed by an
invariant recorded after a production incident.
"""

from __future__ import annotations

from icn import compiler, diagnostics
from icn import search as search_mod
from icn import workspace as ws_mod
from icn.db import one, rows


def rec(ws, **payload):
    return compiler.record_event(ws.store, ws.catalog, ws.repo_id, ws.root, ws.commit, payload)


def active_symbol_ids(ws) -> list[str]:
    return [r["symbol_id"] for r in rows(ws.store.execute(
        "SELECT symbol_id FROM symbols WHERE status='ACTIVE'"))]


# ------------------------------------------------------------------- bypass

BYPASS_SOURCE = '''
class TokenCoordinator:
    def issue(self, session_id):
        return mint_token(session_id)


def mint_token(session_id):
    return session_id


def legacy_login(session_id):
    return mint_token(session_id)


def admin_backdoor(session_id):
    return mint_token(session_id)
'''


def test_a_caller_bypassing_a_guarded_wrapper_is_reported(workspace, project):
    """"Is there a caller bypassing the expected wrapper?" - PLAN.md."""
    project.write("tokens.py", BYPASS_SOURCE)
    ws_mod.ensure_indexed(workspace)

    rec(workspace, kind="decision", summary="all tokens go through the coordinator",
        warnings=["Every token must be issued via TokenCoordinator.issue"],
        symbols=["TokenCoordinator.issue"])

    findings = diagnostics.bypassed_wrappers(workspace.store, active_symbol_ids(workspace))
    assert findings, "expected a bypass finding"
    detail = findings[0]["detail"]
    assert "mint_token" in detail
    assert "TokenCoordinator" in detail
    assert findings[0]["confidence"] < 1.0, "an inferred finding must not claim certainty"


# ------------------------------------------------------------ untested rule

def test_an_invariant_with_no_test_reaching_it_is_reported(workspace, project):
    """"Are there callers without tests?" - scoped to governed code."""
    project.write("billing.py", "def settle(invoice):\n    return invoice\n")
    project.write("caller.py", "from billing import settle\n\n\n"
                               "def run(x):\n    return settle(x)\n")
    ws_mod.ensure_indexed(workspace)

    rec(workspace, kind="decision", summary="settlement must be idempotent",
        invariants=["settle must be idempotent"], symbols=["settle"])

    findings = diagnostics.untested_callers(workspace.store, active_symbol_ids(workspace))
    assert any("settle" in f["detail"] for f in findings)


def test_recording_a_test_clears_the_untested_finding(workspace, project):
    """A GUARDED_BY edge is the signal that the invariant is covered."""
    project.write("billing.py", "def settle(invoice):\n    return invoice\n")
    ws_mod.ensure_indexed(workspace)

    rec(workspace, kind="decision", summary="settlement must be idempotent",
        invariants=["settle must be idempotent"], symbols=["settle"],
        tests=["test_settle_is_idempotent"])

    findings = diagnostics.untested_callers(workspace.store, active_symbol_ids(workspace))
    assert not any("settle" in f["detail"] for f in findings), \
        "a recorded test must clear the finding"


# --------------------------------------------------------------- deprecated

def test_a_deprecated_symbol_with_live_callers_is_reported(workspace, project):
    """"Does a deprecated symbol still have live callers?" - PLAN.md.

    Deprecation is read from recorded knowledge, which is more current than a
    decorator someone added years ago.
    """
    project.write("old.py", "def old_charge(x):\n    return x\n")
    project.write("newcaller.py", "from old import old_charge\n\n\n"
                                  "def checkout(x):\n    return old_charge(x)\n")
    ws_mod.ensure_indexed(workspace)

    rec(workspace, kind="decision", summary="old_charge is deprecated",
        warnings=["old_charge is deprecated, use charge_v2 instead"],
        symbols=["old_charge"])

    findings = diagnostics.deprecated_with_live_callers(
        workspace.store, active_symbol_ids(workspace))
    assert findings
    assert "old_charge" in findings[0]["detail"]
    assert "checkout" in findings[0]["detail"]


# ------------------------------------------------------ unguarded equivalent

EQUIVALENT_SOURCE = '''
def handle_alpha(payload, context):
    validated = check(payload)
    stored = persist(validated, context)
    return stored


def handle_beta(payload, context):
    validated = check(payload)
    stored = persist(validated, context)
    return stored


def check(p):
    return p


def persist(v, c):
    return v
'''


def test_a_structurally_identical_sibling_missing_the_rule_is_reported(workspace, project):
    """Covers two plan questions: diverging implementations of one invariant,
    and a warning attached to one implementation but not its equivalent."""
    project.write("handlers.py", EQUIVALENT_SOURCE)
    ws_mod.ensure_indexed(workspace)

    alpha = one(workspace.store.execute(
        "SELECT skeleton_fingerprint FROM symbols WHERE symbol_path='handle_alpha'"))
    beta = one(workspace.store.execute(
        "SELECT skeleton_fingerprint FROM symbols WHERE symbol_path='handle_beta'"))
    assert alpha["skeleton_fingerprint"] == beta["skeleton_fingerprint"], \
        "fixture must produce structurally identical siblings"

    rec(workspace, kind="decision", summary="alpha must validate before persisting",
        invariants=["handle_alpha must validate payloads before persisting"],
        symbols=["handle_alpha"])

    findings = diagnostics.diverging_implementations(
        workspace.store, active_symbol_ids(workspace))
    assert findings
    assert "handle_beta" in findings[0]["detail"]
    assert findings[0]["confidence"] < 0.7, "structural equivalence is a hint, not proof"


# ------------------------------------------------------------ decision drift

def test_a_decision_whose_code_changed_is_reported(workspace, project):
    """"Did the implementation diverge from a documented decision?" - PLAN.md."""
    rec(workspace, kind="decision", summary="refresh is serialized per session",
        decisions=["refresh_session serializes per session id"],
        symbols=["refresh_session"])

    project.write("auth.py", project.read("auth.py").replace(
        "def refresh_session(session_id):\n"
        "    coordinator = RefreshCoordinator()\n"
        "    lock = coordinator.acquire(session_id)\n"
        "    return rotate_token(lock)",
        "def refresh_session(session_id):\n"
        "    return rotate_token(session_id)  # serialization dropped"))
    ws_mod.ensure_indexed(workspace)

    findings = diagnostics.drifted_from_decision(
        workspace.store, active_symbol_ids(workspace))
    assert findings
    assert "refresh" in findings[0]["detail"]
    assert findings[0]["severity"] == "high"


# ------------------------------------------------------------------ wiring

def test_all_detectors_are_reachable_from_investigate(workspace, project):
    project.write("tokens.py", BYPASS_SOURCE)
    ws_mod.ensure_indexed(workspace)
    rec(workspace, kind="decision", summary="all tokens go through the coordinator",
        warnings=["Every token must be issued via TokenCoordinator.issue"],
        symbols=["TokenCoordinator.issue"])

    result = search_mod.investigate(workspace.store, workspace.catalog, workspace.root,
                                    "token coordinator issue", intent="audit",
                                    commit=workspace.commit)
    kinds = {p["kind"] for p in result["problems"]}
    assert "bypassed_wrapper" in kinds


def test_a_failing_detector_never_breaks_the_search(workspace, monkeypatch):
    """A diagnostic enhances the answer; it is not a precondition for one."""
    import sqlite3

    def explode(conn, symbol_ids):
        raise sqlite3.OperationalError("simulated failure")

    monkeypatch.setattr(diagnostics, "ALL_DETECTORS", (explode,))
    assert diagnostics.run_all(workspace.store, active_symbol_ids(workspace)) == []


def test_detectors_are_bounded_to_the_given_subgraph(workspace, project):
    """Never scan the workspace. An empty candidate set means no work."""
    project.write("tokens.py", BYPASS_SOURCE)
    ws_mod.ensure_indexed(workspace)
    assert diagnostics.run_all(workspace.store, []) == []
