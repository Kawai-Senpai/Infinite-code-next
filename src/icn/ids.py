"""Permanent entity IDs.

PLAN.md section 2: every important entity gets a permanent ID that is never
reused, and is never a path and never a git remote URL. Both of those are
mutable; the identity must not be.

IDs are opaque. Nothing may parse anything out of them except the prefix, which
exists so a bare ID is debuggable and so the resolver can route a lookup to the
right table without a round-trip.
"""

from __future__ import annotations

import uuid

# Prefix -> what it identifies. The resolver keys off this.
REPO = "repo"
CHECKOUT = "ck"
FILE = "file"
SYMBOL = "sym"
MEMORY = "mem"
ANCHOR = "anc"
EVENT = "evt"
EDGE = "edg"
INVESTIGATION = "inv"


def new_id(prefix: str) -> str:
    """Mint a fresh permanent ID. 96 bits of randomness is ample here."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def prefix_of(entity_id: str) -> str:
    """Return the kind prefix of an ID, or '' if it is not shaped like one."""
    if not entity_id or "_" not in entity_id:
        return ""
    return entity_id.split("_", 1)[0]


def is_id(value: str, prefix: str | None = None) -> bool:
    """True if `value` looks like one of our IDs (optionally of a given kind)."""
    if not isinstance(value, str) or "_" not in value:
        return False
    head, _, tail = value.partition("_")
    if not tail or any(c not in "0123456789abcdef" for c in tail):
        return False
    return head == prefix if prefix else bool(head)
