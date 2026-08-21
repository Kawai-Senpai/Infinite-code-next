"""Export and import a repository's knowledge.

Three formats, one graph:

  graph   JSON nodes and edges, for the explorer UI.
  md      Markdown, for humans and for reading in any editor. Round-trips:
          the importer reads back what the exporter wrote.
  icn     A zip of the markdown plus the raw graph, for handing a codebase's
          knowledge to someone else.

The markdown form is deliberately the canonical shareable one. A knowledge
base nobody can read without the tool is a knowledge base nobody checks, and
`.md` renders everywhere - in an editor, in a diff, on a wiki, in a pull
request. The embedded JSON is what makes the import lossless.

Import never overwrites. Everything arriving from outside is recorded as
`authority='imported'` with its provenance attached, because a memory whose
author you cannot establish should not silently outrank one you wrote.
"""

from __future__ import annotations

import json
import sqlite3
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import ids
from .db import jload, one, rows, write_tx

FORMAT_VERSION = 1

# Node kinds the explorer draws. Keep in step with the legend in explorer.py.
NODE_KINDS = ("repo", "file", "symbol", "memory", "event")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# --------------------------------------------------------------------- graph

def build_graph(conn: sqlite3.Connection, catalog: sqlite3.Connection, repo_id: str,
                include_deleted: bool = False) -> dict[str, Any]:
    """The whole knowledge graph as nodes and edges.

    One pass over each table rather than per-node queries: the explorer wants
    everything at once, and a repository with thousands of symbols would
    otherwise issue thousands of round trips.
    """
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    status_filter = "" if include_deleted else " WHERE status = 'ACTIVE'"

    repo = one(catalog.execute(
        "SELECT repo_id, name, status, vcs FROM repositories WHERE repo_id = ?", (repo_id,)))
    if repo:
        nodes.append({"id": repo["repo_id"], "kind": "repo", "label": repo["name"] or repo_id,
                      "status": repo["status"], "detail": {"vcs": repo["vcs"]}})

    for row in rows(conn.execute(
            "SELECT file_id, path, lang, size, status FROM files" + status_filter)):
        nodes.append({"id": row["file_id"], "kind": "file", "label": row["path"],
                      "status": row["status"],
                      "detail": {"lang": row["lang"], "size": row["size"]}})
        if repo:
            edges.append({"from": repo["repo_id"], "to": row["file_id"], "kind": "CONTAINS",
                          "edge_class": "deterministic", "status": "ACTIVE"})

    for row in rows(conn.execute(
            "SELECT symbol_id, file_id, name, symbol_path, kind, lang, signature,"
            " line_start, line_end, last_known_path, status FROM symbols" + status_filter)):
        nodes.append({
            "id": row["symbol_id"], "kind": "symbol", "label": row["symbol_path"],
            "status": row["status"],
            "detail": {"name": row["name"], "symbol_kind": row["kind"], "lang": row["lang"],
                       "signature": row["signature"], "path": row["last_known_path"],
                       "lines": f"{row['line_start']}-{row['line_end']}"},
        })
        if row["file_id"]:
            edges.append({"from": row["file_id"], "to": row["symbol_id"], "kind": "DEFINES",
                          "edge_class": "deterministic", "status": "ACTIVE"})

    for row in rows(conn.execute(
            "SELECT edge_id, from_id, to_id, kind, edge_class, status, confidence"
            " FROM code_edges" + (" WHERE status = 'ACTIVE'" if not include_deleted else ""))):
        edges.append({"from": row["from_id"], "to": row["to_id"], "kind": row["kind"],
                      "edge_class": row["edge_class"], "status": row["status"],
                      "confidence": row["confidence"]})

    anchors: dict[str, list[dict[str, Any]]] = {}
    for row in rows(conn.execute(
            "SELECT memory_id, symbol_id, file_id, symbol_path, file_path, status,"
            " anchor_confidence FROM anchors")):
        anchors.setdefault(row["memory_id"], []).append(dict(row))

    memory_filter = "" if include_deleted else " WHERE status = 'ACTIVE'"
    for row in rows(conn.execute(
            "SELECT memory_id, kind, severity, title, body, authority, confidence,"
            " status, scope, created_at, last_verified_at, source_event"
            " FROM memories" + memory_filter)):
        attached = anchors.get(row["memory_id"], [])
        worst = _worst_anchor(attached)
        nodes.append({
            "id": row["memory_id"], "kind": "memory", "label": row["title"],
            "status": row["status"],
            "detail": {"memory_kind": row["kind"], "severity": row["severity"],
                       "body": row["body"], "authority": row["authority"],
                       "anchor_status": worst, "created_at": row["created_at"],
                       "last_verified_at": row["last_verified_at"],
                       "scope": row["scope"]},
        })
        for anchor in attached:
            target = anchor["symbol_id"] or anchor["file_id"]
            if target:
                edges.append({"from": row["memory_id"], "to": target, "kind": "ANCHORED_TO",
                              "edge_class": "asserted", "status": anchor["status"],
                              "confidence": anchor["anchor_confidence"]})

    for row in rows(conn.execute(
            "SELECT from_id, to_id, kind, edge_class, status, confidence FROM memory_edges"
            + (" WHERE status = 'ACTIVE'" if not include_deleted else ""))):
        edges.append({"from": row["from_id"], "to": row["to_id"], "kind": row["kind"],
                      "edge_class": row["edge_class"], "status": row["status"],
                      "confidence": row["confidence"]})

    known = {n["id"] for n in nodes}
    kept = [e for e in edges if e["from"] in known and e["to"] in known]

    return {
        "format": "infinite-code-next.graph",
        "version": FORMAT_VERSION,
        "repo_id": repo_id,
        "repo_name": (repo["name"] if repo else repo_id),
        "exported_at": now(),
        "nodes": nodes,
        "edges": kept,
        "stats": _stats(nodes, kept),
    }


def _worst_anchor(anchors: list[dict[str, Any]]) -> str:
    """The least trustworthy anchor decides how a memory is shown."""
    order = ["ORPHANED", "NEEDS_REVIEW", "DRIFTED", "SUPERSEDED", "RESOLVED", "ACTIVE"]
    for status in order:
        if any(a["status"] == status for a in anchors):
            return status
    return "ACTIVE"


def _stats(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind: dict[str, int] = {}
    for node in nodes:
        by_kind[node["kind"]] = by_kind.get(node["kind"], 0) + 1
    by_edge: dict[str, int] = {}
    for edge in edges:
        by_edge[edge["kind"]] = by_edge.get(edge["kind"], 0) + 1
    return {"nodes": len(nodes), "edges": len(edges),
            "by_node_kind": by_kind, "by_edge_kind": by_edge}


# ------------------------------------------------------------------ markdown

def to_markdown(graph: dict[str, Any]) -> str:
    """Human-readable knowledge, with a machine-readable tail.

    The prose is the point; the JSON block at the end is what makes import
    lossless. Anything that reads markdown can show this file usefully without
    knowing the format exists.
    """
    memories = [n for n in graph["nodes"] if n["kind"] == "memory"]
    lines: list[str] = [
        f"# Knowledge: {graph['repo_name']}",
        "",
        f"Exported {graph['exported_at']} from `{graph['repo_id']}`.",
        f"{len(memories)} memories over {graph['stats']['nodes']} nodes"
        f" and {graph['stats']['edges']} edges.",
        "",
        "Import with `icn-explore import <this file>`.",
        "",
    ]

    by_kind: dict[str, list[dict[str, Any]]] = {}
    for memory in memories:
        by_kind.setdefault(memory["detail"]["memory_kind"], []).append(memory)

    # Rules first, then history, then the softer material. An agent reading
    # top-down should meet the constraints before the commentary.
    order = ["security", "invariant", "warning", "contract", "failed_attempt", "decision",
             "bug_history", "fix_history", "migration", "performance", "convention",
             "rationale", "test_evidence"]
    ordered = [k for k in order if k in by_kind] + [k for k in sorted(by_kind) if k not in order]

    anchors_by_memory: dict[str, list[str]] = {}
    labels = {n["id"]: n["label"] for n in graph["nodes"]}
    for edge in graph["edges"]:
        if edge["kind"] == "ANCHORED_TO":
            anchors_by_memory.setdefault(edge["from"], []).append(labels.get(edge["to"], edge["to"]))

    for kind in ordered:
        lines.append(f"## {kind.replace('_', ' ').title()}")
        lines.append("")
        for memory in sorted(by_kind[kind], key=lambda m: m["label"] or ""):
            detail = memory["detail"]
            flag = "" if detail["anchor_status"] == "ACTIVE" else f" **[{detail['anchor_status']}]**"
            lines.append(f"### {memory['label']}{flag}")
            lines.append("")
            lines.append(f"- severity: {detail['severity']} | authority: {detail['authority']}")
            attached = anchors_by_memory.get(memory["id"], [])
            if attached:
                lines.append("- applies to: " + ", ".join(f"`{a}`" for a in attached[:8]))
            lines.append("")
            body = (detail.get("body") or "").strip()
            if body and body != memory["label"]:
                lines.extend([body, ""])

    lines.extend([
        "---",
        "",
        "<!-- infinite-code-next:graph -->",
        "```json",
        json.dumps(graph, indent=1, ensure_ascii=False),
        "```",
        "",
    ])
    return "\n".join(lines)


def from_markdown(text: str) -> dict[str, Any] | None:
    """Recover the graph from an exported markdown file."""
    marker = "<!-- infinite-code-next:graph -->"
    if marker not in text:
        return None
    tail = text.split(marker, 1)[1]
    start = tail.find("```json")
    if start < 0:
        return None
    body = tail[start + len("```json"):]
    end = body.find("```")
    if end < 0:
        return None
    try:
        return json.loads(body[:end])
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------- .icn

def write_bundle(graph: dict[str, Any], target: Path) -> Path:
    """A zip carrying both the readable and the machine-readable form."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("knowledge.md", to_markdown(graph))
        bundle.writestr("graph.json", json.dumps(graph, indent=1, ensure_ascii=False))
        bundle.writestr("MANIFEST.json", json.dumps({
            "format": "infinite-code-next.bundle",
            "version": FORMAT_VERSION,
            "repo_id": graph["repo_id"],
            "repo_name": graph["repo_name"],
            "exported_at": graph["exported_at"],
            "stats": graph["stats"],
        }, indent=1))
    return target


def read_bundle(source: Path) -> dict[str, Any] | None:
    """Load a graph from .icn, .zip, .md or .json - whatever was handed over."""
    source = Path(source)
    if not source.exists():
        return None

    if source.suffix.lower() in (".icn", ".zip"):
        try:
            with zipfile.ZipFile(source) as bundle:
                names = set(bundle.namelist())
                if "graph.json" in names:
                    return json.loads(bundle.read("graph.json").decode("utf-8"))
                if "knowledge.md" in names:
                    return from_markdown(bundle.read("knowledge.md").decode("utf-8"))
                # A folder of markdown files is a legitimate hand-off too.
                merged = None
                for name in sorted(names):
                    if not name.lower().endswith(".md"):
                        continue
                    graph = from_markdown(bundle.read(name).decode("utf-8"))
                    merged = graph if merged is None else _merge(merged, graph)
                return merged
        except (zipfile.BadZipFile, KeyError, UnicodeDecodeError, json.JSONDecodeError):
            return None

    try:
        text = source.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if source.suffix.lower() == ".json":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
    return from_markdown(text)


def read_many(paths: list[Path]) -> dict[str, Any] | None:
    """Merge several exports - a directory of markdown, or a set of bundles."""
    merged = None
    for path in paths:
        graph = read_bundle(path)
        if graph is None:
            continue
        merged = graph if merged is None else _merge(merged, graph)
    return merged


def _merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    seen = {n["id"] for n in left["nodes"]}
    left["nodes"].extend(n for n in right.get("nodes", []) if n["id"] not in seen)
    pairs = {(e["from"], e["to"], e["kind"]) for e in left["edges"]}
    left["edges"].extend(e for e in right.get("edges", [])
                         if (e["from"], e["to"], e["kind"]) not in pairs)
    left["stats"] = _stats(left["nodes"], left["edges"])
    return left


# -------------------------------------------------------------------- import

def import_graph(conn: sqlite3.Connection, catalog: sqlite3.Connection, repo_id: str,
                 graph: dict[str, Any], source: str = "import") -> dict[str, Any]:
    """Bring another repository's memories in, without pretending they are ours.

    Imported memories are `authority='imported'` and carry their origin. They
    are anchored where a symbol path matches locally and left repo-scoped where
    it does not - a memory about code we do not have is still worth keeping,
    but it must not claim to describe a span it never saw.
    """
    from . import anchors as anchor_mod
    from . import catalog as catalog_mod

    incoming = [n for n in graph.get("nodes", []) if n["kind"] == "memory"]
    if not incoming:
        return {"ok": False, "error": "no memories found in that file"}

    origin = graph.get("repo_name") or graph.get("repo_id") or source
    added = skipped = anchored = 0
    id_map: dict[str, str] = {}

    for node in incoming:
        detail = node.get("detail") or {}
        body = (detail.get("body") or node.get("label") or "").strip()
        if not body:
            continue
        # Same text already here means the knowledge is not new.
        if one(conn.execute("SELECT memory_id FROM memories WHERE body = ? AND status='ACTIVE'",
                            (body,))):
            skipped += 1
            continue

        memory_id = ids.new_id(ids.MEMORY)
        id_map[node["id"]] = memory_id
        title = (node.get("label") or body.split("\n", 1)[0])[:200]
        with write_tx(conn):
            conn.execute(
                "INSERT INTO memories (memory_id, kind, severity, title, body, authority,"
                " confidence, status, scope, source_event, version, created_at, updated_at,"
                " created_commit, last_verified_commit, last_verified_at, valid_from)"
                " VALUES (?,?,?,?,?,'imported',?,'ACTIVE',?,?,1,?,?,NULL,NULL,?,?)",
                (memory_id, detail.get("memory_kind") or "rationale",
                 detail.get("severity") or "medium", title, body,
                 float(detail.get("confidence") or 0.6),
                 # A per-memory event id: imported memories did not happen
                 # together here, and a shared one makes same-event joins
                 # (test coverage, causal siblings) link everything to
                 # everything.
                 "repo", f"import:{origin}:{memory_id}", now(), now(), now(), now()),
            )
            conn.execute(
                "INSERT INTO fts_memories (memory_id, title, body, kind) VALUES (?,?,?,?)",
                (memory_id, title, body[:4000], detail.get("memory_kind") or "rationale"))
        added += 1
        catalog_mod.register_memory(catalog, memory_id, repo_id,
                                    detail.get("memory_kind") or "rationale",
                                    detail.get("severity") or "medium", "ACTIVE", title)

    # Re-anchor where the same symbol path exists locally.
    labels = {n["id"]: n for n in graph.get("nodes", [])}
    for edge in graph.get("edges", []):
        if edge["kind"] != "ANCHORED_TO" or edge["from"] not in id_map:
            continue
        target = labels.get(edge["to"])
        if not target:
            continue
        local = one(conn.execute(
            "SELECT * FROM symbols WHERE symbol_path = ? AND status='ACTIVE'",
            (target.get("label"),)))
        with write_tx(conn):
            if local:
                anchor_mod.create_anchor(conn, id_map[edge["from"]], dict(local), None, None)
                anchored += 1
            else:
                anchor_mod.create_anchor(conn, id_map[edge["from"]], None, None, None,
                                         target_kind="repo")

    return {"ok": True, "imported": added, "skipped_duplicates": skipped,
            "anchored_to_local_code": anchored, "origin": origin,
            "note": "imported memories are marked authority='imported' and anchored only"
                    " where a matching symbol exists here"}
