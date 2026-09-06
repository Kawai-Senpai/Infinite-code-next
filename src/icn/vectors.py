"""Storage and brute-force search for semantic vectors.

There is no vector database here, and that is a deliberate design decision
rather than a shortcut. Approximate indexes (FAISS, HNSW, a hosted service)
exist to avoid scanning millions of vectors. A large repository indexed by ICN
holds a few thousand symbols; every repository in a working catalogue together
is tens of thousands. Measured on a Tiger Lake laptop, an exact scan of 30,000
256-dimensional vectors takes 1.1ms and 31MB.

An approximate index would therefore trade exact results, a second store that
can drift out of sync with SQLite, and a new class of "the index is down"
failure, for a saving of under a millisecond. Brute force is not the cheap
option here, it is the correct one. Revisit only above roughly 500k vectors.
"""

from __future__ import annotations

import hashlib
import sqlite3
from typing import Any, Iterable

from datetime import datetime, timezone

from .db import rows, write_tx

SYMBOL, MEMORY = "symbol", "memory"


def now() -> str:
    """Local, rather than imported from compiler: this module sits below the
    compiler in the import order and must not pull it in."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

# Encoding is the slow half of indexing, so it happens in batches. Large
# enough to amortise the call overhead, small enough that a huge repository
# does not build one enormous list of strings in memory.
BATCH = 256


def content_hash(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8", "replace"), digest_size=16).hexdigest()


def _pack(vector: Any) -> bytes:
    return vector.astype("float32", copy=False).tobytes()


def stored_encoder(conn: sqlite3.Connection) -> tuple[str, int] | None:
    """Which encoder the stored vectors came from, if any."""
    row = conn.execute("SELECT encoder_id, dim FROM embeddings LIMIT 1").fetchone()
    return (row["encoder_id"], row["dim"]) if row else None


def clear(conn: sqlite3.Connection, scope: str | None = None) -> int:
    with write_tx(conn):
        cur = (conn.execute("DELETE FROM embeddings WHERE scope = ?", (scope,))
               if scope else conn.execute("DELETE FROM embeddings"))
    return cur.rowcount or 0


def _existing(conn: sqlite3.Connection, scope: str, encoder_id: str) -> dict[str, str]:
    """item_id -> content_hash for rows this encoder already produced.

    Rows from a different encoder are deliberately absent, so they read as
    missing and get re-encoded rather than being compared across models.
    """
    return {r["item_id"]: r["content_hash"] for r in rows(conn.execute(
        "SELECT item_id, content_hash FROM embeddings WHERE scope = ? AND encoder_id = ?",
        (scope, encoder_id)))}


def _symbol_texts(conn: sqlite3.Connection) -> Iterable[tuple[str, str]]:
    """What a symbol looks like to the embedding model.

    Path, name and signature come first because they carry most of the intent
    in the fewest tokens, and a static model has no attention to recover
    meaning from a long tail. The body is included but truncated: beyond a
    few hundred characters, mean pooling washes the distinctive parts out.
    """
    for row in rows(conn.execute(
            "SELECT s.symbol_id, s.symbol_path, s.name, s.kind, s.signature,"
            " f.body AS body FROM symbols s"
            " LEFT JOIN fts_symbols f ON f.symbol_id = s.symbol_id"
            " WHERE s.status = 'ACTIVE'")):
        parts = [row["symbol_path"] or "", row["name"] or "", row["kind"] or "",
                 row["signature"] or "", (row["body"] or "")[:600]]
        yield row["symbol_id"], "\n".join(p for p in parts if p)


def _memory_texts(conn: sqlite3.Connection) -> Iterable[tuple[str, str]]:
    for row in rows(conn.execute(
            "SELECT memory_id, title, body, kind FROM memories WHERE status = 'ACTIVE'")):
        parts = [row["kind"] or "", row["title"] or "", (row["body"] or "")[:900]]
        yield row["memory_id"], "\n".join(p for p in parts if p)


def refresh(conn: sqlite3.Connection, encoder: Any, scope: str | None = None,
            limit: int | None = None) -> dict[str, Any]:
    """Encode anything new or changed. Safe to call on every index.

    Incremental by content hash, so a repository whose code did not change
    costs one SELECT and no encoding at all.
    """
    if encoder is None:
        return {"encoded": 0, "skipped": 0, "state": "no encoder"}

    sources = {SYMBOL: _symbol_texts, MEMORY: _memory_texts}
    if scope:
        sources = {scope: sources[scope]}

    # A model change makes every stored vector meaningless: different space,
    # possibly different width. Drop them rather than mixing two spaces in one
    # similarity computation, which would silently return nonsense.
    previous = stored_encoder(conn)
    if previous and previous != (encoder.encoder_id, encoder.dimensions):
        clear(conn)

    encoded = skipped = 0
    stamp = now()
    for name, source in sources.items():
        seen = _existing(conn, name, encoder.encoder_id)
        pending: list[tuple[str, str, str]] = []

        def flush(batch: list[tuple[str, str, str]]) -> int:
            if not batch:
                return 0
            vectors = encoder.encode([text for _, text, _ in batch])
            with write_tx(conn):
                conn.executemany(
                    "INSERT OR REPLACE INTO embeddings"
                    " (scope, item_id, encoder_id, dim, content_hash, vector, created_at)"
                    " VALUES (?,?,?,?,?,?,?)",
                    [(name, item_id, encoder.encoder_id, encoder.dimensions,
                      digest, _pack(vectors[i]), stamp)
                     for i, (item_id, _, digest) in enumerate(batch)])
            return len(batch)

        for item_id, text in source(conn):
            digest = content_hash(text)
            if seen.get(item_id) == digest:
                skipped += 1
                continue
            pending.append((item_id, text, digest))
            if len(pending) >= BATCH:
                encoded += flush(pending)
                pending = []
            if limit is not None and encoded >= limit:
                break
        encoded += flush(pending)

    # Vectors for content that no longer exists would surface deleted symbols
    # in search results.
    with write_tx(conn):
        conn.execute(
            "DELETE FROM embeddings WHERE scope = 'symbol' AND item_id NOT IN"
            " (SELECT symbol_id FROM symbols WHERE status = 'ACTIVE')")
        conn.execute(
            "DELETE FROM embeddings WHERE scope = 'memory' AND item_id NOT IN"
            " (SELECT memory_id FROM memories WHERE status = 'ACTIVE')")

    return {"encoded": encoded, "skipped": skipped,
            "encoder": encoder.encoder_id, "dimensions": encoder.dimensions}


def search(conn: sqlite3.Connection, encoder: Any, query: str, scope: str,
           limit: int = 60) -> list[tuple[str, float]]:
    """The nearest stored items to `query`, best first.

    Returns (item_id, cosine similarity). An exact scan: see the module
    docstring for why there is no index.
    """
    if encoder is None or not (query or "").strip():
        return []

    import numpy as np

    stored = rows(conn.execute(
        "SELECT item_id, vector FROM embeddings WHERE scope = ? AND encoder_id = ? AND dim = ?",
        (scope, encoder.encoder_id, encoder.dimensions)))
    if not stored:
        return []

    matrix = np.frombuffer(b"".join(r["vector"] for r in stored), dtype=np.float32)
    matrix = matrix.reshape(len(stored), encoder.dimensions)

    # Both sides are unit vectors (the encoder normalises), so the dot product
    # is cosine similarity and needs no further division.
    scores = matrix @ encoder.encode([query])[0]

    top = min(limit, len(stored))
    # argpartition finds the top-k without sorting the whole array; only the
    # k survivors are then ordered.
    picked = np.argpartition(-scores, top - 1)[:top] if top < len(stored) else np.arange(len(stored))
    picked = picked[np.argsort(-scores[picked])]
    return [(stored[i]["item_id"], float(scores[i])) for i in picked]


def stats(conn: sqlite3.Connection) -> dict[str, Any]:
    counts = {r["scope"]: r["n"] for r in rows(conn.execute(
        "SELECT scope, COUNT(*) AS n FROM embeddings GROUP BY scope"))}
    current = stored_encoder(conn)
    return {"symbols": counts.get(SYMBOL, 0), "memories": counts.get(MEMORY, 0),
            "encoder": current[0] if current else None,
            "dimensions": current[1] if current else None}
