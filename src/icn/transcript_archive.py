"""Raw transcript archive: our own copy of what the vendors are about to delete.

The vendors treat their transcripts as scratch. Claude Code prunes projects/
after 30 days by default, Codex rolls sessions away, VS Code drops chat storage
with the workspace. Anything indexed only from those files disappears with them,
including the normalised copy's ability to be rebuilt.

So the archive, not the index, is canonical:

    vendor file --sync--> archive/<provider>/<key>/transcript.jsonl
                                                  companions/<name>.jsonl
                                                  versions/<stamp>.jsonl
                                                  manifest.json

    archive + live files --refresh--> transcripts.db   (rebuildable, always)

Layout, per session key:
    transcript.jsonl   the bytes, appended to as the vendor appends
    companions/        sidecar files that belong to the session (Claude Code
                       subagent transcripts live in <session>/subagents/)
    versions/<iso>.jsonl  a previous generation, kept when the live file stopped
                       being an extension of what we already had
    manifest.json      provenance: provider, surface, live path, sizes, the
                       hash of the prefix we last anchored on, sync history

Append-only sync. A vendor JSONL only ever grows, so the normal case is "the
first N bytes are identical, copy the tail". We verify that with a hash of the
first min(archived, live) bytes rather than trusting mtime. When the prefix
does not match - the vendor rewrote, compacted, or reused the id - the existing
copy rotates into versions/ and the new bytes start a fresh transcript.jsonl.
Nothing is ever overwritten in place, so a truncation upstream cannot destroy
history down here.

Retention is unlimited by default (ICN_TRANSCRIPTS_ARCHIVE_RETENTION_DAYS=0).

Deletion is explicit and sticky. Removing an archive directory alone would
achieve nothing: the next sync would copy it straight back from the vendor
store. A delete therefore writes a tombstone, in one of two modes:

    archive_only  stop archiving it; the live file stays searchable while the
                  vendor keeps it
    exclude_all   never index it from anywhere, live or archived

Tombstones live in archive.sqlite, which is itself rebuildable from the
manifests except for the tombstones - those are the one row type that must
survive, and they are what the sqlite file is really for.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from . import paths
from .db import jdump, one, rows, write_tx

ENV_ARCHIVE_ENABLED = "ICN_TRANSCRIPTS_ARCHIVE"              # 1 (default) | 0
ENV_ARCHIVE_RETENTION = "ICN_TRANSCRIPTS_ARCHIVE_RETENTION_DAYS"   # 0 = unlimited
ENV_ARCHIVE_MAX_VERSIONS = "ICN_TRANSCRIPTS_ARCHIVE_MAX_VERSIONS"  # 0 = unlimited
ENV_ARCHIVE_MAX_BYTES = "ICN_TRANSCRIPTS_ARCHIVE_MAX_FILE_MB"      # skip huge files

PREFIX_SAMPLE = 1 << 20        # bytes hashed to prove append-only continuity
COPY_CHUNK = 1 << 20
DEFAULT_MAX_FILE_MB = 512

TOMBSTONE_MODES = ("archive_only", "exclude_all")

SCHEMA = """
CREATE TABLE IF NOT EXISTS archives (
    archive_key   TEXT PRIMARY KEY,   -- provider/<hash of surface+identity>
    provider      TEXT NOT NULL,
    surface       TEXT NOT NULL,
    live_path     TEXT NOT NULL,
    archive_path  TEXT NOT NULL,
    native_id     TEXT,
    bytes_copied  INTEGER NOT NULL DEFAULT 0,
    live_size     INTEGER,
    live_mtime_ns INTEGER,
    generations   INTEGER NOT NULL DEFAULT 1,
    first_seen    TEXT NOT NULL,
    last_sync     TEXT NOT NULL,
    live_missing  INTEGER NOT NULL DEFAULT 0,
    meta_json     TEXT
);
CREATE INDEX IF NOT EXISTS idx_archives_live ON archives(live_path);
CREATE INDEX IF NOT EXISTS idx_archives_provider ON archives(provider, surface);

-- The one table that is not rebuildable. A purge must stay purged even though
-- the vendor file it came from is still sitting on disk.
CREATE TABLE IF NOT EXISTS tombstones (
    target      TEXT PRIMARY KEY,     -- live path, archive path, or archive key
    mode        TEXT NOT NULL,        -- archive_only | exclude_all
    reason      TEXT,
    created_at  TEXT NOT NULL
);
"""


# ---------------------------------------------------------------- settings


def archive_enabled() -> bool:
    return os.environ.get(ENV_ARCHIVE_ENABLED, "1").strip().lower() not in ("0", "false", "no", "off")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def retention_days() -> int:
    """0 means keep forever, which is the default and the point of the archive."""
    return max(0, _env_int(ENV_ARCHIVE_RETENTION, 0))


def max_versions() -> int:
    return max(0, _env_int(ENV_ARCHIVE_MAX_VERSIONS, 0))


def max_file_bytes() -> int:
    return max(1, _env_int(ENV_ARCHIVE_MAX_BYTES, DEFAULT_MAX_FILE_MB)) * 1024 * 1024


def archive_root() -> Path:
    return paths.storage_root() / "data" / "transcripts-archive"


def archive_db_path() -> Path:
    return archive_root() / "archive.sqlite"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ------------------------------------------------------------------ keying


def archive_key(provider: str, surface: str, identity: str) -> str:
    """Stable directory name for one session's copy.

    Keyed on the vendor path rather than the session id: the id is not known
    until the file is parsed, and the archive must be able to copy bytes it has
    not parsed yet. Paths are case-folded on Windows so a drive-letter or case
    difference does not fork the archive.
    """
    try:
        identity = os.path.realpath(identity)   # 8.3 short names, symlinks
    except (OSError, ValueError):
        pass
    norm = identity.replace("\\", "/")
    if os.name == "nt":
        norm = norm.lower()
    digest = hashlib.sha256(f"{surface}\x00{norm}".encode("utf-8")).hexdigest()[:24]
    return f"{provider}/{digest}"


def _safe_provider(provider: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in provider) or "other"


def archive_dir(key: str) -> Path:
    provider, _, digest = key.partition("/")
    return archive_root() / _safe_provider(provider) / digest


# -------------------------------------------------------------- byte tools


def _hash_prefix(path: Path, length: int) -> str | None:
    """sha256 of the first `length` bytes, capped at PREFIX_SAMPLE.

    Hashing the whole file every sync would make the archive O(total bytes) per
    run on a 459MB store. A megabyte of prefix is enough to catch a rewrite:
    every vendor writes its session header first, and a compaction or id reuse
    changes it.
    """
    if length <= 0:
        return None
    want = min(length, PREFIX_SAMPLE)
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            left = want
            while left > 0:
                chunk = fh.read(min(COPY_CHUNK, left))
                if not chunk:
                    return None          # file shrank while reading: not a prefix
                left -= len(chunk)
                h.update(chunk)
    except OSError:
        return None
    return f"{want}:{h.hexdigest()}"


def _append_tail(src: Path, dst: Path, offset: int) -> int:
    """Copy src[offset:] onto the end of dst. Returns bytes written.

    Written through a flush+fsync so a crash mid-sync leaves a short file, not
    a file with a hole in it; the next sync re-derives the offset from dst's
    actual size and continues from there.
    """
    written = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as fin:
        fin.seek(offset)
        with open(dst, "ab") as fout:
            while True:
                chunk = fin.read(COPY_CHUNK)
                if not chunk:
                    break
                fout.write(chunk)
                written += len(chunk)
            fout.flush()
            os.fsync(fout.fileno())
    return written


# ------------------------------------------------------------------ store


class ArchiveStore:
    """archive.sqlite: what has been copied, and what must never be."""

    def __init__(self, db_path: Path | None = None):
        self.path = db_path or archive_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=10.0, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.conn.execute("PRAGMA busy_timeout = 10000")
        self.conn.executescript(SCHEMA)
        if os.name != "nt":
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass
        self._tombstones: dict[str, str] | None = None

    def close(self) -> None:
        self.conn.close()

    # -- tombstones --------------------------------------------------------

    def tombstones(self, refresh: bool = False) -> dict[str, str]:
        """{normalised target: mode}. Cached: consulted once per source per sync."""
        if self._tombstones is None or refresh:
            self._tombstones = {
                _norm_target(r["target"]): r["mode"]
                for r in rows(self.conn.execute("SELECT target, mode FROM tombstones"))
            }
        return self._tombstones

    def tombstone_mode(self, *targets: str | None) -> str | None:
        """The strictest mode covering any of these identifiers."""
        table = self.tombstones()
        best: str | None = None
        for t in targets:
            if not t:
                continue
            mode = table.get(_norm_target(t))
            if mode == "exclude_all":
                return "exclude_all"
            if mode:
                best = mode
        return best

    def add_tombstone(self, target: str, mode: str, reason: str | None = None) -> dict[str, Any]:
        if mode not in TOMBSTONE_MODES:
            raise ValueError(f"mode must be one of {TOMBSTONE_MODES}")
        with write_tx(self.conn):
            self.conn.execute(
                "INSERT INTO tombstones (target, mode, reason, created_at) VALUES (?,?,?,?)"
                " ON CONFLICT(target) DO UPDATE SET mode=excluded.mode, reason=excluded.reason,"
                " created_at=excluded.created_at",
                (target, mode, reason, _now()),
            )
        self._tombstones = None
        return {"target": target, "mode": mode, "reason": reason}

    def remove_tombstone(self, target: str) -> bool:
        with write_tx(self.conn):
            cur = self.conn.execute("DELETE FROM tombstones WHERE target = ?", (target,))
        self._tombstones = None
        return cur.rowcount > 0

    def list_tombstones(self) -> list[dict[str, Any]]:
        return rows(self.conn.execute(
            "SELECT target, mode, reason, created_at FROM tombstones ORDER BY created_at DESC"))

    # -- archives ----------------------------------------------------------

    def get(self, key: str) -> dict[str, Any] | None:
        return one(self.conn.execute("SELECT * FROM archives WHERE archive_key = ?", (key,)))

    def upsert(self, row: dict[str, Any]) -> None:
        existing = self.get(row["archive_key"])
        now = _now()
        with write_tx(self.conn):
            if existing:
                self.conn.execute(
                    "UPDATE archives SET live_path=?, archive_path=?, native_id=COALESCE(?, native_id),"
                    " bytes_copied=?, live_size=?, live_mtime_ns=?, generations=?, last_sync=?,"
                    " live_missing=?, meta_json=? WHERE archive_key=?",
                    (row["live_path"], row["archive_path"], row.get("native_id"),
                     row["bytes_copied"], row.get("live_size"), row.get("live_mtime_ns"),
                     row.get("generations", existing["generations"]), now,
                     int(row.get("live_missing", 0)), jdump(row.get("meta")), row["archive_key"]),
                )
            else:
                self.conn.execute(
                    "INSERT INTO archives (archive_key, provider, surface, live_path, archive_path,"
                    " native_id, bytes_copied, live_size, live_mtime_ns, generations, first_seen,"
                    " last_sync, live_missing, meta_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (row["archive_key"], row["provider"], row["surface"], row["live_path"],
                     row["archive_path"], row.get("native_id"), row["bytes_copied"],
                     row.get("live_size"), row.get("live_mtime_ns"), row.get("generations", 1),
                     now, now, int(row.get("live_missing", 0)), jdump(row.get("meta"))),
                )

    def mark_live_missing(self, keys: Iterable[str]) -> int:
        keys = list(keys)
        if not keys:
            return 0
        with write_tx(self.conn):
            for key in keys:
                self.conn.execute("UPDATE archives SET live_missing = 1 WHERE archive_key = ?", (key,))
        return len(keys)

    def drop(self, key: str) -> None:
        with write_tx(self.conn):
            self.conn.execute("DELETE FROM archives WHERE archive_key = ?", (key,))

    def list_archives(self, *, provider: str | None = None, live_missing: bool | None = None,
                      limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        where, params = [], []
        if provider:
            where.append("provider = ?")
            params.append(provider)
        if live_missing is not None:
            where.append("live_missing = ?")
            params.append(1 if live_missing else 0)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        params.extend([limit, offset])
        return rows(self.conn.execute(
            f"SELECT * FROM archives {clause} ORDER BY last_sync DESC LIMIT ? OFFSET ?", params))

    def stats(self) -> dict[str, Any]:
        by_provider = rows(self.conn.execute(
            "SELECT provider, surface, COUNT(*) AS archives, COALESCE(SUM(bytes_copied),0) AS bytes,"
            " SUM(live_missing) AS live_missing FROM archives GROUP BY provider, surface"
            " ORDER BY provider, surface"))
        total = one(self.conn.execute(
            "SELECT COUNT(*) AS archives, COALESCE(SUM(bytes_copied),0) AS bytes,"
            " COALESCE(SUM(live_missing),0) AS live_missing FROM archives")) or {}
        return {
            "root": str(archive_root()),
            "enabled": archive_enabled(),
            "retention_days": retention_days() or "unlimited",
            "archives": total.get("archives", 0),
            "bytes": total.get("bytes", 0),
            "live_missing": total.get("live_missing", 0),
            "tombstones": len(self.list_tombstones()),
            "by_provider": by_provider,
        }


def _norm_target(value: str) -> str:
    v = value.replace("\\", "/").rstrip("/")
    return v.lower() if os.name == "nt" else v


# ------------------------------------------------------------------- sync


@dataclass
class SyncResult:
    key: str
    action: str          # created | appended | unchanged | rotated | skipped | error
    bytes_copied: int = 0
    detail: str | None = None


def _write_manifest(dir_path: Path, data: dict[str, Any]) -> None:
    tmp = dir_path / "manifest.json.tmp"
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, dir_path / "manifest.json")


def _read_manifest(dir_path: Path) -> dict[str, Any]:
    try:
        return json.loads((dir_path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _claude_companions(live: Path) -> list[Path]:
    """Claude Code stores subagent transcripts beside the session, in a directory
    named after it. They are part of the same conversation and vanish with it."""
    sub = live.parent / live.stem / "subagents"
    if not sub.is_dir():
        return []
    return sorted(p for p in sub.glob("*.jsonl") if p.is_file())


def companions_for(surface: str, live: Path) -> list[Path]:
    if surface == "claude_code":
        return _claude_companions(live)
    return []


def sync_file(store: ArchiveStore, provider: str, surface: str, live_path: str,
              native_id: str | None = None) -> SyncResult:
    """Copy one vendor file into the archive, appending where it is an extension
    of what we already hold and rotating the old generation where it is not."""
    live = Path(live_path)
    key = archive_key(provider, surface, live_path)

    mode = store.tombstone_mode(live_path, key, str(archive_dir(key)))
    if mode:
        return SyncResult(key, "skipped", detail=f"tombstoned ({mode})")

    try:
        st = live.stat()
    except OSError as exc:
        return SyncResult(key, "error", detail=f"stat failed: {exc}")
    if not live.is_file():
        return SyncResult(key, "skipped", detail="not a regular file")
    if st.st_size > max_file_bytes():
        return SyncResult(key, "skipped",
                          detail=f"{st.st_size} bytes exceeds {ENV_ARCHIVE_MAX_BYTES} cap")

    dir_path = archive_dir(key)
    dest = dir_path / "transcript.jsonl"
    manifest = _read_manifest(dir_path)
    have = dest.stat().st_size if dest.is_file() else 0

    action = "unchanged"
    copied = 0
    generations = int(manifest.get("generations", 1))

    if have == 0:
        dir_path.mkdir(parents=True, exist_ok=True)
        copied = _append_tail(live, dest, 0)
        action = "created"
    elif st.st_size < have or _hash_prefix(live, have) != manifest.get("prefix_hash"):
        # Not an extension of what we hold: the vendor rewrote, compacted, or
        # reused the path. Keep both - the old bytes are history the live file
        # no longer has.
        versions = dir_path / "versions"
        versions.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(dest, versions / f"{_stamp()}.jsonl")
        except OSError as exc:
            return SyncResult(key, "error", detail=f"rotate failed: {exc}")
        generations += 1
        copied = _append_tail(live, dest, 0)
        action = "rotated"
        _prune_versions(versions)
    elif st.st_size > have:
        copied = _append_tail(live, dest, have)
        action = "appended"

    comp_copied = 0
    comp_map: dict[str, str] = dict(manifest.get("companions") or {})
    for comp in companions_for(surface, live):
        target = dir_path / "companions" / comp.name
        comp_map[comp.name] = str(comp)
        try:
            chave = target.stat().st_size if target.is_file() else 0
            csize = comp.stat().st_size
            if csize > chave:
                comp_copied += _append_tail(comp, target, chave)
            elif csize < chave:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(comp, target.with_suffix(target.suffix + f".{_stamp()}"))
                comp_copied += _append_tail(comp, target, 0)
        except OSError:
            continue

    if action == "unchanged" and comp_copied == 0 and comp_map == (manifest.get("companions") or {}):
        # Still record that we looked, so status reflects reality.
        store.upsert({"archive_key": key, "provider": provider, "surface": surface,
                      "live_path": live_path, "archive_path": str(dest), "native_id": native_id,
                      "bytes_copied": have, "live_size": st.st_size,
                      "live_mtime_ns": st.st_mtime_ns, "generations": generations,
                      "live_missing": 0})
        return SyncResult(key, "unchanged")

    total = dest.stat().st_size if dest.is_file() else 0
    _write_manifest(dir_path, {
        "archive_key": key,
        "provider": provider,
        "surface": surface,
        "native_id": native_id,
        "live_path": live_path,
        "bytes": total,
        "generations": generations,
        "prefix_hash": _hash_prefix(dest, total),
        "companions": comp_map,
        "first_seen": manifest.get("first_seen") or _now(),
        "last_sync": _now(),
    })
    store.upsert({"archive_key": key, "provider": provider, "surface": surface,
                  "live_path": live_path, "archive_path": str(dest), "native_id": native_id,
                  "bytes_copied": total, "live_size": st.st_size,
                  "live_mtime_ns": st.st_mtime_ns, "generations": generations,
                  "live_missing": 0})
    return SyncResult(key, action, bytes_copied=copied + comp_copied)


def _prune_versions(versions: Path) -> None:
    keep = max_versions()
    if keep <= 0:
        return
    files = sorted(versions.glob("*.jsonl"))
    for old in files[:-keep]:
        try:
            old.unlink()
        except OSError:
            pass


def sync_all(sources: list[Any], store: ArchiveStore | None = None,
             limit_seconds: float | None = None, heartbeat: Any = None) -> dict[str, Any]:
    """Archive every file-backed source. Sources whose bytes are rows in a vendor
    database (cursor_composer) are not copied here: the index is their archive.
    """
    if not archive_enabled():
        return {"enabled": False}
    owned = store is None
    store = store or ArchiveStore()
    started = time.monotonic()
    counts = {"enabled": True, "considered": 0, "created": 0, "appended": 0, "unchanged": 0,
              "rotated": 0, "skipped": 0, "error": 0, "bytes": 0, "skipped_time_budget": 0}
    errors: list[str] = []
    seen_keys: set[str] = set()
    try:
        for source in sources:
            if getattr(source, "origin", "live") != "live":
                continue
            file = getattr(source, "file", "")
            if not file or "#" in file:
                continue
            counts["considered"] += 1
            if limit_seconds is not None and time.monotonic() - started > limit_seconds:
                counts["skipped_time_budget"] += 1
                continue
            if heartbeat is not None:
                heartbeat()
            result = sync_file(store, source.provider, source.surface, file)
            seen_keys.add(result.key)
            counts[result.action] = counts.get(result.action, 0) + 1
            counts["bytes"] += result.bytes_copied
            if result.action == "error" and len(errors) < 20:
                errors.append(f"{file}: {result.detail}")

        # Anything archived whose vendor file is gone is exactly the case the
        # archive exists for. Flag it so the index can keep serving it.
        gone = [r["archive_key"] for r in store.list_archives(live_missing=False, limit=100000)
                if r["archive_key"] not in seen_keys and not os.path.exists(r["live_path"])]
        counts["live_missing_new"] = store.mark_live_missing(gone)
        counts["seconds"] = round(time.monotonic() - started, 2)
        if errors:
            counts["errors"] = errors
        return counts
    finally:
        if owned:
            store.close()


# -------------------------------------------------------------- discovery


def _manifest_dirs() -> list[Path]:
    root = archive_root()
    if not root.is_dir():
        return []
    out = []
    for provider_dir in root.iterdir():
        if not provider_dir.is_dir():
            continue
        for entry in provider_dir.iterdir():
            if entry.is_dir() and (entry / "manifest.json").is_file():
                out.append(entry)
    return out


def archived_sources(source_factory: Any, store: ArchiveStore | None = None,
                     include_versions: bool = True) -> list[Any]:
    """Build Source objects for the archive copies, for the index to read.

    `source_factory(provider, surface, file, live_path, origin)` builds the
    caller's Source; this module does not import transcripts, so the dependency
    runs one way only (transcripts -> archive).

    Origin tells the index what it is looking at:
        archive        a copy whose live file still exists (same session identity)
        archive_only   the vendor deleted the original; only we have it now
        archive_version an older generation from versions/ - its own session
    """
    owned = store is None
    store = store or ArchiveStore()
    out: list[Any] = []
    # A Claude subagent transcript is archived twice over: once as a companion
    # of its parent session, and once in its own right, because discovery walks
    # projects/ and finds the file directly. Both copies are the same bytes, so
    # emitting both would index the conversation twice.
    emitted: set[str] = set()
    try:
        tombs = store.tombstones(refresh=True)
        for entry in _manifest_dirs():
            manifest = _read_manifest(entry)
            provider = manifest.get("provider")
            surface = manifest.get("surface")
            live_path = manifest.get("live_path")
            if not provider or not surface or not live_path:
                continue
            key = manifest.get("archive_key") or ""
            if any(tombs.get(_norm_target(t)) == "exclude_all"
                   for t in (live_path, key, str(entry))):
                continue
            transcript = entry / "transcript.jsonl"
            if transcript.is_file() and _norm_target(live_path) not in emitted:
                emitted.add(_norm_target(live_path))
                live_exists = os.path.exists(live_path)
                out.append(source_factory(provider, surface, str(transcript), live_path,
                                          "archive" if live_exists else "archive_only"))
            if include_versions:
                for old in sorted((entry / "versions").glob("*.jsonl")):
                    out.append(source_factory(provider, surface, str(old), None, "archive_version"))
            # Companions are indexed against the vendor path they were copied
            # from, so the archived copy and the live file are one session
            # rather than two. The mapping is recorded at sync time; without it
            # a subagent transcript would be indexed twice, once per origin.
            comp_map = manifest.get("companions") or {}
            for comp in sorted((entry / "companions").glob("*.jsonl")):
                comp_live = comp_map.get(comp.name)
                if not comp_live or _norm_target(comp_live) in emitted:
                    continue
                emitted.add(_norm_target(comp_live))
                exists = os.path.exists(comp_live)
                out.append(source_factory(provider, surface, str(comp), comp_live,
                                          "archive" if exists else "archive_only"))
        return out
    finally:
        if owned:
            store.close()


# --------------------------------------------------------------- deletion


def delete_archive(store: ArchiveStore, key_or_path: str, mode: str = "archive_only",
                   reason: str | None = None) -> dict[str, Any]:
    """Remove an archived copy and write the tombstone that keeps it removed.

    Without the tombstone this is pointless: the next sync copies the same bytes
    back from the vendor store within minutes.
    """
    row = store.get(key_or_path)
    if row is None:
        row = one(store.conn.execute(
            "SELECT * FROM archives WHERE live_path = ? OR archive_path = ?",
            (key_or_path, key_or_path)))
    if row is None:
        # Nothing archived under that name, but the tombstone still applies -
        # it is how you stop a file being archived in the first place.
        store.add_tombstone(key_or_path, mode, reason)
        return {"deleted": False, "tombstoned": key_or_path, "mode": mode,
                "note": "no archive found; future syncs will skip this target"}

    key = row["archive_key"]
    dir_path = archive_dir(key)
    removed_bytes = int(row["bytes_copied"] or 0)
    try:
        shutil.rmtree(dir_path)
    except OSError as exc:
        return {"deleted": False, "error": f"could not remove {dir_path}: {exc}"}
    store.drop(key)
    store.add_tombstone(row["live_path"], mode, reason)
    store.add_tombstone(key, mode, reason)
    return {"deleted": True, "archive_key": key, "path": str(dir_path),
            "bytes_freed": removed_bytes, "mode": mode,
            "live_path": row["live_path"], "reason": reason}


def prune_retention(store: ArchiveStore) -> dict[str, Any]:
    """Apply ICN_TRANSCRIPTS_ARCHIVE_RETENTION_DAYS. Does nothing by default.

    Retention here deletes nothing while the value is 0, which is deliberate:
    the archive's whole purpose is outliving the vendors' own retention, so a
    default that expired data would defeat it.
    """
    days = retention_days()
    if days <= 0:
        return {"pruned": 0, "retention_days": "unlimited"}
    cutoff = time.time() - days * 86400
    pruned, freed = 0, 0
    for row in store.list_archives(limit=1000000):
        try:
            last = datetime.fromisoformat(row["last_sync"]).timestamp()
        except (ValueError, TypeError):
            continue
        if last >= cutoff:
            continue
        # Retention prunes without a tombstone: this is age-based expiry, not a
        # decision that the content is junk, so re-archiving later is correct.
        try:
            shutil.rmtree(archive_dir(row["archive_key"]))
        except OSError:
            continue
        store.drop(row["archive_key"])
        pruned += 1
        freed += int(row["bytes_copied"] or 0)
    return {"pruned": pruned, "bytes_freed": freed, "retention_days": days}
