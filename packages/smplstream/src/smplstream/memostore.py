"""Memo cache store (spec → *Memoization*, NORMATIVE).

:mod:`smplstream.memo` computes the key; this module is the **store** that key indexes —
the piece that makes memoization live behavior rather than protocol design. It maps

    memo_key  →  CAS hash of the op's output

so a cacheable op can look its key up, and on a hit emit the cached output *without
computing*. The heavy bytes always live in the CAS (:mod:`smplstream.cas`); this index
holds nothing but the mapping, so GC and integrity rules keep applying unchanged.

Layout: ``<cas_dir>/.memo/<aa>/<memo-key-hex>.json`` (override the whole directory with
``SMPL_MEMO_DIR``). It lives *inside* the CAS root by default because an entry is only
meaningful for the CAS it points into — point ``SMPL_CAS_DIR`` somewhere else and the memo
index follows, so a cache can never serve a hash the store doesn't hold. The dotted name
keeps it out of :func:`smplstream.cas.iter_blobs` (which only walks 2-char shard dirs), so
``smpl gc`` never mistakes an index entry for a blob.

One small file per key (atomic temp-write + ``rename``), NOT one shared index document:
memoization plus fan-out pipes routinely run the same op concurrently, and a shared
read-modify-write index loses entries under exactly that load.

A lookup verifies the referenced blob is still present (``cas.exists``) before reporting a
hit, so a GC'd blob degrades to a miss (recompute) instead of a dangling reference.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from . import cas
from .errors import PathSafetyError

KEY_RE = re.compile(r"^blake3:[0-9a-f]{64}$")


def memo_dir() -> Path:
    """Resolve the memo-index root (reads the env each call so tests can override)."""
    override = os.environ.get("SMPL_MEMO_DIR")
    if override:
        return Path(override).expanduser()
    return cas.cas_dir() / ".memo"


def _entry_path(key: str) -> Path:
    if not isinstance(key, str) or not KEY_RE.match(key):
        raise PathSafetyError(f"unsafe or malformed memo key: {key!r}")
    hexd = key.split(":", 1)[1]
    return memo_dir() / hexd[:2] / f"{hexd}.json"


def _atomic_write(dest: Path, data: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(dest.parent), prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, dest)  # atomic within the same directory
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def lookup(key: str) -> Optional[dict]:
    """Return the recorded entry for ``key``, or None on a miss.

    A recorded entry whose blob is gone (GC, moved CAS) is treated as a **miss** — the
    caller recomputes rather than resolving a dangling hash.
    """
    path = _entry_path(key)
    if not path.exists():
        return None
    try:
        entry = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None  # a corrupt entry is a miss, never an error
    h = entry.get("hash")
    if not isinstance(h, str) or not cas.exists(h):
        return None
    return entry


def record(
    key: str,
    cas_hash: str,
    *,
    media: str,
    op: Optional[str] = None,
    op_version: Optional[str] = None,
) -> dict:
    """Record ``memo_key → cas_hash``. Overwrites any prior entry (a forced recompute repairs)."""
    cas.validate_hash(cas_hash)
    entry = {
        "memo_key": key,
        "hash": cas_hash,
        "media": media,
        "op": op,
        "op_version": op_version,
        "recorded": round(time.time(), 3),
    }
    _atomic_write(_entry_path(key), json.dumps(entry, sort_keys=True).encode("utf-8"))
    return entry


def get_json(key: str) -> Optional[Any]:
    """Read the cached JSON payload for ``key``, or None on a miss/unreadable blob."""
    entry = lookup(key)
    if entry is None:
        return None
    try:
        return json.loads(cas.get_path(entry["hash"]).read_text())
    except (OSError, json.JSONDecodeError, FileNotFoundError):
        return None


def put_json(
    key: str, payload: Any, *, op: Optional[str] = None, op_version: Optional[str] = None
) -> str:
    """Store an op's JSON result in the CAS and record it under ``key``. Returns the CAS hash."""
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    h = cas.put_blob(blob, "application/json")
    record(key, h, media="application/json", op=op, op_version=op_version)
    return h
