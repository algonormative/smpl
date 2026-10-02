"""Tests for smplstream.memostore — the memo_key → CAS mapping behind live memoization."""

from __future__ import annotations

import json

import pytest

from smplstream import cas, memo, memostore
from smplstream.errors import PathSafetyError


@pytest.fixture()
def key():
    return memo.memo_key("demo", "demo@1", ["blake3:" + "ab" * 32], params={"x": 1})


def test_lookup_miss_on_unrecorded_key(isolated_cas, key):
    assert memostore.lookup(key) is None
    assert memostore.get_json(key) is None


def test_put_then_get_json_roundtrips(isolated_cas, key):
    h = memostore.put_json(key, {"a": 1, "b": [2, 3]}, op="demo", op_version="demo@1")
    assert h.startswith("blake3:")
    assert memostore.get_json(key) == {"a": 1, "b": [2, 3]}

    entry = memostore.lookup(key)
    assert entry["hash"] == h
    assert entry["op"] == "demo"
    assert entry["op_version"] == "demo@1"
    assert entry["media"] == "application/json"


def test_record_maps_key_to_existing_blob(isolated_cas, key):
    h = cas.put_blob(b"\x89PNG\r\n\x1a\n-not-really", "image/png")
    memostore.record(key, h, media="image/png", op="demo", op_version="demo@1")
    assert memostore.lookup(key)["hash"] == h


def test_distinct_params_are_distinct_entries(isolated_cas):
    k1 = memo.memo_key("demo", "demo@1", ["blake3:" + "ab" * 32], params={"x": 1})
    k2 = memo.memo_key("demo", "demo@1", ["blake3:" + "ab" * 32], params={"x": 2})
    memostore.put_json(k1, {"which": 1})
    assert memostore.get_json(k1) == {"which": 1}
    assert memostore.get_json(k2) is None  # a different key must not serve k1's result


def test_op_version_bump_invalidates(isolated_cas):
    k1 = memo.memo_key("demo", "demo@1", ["blake3:" + "ab" * 32])
    k2 = memo.memo_key("demo", "demo@2", ["blake3:" + "ab" * 32])
    memostore.put_json(k1, {"v": 1})
    assert memostore.get_json(k2) is None


def test_collected_blob_degrades_to_miss(isolated_cas, key):
    """A recorded entry whose blob is gone is a MISS, never a dangling reference."""
    h = memostore.put_json(key, {"a": 1})
    assert memostore.lookup(key) is not None
    for path in (cas.get_path(h), isolated_cas / (h.split(":")[1][:2]) / f"{h.split(':')[1]}.meta.json"):
        path.unlink()
    assert memostore.lookup(key) is None
    assert memostore.get_json(key) is None


def test_corrupt_entry_is_a_miss_not_an_error(isolated_cas, key):
    memostore.put_json(key, {"a": 1})
    entry_path = next(memostore.memo_dir().rglob("*.json"))
    entry_path.write_text("{not json")
    assert memostore.lookup(key) is None


def test_malformed_key_is_rejected(isolated_cas):
    for bad in ("blake3:../../etc/passwd", "not-a-key", "blake3:zz" + "ab" * 31):
        with pytest.raises(PathSafetyError):
            memostore.lookup(bad)


def test_index_lives_under_the_cas_root_and_follows_it(isolated_cas, tmp_path, monkeypatch, key):
    """The index belongs to the CAS it points into: repoint the CAS, get a cold cache."""
    memostore.put_json(key, {"a": 1})
    assert memostore.memo_dir() == isolated_cas / ".memo"
    assert json.loads(next(memostore.memo_dir().rglob("*.json")).read_text())["memo_key"] == key

    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "other-cas"))
    assert memostore.get_json(key) is None


def test_memo_dir_env_override(isolated_cas, tmp_path, monkeypatch, key):
    monkeypatch.setenv("SMPL_MEMO_DIR", str(tmp_path / "elsewhere"))
    memostore.put_json(key, {"a": 1})
    assert memostore.memo_dir() == tmp_path / "elsewhere"
    assert (tmp_path / "elsewhere").exists()
    assert memostore.get_json(key) == {"a": 1}


def test_gc_does_not_walk_the_memo_index(isolated_cas, key):
    """`.memo` must not look like a CAS shard — gc iterates 2-char shard dirs only."""
    memostore.put_json(key, {"a": 1})
    blobs = list(cas.iter_blobs())
    assert len(blobs) == 1  # the JSON payload only; no index entry mistaken for a blob
