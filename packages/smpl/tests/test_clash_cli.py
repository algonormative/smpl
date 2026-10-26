"""CLI integration tests for `smpl clash` (ticket vault-16dd.4) — real subprocesses.

Two layers in, a pair out (document + scalars + markers, both layers named); one layer in,
the seam check; ``--seam`` on a pair; the usage failure. Audio is synthesized here.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

SMPL = shutil.which("smpl")
pytestmark = pytest.mark.skipif(SMPL is None, reason="`smpl` console script not on PATH")


@pytest.fixture()
def env(tmp_path):
    e = dict(os.environ)
    e["SMPL_CAS_DIR"] = str(tmp_path / "cas")
    e.pop("VIRTUAL_ENV", None)
    return e


def _wav(path, f, sr, n_samples=None, amp=0.5):
    n = n_samples or sr * 3
    sf.write(str(path), (amp * np.sin(2 * np.pi * f * np.arange(n) / sr)).astype("float32"), sr)
    return str(path)


def _run(args, env, stdin=b""):
    return subprocess.run(args, input=stdin, capture_output=True, env=env, timeout=300,
                          check=False)


def _frames(out):
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def _read(env, *paths, role="source"):
    r = _run(["smpl", "read", "--role", role, *paths], env)
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_two_layers_make_a_pair_that_names_both(env, tmp_path):
    drone = _wav(tmp_path / "drone.wav", 48, 44100)
    kick = _wav(tmp_path / "kick.wav", 50, 48000)             # different rates: both resampled
    stream = _read(env, drone) + _read(env, kick, role="kick")
    r = _run(["smpl", "clash", "--out", str(tmp_path / "c.json")], env, stdin=stream)
    assert r.returncode == 0, r.stderr
    frames = _frames(r.stdout)
    assert [f["kind"] for f in frames[:2]] == ["audio", "audio"]      # passthrough first
    ids = [f["id"] for f in frames[:2]]
    derived = frames[2:]
    doc = next(f for f in derived if f["kind"] == "feature" and f["role"] == "clash")
    feats = next(f for f in derived if f.get("role") == "clash-features")
    marks = next(f for f in derived if f["kind"] == "marker")
    assert doc["data"]["a"]["id"] == ids[0] and doc["data"]["b"]["id"] == ids[1]
    assert doc["data"]["pair"] == "source|kick"
    assert doc["lineage"] == ids and doc["op"] == "clash"
    assert feats["data"]["clash.sub_collision"] > 0.9
    assert marks["data"] and all("t" in p and "label" in p for p in marks["data"])
    assert not any(f.get("role") == "clash-seam" for f in derived)    # pair only, no --seam
    assert json.loads((tmp_path / "c.json").read_text())[0]["pair"] == "source|kick"


def test_a_single_layer_runs_the_seam_check(env, tmp_path):
    sr = 48000
    closed = _wav(tmp_path / "closed.wav", 240, sr, n_samples=sr)
    cut = _wav(tmp_path / "cut.wav", 240, sr, n_samples=sr - 70)
    for path, want in ((closed, False), (cut, True)):
        r = _run(["smpl", "clash"], env, stdin=_read(env, path))
        assert r.returncode == 0, r.stderr
        seam = next(f for f in _frames(r.stdout) if f.get("role") == "clash-seam"
                    and f["kind"] == "feature")
        assert seam["data"]["flagged"] is want
        assert set(seam["data"]["features"]) >= {"clash.seam_jump_ratio", "clash.seam_flagged"}


def test_seam_flag_adds_the_check_to_a_pair(env, tmp_path):
    stream = _read(env, _wav(tmp_path / "a.wav", 55, 48000)) \
        + _read(env, _wav(tmp_path / "b.wav", 60, 48000), role="kick")
    r = _run(["smpl", "clash", "--seam"], env, stdin=stream)
    roles = [f.get("role") for f in _frames(r.stdout) if f["kind"] == "feature"]
    assert roles.count("clash") == 1 and roles.count("clash-seam") == 2


def test_no_audio_is_an_error_frame(env):
    r = _run(["smpl", "clash"], env)
    assert r.returncode == 1
    assert _frames(r.stdout)[-1]["kind"] == "error"
