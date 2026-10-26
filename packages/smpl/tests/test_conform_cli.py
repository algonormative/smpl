"""CLI integration tests for `smpl conform` (vault-16dd.2) — real subprocesses."""

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

SR = 22050
TIMES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]


@pytest.fixture()
def env(tmp_path):
    e = dict(os.environ)
    e["SMPL_CAS_DIR"] = str(tmp_path / "cas")
    e.pop("VIRTUAL_ENV", None)
    return e


def _run(args, env, stdin=None):
    return subprocess.run(args, input=stdin, capture_output=True, env=env, timeout=300)


def _frames(out: bytes):
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def _loop(tmp_path, late_ms, sidecar=True):
    t = np.arange(int(SR * 0.1)) / SR
    hit = np.exp(-t * 30) * np.sin(2 * np.pi * 80 * t) + 0.5 * np.exp(-t * 300) * np.sin(2 * np.pi * 1800 * t)
    y = np.zeros(SR * 3)
    for h in TIMES:
        i = round((h + late_ms / 1000) * SR)
        y[i:i + len(hit)] += hit
    wav = tmp_path / "loop.wav"
    sf.write(str(wav), y, SR, subtype="FLOAT")
    if sidecar:
        (tmp_path / "loop.wav.intent.json").write_text(json.dumps(
            {"dur_s": 3.0, "pitch_classes": ["A", "G#"], "lufs": -20.0,
             "hits": [{"t": h, "track": "kick"} for h in TIMES]}))
    return str(wav)


def test_conform_as_a_pipe_stage(env, tmp_path):
    src = _run(["smpl", "read", _loop(tmp_path, 20)], env)
    assert src.returncode == 0, src.stderr
    r = _run(["smpl", "conform", "--allowed-pcs", "A,B,C,D,E,F,G", "--checks", "length,timing,pitch,loudness"],
             env, stdin=src.stdout)
    assert r.returncode == 0, r.stderr
    frames = _frames(r.stdout)
    assert frames[0]["kind"] == "audio"                       # passthrough first
    feat = next(f for f in frames if f["kind"] == "feature" and f["role"] == "conform")
    d = feat["data"]
    assert feat["of"] == frames[0]["id"] and feat["op"] == "conform"
    assert d["conform.length_delta_ms"] == 0.0
    assert d["conform.timing_median_ms"] == pytest.approx(20.0, abs=1.5)
    assert d["conform.timing_mean_ms"] > 0                     # late
    assert d["conform.pc_outside_allowed"] == ["G#"]
    assert d["conform.lufs_intended"] == -20.0 and d["conform.lufs_delta"] is not None
    assert {"allowed", "timing"} <= set(d["conform.over_tol"])
    marker = next(f for f in frames if f["kind"] == "marker" and f["role"] == "conform")
    assert len(marker["data"]) == len(TIMES) and "kick" in marker["data"][0]["label"]


def test_conform_tolerance_flag_silences_markers(env, tmp_path):
    src = _run(["smpl", "read", _loop(tmp_path, 20)], env)
    r = _run(["smpl", "conform", "--timing-tol-ms", "30", "--checks", "timing"], env, stdin=src.stdout)
    assert r.returncode == 0, r.stderr
    assert not [f for f in _frames(r.stdout) if f["kind"] == "marker"]


def test_conform_without_intent_reports_none_and_cli_overrides_apply(env, tmp_path):
    src = _run(["smpl", "read", _loop(tmp_path, 0, sidecar=False)], env)
    r = _run(["smpl", "conform", "--lufs", "-14", "--true-peak-db", "-1"], env, stdin=src.stdout)
    assert r.returncode == 0, r.stderr
    d = next(f for f in _frames(r.stdout) if f["kind"] == "feature")["data"]
    assert d["conform.length_delta_ms"] is None and d["conform.timing_median_ms"] is None
    assert d["conform.lufs_intended"] == -14.0 and d["conform.true_peak_intended_dbtp"] == -1.0


def test_conform_rejects_bad_flags_with_an_error_frame(env, tmp_path):
    src = _run(["smpl", "read", _loop(tmp_path, 0)], env)
    r = _run(["smpl", "conform", "--checks", "vibes"], env, stdin=src.stdout)
    assert r.returncode == 2 and any(f["kind"] == "error" for f in _frames(r.stdout))
    r = _run(["smpl", "conform", "--allowed-pcs", "H"], env, stdin=src.stdout)
    assert r.returncode == 2


def test_conform_is_listed_as_a_builtin(env):
    assert b"conform" in _run(["smpl", "--help"], env).stdout
