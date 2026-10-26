"""CLI integration test for `smpl groove-feel` (vault-16dd.7) — real subprocesses.

A synthetic click loop with a `.provenance.json` sidecar: intent is found next to the source,
the frames come out in order, `--text-out` carries the step grid, `--ref` adds the deltas.
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

SR = 44100
STEP = 0.125                       # 120 bpm, 16 steps of 4/4
RECIPE = {"bpm": 120, "beats_per_bar": 4, "grid_steps": 16, "swing": 0.0,
          "tracks": [{"name": "kick", "hits": [{"bar": b, "step": s, "velocity": v}
                                                for b in (1, 2) for s, v in ((1, 1.0), (7, 0.6), (11, 0.3))]}]}


@pytest.fixture()
def env(tmp_path):
    e = dict(os.environ)
    e["SMPL_CAS_DIR"] = str(tmp_path / "cas")
    e.pop("VIRTUAL_ENV", None)
    return e


def _loop(path, hits):
    n = int(SR * 4.05)
    y = np.zeros(n, dtype="float32")
    t = np.arange(int(SR * 0.03)) / SR
    click = (np.exp(-t * 200) * np.sin(2 * np.pi * 1500 * t)).astype("float32")
    for bar, step in hits:
        i = round((((bar - 1) * 16 + step - 1) * STEP) * SR)
        y[i:i + len(click)] += 0.5 * click
    sf.write(str(path), y, SR, subtype="FLOAT")


def _frames(out: bytes):
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def _run(args, env, stdin=None):
    return subprocess.run(args, input=stdin, capture_output=True, env=env, timeout=300)


def test_groove_feel_pipe_stage_reads_the_sidecar_and_writes_text(tmp_path, env):
    wav = tmp_path / "loop.wav"
    _loop(wav, [(b, s) for b in (1, 2) for s in (1, 7, 11)])
    (tmp_path / "loop.wav.provenance.json").write_text(json.dumps({"bpm": 120, "bars": 2, "recipe": RECIPE}))
    ref = tmp_path / "ref.wav"
    _loop(ref, [(b, s) for b in (1, 2) for s in (1, 7, 11)])

    read = _run([SMPL, "read", str(wav)], env)
    assert read.returncode == 0
    text = tmp_path / "feel.txt"
    out = _run([SMPL, "groove-feel", "--ref", str(ref), "--text-out", str(text),
                "--out", str(tmp_path / "feel.json")], env, stdin=read.stdout)
    assert out.returncode == 0, out.stderr.decode()
    frames = _frames(out.stdout)
    assert [f["kind"] for f in frames] == ["audio", "feature", "feature"]
    assert frames[0].get("role") == "source"                       # passthrough first
    doc, feats = frames[1]["data"], frames[2]["data"]
    assert frames[1]["role"] == "groove-feel" and frames[2]["role"] == "groove-feel-features"
    assert doc["source"] == "intent" and doc["measured"] is True
    assert set(feats) >= {"rhythm.syncopation_lhl", "rhythm.syncopation_norm",
                          "rhythm.microtiming_abs_ms", "rhythm.groove_ref_delta_ms"}
    assert feats["rhythm.microtiming_abs_ms"] < 2.0                # clicks sit on the grid
    body = text.read_text()
    assert "kick" in body
    assert "|X... ..x. ..o. ....|" in body                         # X at 1, x at 7, ghost at 11
    assert json.loads((tmp_path / "feel.json").read_text())["text"] == body


def test_groove_feel_without_intent_falls_back_to_detected_onsets(tmp_path, env):
    wav = tmp_path / "bare.wav"
    _loop(wav, [(b, s) for b in (1, 2) for s in (1, 5, 9, 13)])
    read = _run([SMPL, "read", str(wav)], env)
    out = _run([SMPL, "groove-feel", "--bpm", "120"], env, stdin=read.stdout)
    assert out.returncode == 0, out.stderr.decode()
    doc = _frames(out.stdout)[1]["data"]
    assert doc["source"] == "detected" and list(doc["tracks"]) == ["all"]
