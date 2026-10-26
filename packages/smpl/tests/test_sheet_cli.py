"""CLI integration tests for `smpl sheet` (vault-16dd.3) — real subprocesses.

Path-in and pipe-stage entry, `--out` files, `--ab` on two piped frames, and the usage
failure when `--ab` gets one clip.
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
SR = 16000


@pytest.fixture()
def env(tmp_path):
    e = dict(os.environ)
    e["SMPL_CAS_DIR"] = str(tmp_path / "cas")
    e.pop("VIRTUAL_ENV", None)
    return e


def _wav(tmp_path, name, amp):
    t = np.arange(SR * 4) / SR
    p = tmp_path / name
    sf.write(str(p), (amp * np.sin(2 * np.pi * 220 * t)).astype("float32"), SR, subtype="FLOAT")
    return str(p)


def _run(args, env, stdin=None):
    return subprocess.run(args, input=stdin, stdin=None if stdin is not None else subprocess.DEVNULL,
                          capture_output=True, env=env, timeout=300, check=False)


def _frames(out: bytes):
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def test_path_in_writes_png_json_md(env, tmp_path):
    src = _wav(tmp_path, "tone.wav", 0.2)
    out_dir = tmp_path / "sheets"
    r = _run(["smpl", "sheet", src, "--bpm", "120", "--out", str(out_dir)], env)
    assert r.returncode == 0, r.stderr
    frames = _frames(r.stdout)
    assert frames[0]["kind"] == "audio"
    assert [f["role"] for f in frames[1:]] == ["sheet", "sheet", "sheet:summary"]
    assert (out_dir / "tone.sheet.png").read_bytes()[:4] == b"\x89PNG"
    doc = json.loads((out_dir / "tone.sheet.json").read_text())
    assert doc["grid"]["source"] == "flag" and doc["grid"]["n"] == 2
    assert "| bar |" in (out_dir / "tone.sheet.md").read_text()


def test_ab_on_two_piped_frames(env, tmp_path):
    a, b = _wav(tmp_path, "a.wav", 0.2), _wav(tmp_path, "b.wav", 0.1)
    src = _run(["smpl", "read", a, b], env)
    assert src.returncode == 0, src.stderr
    r = _run(["smpl", "sheet", "--ab", "--bpm", "120"], env, stdin=src.stdout)
    assert r.returncode == 0, r.stderr
    frames = _frames(r.stdout)
    feat = next(f for f in frames if f["kind"] == "feature" and f["role"] == "sheet:ab")
    assert feat["data"]["ab"]["gain_db_a"] == pytest.approx(-6.02, abs=0.05)
    assert len(feat["lineage"]) == 2


def test_ab_with_one_clip_fails(env, tmp_path):
    r = _run(["smpl", "sheet", "--ab", _wav(tmp_path, "a.wav", 0.2)], env)
    assert r.returncode == 1
    assert any(f["kind"] == "error" for f in _frames(r.stdout))
