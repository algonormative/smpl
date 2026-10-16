"""`smpl qc --report` / `--audiobox`: batch gate in one call (vault-1gdf1). Audiobox is stubbed."""

from __future__ import annotations

import json
import os
import subprocess
import shutil

import numpy as np
import pytest
import soundfile as sf

SR = 44100
SMPL = shutil.which("smpl")
pytestmark = pytest.mark.skipif(SMPL is None, reason="`smpl` console script not on PATH")


def _write(path, sig):
    sf.write(str(path), sig.astype("float32"), SR, subtype="PCM_16")
    return str(path)


@pytest.fixture()
def takes(tmp_path):
    t = np.arange(int(2.5 * SR)) / SR
    hit = 0.8 * np.exp(-t[: SR // 2] / 0.05) * np.sin(2 * np.pi * 150 * t[: SR // 2])
    long_hit = 0.8 * np.exp(-t / 0.05) * np.sin(2 * np.pi * 150 * t)
    swell = np.sin(np.pi * t / 2.5) * np.sin(2 * np.pi * 220 * t)
    pad = 0.4 * np.sin(2 * np.pi * 330 * t)
    return [_write(tmp_path / f"{n}.wav", s)
            for n, s in [("hit", hit), ("long", long_hit), ("swell", swell), ("pad", pad)]]


@pytest.fixture()
def env(tmp_path):
    e = dict(os.environ)
    e["SMPL_CAS_DIR"] = str(tmp_path / "cas")
    e.pop("VIRTUAL_ENV", None)
    return e


def _pipe(files, qc_args, env):
    src = subprocess.run(["smpl", "read", *files],
                         capture_output=True, env=env, timeout=120)
    assert src.returncode == 0, src.stderr
    return subprocess.run(["smpl", "qc", *qc_args], input=src.stdout,
                          capture_output=True, env=env, timeout=120)


def test_report_four_takes_one_document(takes, env):
    r = _pipe(takes, ["--report"], env)
    assert r.returncode == 0, r.stderr
    doc = json.loads(r.stdout)  # a single JSON document, not NDJSON
    files = doc["files"]
    assert [os.path.basename(f["file"]) for f in files] == ["hit.wav", "long.wav", "swell.wav", "pad.wav"]
    by = {os.path.basename(f["file"]): f for f in files}
    assert all(f["audiobox"] is None for f in files)
    assert by["hit.wav"]["attack_time_s"] < 0.02 and by["hit.wav"]["late_energy_share"] == 0.0
    assert by["long.wav"]["attack_time_s"] < 0.02 and by["long.wav"]["late_energy_share"] < 0.01
    assert by["swell.wav"]["attack_time_s"] == pytest.approx(1.25, abs=0.05)
    assert by["swell.wav"]["late_energy_share"] > 0.5
    assert by["pad.wav"]["late_energy_share"] > 0.5


def test_audiobox_missing_extra_fails_cleanly(takes, env):
    r = _pipe(takes[:1], ["--report", "--audiobox"], env)
    # extra is not installed in the test env -> clear message, no traceback
    if r.returncode == 0:
        pytest.skip("audiobox extra is installed here")
    assert r.returncode == 2
    assert b"Traceback" not in r.stderr and b"smpl-analysis[audiobox]" in r.stderr


def test_audiobox_stubbed_scores_in_report_and_frames(takes, env, monkeypatch, capsys):
    """In-process with the scorer stubbed: no torch, no model download."""
    import argparse

    from smpl_analysis import audiobox
    from smpl_cli.subcommands import qc as qc_cmd

    src = subprocess.run(["smpl", "read", *takes], capture_output=True, env=env, timeout=120)
    frames = [json.loads(l) for l in src.stdout.splitlines() if l.strip()]
    monkeypatch.setenv("SMPL_CAS_DIR", env["SMPL_CAS_DIR"])
    monkeypatch.setattr(qc_cmd, "read_stdin_frames", lambda: frames)
    monkeypatch.setattr(audiobox, "check_available", lambda: None)
    monkeypatch.setattr(audiobox, "score_path",
                        lambda p: {"CE": 5.0, "CU": 6.5, "PC": 3.0, "PQ": 7.0})

    p = argparse.ArgumentParser()
    qc_cmd.add_arguments(p)
    rc = qc_cmd.run(p.parse_args(["--report", "--audiobox"]))
    assert rc == 0
    files = json.loads(capsys.readouterr().out)["files"]
    assert len(files) == 4
    assert all(f["audiobox"] == {"CE": 5.0, "CU": 6.5, "PC": 3.0, "PQ": 7.0} for f in files)
    assert all(f["attack_time_s"] is not None for f in files)
