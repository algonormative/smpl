"""`smpl crosscheck` subcommand wiring: stdin frames in, verdict frames out, stub claims only."""

from __future__ import annotations

import argparse
import json

import pytest
from smpl_cli.subcommands import crosscheck as cmd
from smplstream import frames as F


def _run(monkeypatch, capsys, frames, *argv):
    monkeypatch.setattr(cmd, "read_stdin_frames", lambda: frames)
    p = argparse.ArgumentParser()
    cmd.add_arguments(p)
    rc = cmd.run(p.parse_args(list(argv)))
    out = capsys.readouterr()
    return rc, [json.loads(line) for line in out.out.splitlines() if line.strip()], out.err


@pytest.fixture()
def stream():
    a = F.audio_frame("blake3:" + "1" * 64, sr=44100, ch=1, dur=4.0, role="source")
    beats = F.feature_frame({"rhythm.bpm": 128.0, "rhythm.bpm_candidates": []}, role="beats", of=a["id"])
    claim = F.text_frame("tempo 90", role="claim", of=a["id"], params={"question_id": "tempo_bpm", "answer": 90})
    return [a, beats, claim]


def test_marks_and_reports_a_contradicted_claim(monkeypatch, capsys, stream):
    rc, out, err = _run(monkeypatch, capsys, stream, "--report")
    assert rc == 0 and out[:3] == stream
    res = next(f for f in out if f.get("role") == "crosscheck")["data"]
    assert res["status"] == "contradicted" and res["claim_id"] == stream[2]["id"]
    assert "contradicted" in err and "tempo_bpm" in err


def test_drop_contradicted_and_tolerance_flag(monkeypatch, capsys, stream):
    _, out, _ = _run(monkeypatch, capsys, stream, "--drop-contradicted")
    assert stream[2]["id"] not in {f["id"] for f in out}
    _, out, _ = _run(monkeypatch, capsys, stream, "--tol", "bpm_rel=0.5")     # 90 vs 128 is 30% off
    assert next(f for f in out if f.get("role") == "crosscheck")["data"]["status"] == "supported"


def test_bad_tolerance_is_a_usage_error(monkeypatch, capsys, stream):
    rc, out, err = _run(monkeypatch, capsys, stream, "--tol", "nope=1")
    assert rc == 2 and out == [] and "bad --tol" in err
