"""CLI integration tests for `smpl arc` (ticket vault-3te4) — real subprocesses.

Covers both entry shapes (pipe stage and path-in), the emitted frame ordering, the
`--out` sidecars, and the usage failures. The narrative fixture is shared with the
smpl-analysis tests; the audio is synthesized here so the test is self-contained.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

SMPL = shutil.which("smpl")
pytestmark = pytest.mark.skipif(SMPL is None, reason="`smpl` console script not on PATH")

NARRATIVE = str(Path(__file__).resolve().parents[2] / "smpl-analysis" / "tests"
                / "fixtures" / "arc_narrative.yaml")
SR = 16000


@pytest.fixture()
def env(tmp_path):
    e = dict(os.environ)
    e["SMPL_CAS_DIR"] = str(tmp_path / "cas")
    e.pop("VIRTUAL_ENV", None)
    return e


@pytest.fixture()
def recording(tmp_path):
    """A short set-shaped recording: quiet · pad · loud wall · quiet tail."""
    rng = np.random.default_rng(11)
    n = int(SR * 15.0)
    t = np.arange(n) / SR
    y = np.concatenate([
        0.03 * np.sin(2 * np.pi * 110 * t),
        0.15 * (np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 330 * t)),
        0.9 * np.tanh(3.0 * (0.4 * rng.standard_normal(n) + 0.3 * np.sin(2 * np.pi * 55 * t))),
        0.02 * np.sin(2 * np.pi * 80 * t),
    ]).astype("float32")
    p = tmp_path / "set.wav"
    sf.write(str(p), y, SR, subtype="FLOAT")
    return str(p)


def _run(args, env, stdin=None):
    return subprocess.run(args, input=stdin, capture_output=True, env=env, timeout=300)


def _frames(out: bytes):
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def test_arc_as_a_pipe_stage(env, recording):
    src = _run(["smpl", "read", recording], env)
    assert src.returncode == 0, src.stderr
    r = _run(["smpl", "arc", "--narrative", NARRATIVE], env, stdin=src.stdout)
    assert r.returncode == 0, r.stderr

    frames = _frames(r.stdout)
    assert frames[0]["kind"] == "audio"          # passthrough first (spec: stream ordering)
    audio_id = frames[0]["id"]
    derived = frames[1:]
    assert derived[0]["kind"] == "image" and derived[0]["role"] == "arc:overlay"
    assert derived[-1]["role"] == "arc:summary"

    sections = [f for f in derived if f.get("role") == "arc:section"]
    assert sections, "at least one per-section feature frame"
    for f in derived:
        assert f["of"] == audio_id
        assert f["op"] == "arc" and f["op_version"] == "arc@1"
        assert f["params"]["narrative"] == NARRATIVE
    for s in sections:
        assert s["kind"] == "feature"
        assert {"index", "t0", "t1", "anchor", "measured_energy",
                "intended_tension", "divergence"} <= set(s["data"])
        assert "arc.section.crest_db" in s["data"]


def test_arc_path_in_mode_ingests_analyzes_and_writes_sidecars(env, recording, tmp_path):
    prefix = str(tmp_path / "arc")
    r = _run(["smpl", "arc", recording, "--narrative", NARRATIVE, "--out", prefix], env)
    assert r.returncode == 0, r.stderr
    frames = _frames(r.stdout)
    assert frames[0]["kind"] == "audio" and frames[0]["role"] == "source"
    assert any(f.get("role") == "arc:overlay" for f in frames)

    assert Path(f"{prefix}.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    rows = [json.loads(x) for x in
            Path(f"{prefix}.sections.ndjson").read_text(encoding="utf-8").splitlines() if x.strip()]
    assert rows and all(row["role"] == "arc:section" for row in rows)


def test_arc_sections_override(env, recording):
    r = _run(["smpl", "arc", recording, "--narrative", NARRATIVE, "--sections", "2"], env)
    assert r.returncode == 0, r.stderr
    sections = [f for f in _frames(r.stdout) if f.get("role") == "arc:section"]
    assert 0 < len(sections) <= 2


def test_arc_without_audio_is_a_usage_failure(env):
    r = _run(["smpl", "arc", "--narrative", NARRATIVE], env, stdin=b"")
    assert r.returncode == 1 and b"no audio frame" in r.stderr


def test_arc_with_a_bad_narrative_emits_an_error_frame(env, recording, tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema: narrative/v1\nset: x\n", encoding="utf-8")
    r = _run(["smpl", "arc", recording, "--narrative", str(bad)], env)
    assert r.returncode == 1
    errors = [f for f in _frames(r.stdout) if f["kind"] == "error"]
    assert errors and errors[0]["data"]["code"] == "op_failed"


def test_arc_is_listed_as_a_builtin(env):
    assert b"arc" in _run(["smpl", "--help"], env).stdout
