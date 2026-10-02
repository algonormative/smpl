"""CLI integration tests: the real composable pipes via subprocess against installed shims.

Exercises the contract end-to-end — read → cat passthrough, the raw-WAV bridge round-trip,
resolve, gc safety, multicall (`smpl-cat` ≡ `smpl cat`), and PATH-discovery fall-through.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

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


@pytest.fixture()
def tone(tmp_path):
    p = tmp_path / "tone.wav"
    t = np.arange(22050) / 44100
    sf.write(str(p), (0.5 * np.sin(2 * np.pi * 220 * t)).astype("float32"), 44100, subtype="PCM_16")
    return str(p)


def _run(args, env, stdin=None):
    return subprocess.run(args, input=stdin, capture_output=True, env=env, timeout=60)


def _frames(out: bytes):
    return [json.loads(l) for l in out.splitlines() if l.strip()]


def test_read_emits_audio_frame(env, tone):
    r = _run(["smpl", "read", tone], env)
    assert r.returncode == 0, r.stderr
    frames = _frames(r.stdout)
    assert frames[0]["kind"] == "audio"
    assert frames[0]["role"] == "source"
    assert frames[0]["hash"].startswith("blake3:")


def test_read_cat_passthrough(env, tone):
    p1 = _run(["smpl", "read", tone], env)
    p2 = _run(["smpl", "cat"], env, stdin=p1.stdout)
    kinds = [f["kind"] for f in _frames(p2.stdout)]
    # Passthrough first (audio), then derived (feature + text).
    assert kinds[0] == "audio"
    assert "feature" in kinds and "text" in kinds


def test_multicall_shim_equiv(env, tone):
    src = _run(["smpl", "read", tone], env).stdout
    via_sub = _run(["smpl", "cat"], env, stdin=src).stdout
    via_shim = _run(["smpl-cat"], env, stdin=src).stdout
    assert len(_frames(via_sub)) == len(_frames(via_shim))


def test_raw_wav_bridge_roundtrip(env, tone):
    """as-wav → (identity cat) → from-wav preserves duration + reattaches provenance."""
    src = _run(["smpl", "read", tone], env).stdout
    wav = _run(["smpl", "as-wav"], env, stdin=src).stdout
    assert wav[:4] == b"RIFF" and len(wav) > 1000
    fw = _run(["smpl", "from-wav", "--role", "x.wet", "--derives-from", "source"], env, stdin=wav)
    frame = _frames(fw.stdout)[0]
    assert frame["kind"] == "audio" and frame["op"] == "from-wav"
    assert frame["meta"]["dur"] == pytest.approx(0.5, abs=0.05)
    assert frame["params"]["derives_from"] == "source"  # role → provenance, not dangling lineage


def test_resolve_role_path(env, tone):
    src = _run(["smpl", "read", tone], env).stdout
    r = _run(["smpl", "resolve", "--role", "source"], env, stdin=src)
    assert r.returncode == 0
    assert r.stdout.decode().strip().endswith(".wav")


def test_gc_keeps_referenced(env, tone):
    src = _run(["smpl", "read", tone], env).stdout
    r = _run(["smpl", "gc", "--json"], env, stdin=src)
    summary = json.loads(r.stdout)
    assert summary["dry_run"] is True
    assert len(summary["removed"]) == 0  # the one blob is referenced + in grace


def test_unknown_command_path_discovery(env):
    # A non-built-in with no `smpl-<x>` on PATH → exit 127 with a helpful message.
    r = _run(["smpl", "definitely-not-a-command"], env)
    assert r.returncode == 127
    assert b"not a built-in" in r.stderr


@pytest.mark.parametrize(
    ("name", "project"),
    [
        ("stems", "smpl-stems"),
        ("transcribe", "smpl-transcribe"),
        ("embed", "smpl-embed"),
        ("gen", "smpl-gen"),
        ("cloud", "smpl-cloud"),
        ("synth", "smpl-synth"),
        # The MIDI twins: two console scripts, one `tools/smpl-midi` project.
        ("transcribe-midi", "smpl-midi"),
        ("render-midi", "smpl-midi"),
    ],
)
def test_missing_first_party_tool_prints_install_command(env, tmp_path, name, project):
    # A missing heavy tool must hand back the exact copy-pasteable install line, not just its
    # name. PATH is emptied so an actually-installed `smpl-<x>` can't be exec'd instead.
    e = dict(env)
    e["PATH"] = str(tmp_path)
    r = _run([SMPL, name], e)
    assert r.returncode == 127
    expected = (
        f"uv tool install git+https://github.com/chronick/smpl#subdirectory=tools/{project}"
    )
    assert expected in r.stderr.decode(), r.stderr


def test_missing_unknown_tool_has_no_install_command(env, tmp_path):
    # We can't invent a source for someone else's `smpl-<x>` — generic message only.
    e = dict(env)
    e["PATH"] = str(tmp_path)
    r = _run([SMPL, "frobnicate"], e)
    assert r.returncode == 127
    err = r.stderr.decode()
    assert b"uv tool install" not in r.stderr
    assert "not a built-in" in err and "install the tool that provides it" in err


def test_help_lists_builtins(env):
    r = _run(["smpl", "--help"], env)
    assert b"read" in r.stdout and b"as-wav" in r.stdout and b"external commands" in r.stdout


def test_external_discovery_does_not_exec_loop(env, tmp_path):
    # A `smpl-<x>` shim that re-invokes `smpl <x>` would spin forever without the guard.
    # Plant exactly such a recursive shim and assert the env-sentinel breaks the loop (127),
    # not a 60s timeout. This is the regression test for the exec-loop blocker.
    smpl = shutil.which("smpl")
    shim = tmp_path / "smpl-loopy"
    shim.write_text(f'#!/bin/sh\nexec "{smpl}" loopy "$@"\n')
    shim.chmod(0o755)
    e = dict(env)
    e["PATH"] = f"{tmp_path}:{e['PATH']}"
    r = _run(["smpl", "loopy"], e)  # not built-in → execs smpl-loopy → re-enters → guard
    assert r.returncode == 127
    assert b"recurse" in r.stderr or b"unknown" in r.stderr


def test_id_collision_emits_error_frame(env, tone):
    # Two distinct frames sharing an id on stdin → an id_collision error frame is emitted.
    src = _run(["smpl", "read", tone], env).stdout
    frame = json.loads(src.splitlines()[0])
    dup = dict(frame)
    dup["role"] = "different"  # distinct frame, same id
    dup_line = json.dumps({**dup, "id": frame["id"]}).encode()
    collided = src + b"\n" + dup_line + b"\n"
    out = _run(["smpl", "resolve", frame["id"]], env, stdin=collided)
    assert any(json.loads(l).get("kind") == "error" and json.loads(l)["data"]["code"] == "id_collision"
               for l in out.stdout.splitlines() if l.strip()) or out.returncode != 0


def _loudness_feature(out: bytes):
    return next(f for f in _frames(out) if f["kind"] == "feature" and f["role"] == "loudness")


def test_loudness_pipe_is_memoized(env, tone):
    """A warm re-run of the same pipe reports a cache hit — the memo lookup is live."""
    src = _run(["smpl", "read", tone], env).stdout

    cold = _run(["smpl", "loudness"], env, stdin=src)
    assert cold.returncode == 0, cold.stderr
    cold_feat = _loudness_feature(cold.stdout)
    assert cold_feat["params"]["cache_hit"] is False

    warm = _run(["smpl", "loudness"], env, stdin=src)
    warm_feat = _loudness_feature(warm.stdout)
    assert warm_feat["params"]["cache_hit"] is True
    assert warm_feat["data"] == cold_feat["data"]  # same measurement, no recompute


def test_loudness_no_cache_flag_bypasses(env, tone):
    src = _run(["smpl", "read", tone], env).stdout
    _run(["smpl", "loudness"], env, stdin=src)  # warm the entry

    bypass = _run(["smpl", "loudness", "--no-cache"], env, stdin=src)
    assert bypass.returncode == 0, bypass.stderr
    assert _loudness_feature(bypass.stdout)["params"]["cache_hit"] is False
