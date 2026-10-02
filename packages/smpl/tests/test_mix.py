"""`smpl mix` — the combinator (N audio frames → one rendered audio frame).

Covers the position grammar, N-input lineage, lazy `--dry-run` planning, memoized render,
sample-accurate placement against ``marker.sample``, the clip guard, and the session
round-trip (stateless verbs → the same render, deterministically).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from smpl_cli import mixsession as M

SMPL = shutil.which("smpl")
pytestmark = pytest.mark.skipif(SMPL is None, reason="`smpl` console script not on PATH")


@pytest.fixture()
def env(tmp_path):
    e = dict(os.environ)
    e["SMPL_CAS_DIR"] = str(tmp_path / "cas")
    e["SMPL_MIX_DIR"] = str(tmp_path / "mix")
    e.pop("VIRTUAL_ENV", None)
    return e


def _tone(path, *, freq=220.0, dur=0.5, sr=44100, amp=0.5):
    t = np.arange(int(sr * dur)) / sr
    sf.write(str(path), (amp * np.sin(2 * np.pi * freq * t)).astype("float32"), sr,
             subtype="FLOAT")
    return str(path)


@pytest.fixture()
def two_tones(tmp_path):
    return _tone(tmp_path / "a.wav", freq=220.0), _tone(tmp_path / "b.wav", freq=330.0)


def _run(args, env, stdin=None):
    return subprocess.run(args, input=stdin, capture_output=True, env=env, timeout=120)


def _frames(out: bytes):
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def _stream(env, a, b):
    """A two-audio-frame stream: role ``source`` (a) + role ``pad`` (b)."""
    first = _run(["smpl", "read", a], env).stdout
    return _run(["smpl", "read", "--role", "pad", b], env, stdin=first).stdout


def _mix_frame(frames):
    return [f for f in frames if f.get("op") == "mix" and f["kind"] == "audio"][-1]


# ---------------------------------------------------------------------------
# Position + source grammar (the sample-accurate timebase).
# ---------------------------------------------------------------------------
def test_at_grammar_forms():
    assert M.parse_at("sample:1024") == {"sample": 1024}
    assert M.parse_at("sec:1.5") == {"sec": 1.5}
    assert M.parse_at("1.5s") == {"sec": 1.5}
    assert M.parse_at("marker:beat#3") == {"marker": {"role": "beat", "index": 3}}
    assert M.parse_at("4410") == {"sample": 4410}


def test_bare_dotted_position_is_bars_not_seconds():
    """`smpl pattern` writes ``at: "1.3"`` meaning bar 1 beat 3 — never 1.3 seconds."""
    assert M.parse_at("1.3") == {"bar": "1.3"}
    assert M.parse_at("bar:2.1.25") == {"bar": "2.1.25"}
    with pytest.raises(M.MixError):
        M.parse_at("halfway")


def test_source_ref_grammar():
    assert M.parse_source_ref("role:stem:drums") == {"role": "stem:drums"}
    assert M.parse_source_ref("blake3:" + "0" * 64) == {"hash": "blake3:" + "0" * 64}
    assert M.parse_source_ref("kick.wav") == {"path": "kick.wav"}
    assert M.parse_source_ref("source") == {"role": "source"}


# ---------------------------------------------------------------------------
# The combinator contract.
# ---------------------------------------------------------------------------
def test_mix_selects_n_inputs_and_emits_one_frame_with_lineage(env, two_tones):
    a, b = two_tones
    src = _stream(env, a, b)
    r = _run(["smpl", "mix", "--stream",
              "--clip", "source=role:source,at=sample:0",
              "--clip", "source=role:pad,at=sample:22050,gain_db=-6,pan=-0.5"], env, stdin=src)
    assert r.returncode == 0, r.stderr
    frames = _frames(r.stdout)
    mixes = [f for f in frames if f.get("op") == "mix"]
    assert len(mixes) == 1  # N in, exactly ONE out
    mix = mixes[0]
    assert mix["kind"] == "audio" and mix["role"] == "mix"
    assert mix["op_version"] == "mix@1"
    # Lineage covers EVERY input, and every target appears earlier in the stream.
    ids = [f["id"] for f in frames]
    assert len(mix["lineage"]) == 2
    for parent in mix["lineage"]:
        assert ids.index(parent) < ids.index(mix["id"])
    assert mix["params"]["clips"][1]["start_sample"] == 22050
    assert mix["params"]["clips"][1]["gain_db"] == -6.0
    assert mix["meta"]["ch"] == 2  # a panned clip forces the stereo bus


def test_bare_mix_sums_every_audio_frame(env, two_tones):
    a, b = two_tones
    r = _run(["smpl", "mix"], env, stdin=_stream(env, a, b))
    assert r.returncode == 0, r.stderr
    mix = _mix_frame(_frames(r.stdout))
    assert len(mix["params"]["clips"]) == 2
    assert mix["meta"]["dur"] == pytest.approx(0.5, abs=1e-6)


def test_dry_run_plans_without_rendering(env, two_tones):
    a, b = two_tones
    src = _stream(env, a, b)
    r = _run(["smpl", "mix", "--dry-run"], env, stdin=src)
    assert r.returncode == 0, r.stderr
    frames = _frames(r.stdout)
    assert not [f for f in frames if f.get("op") == "mix" and f["kind"] == "audio"]
    plan = [f for f in frames if f["kind"] == "control" and f["role"] == "mix.plan"][-1]
    assert plan["params"]["rendered"] is False
    assert plan["params"]["memo_key"].startswith("blake3:")
    assert len(plan["data"]["clips"]) == 2
    assert plan["data"]["length_samples"] == 22050


def test_dry_run_memo_key_matches_the_render_it_would_produce(env, two_tones):
    a, b = two_tones
    src = _stream(env, a, b)
    planned = _frames(_run(["smpl", "mix", "--dry-run"], env, stdin=src).stdout)[-1]
    rendered = _mix_frame(_frames(_run(["smpl", "mix"], env, stdin=src).stdout))
    assert planned["params"]["memo_key"] == rendered["params"]["memo_key"]


def test_render_is_memoized_on_inputs_and_arrangement(env, two_tones):
    a, b = two_tones
    src = _stream(env, a, b)
    first = _mix_frame(_frames(_run(["smpl", "mix"], env, stdin=src).stdout))
    second = _mix_frame(_frames(_run(["smpl", "mix"], env, stdin=src).stdout))
    assert first["params"]["cache_hit"] is False
    assert second["params"]["cache_hit"] is True
    assert first["hash"] == second["hash"]

    # Change the arrangement (not the inputs) → a different key and a real render.
    moved = _mix_frame(_frames(_run(
        ["smpl", "mix", "--clip", "source=role:source,at=sample:0",
         "--clip", "source=role:pad,at=sample:100"], env, stdin=src).stdout))
    assert moved["params"]["memo_key"] != first["params"]["memo_key"]
    assert moved["params"]["cache_hit"] is False


def test_placement_is_sample_accurate_against_marker_sample(env, two_tones):
    """A clip placed at ``marker:onset#i`` lands on that marker's exact ``sample``."""
    a, b = two_tones
    src = _stream(env, a, b)
    sliced = _run(["smpl", "slice", "--role", "source"], env, stdin=src).stdout
    markers = [f for f in _frames(sliced) if f["kind"] == "marker"][-1]
    if len(markers["data"]) < 2:
        pytest.skip("test tone produced too few onsets")
    target = markers["data"][1]["sample"]

    r = _run(["smpl", "mix", "--dry-run",
              "--clip", "source=role:pad,at=marker:onset#1"], env, stdin=sliced)
    plan = _frames(r.stdout)[-1]["data"]
    assert plan["clips"][0]["start_sample"] == target  # exact, not a rounded float second


def test_clip_guard_scales_the_bus_once(env, tmp_path):
    """Summing two hot clips exceeds 0 dBFS → ONE bus gain, relative balance preserved."""
    loud = _tone(tmp_path / "loud.wav", amp=0.9)
    src = _run(["smpl", "read", loud], env).stdout
    r = _run(["smpl", "mix",
              "--clip", "source=role:source,at=sample:0",
              "--clip", "source=role:source,at=sample:0"], env, stdin=src)
    mix = _mix_frame(_frames(r.stdout))
    assert mix["params"]["peak_before"] > 1.0
    assert mix["params"]["guard_gain_db"] < 0.0

    out = tmp_path / "guarded.wav"
    _run(["smpl", "write", str(out), "--role", "mix"], env, stdin=r.stdout)
    data, _ = sf.read(str(out), dtype="float32")
    assert float(np.max(np.abs(data))) == pytest.approx(10 ** (-0.3 / 20), abs=1e-4)


def test_unresolvable_role_is_an_error_frame_and_nonzero_exit(env, two_tones):
    a, b = two_tones
    r = _run(["smpl", "mix", "--clip", "source=role:nope"], env, stdin=_stream(env, a, b))
    assert r.returncode == 1
    err = [f for f in _frames(r.stdout) if f["kind"] == "error"][-1]
    assert err["data"]["code"] == "op_failed" and "nope" in err["data"]["message"]


# ---------------------------------------------------------------------------
# The session control plane (canonical data on disk; stateless verbs).
# ---------------------------------------------------------------------------
def test_session_roundtrips_to_a_deterministic_render(env, tmp_path, two_tones):
    """Verbs → session file → render; re-running the verbs reproduces the same bytes."""
    a, b = two_tones

    def build(path):
        assert _run(["smpl", "mix", "init", "--session", path,
                     "--sr", "44100", "--bpm", "120"], env).returncode == 0
        assert _run(["smpl", "mix", "add-clip", "--session", path,
                     "--source", f"path:{a}", "--at", "sample:0"], env).returncode == 0
        assert _run(["smpl", "mix", "add-clip", "--session", path, "--track", "pad",
                     "--source", f"path:{b}", "--at", "bar:1.3"], env).returncode == 0
        assert _run(["smpl", "mix", "set-gain", "--session", path, "--track", "pad",
                     "--db", "-4.5"], env).returncode == 0

    one, two = str(tmp_path / "one.smplset.json"), str(tmp_path / "two.smplset.json")
    build(one)
    build(two)
    assert json.loads(open(one).read()) == json.loads(open(two).read())

    first = _mix_frame(_frames(_run(["smpl", "mix", "render", "--session", one], env).stdout))
    # Wipe the memo index so the second run genuinely re-renders rather than serving cache.
    shutil.rmtree(env["SMPL_MIX_DIR"], ignore_errors=True)
    second = _mix_frame(_frames(_run(["smpl", "mix", "render", "--arrange", two], env).stdout))
    assert second["params"]["cache_hit"] is False
    assert first["hash"] == second["hash"]  # deterministic: same session ⇒ same output hash
    # bar 1 beat 3 @120bpm = 2 beats = 1.0s = 44100 samples
    assert first["params"]["clips"][1]["start_sample"] == 44100
    assert first["params"]["clips"][1]["gain_db"] == -4.5


def test_verbs_emit_the_session_as_a_control_frame(env, tmp_path, two_tones):
    a, _ = two_tones
    path = str(tmp_path / "s.smplset.json")
    _run(["smpl", "mix", "init", "--session", path, "--sr", "44100"], env)
    r = _run(["smpl", "mix", "add-clip", "--session", path, "--source", f"path:{a}"], env)
    frame = _frames(r.stdout)[-1]
    assert frame["kind"] == "control" and frame["role"] == "mix.session"
    assert frame["params"]["path"] == path
    assert frame["data"]["tracks"][0]["clips"][0]["source"] == {"path": a}

    shown = _frames(_run(["smpl", "mix", "show", "--session", path], env).stdout)[-1]
    assert shown["data"] == frame["data"]


def test_session_on_the_stream_renders_without_a_file(env, tmp_path, two_tones):
    """A `control` mix.session frame round-trips: pipe it back in and it renders."""
    a, b = two_tones
    path = str(tmp_path / "s.smplset.json")
    _run(["smpl", "mix", "init", "--session", path, "--sr", "44100"], env)
    _run(["smpl", "mix", "add-clip", "--session", path, "--source", "role:source"], env)
    session_frame = _run(["smpl", "mix", "show", "--session", path], env).stdout
    stream = _stream(env, a, b) + session_frame
    mix = _mix_frame(_frames(_run(["smpl", "mix"], env, stdin=stream).stdout))
    assert len(mix["params"]["clips"]) == 1


def test_pattern_session_renders_through_mix(env, tmp_path):
    """`smpl pattern | smpl mix` — the pattern DSL's smplset IS the mix session format."""
    tone = _tone(tmp_path / "hit.wav", dur=0.25)
    dsl = json.dumps({"bpm": 120, "grid_steps": 4, "bars": 1,
                      "tracks": [{"name": "kick", "source": tone, "steps": [1, 3],
                                  "velocity": 0.5}]}).encode()
    pat = _run(["smpl", "pattern"], env, stdin=dsl)
    assert pat.returncode == 0, pat.stderr
    r = _run(["smpl", "mix"], env, stdin=pat.stdout)
    assert r.returncode == 0, r.stderr
    mix = _mix_frame(_frames(r.stdout))
    starts = [c["start_sample"] for c in mix["params"]["clips"]]
    assert starts == [0, 44100]  # step 1 and step 3 = 2 beats @ 120bpm
    assert mix["params"]["clips"][0]["gain_db"] == pytest.approx(-6.02, abs=0.01)


def test_rm_clip_and_set_pan(env, tmp_path, two_tones):
    a, b = two_tones
    path = str(tmp_path / "s.smplset.json")
    _run(["smpl", "mix", "init", "--session", path, "--sr", "44100"], env)
    _run(["smpl", "mix", "add-clip", "--session", path, "--source", f"path:{a}"], env)
    _run(["smpl", "mix", "add-clip", "--session", path, "--source", f"path:{b}"], env)
    _run(["smpl", "mix", "set-pan", "--session", path, "--index", "1", "--pan", "0.5"], env)
    _run(["smpl", "mix", "rm-clip", "--session", path, "--index", "0"], env)
    session = json.loads(open(path).read())
    clips = session["tracks"][0]["clips"]
    assert len(clips) == 1 and clips[0]["pan"] == 0.5 and clips[0]["source"] == {"path": b}


def test_legacy_clip_fields_survive_normalization_and_are_flagged():
    """`len` / `transform` aren't rendered in v1 — but they're kept, and reported."""
    session = M.normalize_session({
        "bpm": 120, "swing": 0.1,
        "tracks": [{"name": "t", "mute": False,
                    "clips": [{"source": "kick.wav", "at": "1.3", "len": "1/16",
                               "transform": {"transpose_semitones": 3}}]}],
    })
    clip = session["tracks"][0]["clips"][0]
    assert clip["len"] == "1/16" and clip["transform"] == {"transpose_semitones": 3}
    assert clip["unsupported"] == ["len", "transform"]
    assert clip["at"] == {"bar": "1.3"}
    assert session["tracks"][0]["mute"] is False and session["swing"] == 0.1


def test_mismatched_sample_rate_is_refused_not_silently_resampled(env, tmp_path):
    a = _tone(tmp_path / "a.wav", sr=44100)
    b = _tone(tmp_path / "b.wav", sr=48000)
    src = _run(["smpl", "read", "--role", "pad", b], env,
               stdin=_run(["smpl", "read", a], env).stdout).stdout
    r = _run(["smpl", "mix"], env, stdin=src)
    assert r.returncode == 1
    err = [f for f in _frames(r.stdout) if f["kind"] == "error"][-1]
    assert "resample" in err["data"]["message"]
