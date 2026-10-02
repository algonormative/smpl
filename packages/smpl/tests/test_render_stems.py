"""`smpl render-stems` — grouping/planning units + an end-to-end run on a STUB renderer.

smplmix is an external binary and is not required to test this op: the e2e tests put a tiny
stub `smplmix` on PATH that consumes the session JSON the op hands it (logging every session
so the tests can assert on what was requested) and writes a deterministic per-track sine mix.
That is enough to verify the things that actually matter — grouping, the preserved gain
ladder, the bus glue, output naming/nesting, and determinism — against real audio files.
"""

from __future__ import annotations

import json
import os
import stat
import sys

import numpy as np
import pytest
import soundfile as sf

from smpl_cli import _stems as ST
from smpl_cli import dispatch

# ---------------------------------------------------------------------------
# The stub renderer. `smplmix render <session> -o <out>`: one sine per session track, at the
# frequency encoded in its source path (``tone-220.wav`` → 220 Hz) and the amplitude implied
# by the track's `gain_db`, hard-panned nowhere (identical channels, so a `widen` stage is a
# level no-op and the ladder assertions stay exact).
# ---------------------------------------------------------------------------
STUB = '''
import json, os, sys
import numpy as np
import soundfile as sf

args = sys.argv[1:]
session_path = args[1]
out = args[args.index("-o") + 1]
sess = json.load(open(session_path))

log = os.environ.get("SMPL_STUB_SESSION_LOG")
if log:
    with open(log, "a") as fh:
        fh.write(json.dumps({"out": out, "session": sess}) + "\\n")

sr = 44100
bpm = float(sess["bpm"]); bpb = float(sess.get("beats_per_bar", 4)); bars = int(sess.get("bars", 1))
n = int(round(bars * bpb * 60.0 / bpm * sr))
t = np.arange(n) / sr
mix = np.zeros(n)
for tr in sess["tracks"]:
    path = tr["clips"][0]["source"].get("path", "")
    digits = "".join(c for c in os.path.basename(path) if c.isdigit())
    freq = float(digits) if digits else 440.0
    mix += 0.3 * (10.0 ** (float(tr.get("gain_db", 0.0)) / 20.0)) * np.sin(2 * np.pi * freq * t)
sf.write(out, np.stack([mix, mix], axis=1).astype("float32"), sr, subtype="FLOAT")
'''


@pytest.fixture()
def stub_env(tmp_path, monkeypatch):
    """An isolated CAS + a stub `smplmix` first on PATH. Returns the session log path."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    stub = bindir / "smplmix"
    stub.write_text(f"#!{sys.executable}\n{STUB}")
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    log = tmp_path / "sessions.jsonl"
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    monkeypatch.setenv("SMPL_STUB_SESSION_LOG", str(log))
    monkeypatch.delenv(ST.SMPLMIX_ENV, raising=False)
    return log


class _StdoutShim:
    """`emit` writes NDJSON to ``sys.stdout.buffer`` — hand it a real binary file."""

    def __init__(self, fh):
        self.buffer = fh

    def write(self, text):
        self.buffer.write(text.encode())

    def flush(self):
        self.buffer.flush()

    def isatty(self):
        return False


def run_op(argv, tmp_path, name="frames.ndjson"):
    """Dispatch `smpl render-stems …` in-process; return ``(rc, frames)``.

    stdout is swapped by hand (not via `monkeypatch`) so undoing it can't also undo the
    fixture's PATH/CAS environment.
    """
    out = tmp_path / name
    saved = sys.stdout
    with open(out, "wb") as fh:
        sys.stdout = _StdoutShim(fh)
        try:
            rc = dispatch.main(["smpl", "render-stems", *argv])
        finally:
            sys.stdout = saved
    frames = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    return rc, frames


def write_recipe(tmp_path, recipe, name="demo.pattern.json"):
    path = tmp_path / name
    path.write_text(json.dumps(recipe))
    return str(path)


def rms(path):
    data, _ = sf.read(path, dtype="float64", always_2d=True)
    return float(np.sqrt(np.mean(data ** 2)))


def band_db(path, freq, halfwidth=20.0):
    """Energy (dB) in a narrow band around ``freq`` of the file's mono sum."""
    data, sr = sf.read(path, dtype="float64", always_2d=True)
    mono = data.mean(axis=1)
    spec = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
    fr = np.fft.rfftfreq(len(mono), 1.0 / sr)
    sel = (fr >= freq - halfwidth) & (fr <= freq + halfwidth)
    return 20.0 * np.log10(float(np.sqrt(np.sum(spec[sel] ** 2))) + 1e-12)


LADDER_RECIPE = {
    "name": "demo", "bpm": 130, "bars": 1, "grid_steps": 16,
    "tracks": [
        {"name": "kick", "source": "tone-220.wav", "steps": [1], "gain_db": 0.0},
        {"name": "sub", "source": "tone-220.wav", "steps": [1], "gain_db": -12.0},
        {"name": "pad", "source": "tone-50.wav", "steps": [1], "gain_db": 0.0},
    ],
}


# ---------------------------------------------------------------------------
# Grouping (pure).
# ---------------------------------------------------------------------------
def test_explicit_tag_beats_prefix():
    """An explicit `stem` wins over what the name would imply."""
    assert ST.stem_of({"name": "kick", "stem": "atmos"}) == "atmos"
    assert ST.stem_of({"name": "pad", "stem": "PERC"}) == "perc"


@pytest.mark.parametrize("name,stem", [
    ("kick_909", "perc"), ("clap", "perc"), ("chat", "perc"), ("ride1", "perc"),
    ("tom-hi", "perc"), ("sub", "bass"), ("bassline", "bass"),
    ("stab", "atmos"), ("texture-2", "atmos"), ("drone", "atmos"), ("noisefloor", "atmos"),
])
def test_prefix_table(name, stem):
    assert ST.stem_of({"name": name}) == stem


def test_unknown_name_defaults_to_perc():
    assert ST.stem_of({"name": "zither"}) == "perc"
    assert ST.stem_of({}) == "perc"


def test_unknown_tag_falls_back_to_inference():
    """A junk tag must not create a fourth bus — the name decides instead."""
    assert ST.stem_of({"name": "sub", "stem": "lead"}) == "bass"


def test_group_tracks_covers_every_bus():
    groups = ST.group_tracks(LADDER_RECIPE["tracks"])
    assert list(groups) == list(ST.STEMS)
    assert [t["name"] for t in groups["perc"]] == ["kick"]
    assert [t["name"] for t in groups["bass"]] == ["sub"]
    assert [t["name"] for t in groups["atmos"]] == ["pad"]


def test_parse_stems():
    assert ST.parse_stems("all") == ("perc", "bass", "atmos")
    assert ST.parse_stems("atmos,perc") == ("perc", "atmos")   # canonical order
    with pytest.raises(ValueError):
        ST.parse_stems("perc,lead")


# ---------------------------------------------------------------------------
# Planning (pure).
# ---------------------------------------------------------------------------
def test_plan_emits_full_plus_non_empty_stems():
    jobs = ST.plan(LADDER_RECIPE, outdir="out")["jobs"]
    assert [j["kind"] for j in jobs] == ["full", "perc", "bass", "atmos"]
    assert [os.path.basename(j["out"]) for j in jobs] == [
        "demo.full.wav", "demo.perc.wav", "demo.bass.wav", "demo.atmos.wav"]


def test_plan_skips_empty_groups():
    recipe = {"name": "d", "bpm": 130, "tracks": [{"name": "kick", "steps": [1]}]}
    assert [j["kind"] for j in ST.plan(recipe, outdir="out")["jobs"]] == ["full", "perc"]


def test_plan_honors_requested_stems():
    jobs = ST.plan(LADDER_RECIPE, outdir="out", stems=("bass",))["jobs"]
    assert [j["kind"] for j in jobs] == ["full", "bass"]


def test_plan_nests_under_set():
    recipe = dict(LADDER_RECIPE, _set="phase-lock")
    layout = ST.plan(recipe, outdir="renders")
    assert layout["outdir"] == os.path.join("renders", "phase-lock")
    assert layout["jobs"][0]["out"] == os.path.join("renders", "phase-lock", "demo.full.wav")


def test_plan_names_from_file_when_recipe_is_anonymous():
    recipe = {"bpm": 130, "tracks": [{"name": "kick", "steps": [1]}]}
    layout = ST.plan(recipe, outdir="o", pattern_path="/x/pl-v2-01.pattern.json")
    assert layout["name"] == "pl-v2-01"


def test_every_render_disables_loudnorm():
    """The ladder rule at the plan level: no sub-session may ask smplmix to loudnorm."""
    for job in ST.plan(LADDER_RECIPE, outdir="o")["jobs"]:
        assert job["recipe"]["master"]["loudnorm"] is False
        assert job["recipe"]["master"]["limiter"] is True


def test_stem_chains_carry_no_loudness_op():
    for stem in ST.STEMS:
        ops = [op for op, _ in ST.stem_chain(stem)]
        assert "normalize" not in ops and "limit" not in ops
    assert ("mono", {}) in ST.stem_chain("bass")
    assert ST.stem_chain("perc")[0] == ("gain", {"db": -3.0})
    assert ST.stem_chain("bass", widen=True) == [("mono", {})]      # bass never widened


def test_full_chain_master_toggle():
    assert [op for op, _ in ST.full_chain()] == ["widen", "eq", "normalize"]
    assert ST.full_chain(master=False) == []


def test_sub_recipe_does_not_mutate_the_source():
    recipe = dict(LADDER_RECIPE, master={"loudnorm": True, "limiter": True})
    sub = ST.sub_recipe(recipe, recipe["tracks"][:1])
    assert recipe["master"]["loudnorm"] is True
    assert sub["master"]["loudnorm"] is False
    assert len(sub["tracks"]) == 1


# ---------------------------------------------------------------------------
# End-to-end, against the stub renderer.
# ---------------------------------------------------------------------------
def test_renders_full_and_every_stem(tmp_path, stub_env):
    recipe = write_recipe(tmp_path, LADDER_RECIPE)
    outdir = tmp_path / "renders"
    rc, frames = run_op(["--pattern-file", recipe, "--outdir", str(outdir), "--no-master"],
                        tmp_path)
    assert rc == 0
    assert sorted(p.name for p in outdir.iterdir()) == [
        "demo.atmos.wav", "demo.bass.wav", "demo.full.wav", "demo.perc.wav"]

    kinds = [f["params"]["kind"] for f in frames if f["kind"] == "audio"]
    assert kinds == ["full", "perc", "bass", "atmos"]
    manifest = [f for f in frames if f["kind"] == "feature"][-1]
    assert manifest["data"]["outputs"]["bass"].endswith("demo.bass.wav")
    assert manifest["data"]["groups"] == {"perc": ["kick"], "bass": ["sub"], "atmos": ["pad"]}

    sessions = [json.loads(line) for line in stub_env.read_text().splitlines()]
    assert len(sessions) == 4
    assert all(s["session"]["master"]["loudnorm"] is False for s in sessions)


def test_ladder_is_preserved_across_stems(tmp_path, stub_env):
    """kick (0 dB) and sub (−12 dB) must stay 12 dB apart — minus the documented perc pad.

    A per-stem loudnorm would land both at the same loudness (ratio ≈ 1); this is the
    assertion that fails the moment one sneaks in.
    """
    recipe = write_recipe(tmp_path, LADDER_RECIPE)
    outdir = tmp_path / "renders"
    rc, _ = run_op(["--pattern-file", recipe, "--outdir", str(outdir), "--no-master"],
                   tmp_path)
    assert rc == 0
    ratio = rms(str(outdir / "demo.bass.wav")) / rms(str(outdir / "demo.perc.wav"))
    assert ratio == pytest.approx(10 ** (-9 / 20.0), rel=0.02)   # −12 dB track, −3 dB perc pad


def test_perc_pad_is_the_only_level_change(tmp_path, stub_env):
    """perc is padded exactly −3 dB; bass's mono downmix is level-neutral."""
    recipe = write_recipe(tmp_path, {
        "name": "lvl", "bpm": 130, "bars": 1,
        "tracks": [{"name": "kick", "source": "tone-220.wav", "steps": [1], "gain_db": 0.0},
                   {"name": "sub", "source": "tone-220.wav", "steps": [1], "gain_db": 0.0}]})
    outdir = tmp_path / "renders"
    rc, _ = run_op(["--pattern-file", recipe, "--outdir", str(outdir), "--no-master"],
                   tmp_path)
    assert rc == 0
    bass = rms(str(outdir / "lvl.bass.wav"))
    perc = rms(str(outdir / "lvl.perc.wav"))
    assert bass == pytest.approx(0.3 / np.sqrt(2), rel=0.02)      # untouched by the downmix
    assert perc / bass == pytest.approx(10 ** (-3 / 20.0), rel=0.02)


def test_bass_stem_is_mono(tmp_path, stub_env):
    recipe = write_recipe(tmp_path, LADDER_RECIPE)
    outdir = tmp_path / "renders"
    run_op(["--pattern-file", recipe, "--outdir", str(outdir), "--no-master"],
           tmp_path)
    assert sf.info(str(outdir / "demo.bass.wav")).channels == 1
    assert sf.info(str(outdir / "demo.perc.wav")).channels == 2


def test_atmos_stem_is_high_passed(tmp_path, stub_env):
    """A 50 Hz drone and a 1 kHz texture go in at the same level; only the sub is removed."""
    recipe = write_recipe(tmp_path, {
        "name": "air", "bpm": 130, "bars": 1,
        "tracks": [{"name": "drone", "source": "tone-50.wav", "steps": [1], "gain_db": 0.0},
                   {"name": "texture", "source": "tone-1000.wav", "steps": [1], "gain_db": 0.0}]})
    outdir = tmp_path / "renders"
    rc, _ = run_op(["--pattern-file", recipe, "--outdir", str(outdir), "--no-master"],
                   tmp_path)
    assert rc == 0
    out = str(outdir / "air.atmos.wav")
    assert band_db(out, 1000.0) - band_db(out, 50.0) > 30.0      # sub gone, top intact
    assert band_db(out, 1000.0) > band_db(out, 50.0)


def test_set_nesting_and_empty_groups(tmp_path, stub_env):
    recipe = write_recipe(tmp_path, {
        "name": "pl-01", "_set": "phase-lock", "bpm": 130, "bars": 1,
        "tracks": [{"name": "kick", "source": "tone-220.wav", "steps": [1]},
                   {"name": "hat", "source": "tone-4000.wav", "steps": [3]}]})
    outdir = tmp_path / "renders"
    rc, frames = run_op(["--pattern-file", recipe, "--outdir", str(outdir), "--no-master"],
                        tmp_path)
    assert rc == 0
    nested = outdir / "phase-lock"
    assert sorted(p.name for p in nested.iterdir()) == ["pl-01.full.wav", "pl-01.perc.wav"]
    manifest = [f for f in frames if f["kind"] == "feature"][-1]
    assert set(manifest["data"]["outputs"]) == {"full", "perc"}   # no empty bass/atmos files


def test_master_chain_applies_only_to_the_full_mix(tmp_path, stub_env):
    recipe = write_recipe(tmp_path, LADDER_RECIPE)
    mastered = tmp_path / "m"
    raw = tmp_path / "r"
    assert run_op(["--pattern-file", recipe, "--outdir", str(mastered)],
                  tmp_path, name="a.ndjson")[0] == 0
    assert run_op(["--pattern-file", recipe, "--outdir", str(raw), "--no-master"],
                  tmp_path, name="b.ndjson")[0] == 0
    assert (mastered / "demo.full.wav").read_bytes() != (raw / "demo.full.wav").read_bytes()
    # …and the stems are byte-identical either way (mastering never reaches them).
    for stem in ("perc", "bass", "atmos"):
        assert (mastered / f"demo.{stem}.wav").read_bytes() == (raw / f"demo.{stem}.wav").read_bytes()


def test_deterministic(tmp_path, stub_env):
    recipe = write_recipe(tmp_path, LADDER_RECIPE)
    a, b = tmp_path / "a", tmp_path / "b"
    run_op(["--pattern-file", recipe, "--outdir", str(a)], tmp_path, name="a.ndjson")
    run_op(["--pattern-file", recipe, "--outdir", str(b)], tmp_path, name="b.ndjson")
    for kind in ("full", "perc", "bass", "atmos"):
        assert (a / f"demo.{kind}.wav").read_bytes() == (b / f"demo.{kind}.wav").read_bytes()


def test_missing_smplmix_is_an_actionable_error(tmp_path, monkeypatch, capsys, stub_env):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    recipe = write_recipe(tmp_path, LADDER_RECIPE)
    outdir = tmp_path / "renders"
    rc, frames = run_op(["--pattern-file", recipe, "--outdir", str(outdir)], tmp_path)
    assert rc == 1 and frames == []
    err = capsys.readouterr().err
    assert "smplmix" in err and "--smplmix" in err
    assert not outdir.exists()


def test_bad_smplmix_override_is_rejected(tmp_path, capsys, stub_env):
    recipe = write_recipe(tmp_path, LADDER_RECIPE)
    rc, _ = run_op(["--pattern-file", recipe, "--outdir", str(tmp_path / "o"),
                    "--smplmix", str(tmp_path / "nope")], tmp_path)
    assert rc == 1
    assert "not an executable file" in capsys.readouterr().err


def test_renderer_failure_is_reported(tmp_path, capsys, stub_env):
    """A non-zero smplmix exit aborts the run instead of writing a silent half-render."""
    broken = tmp_path / "bin" / "smplmix"
    broken.write_text(f"#!{sys.executable}\nimport sys\nsys.stderr.write('boom\\n')\nsys.exit(2)\n")
    recipe = write_recipe(tmp_path, LADDER_RECIPE)
    rc, _ = run_op(["--pattern-file", recipe, "--outdir", str(tmp_path / "o")],
                   tmp_path)
    assert rc == 1
    assert "smplmix render failed" in capsys.readouterr().err
