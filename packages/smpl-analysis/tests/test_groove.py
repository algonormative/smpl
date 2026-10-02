"""Tests for groove extraction (smpl_analysis.groove; vault-6m62).

**How the round-trip is verified.** No render binary is guaranteed here, so "round-trips
into a generated loop" is proved in its deterministic form instead: every fixture is
rendered from a real ``smpl pattern`` DSL, placed by ``pattern._expand`` — the *actual*
consumer's timing math, not a reimplementation of it — and the clicks are stamped at the
beat positions it computes. The extractor then has to recover the ``swing`` and per-hit
``nudge`` values that went in. :func:`test_groove_round_trips_through_a_generated_pattern`
closes the loop the other way: the extracted groove is fed back into a *new* pattern as
its global ``swing`` plus per-step ``nudge``, and the loop `pattern` generates from it is
asserted to land on the reference's own hit times.
"""

from __future__ import annotations

import numpy as np
import pytest

from smpl_analysis import groove

SR = 44100
BPM = 120.0
GRID = 16
BPB = 4.0

# The exact keys the registry (feature-keys.md) assigns to this op.
EXPECTED_KEYS = {
    "rhythm.swing",
    "rhythm.swing_confidence",
    "rhythm.microtiming_beats",
}


# ---------------------------------------------------------------------------
# Fixtures: render a real `smpl pattern` DSL to clicks.
# ---------------------------------------------------------------------------
def _click(sr: int = SR, seconds: float = 0.03) -> np.ndarray:
    n = int(sr * seconds)
    t = np.linspace(0.0, seconds, n, endpoint=False)
    return (np.exp(-np.linspace(0.0, 12.0, n)) * np.sin(2 * np.pi * 1200.0 * t)).astype("float32")


def _at_to_beats(at: str, bpb: float = BPB) -> float:
    """Invert `pattern`'s ``bar.beat.frac`` back to total beats from the start."""
    parts = at.split(".")
    bar, beat = int(parts[0]), int(parts[1])
    frac = float("0." + parts[2]) if len(parts) > 2 else 0.0
    return (bar - 1) * bpb + (beat - 1) + frac


def pattern_hit_times(dsl: dict, *, bpm: float = BPM, bpb: float = BPB) -> list[float]:
    """The seconds at which ``smpl pattern`` would place every hit in ``dsl``."""
    from smpl_cli.subcommands import pattern

    session = pattern._expand(dsl)
    times = [
        _at_to_beats(clip["at"], bpb) * 60.0 / bpm
        for track in session["tracks"]
        for clip in track["clips"]
    ]
    return sorted(times)


def render(dsl: dict, *, bpm: float = BPM, bpb: float = BPB, sr: int = SR) -> np.ndarray:
    """Click track at exactly the positions ``smpl pattern`` would generate."""
    times = pattern_hit_times(dsl, bpm=bpm, bpb=bpb)
    click = _click(sr)
    y = np.zeros(int(round((max(times) + 1.0) * sr)) + len(click), dtype="float32")
    for t in times:
        start = int(round(t * sr))
        y[start : start + len(click)] += click * 0.5
    return y


def dsl(*, swing: float = 0.0, steps=None, hits=None, bars: int = 2, bpm: float = BPM) -> dict:
    track: dict = {"name": "click", "source": "click.wav"}
    if hits is not None:
        track["hits"] = hits
    else:
        track["steps"] = list(steps if steps is not None else range(1, GRID + 1))
    return {"bpm": bpm, "beats_per_bar": BPB, "grid_steps": GRID,
            "bars": bars, "swing": swing, "tracks": [track]}


def extract(dsl_dict: dict, **kw) -> dict:
    return groove.extract_groove(render(dsl_dict), SR, bpm=BPM,
                                 beats_per_bar=BPB, grid_steps=GRID, **kw)


# ---------------------------------------------------------------------------
# Onset detection: the measurement floor.
# ---------------------------------------------------------------------------
def test_onsets_include_a_hit_sitting_exactly_at_zero():
    """The downbeat of a trimmed loop is at t=0; without the lead-in pad it is missed."""
    times = groove.onset_times(render(dsl(steps=[1, 5, 9, 13])), SR)
    assert len(times) == 8  # 4 hits x 2 bars
    assert times[0] < 0.01


def test_onset_times_track_the_true_hit_positions():
    loop = dsl(swing=0.2, steps=[1, 4, 7, 11, 14])
    got = np.array(groove.onset_times(render(loop), SR))
    for expected in pattern_hit_times(loop):
        assert np.min(np.abs(got - expected)) < 0.006, expected


# ---------------------------------------------------------------------------
# Swing recovery.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("injected", [0.0, 0.08, 0.12, 0.25, 0.3, 0.5, 0.6])
def test_recovers_the_injected_swing(injected):
    """A global search, not a refinement: swing past 0.5 puts an offbeat nearer the
    FOLLOWING step, which pins an iterate-from-zero estimator at 0 forever."""
    got = extract(dsl(swing=injected))["groove"]["swing"]
    assert abs(got - injected) < 0.02, got


def test_recovers_swing_from_a_sparse_pattern():
    loop = dsl(swing=0.3, steps=[1, 3, 5, 7, 9, 11, 13, 15, 2, 10])
    assert abs(extract(loop)["groove"]["swing"] - 0.3) < 0.02


def test_a_straight_grid_reports_no_swing_and_near_zero_microtiming():
    result = extract(dsl(swing=0.0))
    assert result["groove"]["swing"] == 0.0
    assert result["features"]["rhythm.microtiming_beats"] < 0.01


def test_no_offbeat_hits_means_no_swing_evidence():
    """Odd steps only: the swing model touches nothing, so 0 with zero confidence."""
    result = extract(dsl(swing=0.4, steps=[1, 3, 5, 9, 13]))
    assert result["groove"]["swing"] == 0.0
    assert result["features"]["rhythm.swing_confidence"] == 0.0


# ---------------------------------------------------------------------------
# Per-step timing deviation + nudge residuals.
# ---------------------------------------------------------------------------
def _nudge_by_step(result: dict) -> dict[int, float]:
    return {h["step"]: h["nudge"] for h in result["groove"]["hits"]}


def test_recovers_per_hit_nudges_after_the_swing_model_is_removed():
    injected = {5: 0.03, 12: -0.025}
    hits = [{"step": s, "nudge": injected.get(s, 0.0)} for s in range(1, GRID + 1)]
    result = extract(dsl(swing=0.2, hits=hits))
    assert abs(result["groove"]["swing"] - 0.2) < 0.02
    nudges = _nudge_by_step(result)
    for step in range(1, GRID + 1):
        assert abs(nudges[step] - injected.get(step, 0.0)) < 0.01, step


def test_per_step_table_covers_every_played_step_and_counts_the_bars():
    steps = [1, 5, 9, 13, 4, 12]
    result = extract(dsl(swing=0.15, steps=steps, bars=3))
    hits = result["groove"]["hits"]
    assert [h["step"] for h in hits] == sorted(steps)  # ascending, one row per step
    assert all(h["count"] == 3 for h in hits)  # aggregated across all three bars


def test_deviation_is_reported_in_beats_and_milliseconds():
    result = extract(dsl(swing=0.25, steps=[1, 2]))
    step_gap = BPB / GRID
    by_step = {h["step"]: h for h in result["groove"]["hits"]}
    # Step 2 is swung: its total deviation is swing x stepGap beats, the ms mirror is
    # the same number at this tempo.
    assert abs(by_step[2]["deviation"] - 0.25 * step_gap) < 0.01
    assert abs(by_step[2]["deviation_ms"] - by_step[2]["deviation"] * 60000.0 / BPM) < 0.01
    assert abs(by_step[1]["deviation"]) < 0.01  # odd steps are the un-swung anchors


def test_grid_origin_alignment_survives_a_shifted_loop():
    """A loop that does not start exactly on its grid (or an onset detector with a
    constant bias — the same unknown) still reads as the same groove."""
    shifted = np.concatenate([np.zeros(int(0.004 * SR), dtype="float32"),
                              render(dsl(swing=0.2))])
    result = groove.extract_groove(shifted, SR, bpm=BPM, beats_per_bar=BPB, grid_steps=GRID)
    assert abs(result["groove"]["swing"] - 0.2) < 0.02
    assert result["features"]["rhythm.microtiming_beats"] < 0.01


# ---------------------------------------------------------------------------
# The round trip: extracted groove -> a generated loop.
# ---------------------------------------------------------------------------
def test_groove_round_trips_through_a_generated_pattern():
    """Reference in, groove out, groove back into `smpl pattern` — the loop it generates
    must land on the reference's own hit times.

    This is the acceptance's "round-trips into a generated loop" in deterministic form:
    both the reference and the borrowed-groove loop are placed by ``pattern._expand``, so
    what is compared is the timing the real consumer produces, not a model of it.
    """
    injected = {3: 0.02, 8: -0.03, 14: 0.015}
    steps = [1, 3, 5, 8, 9, 11, 14]
    reference = dsl(
        swing=0.24,
        hits=[{"step": s, "nudge": injected.get(s, 0.0)} for s in steps],
    )
    result = extract(reference)

    # Feed the measured groove straight back in: its `swing` IS pattern's global swing,
    # and each hit row's `nudge` IS pattern's per-hit nudge.
    borrowed = dsl(
        swing=result["groove"]["swing"],
        hits=[{"step": h["step"], "nudge": h["nudge"]} for h in result["groove"]["hits"]],
    )
    assert borrowed["swing"] == result["groove"]["swing"]
    assert [h["step"] for h in borrowed["tracks"][0]["hits"]] == steps

    reference_times = pattern_hit_times(reference)
    borrowed_times = pattern_hit_times(borrowed)
    assert len(borrowed_times) == len(reference_times)
    for got, expected in zip(borrowed_times, reference_times):
        assert abs(got - expected) < 0.006, (got, expected)  # ~2 onset hops


# ---------------------------------------------------------------------------
# Registered features.
# ---------------------------------------------------------------------------
def test_emits_exactly_the_registered_keys():
    assert set(extract(dsl(swing=0.2))["features"].keys()) == EXPECTED_KEYS


def test_confidence_is_a_unit_interval_and_higher_for_a_consistent_groove():
    consistent = extract(dsl(swing=0.25))["features"]["rhythm.swing_confidence"]
    rng = np.random.default_rng(0)
    scattered = extract(dsl(
        swing=0.25,
        hits=[{"step": s, "nudge": float(rng.uniform(-0.06, 0.06))}
              for s in range(1, GRID + 1)],
    ))["features"]["rhythm.swing_confidence"]
    for value in (consistent, scattered):
        assert 0.0 <= value <= 1.0
    assert consistent > scattered


def test_microtiming_reports_the_residual_the_swing_model_cannot_express():
    straight = extract(dsl(swing=0.2))["features"]["rhythm.microtiming_beats"]
    nudged = extract(dsl(
        swing=0.2,
        hits=[{"step": s, "nudge": 0.03 if s % 4 == 3 else 0.0} for s in range(1, GRID + 1)],
    ))["features"]["rhythm.microtiming_beats"]
    assert nudged > straight
    assert nudged > 0.005


def test_silence_yields_an_empty_groove_not_a_crash():
    result = groove.extract_groove(np.zeros(SR * 2, dtype="float32"), SR, bpm=BPM)
    assert result["groove"]["hits"] == []
    assert result["onsets"] == []
    assert set(result["features"].keys()) == EXPECTED_KEYS


def test_bpm_is_estimated_when_not_given_and_flagged_as_such():
    result = groove.extract_groove(render(dsl(steps=[1, 5, 9, 13], bars=4)), SR,
                                   beats_per_bar=BPB, grid_steps=GRID)
    assert result["bpm_source"] == "estimated"
    assert result["groove"]["bpm"] > 0
    assert extract(dsl())["bpm_source"] == "given"


def test_extraction_is_deterministic():
    y = render(dsl(swing=0.2))
    first = groove.extract_groove(y, SR, bpm=BPM)
    second = groove.extract_groove(y, SR, bpm=BPM)
    assert first == second


def test_rejects_a_degenerate_grid():
    with pytest.raises(ValueError):
        groove.extract_groove(render(dsl()), SR, bpm=BPM, grid_steps=0)


# ---------------------------------------------------------------------------
# Frames: shape, lineage, consumability.
# ---------------------------------------------------------------------------
def _frames(tmp_path, y=None, **kw):
    import soundfile as sf

    from smplstream import cas, frames as F

    y = render(dsl(swing=0.2)) if y is None else y
    wav = tmp_path / "groove.wav"
    sf.write(str(wav), y, SR, subtype="FLOAT")
    blob = cas.put_audio_bytes(wav.read_bytes())
    audio = F.audio_frame(blob, sr=SR, ch=1, dur=len(y) / SR, role="source")
    kw.setdefault("bpm", BPM)
    return audio, groove.groove_audio_frame(audio, **kw), len(y)


def test_audio_frame_emits_one_marker_and_two_features(tmp_path):
    _, out, _ = _frames(tmp_path)
    assert [f["kind"] for f in out] == ["marker", "feature", "feature"]
    assert [f["role"] for f in out] == ["groove", "groove", "groove-features"]
    assert set(out[2]["data"].keys()) == EXPECTED_KEYS
    assert set(out[1]["data"].keys()) == {"bpm", "beats_per_bar", "grid_steps", "swing", "hits"}


def test_marker_points_carry_sample_accuracy_and_their_grid_step(tmp_path):
    _, out, n_samples = _frames(tmp_path)
    previous = -1
    for point in out[0]["data"]:
        assert set(point.keys()) == {"t", "sample", "label"}
        assert isinstance(point["sample"], int)
        assert abs(point["sample"] - round(point["t"] * SR)) <= 1
        assert 0 <= point["sample"] <= n_samples
        assert point["sample"] > previous
        previous = point["sample"]
        step = int(point["label"].removeprefix("step-"))
        assert 1 <= step <= GRID


def test_lineage_and_params_on_every_emitted_frame(tmp_path):
    audio, out, _ = _frames(tmp_path)
    for frame in out:
        assert frame["of"] == audio["id"]
        assert frame["lineage"] == [audio["id"]]
        assert frame["op"] == "groove"
        assert frame["op_version"] == groove.OP_VERSION
        # The tempo the groove was measured against rides on params, NOT as rhythm.bpm:
        # that key stays owned by the beat grid (feature-keys.md ownership notes).
        assert frame["params"]["bpm"] == BPM
        assert frame["params"]["grid_steps"] == GRID
        assert frame["params"]["sr_hz"] == SR
    assert "rhythm.bpm" not in out[2]["data"]


def test_emitted_frames_validate(tmp_path):
    from smplstream.frames import validate_frame

    _, out, _ = _frames(tmp_path)
    for frame in out:
        assert validate_frame(frame) == []
