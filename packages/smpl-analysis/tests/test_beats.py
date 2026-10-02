"""Tests for the downbeat-aware beat grid (smpl_analysis.beats; vault-32n3)."""

from __future__ import annotations

import numpy as np

from smpl_analysis import beats

SR = 44100
BPM = 120.0

# The exact keys the registry (feature-keys.md) assigns to this op.
EXPECTED_KEYS = {
    "rhythm.bpm",
    "rhythm.bpm_confidence",
    "rhythm.bpm_candidates",
    "rhythm.time_signature",
}


def _click(sr: int = SR, seconds: float = 0.03) -> np.ndarray:
    n = int(sr * seconds)
    t = np.linspace(0.0, seconds, n, endpoint=False)
    return (np.exp(-np.linspace(0.0, 12.0, n)) * np.sin(2 * np.pi * 1200.0 * t)).astype("float32")


def click_track(
    *, bpm: float = BPM, meter: int = 4, bars: int = 10, sr: int = SR, accent: float = 3.0
) -> np.ndarray:
    """A metronome: one click per beat, the bar's first click ``accent``x louder."""
    spb = 60.0 / bpm
    click = _click(sr)
    y = np.zeros(int(sr * spb * meter * bars) + sr // 4, dtype="float32")
    for i in range(meter * bars):
        start = int(round(i * spb * sr))
        y[start : start + len(click)] += click * (accent if i % meter == 0 else 1.0) * 0.25
    return y


def _grid(**kw) -> dict:
    return beats.beat_grid(click_track(**kw), kw.get("sr", SR))


# ---------------------------------------------------------------------------
# Beats and downbeats.
# ---------------------------------------------------------------------------
def test_beats_land_on_the_true_grid():
    grid = _grid()
    got = np.array(grid["beats"], dtype="float64") / SR
    truth = np.arange(40) * (60.0 / BPM)
    assert abs(len(got) - len(truth)) <= 2  # edge beats may be trimmed/added
    # Every true beat has a detected beat within 40 ms (one analysis hop is ~12 ms).
    for expected in truth:
        assert np.min(np.abs(got - expected)) < 0.04, expected


def test_beat_samples_are_ordered_and_in_range():
    y = click_track()
    grid = beats.beat_grid(y, SR)
    samples = grid["beats"]
    assert samples == sorted(samples)
    assert all(isinstance(s, int) for s in samples)
    assert all(0 <= s <= len(y) for s in samples)


def test_downbeats_are_a_subset_of_beats_on_the_accents():
    grid = _grid()
    assert set(grid["downbeats"]).issubset(set(grid["beats"]))
    assert len(grid["downbeats"]) == len(grid["beats"][::4])
    # Bars are 2 s at 120 BPM in 4/4; every downbeat sits on one (± one hop).
    for sample in grid["downbeats"]:
        t = sample / SR
        assert abs(t - round(t / 2.0) * 2.0) < 0.04, t


def test_bar_phase_prefers_the_accented_phase():
    accent = np.array([3.0, 1.0, 1.0, 1.0] * 6)
    assert beats.bar_phase(accent) == (4, 0)
    assert beats.bar_phase(np.roll(accent, 1)) == (4, 1)


def test_bar_phase_is_deterministic_on_a_flat_accent():
    # No accent structure at all: fall back to the declared preference order, not chance.
    flat = np.ones(24)
    assert beats.bar_phase(flat) == beats.bar_phase(flat) == (4, 0)


# ---------------------------------------------------------------------------
# rhythm.* features.
# ---------------------------------------------------------------------------
def test_emits_exactly_the_registered_keys():
    assert set(_grid()["features"].keys()) == EXPECTED_KEYS


def test_bpm_matches_the_synthesized_tempo():
    features = _grid()["features"]
    assert abs(features["rhythm.bpm"] - BPM) < 3.0


def test_two_octave_robust_candidates():
    candidates = _grid()["features"]["rhythm.bpm_candidates"]
    assert len(candidates) == 2
    assert all(isinstance(c, float) and c > 0 for c in candidates)
    # The primary hypothesis is the true tempo, and the runner-up is a DIFFERENT
    # hypothesis — not the same tempo an octave away (that is the folding's whole job).
    assert abs(candidates[0] - BPM) < 5.0
    for ratio in (0.5, 1.0, 2.0):
        assert abs(candidates[1] - candidates[0] * ratio) > 0.05 * candidates[0]


def test_confidence_is_a_unit_interval_and_higher_for_a_metronome():
    steady = _grid()["features"]["rhythm.bpm_confidence"]
    rng = np.random.default_rng(0)
    noise = beats.beat_grid((rng.standard_normal(SR * 12) * 0.2).astype("float32"), SR)
    noisy = noise["features"]["rhythm.bpm_confidence"]
    for value in (steady, noisy):
        assert 0.0 <= value <= 1.0
    assert steady > noisy


def test_time_signature_four_four():
    assert _grid(meter=4)["features"]["rhythm.time_signature"] == "4/4"


def test_time_signature_three_four():
    assert _grid(meter=3)["features"]["rhythm.time_signature"] == "3/4"


def test_fold_octave_collapses_octave_relatives():
    folded = {beats._fold_octave(b) for b in (60.0, 120.0, 240.0, 480.0)}
    assert len(folded) == 1
    assert beats.FOLD_MIN <= folded.pop() < beats.FOLD_MIN * 2


# ---------------------------------------------------------------------------
# Tempo changes.
# ---------------------------------------------------------------------------
def test_steady_tempo_emits_no_tempo_change_points():
    assert _grid()["tempo_changes"] == []


def test_tempo_change_is_detected_and_snapped_to_a_beat():
    y = np.concatenate([click_track(bpm=100.0, bars=6), click_track(bpm=150.0, bars=8)])
    grid = beats.beat_grid(y, SR)
    changes = grid["tempo_changes"]
    assert len(changes) >= 1
    assert set(changes).issubset(set(grid["beats"]))  # snapped onto the grid
    boundary = len(click_track(bpm=100.0, bars=6)) / SR
    assert min(abs(c / SR - boundary) for c in changes) < 2.0


# ---------------------------------------------------------------------------
# Frames: sample-accuracy, lineage, determinism.
# ---------------------------------------------------------------------------
def _frames(tmp_path, y=None, sr: int = SR):
    import soundfile as sf

    from smplstream import cas, frames as F

    y = click_track() if y is None else y
    wav = tmp_path / "grid.wav"
    sf.write(str(wav), y, sr, subtype="FLOAT")
    blob = cas.put_audio_bytes(wav.read_bytes())
    audio = F.audio_frame(blob, sr=sr, ch=1, dur=len(y) / sr, role="source")
    return audio, beats.beats_audio_frame(audio), len(y)


def test_audio_frame_emits_three_markers_and_one_feature(tmp_path):
    _, out, _ = _frames(tmp_path)
    assert [f["kind"] for f in out] == ["marker", "marker", "marker", "feature"]
    assert [f["role"] for f in out] == ["beat", "downbeat", "tempo-change", "beats"]
    assert set(out[-1]["data"].keys()) == EXPECTED_KEYS


def test_every_marker_point_carries_a_consistent_integer_sample(tmp_path):
    _, out, n_samples = _frames(tmp_path)
    for frame in out[:3]:
        previous = -1
        for point in frame["data"]:
            assert set(point.keys()) == {"t", "sample", "label"}
            assert isinstance(point["sample"], int)
            assert abs(point["sample"] - round(point["t"] * SR)) <= 1
            assert 0 <= point["sample"] <= n_samples
            assert point["sample"] > previous
            previous = point["sample"]
        assert frame["data"] or frame["role"] == "tempo-change"


def test_marker_labels_distinguish_the_roles(tmp_path):
    _, out, _ = _frames(tmp_path)
    assert all(p["label"].startswith("beat-") for p in out[0]["data"])
    assert all(p["label"].startswith("downbeat-") for p in out[1]["data"])


def test_lineage_on_every_emitted_frame(tmp_path):
    audio, out, _ = _frames(tmp_path)
    for frame in out:
        assert frame["of"] == audio["id"]
        assert frame["lineage"] == [audio["id"]]
        assert frame["op"] == "beats"
        assert frame["op_version"] == "beats@1"
        assert frame["params"] == {"hop_length": 512, "start_bpm": 120.0, "sr_hz": SR}


def test_deterministic_across_runs():
    y = click_track(bars=6)
    first, second = beats.beat_grid(y, SR), beats.beat_grid(y, SR)
    assert first == second


def test_op_version_constant():
    assert beats.OP == "beats"
    assert beats.OP_VERSION == "beats@1"
