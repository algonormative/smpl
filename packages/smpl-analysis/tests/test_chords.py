"""Tests for the chord timeline + key/tuning op (smpl_analysis.chords; vault-379o)."""

from __future__ import annotations

import numpy as np
import pytest

from smpl_analysis import chords

SR = 22050

# The exact keys the registry (feature-keys.md) assigns to this op.
TONAL_KEYS = {"tonal.key_key", "tonal.key_scale", "tonal.tuning_frequency"}

MAJOR = (0, 4, 7)
MINOR = (0, 3, 7)


def _triad(root_midi: int, intervals=MAJOR, *, sr: int = SR, dur: float = 2.0,
           tuning_hz: float = 440.0) -> np.ndarray:
    """A sustained triad with a few harmonics, tuned to ``tuning_hz`` for A4."""
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    y = np.zeros_like(t)
    for iv in intervals:
        f0 = tuning_hz * 2.0 ** ((root_midi + iv - 69) / 12.0)
        for harmonic, amp in ((1, 1.0), (2, 0.4), (3, 0.2)):
            y += amp * np.sin(2 * np.pi * f0 * harmonic * t)
    return (0.2 * y).astype("float32")


def _progression(specs, *, dur: float = 2.0, tuning_hz: float = 440.0) -> np.ndarray:
    return np.concatenate([
        _triad(root, iv, dur=dur, tuning_hz=tuning_hz) for root, iv in specs
    ])


# C:maj → F:maj → G:maj → C:maj, two seconds each.
C_F_G_C = [(60, MAJOR), (65, MAJOR), (67, MAJOR), (60, MAJOR)]
C_F_G_C_LABELS = ["C:maj", "F:maj", "G:maj", "C:maj"]


def test_detects_the_major_progression():
    result = chords.chord_timeline(_progression(C_F_G_C), SR)
    assert [p["label"] for p in result["markers"]] == C_F_G_C_LABELS


def test_span_boundaries_match_the_true_progression():
    markers = chords.chord_timeline(_progression(C_F_G_C), SR)["markers"]
    # Chroma smoothing shifts a change point by a few frames; allow 100 ms.
    for i, expected_start in enumerate([0.0, 2.0, 4.0, 6.0]):
        assert markers[i]["t"] == pytest.approx(expected_start, abs=0.1)
        assert markers[i]["dur"] == pytest.approx(2.0, abs=0.15)


def test_marker_points_use_the_spec_shape():
    markers = chords.chord_timeline(_progression(C_F_G_C), SR)["markers"]
    assert markers
    for p in markers:
        # SPEC marker shape — NOT t_start/t_end/chord_label.
        assert set(p.keys()) == {"t", "dur", "label", "sample"}
        assert isinstance(p["t"], float) and isinstance(p["dur"], float)
        assert isinstance(p["label"], str)
        assert isinstance(p["sample"], int) and not isinstance(p["sample"], bool)


def test_sample_is_the_start_sample_at_native_sr():
    markers = chords.chord_timeline(_progression(C_F_G_C), SR)["markers"]
    n_samples = int(SR * 8.0)
    for p in markers:
        assert abs(p["sample"] - round(p["t"] * SR)) <= 1
        assert 0 <= p["sample"] <= n_samples


def test_spans_are_ascending_contiguous_and_positive():
    markers = chords.chord_timeline(_progression(C_F_G_C), SR)["markers"]
    dur = 8.0
    for a, b in zip(markers, markers[1:]):
        assert a["t"] < b["t"]
        assert a["sample"] < b["sample"]
        assert a["t"] + a["dur"] == pytest.approx(b["t"], abs=1e-6)  # no gaps/overlaps
    for p in markers:
        assert p["dur"] > 0.0
    last = markers[-1]
    # `t + dur` must not run past the audio (one hop of slack).
    assert last["t"] + last["dur"] <= dur + chords.HOP_LENGTH / SR


def test_minor_triads_are_labelled_min():
    result = chords.chord_timeline(_progression([(69, MINOR), (62, MINOR)]), SR)
    assert [p["label"] for p in result["markers"]] == ["A:min", "D:min"]


def test_key_detection_major():
    tonal = chords.chord_timeline(_progression(C_F_G_C), SR)["tonal"]
    assert set(tonal.keys()) == TONAL_KEYS
    assert tonal["tonal.key_key"] == "C"
    assert tonal["tonal.key_scale"] == "major"


def test_key_detection_minor():
    # A natural minor: Am → Dm → Em → Am.
    prog = [(69, MINOR), (62, MINOR), (64, MINOR), (69, MINOR)]
    tonal = chords.chord_timeline(_progression(prog), SR)["tonal"]
    assert tonal["tonal.key_key"] == "A"
    assert tonal["tonal.key_scale"] == "minor"


def test_tuning_frequency_near_440_for_a440_audio():
    tonal = chords.chord_timeline(_progression(C_F_G_C), SR)["tonal"]
    assert tonal["tonal.tuning_frequency"] == pytest.approx(440.0, abs=1.5)


def test_tuning_frequency_tracks_a_detuned_reference():
    # Synthesize the same progression against A=452 Hz (~+46 cents).
    tonal = chords.chord_timeline(_progression(C_F_G_C, tuning_hz=452.0), SR)["tonal"]
    assert tonal["tonal.tuning_frequency"] > 445.0
    assert tonal["tonal.tuning_frequency"] == pytest.approx(452.0, abs=4.0)


def test_deterministic_across_runs():
    y = _progression(C_F_G_C)
    first = chords.chord_timeline(y, SR)
    second = chords.chord_timeline(y, SR)
    assert first == second


def test_silence_yields_a_no_chord_span_and_no_key():
    result = chords.chord_timeline(np.zeros(SR, dtype="float32"), SR)
    assert [p["label"] for p in result["markers"]] == [chords.NO_CHORD]
    assert result["tonal"]["tonal.key_key"] is None
    assert result["tonal"]["tonal.key_scale"] is None
    assert isinstance(result["tonal"]["tonal.tuning_frequency"], float)


def test_empty_signal_yields_an_empty_timeline():
    result = chords.chord_timeline(np.zeros(0, dtype="float32"), SR)
    assert result["markers"] == []
    assert set(result["tonal"].keys()) == TONAL_KEYS


def test_stereo_input_is_collapsed_to_mono():
    mono = _progression(C_F_G_C)
    stereo = np.stack([mono, mono])  # (ch, n)
    assert chords.chord_timeline(stereo, SR) == chords.chord_timeline(mono, SR)


def test_chord_templates_cover_all_24_triads():
    labels, T = chords.chord_templates()
    assert len(labels) == 24 and T.shape == (24, 12)
    assert labels[0] == "C:maj" and labels[12] == "C:min"
    assert np.allclose(np.linalg.norm(T, axis=1), 1.0)


def test_audio_frame_roundtrip(tmp_path):
    import soundfile as sf

    from smplstream import cas, frames as F

    y = _progression(C_F_G_C)
    wav = tmp_path / "prog.wav"
    sf.write(str(wav), y, SR, subtype="FLOAT")
    blob_hash = cas.put_audio_bytes(wav.read_bytes())
    af = F.audio_frame(blob_hash, sr=SR, ch=1, dur=8.0, role="source")

    derived = chords.chords_audio_frame(af)
    assert len(derived) == 2
    marker, feature = derived

    expected_params = {
        "hop_length": 512,
        "smooth_frames": 9,
        "silence_db": -60.0,
        "sr_hz": SR,
    }

    assert marker["kind"] == "marker"
    assert marker["role"] == "chord"
    assert marker["of"] == af["id"]
    assert marker["lineage"] == [af["id"]]
    assert marker["op"] == "chords"
    assert marker["op_version"] == "chords@1"
    assert marker["params"] == expected_params
    assert [p["label"] for p in marker["data"]] == C_F_G_C_LABELS

    assert feature["kind"] == "feature"
    assert feature["role"] == "key"
    assert feature["of"] == af["id"]
    assert feature["lineage"] == [af["id"]]
    assert feature["op"] == "chords"
    assert feature["op_version"] == "chords@1"
    assert feature["params"] == expected_params
    assert set(feature["data"].keys()) == TONAL_KEYS
    assert feature["data"]["tonal.key_key"] == "C"


def test_frames_validate_against_the_spec(tmp_path):
    import soundfile as sf

    from smplstream import cas, frames as F

    wav = tmp_path / "prog.wav"
    sf.write(str(wav), _progression(C_F_G_C), SR, subtype="FLOAT")
    af = F.audio_frame(cas.put_audio_bytes(wav.read_bytes()), sr=SR, ch=1, dur=8.0)

    for frame in chords.chords_audio_frame(af):
        assert F.validate_frame(frame) == []


def test_chord_spans_join_against_downbeat_style_markers():
    """Chord-per-bar is derivable: chord spans and beat markers share the t/sample timebase."""
    markers = chords.chord_timeline(_progression(C_F_G_C), SR)["markers"]
    # A downbeat marker frame (vault-32n3) carries points of the same {t, sample} shape.
    downbeats = [{"t": float(b), "sample": int(round(b * SR))} for b in (0.0, 2.0, 4.0, 6.0)]

    def chord_at(t: float):
        for p in markers:
            if p["t"] <= t < p["t"] + p["dur"]:
                return p["label"]
        return None

    # Sample the chord just inside each bar so a few-frame boundary shift can't alias.
    assert [chord_at(d["t"] + 0.25) for d in downbeats] == C_F_G_C_LABELS


def test_op_version_constant():
    assert chords.OP == "chords"
    assert chords.OP_VERSION == "chords@1"
