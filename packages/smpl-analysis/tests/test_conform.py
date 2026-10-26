"""Tests for `conform` (smpl_analysis.conform; vault-16dd.2).

Controls are synthetic with known answers: a hit loop whose hits sit exactly where its intent
says (reads ~0 ms) and the same loop shifted 20 ms late (reads ~20 ms, signed +), and a
three-note tonal clip played as intended (rotation 0, nothing flagged) next to the same
notes two semitones up (rotation +2, off-set classes flagged).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf
from smpl_analysis import conform

SR = 22050
BPM = 120.0


def _hit(sr=SR, seconds=0.12):
    n = int(sr * seconds)
    t = np.arange(n) / sr
    env = np.exp(-t * 30)
    return (env * np.sin(2 * np.pi * 70 * t) + 0.5 * np.exp(-t * 300) * np.sin(2 * np.pi * 1800 * t)).astype("float64")


def hit_loop(times, dur=4.0, sr=SR):
    y = np.zeros(int(dur * sr))
    h = _hit(sr)
    for t in times:
        i = round(t * sr)
        seg = h[: len(y) - i]
        y[i:i + len(seg)] += seg
    return y[:, None]


def shifted(x, ms):
    """The loop circularly shifted `ms` later (the loop is tiled, so wrap is the loop's own tail)."""
    return np.roll(x, round(ms / 1000 * SR), axis=0)


HIT_TIMES = [0.0, 0.5, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5]
INTENT_HITS = {"hits": [{"t": t, "velocity": 1.0, "track": "kick"} for t in HIT_TIMES], "dur_s": 4.0}


def chord(semitones=0.0, dur=3.0, sr=SR):
    t = np.arange(int(dur * sr)) / sr
    y = np.zeros_like(t)
    for f0 in (220.0, 261.63, 329.63):                        # A3 C4 E4
        f0 *= 2 ** (semitones / 12)
        for k, a in ((1, 1.0), (2, 0.4), (3, 0.2)):
            y += a * np.sin(2 * np.pi * f0 * k * t)
    y *= 0.2 * np.minimum(1.0, np.minimum(t, dur - t) / 0.05)
    return y[:, None]


TONAL = {"pitch_classes": ["A", "C", "E"]}


# --- timing -----------------------------------------------------------------
def test_timing_control_on_grid_reads_zero():
    r = conform.conform_signal(hit_loop(HIT_TIMES), SR, INTENT_HITS, checks=("timing",))
    f = r["features"]
    assert f["conform.timing_n"] == len(HIT_TIMES)
    assert f["conform.timing_median_ms"] < 1.0
    assert abs(f["conform.timing_mean_ms"]) < 1.0
    assert r["markers"] == [] and f["conform.over_tol"] == []


def test_timing_positive_control_20ms_late_reads_20():
    r = conform.conform_signal(shifted(hit_loop(HIT_TIMES), 20), SR, INTENT_HITS, checks=("timing",))
    f = r["features"]
    assert f["conform.timing_median_ms"] == pytest.approx(20.0, abs=1.5)
    assert f["conform.timing_p90_ms"] == pytest.approx(20.0, abs=2.0)
    assert f["conform.timing_mean_ms"] == pytest.approx(20.0, abs=1.5)      # signed: late
    assert len(r["markers"]) == f["conform.timing_n"]                       # every hit is > 8 ms off
    assert r["markers"][0]["error_ms"] == pytest.approx(20.0, abs=2.0)
    assert f["conform.over_tol"] == ["timing"]


def test_timing_early_is_negative_and_tolerance_is_a_flag():
    r = conform.conform_signal(shifted(hit_loop(HIT_TIMES), -12), SR, INTENT_HITS,
                               checks=("timing",), timing_tol_ms=15.0)
    assert r["features"]["conform.timing_mean_ms"] == pytest.approx(-12.0, abs=1.5)
    assert r["features"]["conform.timing_median_ms"] == pytest.approx(12.0, abs=1.5)
    assert r["markers"] == [] and r["features"]["conform.over_tol"] == []


def test_ghosts_and_crowded_hits_are_not_measured_and_few_hits_read_none():
    intent = {"hits": [{"t": 0.0}, {"t": 0.5, "velocity": 0.3}, {"t": 1.0},
                       {"t": 1.03}, {"t": 2.0}, {"t": 3.0}]}
    f = conform.conform_signal(hit_loop([0.0, 0.5, 1.0, 1.03, 2.0, 3.0]), SR, intent,
                               checks=("timing",))["features"]
    assert f["conform.timing_n"] == 3 and f["conform.timing_intended_n"] == 6   # ghost + the 30 ms pair out
    two = {"hits": [{"t": 0.0}, {"t": 2.0}]}
    f = conform.conform_signal(hit_loop([0.0, 2.0]), SR, two, checks=("timing",))["features"]
    assert f["conform.timing_n"] == 2 and f["conform.timing_median_ms"] is None


def test_clip_shorter_than_the_search_window_reads_none_not_garbage():
    f = conform.conform_signal(hit_loop([0.0], dur=0.02), SR, {"hits": [{"t": 0.0}] * 1}, checks=("timing",))["features"]
    assert f["conform.timing_n"] == 0 and f["conform.timing_median_ms"] is None


def test_no_hits_intended_means_timing_is_none_not_zero():
    f = conform.conform_signal(hit_loop(HIT_TIMES), SR, {}, checks=("timing",))["features"]
    assert f["conform.timing_median_ms"] is None and f["conform.timing_n"] is None


# --- length -----------------------------------------------------------------
def test_length_measured_intended_delta():
    x = hit_loop(HIT_TIMES)
    f = conform.conform_signal(x, SR, INTENT_HITS, checks=("length",))["features"]
    assert f["conform.length_s"] == 4.0 and f["conform.length_delta_ms"] == 0.0
    assert f["conform.over_tol"] == []
    long = np.concatenate([x, np.zeros((SR // 50, 1))])                     # 20 ms long
    f = conform.conform_signal(long, SR, INTENT_HITS, checks=("length",))["features"]
    assert f["conform.length_delta_ms"] == pytest.approx(20.0, abs=0.1) and f["conform.over_tol"] == ["length"]
    f = conform.conform_signal(long, SR, INTENT_HITS, checks=("length",), length_tol_ms=25.0)["features"]
    assert f["conform.over_tol"] == []
    f = conform.conform_signal(long, SR, {}, checks=("length",))["features"]
    assert f["conform.length_s"] > 4.0 and f["conform.length_delta_ms"] is None


# --- pitch classes ----------------------------------------------------------
def test_pitch_control_original_fits_and_shifted_copy_flags():
    ok = conform.conform_signal(chord(0), SR, TONAL, checks=("pitch",))["features"]
    assert ok["conform.pc_best_rotation"] == 0 and ok["conform.pc_fit_margin"] > 0.2
    assert ok["conform.pc_offset_flagged"] == {} and ok["conform.over_tol"] == []
    assert ok["conform.pc_in_set_share"] > 0.7
    assert sum(ok["conform.pc_shares"].values()) == pytest.approx(1.0, abs=0.01)

    up = conform.conform_signal(chord(2), SR, TONAL, checks=("pitch",))["features"]
    assert up["conform.pc_best_rotation"] == 2 and up["conform.pc_fit_margin"] < 0
    assert set(up["conform.pc_offset_flagged"]) >= {"B", "D", "F#"}
    assert up["conform.pc_in_set_share"] < 0.2 and up["conform.over_tol"] == ["pitch"]


def test_pitch_threshold_is_a_flag():
    up = conform.conform_signal(chord(2), SR, TONAL, checks=("pitch",), pc_flag_share=0.99)["features"]
    assert up["conform.pc_offset_flagged"] == {}
    assert up["conform.pc_best_rotation"] == 2          # the rotation test still sees it


def test_no_pitch_intent_is_unmeasured_and_allowed_pcs_reproduce_the_floor():
    f = conform.conform_signal(chord(0), SR, {}, checks=("pitch",))["features"]
    assert f["conform.pc_in_set_share"] is None and f["conform.pc_shares"] is None
    white = list("ABCDEFG")
    f = conform.conform_signal(chord(0), SR, {"pitch_classes": ["A", "C", "G#"]}, checks=("pitch",),
                               allowed_pcs=white)["features"]
    assert f["conform.pc_outside_allowed"] == ["G#"] and "allowed" in f["conform.over_tol"]
    f = conform.conform_signal(chord(0), SR, TONAL, checks=("pitch",), allowed_pcs=white)["features"]
    assert f["conform.pc_outside_allowed"] == []


def test_pitch_fit_rotation_symmetric_set_has_no_margin():
    shares = np.full(12, 1 / 12)
    fit = conform.pitch_fit(shares, ["C", "E", "G#"])      # augmented triad maps onto itself
    assert fit["fit_margin"] is None and fit["best_rotation"] == 0
    assert conform.pitch_fit(None, ["C"]) is None


# --- loudness ---------------------------------------------------------------
def test_loudness_measured_intended_delta_and_flag_override():
    t = np.arange(SR * 2) / SR
    x = (0.1 * np.sin(2 * np.pi * 997 * t))[:, None]        # -20 dBFS sine: ~ -23 LUFS, -20 dBTP
    f = conform.conform_signal(x, SR, {"lufs": -14.0, "true_peak_db": -1.0}, checks=("loudness",))["features"]
    assert f["conform.lufs"] == pytest.approx(-23.0, abs=0.3)
    assert f["conform.true_peak_dbtp"] == pytest.approx(-20.0, abs=0.1)
    assert f["conform.lufs_delta"] == pytest.approx(f["conform.lufs"] + 14.0, abs=0.01)
    assert f["conform.true_peak_delta_db"] == pytest.approx(-19.0, abs=0.1)
    f = conform.conform_signal(x, SR, {"lufs": -14.0}, checks=("loudness",), lufs=-23.0)["features"]
    assert f["conform.lufs_intended"] == -23.0 and abs(f["conform.lufs_delta"]) < 0.3
    f = conform.conform_signal(x, SR, {}, checks=("loudness",))["features"]
    assert f["conform.lufs_delta"] is None and f["conform.true_peak_delta_db"] is None


def test_loudness_pads_sub_second_clips_like_the_gate():
    t = np.arange(int(SR * 0.5)) / SR
    x = (0.5 * np.sin(2 * np.pi * 997 * t))[:, None]
    f = conform.conform_signal(x, SR, {}, checks=("loudness",))["features"]
    assert f["conform.lufs"] is not None and f["conform.true_peak_dbtp"] == pytest.approx(-6.0, abs=0.1)


# --- files and frames -------------------------------------------------------
def test_conform_file_reads_the_provenance_sidecar(tmp_path):
    wav = tmp_path / "loop.wav"
    sf.write(str(wav), shifted(hit_loop(HIT_TIMES), 20), SR, subtype="FLOAT")
    (tmp_path / "loop.wav.provenance.json").write_text(json.dumps(
        {"bpm": 120, "bars": 2, "pitch_classes": ["A"],
         "recipe": {"bpm": 120, "beats_per_bar": 4, "grid_steps": 16, "tracks": [{"name": "k", "hits": [
             {"bar": 1, "step": 1}, {"bar": 1, "step": 5}, {"bar": 1, "step": 9},
             {"bar": 2, "step": 1}, {"bar": 2, "step": 5}]}]}}))
    r = conform.conform_file(wav, checks=("length", "timing"))
    f = r["features"]
    assert f["conform.length_intended_s"] == 4.0 and f["conform.length_delta_ms"] == 0.0
    assert f["conform.timing_n"] == 5 and f["conform.timing_median_ms"] == pytest.approx(20.0, abs=1.5)
    assert [m["error_ms"] > 0 for m in r["markers"]] == [True] * 5


def test_audio_frame_emits_a_feature_and_a_marker_frame(tmp_path, monkeypatch):
    from smplstream import cas
    from smplstream import frames as F

    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    wav = tmp_path / "a.wav"
    sf.write(str(wav), shifted(hit_loop(HIT_TIMES), 20), SR, subtype="FLOAT")
    intent_file = tmp_path / "a.intent.json"
    intent_file.write_text(json.dumps({"hits": INTENT_HITS["hits"], "dur_s": 4.0}))
    blob = cas.put_audio_bytes(wav.read_bytes())
    audio = F.audio_frame(blob, sr=SR, ch=1, dur=4.0, role="source")
    out = conform.conform_audio_frame(audio, intent_path=str(intent_file), checks=("length", "timing"))
    assert [(f["kind"], f["role"]) for f in out] == [("feature", "conform"), ("marker", "conform")]
    assert out[0]["op"] == "conform" and out[0]["of"] == audio["id"]
    assert out[0]["data"]["conform.timing_median_ms"] == pytest.approx(20.0, abs=1.5)


def test_every_feature_key_is_present_even_with_no_intent():
    f = conform.conform_signal(chord(0), SR, None)["features"]
    assert set(f) == set(conform.FEATURE_KEYS)


def test_pc_concentration_flat_vs_single_class():
    import numpy as np
    from smpl_analysis.conform import pc_concentration

    assert pc_concentration(np.full(12, 1 / 12)) == 0.0
    one = np.zeros(12)
    one[9] = 1.0
    assert pc_concentration(one) > 0.99 and pc_concentration(None) is None
