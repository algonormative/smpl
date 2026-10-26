"""Tests for layer interaction (smpl_analysis.clash; ticket vault-16dd.4).

Every case is a synthetic signal with a known answer, with the negative control beside it:
two sub sines collide, a sub sine and a hi-hat do not; a minor second is more dissonant than
an octave (and an octave is near zero); a pitch class one layer plays and the other lacks is
named; a loop that closes on a whole number of cycles passes the seam check and the same
tone cut mid-cycle is flagged. Signals are synthesized at the analysis rate so nothing is
resampled; the frame wrapper is covered in ``packages/smpl/tests/test_clash_cli.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from smpl_analysis import clash as C

SR = C.SR
DUR = 4.0


def _t(dur=DUR, sr=SR):
    return np.arange(int(dur * sr)) / sr


def sine(f, amp=0.5, dur=DUR, sr=SR):
    return amp * np.sin(2 * np.pi * f * _t(dur, sr))


def hats(dur=DUR, sr=SR, seed=7):
    """Bursts of high-passed noise every 250 ms: energy only above 6 kHz."""
    from scipy.signal import butter, sosfilt

    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    burst = rng.standard_normal(n)
    env = np.zeros(n)
    for start in range(0, n, sr // 4):
        k = min(int(0.04 * sr), n - start)
        env[start:start + k] = np.exp(-np.linspace(0, 8, k))
    sos = butter(6, 6000, "hp", fs=sr, output="sos")
    return 0.4 * sosfilt(sos, burst) * env


def pair(a, b, **kw):
    return C.measure_pair(C.analyse_layer(a, SR), C.analyse_layer(b, SR), **kw)


# --- sub collision + ERB overlap --------------------------------------------------------------


def test_sub_sine_pair_collides_and_sub_vs_hat_does_not():
    sub_pair = pair(sine(45), sine(52, 0.4))
    sub_hat = pair(sine(45), hats())
    assert sub_pair["sub"]["share_a"] > 0.95 and sub_pair["sub"]["share_b"] > 0.95
    assert sub_pair["sub"]["collision"] > 0.95
    assert sub_hat["sub"]["share_b"] < 0.05           # the hat has nothing below 80 Hz
    assert sub_hat["sub"]["collision"] < 0.05


def test_erb_overlap_sees_the_shared_band_not_the_disjoint_ones():
    shared = pair(sine(45), sine(52, 0.4))["erb"]
    apart = pair(sine(45), hats())["erb"]
    assert shared["overlap"] > 0.3
    assert apart["overlap"] < 0.05
    top = shared["worst_bands"][0]
    assert top["lo_hz"] <= 45 <= top["hi_hz"] and top["co_active"] > 0.9


def test_crossover_is_a_flag_not_a_constant():
    # 100 Hz sits above the default 80 Hz crossover and below a 120 Hz one.
    a, b = sine(100), sine(100, 0.3)
    assert pair(a, b)["sub"]["collision"] < 0.05
    la, lb = C.analyse_layer(a, SR, crossover_hz=120), C.analyse_layer(b, SR, crossover_hz=120)
    assert C.measure_pair(la, lb)["sub"]["collision"] > 0.9


def test_high_pass_on_the_drone_reduces_the_collision():
    from scipy.signal import butter, sosfilt

    rng = np.random.default_rng(3)
    drone = sine(55, 0.5) + sine(110, 0.3) + 0.01 * rng.standard_normal(int(DUR * SR))
    kick = sine(50, 0.5) * np.exp(-3 * (_t() % 0.5))
    before = pair(drone, kick)["sub"]["collision"]
    after = pair(sosfilt(butter(4, 60, "hp", fs=SR, output="sos"), drone), kick)["sub"]["collision"]
    assert before > 0.5 and after < before - 0.2


# --- pitch classes + dissonance ----------------------------------------------------------------


def test_minor_second_is_more_dissonant_than_an_octave():
    minor2 = pair(sine(440), sine(466.16))["pitch"]["dissonance"]
    octave = pair(sine(440), sine(880))["pitch"]["dissonance"]
    unison = pair(sine(440), sine(440, 0.3))["pitch"]["dissonance"]
    assert minor2 > 0.1
    assert octave < 0.02 and unison < 0.02
    assert minor2 > 5 * max(octave, 1e-6)


def test_foreign_pitch_classes_are_named():
    p = pair(sine(440), sine(466.16))["pitch"]       # A against A#
    assert p["dominant_a"] == ["A"] and p["dominant_b"] == ["A#"]
    assert p["foreign_a"] == ["A"] and p["foreign_b"] == ["A#"]
    assert ["A", "A#"] in p["semitone_a"] and ["A#", "A"] in p["semitone_b"]
    same = pair(sine(440), sine(880, 0.3))["pitch"]   # an octave is the same class
    assert same["foreign_a"] == [] and same["foreign_b"] == []


def test_pair_features_carry_only_registered_keys():
    feats = C.pair_features(pair(sine(440), sine(466.16)))
    assert set(feats) == set(C.PAIR_KEYS)


def test_markers_land_where_the_collision_is():
    # The sub sine only plays from 6 s to the end of a 10 s clip; the kick sine plays throughout.
    n = int(10 * SR)
    a = np.zeros(n)
    a[int(6 * SR):] = sine(50, 0.5, dur=4.0)
    spans = pair(a, sine(48, 0.5, dur=10.0))["spans"]["sub-collision"]
    assert spans and spans[0][0] >= 5.5


# --- loop seam ---------------------------------------------------------------------------------


def _tone(n, f=240.0, sr=48000, seed=1):
    t = np.arange(n) / sr
    rng = np.random.default_rng(seed)
    return 0.5 * np.sin(2 * np.pi * f * t) + 0.15 * np.sin(2 * np.pi * 3 * f * t) \
        + 0.002 * rng.standard_normal(n)


def test_a_whole_number_of_cycles_closes_and_a_mid_cycle_cut_does_not():
    closed = C.measure_seam(_tone(48000), 48000)        # 240 whole cycles
    cut = C.measure_seam(_tone(47930), 48000)           # 0.35 cycle short: lands mid-note
    assert not closed["flagged"] and closed["jump_ratio"] < 3.0
    assert cut["flagged"] and "sample-jump" in cut["reasons"]
    assert cut["jump_ratio"] > 4 * closed["jump_ratio"]


def test_a_crossfaded_loop_passes_where_the_hard_cut_fails():
    sr, n, cf = 48000, 48000 + 7000, 7000
    rng = np.random.default_rng(5)
    noise = np.convolve(rng.standard_normal(n), np.ones(40) / 40, "same")
    tone = 0.4 * np.sin(2 * np.pi * 173.3 * np.arange(n) / sr) + 3 * noise
    hard = tone[:48000]
    loop = tone[:48000].copy()
    th = np.linspace(0, np.pi / 2, cf)
    loop[:cf] = tone[:cf] * np.sin(th) + tone[48000:48000 + cf] * np.cos(th)
    assert C.measure_seam(hard, sr)["jump_ratio"] > 4 * C.measure_seam(loop, sr)["jump_ratio"]
    assert not C.measure_seam(loop, sr)["flagged"]


def test_seam_tolerances_are_flags():
    cut = _tone(47930)
    assert C.measure_seam(cut, 48000)["flagged"]
    assert not C.measure_seam(cut, 48000, jump_ratio=1e6, rms_excess_db=1e6,
                              flux_ratio=1e9)["flagged"]


def test_seam_reports_every_number_and_rejects_a_tiny_clip():
    s = C.measure_seam(_tone(48000), 48000)
    assert set(C.seam_features(s)) == set(C.SEAM_KEYS)
    for k in ("jump", "deriv_jump", "rms_jump_db", "flux_wrap", "flux_median"):
        assert s[k] is not None
    with pytest.raises(ValueError):
        C.measure_seam(np.zeros(10), 48000)
