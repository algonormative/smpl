"""Tests for the AudioCommons timbral descriptors (smpl_analysis.timbre; vault-14ia).

Directional sanity + range/determinism, never golden values: this is a reimplementation
calibrated on an internal 0–100 scale, so pinning exact numbers would pin the calibration
rather than the behaviour.
"""

from __future__ import annotations

import numpy as np

from smpl_analysis import timbre

SR = 44100

# The exact keys the registry (feature-keys.md, owner vault-14ia) assigns to this op.
EXPECTED_KEYS = {
    "timbre.hardness",
    "timbre.depth",
    "timbre.brightness",
    "timbre.roughness",
    "timbre.warmth",
    "timbre.sharpness",
    "timbre.boominess",
    "timbre.reverb",
}


# --- signal fixtures (synthesized, no binary fixtures) ---------------------------------------


def _t(dur: float, sr: int = SR) -> np.ndarray:
    return np.arange(int(sr * dur)) / sr


def _sine(freq: float, dur: float = 1.0, amp: float = 0.5) -> np.ndarray:
    return (amp * np.sin(2 * np.pi * freq * _t(dur))).astype("float32")


def _noise(dur: float = 1.0, amp: float = 0.3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (amp * rng.standard_normal(int(SR * dur))).astype("float32")


def _hiss(dur: float = 1.0, seed: int = 1) -> np.ndarray:
    """Bright noise: white noise high-passed at 5 kHz."""
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, 5000 / (SR / 2), btype="high", output="sos")
    return sosfiltfilt(sos, _noise(dur, 0.4, seed)).astype("float32")


def _kick(dur: float = 0.4, decay: float = 25.0) -> np.ndarray:
    """A dry kick: pitch-swept sine with a fast exponential decay."""
    tt = _t(dur)
    freq = 120 * np.exp(-tt * 30) + 50
    phase = 2 * np.pi * np.cumsum(freq) / SR
    return (0.8 * np.exp(-tt * decay) * np.sin(phase)).astype("float32")


def _beating(dur: float = 1.0) -> np.ndarray:
    """Two close tones (500/530 Hz) — 30 Hz beating, squarely in the roughness band."""
    tt = _t(dur)
    return (0.4 * (np.sin(2 * np.pi * 500 * tt) + np.sin(2 * np.pi * 530 * tt))).astype("float32")


def _pad_out(y: np.ndarray, tail_s: float = 1.5) -> np.ndarray:
    return np.concatenate([y, np.zeros(int(SR * tail_s), dtype="float32")]).astype("float32")


def _reverbed(dry: np.ndarray, rt60_s: float = 1.2, seed: int = 3) -> np.ndarray:
    """Convolve with an exponentially-decaying noise impulse response."""
    rng = np.random.default_rng(seed)
    n = int(SR * rt60_s)
    ir = rng.standard_normal(n) * np.exp(-6.9078 * np.arange(n) / n)
    ir[0] += 3.0  # direct sound
    wet = np.convolve(dry, ir)[: dry.size + n]
    return (0.5 * wet / (np.max(np.abs(wet)) + 1e-9)).astype("float32")


# --- shape / contract -------------------------------------------------------------------------


def test_emits_exactly_the_registered_keys():
    data = timbre.timbral_descriptors(_sine(440.0), SR)
    assert set(data.keys()) == EXPECTED_KEYS


def test_all_scores_in_range_and_jsonable():
    for sig in (_sine(220.0), _noise(), _hiss(), _kick(), _beating()):
        data = timbre.timbral_descriptors(sig, SR)
        for key, val in data.items():
            if key == "timbre.reverb":
                continue
            assert isinstance(val, float), key
            assert np.isfinite(val), key
            assert 0.0 <= val <= 100.0, (key, val)


def test_reverb_is_exactly_binary():
    for sig in (_sine(220.0), _noise(), _pad_out(_kick()), _reverbed(_pad_out(_kick()))):
        val = timbre.timbral_descriptors(sig, SR)["timbre.reverb"]
        assert val in (0, 1), val
        assert isinstance(val, int)


def test_deterministic_across_runs():
    sig = _noise()
    assert timbre.timbral_descriptors(sig, SR) == timbre.timbral_descriptors(sig, SR)


def test_silence_is_finite_and_in_range():
    silence = np.zeros(SR, dtype="float32")
    data = timbre.timbral_descriptors(silence, SR)
    assert set(data.keys()) == EXPECTED_KEYS
    for key, val in data.items():
        assert np.isfinite(val), key
        assert 0.0 <= val <= 100.0, (key, val)


def test_stereo_input_collapses_to_mono():
    mono = _kick()
    stereo = np.vstack([mono, mono])  # (ch, n)
    assert timbre.timbral_descriptors(stereo, SR) == timbre.timbral_descriptors(mono, SR)


# --- directional sanity ------------------------------------------------------------------------


def test_brightness_hiss_over_dark_sine():
    bright = timbre.timbral_descriptors(_hiss(), SR)["timbre.brightness"]
    dark = timbre.timbral_descriptors(_sine(80.0), SR)["timbre.brightness"]
    assert bright > dark + 40.0


def test_sharpness_hiss_over_low_tone():
    hiss = timbre.timbral_descriptors(_hiss(), SR)["timbre.sharpness"]
    low = timbre.timbral_descriptors(_sine(80.0), SR)["timbre.sharpness"]
    assert hiss > low + 40.0


def test_boominess_kick_over_white_noise():
    kick = timbre.timbral_descriptors(_kick(), SR)["timbre.boominess"]
    noise = timbre.timbral_descriptors(_noise(), SR)["timbre.boominess"]
    assert kick > noise + 40.0


def test_depth_low_sine_over_hiss():
    low = timbre.timbral_descriptors(_sine(60.0), SR)["timbre.depth"]
    hiss = timbre.timbral_descriptors(_hiss(), SR)["timbre.depth"]
    assert low > hiss + 40.0


def test_warmth_low_mid_tone_over_hiss():
    warm = timbre.timbral_descriptors(_sine(220.0), SR)["timbre.warmth"]
    hiss = timbre.timbral_descriptors(_hiss(), SR)["timbre.warmth"]
    assert warm > hiss + 40.0


def test_roughness_beating_pair_over_pure_tone():
    rough = timbre.timbral_descriptors(_beating(), SR)["timbre.roughness"]
    smooth = timbre.timbral_descriptors(_sine(515.0), SR)["timbre.roughness"]
    assert rough > smooth + 20.0


def test_hardness_percussive_hit_over_slow_swell():
    hit = timbre.timbral_descriptors(_pad_out(_noise(0.05, 0.6), 0.5), SR)["timbre.hardness"]
    tt = _t(2.0)
    swell = (0.3 * np.minimum(tt / 1.5, 1.0) * np.sin(2 * np.pi * 220 * tt)).astype("float32")
    soft = timbre.timbral_descriptors(swell, SR)["timbre.hardness"]
    assert hit > soft + 20.0


def test_reverb_flags_wet_tail_not_dry():
    dry = _pad_out(_kick())
    wet = _reverbed(dry)
    assert timbre.timbral_descriptors(dry, SR)["timbre.reverb"] == 0
    assert timbre.timbral_descriptors(wet, SR)["timbre.reverb"] == 1


def test_reverb_zero_for_steady_tone_and_noise():
    # A steady tone / steady noise spans 20 dB on the EDC only because it stops; the decay
    # is not exponential, so the linearity gate must keep it at 0.
    assert timbre.timbral_descriptors(_sine(440.0), SR)["timbre.reverb"] == 0
    assert timbre.timbral_descriptors(_noise(), SR)["timbre.reverb"] == 0


# --- helper-level behaviour ---------------------------------------------------------------------


def test_attack_ms_fast_hit_under_slow_swell():
    fast = timbre.attack_ms(_pad_out(_noise(0.05, 0.6), 0.5), SR)
    tt = _t(2.0)
    swell = (0.3 * np.minimum(tt / 1.5, 1.0) * np.sin(2 * np.pi * 220 * tt)).astype("float32")
    assert fast < timbre.attack_ms(swell, SR)


def test_decay_estimate_linear_for_exponential_tail():
    rt60, r2 = timbre.decay_estimate(_pad_out(_kick(decay=8.0)), SR)
    assert r2 > 0.98           # an exponential decay is a straight line in dB
    assert rt60 > 400.0


# --- frame emission ------------------------------------------------------------------------------


def test_audio_frame_roundtrip(tmp_path):
    import soundfile as sf

    from smplstream import cas, frames as F

    wav = tmp_path / "kick.wav"
    sf.write(str(wav), _kick(), SR, subtype="FLOAT")
    blob_hash = cas.put_audio_bytes(wav.read_bytes())
    af = F.audio_frame(blob_hash, sr=SR, ch=1, dur=0.4, role="source")

    derived = timbre.timbre_audio_frame(af)
    assert len(derived) == 1
    feat = derived[0]
    assert feat["kind"] == "feature"
    assert feat["role"] == "timbre"
    assert feat["of"] == af["id"]
    assert feat["op"] == "timbre"
    assert feat["op_version"] == "timbre@1"
    assert set(feat["data"].keys()) == EXPECTED_KEYS
    assert feat["params"]["n_fft"] == 2048
    assert feat["params"]["hop_length"] == 512


def test_op_version_constant():
    assert timbre.OP == "timbre"
    assert timbre.OP_VERSION == "timbre@1"
