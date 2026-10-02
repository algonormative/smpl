"""Tests for the spectral-shape feature family (smpl_analysis.spectral; vault-3uap)."""

from __future__ import annotations

import numpy as np
import pytest

from smpl_analysis import spectral

# The exact keys the registry (feature-keys.md) assigns to this op.
EXPECTED_KEYS = {
    "lowlevel.spectral_flatness_db",
    "lowlevel.spectral_crest",
    "lowlevel.spectral_spread",
    "lowlevel.spectral_rolloff",
    "lowlevel.spectral_contrast",
    "lowlevel.spectral_slope",
    "lowlevel.spectral_skewness",
    "lowlevel.spectral_kurtosis",
}


def _sine(freq: float, sr: int = 22050, dur: float = 1.0) -> np.ndarray:
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype("float32")


def _noise(sr: int = 22050, dur: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (0.5 * rng.standard_normal(int(sr * dur))).astype("float32")


def test_emits_exactly_the_registered_keys():
    data = spectral.spectral_shape(_sine(440.0), 22050)
    assert set(data.keys()) == EXPECTED_KEYS


def test_mean_stdev_shape_and_jsonable():
    data = spectral.spectral_shape(_sine(440.0), 22050)
    for key, val in data.items():
        assert set(val.keys()) == {"mean", "stdev"}, key
        assert isinstance(val["mean"], float), key
        assert isinstance(val["stdev"], float), key
        assert np.isfinite(val["mean"]) and np.isfinite(val["stdev"]), key


def test_flatness_db_distinguishes_tone_from_noise():
    # A pure tone is tonal (very low flatness, very negative dB); white noise is flat
    # (flatness near 1, ~0 dB). Noise flatness_db MUST be much higher than a tone's.
    tone = spectral.spectral_shape(_sine(440.0), 22050)
    noise = spectral.spectral_shape(_noise(), 22050)
    assert tone["lowlevel.spectral_flatness_db"]["mean"] < -20.0
    assert noise["lowlevel.spectral_flatness_db"]["mean"] > tone["lowlevel.spectral_flatness_db"]["mean"] + 10.0


def test_spread_and_rolloff_in_hz_range():
    data = spectral.spectral_shape(_noise(), 22050)
    # Spread/rolloff are Hz quantities bounded by Nyquist.
    assert 0.0 <= data["lowlevel.spectral_spread"]["mean"] <= 22050 / 2
    assert 0.0 <= data["lowlevel.spectral_rolloff"]["mean"] <= 22050 / 2


def test_crest_at_least_one():
    # Crest is peak/mean of the magnitude spectrum, so >= 1 by construction.
    data = spectral.spectral_shape(_sine(440.0), 22050)
    assert data["lowlevel.spectral_crest"]["mean"] >= 1.0


def test_audio_frame_roundtrip(tmp_path, monkeypatch):
    import soundfile as sf

    from smplstream import cas, frames as F

    # Isolated CAS (and therefore isolated memo index) so the run is a genuine cold start
    # and the real ~/.smpl is never touched.
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))

    wav = tmp_path / "tone.wav"
    sf.write(str(wav), _sine(440.0), 22050, subtype="FLOAT")
    blob_hash = cas.put_audio_bytes(wav.read_bytes())
    af = F.audio_frame(blob_hash, sr=22050, ch=1, dur=1.0, role="source")

    derived = spectral.spectral_audio_frame(af)
    assert len(derived) == 1
    feat = derived[0]
    assert feat["kind"] == "feature"
    assert feat["role"] == "spectral"
    assert feat["of"] == af["id"]
    assert feat["op"] == "spectral"
    assert feat["op_version"] == "spectral@1"
    assert set(feat["data"].keys()) == EXPECTED_KEYS
    # STFT params plus the memo marker (a cold first run computes).
    assert feat["params"] == {"n_fft": 2048, "hop_length": 512, "cache_hit": False}


def test_op_version_constant():
    assert spectral.OP_VERSION == "spectral@1"
    assert spectral.OP == "spectral"
