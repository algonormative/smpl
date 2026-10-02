"""Short one-shot regression for the FEATURE tier (ticket vault-2puw).

Sibling of the image-tier regression in ``test_spectrogram.py`` (vault-2t9g). The
feature-tier ops pass a fixed 2048-sample analysis window to librosa: ``spectral``
and ``clarity`` via ``librosa.stft``, ``movement`` via ``librosa.feature.rms``'s
``frame_length``. A sub-window one-shot (the 1h1v campaign generates many) used to
trip librosa's "n_fft too large for input signal" path — a warning plus a
degenerate transform on librosa 0.11, a hard ``ParameterError`` (dropped/corrupt
stats) on older librosa. All three now pad up to one full window first
(``spectrogram.pad_short_signal``), so the analysis is always valid and the warning
never fires.
"""

from __future__ import annotations

import io
import warnings

import numpy as np
import pytest
import soundfile as sf

from smpl_analysis import clarity as CL, movement as MV, spectral as SP

SR = 44100

_TOO_LARGE = "too large for input signal"


def _one_shot(n: int, sr: int = SR) -> np.ndarray:
    """A sub-window blip: ``n`` samples of a decaying 220 Hz tone."""
    t = np.arange(n) / sr
    return (0.3 * np.sin(2 * np.pi * 220 * t) * np.exp(-t * 400.0)).astype("float32")


def _put_wav(samples, sr: int = SR):
    from smplstream import cas, frames as F

    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="FLOAT")
    h = cas.put_audio_bytes(buf.getvalue())
    meta = cas.read_meta(h) or {}
    return F.audio_frame(h, sr=meta.get("sr", sr), ch=meta.get("ch", samples.shape[1]),
                         dur=meta.get("dur", samples.shape[0] / sr), role="source")


def _no_too_large(caught) -> bool:
    return not any(_TOO_LARGE in str(w.message) for w in caught)


# --- scalar paths ----------------------------------------------------------------------------


@pytest.mark.parametrize("n", [255, 173, 64, 4])  # sub-256 one-shots down to an ultra-short blip
def test_spectral_shape_short_one_shot(n):
    y = _one_shot(n)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data = SP.spectral_shape(y, SR)
    assert _no_too_large(caught)
    assert set(data) == {
        "lowlevel.spectral_flatness_db", "lowlevel.spectral_crest", "lowlevel.spectral_spread",
        "lowlevel.spectral_rolloff", "lowlevel.spectral_contrast", "lowlevel.spectral_slope",
        "lowlevel.spectral_skewness", "lowlevel.spectral_kurtosis",
    }
    for key, stat in data.items():
        assert np.isfinite(stat["mean"]), key
        assert np.isfinite(stat["stdev"]), key


@pytest.mark.parametrize("n", [255, 173, 64, 4])
def test_clarity_scalars_short_one_shot(n):
    y = _one_shot(n)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data = CL.clarity_scalars(y, SR)
    assert _no_too_large(caught)
    assert set(data) == set(CL.KEYS)
    assert all(v is not None and np.isfinite(v) for v in data.values())


@pytest.mark.parametrize("n", [255, 173, 64, 4])
def test_movement_scalars_short_one_shot(n):
    y = _one_shot(n)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data = MV.movement_scalars(y, SR)
    assert _no_too_large(caught)
    assert set(data) == set(MV.KEYS)
    assert all(v is not None and np.isfinite(v) for v in data.values())


@pytest.mark.parametrize("n", [255, 173, 64, 4])
def test_movement_rms_envelope_not_collapsed(n):
    """The RMS envelope of a sub-window one-shot keeps real frames (padded, not degenerate)."""
    env = MV._rms_env(_one_shot(n), SR)
    # padded to one FRAME_LENGTH window, centered → 1 + FRAME_LENGTH // HOP_LENGTH frames
    assert env.size == 1 + MV.FRAME_LENGTH // MV.HOP_LENGTH
    assert np.all(np.isfinite(env))


# --- frame-emitting paths --------------------------------------------------------------------


def test_spectral_audio_frame_short_one_shot(tmp_path, monkeypatch):
    """A <256-sample one-shot still emits exactly one spectral feature frame, warning-free."""
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    af = _put_wav(_one_shot(120))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        derived = SP.spectral_audio_frame(af)
    assert _no_too_large(caught)
    assert len(derived) == 1
    feat = derived[0]
    assert feat["kind"] == "feature" and feat["role"] == "spectral"
    assert all(np.isfinite(stat["mean"]) for stat in feat["data"].values())


@pytest.mark.parametrize(
    "mod, role",
    [(CL, "clarity"), (MV, "movement")],
    ids=["clarity", "movement"],
)
def test_gated_families_still_emit_one_frame(mod, role, tmp_path, monkeypatch):
    """clarity/movement duration-gate a <256-sample one-shot — gated, but never dropped."""
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    af = _put_wav(_one_shot(120))
    fn = getattr(mod, f"{role}_audio_frame")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        derived = fn(af)
    assert _no_too_large(caught)
    assert len(derived) == 1
    feat = derived[0]
    assert feat["kind"] == "feature" and feat["role"] == role
    assert set(feat["data"]) == set(mod.KEYS)
    assert feat["params"]["gated"] is True
