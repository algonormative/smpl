"""Tests for the paulstretch edit op (`smpl stretch --paul`, ticket vault-19agn).

Self-contained, same shape as ``test_edit_primitives.py``: CAS a synthetic signal, run the op,
resolve the wet frame, assert. Covers the acceptance bar carried over from the basilica
``build_ambience.py`` stage_proof — exact duration ratio, band-energy preservation of a test
tone, and the OLA-defect detector (no hop-rate line in the energy envelope).

    uv run pytest packages/smpl-analysis/tests/test_paulstretch.py -q
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from smpl_analysis import edit
from smplstream import cas, frames as F

SR = 44_100
WINDOW_S = 0.28


def _frame(x, *, role="tone"):
    """CAS a (frames,) or (frames, ch) float array as WAV and return an `audio` frame."""
    arr = np.asarray(x, dtype="float32")
    ch = 1 if arr.ndim == 1 else arr.shape[1]
    buf = io.BytesIO()
    sf.write(buf, arr, SR, format="WAV", subtype="FLOAT")
    h = cas.put_audio_bytes(buf.getvalue())
    return F.audio_frame(h, sr=SR, ch=ch, dur=arr.shape[0] / SR, role=role)


def _tone_frame(f0=400.0, dur=1.0):
    t = np.arange(int(SR * dur)) / SR
    return _frame(0.6 * np.sin(2 * np.pi * f0 * t)), dur


def _noise_frame(dur=1.0, seed=7):
    rng = np.random.default_rng(seed)
    return _frame(0.3 * rng.standard_normal(int(SR * dur))), dur


def _load(frame):
    d, s = sf.read(str(cas.get_path(frame["hash"])), dtype="float64", always_2d=True)
    return d, int(s)


def _mono(frame):
    d, s = _load(frame)
    return d.mean(axis=1), s


def _band_energy(y, sr, lo, hi):
    spec = np.abs(np.fft.rfft(y))
    f = np.fft.rfftfreq(len(y), 1 / sr)
    return float(np.sum(spec[(f >= lo) & (f < hi)] ** 2))


def _hop_line_vs_floor(y, sr, *, window_s=WINDOW_S, trim_s=0.5):
    """OLA-defect detector: (hop-rate envelope line) / (median 2–20 Hz envelope floor).

    Paulstretch overlap-adds half-windows, so a botched OLA (bad window, or the reference
    impl's ``hinv_buf`` demodulation re-applied) shows up as a spectral line at the output
    hop rate ``2/window_s`` in the signal's energy envelope.
    """
    y = y[int(trim_s * sr): len(y) - int(trim_s * sr)]   # drop the OLA fade in/out edges
    frame_len, hop = 256, 128
    n_frames = (len(y) - frame_len) // hop
    env = np.array([np.sqrt(np.mean(y[i * hop: i * hop + frame_len] ** 2))
                    for i in range(n_frames)])
    env = env - env.mean()
    spec = np.abs(np.fft.rfft(env * np.hanning(len(env))))
    f = np.fft.rfftfreq(len(env), hop / sr)

    f_hop = 2.0 / window_s
    line = float(spec[np.abs(f - f_hop) <= 0.5].max())
    floor_band = (f >= 2.0) & (f <= 20.0) & (np.abs(f - f_hop) > 1.0)
    return line / float(np.median(spec[floor_band]))


def test_duration_ratio_is_exactly_the_factor():
    fr, dur = _tone_frame(dur=1.0)
    y, sr = _mono(edit.apply_paulstretch(fr, factor=8.0))
    assert (len(y) / sr) / dur == pytest.approx(8.0, abs=0.05)


def test_band_energy_of_test_tone_is_preserved():
    fr, _ = _tone_frame(f0=400.0, dur=1.0)
    y, sr = _mono(edit.apply_paulstretch(fr, factor=8.0))
    in_band = _band_energy(y, sr, 350.0, 450.0)
    neighbor = _band_energy(y, sr, 750.0, 850.0)
    assert in_band > 10.0 * neighbor


def test_no_hop_rate_line_in_the_energy_envelope():
    for frame, _ in (_noise_frame(dur=1.0), _tone_frame(f0=400.0, dur=1.0)):
        y, sr = _mono(edit.apply_paulstretch(frame, factor=8.0))
        assert _hop_line_vs_floor(y, sr) < 8.0


def test_params_and_lineage_recorded():
    fr, _ = _tone_frame(dur=0.5)
    wet = edit.apply_paulstretch(fr, factor=8.0, window_s=0.2)
    assert wet["op"] == "paulstretch"
    assert wet["op_version"] == edit.PAULSTRETCH_OP_VERSION == "paulstretch@1"
    assert wet["role"] == "tone.wet"
    assert wet["of"] == fr["id"] and wet["lineage"] == [fr["id"]]
    assert wet["params"]["mode"] == "paul"
    assert wet["params"]["factor"] == 8.0
    assert wet["params"]["window_s"] == 0.2


def test_stereo_decorrelate_flag_recorded_and_skips_the_reblend():
    rng = np.random.default_rng(3)
    fr = _frame(0.3 * rng.standard_normal((SR // 2, 2)), role="pad")

    blended = edit.apply_paulstretch(fr, factor=8.0)
    decorrelated = edit.apply_paulstretch(fr, factor=8.0, stereo_decorrelate=True)

    assert blended["params"]["stereo_decorrelate"] is False
    assert decorrelated["params"]["stereo_decorrelate"] is True

    b, _ = _load(blended)
    d, _ = _load(decorrelated)
    assert np.corrcoef(b[:, 0], b[:, 1])[0, 1] == pytest.approx(0.6, abs=0.05)
    assert abs(np.corrcoef(d[:, 0], d[:, 1])[0, 1]) < 0.1


def test_bad_factor_and_window():
    fr, _ = _tone_frame(dur=0.2)
    with pytest.raises(ValueError):
        edit.apply_paulstretch(fr, factor=0.0)
    with pytest.raises(ValueError):
        edit.apply_paulstretch(fr, factor=8.0, window_s=0.0)
