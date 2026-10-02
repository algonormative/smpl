"""Memo-cache wiring for the cacheable analysis ops (loudness / spectral / qc / spectrogram).

The proof that a warm run *skips compute* is structural, not wall-clock: each test breaks
the op's compute path (monkeypatched to raise) and then asserts the warm run still returns
the same frames. If the lookup were not wired, the run would blow up.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from smplstream import cas, frames as F, memostore

from smpl_analysis import loudness as L
from smpl_analysis import qc as QC
from smpl_analysis import spectral as SP
from smpl_analysis import spectrogram as IMG


def _tone(sr: int = 22050, dur: float = 0.5, freq: float = 220.0) -> np.ndarray:
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture()
def audio(tmp_path, monkeypatch):
    """An `audio` frame in an isolated CAS (which is also where the memo index lives)."""
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    wav = tmp_path / "tone.wav"
    sf.write(str(wav), _tone(), 22050, subtype="FLOAT")
    h = cas.put_audio_file(str(wav))
    meta = cas.read_meta(h) or {}
    return F.audio_frame(h, sr=meta["sr"], ch=meta["ch"], dur=meta["dur"], role="source")


def _boom(*a, **kw):
    raise AssertionError("compute ran on what should have been a cache hit")


def _feature(frames):
    return next(f for f in frames if f["kind"] == "feature")


# --------------------------------------------------------------------------- loudness


def test_loudness_cold_miss_then_warm_hit(audio, monkeypatch):
    cold = _feature(L.loudness_frames(audio))
    assert cold["params"]["cache_hit"] is False

    monkeypatch.setattr(L, "analyze_path", _boom)  # compute must not run on the warm pass
    warm = _feature(L.loudness_frames(audio))
    assert warm["params"]["cache_hit"] is True
    assert warm["data"] == cold["data"]
    assert warm["role"] == cold["role"] and warm["of"] == cold["of"]
    assert warm["op"] == cold["op"] and warm["op_version"] == cold["op_version"]


def test_loudness_markers_survive_a_cache_hit(audio, monkeypatch):
    """Over-points ride in the cached payload, so `--markers` output is identical warm."""
    cold = L.loudness_frames(audio, over_ceiling_dbtp=-30.0)
    cold_markers = [f for f in cold if f["kind"] == "marker"]
    assert cold_markers, "fixture should breach a -30 dBTP ceiling"

    monkeypatch.setattr(L, "analyze_path", _boom)
    warm_markers = [f for f in L.loudness_frames(audio, over_ceiling_dbtp=-30.0)
                    if f["kind"] == "marker"]
    assert warm_markers[0]["data"] == cold_markers[0]["data"]


def test_loudness_param_change_is_a_miss(audio):
    L.loudness_frames(audio)
    other = _feature(L.loudness_frames(audio, over_ceiling_dbtp=-3.0))
    assert other["params"]["cache_hit"] is False  # different params → different memo key


def test_loudness_no_cache_forces_recompute(audio, monkeypatch):
    cold = _feature(L.loudness_frames(audio))
    assert cold["params"]["cache_hit"] is False

    calls = []
    real = L.analyze_path
    monkeypatch.setattr(L, "analyze_path", lambda *a, **kw: (calls.append(1), real(*a, **kw))[1])

    bypass = _feature(L.loudness_frames(audio, use_cache=False))
    assert calls == [1]                              # recomputed despite a warm entry
    assert bypass["params"]["cache_hit"] is False
    assert bypass["data"] == cold["data"]

    # and the forced run refreshed the entry, so the next default run still hits
    monkeypatch.setattr(L, "analyze_path", _boom)
    assert _feature(L.loudness_frames(audio))["params"]["cache_hit"] is True


# --------------------------------------------------------------------------- spectral


def test_spectral_cold_miss_then_warm_hit(audio, monkeypatch):
    cold = _feature(SP.spectral_audio_frame(audio))
    assert cold["params"]["cache_hit"] is False

    monkeypatch.setattr(SP, "spectral_shape", _boom)
    warm = _feature(SP.spectral_audio_frame(audio))
    assert warm["params"]["cache_hit"] is True
    assert warm["data"] == cold["data"]


def test_spectral_param_change_is_a_miss(audio):
    SP.spectral_audio_frame(audio)
    other = _feature(SP.spectral_audio_frame(audio, n_fft=1024))
    assert other["params"]["cache_hit"] is False


def test_spectral_no_cache_forces_recompute(audio, monkeypatch):
    SP.spectral_audio_frame(audio)
    monkeypatch.setattr(SP, "spectral_shape", _boom)
    with pytest.raises(AssertionError):
        SP.spectral_audio_frame(audio, use_cache=False)  # bypass really recomputes


# --------------------------------------------------------------------------- qc


def test_qc_cold_miss_then_warm_hit(audio, monkeypatch):
    cold_frames = QC.qc_audio_frame(audio)
    cold = _feature(cold_frames)
    assert cold["params"]["cache_hit"] is False

    monkeypatch.setattr(QC, "_load", _boom)  # not even the decode may run
    warm_frames = QC.qc_audio_frame(audio)
    warm = _feature(warm_frames)
    assert warm["params"]["cache_hit"] is True
    assert warm["data"] == cold["data"]
    assert warm["params"]["sr"] == cold["params"]["sr"]
    assert warm["params"]["ch"] == cold["params"]["ch"]
    assert [f["kind"] for f in warm_frames] == [f["kind"] for f in cold_frames]


def test_qc_markers_flag_shares_one_entry(audio, monkeypatch):
    """`want_markers` gates emission only — it is not part of the key."""
    QC.qc_audio_frame(audio, want_markers=False)
    monkeypatch.setattr(QC, "_load", _boom)
    warm = QC.qc_audio_frame(audio, want_markers=True)
    assert _feature(warm)["params"]["cache_hit"] is True


def test_qc_no_cache_forces_recompute(audio, monkeypatch):
    QC.qc_audio_frame(audio)
    monkeypatch.setattr(QC, "_load", _boom)
    with pytest.raises(AssertionError):
        QC.qc_audio_frame(audio, use_cache=False)


# --------------------------------------------------------------------------- spectrogram


def test_spectrogram_cold_miss_then_warm_hit(audio, monkeypatch):
    cold = IMG.render_audio_frame(audio)[0]
    assert cold["params"]["cache_hit"] is False

    monkeypatch.setattr(IMG, "render_array", _boom)
    monkeypatch.setattr(IMG, "_load_mono", _boom)  # a full hit never decodes the audio
    warm = IMG.render_audio_frame(audio)[0]
    assert warm["params"]["cache_hit"] is True
    assert warm["hash"] == cold["hash"]            # same PNG blob, not a re-render
    assert warm["role"] == cold["role"] and warm["meta"] == cold["meta"]


def test_spectrogram_memoizes_each_kind_separately(audio, monkeypatch):
    IMG.render_audio_frame(audio, kinds=["mel"])

    calls = []
    real = IMG.render_array
    monkeypatch.setattr(
        IMG, "render_array", lambda y, sr, kind: (calls.append(kind), real(y, sr, kind))[1]
    )
    out = IMG.render_audio_frame(audio, kinds=["mel", "waveform"])
    assert calls == ["waveform"]                    # mel hit; only the new kind rendered
    assert [f["params"]["cache_hit"] for f in out] == [True, False]


def test_spectrogram_no_cache_forces_rerender(audio, monkeypatch):
    IMG.render_audio_frame(audio)
    monkeypatch.setattr(IMG, "render_array", _boom)
    with pytest.raises(AssertionError):
        IMG.render_audio_frame(audio, use_cache=False)


# --------------------------------------------------------------------------- store wiring


def test_ops_record_entries_under_their_own_keys(audio):
    L.loudness_frames(audio)
    SP.spectral_audio_frame(audio)
    QC.qc_audio_frame(audio)
    IMG.render_audio_frame(audio)

    entries = [f for f in memostore.memo_dir().rglob("*.json")]
    ops = sorted(json.loads(p.read_text())["op"] for p in entries)
    assert ops == ["loudness", "qc", "spectral", "spectrogram"]


def test_a_different_input_is_a_miss(audio, tmp_path):
    L.loudness_frames(audio)
    other_wav = tmp_path / "other.wav"
    sf.write(str(other_wav), _tone(freq=440.0), 22050, subtype="FLOAT")
    h = cas.put_audio_file(str(other_wav))
    meta = cas.read_meta(h) or {}
    other = F.audio_frame(h, sr=meta["sr"], ch=meta["ch"], dur=meta["dur"], role="source")

    assert _feature(L.loudness_frames(other))["params"]["cache_hit"] is False
