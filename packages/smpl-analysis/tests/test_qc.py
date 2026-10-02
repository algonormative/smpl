"""Tests for smpl_analysis.qc — deterministic QC top-6 (ticket vault-1e9a).

Each test synthesizes a signal with a KNOWN defect and asserts the corresponding QC measure
fires (and that clean signals don't). The end-to-end test exercises the frame-emitting path
through the CAS, asserting registry-correct keys and lineage.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from smpl_analysis import qc


SR = 44100


def _tone(freq=440.0, dur=1.0, amp=0.5, sr=SR, channels=1):
    t = np.arange(int(dur * sr)) / sr
    sig = amp * np.sin(2 * np.pi * freq * t)
    if channels == 1:
        return sig.reshape(-1, 1).astype(np.float32)
    return np.column_stack([sig] * channels).astype(np.float32)


# --- 1. clipping / true-peak ------------------------------------------------------------


def test_true_peak_clean_below_threshold():
    s = _tone(amp=0.5)
    tp = qc.true_peak_dbtp(s, SR)
    assert tp < qc._CLIP_DBTP_THRESHOLD  # −0.5 amp ≈ −6 dBTP, well clear


def test_true_peak_clipped_full_scale():
    s = _tone(amp=1.0)
    # hard-clip to full scale so inter-sample peaks exceed 0 dBTP
    s = np.clip(s * 4.0, -1.0, 1.0).astype(np.float32)
    tp = qc.true_peak_dbtp(s, SR)
    assert tp >= qc._CLIP_DBTP_THRESHOLD


# --- 2. phase correlation ---------------------------------------------------------------


def test_phase_correlation_mono_is_none():
    assert qc.phase_correlation(_tone(channels=1)) is None


def test_phase_correlation_in_phase_near_plus_one():
    s = _tone(channels=2)  # identical L/R
    corr = qc.phase_correlation(s)
    assert corr is not None and corr > 0.99


def test_phase_correlation_anti_phase_near_minus_one():
    s = _tone(channels=2)
    s[:, 1] *= -1.0  # invert right channel
    corr = qc.phase_correlation(s)
    assert corr is not None and corr < -0.99


# --- 3. DC offset -----------------------------------------------------------------------


def test_dc_offset_clean_is_very_low():
    s = _tone()
    assert qc.dc_offset_dbfs(s) < -60.0


def test_dc_offset_detects_bias():
    s = _tone(amp=0.3) + 0.1  # +0.1 DC bias (≈ −20 dBFS)
    s = s.astype(np.float32)
    dc = qc.dc_offset_dbfs(s)
    assert -25.0 < dc < -15.0


# --- 4. SNR -----------------------------------------------------------------------------


def _tone_with_silence(amp=0.5, tone_s=0.5, sil_s=0.3, sr=SR, noise=0.0):
    """A tone body bracketed by (optionally noisy) silence — the regime SNR is built for."""
    rng = np.random.default_rng(7)
    body = _tone(amp=amp, dur=tone_s, sr=sr)[:, 0]
    sil = np.zeros(int(sil_s * sr), dtype=np.float32)
    sig = np.concatenate([sil, body, sil]).astype(np.float64)
    if noise > 0:
        sig = sig + rng.normal(0, noise, sig.shape[0])
    return sig.reshape(-1, 1).astype(np.float32)


def test_snr_clean_signal_with_silence_is_high():
    # body well above a near-silent floor → wide dynamic range
    snr = qc.snr_db(_tone_with_silence(amp=0.5, noise=0.0), SR)
    assert snr is not None and snr > 30.0


def test_snr_noisy_floor_is_lower_than_clean():
    snr_clean = qc.snr_db(_tone_with_silence(amp=0.5, noise=1e-5), SR)
    snr_noisy = qc.snr_db(_tone_with_silence(amp=0.5, noise=0.02), SR)
    assert snr_noisy is not None and snr_clean is not None
    # raising the floor noise narrows the loud-to-quiet gap
    assert snr_noisy < snr_clean


# --- 5. clicks / gaps -------------------------------------------------------------------


def test_detect_clicks_finds_injected_spike():
    s = _tone(amp=0.3).copy()
    pos = SR // 2
    s[pos, 0] = 0.95  # a sharp discontinuity
    points = qc.detect_clicks(s, SR)
    assert points, "expected at least one click marker"
    # the click should land near the injected sample
    assert any(abs(p["sample"] - pos) <= 2 for p in points)
    assert all(p["label"] == "click" for p in points)


def test_detect_clicks_clean_tone_none():
    assert qc.detect_clicks(_tone(amp=0.3), SR) == []


def test_detect_gaps_finds_internal_silence():
    s = _tone(amp=0.4).copy()
    g0, g1 = SR // 3, SR // 3 + int(SR * 0.05)  # 50 ms internal gap
    s[g0:g1, 0] = 0.0
    points = qc.detect_gaps(s, SR)
    assert points, "expected an internal gap marker"
    assert any(abs(p["sample"] - g0) <= 2 for p in points)
    assert all(p["label"] == "gap" for p in points)


def test_detect_gaps_ignores_trailing_silence():
    s = _tone(amp=0.4, dur=0.5).copy()
    s = np.vstack([s, np.zeros((int(SR * 0.2), 1), dtype=np.float32)])  # trailing silence only
    assert qc.detect_gaps(s, SR) == []


# --- 6. lossy origin --------------------------------------------------------------------


def test_lossy_full_band_low_confidence():
    rng = np.random.default_rng(1)
    s = rng.normal(0, 0.2, (SR, 1)).astype(np.float32)  # white noise → full-band
    res = qc.lossy_origin(s, SR)
    assert res["qc.lossy.expected_nyquist_hz"] == pytest.approx(SR / 2.0, abs=1.0)
    assert res["qc.lossy.confidence"] < 0.3


def test_lossy_brickwalled_flags_cutoff():
    # white noise low-passed hard at ~16 kHz, emulating a 128 kbps LAME brickwall
    rng = np.random.default_rng(2)
    from scipy.signal import butter, sosfiltfilt

    noise = rng.normal(0, 0.3, SR).astype(np.float64)
    sos = butter(12, 16000 / (SR / 2), btype="low", output="sos")
    band = sosfiltfilt(sos, noise).reshape(-1, 1).astype(np.float32)
    res = qc.lossy_origin(band, SR)
    assert res["qc.lossy.spectral_cutoff_hz"] < 18000.0
    assert res["qc.lossy.confidence"] > 0.3


def test_lossy_sub_only_kick_not_flagged():
    # A genuinely sub-only synth kick body: energy all below ~140 Hz, dead above. The old
    # detector scored this 1.0 (absence-of-highs read as a codec brickwall). It must not
    # flag — the knee is implausibly low for any codec cutoff (vault-3t1l).
    rng = np.random.default_rng(3)
    from scipy.signal import butter, sosfiltfilt

    noise = rng.normal(0, 0.3, SR).astype(np.float64)
    sos = butter(12, 140 / (SR / 2), btype="low", output="sos")
    sub = sosfiltfilt(sos, noise).reshape(-1, 1).astype(np.float32)
    res = qc.lossy_origin(sub, SR)
    assert res["qc.lossy.confidence"] < 0.3


def test_lossy_pure_sine_not_flagged():
    # A pure 50 Hz sine — the pathological pure-sub case (mirrors the refmatch centroid
    # finding). Absence of highs is not shelf-shape evidence; must not flag (vault-3t1l).
    t = np.arange(SR, dtype=np.float64) / SR
    sine = (0.5 * np.sin(2 * np.pi * 50 * t)).reshape(-1, 1).astype(np.float32)
    res = qc.lossy_origin(sine, SR)
    assert res["qc.lossy.confidence"] < 0.3


# --- 6b. lossy origin on steep-HF-decay content (vault-3fb7b) ---------------------------
# The 2026-08-07 probe: a 64 kbps mp3 roundtrip of a synthetic techno loop located its
# brickwall correctly but scored confidence 0.053. The loop's natural HF decay is steep
# (most of its energy lives under 1 kHz), so the 0.999-cumulative-energy knee lands over a
# kilohertz BELOW the encoder ceiling — and the slope/shelf evidence, gathered at the knee,
# straddled live content and read as a gentle taper. The fixtures below reproduce that
# spectral shape without needing an encoder: an FFT-zeroing brickwall is a harder ceiling
# than any codec's, so anything a codec does is a weaker case of the same signature.


def _techno_loop(dur=2.0, sr=SR, seed=11):
    """Techno-loop-like signal: kick + saw bass + hats. Full band, steep natural HF decay."""
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    sig = np.zeros(n, dtype=np.float64)

    def add(i0, seg):
        end = min(n, i0 + seg.shape[0])
        if end > i0:
            sig[i0:end] += seg[: end - i0]

    beat = 60.0 / 130.0  # 130 BPM
    for k in range(int(dur / beat)):  # kick: pitch-swept sine body on every beat
        tt = np.arange(int(0.25 * sr)) / sr
        f = 120.0 * np.exp(-tt * 40) + 45.0
        add(int(k * beat * sr), 0.9 * np.exp(-tt * 28) * np.sin(2 * np.pi * np.cumsum(f) / sr))
    for k in range(int(dur / (beat / 2))):  # saw bass on offbeat eighths
        tt = np.arange(int(0.18 * sr)) / sr
        f0 = 55.0 * (2 ** ((k % 3) / 12.0))
        add(int(k * (beat / 2) * sr), 0.25 * np.exp(-tt * 12) * (2 * ((f0 * tt) % 1.0) - 1.0))
    for k in range(int(dur / (beat / 4))):  # closed hats on 16ths (HF-tilted noise bursts)
        ln = int(0.03 * sr)
        tt = np.arange(ln) / sr
        nz = np.diff(np.concatenate([[0.0], rng.normal(0, 1.0, ln)]))
        add(int(k * (beat / 4) * sr), 0.06 * np.exp(-tt * 220) * nz)

    return sig / (np.max(np.abs(sig)) + 1e-12) * 0.8


def _brickwall(sig, fc, sr=SR):
    """Hard FFT-zeroing low-pass at `fc` — a ceiling at least as hard as any encoder's."""
    spec = np.fft.rfft(sig)
    spec[np.fft.rfftfreq(sig.shape[0], d=1.0 / sr) >= fc] = 0.0
    return np.fft.irfft(spec, n=sig.shape[0])


def _as_frames(sig):
    return np.asarray(sig, dtype=np.float64).reshape(-1, 1).astype(np.float32)


def test_lossy_synthetic_brickwall_high_confidence():
    # THE REGRESSION (vault-3fb7b): a hard 12 kHz ceiling on steep-HF-decay synthetic
    # content scored 0.15 when the evidence was gathered at the energy knee. It is an
    # unambiguous brickwall and must score high.
    res = qc.lossy_origin(_as_frames(_brickwall(_techno_loop(), 12000.0)), SR)
    assert res["qc.lossy.confidence"] > 0.5
    # and the reported cutoff must be the WALL, not the energy knee a kilohertz below it
    assert res["qc.lossy.spectral_cutoff_hz"] == pytest.approx(12000.0, abs=100.0)


def test_lossy_synthetic_brickwall_beats_energy_knee():
    """Pin the root cause: the 0.999-energy knee is NOT the wall on this content."""
    sig = _brickwall(_techno_loop(), 12000.0)
    mono = _as_frames(sig)[:, 0].astype(np.float64)
    nfft, hop = 4096, 2048
    win = np.hanning(nfft)
    n_frames = 1 + (mono.shape[0] - nfft) // hop
    acc = np.zeros(nfft // 2 + 1)
    for i in range(n_frames):
        acc += np.abs(np.fft.rfft(mono[i * hop : i * hop + nfft] * win))
    power = (acc / n_frames) ** 2
    freqs = np.fft.rfftfreq(nfft, d=1.0 / SR)
    knee = float(freqs[int(np.searchsorted(np.cumsum(power) / power.sum(), qc._LOSSY_ENERGY_FRAC))])
    # the knee undershoots the 12 kHz wall by a kilohertz — that is the whole bug
    assert knee < 11500.0
    # ...and the edge search recovers the wall from the same spectrum
    assert qc.lossy_origin(_as_frames(sig), SR)["qc.lossy.spectral_cutoff_hz"] > knee + 500.0


def test_lossy_synthetic_full_band_not_flagged():
    # The SAME loop without the brickwall must stay low — otherwise the fixture proves
    # nothing about brickwalls and only that this content is dark.
    res = qc.lossy_origin(_as_frames(_techno_loop()), SR)
    assert res["qc.lossy.confidence"] < 0.3


def test_lossy_natural_full_band_brickwall_high_confidence():
    # The natural-content half of the acceptance: broadband material with a hard 16 kHz
    # ceiling (a 128 kbps LAME-shaped wall) also scores high.
    rng = np.random.default_rng(9)
    body = rng.normal(0, 0.25, int(1.5 * SR))
    res = qc.lossy_origin(_as_frames(_brickwall(body, 16000.0)), SR)
    assert res["qc.lossy.confidence"] > 0.5
    assert res["qc.lossy.spectral_cutoff_hz"] == pytest.approx(16000.0, abs=100.0)


def test_lossy_gentle_dark_rolloff_not_flagged():
    # A naturally dark source with a GENTLE taper: band-limited-looking but no wall. The
    # edge scan must not manufacture one out of the spectrum's own roll-off (nor out of the
    # collapse every digital low-pass shows approaching Nyquist).
    from scipy.signal import butter, sosfiltfilt

    rng = np.random.default_rng(4)
    sos = butter(1, 3000 / (SR / 2), btype="low", output="sos")
    pad = sosfiltfilt(sos, rng.normal(0, 0.3, int(1.5 * SR)))
    assert qc.lossy_origin(_as_frames(pad), SR)["qc.lossy.confidence"] < 0.3


def test_lossy_confidence_tracks_ceiling_hardness():
    # Confidence is evidence of a WALL, so it must rise with how wall-like the band limit
    # is: a gentle taper at 12 kHz < a steep one < a hard FFT ceiling at the same corner.
    from scipy.signal import butter, sosfiltfilt

    base = _techno_loop()
    confs = []
    for order in (1, 3):
        sos = butter(order, 12000 / (SR / 2), btype="low", output="sos")
        confs.append(qc.lossy_origin(_as_frames(sosfiltfilt(sos, base)), SR)["qc.lossy.confidence"])
    hard = qc.lossy_origin(_as_frames(_brickwall(base, 12000.0)), SR)["qc.lossy.confidence"]
    assert confs[0] < confs[1] <= hard
    assert confs[0] < 0.5 < hard


# --- end-to-end: frame emission through the CAS -----------------------------------------


def _put_wav(samples, sr):
    """CAS a numpy buffer as a WAV and return its audio frame."""
    from smplstream import cas, frames as F

    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="FLOAT")
    h = cas.put_audio_bytes(buf.getvalue())
    meta = cas.read_meta(h) or {}
    return F.audio_frame(
        h,
        sr=meta.get("sr", sr),
        ch=meta.get("ch", samples.shape[1]),
        dur=meta.get("dur", samples.shape[0] / sr),
        role="source",
    )


def test_qc_audio_frame_emits_feature_and_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    # stereo tone with an injected click so we also get a marker frame
    s = _tone(amp=0.4, channels=2).copy()
    s[SR // 2, 0] = 0.95
    af = _put_wav(s, SR)

    derived = qc.qc_audio_frame(af, want_markers=True)
    features = [f for f in derived if f["kind"] == "feature"]
    markers = [f for f in derived if f["kind"] == "marker"]

    assert len(features) == 1
    data = features[0]["data"]
    # registry keys present
    for key in (
        "qc.phase.correlation",
        "qc.dc_offset_dbfs",
        "qc.snr_db",
        "qc.lossy.spectral_cutoff_hz",
        "qc.lossy.expected_nyquist_hz",
        "qc.lossy.confidence",
    ):
        assert key in data, f"missing registry key {key}"
    # clipping flag + context true-peak
    assert "qc.clipping.detected" in data
    assert data["qc.clipping.detected"] is False  # −0.4 amp tone is clean
    # the loudness-owned KEY must NOT be emitted here (ownership boundary)
    assert "loudness.true_peak_dbtp" not in data

    # lineage / op metadata on every derived frame
    for f in derived:
        assert f.get("of") == af["id"]
        assert f.get("op") == "qc"
        assert f.get("op_version") == "qc@1"

    # the injected click yields a defect marker frame
    assert markers, "expected a marker frame for the injected click"
    assert any(p.get("label") == "click" for m in markers for p in m["data"])


def test_qc_audio_frame_no_markers(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    af = _put_wav(_tone(amp=0.4), SR)
    derived = qc.qc_audio_frame(af, want_markers=False)
    assert all(f["kind"] == "feature" for f in derived)
