"""Tests for the reference-match edit ops: widen + spectral-match (smpl_analysis.edit).

widen: side-channel gain above a crossover with a mono-safe low end. spectral-match: a bounded
corrective EQ that moves a source's per-band balance toward a reference's, shape-only (level
untouched), with the sub protected.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from smpl_analysis import edit

SR = 44100


def _put_wav(samples, sr=SR, role="source"):
    from smplstream import cas, frames as F

    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="FLOAT")
    h = cas.put_audio_bytes(buf.getvalue())
    meta = cas.read_meta(h) or {}
    return F.audio_frame(h, sr=meta.get("sr", sr), ch=meta.get("ch", samples.shape[1]),
                         dur=meta.get("dur", samples.shape[0] / sr), role=role)


def _load(frame):
    from smplstream import cas

    data, sr = sf.read(str(cas.get_path(frame["hash"])), dtype="float64", always_2d=True)
    return data, sr


def _side_band_energy(samples, sr, lo, hi, trim_s=0.05):
    """Time-domain RMS of the side channel (L−R)/2 within a band, excluding ``trim_s`` of
    each edge (zero-phase filter edge transients are confined there and would otherwise
    swamp a near-silent band — see the widen edge-transient analysis)."""
    from scipy.signal import butter, sosfiltfilt

    side = (samples[:, 0] - samples[:, 1]) / 2.0
    nyq = sr / 2.0
    sos = butter(4, [max(lo / nyq, 1e-6), min(hi / nyq, 0.999999)],
                 btype="bandpass", output="sos")
    band = sosfiltfilt(sos, side)
    n = int(trim_s * sr)
    core = band[n:-n] if len(band) > 2 * n else band
    return float(np.sqrt(np.mean(core ** 2)))


def _stereo_two_band(low=80.0, high=6000.0, dur=1.0, amp=0.3, sr=SR):
    """Stereo signal where BOTH bands carry real side content: the low tone is partly panned
    (a non-zero, meaningful low-side baseline to compare against) and the high tone is fully
    anti-phase (lives entirely in the side). Lets us prove the crossover preserves the low
    side while boosting the high side — without a degenerate near-zero baseline."""
    t = np.arange(int(dur * sr)) / sr
    lo = amp * np.sin(2 * np.pi * low * t)
    hi = amp * np.sin(2 * np.pi * high * t)
    left = lo + hi
    right = 0.5 * lo - hi  # low tone partly in the side; high tone fully anti-phase
    return np.column_stack([left, right]).astype(np.float32)


# --- widen --------------------------------------------------------------------------------


def test_widen_boosts_high_side_keeps_low_mono(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    src = _put_wav(_stereo_two_band(), SR)
    wet = edit.apply_widen(src, side_gain_db=6.0, crossover_hz=500.0)

    assert wet["role"] == "source.wet"
    assert wet.get("op") == "widen" and wet.get("op_version") == edit.WIDEN_OP_VERSION
    assert wet.get("of") == src["id"] and wet.get("lineage") == [src["id"]]
    assert wet["hash"] != src["hash"]

    before, sr = _load(src)
    after, _ = _load(wet)
    # The high (anti-phase) side energy should rise ~+6 dB (×~2); the low band stays mono (~0 side).
    hi_before = _side_band_energy(before, sr, 4000, 8000)
    hi_after = _side_band_energy(after, sr, 4000, 8000)
    assert hi_after > 1.6 * hi_before  # ~+6 dB widen on the high band
    lo_before = _side_band_energy(before, sr, 40, 200)
    lo_after = _side_band_energy(after, sr, 40, 200)
    assert abs(lo_after - lo_before) < 0.12 * lo_before  # low end side preserved (mono-safe)


def test_widen_negative_narrows(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    src = _put_wav(_stereo_two_band(), SR)
    wet = edit.apply_widen(src, side_gain_db=-12.0, crossover_hz=500.0)
    before, sr = _load(src)
    after, _ = _load(wet)
    assert _side_band_energy(after, sr, 5000, 7000) < 0.5 * _side_band_energy(before, sr, 5000, 7000)


def test_widen_mono_passthrough(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    t = np.arange(SR) / SR
    mono = (0.3 * np.sin(2 * np.pi * 440 * t)).reshape(-1, 1).astype(np.float32)
    src = _put_wav(mono, SR)
    wet = edit.apply_widen(src, side_gain_db=6.0)
    assert "note" in wet["params"]
    before, _ = _load(src)
    after, _ = _load(wet)
    assert np.allclose(before, after, atol=1e-6)  # mono untouched


# --- spectral-match -----------------------------------------------------------------------


def _band_db(mono, sr, lo, hi):
    spec = np.abs(np.fft.rfft(mono * np.hanning(len(mono)))) ** 2
    freqs = np.fft.rfftfreq(len(mono), 1.0 / sr)
    mask = (freqs >= lo) & (freqs < hi)
    return 10.0 * np.log10(max(float(np.sum(spec[mask])), 1e-20))


def _noise_shaped(sr=SR, dur=2.0, hp=None, tilt=0.0, seed=0):
    """White-ish noise, optionally tilted (tilt>0 = brighter) via a simple high-shelf-ish ramp."""
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    x = rng.standard_normal(n)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    if tilt:
        X = X * (1.0 + tilt * (f / (sr / 2)))  # ramp emphasis toward Nyquist
    y = np.fft.irfft(X, n=n)
    y = 0.2 * y / (np.max(np.abs(y)) + 1e-9)
    return y.reshape(-1, 1).astype(np.float32)


def test_spectral_match_moves_balance_toward_reference(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    # source: dull (no tilt). reference: bright (high-frequency emphasis).
    src = _put_wav(_noise_shaped(tilt=0.0, seed=1), SR)
    ref_path = tmp_path / "ref.wav"
    sf.write(str(ref_path), _noise_shaped(tilt=6.0, seed=2), SR, format="WAV", subtype="FLOAT")

    wet = edit.apply_spectral_match(src, reference_path=str(ref_path), strength=1.0, max_correction_db=12.0)
    assert wet.get("op") == "spectral-match"
    assert wet.get("op_version") == edit.SPECTRAL_MATCH_OP_VERSION
    assert wet.get("of") == src["id"]
    assert len(wet["params"]["correction_db"]) == wet["params"]["n_bands"]

    before, sr = _load(src)
    after, _ = _load(wet)
    bmono, amono = before.mean(axis=1), after.mean(axis=1)
    # high/low balance should shift UP toward the bright reference
    bal_before = _band_db(bmono, sr, 4000, 12000) - _band_db(bmono, sr, 100, 500)
    bal_after = _band_db(amono, sr, 4000, 12000) - _band_db(amono, sr, 100, 500)
    assert bal_after > bal_before + 1.0  # got measurably brighter


def test_spectral_match_clamps_correction(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    src = _put_wav(_noise_shaped(tilt=0.0, seed=3), SR)
    ref_path = tmp_path / "ref.wav"
    sf.write(str(ref_path), _noise_shaped(tilt=20.0, seed=4), SR, format="WAV", subtype="FLOAT")
    wet = edit.apply_spectral_match(src, reference_path=str(ref_path), strength=1.0, max_correction_db=3.0)
    corr = wet["params"]["correction_db"]
    assert all(abs(c) <= 3.0 + 1e-6 for c in corr)  # every band within the clamp


def test_spectral_match_protects_sub(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    src = _put_wav(_noise_shaped(tilt=0.0, seed=5), SR)
    ref_path = tmp_path / "ref.wav"
    sf.write(str(ref_path), _noise_shaped(tilt=10.0, seed=6), SR, format="WAV", subtype="FLOAT")
    wet = edit.apply_spectral_match(src, reference_path=str(ref_path), protect_below_hz=80.0)
    centers = wet["params"]["band_centers_hz"]
    corr = wet["params"]["correction_db"]
    for fc, c in zip(centers, corr):
        if fc < 80.0:
            assert c == 0.0  # sub protected


def test_spectral_match_cross_sample_rate(tmp_path, monkeypatch):
    """A reference at a LOWER sample rate than the source must not corrupt the match: the band
    grid clamps below the reference Nyquist (no empty -inf bands polluting the mean)."""
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    src = _put_wav(_noise_shaped(tilt=0.0, seed=11, sr=SR), SR)            # 44.1k, dull
    ref_path = tmp_path / "ref16k.wav"
    sf.write(str(ref_path), _noise_shaped(tilt=8.0, seed=12, sr=16000), 16000,
             format="WAV", subtype="FLOAT")                                # 16k ref (Nyquist 8k), bright
    wet = edit.apply_spectral_match(src, reference_path=str(ref_path), strength=1.0, max_correction_db=12.0)
    # the grid clamped below the ref Nyquist (8000) — proves the cross-SR guard engaged
    assert wet["params"]["hi_hz_effective"] <= 8000.0
    assert wet["params"]["ref_sr_hz"] == 16000
    # corrections are finite, varied (NOT a uniform garbage offset), and within the clamp
    corr = wet["params"]["correction_db"]
    assert all(c == c and abs(c) <= 12.0 for c in corr)  # c==c rejects NaN
    assert max(corr) - min(corr) > 1.0                   # a real curve, not a flat offset
    # and the match actually brightened the source within the shared band range
    before, sr = _load(src)
    after, _ = _load(wet)
    bal_b = _band_db(before.mean(axis=1), sr, 3000, 7000) - _band_db(before.mean(axis=1), sr, 100, 500)
    bal_a = _band_db(after.mean(axis=1), sr, 3000, 7000) - _band_db(after.mean(axis=1), sr, 100, 500)
    assert bal_a > bal_b


def test_spectral_match_strength_scales(tmp_path, monkeypatch):
    """strength=0 → no correction; strength=0.5 ≈ half of strength=1.0 per band."""
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    src = _put_wav(_noise_shaped(tilt=0.0, seed=13), SR)
    ref_path = tmp_path / "ref.wav"
    sf.write(str(ref_path), _noise_shaped(tilt=8.0, seed=14), SR, format="WAV", subtype="FLOAT")
    z = edit.apply_spectral_match(src, reference_path=str(ref_path), strength=0.0, max_correction_db=24.0)
    assert all(c == 0.0 for c in z["params"]["correction_db"])
    half = edit.apply_spectral_match(src, reference_path=str(ref_path), strength=0.5, max_correction_db=24.0)
    full = edit.apply_spectral_match(src, reference_path=str(ref_path), strength=1.0, max_correction_db=24.0)
    # away from the clamp, half-strength corrections are ~half the full-strength ones
    for h, f in zip(half["params"]["correction_db"], full["params"]["correction_db"]):
        if abs(f) > 0.1 and abs(f) < 23.0:
            assert abs(h - 0.5 * f) < 0.2


def test_spectral_match_rejects_bad_params(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    src = _put_wav(_noise_shaped(seed=15), SR)
    ref_path = tmp_path / "ref.wav"
    sf.write(str(ref_path), _noise_shaped(tilt=4.0, seed=16), SR, format="WAV", subtype="FLOAT")
    with pytest.raises(ValueError):
        edit.apply_spectral_match(src, reference_path=str(ref_path), max_correction_db=-3.0)
    with pytest.raises(ValueError):
        edit.apply_spectral_match(src, reference_path=str(ref_path), n_bands=0)
    with pytest.raises(ValueError):  # anchor below the grid's low edge — no band contains it
        edit.apply_spectral_match(src, reference_path=str(ref_path), lo_hz=30.0, anchor_hz=10.0)
    with pytest.raises(ValueError):  # anchor above the grid's effective high edge
        edit.apply_spectral_match(src, reference_path=str(ref_path), hi_hz=16000.0, anchor_hz=20000.0)


def test_spectral_match_records_ab_spectra(tmp_path, monkeypatch):
    """The wet frame carries the A/B block: source (before), reference, matched (after) — all
    on the same band grid and all mean-normalized, so the three curves are comparable."""
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    src = _put_wav(_noise_shaped(tilt=0.0, seed=21), SR)
    ref_path = tmp_path / "ref.wav"
    sf.write(str(ref_path), _noise_shaped(tilt=6.0, seed=22), SR, format="WAV", subtype="FLOAT")

    wet = edit.apply_spectral_match(src, reference_path=str(ref_path), strength=1.0,
                                    max_correction_db=12.0)
    spec = wet["params"]["spectrum_db"]
    assert set(spec) == {"source", "reference", "matched"}
    n = wet["params"]["n_bands"]
    for key, curve in spec.items():
        assert len(curve) == n, key
        assert abs(float(np.mean(curve))) < 1e-6, key   # mean-normalized: balance, not level
    assert set(wet["params"]["residual_db"]) == {"before", "after"}


def test_spectral_match_residual_improves(tmp_path, monkeypatch):
    """A/B proof: the matched target sits CLOSER to the reference than the source did."""
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    # same dull-source / bright-reference fixture as the balance test above
    src = _put_wav(_noise_shaped(tilt=0.0, seed=1), SR)
    ref_path = tmp_path / "ref.wav"
    sf.write(str(ref_path), _noise_shaped(tilt=6.0, seed=2), SR, format="WAV", subtype="FLOAT")

    wet = edit.apply_spectral_match(src, reference_path=str(ref_path), strength=1.0,
                                    max_correction_db=12.0)
    residual = wet["params"]["residual_db"]
    assert residual["after"] < residual["before"]


def test_spectral_match_zero_strength_ab_is_flat(tmp_path, monkeypatch):
    """strength=0 applies no EQ, so the after-curve IS the before-curve and the residual to the
    reference is unchanged — the A/B block can't manufacture an improvement."""
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    src = _put_wav(_noise_shaped(tilt=0.0, seed=23), SR)
    ref_path = tmp_path / "ref.wav"
    sf.write(str(ref_path), _noise_shaped(tilt=8.0, seed=24), SR, format="WAV", subtype="FLOAT")

    wet = edit.apply_spectral_match(src, reference_path=str(ref_path), strength=0.0,
                                    max_correction_db=24.0)
    spec = wet["params"]["spectrum_db"]
    for before, after in zip(spec["source"], spec["matched"]):
        assert abs(after - before) < 1e-6
    residual = wet["params"]["residual_db"]
    assert abs(residual["after"] - residual["before"]) < 1e-6


def test_spectral_match_anchor_holds_low_end(tmp_path, monkeypatch):
    """A DARK source matched to a BRIGHT reference: mean-normalizing cuts the (decent) low end
    because the dark curve's mean sits low; anchoring at the sub pivots on that band instead, so
    the low end is held and only the mids/highs rise (vault-9s8c).

    ``protect_below_hz=0.0`` in both runs — otherwise sub-protection, not the anchor, would be
    what holds the low end.
    """
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    src = _put_wav(_noise_shaped(tilt=-0.95, seed=31), SR)      # dark: highs rolled off
    ref_path = tmp_path / "ref.wav"
    sf.write(str(ref_path), _noise_shaped(tilt=10.0, seed=32), SR, format="WAV", subtype="FLOAT")

    kw = dict(reference_path=str(ref_path), strength=1.0, max_correction_db=12.0,
              protect_below_hz=0.0)
    anchored = edit.apply_spectral_match(src, anchor_hz=45.0, **kw)
    meaned = edit.apply_spectral_match(src, **kw)

    # the pivot band (the one containing 45 Hz) is recorded and reads 0 in all three curves
    params = anchored["params"]
    assert params["anchor_hz"] == 45.0
    k = params["band_centers_hz"].index(params["anchor_band_hz"])
    assert abs(params["correction_db"][k]) < 1e-6
    for curve in params["spectrum_db"].values():
        assert abs(curve[k]) < 1e-6
    assert "anchor_band_hz" not in meaned["params"] and meaned["params"]["anchor_hz"] is None

    before, sr = _load(src)
    a_mono = _load(anchored)[0].mean(axis=1)
    m_mono = _load(meaned)[0].mean(axis=1)
    b_mono = before.mean(axis=1)

    sub_before = _band_db(b_mono, sr, 30, 60)
    assert _band_db(a_mono, sr, 30, 60) >= sub_before - 0.5     # anchored: sub held
    assert _band_db(a_mono, sr, 1000, 8000) > _band_db(b_mono, sr, 1000, 8000) + 1.0  # mids/highs up
    # the bug this mode exists for: mean-normalization CUTS the dark source's low bands
    assert meaned["params"]["correction_db"][k] < -1.0
    assert _band_db(m_mono, sr, 30, 60) < sub_before - 1.0


def _crest(x):
    return float(np.max(np.abs(x)) / (np.sqrt(np.mean(x ** 2)) + 1e-12))


def test_compress_reduces_crest(tmp_path, monkeypatch):
    """A quiet tone with periodic loud bursts (high crest) → compression lowers the crest."""
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    n = SR * 2
    t = np.arange(n) / SR
    sig = 0.08 * np.sin(2 * np.pi * 200 * t)
    for k in range(SR // 2, n - 1200, SR // 2):   # loud bursts well inside bounds
        sig[k:k + 1000] += 0.85 * np.hanning(1000)
    src = _put_wav(sig.reshape(-1, 1).astype(np.float32), SR)
    wet = edit.apply_compress(src, threshold_db=-24.0, ratio=4.0, attack_ms=2.0, release_ms=80.0)
    assert wet.get("op") == "compress" and wet.get("op_version") == edit.COMPRESS_OP_VERSION
    assert wet["params"]["mean_gain_reduction_db"] < 0  # it actually compressed
    before, _ = _load(src)
    after, _ = _load(wet)
    assert _crest(after.mean(axis=1)) < _crest(before.mean(axis=1))


def test_compress_below_threshold_untouched(tmp_path, monkeypatch):
    """A signal entirely below threshold passes through ~unchanged."""
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    quiet = (0.02 * np.sin(2 * np.pi * 200 * np.arange(SR) / SR)).reshape(-1, 1).astype(np.float32)
    src = _put_wav(quiet, SR)
    wet = edit.apply_compress(src, threshold_db=-12.0, ratio=4.0)  # signal ≈ -34 dB
    assert abs(wet["params"]["mean_gain_reduction_db"]) < 0.5
    before, _ = _load(src)
    after, _ = _load(wet)
    assert np.allclose(before, after, atol=2e-3)


def test_compress_reduces_dynamic_range(tmp_path, monkeypatch):
    """A loud half + a (below-threshold) quiet half: compression pulls the loud half down and
    leaves the quiet half alone, so the loud/quiet level ratio shrinks — the core behavior, and
    the mechanism behind loudness parity (less crest → normalize can push louder at a ceiling)."""
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    half = SR
    th = np.arange(half) / SR
    loud = 0.6 * np.sin(2 * np.pi * 200 * th)     # ≈ -4 dBFS, above threshold
    quiet = 0.05 * np.sin(2 * np.pi * 200 * th)   # ≈ -26 dBFS, below threshold
    sig = np.concatenate([loud, quiet]).reshape(-1, 1).astype(np.float32)
    src = _put_wav(sig, SR)
    wet = edit.apply_compress(src, threshold_db=-18.0, ratio=4.0, attack_ms=5.0, release_ms=80.0)
    assert wet["params"]["mean_gain_reduction_db"] < 0
    before, _ = _load(src)
    after, _ = _load(wet)

    def seg_rms(x, a, b):
        return float(np.sqrt(np.mean(x[a:b, 0] ** 2)))

    # mid-segment windows avoid the attack/release edge transients
    dr_before = seg_rms(before, SR // 4, SR // 2) / (seg_rms(before, SR + SR // 4, SR + SR // 2) + 1e-9)
    dr_after = seg_rms(after, SR // 4, SR // 2) / (seg_rms(after, SR + SR // 4, SR + SR // 2) + 1e-9)
    assert dr_after < dr_before  # loud/quiet ratio reduced (dynamics compressed)


def test_compress_rejects_bad_ratio(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    src = _put_wav((0.3 * np.sin(2 * np.pi * 200 * np.arange(SR) / SR)).reshape(-1, 1).astype(np.float32), SR)
    with pytest.raises(ValueError):
        edit.apply_compress(src, ratio=0.5)
    with pytest.raises(ValueError):
        edit.apply_compress(src, attack_ms=-1.0)


def test_compress_empty_input_passthrough(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    src = _put_wav(np.zeros((0, 2), dtype=np.float32), SR)
    wet = edit.apply_compress(src, threshold_db=-18.0)  # must not raise on empty
    assert "note" in wet["params"]


def test_widen_short_input_passthrough(tmp_path, monkeypatch):
    """A stereo slice too short for the crossover filter passes through (no crash)."""
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    tiny = np.column_stack([np.zeros(8), np.zeros(8)]).astype(np.float32)
    tiny[2, 0] = 0.5  # a tiny transient, L≠R so it's genuinely stereo
    src = _put_wav(tiny, SR)
    wet = edit.apply_widen(src, side_gain_db=6.0)
    assert "too short" in wet["params"].get("note", "")
    before, _ = _load(src)
    after, _ = _load(wet)
    assert np.allclose(before, after, atol=1e-6)


# --- automate -----------------------------------------------------------------------------

def _band_rms_fft(mono, sr, lo, hi):
    spec = np.abs(np.fft.rfft(mono))
    fr = np.fft.rfftfreq(len(mono), 1.0 / sr)
    m = (fr >= lo) & (fr < hi)
    return float(np.sqrt(np.mean(spec[m] ** 2))) if m.any() else 0.0


def test_automate_gain_square_gates_second_half():
    """square gain at depth 1.0, duty 0.5, 1 cycle → first half passes, second half silenced."""
    sr = SR
    sig = (0.3 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)).astype(np.float32)[:, None]
    src = _put_wav(sig, sr=sr)
    wet = edit.apply_automate(src, target="gain", shape="square", cycles=1.0, depth=1.0, duty=0.5)
    out, _ = _load(wet)
    half = len(out) // 2
    first = np.sqrt(np.mean(out[:half] ** 2))
    second = np.sqrt(np.mean(out[half + 100:] ** 2))  # skip the edge sample at the switch
    assert first > 0.1
    assert second < 1e-3


def test_automate_gain_depth_zero_is_noop():
    sr = SR
    sig = (0.3 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)).astype(np.float32)[:, None]
    src = _put_wav(sig, sr=sr)
    wet = edit.apply_automate(src, target="gain", shape="sine", cycles=4.0, depth=0.0)
    before, _ = _load(src)
    after, _ = _load(wet)
    assert np.allclose(before, after, atol=1e-6)


def test_automate_gain_saw_up_rises_within_cycle():
    """saw-up over 1 cycle: a constant tone should be quieter at the cycle start than its end."""
    sr = SR
    sig = (0.3 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)).astype(np.float32)[:, None]
    src = _put_wav(sig, sr=sr)
    wet = edit.apply_automate(src, target="gain", shape="saw-up", cycles=1.0, depth=0.8)
    out, _ = _load(wet)
    q = len(out) // 4
    start_rms = np.sqrt(np.mean(out[:q] ** 2))
    end_rms = np.sqrt(np.mean(out[-q:] ** 2))
    assert end_rms > 2.0 * start_rms


def test_automate_cutoff_darkens_highs():
    """A held-low cutoff (constant v=0 via points) lowpasses noise: high band collapses."""
    sr = SR
    rng = np.random.default_rng(0)
    noise = (0.2 * rng.standard_normal(sr)).astype(np.float32)[:, None]
    src = _put_wav(noise, sr=sr)
    wet = edit.apply_automate(src, target="cutoff", shape="points", points=[(0.0, 0.0), (1.0, 0.0)],
                              cycles=1.0, lo_hz=400.0, hi_hz=8000.0, resonance=0.707)
    before, _ = _load(src); after, _ = _load(wet)
    hi_before = _band_rms_fft(before[:, 0], sr, 4000, 10000)
    hi_after = _band_rms_fft(after[:, 0], sr, 4000, 10000)
    lo_before = _band_rms_fft(before[:, 0], sr, 100, 300)
    lo_after = _band_rms_fft(after[:, 0], sr, 100, 300)
    assert hi_after < 0.2 * hi_before          # highs strongly attenuated
    assert lo_after > 0.5 * lo_before          # lows largely preserved


def test_automate_cutoff_resonance_peaks():
    """High Q boosts energy right at the cutoff relative to a low-Q (flat) lowpass."""
    sr = SR
    rng = np.random.default_rng(1)
    noise = (0.2 * rng.standard_normal(sr)).astype(np.float32)[:, None]
    src = _put_wav(noise, sr=sr)
    pts = [(0.0, 0.0), (1.0, 0.0)]  # hold cutoff at lo_hz = 1000
    flat = edit.apply_automate(src, target="cutoff", shape="points", points=pts, cycles=1.0,
                               lo_hz=1000.0, hi_hz=8000.0, resonance=0.707)
    acid = edit.apply_automate(src, target="cutoff", shape="points", points=pts, cycles=1.0,
                               lo_hz=1000.0, hi_hz=8000.0, resonance=8.0)
    fa, _ = _load(flat); ac, _ = _load(acid)
    # ratio of energy at the cutoff (900-1100) to energy well below it (300-500)
    flat_ratio = _band_rms_fft(fa[:, 0], sr, 900, 1100) / (_band_rms_fft(fa[:, 0], sr, 300, 500) + 1e-12)
    acid_ratio = _band_rms_fft(ac[:, 0], sr, 900, 1100) / (_band_rms_fft(ac[:, 0], sr, 300, 500) + 1e-12)
    assert acid_ratio > 1.5 * flat_ratio


def test_automate_pan_mono_to_stereo():
    sr = SR
    sig = (0.3 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)).astype(np.float32)[:, None]
    src = _put_wav(sig, sr=sr)
    wet = edit.apply_automate(src, target="pan", shape="sine", cycles=2.0, depth=1.0)
    out, _ = _load(wet)
    assert out.shape[1] == 2
    assert not np.allclose(out[:, 0], out[:, 1], atol=1e-3)  # genuinely panned


def test_automate_rejects_bad_params():
    sr = SR
    src = _put_wav((0.1 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)).astype(np.float32)[:, None], sr=sr)
    with pytest.raises(ValueError):
        edit.apply_automate(src, target="bogus", shape="sine", cycles=1.0)
    with pytest.raises(ValueError):
        edit.apply_automate(src, target="gain", shape="sine", cycles=0.0)
    with pytest.raises(ValueError):
        edit.apply_automate(src, target="gain", shape="sine", cycles=1.0, depth=1.5)
    with pytest.raises(ValueError):
        edit.apply_automate(src, target="gain", shape="points", cycles=1.0, points=None)


def test_automate_emits_wet_lineage():
    sr = SR
    src = _put_wav((0.2 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)).astype(np.float32)[:, None],
                   sr=sr, role="bass")
    wet = edit.apply_automate(src, target="gain", shape="sine", cycles=4.0, depth=0.5)
    assert wet["role"] == "bass.wet"
    assert wet["op"] == "automate"
    assert wet["op_version"] == "automate@1"
    assert wet["params"]["target"] == "gain"


# --- stereoize ----------------------------------------------------------------------------

def _side_mid(samples):
    L, R = samples[:, 0], samples[:, 1]
    mid = 0.5 * (L + R); side = 0.5 * (L - R)
    return float(np.sqrt(np.mean(side ** 2)) / (np.sqrt(np.mean(mid ** 2)) + 1e-12))


def test_stereoize_widens_mono_source():
    """A dual-mono input (side≈0) gains real side energy after stereoize."""
    sr = SR
    rng = np.random.default_rng(3)
    m = 0.2 * rng.standard_normal(sr)
    src = _put_wav(np.column_stack([m, m]).astype(np.float32), sr=sr)
    before, _ = _load(src)
    wet = edit.apply_stereoize(src, amount=0.6, crossover_hz=200.0)
    after, _ = _load(wet)
    assert _side_mid(before) < 1e-6          # started mono
    assert _side_mid(after) > 0.2            # genuinely widened


def test_stereoize_preserves_mono_sum():
    """The defining property: L+R (mono downmix) is unchanged → mono-measured spectrum intact."""
    sr = SR
    rng = np.random.default_rng(4)
    m = 0.2 * rng.standard_normal(sr)
    src = _put_wav(np.column_stack([m, m]).astype(np.float32), sr=sr)
    before, _ = _load(src)
    after, _ = _load(edit.apply_stereoize(src, amount=0.8, crossover_hz=200.0))
    mono_before = before[:, 0] + before[:, 1]
    mono_after = after[:, 0] + after[:, 1]
    # centroid of the mono sum must be essentially unchanged (decoupled from width)
    def centroid(x):
        spec = np.abs(np.fft.rfft(x)); fr = np.fft.rfftfreq(len(x), 1.0 / sr)
        return float(np.sum(fr * spec) / (np.sum(spec) + 1e-12))
    assert np.allclose(mono_before, mono_after, atol=2e-3)
    assert abs(centroid(mono_before) - centroid(mono_after)) < 5.0   # Hz


def test_stereoize_keeps_low_end_mono():
    """Low-band content (below the crossover) should stay mono; only highs widen."""
    sr = SR
    t = np.arange(sr) / sr
    low = 0.3 * np.sin(2 * np.pi * 60 * t)      # below 200 Hz crossover
    high = 0.3 * np.sin(2 * np.pi * 6000 * t)
    m = (low + high).astype(np.float32)
    src = _put_wav(np.column_stack([m, m]), sr=sr)
    after, _ = _load(edit.apply_stereoize(src, amount=0.8, crossover_hz=200.0))
    low_side = _side_band_energy(after, sr, 40, 120)
    high_side = _side_band_energy(after, sr, 4000, 8000)
    assert low_side < 0.02 * high_side          # bass stays centred, highs spread


def test_stereoize_amount_zero_noop_and_rejects_negative():
    sr = SR
    m = 0.2 * np.sin(2 * np.pi * 300 * np.arange(sr) / sr)
    src = _put_wav(np.column_stack([m, m]).astype(np.float32), sr=sr)
    before, _ = _load(src)
    after, _ = _load(edit.apply_stereoize(src, amount=0.0))
    assert np.allclose(before, after, atol=1e-6)
    with pytest.raises(ValueError):
        edit.apply_stereoize(src, amount=-0.5)
    with pytest.raises(ValueError):
        edit.apply_stereoize(src, amount=0.5, order=0)


def test_stereoize_short_input_passthrough():
    """A signal too short to crossover passes through unchanged (no bass leak into the side)."""
    sr = SR
    m = (0.2 * np.sin(2 * np.pi * 300 * np.arange(20) / sr)).astype(np.float32)
    src = _put_wav(np.column_stack([m, m]), sr=sr)
    before, _ = _load(src)
    wet = edit.apply_stereoize(src, amount=0.8, crossover_hz=200.0, order=4)
    after, _ = _load(wet)
    assert "too short" in wet["params"].get("note", "")
    assert np.allclose(before, after, atol=1e-6)


def _crest_db(mono):
    rms = np.sqrt(np.mean(mono ** 2)) + 1e-12
    return 20 * np.log10(np.max(np.abs(mono)) / rms)


def test_maximize_reduces_crest_and_caps_peaks():
    """A peaky signal (transient spikes over a quiet body) gets lower crest + a respected ceiling."""
    sr = SR
    body = 0.05 * np.sin(2 * np.pi * 200 * np.arange(sr) / sr)
    body[::4000] += 0.9   # sharp transient spikes → high crest
    src = _put_wav(body.astype(np.float32)[:, None], sr=sr)
    before, _ = _load(src)
    wet = edit.apply_maximize(src, ceiling_dbtp=-1.0, makeup_db=6.0)
    after, _ = _load(wet)
    ceiling = 10 ** (-1.0 / 20.0)
    assert np.max(np.abs(after)) <= ceiling + 1e-3          # brickwall respected
    assert _crest_db(after[:, 0]) < _crest_db(before[:, 0]) - 2.0   # crest meaningfully reduced


def test_maximize_makeup_raises_rms():
    sr = SR
    sig = (0.1 * np.sin(2 * np.pi * 200 * np.arange(sr) / sr)).astype(np.float32)[:, None]
    src = _put_wav(sig, sr=sr)
    before, _ = _load(src); after, _ = _load(edit.apply_maximize(src, ceiling_dbtp=-1.0, makeup_db=8.0))
    assert np.sqrt(np.mean(after ** 2)) > np.sqrt(np.mean(before ** 2))


def test_maximize_rejects_bad_params_and_handles_empty():
    sr = SR
    src = _put_wav((0.1 * np.sin(2 * np.pi * 200 * np.arange(sr) / sr)).astype(np.float32)[:, None], sr=sr)
    with pytest.raises(ValueError):
        edit.apply_maximize(src, makeup_db=-3.0)
    with pytest.raises(ValueError):
        edit.apply_maximize(src, lookahead_ms=0.0)


def test_automate_cutoff_drops_depth_from_params():
    """depth is gain/pan only — the cutoff frame must not carry a misleading depth param."""
    sr = SR
    src = _put_wav((0.2 * np.sin(2 * np.pi * 300 * np.arange(sr) / sr)).astype(np.float32)[:, None], sr=sr)
    wet = edit.apply_automate(src, target="cutoff", shape="tri", cycles=2.0, depth=0.5,
                              lo_hz=300, hi_hz=6000)
    assert "depth" not in wet["params"]
    assert wet["params"]["target"] == "cutoff"


# --- loopify (tile-safety) ----------------------------------------------------------------


def _onset_idx(x, floor_db=-45.0):
    """First sample above ``floor_db`` of the peak — the same downbeat detector loopify uses."""
    mag = np.abs(x).max(axis=1)
    above = np.where(mag > 10 ** (floor_db / 20) * float(mag.max()))[0]
    return int(above[0]) if len(above) else -1


def _kickish(n, sr=SR, freq=60.0, decay=8.0, amp=0.8, sustain=0.15):
    """A decaying cosine over a sustained tone — a downbeat that is LOUD at sample 0 (a sine would
    start at zero and make the onset detector's answer a property of the phase rather than of the
    trim), with a tail that is still ringing at the cut (which is what makes a naive repeat click)."""
    t = np.arange(n) / sr
    body = amp * np.exp(-decay * t) * np.cos(2 * np.pi * freq * t)
    return (body + sustain * np.cos(2 * np.pi * 220 * t)).astype(np.float32)[:, None]


def _rendered_loop(lead_ms, body_len, sr=SR):
    """A smplmix-style render: ``lead_ms`` of silence before the downbeat, then the body."""
    lead = int(lead_ms / 1000 * sr)
    return np.concatenate([np.zeros((lead, 1), dtype=np.float32), _kickish(body_len, sr=sr)])


def test_loopify_trims_render_offset_to_downbeat_zero(tmp_path, monkeypatch):
    """The ~5 ms smplmix render-offset is trimmed so the downbeat lands at sample 0."""
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    target = 4 * 60 * SR // 120                       # 1 bar @ 120 BPM, 4/4 → 2 s
    src = _put_wav(_rendered_loop(5.0, target))
    before, _ = _load(src)
    assert _onset_idx(before) == int(0.005 * SR)      # the render offset is really there

    wet = edit.apply_loopify(src, bpm=120.0, bars=1, beats_per_bar=4)
    assert wet["role"] == "source.wet"
    assert wet.get("op") == "loopify" and wet.get("op_version") == edit.LOOPIFY_OP_VERSION
    assert wet.get("of") == src["id"] and wet.get("lineage") == [src["id"]]
    assert wet["params"]["lead_trimmed_samples"] == int(0.005 * SR)

    after, _ = _load(wet)
    # ≤2, not ==0: the 64-sample anti-DC fade-in zeroes sample 0 by design (it kills start DC
    # without softening the transient), so the first audible sample is 1.
    assert _onset_idx(after) <= 2


def test_loopify_sets_exact_bar_length(tmp_path, monkeypatch):
    """Both a long render (truncated) and a short one (zero-padded) come out on the bar grid."""
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    target = int(round(2 * 4 * 60.0 / 132.0 * SR))    # 2 bars @ 132 BPM, 4/4
    for body in (target + 5000, target - 5000):
        wet = edit.apply_loopify(_put_wav(_rendered_loop(5.0, body)), bpm=132.0, bars=2)
        assert wet["params"]["target_len_samples"] == target
        after, _ = _load(wet)
        assert after.shape[0] == target               # exact — N repeats stay on the grid


def test_loopify_seam_is_declicked(tmp_path, monkeypatch):
    """The seam fades to zero, so tiling the output twice has ~no discontinuity (no click)."""
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    target = 4 * 60 * SR // 120
    src = _put_wav(_rendered_loop(5.0, target + 3000))
    raw, _ = _load(src)
    wet = edit.apply_loopify(src, bpm=120.0, bars=1, declick_ms=5.0)
    after, _ = _load(wet)

    assert abs(float(after[-1, 0])) < 1e-9            # end fade actually reaches zero
    tiled = np.concatenate([after, after])
    seam = abs(float(tiled[target, 0] - tiled[target - 1, 0]))
    naive = np.concatenate([raw, raw])                # the un-loopified repeat, for contrast
    naive_seam = abs(float(naive[len(raw), 0] - naive[len(raw) - 1, 0]))
    assert seam < 1e-6
    assert naive_seam > 0.05                          # the click loopify removes (~0.15 here)


def test_loopify_keeps_a_musical_fade_in(tmp_path, monkeypatch):
    """A lead LONGER than max_trim_ms is a musical fade-in/pickup, not the render offset — keep it."""
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    target = 4 * 60 * SR // 120
    lead = int(0.05 * SR)                             # 50 ms ≫ the 12 ms render-offset bound
    wet = edit.apply_loopify(_put_wav(_rendered_loop(50.0, target)), bpm=120.0, bars=1,
                             max_trim_ms=12.0)
    assert wet["params"]["lead_trimmed_samples"] == 0
    after, _ = _load(wet)
    assert np.max(np.abs(after[:lead - 1])) < 1e-6    # the lead survives untouched
    assert _onset_idx(after) >= lead - 1


def test_loopify_rejects_bad_params(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    src = _put_wav(_kickish(SR))
    with pytest.raises(ValueError):
        edit.apply_loopify(src, bpm=0.0)
    with pytest.raises(ValueError):
        edit.apply_loopify(src, bpm=120.0, bars=0)
