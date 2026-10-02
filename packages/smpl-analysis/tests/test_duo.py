"""Tests for smpl_analysis.duo — the 2-audio-input op family (tickets vault-nkvw, vault-d3qk).

Covers the ``vocode`` channel vocoder: modulator-envelope tracking, carrier-timbre transfer,
determinism, sample-rate mismatch, and the 2-input lineage convention (lineage carries BOTH
the carrier and modulator ids).

Covers the ``sidechain`` duck: the A/B on a sub under a kick (RMS pulled down under each
transient, unchanged between them), stereo passthrough, the depth-0 no-op, parameter
validation, and the same 2-input lineage convention.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from smpl_analysis import duo


SR = 44100


# ---------------------------------------------------------------------------
# CAS isolation — a fresh SMPL_CAS_DIR per test (cas_dir() reads the env each call).
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _cas_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))


def _put_wav(samples, sr, role="source"):
    from smplstream import cas, frames as F

    samples = np.asarray(samples, dtype="float32")
    if samples.ndim == 1:
        samples = samples[:, None]
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="FLOAT")
    h = cas.put_audio_bytes(buf.getvalue())
    meta = cas.read_meta(h) or {}
    return F.audio_frame(
        h,
        sr=meta.get("sr", sr),
        ch=meta.get("ch", samples.shape[1]),
        dur=meta.get("dur", samples.shape[0] / sr),
        role=role,
    )


def _load_mono(frame):
    from smplstream import cas

    data, sr = sf.read(str(cas.get_path(frame["hash"])), dtype="float64", always_2d=True)
    return data.mean(axis=1), sr


def _harmonic_carrier(f0=60.0, n_harm=40, dur=1.5, sr=SR):
    """A harmonic-rich tone at ``f0`` with 1/k-weighted harmonics (fundamental dominates)."""
    t = np.arange(int(dur * sr)) / sr
    sig = np.zeros_like(t)
    for k in range(1, n_harm + 1):
        f = f0 * k
        if f >= 0.45 * sr:
            break
        sig += (1.0 / k) * np.sin(2 * np.pi * f * t)
    sig /= np.max(np.abs(sig))
    return (0.8 * sig).astype("float32")


def _noise_bursts(dur=1.5, sr=SR, n_bursts=3, seed=1234):
    """``n_bursts`` white-noise bursts separated by silence — spanning the full ``dur``."""
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    sig = np.zeros(n, dtype="float64")
    seg = n // (2 * n_bursts)  # alternating burst / gap of equal length
    burst_ranges = []
    for b in range(n_bursts):
        start = (2 * b) * seg
        end = start + seg
        sig[start:end] = rng.standard_normal(end - start)
        burst_ranges.append((start, end))
    sig *= 0.5
    return sig.astype("float32"), burst_ranges, seg


def _short_time_rms(x, win):
    """Non-overlapping short-time RMS frames."""
    n = (len(x) // win) * win
    frames = x[:n].reshape(-1, win)
    return np.sqrt(np.mean(frames ** 2, axis=1) + 1e-20)


# ---------------------------------------------------------------------------
# 1. Envelope tracking: output loudness follows the modulator; gaps go quiet.
# ---------------------------------------------------------------------------
def test_envelope_tracking_follows_modulator():
    carrier = _put_wav(_harmonic_carrier(), SR, role="carrier")
    mod_sig, burst_ranges, seg = _noise_bursts()
    modulator = _put_wav(mod_sig, SR, role="mod")

    wet = duo.apply_vocode(carrier, modulator)
    out, sr = _load_mono(wet)

    # Short-time RMS of output should correlate with the modulator's short-time RMS.
    win = 2048
    out_rms = _short_time_rms(out, win)
    mod_rms = _short_time_rms(mod_sig.astype("float64"), win)
    m = min(len(out_rms), len(mod_rms))
    r = np.corrcoef(out_rms[:m], mod_rms[:m])[0, 1]
    assert r > 0.7, f"output RMS envelope correlation with modulator too low: r={r:.3f}"

    # Gap energy should be a small fraction of burst energy.
    burst_e = 0.0
    for (s, e) in burst_ranges:
        burst_e += float(np.sum(out[s:e] ** 2))
    gap_e = 0.0
    for b in range(len(burst_ranges) - 1):
        gs = burst_ranges[b][1]
        ge = burst_ranges[b + 1][0]
        gap_e += float(np.sum(out[gs:ge] ** 2))
    assert gap_e < 0.10 * burst_e, f"gap energy {gap_e:.4g} not < 10% of burst energy {burst_e:.4g}"


# ---------------------------------------------------------------------------
# 2. Timbre from the carrier: output spectrum concentrated at carrier harmonics (low),
#    not at the broadband modulator noise.
# ---------------------------------------------------------------------------
def test_timbre_comes_from_carrier():
    carrier = _put_wav(_harmonic_carrier(), SR, role="carrier")
    # Sustained (full-length) red noise → roughly constant, low-tilted envelope (the WRUM case:
    # a deep-voiced modulator), so the output spectrum is dominated by the carrier's low harmonic
    # line structure rather than by modulator dynamics.
    rng = np.random.default_rng(7)
    red = np.cumsum(rng.standard_normal(int(1.5 * SR)))
    red -= red.mean()
    red = (0.5 * red / np.max(np.abs(red))).astype("float32")
    modulator = _put_wav(red, SR, role="mod")

    # ess_mix=0 isolates the vocoder BODY: the ess path deliberately injects fixed HF energy
    # (consonant intelligibility), which would otherwise swamp the carrier's low harmonics here.
    wet = duo.apply_vocode(carrier, modulator, ess_mix=0.0)
    out, sr = _load_mono(wet)

    spec = np.abs(np.fft.rfft(out * np.hanning(len(out))))
    freqs = np.fft.rfftfreq(len(out), 1.0 / sr)
    peak_freq = freqs[int(np.argmax(spec))]
    # Dominant peak is low AND sits on a carrier harmonic (multiple of 60 Hz) — the line
    # structure is the carrier's timbre, not the continuous-spectrum modulator noise.
    assert peak_freq < 200.0, f"dominant spectral peak at {peak_freq:.1f} Hz, expected < 200 Hz"
    assert abs(peak_freq % 60.0) < 6.0 or abs((peak_freq % 60.0) - 60.0) < 6.0, (
        f"dominant peak {peak_freq:.1f} Hz is not on a carrier harmonic (multiple of 60)")


# ---------------------------------------------------------------------------
# 3. Determinism: two runs are bit-identical.
# ---------------------------------------------------------------------------
def test_determinism():
    carrier = _put_wav(_harmonic_carrier(dur=1.0), SR, role="carrier")
    mod_sig, _, _ = _noise_bursts(dur=1.0)
    modulator = _put_wav(mod_sig, SR, role="mod")

    a = duo.apply_vocode(carrier, modulator)
    b = duo.apply_vocode(carrier, modulator)
    assert a["hash"] == b["hash"]
    assert a["id"] == b["id"]
    oa, _ = _load_mono(a)
    ob, _ = _load_mono(b)
    assert np.array_equal(oa, ob)


# ---------------------------------------------------------------------------
# 4. Sample-rate mismatch: a 22050 Hz modulator is resampled to the carrier's sr.
# ---------------------------------------------------------------------------
def test_sr_mismatch_modulator_resampled():
    carrier = _put_wav(_harmonic_carrier(dur=1.0), SR, role="carrier")
    mod_sig, _, _ = _noise_bursts(dur=1.0, sr=22050)
    modulator = _put_wav(mod_sig, 22050, role="mod")

    wet = duo.apply_vocode(carrier, modulator)
    assert wet["meta"]["sr"] == SR
    out, sr = _load_mono(wet)
    assert sr == SR
    assert np.max(np.abs(out)) > 0.0
    assert wet["params"]["sr_hz"] == SR


# ---------------------------------------------------------------------------
# 5. Lineage: the wet frame's lineage carries BOTH input ids (2-input convention).
# ---------------------------------------------------------------------------
def test_lineage_carries_both_inputs():
    carrier = _put_wav(_harmonic_carrier(dur=0.5), SR, role="carrier")
    mod_sig, _, _ = _noise_bursts(dur=0.5)
    modulator = _put_wav(mod_sig, SR, role="mod")

    wet = duo.apply_vocode(carrier, modulator)
    assert wet["kind"] == "audio"
    assert wet["role"] == "carrier.wet"
    assert wet.get("of") == carrier["id"]
    assert wet.get("lineage") == [carrier["id"], modulator["id"]]
    assert wet.get("op") == "vocode"
    assert wet.get("op_version") == duo.VOCODE_OP_VERSION
    assert carrier["id"] in wet["lineage"] and modulator["id"] in wet["lineage"]
    # vocode@2 contract: BODY normalized to 0.9, then the ess add may raise the final
    # peak up to the 0.98 safety cap (intelligibility fix — vowels no longer crushed).
    out, _ = _load_mono(wet)
    assert 0.85 < np.max(np.abs(out)) <= 0.98 + 1e-4


# ===========================================================================
# sidechain (vault-d3qk) — duck a TARGET under a TRIGGER's transients.
# ===========================================================================
def _sub_sine(dur=2.0, sr=SR, f0=55.0, amp=0.5):
    """A steady sub-bass sine — the duck target (constant RMS, so any dip IS the duck)."""
    t = np.arange(int(dur * sr)) / sr
    return (amp * np.sin(2 * np.pi * f0 * t)).astype("float32")


def _kick_train(dur=2.0, sr=SR, period=0.5, decay_s=0.05, amp=0.9):
    """Short decaying 60 Hz bursts on a fixed grid with silence between — the trigger."""
    n = int(dur * sr)
    sig = np.zeros(n, dtype="float64")
    onsets = []
    k = 0
    while int(k * period * sr) < n:
        start = int(k * period * sr)
        ln = min(int(0.15 * sr), n - start)
        tt = np.arange(ln) / sr
        sig[start:start + ln] += amp * np.sin(2 * np.pi * 60.0 * tt) * np.exp(-tt / decay_s)
        onsets.append(start)
        k += 1
    return sig.astype("float32"), onsets


def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype="float64") ** 2) + 1e-20))


def _rms_delta_db(wet, dry, lo_s, hi_s, onset, sr=SR):
    """dB difference of wet vs dry RMS over ``[onset+lo_s, onset+hi_s)``."""
    a, b = onset + int(lo_s * sr), onset + int(hi_s * sr)
    return 20.0 * np.log10(_rms(wet[a:b]) / _rms(dry[a:b]))


# ---------------------------------------------------------------------------
# 6. The A/B: the sub ducks under each kick and is untouched between kicks.
# ---------------------------------------------------------------------------
def test_sidechain_ducks_sub_under_kick():
    dry = _sub_sine()
    trig_sig, onsets = _kick_train()
    target = _put_wav(dry, SR, role="sub")
    trigger = _put_wav(trig_sig, SR, role="kick")

    wet_frame = duo.apply_sidechain(target, trigger, attack_ms=5.0, release_ms=120.0, depth_db=12.0)
    wet, sr = _load_mono(wet_frame)
    assert sr == SR and len(wet) == len(dry)

    # Under each kick (8–45 ms after onset): a deep duck. The realized depth sits a little
    # under the 12 dB setting because the peak follower's attack ramp never quite reaches the
    # trigger's own peak — which is exactly what the frame reports as gr_max_db.
    gr_max = wet_frame["params"]["gr_max_db"]
    for onset in onsets:
        d = _rms_delta_db(wet, dry, 0.008, 0.045, onset)
        assert -12.0 - 0.5 <= d <= -8.0, f"duck under kick @{onset} was {d:.2f} dB"
        assert abs(d + gr_max) < 0.5, f"duck {d:.2f} dB disagrees with reported gr_max_db {gr_max}"

    # Between kicks (400–490 ms after onset, i.e. just before the next one): recovered.
    for onset in onsets[:-1]:
        d = _rms_delta_db(wet, dry, 0.40, 0.49, onset)
        assert abs(d) < 0.5, f"target changed by {d:.2f} dB between kicks (expected < 0.5)"


# ---------------------------------------------------------------------------
# 7. params carry the MEASURED reduction (the A/B evidence, readable off the frame).
# ---------------------------------------------------------------------------
def test_sidechain_params_carry_measured_reduction():
    trig_sig, _ = _kick_train(dur=1.0)
    target = _put_wav(_sub_sine(dur=1.0), SR, role="sub")
    trigger = _put_wav(trig_sig, SR, role="kick")

    wet = duo.apply_sidechain(target, trigger, depth_db=12.0)
    p = wet["params"]
    assert p["attack_ms"] == 5.0 and p["release_ms"] == 120.0
    assert p["depth_db"] == 12.0 and p["threshold_db"] == -30.0
    assert 8.0 < p["gr_max_db"] <= 12.0, p["gr_max_db"]
    assert 0.0 < p["ducked_fraction"] < 1.0, p["ducked_fraction"]


# ---------------------------------------------------------------------------
# 8. Lineage: 2-input convention, wet role, op/op_version.
# ---------------------------------------------------------------------------
def test_sidechain_lineage_carries_both_inputs():
    trig_sig, _ = _kick_train(dur=0.5)
    target = _put_wav(_sub_sine(dur=0.5), SR, role="sub")
    trigger = _put_wav(trig_sig, SR, role="kick")

    wet = duo.apply_sidechain(target, trigger)
    assert wet["kind"] == "audio"
    assert wet["role"] == "sub.wet"
    assert wet.get("of") == target["id"]
    assert wet.get("lineage") == [target["id"], trigger["id"]]
    assert wet.get("op") == "sidechain"
    assert wet.get("op_version") == duo.SIDECHAIN_OP_VERSION


# ---------------------------------------------------------------------------
# 9. A stereo target stays stereo — one gain curve over both channels.
# ---------------------------------------------------------------------------
def test_sidechain_stereo_target_keeps_channels():
    mono = _sub_sine(dur=0.5)
    stereo = np.stack([mono, 0.5 * mono], axis=1)
    target = _put_wav(stereo, SR, role="mix")
    trig_sig, _ = _kick_train(dur=0.5)
    trigger = _put_wav(trig_sig, SR, role="kick")

    from smplstream import cas

    wet = duo.apply_sidechain(target, trigger)
    assert wet["meta"]["ch"] == 2
    data, sr = sf.read(str(cas.get_path(wet["hash"])), dtype="float64", always_2d=True)
    assert data.shape[1] == 2 and data.shape[0] == len(mono)
    # Same curve on both channels → the L/R ratio of the source survives.
    loud = np.abs(data[:, 0]) > 1e-4
    ratio = data[loud, 1] / data[loud, 0]
    assert np.allclose(ratio, 0.5, atol=1e-3)


# ---------------------------------------------------------------------------
# 10. depth 0 is an exact no-op; a short / off-sr trigger aligns to the target.
# ---------------------------------------------------------------------------
def test_sidechain_depth_zero_is_noop():
    dry = _sub_sine(dur=0.5)
    target = _put_wav(dry, SR, role="sub")
    trig_sig, _ = _kick_train(dur=0.5)
    trigger = _put_wav(trig_sig, SR, role="kick")

    wet = duo.apply_sidechain(target, trigger, depth_db=0.0)
    out, _ = _load_mono(wet)
    assert np.allclose(out, dry.astype("float64"), atol=1e-6)
    assert wet["params"]["gr_max_db"] == 0.0
    assert wet["params"]["ducked_fraction"] == 0.0


def test_sidechain_short_and_off_sr_trigger_aligns_to_target():
    dry = _sub_sine(dur=1.0)
    target = _put_wav(dry, SR, role="sub")
    # Half as long as the target AND at a different sample rate: resampled, then zero-padded.
    trig_sig, _ = _kick_train(dur=0.5, sr=22050, period=0.25)
    trigger = _put_wav(trig_sig, 22050, role="kick")

    wet = duo.apply_sidechain(target, trigger, release_ms=60.0)
    out, sr = _load_mono(wet)
    assert sr == SR and len(out) == len(dry)
    assert wet["params"]["sr_hz"] == SR
    # First half ducked, tail (past the padded-out trigger) untouched.
    assert _rms(out[: SR // 2]) < 0.9 * _rms(dry[: SR // 2])
    assert np.allclose(out[-SR // 4:], dry[-SR // 4:].astype("float64"), atol=1e-6)


# ---------------------------------------------------------------------------
# 11. Parameter validation.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kwargs", [
    {"attack_ms": 0.0},
    {"attack_ms": -1.0},
    {"release_ms": 0.0},
    {"depth_db": -3.0},
    {"threshold_db": 0.0},
    {"threshold_db": 6.0},
])
def test_sidechain_rejects_bad_params(kwargs):
    target = _put_wav(_sub_sine(dur=0.2), SR, role="sub")
    trig_sig, _ = _kick_train(dur=0.2)
    trigger = _put_wav(trig_sig, SR, role="kick")
    with pytest.raises(ValueError):
        duo.apply_sidechain(target, trigger, **kwargs)
