"""Tests for smpl_analysis.edit.render_cutoff_variants — the palette variants op (vault-5yeb).

Covers: N frames emitted with distinct roles / params / lineage, the log-spaced cutoff ladder,
the spectral ordering that makes a palette read closed→open (more HF energy at a higher cutoff),
determinism (same input + params ⇒ identical CAS hash), and the CLI shim's passthrough.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from smpl_analysis import edit


SR = 44100


def _broadband(dur=0.5, sr=SR, channels=1):
    """A deterministic harmonic stack spanning lo→hi, for cutoff-ordering assertions."""
    t = np.arange(int(dur * sr)) / sr
    sig = np.zeros_like(t)
    for f in (110.0, 330.0, 880.0, 2200.0, 5500.0, 11000.0):
        sig += np.sin(2 * np.pi * f * t)
    sig = 0.15 * sig
    if channels == 1:
        return sig.reshape(-1, 1).astype(np.float32)
    return np.column_stack([sig] * channels).astype(np.float32)


def _put_wav(samples, sr, role="source"):
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
        role=role,
    )


def _load_frame_samples(frame):
    from smplstream import cas

    data, sr = sf.read(str(cas.get_path(frame["hash"])), dtype="float32", always_2d=True)
    return data, sr


def _band_energy(samples, sr, lo, hi):
    mono = samples.mean(axis=1)
    spec = np.abs(np.fft.rfft(mono))
    freqs = np.fft.rfftfreq(len(mono), 1.0 / sr)
    mask = (freqs >= lo) & (freqs < hi)
    return float(np.sqrt(np.mean(spec[mask] ** 2))) if mask.any() else 0.0


# --- frames / lineage ------------------------------------------------------------------


def test_emits_n_variants_with_roles_params_and_lineage(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    src = _put_wav(_broadband(), SR)
    out = edit.render_cutoff_variants(src, lo_hz=200.0, hi_hz=8000.0, steps=5)

    assert len(out) == 5
    assert [f["role"] for f in out] == [f"source.variant:{k}" for k in range(1, 6)]
    hashes = {f["hash"] for f in out}
    assert len(hashes) == 5                      # every step is a distinct render
    assert src["hash"] not in hashes

    for k, f in enumerate(out, start=1):
        assert f["kind"] == "audio"
        assert f.get("of") == src["id"]
        assert f.get("lineage") == [src["id"]]
        assert f.get("op") == "variants"
        assert f.get("op_version") == edit.VARIANTS_OP_VERSION
        p = f["params"]
        assert p["variant_index"] == k
        assert p["steps"] == 5
        assert p["lo_hz"] == 200.0 and p["hi_hz"] == 8000.0
        assert p["resonance"] == 0.707
        assert p["spacing"] == "log"
        assert p["sr_hz"] == SR


def test_role_derives_from_source_role_and_strips_wet(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    src = _put_wav(_broadband(dur=0.2), SR, role="pad.wet")
    out = edit.render_cutoff_variants(src, steps=2)
    assert [f["role"] for f in out] == ["pad.variant:1", "pad.variant:2"]


def test_resonance_recorded_in_params(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    src = _put_wav(_broadband(dur=0.2), SR)
    out = edit.render_cutoff_variants(src, steps=2, resonance=6.0)
    assert all(f["params"]["resonance"] == 6.0 for f in out)


# --- cutoff ladder ---------------------------------------------------------------------


def test_cutoffs_are_log_spaced_and_inclusive(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    src = _put_wav(_broadband(dur=0.2), SR)
    out = edit.render_cutoff_variants(src, lo_hz=200.0, hi_hz=8000.0, steps=5)
    cutoffs = [f["params"]["cutoff_hz"] for f in out]

    assert cutoffs[0] == pytest.approx(200.0, rel=1e-6)
    assert cutoffs[-1] == pytest.approx(8000.0, rel=1e-6)
    # log spacing ⇒ constant RATIO between consecutive steps
    ratios = [b / a for a, b in zip(cutoffs, cutoffs[1:])]
    expected = (8000.0 / 200.0) ** (1 / 4)
    assert ratios == pytest.approx([expected] * 4, rel=1e-6)
    assert cutoffs == pytest.approx(edit.variant_cutoffs(200.0, 8000.0, 5), rel=1e-6)


def test_hi_clamped_to_nyquist_and_lo_floored(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    src = _put_wav(_broadband(dur=0.2), SR)
    out = edit.render_cutoff_variants(src, lo_hz=1.0, hi_hz=99999.0, steps=2)
    assert out[0]["params"]["lo_hz"] == 20.0
    assert out[-1]["params"]["hi_hz"] == round(0.49 * SR, 1)


def test_single_step_degenerates_to_lo(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    src = _put_wav(_broadband(dur=0.2), SR)
    out = edit.render_cutoff_variants(src, lo_hz=500.0, hi_hz=8000.0, steps=1)
    assert len(out) == 1
    assert out[0]["params"]["cutoff_hz"] == pytest.approx(500.0, rel=1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"steps": 0},
        {"steps": edit.MAX_VARIANT_STEPS + 1},
        {"resonance": 0.0},
        {"lo_hz": 8000.0, "hi_hz": 200.0, "steps": 3},
    ],
)
def test_invalid_params_raise(tmp_path, monkeypatch, kwargs):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    src = _put_wav(_broadband(dur=0.2), SR)
    with pytest.raises(ValueError):
        edit.render_cutoff_variants(src, **kwargs)


# --- the actual DSP: closed → open ------------------------------------------------------


def test_higher_cutoff_variant_keeps_more_high_frequency_energy(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    src = _put_wav(_broadband(), SR)
    out = edit.render_cutoff_variants(src, lo_hz=200.0, hi_hz=8000.0, steps=5)

    hf = []
    for f in out:
        data, sr = _load_frame_samples(f)
        hf.append(_band_energy(data, sr, 4000, 12000))
    # monotone rise: each step opens further than the last (closed → open palette)
    assert all(b > a for a, b in zip(hf, hf[1:])), hf

    before, sr = _load_frame_samples(src)
    dry_hf = _band_energy(before, sr, 4000, 12000)
    assert hf[0] < 0.2 * dry_hf          # the closed variant is genuinely dark
    assert hf[-1] > 3 * hf[0]            # and the open one is genuinely brighter


def test_low_band_survives_every_variant(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    src = _put_wav(_broadband(), SR)
    out = edit.render_cutoff_variants(src, lo_hz=400.0, hi_hz=8000.0, steps=4)
    before, sr = _load_frame_samples(src)
    dry_lo = _band_energy(before, sr, 60, 200)
    for f in out:
        data, _ = _load_frame_samples(f)
        assert _band_energy(data, sr, 60, 200) > 0.5 * dry_lo


def test_stereo_source_keeps_channel_count(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    src = _put_wav(_broadband(dur=0.2, channels=2), SR)
    out = edit.render_cutoff_variants(src, steps=3)
    for f in out:
        data, _ = _load_frame_samples(f)
        assert data.shape[1] == 2


def test_empty_source_emits_noop_variants(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    src = _put_wav(np.zeros((0, 1), dtype=np.float32), SR)
    out = edit.render_cutoff_variants(src, steps=3)
    assert len(out) == 3
    assert all(f["params"]["noop"] == "empty" for f in out)


# --- determinism -----------------------------------------------------------------------


def test_same_input_and_params_render_identical_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    src = _put_wav(_broadband(dur=0.3), SR)
    first = edit.render_cutoff_variants(src, lo_hz=300.0, hi_hz=6000.0, steps=4, resonance=3.0)
    second = edit.render_cutoff_variants(src, lo_hz=300.0, hi_hz=6000.0, steps=4, resonance=3.0)

    assert [f["hash"] for f in first] == [f["hash"] for f in second]
    for a, b in zip(first, second):
        da, _ = _load_frame_samples(a)
        db, _ = _load_frame_samples(b)
        assert np.array_equal(da, db)
        assert a["params"] == b["params"]


# --- CLI shim --------------------------------------------------------------------------

SMPL = shutil.which("smpl")


@pytest.mark.skipif(SMPL is None, reason="`smpl` console script not on PATH")
def test_cli_passthrough_then_variants(tmp_path):
    env = dict(os.environ)
    env["SMPL_CAS_DIR"] = str(tmp_path / "cas")
    env.pop("VIRTUAL_ENV", None)
    wav = tmp_path / "tone.wav"
    sf.write(str(wav), _broadband(dur=0.25), SR, subtype="FLOAT")

    read = subprocess.run(["smpl", "read", str(wav)], capture_output=True, env=env, timeout=120)
    assert read.returncode == 0, read.stderr
    run = subprocess.run(["smpl", "variants", "--lo", "300", "--hi", "6000", "--steps", "3"],
                         input=read.stdout, capture_output=True, env=env, timeout=180)
    assert run.returncode == 0, run.stderr

    frames = [json.loads(l) for l in run.stdout.splitlines() if l.strip()]
    src = json.loads(read.stdout.splitlines()[0])
    # passthrough first, then one wet frame per step
    assert frames[0]["id"] == src["id"] and frames[0]["role"] == "source"
    variants = frames[1:]
    assert [f["role"] for f in variants] == [f"source.variant:{k}" for k in (1, 2, 3)]
    assert all(f["op"] == "variants" and f["of"] == src["id"] for f in variants)
    assert [f["params"]["variant_index"] for f in variants] == [1, 2, 3]


@pytest.mark.skipif(SMPL is None, reason="`smpl` console script not on PATH")
def test_cli_rejects_inverted_range(tmp_path):
    env = dict(os.environ)
    env["SMPL_CAS_DIR"] = str(tmp_path / "cas")
    env.pop("VIRTUAL_ENV", None)
    run = subprocess.run(["smpl", "variants", "--lo", "8000", "--hi", "200"],
                         input=b"", capture_output=True, env=env, timeout=120)
    assert run.returncode == 2
    assert b"--lo must be below --hi" in run.stderr
