"""Edit operations — the `smpl filter / eq / env / fx / slice` DSP tier (ticket vault-3l83).

Pure functions over a resolved audio frame: each loads the canonical PCM from the CAS,
applies a transform, re-CASes the result, and returns new smplstream frame dicts. The thin
CLI subcommands in the `smpl` package call into these.

Two families:
  - **Audio-producing ops** (filter / eq / env / gain / normalize / limit / widen /
    spectral-match, plus sox-backed fx) return a new `audio` frame with role ``<role>.wet``
    (the dry→wet convention) and full lineage (``of`` / ``lineage`` / ``op`` / ``op_version`` /
    ``params``). The DSP ops use scipy/numpy (pure-Python, deterministic, empty
    env-fingerprint); fx (reverb/delay) shells out to ``sox`` and fingerprints the tool version.
  - **Marker-producing op** (`slice_onsets`) runs librosa onset detection and returns a
    ``marker`` frame (role ``onset``) plus, optionally, one sliced ``audio`` frame per
    region (role ``slice:<n>``).

Heavy imports (librosa, scipy, soundfile, matplotlib) stay INSIDE the functions so a cold
pipe stage starts fast. No new dependencies — scipy/librosa/soundfile/numpy are installed,
sox/ffmpeg are on PATH.
"""

from __future__ import annotations

import io
import math
import os
import subprocess
from typing import Optional

# ---------------------------------------------------------------------------
# op_version constants — bumped on ANY behavior change (spec → *Memoization*).
# ---------------------------------------------------------------------------
FILTER_OP_VERSION = "filter@1"
EQ_OP_VERSION = "eq@1"
ENV_OP_VERSION = "env@1"
FX_OP_VERSION = "fx@1"
SLICE_OP_VERSION = "slice@1"
GAIN_OP_VERSION = "gain@1"
NORMALIZE_OP_VERSION = "normalize@1"
LIMIT_OP_VERSION = "limit@1"
WIDEN_OP_VERSION = "widen@1"
MONO_OP_VERSION = "mono@1"
SPECTRAL_MATCH_OP_VERSION = "spectral-match@1"
COMPRESS_OP_VERSION = "compress@1"
CROP_OP_VERSION = "crop@1"
REVERSE_OP_VERSION = "reverse@1"
PITCH_OP_VERSION = "pitch@1"
STRETCH_OP_VERSION = "stretch@1"
LOOPIFY_OP_VERSION = "loopify@1"


# ---------------------------------------------------------------------------
# Shared helpers.
# ---------------------------------------------------------------------------
def _load_audio(audio_frame: dict):
    """Resolve an audio frame's CAS blob to ``(samples (frames, ch) float32, sr)``."""
    import soundfile as sf

    from smplstream import cas

    src = cas.get_path(audio_frame["hash"])
    data, sr = sf.read(str(src), dtype="float32", always_2d=True)
    return data, int(sr)


def _wet_role(audio_frame: dict) -> str:
    """Derive the ``<role>.wet`` role from the source frame's role (default ``edit``)."""
    role = audio_frame.get("role") or "edit"
    # Strip an existing .wet/.dry suffix so re-filtering stays ``<base>.wet`` (not .wet.wet).
    for suffix in (".wet", ".dry"):
        if role.endswith(suffix):
            role = role[: -len(suffix)]
            break
    return f"{role}.wet"


def _emit_wet_audio(
    samples,
    sr: int,
    *,
    src_frame: dict,
    op: str,
    op_version: str,
    params: dict,
    role: Optional[str] = None,
) -> dict:
    """CAS a processed (frames, ch) float32 array as WAV and return a wet `audio` frame.

    ``role`` overrides the default ``<role>.wet`` — used by ops that emit SEVERAL wet frames
    from one source (e.g. ``variants``) and so need a distinct role per output.
    """
    import numpy as np
    import soundfile as sf

    from smplstream import cas, frames as F

    arr = np.ascontiguousarray(np.asarray(samples, dtype="float32"))
    if arr.ndim == 1:
        arr = arr[:, None]
    # WAV back-patches its RIFF size header → needs a seekable sink; render to memory.
    buf = io.BytesIO()
    sf.write(buf, arr, sr, format="WAV", subtype="FLOAT")
    h = cas.put_audio_bytes(buf.getvalue())
    meta = cas.read_meta(h) or {}
    return F.audio_frame(
        h,
        sr=meta.get("sr", sr),
        ch=meta.get("ch", arr.shape[1]),
        dur=meta.get("dur", arr.shape[0] / sr if sr else 0.0),
        role=role or _wet_role(src_frame),
        of=src_frame.get("id"),
        lineage=[src_frame["id"]] if src_frame.get("id") else None,
        op=op,
        op_version=op_version,
        params=params,
        fmt=meta.get("fmt"),
    )


def _sox_version_fingerprint() -> str:
    from smplstream import memo

    return memo.tool_version_fingerprint(["sox", "--version"])


# ---------------------------------------------------------------------------
# filter — high/low/band-pass via scipy Butterworth (deterministic, pure-Python).
# ---------------------------------------------------------------------------
def apply_filter(
    audio_frame: dict,
    *,
    kind: str,
    freq,
    order: int = 4,
) -> dict:
    """Apply an HP/LP/BP Butterworth filter, returning a wet `audio` frame.

    ``kind`` ∈ {"hp", "lp", "bp"}. For ``bp``, ``freq`` is a ``(low_hz, high_hz)`` pair;
    otherwise a single cutoff in Hz. Zero-phase (``filtfilt``) so no group-delay smear.
    """
    import numpy as np
    from scipy.signal import butter, filtfilt, sosfiltfilt

    data, sr = _load_audio(audio_frame)
    nyq = sr / 2.0

    if kind == "bp":
        low, high = float(freq[0]), float(freq[1])
        wn = [max(low / nyq, 1e-6), min(high / nyq, 0.999999)]
        sos = butter(order, wn, btype="bandpass", output="sos")
        out = sosfiltfilt(sos, data, axis=0)
        params = {"kind": "bp", "low_hz": low, "high_hz": high, "order": order, "sr_hz": sr}
    else:
        cutoff = float(freq)
        btype = {"hp": "highpass", "lp": "lowpass"}[kind]
        wn = min(max(cutoff / nyq, 1e-6), 0.999999)
        b, a = butter(order, wn, btype=btype)
        out = filtfilt(b, a, data, axis=0)
        params = {"kind": kind, "freq_hz": cutoff, "order": order, "sr_hz": sr}

    out = np.clip(out, -1.0, 1.0).astype("float32")
    return _emit_wet_audio(
        out, sr, src_frame=audio_frame, op="filter", op_version=FILTER_OP_VERSION, params=params
    )


# ---------------------------------------------------------------------------
# eq — peaking / shelving bands via scipy biquad (RBJ cookbook coefficients).
# ---------------------------------------------------------------------------
def _biquad_peaking(f0: float, q: float, gain_db: float, sr: int):
    import numpy as np

    a_amp = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / sr
    alpha = np.sin(w0) / (2.0 * q)
    cos_w0 = np.cos(w0)
    b0 = 1 + alpha * a_amp
    b1 = -2 * cos_w0
    b2 = 1 - alpha * a_amp
    a0 = 1 + alpha / a_amp
    a1 = -2 * cos_w0
    a2 = 1 - alpha / a_amp
    return np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0])


def _biquad_shelf(f0: float, gain_db: float, sr: int, *, high: bool, slope: float = 1.0):
    import numpy as np

    a_amp = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / sr
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / 2.0 * np.sqrt((a_amp + 1 / a_amp) * (1 / slope - 1) + 2)
    two_sqrt_a_alpha = 2 * np.sqrt(a_amp) * alpha
    if high:
        b0 = a_amp * ((a_amp + 1) + (a_amp - 1) * cos_w0 + two_sqrt_a_alpha)
        b1 = -2 * a_amp * ((a_amp - 1) + (a_amp + 1) * cos_w0)
        b2 = a_amp * ((a_amp + 1) + (a_amp - 1) * cos_w0 - two_sqrt_a_alpha)
        a0 = (a_amp + 1) - (a_amp - 1) * cos_w0 + two_sqrt_a_alpha
        a1 = 2 * ((a_amp - 1) - (a_amp + 1) * cos_w0)
        a2 = (a_amp + 1) - (a_amp - 1) * cos_w0 - two_sqrt_a_alpha
    else:
        b0 = a_amp * ((a_amp + 1) - (a_amp - 1) * cos_w0 + two_sqrt_a_alpha)
        b1 = 2 * a_amp * ((a_amp - 1) - (a_amp + 1) * cos_w0)
        b2 = a_amp * ((a_amp + 1) - (a_amp - 1) * cos_w0 - two_sqrt_a_alpha)
        a0 = (a_amp + 1) + (a_amp - 1) * cos_w0 + two_sqrt_a_alpha
        a1 = -2 * ((a_amp - 1) + (a_amp + 1) * cos_w0)
        a2 = (a_amp + 1) + (a_amp - 1) * cos_w0 - two_sqrt_a_alpha
    return np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0])


def apply_eq(audio_frame: dict, *, bands: list[dict]) -> dict:
    """Apply a chain of EQ bands, returning a wet `audio` frame.

    Each band: ``{"type": "peaking"|"lowshelf"|"highshelf", "freq": Hz, "gain": dB,
    "q": float}`` (``q`` used by peaking; shelves use a unit slope).
    """
    import numpy as np
    from scipy.signal import lfilter

    data, sr = _load_audio(audio_frame)
    out = data.astype("float64", copy=True)
    norm_bands = []
    for band in bands:
        btype = band.get("type", "peaking")
        f0 = float(band["freq"])
        gain_db = float(band.get("gain", 0.0))
        q = float(band.get("q", 1.0))
        if btype == "peaking":
            b, a = _biquad_peaking(f0, q, gain_db, sr)
        elif btype == "lowshelf":
            b, a = _biquad_shelf(f0, gain_db, sr, high=False)
        elif btype == "highshelf":
            b, a = _biquad_shelf(f0, gain_db, sr, high=True)
        else:
            raise ValueError(f"unknown eq band type: {btype!r}")
        out = lfilter(b, a, out, axis=0)
        norm_bands.append({"type": btype, "freq_hz": f0, "gain_db": gain_db, "q": q})

    out = np.clip(out, -1.0, 1.0).astype("float32")
    params = {"bands": norm_bands, "sr_hz": sr}
    return _emit_wet_audio(
        out, sr, src_frame=audio_frame, op="eq", op_version=EQ_OP_VERSION, params=params
    )


# ---------------------------------------------------------------------------
# env — amplitude envelope (pluck / fade / gate), deterministic numpy.
# ---------------------------------------------------------------------------
def apply_env(
    audio_frame: dict,
    *,
    shape: str,
    attack: float = 0.0,
    release: float = 0.0,
    threshold_db: float = -40.0,
) -> dict:
    """Apply an amplitude envelope, returning a wet `audio` frame.

    ``shape`` ∈ {"pluck", "fade", "gate"}:
      - ``pluck`` — fast linear attack (``attack`` s) then exponential decay over ``release`` s.
      - ``fade``  — linear fade-in (``attack`` s) and fade-out (``release`` s).
      - ``gate``  — silence samples whose short-term level is below ``threshold_db`` (dBFS).
    """
    import numpy as np

    data, sr = _load_audio(audio_frame)
    n = data.shape[0]
    t = np.arange(n) / sr if sr else np.arange(n)

    if shape == "pluck":
        gain = np.ones(n, dtype="float64")
        a = max(int(attack * sr), 1)
        gain[:a] = np.linspace(0.0, 1.0, a)
        if release > 0:
            tau = release / 5.0  # ~5 time-constants to ≈0 over the release window
            decay = np.exp(-(t - t[a - 1]) / tau)
            decay[:a] = 1.0
            gain = gain * decay
        params = {"shape": "pluck", "attack_s": attack, "release_s": release, "sr_hz": sr}
    elif shape == "fade":
        gain = np.ones(n, dtype="float64")
        a = min(max(int(attack * sr), 0), n)
        r = min(max(int(release * sr), 0), n)
        if a > 0:
            gain[:a] = np.linspace(0.0, 1.0, a)
        if r > 0:
            gain[n - r:] = np.linspace(1.0, 0.0, r)
        params = {"shape": "fade", "attack_s": attack, "release_s": release, "sr_hz": sr}
    elif shape == "gate":
        mono = data.mean(axis=1)
        win = max(int(0.01 * sr), 1)  # 10 ms RMS window
        kernel = np.ones(win) / win
        env = np.sqrt(np.convolve(mono ** 2, kernel, mode="same") + 1e-12)
        thresh_lin = 10.0 ** (threshold_db / 20.0)
        gain = (env >= thresh_lin).astype("float64")
        params = {"shape": "gate", "threshold_db": threshold_db, "sr_hz": sr}
    else:
        raise ValueError(f"unknown env shape: {shape!r}")

    out = (data * gain[:, None]).astype("float32")
    return _emit_wet_audio(
        out, sr, src_frame=audio_frame, op="env", op_version=ENV_OP_VERSION, params=params
    )


# ---------------------------------------------------------------------------
# fx — reverb / delay via sox (shell-out; env-fingerprinted tool version).
# ---------------------------------------------------------------------------
def apply_fx(
    audio_frame: dict,
    *,
    effect: str,
    amount: float = 50.0,
    delay_ms: float = 250.0,
    decay: float = 0.5,
) -> dict:
    """Apply a sox-driven effect, returning a wet `audio` frame.

    ``effect`` ∈ {"reverb", "delay"}. ``reverb`` uses sox ``reverb <amount>`` (0–100).
    ``delay`` uses sox ``echo`` with one tap at ``delay_ms`` / ``decay``.
    """
    from smplstream import cas

    src = cas.get_path(audio_frame["hash"])
    if effect == "reverb":
        chain = ["reverb", str(float(amount))]
        params = {"effect": "reverb", "amount": float(amount)}
    elif effect == "delay":
        # sox echo: gain-in gain-out <delay_ms decay> ...
        chain = ["echo", "0.8", "0.9", str(float(delay_ms)), str(float(decay))]
        params = {"effect": "delay", "delay_ms": float(delay_ms), "decay": float(decay)}
    else:
        raise ValueError(f"unknown fx effect: {effect!r}")

    # Render float32 WAV to stdout so we never silently truncate bit depth.
    cmd = ["sox", str(src), "-t", "wav", "-e", "floating-point", "-b", "32", "-", *chain]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"sox {effect} failed: {proc.stderr.decode('utf-8', 'replace').strip()}")

    h = cas.put_audio_bytes(proc.stdout)
    meta = cas.read_meta(h) or {}

    from smplstream import frames as F

    params["env_fingerprint"] = _sox_version_fingerprint()
    return F.audio_frame(
        h,
        sr=meta.get("sr", 0),
        ch=meta.get("ch", 1),
        dur=meta.get("dur", 0.0),
        role=_wet_role(audio_frame),
        of=audio_frame.get("id"),
        lineage=[audio_frame["id"]] if audio_frame.get("id") else None,
        op="fx",
        op_version=FX_OP_VERSION,
        params=params,
        fmt=meta.get("fmt"),
    )


# ---------------------------------------------------------------------------
# gain / normalize / limit — level management (ticket vault-3l83 follow-up).
#
# The level-management trio. All deterministic numpy (empty env-fingerprint), all emit a
# wet ``<role>.wet`` audio frame carrying the measured-before / applied-gain in ``params`` so
# the level decision is auditable from the lineage. They compose like every other op — the
# canonical leveling chain is ``read | normalize --lufs -14 | limit | write``:
#
#   - ``gain``      — the primitive: a pure dB scale, NO clipping (float-safe, composable;
#                     pair with ``limit`` for a ceiling). The single-knob building block.
#   - ``normalize`` — LUFS-normalize to a target (BS.1770 integrated), WITH a true-peak
#                     ceiling so it is safe standalone AND composes. Trades exact-LUFS for
#                     no-clipping when the target would breach the ceiling.
#   - ``limit``     — a true-peak ceiling by whole-sample gain reduction (never boosts). Not
#                     a look-ahead compressor: transparent, no pumping, preserves dynamics —
#                     the right primitive for one-shot / sample prep.
# ---------------------------------------------------------------------------
def apply_gain(audio_frame: dict, *, db: float) -> dict:
    """Scale the selected audio frame by ``db`` decibels, returning a wet `audio` frame.

    A pure level change — deliberately **not** clipped, so it is lossless and composes (a
    boost that exceeds 0 dBFS survives in the float CAS blob; follow with ``limit`` for a
    delivery ceiling). The single-knob primitive ``normalize`` is built on.
    """
    import numpy as np

    data, sr = _load_audio(audio_frame)
    factor = 10.0 ** (float(db) / 20.0)
    out = (data.astype("float64") * factor).astype("float32")
    params = {"db": float(db), "sr_hz": sr}
    return _emit_wet_audio(
        out, sr, src_frame=audio_frame, op="gain", op_version=GAIN_OP_VERSION, params=params
    )


def apply_normalize(
    audio_frame: dict,
    *,
    target_lufs: float,
    ceiling_dbtp: Optional[float] = -1.0,
) -> dict:
    """Loudness-normalize to ``target_lufs`` (BS.1770 integrated), returning a wet frame.

    Computes the gain that moves measured integrated LUFS to ``target_lufs``. If applying
    that gain would push the true peak above ``ceiling_dbtp`` (default −1 dBTP), the gain is
    pulled back so the true peak lands exactly at the ceiling — i.e. the op self-limits,
    trading a little loudness for no inter-sample clipping. Pass ``ceiling_dbtp=None`` to
    normalize to exact LUFS with no ceiling (then chain ``limit`` yourself). Silent / too-
    short signals pass through ungained with a ``note`` in params.

    The measured-in loudness, true peak, and applied gain are recorded in ``params`` so the
    level decision is fully auditable from the frame lineage.
    """
    import numpy as np

    from . import loudness

    data, sr = _load_audio(audio_frame)
    res = loudness.analyze_array(data, sr)
    measured_lufs = res["integrated_lufs"]
    measured_tp = res["true_peak_dbtp"]

    note = None
    if measured_lufs is None or not math.isfinite(measured_lufs):
        gain_db = 0.0
        note = "silent_or_too_short: passed through ungained"
    else:
        gain_db = float(target_lufs) - float(measured_lufs)

    ceiling_applied = False
    if (
        ceiling_dbtp is not None
        and measured_tp is not None
        and math.isfinite(measured_tp)
    ):
        projected_tp = measured_tp + gain_db
        if projected_tp > ceiling_dbtp:
            gain_db -= projected_tp - ceiling_dbtp
            ceiling_applied = True

    factor = 10.0 ** (gain_db / 20.0)
    out = (data.astype("float64") * factor).astype("float32")
    params = {
        "target_lufs": float(target_lufs),
        "ceiling_dbtp": ceiling_dbtp,
        "measured_lufs_in": loudness._db_round(measured_lufs),
        "measured_true_peak_dbtp_in": loudness._db_round(measured_tp),
        "applied_gain_db": round(float(gain_db), 3),
        "ceiling_applied": ceiling_applied,
        "sr_hz": sr,
    }
    if note:
        params["note"] = note
    return _emit_wet_audio(
        out, sr, src_frame=audio_frame, op="normalize",
        op_version=NORMALIZE_OP_VERSION, params=params,
    )


def apply_limit(audio_frame: dict, *, ceiling_dbtp: float = -1.0) -> dict:
    """Guarantee true peak ≤ ``ceiling_dbtp`` by whole-sample gain reduction (never boosts).

    Measures the 4×-oversampled true peak and, if it exceeds the ceiling, scales the entire
    sample down so the worst inter-sample peak sits exactly at the ceiling. A transparent,
    deterministic ceiling — not a look-ahead compressor — so it never pumps and preserves
    the sample's dynamics. Below-ceiling input passes through unchanged (gain 0). The right
    safety stage for one-shots and for the tail of a normalize chain.
    """
    import numpy as np

    from . import loudness

    data, sr = _load_audio(audio_frame)
    res = loudness.analyze_array(data, sr)
    measured_tp = res["true_peak_dbtp"]

    gain_db = 0.0
    if measured_tp is not None and math.isfinite(measured_tp) and measured_tp > ceiling_dbtp:
        gain_db = float(ceiling_dbtp) - float(measured_tp)  # always ≤ 0

    factor = 10.0 ** (gain_db / 20.0)
    out = (data.astype("float64") * factor).astype("float32")
    params = {
        "ceiling_dbtp": float(ceiling_dbtp),
        "measured_true_peak_dbtp_in": loudness._db_round(measured_tp),
        "applied_gain_db": round(float(gain_db), 3),
        "sr_hz": sr,
    }
    return _emit_wet_audio(
        out, sr, src_frame=audio_frame, op="limit", op_version=LIMIT_OP_VERSION, params=params
    )


# ---------------------------------------------------------------------------
# compress — feed-forward downward compressor (peak-linked RMS-smoothed detect, soft knee).
#
# The crest-reduction primitive the level trio (gain/normalize/limit) lacked: limit only
# pulls peaks to a ceiling; compress reduces the peak-to-RMS gap so a later ``normalize`` can
# push the loudness UP without breaching the ceiling — i.e. it's how you get a loop/bus to a
# dense reference's level (reference-targets.md loudness parity). Detector = peak-across-channels,
# RMS-time-smoothed by a one-pole at the release tc; soft-knee static curve + attack-smoothed
# gain, fully vectorized scipy/numpy (deterministic, empty env-fingerprint). NOTE: this is NOT a
# look-ahead brickwall limiter — the RMS-smoothed detector lets fast transients through, so it
# reduces dynamic range / glues but does not maximize loudness (use a brickwall for that,
# vault-36pa). Canonical chain: ``compress | normalize``.
# ---------------------------------------------------------------------------
def apply_compress(
    audio_frame: dict,
    *,
    threshold_db: float = -18.0,
    ratio: float = 3.0,
    attack_ms: float = 10.0,
    release_ms: float = 120.0,
    knee_db: float = 6.0,
    makeup_db: float = 0.0,
) -> dict:
    """Downward-compress the selected audio, returning a wet `audio` frame.

    A feed-forward compressor: a peak-across-channels detector, RMS-time-smoothed by a one-pole at
    ``release_ms``, feeds a soft-knee static curve (``threshold_db`` / ``ratio`` / ``knee_db``);
    the resulting gain reduction is smoothed at ``attack_ms`` and applied with ``makeup_db`` of
    make-up gain. Pair with ``normalize`` to raise loudness toward a dense reference (it reduces
    crest so normalize can push further before the true-peak ceiling) — though full loudness
    maximization needs a look-ahead brickwall (vault-36pa). The one-poles start from zero ICs, so
    the detector settles over ~``release_ms`` from the start. Fully vectorized + deterministic; the
    mean/max gain reduction is recorded in ``params``.
    """
    import numpy as np
    from scipy.signal import lfilter

    if ratio < 1.0:
        raise ValueError(f"ratio must be >= 1 (got {ratio})")
    if knee_db < 0:
        raise ValueError(f"knee_db must be >= 0 (got {knee_db})")
    if attack_ms < 0 or release_ms < 0:
        raise ValueError(f"attack_ms/release_ms must be >= 0 (got {attack_ms}/{release_ms})")

    data, sr = _load_audio(audio_frame)
    x = data.astype("float64")
    if x.shape[0] == 0:  # empty slice → passthrough with a note (mirror apply_widen)
        params = {"threshold_db": float(threshold_db), "ratio": float(ratio),
                  "note": "empty input: passthrough", "sr_hz": sr}
        return _emit_wet_audio(x.astype("float32"), sr, src_frame=audio_frame,
                               op="compress", op_version=COMPRESS_OP_VERSION, params=params)
    det = np.max(np.abs(x), axis=1) if x.ndim > 1 else np.abs(x)  # peak-across-channels detector

    # RMS detection: one-pole on the squared detector at the release time constant.
    a_rel = float(np.exp(-1.0 / (sr * max(release_ms, 1e-3) / 1000.0)))
    ms = lfilter([1.0 - a_rel], [1.0, -a_rel], det ** 2)
    env_db = 10.0 * np.log10(np.maximum(ms, 1e-12))

    # Soft-knee static curve → gain reduction (<= 0 dB).
    over = env_db - threshold_db
    half = knee_db / 2.0
    gr = np.zeros_like(env_db)
    above = over >= half
    gr[above] = (1.0 / ratio - 1.0) * (over[above])
    if knee_db > 0:
        ink = (over > -half) & (over < half)
        gr[ink] = (1.0 / ratio - 1.0) * (over[ink] + half) ** 2 / (2.0 * knee_db)

    # Attack smoothing of the gain-reduction signal (one-pole at the attack tc).
    a_att = float(np.exp(-1.0 / (sr * max(attack_ms, 1e-3) / 1000.0)))
    gr_s = lfilter([1.0 - a_att], [1.0, -a_att], gr)

    gain_lin = 10.0 ** ((gr_s + float(makeup_db)) / 20.0)
    out = x * (gain_lin[:, None] if x.ndim > 1 else gain_lin)
    out = np.clip(out, -1.0, 1.0).astype("float32")
    params = {
        "threshold_db": float(threshold_db), "ratio": float(ratio),
        "attack_ms": float(attack_ms), "release_ms": float(release_ms),
        "knee_db": float(knee_db), "makeup_db": float(makeup_db),
        "mean_gain_reduction_db": round(float(np.mean(gr_s)), 3),
        "max_gain_reduction_db": round(float(np.min(gr_s)), 3),
        "sr_hz": sr,
    }
    return _emit_wet_audio(
        out, sr, src_frame=audio_frame, op="compress", op_version=COMPRESS_OP_VERSION, params=params
    )


# ---------------------------------------------------------------------------
# slice — librosa onset detection → marker frame (+ optional sliced audio frames).
# ---------------------------------------------------------------------------
def slice_onsets(
    audio_frame: dict,
    *,
    emit_audio: bool = False,
    backtrack: bool = True,
) -> list[dict]:
    """Detect onsets and return a `marker` frame; optionally one sliced `audio` per region.

    The marker frame (role ``onset``) carries one point per onset with float-second ``t`` and
    sample-accurate ``sample`` (spec → *Units & timebase*: markers destined for sample-exact
    export MUST carry ``sample``). When ``emit_audio`` is set, each inter-onset region is also
    CASed and emitted as an ``audio`` frame with role ``slice:<n>``.
    """
    import librosa
    import numpy as np

    from smplstream import frames as F

    data, sr = _load_audio(audio_frame)
    mono = data.mean(axis=1) if data.ndim > 1 else data

    onset_samples = librosa.onset.onset_detect(
        y=mono, sr=sr, backtrack=backtrack, units="samples"
    )
    onset_samples = [int(s) for s in onset_samples]

    points = [
        {"t": round(s / sr, 6), "sample": s, "label": f"onset-{i}"}
        for i, s in enumerate(onset_samples)
    ]
    out: list[dict] = [
        F.marker_frame(
            points,
            role="onset",
            of=audio_frame.get("id"),
            op="slice",
            op_version=SLICE_OP_VERSION,
            lineage=[audio_frame["id"]] if audio_frame.get("id") else None,
            params={"backtrack": backtrack, "emit_audio": emit_audio, "sr_hz": sr},
        )
    ]

    if emit_audio and onset_samples:
        import io as _io

        import soundfile as sf

        from smplstream import cas

        bounds = onset_samples + [data.shape[0]]
        for i in range(len(onset_samples)):
            start, end = bounds[i], bounds[i + 1]
            if end <= start:
                continue
            region = np.ascontiguousarray(data[start:end], dtype="float32")
            buf = _io.BytesIO()
            sf.write(buf, region, sr, format="WAV", subtype="FLOAT")
            h = cas.put_audio_bytes(buf.getvalue())
            meta = cas.read_meta(h) or {}
            out.append(
                F.audio_frame(
                    h,
                    sr=meta.get("sr", sr),
                    ch=meta.get("ch", region.shape[1] if region.ndim > 1 else 1),
                    dur=meta.get("dur", (end - start) / sr if sr else 0.0),
                    role=f"slice:{i}",
                    of=audio_frame.get("id"),
                    lineage=[audio_frame["id"]] if audio_frame.get("id") else None,
                    op="slice",
                    op_version=SLICE_OP_VERSION,
                    params={"index": i, "start_sample": start, "end_sample": end, "sr_hz": sr},
                    fmt=meta.get("fmt"),
                )
            )
    return out


# ---------------------------------------------------------------------------
# widen — mid-side stereo widening above a crossover (reference-match family).
#
# Reference masters carry stereo width in the upper-mids/highs while the low end
# stays mono (so the kick/sub translate on a club system). This widens by scaling
# the SIDE channel (S = (L−R)/2) only ABOVE a crossover, leaving everything below
# untouched — so the bass stays exactly as-is (mono-safe) and only the air opens up.
# Deterministic scipy (empty env-fingerprint), composes like every other edit op.
# ---------------------------------------------------------------------------
def apply_widen(
    audio_frame: dict,
    *,
    side_gain_db: float = 3.0,
    crossover_hz: float = 200.0,
    order: int = 4,
) -> dict:
    """Widen the stereo image above ``crossover_hz`` by ``side_gain_db``, returning a wet frame.

    Splits the side channel at the crossover (zero-phase Butterworth), scales only the high
    band by ``side_gain_db``, recombines: ``L' = M + S'``, ``R' = M − S'``. Below the crossover
    the side is untouched, so low-frequency mono-compatibility is preserved. A negative
    ``side_gain_db`` narrows (toward mono). Mono input has no side channel → passthrough with a
    ``note``. The amount lives in dB so it composes with the rest of the level vocabulary.
    """
    import numpy as np
    from scipy.signal import butter, sosfiltfilt

    data, sr = _load_audio(audio_frame)
    if data.shape[1] < 2:
        out = data.astype("float32")
        params = {
            "side_gain_db": float(side_gain_db),
            "crossover_hz": float(crossover_hz),
            "note": "mono input: no side channel to widen (passthrough)",
            "sr_hz": sr,
        }
        return _emit_wet_audio(
            out, sr, src_frame=audio_frame, op="widen", op_version=WIDEN_OP_VERSION, params=params
        )

    # sosfiltfilt needs more samples than its pad length; a tiny/empty slice can't be filtered
    # (it would raise) — pass it through, mirroring the mono branch.
    min_len = 3 * (2 * order + 1) + 1
    if data.shape[0] < min_len:
        out = data.astype("float32")
        params = {
            "side_gain_db": float(side_gain_db),
            "crossover_hz": float(crossover_hz),
            "note": f"input too short to widen (< {min_len} samples; passthrough)",
            "sr_hz": sr,
        }
        return _emit_wet_audio(
            out, sr, src_frame=audio_frame, op="widen", op_version=WIDEN_OP_VERSION, params=params
        )

    left = data[:, 0].astype("float64")
    right = data[:, 1].astype("float64")
    mid = (left + right) / 2.0
    side = (left - right) / 2.0

    nyq = sr / 2.0
    wn = min(max(crossover_hz / nyq, 1e-6), 0.999999)
    sos_hp = butter(order, wn, btype="highpass", output="sos")
    side_high = sosfiltfilt(sos_hp, side)

    factor = 10.0 ** (float(side_gain_db) / 20.0)
    # Additive boost of ONLY the high band: below the crossover ``side_high`` ≈ 0, so the side
    # (and thus the low-frequency mono balance) is left intact; above it the side is ×factor.
    side_new = side + (factor - 1.0) * side_high
    out_l = mid + side_new
    out_r = mid - side_new

    out = np.clip(np.stack([out_l, out_r], axis=1), -1.0, 1.0).astype("float32")
    params = {
        "side_gain_db": float(side_gain_db),
        "crossover_hz": float(crossover_hz),
        "order": int(order),
        "sr_hz": sr,
    }
    return _emit_wet_audio(
        out, sr, src_frame=audio_frame, op="widen", op_version=WIDEN_OP_VERSION, params=params
    )


def apply_mono(audio_frame: dict) -> dict:
    """Downmix to ONE channel by averaging the channels (sox ``channels 1``).

    The narrow end of the stereo-image vocabulary and the bass bus's glue: averaging is
    level-neutral for correlated material (identical channels come back at the same
    amplitude) and it *guarantees* mono, which scaling the side channel (``widen`` with a
    big negative gain) only ever approaches. Mono input passes through unchanged.
    """
    import numpy as np

    data, sr = _load_audio(audio_frame)
    ch_in = int(data.shape[1])
    out = np.ascontiguousarray(data.astype("float64").mean(axis=1, keepdims=True), dtype="float32")
    params = {"channels_in": ch_in, "channels_out": 1, "sr_hz": sr}
    return _emit_wet_audio(
        out, sr, src_frame=audio_frame, op="mono", op_version=MONO_OP_VERSION, params=params
    )


# ---------------------------------------------------------------------------
# spectral-match — EQ the source toward a reference's spectral BALANCE.
#
# The reference-replication linchpin (reference-targets.md): measure a reference's
# average per-band spectrum, measure the source's, and apply a bounded corrective
# peaking-EQ chain that moves the source's *balance* toward the reference's. It
# matches SHAPE, not level — both curves are mean-normalized first, so loudness is
# left to `normalize` (you don't want spectral-match also yanking the gain). A
# `strength` (0..1) does partial matches, `max_correction_db` clamps each band, and
# `protect_below_hz` leaves the sub alone (the references rarely want *more* sub).
# ---------------------------------------------------------------------------
def _log_bands(lo_hz: float, hi_hz: float, n: int) -> list[tuple]:
    """``n`` log-spaced (geometric) band edges → list of ``(low_hz, high_hz)`` pairs."""
    import numpy as np

    edges = np.geomspace(max(lo_hz, 1.0), hi_hz, n + 1)
    return [(float(edges[i]), float(edges[i + 1])) for i in range(n)]


# Shared spectral floor (dB). Used for both too-short signals AND empty bands so the
# mean-normalization in spectral-match never sees a -200 dB outlier that biases every band.
_SPECTRUM_FLOOR_DB = -120.0


def _band_power_db(mono, sr: int, bands: list) -> "object":
    """Per-band power (dB) of a mono signal via one windowed rFFT (Parseval).

    Signals shorter than the analysis window (n < 4) and bands with no FFT bins both read the
    shared ``_SPECTRUM_FLOOR_DB`` — a single floor so neither can skew the normalization mean.
    """
    import numpy as np

    n = len(mono)
    if n < 4:
        return np.full(len(bands), _SPECTRUM_FLOOR_DB)
    spec_pow = np.abs(np.fft.rfft(mono * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    out = []
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        p = float(np.sum(spec_pow[mask])) if mask.any() else 0.0
        out.append(max(10.0 * np.log10(max(p, 1e-20)), _SPECTRUM_FLOOR_DB))
    return np.array(out)


def _peaking_q_for_octaves(bw_oct: float) -> float:
    """RBJ peaking Q for a bandwidth in octaves (≈ one band wide so neighbors blend; the −3 dB
    bandwidth is not an exact partition, so adjacent bands overlap/ripple a little)."""
    import numpy as np

    if bw_oct <= 0:
        return 1.0
    return float(np.sqrt(2 ** bw_oct) / (2 ** bw_oct - 1))


def apply_spectral_match(
    audio_frame: dict,
    *,
    reference_path: str,
    strength: float = 1.0,
    max_correction_db: float = 6.0,
    n_bands: int = 12,
    lo_hz: float = 30.0,
    hi_hz: float = 16000.0,
    protect_below_hz: float = 60.0,
) -> dict:
    """Move the source's spectral balance toward ``reference_path``'s, returning a wet frame.

    Measures both signals' per-band power over an ``n_bands`` log-spaced grid, mean-normalizes
    each (so absolute loudness is ignored — that is ``normalize``'s job), and applies a chain of
    peaking bands whose gains are ``strength × (ref − src)`` clamped to ``±max_correction_db``.
    Bands centered below ``protect_below_hz`` are forced to 0 dB so the sub/kick foundation is
    left intact. The full corrective curve is recorded in ``params`` for auditability, together
    with the A/B spectra (``spectrum_db``: the target before, the reference, the target after —
    all mean-normalized on the same band grid) and the ``residual_db`` before/after distances
    to the reference, so a consumer can see what the match actually did.
    """
    import numpy as np
    import soundfile as sf
    from scipy.signal import lfilter

    if n_bands < 1:
        raise ValueError(f"n_bands must be >= 1 (got {n_bands})")
    if max_correction_db < 0:
        raise ValueError(f"max_correction_db must be >= 0 (got {max_correction_db})")

    data, sr = _load_audio(audio_frame)
    src_mono = data.mean(axis=1) if data.ndim > 1 else data

    ref, ref_sr = sf.read(str(reference_path), dtype="float64", always_2d=True)
    ref_mono = ref.mean(axis=1)

    # The band grid must stay below BOTH Nyquists: a band above the reference's Nyquist has no
    # bins (a floor outlier that would bias the normalization mean), and a band center above the
    # source's Nyquist would build a degenerate biquad. Clamp the high edge to just under the
    # lower of the two Nyquists — this is what makes cross-sample-rate references safe.
    eff_hi = min(float(hi_hz), 0.49 * sr, 0.49 * float(ref_sr))
    if eff_hi <= lo_hz:
        raise ValueError(
            f"band grid empty: lo_hz {lo_hz} >= effective hi {eff_hi:.0f} "
            f"(src sr {sr}, ref sr {int(ref_sr)})")
    bands = _log_bands(lo_hz, eff_hi, n_bands)
    src_db = _band_power_db(src_mono, sr, bands)
    ref_db = _band_power_db(ref_mono, int(ref_sr), bands)

    # Match BALANCE not level: remove each curve's mean before differencing.
    src_n = src_db - float(np.mean(src_db))
    ref_n = ref_db - float(np.mean(ref_db))
    delta = float(strength) * (ref_n - src_n)
    delta = np.clip(delta, -float(max_correction_db), float(max_correction_db))

    eq_bands = []
    centers = []
    applied = []
    out = data.astype("float64", copy=True)
    for (lo, hi), g in zip(bands, delta):
        fc = float(np.sqrt(lo * hi))
        # protect the sub, and never build a biquad at/above the source Nyquist (degenerate)
        gain = 0.0 if (fc < protect_below_hz or fc >= 0.5 * sr) else float(g)
        q = _peaking_q_for_octaves(float(np.log2(hi / lo)))
        centers.append(round(fc, 1))
        applied.append(round(gain, 2))
        if abs(gain) > 1e-3:
            b, a = _biquad_peaking(fc, q, gain, sr)
            out = lfilter(b, a, out, axis=0)
        eq_bands.append({"freq_hz": round(fc, 1), "gain_db": round(gain, 2), "q": round(q, 3)})

    out = np.clip(out, -1.0, 1.0).astype("float32")

    # A/B: measure the OUTPUT on the SAME band grid, mean-normalized like the other two curves,
    # so a consumer can read the target's spectrum before vs after against the reference without
    # re-analysing the audio. `residual_db` scores the distance to the reference (RMS over bands,
    # dB) before and after — the single number that says whether the match helped.
    out_mono = out.mean(axis=1) if out.ndim > 1 else out
    out_db = _band_power_db(out_mono, sr, bands)
    out_n = out_db - float(np.mean(out_db))
    # Protected bands are pinned at 0 dB correction, so scoring them would dilute (understate)
    # the improvement; if EVERY band is protected, score them all rather than divide by zero.
    band_fcs = np.array([np.sqrt(lo * hi) for lo, hi in bands])
    mask = band_fcs >= float(protect_below_hz)
    if not mask.any():
        mask = np.ones(len(bands), dtype=bool)

    def _rms(curve):
        return round(float(np.sqrt(np.mean(np.asarray(curve)[mask] ** 2))), 6)

    params = {
        "reference": os.path.basename(str(reference_path)),
        "strength": float(strength),
        "max_correction_db": float(max_correction_db),
        "n_bands": int(n_bands),
        "protect_below_hz": float(protect_below_hz),
        "hi_hz_effective": round(eff_hi, 1),
        "ref_sr_hz": int(ref_sr),
        "band_centers_hz": centers,
        "correction_db": applied,
        "bands": eq_bands,
        # band order is `band_centers_hz`; 6 dp keeps each curve's mean at 0 within 1e-6
        # (2 dp, the house rounding for a single gain, would bias a 12-band mean by ~1e-3).
        "spectrum_db": {
            "source": [round(float(v), 6) for v in src_n],
            "reference": [round(float(v), 6) for v in ref_n],
            "matched": [round(float(v), 6) for v in out_n],
        },
        "residual_db": {"before": _rms(ref_n - src_n), "after": _rms(ref_n - out_n)},
        "sr_hz": sr,
    }
    return _emit_wet_audio(
        out, sr, src_frame=audio_frame, op="spectral-match",
        op_version=SPECTRAL_MATCH_OP_VERSION, params=params,
    )


# ---------------------------------------------------------------------------
# automate — time-varying parameter modulation (LFO / breakpoint envelopes).
#
# The "movement" op: modulate gain (tremolo / sidechain pump), pan (auto-pan), or a
# resonant low-pass cutoff (filter sweep / acid) over time, with an LFO shape OR an
# arbitrary breakpoint curve, optionally tempo-synced (caller converts beats/Hz → cycles).
# Static loops read as loops; automation is what makes a section evolve. Pure numpy/scipy
# (deterministic, no shell-out). cutoff is a per-block resonant RBJ biquad with state
# carried across coefficient changes (smooth for musical sweep rates).
# ---------------------------------------------------------------------------
AUTOMATE_OP_VERSION = "automate@1"

_AUTOMATE_SHAPES = ("sine", "tri", "saw-up", "saw-down", "square", "points")
_AUTOMATE_TARGETS = ("gain", "pan", "cutoff")


def _automation_curve(n, *, shape, cycles, phase, duty, points):
    """Per-sample modulator in [0,1], length ``n``; ``cycles`` LFO periods across the clip.

    Phase-0 start value by shape: sine/tri/saw-up start at 0 (saw-up's 0→1 ramp reads as the
    classic post-kick sidechain recovery); saw-down and square start at 1.0. ``points`` linearly
    interpolates breakpoints ``(t, v)`` (t,v in 0..1) over one cycle — for multi-cycle use
    (cycles>1) the breakpoints should span [0,1] or the cycle wrap will step discontinuously.
    """
    import numpy as np

    if n <= 0:
        return np.zeros(0)
    ph = (np.arange(n) / float(n) * float(cycles) + float(phase)) % 1.0
    if shape == "points":
        pts = sorted((float(t), float(v)) for t, v in points)
        ts = np.array([p[0] for p in pts]); vs = np.array([p[1] for p in pts])
        v = np.interp(ph, ts, vs)            # np.interp clamps outside the breakpoint range
    elif shape == "sine":
        v = 0.5 - 0.5 * np.cos(2.0 * np.pi * ph)
    elif shape == "tri":
        v = 1.0 - np.abs(2.0 * ph - 1.0)
    elif shape == "saw-up":
        v = ph.copy()
    elif shape == "saw-down":
        v = 1.0 - ph
    elif shape == "square":
        v = (ph < float(duty)).astype("float64")
    else:
        raise ValueError(f"unknown automate shape: {shape!r}")
    return np.clip(v, 0.0, 1.0)


def _rbj_lowpass_ba(fc: float, q: float, sr: int):
    """RBJ cookbook resonant low-pass biquad coefficients ``(b, a)`` (q>0.707 peaks → acid)."""
    import numpy as np

    w0 = 2.0 * np.pi * fc / sr
    cw = np.cos(w0); sw = np.sin(w0)
    alpha = sw / (2.0 * max(q, 1e-4))
    b0 = (1.0 - cw) / 2.0; b1 = 1.0 - cw; b2 = (1.0 - cw) / 2.0
    a0 = 1.0 + alpha; a1 = -2.0 * cw; a2 = 1.0 - alpha
    return np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0])


def _sweep_lowpass(data, sr: int, cutoff, q: float, hop: int):
    """Block-wise resonant low-pass; filter state carries across per-block coefficient changes."""
    import numpy as np
    from scipy.signal import lfilter, lfilter_zi

    n, ch = data.shape
    out = np.empty((n, ch), dtype="float64")
    zi = None
    for start in range(0, n, max(hop, 1)):
        end = min(start + max(hop, 1), n)
        fc = float(np.clip(np.median(cutoff[start:end]), 20.0, 0.49 * sr))
        b, a = _rbj_lowpass_ba(fc, q, sr)
        if zi is None:
            zi = np.outer(lfilter_zi(b, a), data[0])   # (2, ch) steady-state init from sample 0
        block, zi = lfilter(b, a, data[start:end], axis=0, zi=zi)
        out[start:end] = block
    return out


def apply_automate(audio_frame: dict, *, target: str, shape: str = "sine", cycles: float = 1.0,
                   depth: float = 1.0, phase: float = 0.0, duty: float = 0.5, points=None,
                   lo_hz: float = 200.0, hi_hz: float = 8000.0, resonance: float = 0.707,
                   hop: int = 128) -> dict:
    """Modulate a parameter over time, returning a wet `audio` frame.

    ``target`` ∈ {"gain" (tremolo / sidechain pump), "pan" (auto-pan, mono→stereo),
    "cutoff" (resonant low-pass sweep — acid / filter movement)}.
    ``shape`` ∈ {sine, tri, saw-up, saw-down, square, points}. ``cycles`` = LFO periods across
    the clip (tempo-sync is the caller's job). ``depth`` 0..1 scales gain/pan ONLY (cutoff ignores
    it); the cutoff range is ``lo_hz``..``hi_hz`` (log-swept by the modulator, ``resonance`` = Q).
    """
    import numpy as np

    if target not in _AUTOMATE_TARGETS:
        raise ValueError(f"unknown automate target: {target!r}")
    if shape not in _AUTOMATE_SHAPES:
        raise ValueError(f"unknown automate shape: {shape!r}")
    if shape == "points" and not points:
        raise ValueError("shape 'points' requires at least one (t, v) breakpoint")
    if cycles <= 0:
        raise ValueError("cycles must be > 0")
    if not (0.0 <= depth <= 1.0):
        raise ValueError("depth must be in [0, 1]")

    data, sr = _load_audio(audio_frame)
    if data.shape[0] == 0:
        return _emit_wet_audio(data, sr, src_frame=audio_frame, op="automate",
                               op_version=AUTOMATE_OP_VERSION,
                               params={"target": target, "shape": shape, "noop": "empty"})
    n = data.shape[0]
    v = _automation_curve(n, shape=shape, cycles=cycles, phase=phase, duty=duty, points=points)

    params = {"target": target, "shape": shape, "cycles": round(float(cycles), 4),
              "depth": float(depth), "phase": float(phase), "sr_hz": sr}
    if shape == "square":
        params["duty"] = float(duty)
    if shape == "points":
        params["points"] = [[float(t), float(val)] for t, val in points]

    if target == "gain":
        mult = 1.0 - depth * (1.0 - v)             # v=1 → unity, v=0 → (1-depth)
        out = data * mult[:, None]
    elif target == "pan":
        if data.shape[1] == 1:
            data = np.repeat(data, 2, axis=1)
        pan = (2.0 * v - 1.0) * depth              # -depth..+depth
        ang = (pan + 1.0) * (np.pi / 4.0)         # 0..pi/2 → constant-power
        gl = np.sqrt(2.0) * np.cos(ang); gr = np.sqrt(2.0) * np.sin(ang)
        out = np.column_stack([data[:, 0] * gl, data[:, 1] * gr])
    else:  # cutoff
        eff_hi = min(float(hi_hz), 0.49 * sr)
        lo = max(float(lo_hz), 20.0)
        cutoff = lo * (eff_hi / lo) ** v          # log sweep lo..eff_hi by the modulator
        out = _sweep_lowpass(data, sr, cutoff, float(resonance), int(hop))
        params.pop("depth", None)                 # depth is gain/pan only — not used for cutoff
        params.update({"lo_hz": round(lo, 1), "hi_hz": round(eff_hi, 1),
                       "resonance": float(resonance)})

    out = np.clip(out, -1.0, 1.0).astype("float32")
    return _emit_wet_audio(out, sr, src_frame=audio_frame, op="automate",
                           op_version=AUTOMATE_OP_VERSION, params=params)


# ---------------------------------------------------------------------------
# variants — N STATIC filter renders across a log-spaced cutoff range (palette variants).
#
# A one-shot palette printed once reads dead: every hit is the same timbre, and with no live
# filter under your hands there is no closed→open, dark→bright gesture to play. `apply_automate
# (target="cutoff")` bakes that movement WITHIN one render; this bakes it ACROSS renders — the
# same sweep frozen at N points, which is the closed / muted / half / open / bright palette a
# sound designer would print by hand. No new DSP: each variant is the automate sweep held at a
# constant cutoff (same resonant RBJ low-pass, same steady-state init), so variant k IS the
# sweep paused at step k. Pure numpy/scipy, no RNG → same input + params ⇒ identical bytes.
# ---------------------------------------------------------------------------
VARIANTS_OP_VERSION = "variants@1"

MAX_VARIANT_STEPS = 64


def _variant_role(audio_frame: dict, index: int) -> str:
    """``<base>.variant:<k>`` (k is 1-based) — distinct per step so `select --role` can pick one."""
    base = _wet_role(audio_frame)
    base = base[: -len(".wet")] if base.endswith(".wet") else base
    return f"{base}.variant:{index}"


def variant_cutoffs(lo_hz: float, hi_hz: float, steps: int) -> list[float]:
    """The log-spaced cutoff ladder ``lo..hi`` (inclusive at both ends), ``steps`` long.

    Log, not linear: pitch/brightness is perceived geometrically, so equal ratios are what
    read as equal steps of "more open". ``steps == 1`` degenerates to ``[lo]``.
    """
    import numpy as np

    return [float(f) for f in np.geomspace(float(lo_hz), float(hi_hz), int(steps))]


def render_cutoff_variants(
    audio_frame: dict,
    *,
    lo_hz: float = 200.0,
    hi_hz: float = 8000.0,
    steps: int = 5,
    resonance: float = 0.707,
) -> list[dict]:
    """Render ``steps`` timbral variants of one source, one per log-spaced cutoff in lo..hi.

    Returns a list of wet `audio` frames, closed (``lo_hz``) → open (``hi_hz``), each with role
    ``<base>.variant:<k>`` (k = 1..steps) and full lineage. ``hi_hz`` is clamped to 0.49·sr and
    ``lo_hz`` floored at 20 Hz (the automate sweep's limits); ``resonance`` is the low-pass Q
    (0.707 flat, 3-10 = a resonant, vocal-sounding edge at the cutoff).
    """
    import numpy as np

    steps = int(steps)
    if steps < 1:
        raise ValueError(f"steps must be >= 1 (got {steps})")
    if steps > MAX_VARIANT_STEPS:
        raise ValueError(f"steps must be <= {MAX_VARIANT_STEPS} (got {steps})")
    if resonance <= 0:
        raise ValueError(f"resonance must be > 0 (got {resonance})")

    data, sr = _load_audio(audio_frame)
    lo = max(float(lo_hz), 20.0)
    eff_hi = min(float(hi_hz), 0.49 * sr)
    if steps > 1 and not lo < eff_hi:
        raise ValueError(f"empty cutoff range: lo {lo:.1f} Hz >= hi {eff_hi:.1f} Hz (sr {sr})")

    cutoffs = variant_cutoffs(lo, eff_hi, steps)
    n = data.shape[0]

    out: list[dict] = []
    for k, fc in enumerate(cutoffs, start=1):
        params = {
            "variant_index": k,
            "steps": steps,
            "cutoff_hz": round(float(fc), 3),
            "lo_hz": round(lo, 1),
            "hi_hz": round(eff_hi, 1),
            "resonance": float(resonance),
            "spacing": "log",
            "sr_hz": sr,
        }
        if n == 0:
            params["noop"] = "empty"
            wet = data
        else:
            # One block over the whole clip: `_sweep_lowpass` with a constant cutoff is exactly
            # the automate sweep frozen at fc (identical coefficients + steady-state init).
            const = np.full(n, float(fc))
            wet = _sweep_lowpass(data, sr, const, float(resonance), n)
            wet = np.clip(wet, -1.0, 1.0).astype("float32")
        out.append(_emit_wet_audio(
            wet, sr, src_frame=audio_frame, op="variants", op_version=VARIANTS_OP_VERSION,
            params=params, role=_variant_role(audio_frame, k),
        ))
    return out


# ---------------------------------------------------------------------------
# stereoize — decorrelation widener: synthesise side energy from a mono(-ish) mid.
#
# `widen` (M-S) only SCALES existing side — useless on a mono/synth source with ~zero side.
# stereoize injects an allpass-decorrelated copy of the high-band mid into the side channel:
#   L = M + (S + amount·D),  R = M − (S + amount·D)   with D = allpass-scrambled high-band M
# so the MONO SUM L+R = 2·M is preserved EXACTLY (mono-compatible, and the mono-measured
# centroid/mid-sub/hi-sub are untouched) while side/mid rises — decoupling width from spectral
# balance. The low end (below crossover) stays mono. Allpass cascade keeps magnitude flat
# (no timbral colouration), only phase is scrambled.
# ---------------------------------------------------------------------------
STEREOIZE_OP_VERSION = "stereoize@1"

_DECORR_STAGES = ((13, 0.7), (29, 0.7), (47, 0.7), (71, 0.65), (113, 0.6))  # (delay, feedback)


def _allpass_delay(x, m: int, g: float):
    """Schroeder allpass: H(z) = (-g + z^-m) / (1 - g z^-m). Flat magnitude, scrambled phase."""
    import numpy as np
    from scipy.signal import lfilter

    b = np.zeros(m + 1); b[0] = -g; b[m] = 1.0
    a = np.zeros(m + 1); a[0] = 1.0; a[m] = -g
    return lfilter(b, a, x)


def _decorrelate(x):
    y = x.astype("float64", copy=True)
    for m, g in _DECORR_STAGES:
        y = _allpass_delay(y, m, g)
    return y


def apply_stereoize(audio_frame: dict, *, amount: float = 0.4, crossover_hz: float = 200.0,
                    order: int = 4) -> dict:
    """Widen by injecting decorrelated high-band mid into the side; mono sum preserved.

    ``amount`` scales the decorrelated side injection (0 = no-op). ``crossover_hz`` keeps the
    low end mono. Returns a wet `audio` frame; the mono downmix is bit-for-bit the input's
    (so spectral-balance metrics are unchanged — only stereo width moves).
    """
    import numpy as np
    from scipy.signal import butter, sosfiltfilt

    if amount < 0:
        raise ValueError("amount must be >= 0")
    if order < 1:
        raise ValueError("order must be >= 1")
    data, sr = _load_audio(audio_frame)
    n = data.shape[0]
    min_len = 3 * (2 * order + 1) + 1
    # passthrough on no-op / empty / too-short-to-crossover (widening a sub-ms clip is meaningless,
    # and skipping the highpass would leak bass into the side — breaking the low-end-mono guarantee)
    if n == 0 or amount == 0.0 or n <= min_len:
        note = "empty" if n == 0 else ("noop" if amount == 0.0 else "too short — passthrough")
        params = {"amount": float(amount), "crossover_hz": float(crossover_hz), "sr_hz": sr,
                  "note": note}
        return _emit_wet_audio(data, sr, src_frame=audio_frame, op="stereoize",
                               op_version=STEREOIZE_OP_VERSION, params=params)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    L = data[:, 0].astype("float64"); R = data[:, 1].astype("float64")
    M = 0.5 * (L + R); S = 0.5 * (L - R)

    nyq = sr / 2.0
    fc = min(max(float(crossover_hz) / nyq, 1e-4), 0.999)
    sos = butter(order, fc, btype="highpass", output="sos")
    high_M = sosfiltfilt(sos, M)
    D = _decorrelate(high_M)
    d_rms = float(np.sqrt(np.mean(D ** 2))) + 1e-12
    h_rms = float(np.sqrt(np.mean(high_M ** 2)))
    D *= (h_rms / d_rms)                       # match decorrelated energy to the high-band mid
    side_new = S + float(amount) * D
    # mono sum L+R = 2M preserved EXACTLY → do NOT clip (clipping would break the invariant);
    # peaks are handled by the downstream normalize/limit. May exceed unity by design.
    out = np.column_stack([M + side_new, M - side_new]).astype("float32")
    params = {"amount": float(amount), "crossover_hz": float(crossover_hz),
              "stages": len(_DECORR_STAGES), "order": int(order), "sr_hz": sr,
              "peak": round(float(np.max(np.abs(out))), 4)}
    return _emit_wet_audio(out, sr, src_frame=audio_frame, op="stereoize",
                           op_version=STEREOIZE_OP_VERSION, params=params)


# ---------------------------------------------------------------------------
# maximize — look-ahead brickwall limiter / loudness maximizer (vault-36pa).
#
# `limit` is a transparent whole-sample ceiling (scales the signal down, preserves crest).
# `compress` is feed-forward RMS (can't catch fast transients without pumping). Neither closes
# the crest gap to a mastered reference. maximize drives the signal in by ``makeup_db`` then
# look-ahead peak-limits to ``ceiling_dbtp`` — peaks are caught (gain ducks BEFORE them via the
# look-ahead window) while the body is raised, so crest drops toward a mastered density. The
# gain drops instantly and releases slowly (no pumping on sustained material).
# ---------------------------------------------------------------------------
MAXIMIZE_OP_VERSION = "maximize@1"


def apply_maximize(audio_frame: dict, *, ceiling_dbtp: float = -1.0, makeup_db: float = 6.0,
                   lookahead_ms: float = 2.0, release_ms: float = 60.0) -> dict:
    """Look-ahead brickwall limiter: drive by ``makeup_db``, cap peaks at ``ceiling_dbtp``.

    Reduces crest factor toward a mastered reference without the pumping of a compressor. Gain
    reduction is computed over a forward look-ahead window (ducks before the transient) and
    releases over ``release_ms``. Returns a wet `audio` frame.
    """
    import numpy as np
    from scipy.ndimage import minimum_filter1d

    if makeup_db < 0:
        raise ValueError("makeup_db must be >= 0 (it is the input drive)")
    if lookahead_ms <= 0 or release_ms <= 0:
        raise ValueError("lookahead_ms and release_ms must be > 0")
    data, sr = _load_audio(audio_frame)
    if data.shape[0] == 0:
        return _emit_wet_audio(data, sr, src_frame=audio_frame, op="maximize",
                               op_version=MAXIMIZE_OP_VERSION,
                               params={"note": "empty", "ceiling_dbtp": float(ceiling_dbtp)})
    ceiling = 10.0 ** (float(ceiling_dbtp) / 20.0)
    x = data.astype("float64") * (10.0 ** (float(makeup_db) / 20.0))   # drive in
    peak = np.max(np.abs(x), axis=1)                                   # per-sample peak across ch
    desired = np.minimum(1.0, ceiling / (peak + 1e-12))               # instantaneous needed gain
    la = max(int(lookahead_ms * sr / 1000.0), 1)
    g_la = minimum_filter1d(desired, size=la, origin=-(la // 2))       # look-ahead: duck before peak
    # release: gain drops instantly, rises at most `inc` per sample (slow recovery, no pump)
    inc = 1.0 / max(int(release_ms * sr / 1000.0), 1)
    g = np.empty_like(g_la)
    g[0] = g_la[0]
    for n in range(1, len(g)):
        prev = g[n - 1] + inc
        g[n] = g_la[n] if g_la[n] < prev else prev
    out = (x * g[:, None])
    out = np.clip(out, -ceiling, ceiling).astype("float32")           # brickwall safety
    in_rms = float(np.sqrt(np.mean(data.astype("float64") ** 2)) + 1e-12)
    out_rms = float(np.sqrt(np.mean(out ** 2)) + 1e-12)
    params = {"ceiling_dbtp": float(ceiling_dbtp), "makeup_db": float(makeup_db),
              "lookahead_ms": float(lookahead_ms), "release_ms": float(release_ms),
              "gain_reduction_db_max": round(float(20 * np.log10(min(float(g.min()), 1.0) + 1e-12)), 2),
              "rms_change_db": round(float(20 * np.log10(out_rms / in_rms)), 2), "sr_hz": sr}
    return _emit_wet_audio(out, sr, src_frame=audio_frame, op="maximize",
                           op_version=MAXIMIZE_OP_VERSION, params=params)


# ---------------------------------------------------------------------------
# crop / reverse / pitch / stretch — sampler-editing primitives (smpledit parity, vault-2s1g).
#
# The mechanical edits every sampler has. crop/reverse are exact numpy; pitch/stretch use a
# compact phase vocoder (STFT → time-scale with phase accumulation → ISTFT) so time and pitch
# move INDEPENDENTLY (unlike a naive resample). These are the fast, timbre-agnostic primitives;
# the formant-CORRECT versions live in `smpl larynx render` (WORLD) — use those for voice.
# ---------------------------------------------------------------------------
def apply_crop(audio_frame: dict, *, start_s: float = 0.0, end_s: Optional[float] = None) -> dict:
    """Keep only ``[start_s, end_s)`` seconds of the audio (end omitted = to the tail)."""
    import numpy as np

    data, sr = _load_audio(audio_frame)
    n = data.shape[0]
    a = min(max(int(round(start_s * sr)), 0), n)
    b = n if end_s is None else min(max(int(round(end_s * sr)), a), n)
    out = np.ascontiguousarray(data[a:b], dtype="float32")
    params = {"start_s": float(start_s), "end_s": (float(end_s) if end_s is not None else None),
              "start_sample": a, "end_sample": b, "sr_hz": sr}
    return _emit_wet_audio(out, sr, src_frame=audio_frame, op="crop",
                           op_version=CROP_OP_VERSION, params=params)


def apply_reverse(audio_frame: dict) -> dict:
    """Reverse the audio in time (per channel)."""
    import numpy as np

    data, sr = _load_audio(audio_frame)
    out = np.ascontiguousarray(data[::-1], dtype="float32")
    return _emit_wet_audio(out, sr, src_frame=audio_frame, op="reverse",
                           op_version=REVERSE_OP_VERSION, params={"sr_hz": sr})


def _phase_vocoder(mono: "object", speed: float, n_fft: int = 2048, hop: int = 512):
    """Time-scale a mono float64 signal by ``speed`` (>1 = FASTER/shorter) at constant pitch.

    Classic phase vocoder: STFT, interpolate frames at fractional time steps while accumulating
    the true instantaneous phase advance per bin, then overlap-add the ISTFT. Pitch is unchanged
    because only the frame *timing* is resampled, not the spectrum. (``speed`` is librosa's
    convention — a speed factor; callers wanting a LENGTH multiplier ``r`` pass ``speed = 1/r``.)
    """
    rate = speed
    import numpy as np

    win = np.hanning(n_fft).astype("float64")
    padded = np.concatenate([np.zeros(n_fft), mono, np.zeros(n_fft)])
    n_frames = 1 + (len(padded) - n_fft) // hop
    stft = np.empty((n_fft // 2 + 1, n_frames), dtype="complex128")
    for i in range(n_frames):
        seg = padded[i * hop: i * hop + n_fft]
        stft[:, i] = np.fft.rfft(win * seg)

    time_steps = np.arange(0, n_frames - 1, rate)
    expected = np.linspace(0, np.pi * hop, stft.shape[0])   # per-bin phase advance per hop
    phase_acc = np.angle(stft[:, 0])
    out_stft = np.empty((stft.shape[0], len(time_steps)), dtype="complex128")
    for t, step in enumerate(time_steps):
        lo = int(np.floor(step))
        frac = step - lo
        mag = (1.0 - frac) * np.abs(stft[:, lo]) + frac * np.abs(stft[:, lo + 1])
        out_stft[:, t] = mag * np.exp(1j * phase_acc)
        dphase = np.angle(stft[:, lo + 1]) - np.angle(stft[:, lo]) - expected
        dphase -= 2.0 * np.pi * np.round(dphase / (2.0 * np.pi))   # wrap to (−π, π]
        phase_acc = phase_acc + expected + dphase

    out_len = int(len(out_stft.T) * hop) + n_fft
    out = np.zeros(out_len, dtype="float64")
    wsum = np.zeros(out_len, dtype="float64")
    for i in range(out_stft.shape[1]):
        seg = np.fft.irfft(out_stft[:, i], n=n_fft) * win
        out[i * hop: i * hop + n_fft] += seg
        wsum[i * hop: i * hop + n_fft] += win ** 2
    out /= np.maximum(wsum, 1e-8)
    return out[n_fft: n_fft + max(int(round(len(mono) / speed)), 1)]


def apply_stretch(audio_frame: dict, *, ratio: float) -> dict:
    """Time-stretch by ``ratio`` (>1 = longer/slower) at constant pitch (phase vocoder)."""
    import numpy as np

    if ratio <= 0:
        raise ValueError(f"ratio must be > 0 (got {ratio})")
    data, sr = _load_audio(audio_frame)
    # ratio is a LENGTH multiplier (>1 longer); the PV takes a SPEED factor → invert.
    chans = [_phase_vocoder(data[:, c].astype("float64"), 1.0 / ratio) for c in range(data.shape[1])]
    out = np.clip(np.stack(chans, axis=1), -1.0, 1.0).astype("float32")
    params = {"ratio": float(ratio), "sr_hz": sr}
    return _emit_wet_audio(out, sr, src_frame=audio_frame, op="stretch",
                           op_version=STRETCH_OP_VERSION, params=params)


# ---------------------------------------------------------------------------
# paulstretch — EXTREME time-stretch into ambience (`smpl stretch --paul`).
#
# Nasca Octavian Paul's algorithm (public domain), promoted out of the basilica
# ambience build script. Window the input, keep each window's magnitude spectrum
# and RANDOMIZE the phases, then overlap-add the resynthesised windows at 50%
# overlap while the *input* read head advances only ``window/(2·factor)`` per
# step. Throwing the phases away is the whole trick: there is no transient left
# to smear, so 8×/50× factors turn a sample into a smooth pad instead of the
# metallic warble a phase vocoder gives at those ratios.
#
# The reference implementation's ``hinv_buf`` amplitude demodulation is
# DELIBERATELY absent (it is commented out upstream too): with the
# ``(1−x²)^1.25`` window at 50% overlap the overlap-add power sum ripples only
# ~2.6%, which is inaudible, and re-applying the correction would put a
# hop-rate line back into the energy envelope.
# ---------------------------------------------------------------------------
PAULSTRETCH_OP_VERSION = "paulstretch@1"

# Fixed phase seed: the algorithm is randomised, but a pipe op has to be
# reproducible (same input + params → same hash) for lineage/memoization.
_PAULSTRETCH_SEED = 0x5A17

# Partial L/R re-blend for the stereo variant (``pstretch_st``). Channels are
# stretched with independent phase streams, so they come out uncorrelated; the
# mix L' = a·L + b·R, R' = a·R + b·L lands inter-channel correlation at
# 2ab/(a²+b²) = 0.6 for b/a = 1/3, with a²+b² = 1 keeping the power constant.
_PSTRETCH_ST_A = 3.0 / (10.0 ** 0.5)
_PSTRETCH_ST_B = 1.0 / (10.0 ** 0.5)


def _paulstretch_mono(x, *, factor: float, windowsize: int, sr: int, rng):
    """Paulstretch one float64 channel by ``factor`` (>1 = longer). Returns float64."""
    import numpy as np

    half = windowsize // 2
    n = len(x)
    if n == 0:
        return np.zeros(0, dtype="float64")

    x = np.asarray(x, dtype="float64").copy()
    # Reference behaviour: taper the last 50 ms so the final window doesn't end on a step.
    end_size = min(max(int(sr * 0.05), 16), n)
    x[n - end_size:] *= np.linspace(1.0, 0.0, end_size)

    window = np.power(1.0 - np.power(np.linspace(-1.0, 1.0, windowsize), 2.0), 1.25)
    displace = half / float(factor)          # INPUT hop; the output hop stays `half`
    n_steps = max(int(math.ceil(n / displace)), 1)

    out = np.zeros(n_steps * half, dtype="float64")
    old = np.zeros(windowsize, dtype="float64")
    pos = 0.0
    for i in range(n_steps):
        istart = int(math.floor(pos))
        buf = x[istart: istart + windowsize]
        if len(buf) < windowsize:
            buf = np.concatenate([buf, np.zeros(windowsize - len(buf), dtype="float64")])
        buf = buf * window
        mags = np.abs(np.fft.rfft(buf))
        # Keep the magnitudes, discard the phases (uniform random, modulus 1).
        phases = rng.uniform(0.0, 2.0 * np.pi, mags.shape[0])
        buf = np.fft.irfft(mags * np.exp(1j * phases), n=windowsize)
        buf = buf * window                   # window again on the way out
        out[i * half: (i + 1) * half] = buf[:half] + old[half:]
        old = buf
        pos += displace
    return out


def apply_paulstretch(
    audio_frame: dict,
    *,
    factor: float,
    window_s: float = 0.28,
    stereo_decorrelate: bool = False,
) -> dict:
    """Extreme time-stretch (paulstretch) by ``factor``, returning a wet `audio` frame.

    ``factor`` is a LENGTH multiplier (8 → eight times longer); ``window_s`` is the
    analysis/synthesis window in seconds (longer = smoother/more smeared). Stereo input is
    stretched per channel with independent phase streams and then partially re-blended so
    the inter-channel correlation lands ~0.6 (``pstretch_st``); ``stereo_decorrelate=True``
    skips the re-blend and leaves the channels fully decorrelated for a wider pad.
    """
    import numpy as np

    if factor <= 0:
        raise ValueError(f"factor must be > 0 (got {factor})")
    if window_s <= 0:
        raise ValueError(f"window_s must be > 0 (got {window_s})")

    data, sr = _load_audio(audio_frame)
    windowsize = max(int(window_s * sr), 16)
    windowsize = (windowsize // 2) * 2       # even → a clean 50% overlap

    n_in = data.shape[0]
    target = max(int(round(n_in * float(factor))), 1) if n_in else 0

    chans = []
    for c in range(data.shape[1]):
        rng = np.random.default_rng(_PAULSTRETCH_SEED + c)
        y = _paulstretch_mono(data[:, c].astype("float64"), factor=float(factor),
                              windowsize=windowsize, sr=sr, rng=rng)
        # The OLA emits whole half-windows; trim (or pad) to the exact stretched length.
        if len(y) >= target:
            y = y[:target]
        else:
            y = np.concatenate([y, np.zeros(target - len(y), dtype="float64")])
        chans.append(y)

    out = np.stack(chans, axis=1) if chans else np.zeros((0, 1), dtype="float64")
    if out.shape[1] >= 2 and not stereo_decorrelate:
        left, right = out[:, 0].copy(), out[:, 1].copy()
        out[:, 0] = _PSTRETCH_ST_A * left + _PSTRETCH_ST_B * right
        out[:, 1] = _PSTRETCH_ST_A * right + _PSTRETCH_ST_B * left

    out = np.clip(out, -1.0, 1.0).astype("float32")
    params = {
        "mode": "paul",
        "factor": float(factor),
        "window_s": float(window_s),
        "window_samples": int(windowsize),
        "sr_hz": sr,
    }
    if data.shape[1] >= 2:
        params["stereo_decorrelate"] = bool(stereo_decorrelate)
    return _emit_wet_audio(out, sr, src_frame=audio_frame, op="paulstretch",
                           op_version=PAULSTRETCH_OP_VERSION, params=params)


def apply_pitch(audio_frame: dict, *, semitones: float) -> dict:
    """Pitch-shift by ``semitones`` at constant DURATION (phase-vocoder stretch + resample).

    Timbre-agnostic (formants shift with pitch — the chipmunk effect). For voice, prefer
    ``smpl larynx render --semitones`` (WORLD, formant-preserving). Time-stretch by ``1/shift``
    then resample by ``shift`` nets a pitch change at the original length.
    """
    import numpy as np
    from scipy.signal import resample_poly

    data, sr = _load_audio(audio_frame)
    shift = 2.0 ** (float(semitones) / 12.0)
    # Time-stretch LONGER by `shift` (speed = 1/shift, pitch unchanged), then resample by 1/shift
    # (up=1000, down=1000·shift) to restore the original length — which raises pitch by `shift`.
    stretched = [_phase_vocoder(data[:, c].astype("float64"), 1.0 / shift)
                 for c in range(data.shape[1])]
    up, down = 1000, max(int(round(1000 * shift)), 1)
    chans = [resample_poly(s, up, down) for s in stretched]
    target = data.shape[0]
    out = np.zeros((target, data.shape[1]), dtype="float64")
    for c, ch in enumerate(chans):
        m = min(len(ch), target)
        out[:m, c] = ch[:m]
    out = np.clip(out, -1.0, 1.0).astype("float32")
    params = {"semitones": float(semitones), "shift_ratio": round(shift, 5), "sr_hz": sr}
    return _emit_wet_audio(out, sr, src_frame=audio_frame, op="pitch",
                           op_version=PITCH_OP_VERSION, params=params)


# ---------------------------------------------------------------------------
# loopify — make a rendered loop TILE-SAFE (promoted from refmatch/loopify.py, vault-11gm).
#
# A render is not a loop. smplmix places bar-1-beat-1 at sample ~249 (a ~5 ms render offset) and
# cuts the tail wherever the render ended, so naive repeats of its output sit progressively LATE
# and click at every seam. Three exact fixes, in order: shift the downbeat to sample 0, force the
# length to the bar grid, fade the seam to zero. Everything here is sample-exact numpy — no
# analysis, no resampling — so the audio between the trim and the fade is bit-identical.
# ---------------------------------------------------------------------------
def apply_loopify(
    audio_frame: dict,
    *,
    bpm: float,
    bars: int = 2,
    beats_per_bar: int = 4,
    declick_ms: float = 5.0,
    max_trim_ms: float = 12.0,
) -> dict:
    """Trim the render-offset (downbeat→0), set the exact bar length, fade the seam to zero.

    The leading offset is found as the first sample above −45 dB of the peak, and is trimmed ONLY
    when it is shorter than ``max_trim_ms`` — that bounds the fix to smplmix's ~5 ms render
    artifact; a longer lead is a musical fade-in or pickup and is left intact (trimming it would
    eat the performance). The body is then truncated or zero-padded to
    ``round(bars * beats_per_bar * 60 / bpm * sr)`` samples so N repeats land exactly on the grid.
    Finally the last ``declick_ms`` fades to 0 — the loop point is where a discontinuity becomes an
    audible click — plus a ~1.3 ms (64-sample) fade-in that kills start DC without softening the
    first transient, so the downbeat keeps its punch. The end fade is skipped for a body shorter
    than ``2 * declick_ms`` (nothing but fade would survive).
    """
    import numpy as np

    if bpm <= 0:
        raise ValueError(f"bpm must be > 0 (got {bpm})")
    if bars <= 0 or beats_per_bar <= 0:
        raise ValueError(f"bars/beats_per_bar must be >= 1 (got {bars}/{beats_per_bar})")
    if declick_ms < 0 or max_trim_ms < 0:
        raise ValueError(f"declick_ms/max_trim_ms must be >= 0 (got {declick_ms}/{max_trim_ms})")

    data, sr = _load_audio(audio_frame)
    x = data
    target = int(round(bars * beats_per_bar * 60.0 / float(bpm) * sr))
    if x.shape[0] == 0:  # empty slice → passthrough with a note (mirror apply_compress)
        params = {"bpm": float(bpm), "bars": int(bars), "beats_per_bar": int(beats_per_bar),
                  "note": "empty input: passthrough", "sr_hz": sr}
        return _emit_wet_audio(x, sr, src_frame=audio_frame, op="loopify",
                               op_version=LOOPIFY_OP_VERSION, params=params)

    # 1. downbeat → 0: first sample above −45 dBpeak, trimmed only if it is the render offset.
    mag = np.abs(x).max(axis=1)
    peak = float(mag.max()) or 1.0
    above = np.where(mag > 10 ** (-45 / 20) * peak)[0]
    lead = int(above[0]) if len(above) else 0
    if 0 < lead < int(max_trim_ms / 1000 * sr):
        x = x[lead:]
    else:
        lead = 0

    # 2. exact bar length: truncate a long render, zero-pad one that got cut short.
    if len(x) >= target:
        x = x[:target].copy()
    else:
        x = np.pad(x, ((0, target - len(x)), (0, 0)))

    # 3. de-click the seam: end → 0, plus a 64-sample fade-in against start DC.
    d = int(declick_ms / 1000 * sr)
    if d and len(x) > 2 * d:
        x[-d:] *= np.linspace(1.0, 0.0, d, dtype="float32")[:, None]
        fi = min(64, len(x))
        x[:fi] *= np.linspace(0.0, 1.0, fi, dtype="float32")[:, None]

    params = {
        "bpm": float(bpm), "bars": int(bars), "beats_per_bar": int(beats_per_bar),
        "lead_trimmed_samples": int(lead), "target_len_samples": int(target),
        "declick_ms": float(declick_ms), "max_trim_ms": float(max_trim_ms), "sr_hz": sr,
    }
    return _emit_wet_audio(x.astype("float32"), sr, src_frame=audio_frame, op="loopify",
                           op_version=LOOPIFY_OP_VERSION, params=params)
