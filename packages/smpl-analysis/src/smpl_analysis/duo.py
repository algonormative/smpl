"""Duo ops — the first **2-audio-input** op family (ticket vault-nkvw).

Every op here consumes TWO resolved audio frames and emits ONE wet `audio` frame whose
``lineage`` names BOTH inputs — establishing the 2-input lineage convention the rest of the
duo family (mix, convolve, …) will follow. The single-input edit ops in
``smpl_analysis.edit`` set ``lineage = [src.id]``; a duo op sets ``lineage = [a.id, b.id]``.

First op: **vocode** — a classic analysis/synthesis channel vocoder. The MODULATOR's
per-band amplitude envelope is imprinted onto the CARRIER's per-band spectrum, so the output
speaks with the modulator's articulation but keeps the carrier's timbre. Art-direction linchpin
for WRUM (the hell-pole voice): FLOW spat render = MODULATOR, deep harsh bass voice = CARRIER.

Second op: **sidechain** — duck a TARGET (sub, rumble, mix) under a TRIGGER's transients
(the kick). The Birmingham rumble + glue move: fast attack, release in the 20–350 ms range.

Heavy imports (scipy, soundfile) stay INSIDE the functions so a cold pipe stage starts fast.
Deterministic scipy/numpy — no shell-out, empty env-fingerprint, bit-identical across runs.
"""

from __future__ import annotations

import io

# ---------------------------------------------------------------------------
# op_version — bumped on ANY behavior change (spec → *Memoization*).
# ---------------------------------------------------------------------------
VOCODE_OP_VERSION = "vocode@2"  # @2: body-normalize before ess (vowel-crush fix); attack 2 ms
SIDECHAIN_OP_VERSION = "sidechain@1"


# ---------------------------------------------------------------------------
# Shared helpers.
# ---------------------------------------------------------------------------
def _load_mono(audio_frame: dict):
    """Resolve an audio frame's CAS blob to ``(mono float64 samples, sr)`` (mean-mixed)."""
    import numpy as np
    import soundfile as sf

    from smplstream import cas

    src = cas.get_path(audio_frame["hash"])
    data, sr = sf.read(str(src), dtype="float64", always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    return np.ascontiguousarray(mono, dtype="float64"), int(sr)


def _load_channels(audio_frame: dict):
    """Resolve an audio frame's CAS blob to ``((n, ch) float64 samples, sr)`` — channels KEPT.

    The mono-mixing sibling of :func:`_load_mono`, for ops that process a target in place
    (sidechain) rather than analyzing it.
    """
    import numpy as np
    import soundfile as sf

    from smplstream import cas

    src = cas.get_path(audio_frame["hash"])
    data, sr = sf.read(str(src), dtype="float64", always_2d=True)
    return np.ascontiguousarray(data, dtype="float64"), int(sr)


def _rms(x) -> float:
    import numpy as np

    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x)) + 1e-20))


def _follow_env(rect, a_att: float, a_rel: float):
    """Asymmetric peak envelope follower over the LAST axis of ``rect`` (nonneg input).

    A first-order recursive one-pole whose coefficient switches on rising vs falling samples:
    ``env = c*env + (1-c)*x`` with ``c = a_att`` when ``x > env`` (fast) else ``a_rel`` (slow).
    ``rect`` is ``(bands, n)``; the loop runs over samples so all bands advance as one vector
    (the recursion is inherently sequential — no lfilter form — but vectorized across bands).
    """
    import numpy as np

    rect = np.atleast_2d(rect)
    b, n = rect.shape
    env = np.zeros(b, dtype="float64")
    out = np.empty((b, n), dtype="float64")
    for i in range(n):
        x = rect[:, i]
        coef = np.where(x > env, a_att, a_rel)
        env = coef * env + (1.0 - coef) * x
        out[:, i] = env
    return out


def _one_pole_coef(sr: int, ms: float) -> float:
    """One-pole smoothing coefficient for a time constant of ``ms`` milliseconds."""
    import numpy as np

    tau = sr * max(float(ms), 1e-3) / 1000.0
    return float(np.exp(-1.0 / tau))


# ---------------------------------------------------------------------------
# vocode — channel vocoder: modulator's band envelopes × carrier's band spectrum.
# ---------------------------------------------------------------------------
def apply_vocode(
    carrier_frame: dict,
    modulator_frame: dict,
    *,
    bands: int = 24,
    lo_hz: float = 70.0,
    hi_hz: float = 11000.0,
    ess_hz: float = 5500.0,
    ess_mix: float = 0.35,
    attack_ms: float = 2.0,
    release_ms: float = 45.0,
) -> dict:
    """Channel-vocode ``carrier_frame`` with ``modulator_frame``, returning a wet `audio` frame.

    Both inputs are mono-mixed; if the sample rates differ the MODULATOR is resampled to the
    carrier's sr (``scipy.resample_poly``) and both are truncated to the shorter length. A
    log-spaced bank of ``bands`` 4th-order Butterworth band-passes splits carrier and modulator;
    per band the modulator's rectified envelope (asymmetric attack/release one-pole) — normalized
    by the modulator's *global* peak envelope so band gains stay RELATIVE — multiplies the
    carrier band. A "Bode ess" path adds the modulator high-passed above ``ess_hz``, level-matched
    to ``ess_mix × carrier_rms/ess_rms``, to keep consonants intelligible. Output peak is
    normalized to 0.9.

    Emits a wet frame with role ``<carrier_role>.wet``, op ``vocode``, and — the 2-input lineage
    convention — ``lineage = [carrier.id, modulator.id]``.
    """
    import numpy as np
    from scipy.signal import butter, resample_poly, sosfilt

    if bands < 1:
        raise ValueError(f"bands must be >= 1 (got {bands})")

    carrier, sr = _load_mono(carrier_frame)
    mod, mod_sr = _load_mono(modulator_frame)

    # Sample-rate reconcile: resample the MODULATOR to the carrier's sr.
    if mod_sr != sr:
        from math import gcd

        g = gcd(int(sr), int(mod_sr)) or 1
        mod = np.ascontiguousarray(resample_poly(mod, sr // g, mod_sr // g), dtype="float64")
        mod_sr = sr

    # Truncate both to the shorter length.
    n = int(min(carrier.shape[0], mod.shape[0]))
    carrier = carrier[:n]
    mod = mod[:n]

    nyq = sr / 2.0
    eff_hi = min(float(hi_hz), 0.45 * sr)
    if eff_hi <= lo_hz:
        raise ValueError(f"band grid empty: lo_hz {lo_hz} >= effective hi {eff_hi:.0f} (sr {sr})")
    edges = np.geomspace(max(float(lo_hz), 1.0), eff_hi, bands + 1)

    a_att = _one_pole_coef(sr, attack_ms)
    a_rel = _one_pole_coef(sr, release_ms)

    out = np.zeros(n, dtype="float64")
    if n > 0:
        # Global peak envelope of the (broadband) modulator → relative-gain normalizer.
        global_env = _follow_env(np.abs(mod)[None, :], a_att, a_rel)[0]
        global_peak = float(np.max(global_env)) if global_env.size else 0.0
        norm = global_peak if global_peak > 1e-9 else 1.0

        # Per-band: bandpass carrier + modulator, follow the modulator band envelope,
        # normalize by the global peak (relative band gains), multiply into the carrier band.
        carrier_bands = np.empty((bands, n), dtype="float64")
        mod_rect = np.empty((bands, n), dtype="float64")
        for bi in range(bands):
            low = max(float(edges[bi]) / nyq, 1e-6)
            high = min(float(edges[bi + 1]) / nyq, 0.999999)
            if high <= low:
                carrier_bands[bi] = 0.0
                mod_rect[bi] = 0.0
                continue
            sos = butter(4, [low, high], btype="bandpass", output="sos")
            carrier_bands[bi] = sosfilt(sos, carrier)
            mod_rect[bi] = np.abs(sosfilt(sos, mod))

        band_env = _follow_env(mod_rect, a_att, a_rel) / norm
        out = np.sum(carrier_bands * band_env, axis=0)

    # Intelligibility fix (vocode@2, research/intelligibility.md #3): normalize the vocoded
    # BODY to 0.9 first, THEN add the ess band relative to the normalized body. The old order
    # (ess first, normalize after) let the ess transients set the peak, crushing the vowels
    # that carry the words. Floor above numerical noise so silence stays silent (review fix).
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1e-6:
        out = out * (0.9 / peak)
    else:
        out = np.zeros_like(out)

    # Bode ess band: modulator high-passed above ess_hz, level-matched to the NORMALIZED body
    # and added (un-vocoded HF) so sibilants/consonants stay intelligible. Silent-carrier
    # guard (review fix): only when the carrier is real.
    if ess_mix > 0.0 and ess_hz < nyq and _rms(carrier) > 1e-6 and peak > 1e-6:
        sos_hp = butter(4, min(float(ess_hz) / nyq, 0.999999), btype="highpass", output="sos")
        ess = sosfilt(sos_hp, mod)
        ess_rms = _rms(ess)
        if ess_rms > 1e-9:
            scale = float(ess_mix) * (_rms(out) / ess_rms)
            out = out + scale * ess
        # safety cap after the ess add — scale, never clip
        peak2 = float(np.max(np.abs(out)))
        if peak2 > 0.98:
            out = out * (0.98 / peak2)
    out = out.astype("float32")

    params = {
        "bands": int(bands),
        "lo_hz": float(lo_hz),
        "hi_hz": float(hi_hz),
        "hi_hz_effective": round(float(eff_hi), 1),
        "band_edges_hz": [round(float(e), 2) for e in edges],
        "ess_hz": float(ess_hz),
        "ess_mix": float(ess_mix),
        "attack_ms": float(attack_ms),
        "release_ms": float(release_ms),
        "sr_hz": int(sr),
        "carrier_hash": carrier_frame.get("hash"),
        "modulator_hash": modulator_frame.get("hash"),
    }
    return _emit_duo_audio(
        out,
        sr,
        carrier_frame=carrier_frame,
        modulator_frame=modulator_frame,
        op="vocode",
        op_version=VOCODE_OP_VERSION,
        params=params,
    )


# ---------------------------------------------------------------------------
# sidechain — duck a TARGET by a TRIGGER's transients (the kick-vs-sub glue move).
# ---------------------------------------------------------------------------
def apply_sidechain(
    target_frame: dict,
    trigger_frame: dict,
    *,
    attack_ms: float = 5.0,
    release_ms: float = 120.0,
    depth_db: float = 12.0,
    threshold_db: float = -30.0,
) -> dict:
    """Duck ``target_frame`` under ``trigger_frame``'s transients — a wet `audio` frame.

    The Birmingham rumble / glue move: a kick (TRIGGER) pushes a sub, rumble, or full mix
    (TARGET) out of the way so the transient lands in cleared space, then the target recovers
    on the release. The TRIGGER is mono-mixed and, if its sample rate differs, resampled to the
    target's sr (``scipy.resample_poly``), then aligned to the target's timeline — zero-padded
    if short, truncated if long. The TARGET keeps all of its channels: one gain curve is applied
    to every channel, so a stereo target stays stereo and its image is untouched.

    Gain curve: a rectified peak envelope of the trigger, followed with the asymmetric one-pole
    (``attack_ms`` rising / ``release_ms`` falling), is read in dB **relative to full scale**
    (0 dBFS). ``threshold_db`` is therefore an absolute dBFS knee: at or below it the target is
    untouched, at 0 dBFS the target is pulled down by the full ``depth_db``, and in between the
    reduction interpolates linearly in dB::

        gr_db = -depth_db * clip((env_dbfs - threshold_db) / (0 - threshold_db), 0, 1)

    The duck's shape is the follower's — fast attack pulls down on the transient, the release
    time constant governs the recovery. Output is **not** re-normalized (a duck that renormalizes
    is not a duck). ``depth_db=0`` is an exact no-op.

    ``params`` records the settings plus the MEASURED result — ``gr_max_db`` (the deepest gain
    reduction actually applied) and ``ducked_fraction`` (share of samples pulled down by more
    than 0.1 dB) — so an A/B can be read off the frame without re-analyzing the audio.

    Emits role ``<target_role>.wet``, op ``sidechain``, ``lineage = [target.id, trigger.id]``.
    """
    import numpy as np
    from scipy.signal import resample_poly

    if attack_ms <= 0:
        raise ValueError(f"attack_ms must be > 0 (got {attack_ms})")
    if release_ms <= 0:
        raise ValueError(f"release_ms must be > 0 (got {release_ms})")
    if depth_db < 0:
        raise ValueError(f"depth_db must be >= 0 (got {depth_db})")
    if threshold_db >= 0:
        raise ValueError(f"threshold_db must be < 0 dBFS (got {threshold_db})")

    target, sr = _load_channels(target_frame)
    trig, trig_sr = _load_mono(trigger_frame)

    # Sample-rate reconcile: resample the TRIGGER to the target's sr.
    if trig_sr != sr:
        from math import gcd

        g = gcd(int(sr), int(trig_sr)) or 1
        trig = np.ascontiguousarray(resample_poly(trig, sr // g, trig_sr // g), dtype="float64")

    # Align the trigger to the TARGET's timeline (the target is never truncated).
    n = int(target.shape[0])
    if trig.shape[0] < n:
        trig = np.concatenate([trig, np.zeros(n - trig.shape[0], dtype="float64")])
    else:
        trig = trig[:n]

    a_att = _one_pole_coef(sr, attack_ms)
    a_rel = _one_pole_coef(sr, release_ms)

    gr_db = np.zeros(n, dtype="float64")
    out = target
    if n > 0:
        env = _follow_env(np.abs(trig)[None, :], a_att, a_rel)[0]
        env_db = 20.0 * np.log10(np.maximum(env, 1e-12))
        drive = np.clip((env_db - float(threshold_db)) / (0.0 - float(threshold_db)), 0.0, 1.0)
        gr_db = -float(depth_db) * drive
        out = target * np.power(10.0, gr_db / 20.0)[:, None]
    out = out.astype("float32")

    gr_max_db = round(abs(float(np.min(gr_db))), 4) if gr_db.size else 0.0
    ducked = float(np.count_nonzero(gr_db < -0.1) / gr_db.size) if gr_db.size else 0.0

    params = {
        "attack_ms": float(attack_ms),
        "release_ms": float(release_ms),
        "depth_db": float(depth_db),
        "threshold_db": float(threshold_db),
        "threshold_ref": "dbfs",
        "sr_hz": int(sr),
        "gr_max_db": gr_max_db,
        "ducked_fraction": round(ducked, 4),
        "target_hash": target_frame.get("hash"),
        "trigger_hash": trigger_frame.get("hash"),
    }
    return _emit_duo_audio(
        out,
        sr,
        carrier_frame=target_frame,
        modulator_frame=trigger_frame,
        op="sidechain",
        op_version=SIDECHAIN_OP_VERSION,
        params=params,
    )


# ---------------------------------------------------------------------------
# 2-input wet-frame emitter (lineage = [carrier.id, modulator.id]).
# ---------------------------------------------------------------------------
def _carrier_wet_role(carrier_frame: dict) -> str:
    """``<carrier_role>.wet`` (strip an existing .wet/.dry so re-vocoding stays single-suffix)."""
    role = carrier_frame.get("role") or "edit"
    for suffix in (".wet", ".dry"):
        if role.endswith(suffix):
            role = role[: -len(suffix)]
            break
    return f"{role}.wet"


def _emit_duo_audio(
    samples,
    sr: int,
    *,
    carrier_frame: dict,
    modulator_frame: dict,
    op: str,
    op_version: str,
    params: dict,
) -> dict:
    """CAS a processed mono/stereo float32 array as WAV → wet `audio` frame with 2-input lineage."""
    import numpy as np
    import soundfile as sf

    from smplstream import cas, frames as F

    arr = np.ascontiguousarray(np.asarray(samples, dtype="float32"))
    if arr.ndim == 1:
        arr = arr[:, None]
    buf = io.BytesIO()
    sf.write(buf, arr, sr, format="WAV", subtype="FLOAT")
    h = cas.put_audio_bytes(buf.getvalue())
    meta = cas.read_meta(h) or {}

    lineage = [f["id"] for f in (carrier_frame, modulator_frame) if f.get("id")]
    return F.audio_frame(
        h,
        sr=meta.get("sr", sr),
        ch=meta.get("ch", arr.shape[1]),
        dur=meta.get("dur", arr.shape[0] / sr if sr else 0.0),
        role=_carrier_wet_role(carrier_frame),
        of=carrier_frame.get("id"),
        lineage=lineage or None,
        op=op,
        op_version=op_version,
        params=params,
        fmt=meta.get("fmt"),
    )
