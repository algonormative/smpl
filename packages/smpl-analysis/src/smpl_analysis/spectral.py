"""Spectral-shape feature family (research §2; ticket vault-3uap).

Computes the spectral distribution-shape descriptors off a single librosa STFT —
flatness, crest, spread, rolloff, contrast, slope, skewness, kurtosis — and emits a
single `feature` frame whose values are frame-aggregated `{mean, stdev}` objects
(spec → *Standards alignment*; the Essentia `{mean, stdev}` statistic convention).

The emitted keys use the EXACT `lowlevel.spectral_*` spellings registered in
`feature-keys.md`. NOTE: those `lowlevel.*` keys are **provisional** pending the
Essentia-vs-lean-stack spike (vault-tkih) — that spike decides whether these features
come from Essentia or stay on librosa, and finalizes the spellings in the registry.

Pure functions returning frame dicts. Heavy imports (librosa/numpy) live inside the
functions so cold pipe stages start fast.
"""

from __future__ import annotations

from typing import Optional

OP = "spectral"
OP_VERSION = "spectral@1"

# STFT analysis defaults (declared so memo params are complete & stable per op_version).
N_FFT = 2048
HOP_LENGTH = 512

# Declared default table for the memo key (spec → *Parameter canonicalization*): omitted
# params are filled from THIS table for THIS op_version before hashing.
MEMO_DEFAULTS = {"n_fft": N_FFT, "hop_length": HOP_LENGTH}


def _mean_std(values) -> dict:
    """Frame-aggregate a per-frame array into the {mean, stdev} statistic shape.

    Ignores non-finite frames (a silent/empty STFT frame can yield NaN for ratio-style
    descriptors); returns 0.0/0.0 when nothing finite remains so the feature stays
    JSON-serializable.
    """
    import numpy as np

    arr = np.asarray(values, dtype="float64").ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": 0.0, "stdev": 0.0}
    return {"mean": round(float(np.mean(arr)), 6), "stdev": round(float(np.std(arr)), 6)}


def spectral_shape(y, sr: int, *, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH) -> dict:
    """Compute the spectral-shape family over a mono signal, returning the feature dict.

    Returns a dict mapping each registered `lowlevel.spectral_*` key to a `{mean, stdev}`
    object. The caller wraps it in a `feature` frame.
    """
    import librosa
    import numpy as np

    from .spectrogram import pad_short_signal

    y = np.asarray(y, dtype="float32")
    if y.ndim > 1:  # collapse to mono for shape descriptors
        y = np.mean(y, axis=0)

    y = pad_short_signal(y, n_fft)  # short one-shots → full window (no drop, no warning)

    # One STFT; magnitude (S) and power spectrum reused across descriptors.
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))  # (freq, frames)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)              # (freq,)
    eps = np.finfo(np.float32).eps

    # --- flatness: geometric-mean / arithmetic-mean of the POWER spectrum, in dB. ---
    # librosa.feature.spectral_flatness returns the linear power ratio (0..1); the
    # registry stores dB (NOT a 0–1 ratio), so convert 10*log10(ratio).
    flat_lin = librosa.feature.spectral_flatness(
        S=S, n_fft=n_fft, hop_length=hop_length, power=2.0
    )[0]
    flat_db = 10.0 * np.log10(np.maximum(flat_lin, eps))

    # --- crest: peak / mean of the magnitude spectrum, per frame (unitless). ---
    peak = np.max(S, axis=0)
    mean_mag = np.mean(S, axis=0)
    crest = peak / (mean_mag + eps)

    # --- centroid / spread / skewness / kurtosis: moments of the magnitude PDF. ---
    # Normalize each frame's magnitude into a probability distribution over freqs.
    col_sum = np.sum(S, axis=0, keepdims=True)
    p = S / (col_sum + eps)                          # (freq, frames)
    f = freqs[:, None]                               # (freq, 1)
    centroid = np.sum(f * p, axis=0)                 # Hz
    var = np.sum(((f - centroid) ** 2) * p, axis=0)  # Hz^2
    spread = np.sqrt(np.maximum(var, 0.0))           # Hz (sqrt of 2nd central moment)
    std = spread + eps
    skewness = np.sum(((f - centroid) ** 3) * p, axis=0) / (std ** 3)   # unitless
    kurtosis = np.sum(((f - centroid) ** 4) * p, axis=0) / (std ** 4)   # unitless (Pearson)

    # --- rolloff: 85% energy frequency, per frame (Hz). ---
    rolloff = librosa.feature.spectral_rolloff(
        S=S, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=0.85
    )[0]

    # --- contrast: peak-vs-valley dB across sub-bands, averaged across bands (dB). ---
    contrast = librosa.feature.spectral_contrast(
        S=S, sr=sr, n_fft=n_fft, hop_length=hop_length
    )  # (bands, frames), already in dB
    contrast_mean_band = np.mean(contrast, axis=0)

    # --- slope: linear regression slope of magnitude vs frequency, per frame (unitless). ---
    # slope = cov(f, mag) / var(f); a single var(f) (freq grid is frame-invariant).
    f1 = freqs
    fmean = np.mean(f1)
    f_centered = f1 - fmean
    f_var = np.sum(f_centered ** 2) + eps
    mag_mean = np.mean(S, axis=0, keepdims=True)
    slope = np.sum(f_centered[:, None] * (S - mag_mean), axis=0) / f_var

    return {
        "lowlevel.spectral_flatness_db": _mean_std(flat_db),
        "lowlevel.spectral_crest": _mean_std(crest),
        "lowlevel.spectral_spread": _mean_std(spread),
        "lowlevel.spectral_rolloff": _mean_std(rolloff),
        "lowlevel.spectral_contrast": _mean_std(contrast_mean_band),
        "lowlevel.spectral_slope": _mean_std(slope),
        "lowlevel.spectral_skewness": _mean_std(skewness),
        "lowlevel.spectral_kurtosis": _mean_std(kurtosis),
    }


def spectral_audio_frame(
    audio_frame: dict,
    *,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    use_cache: bool = True,
) -> list[dict]:
    """Resolve an `audio` frame's PCM from the CAS and emit its spectral-shape feature.

    Returns a one-element list with the derived `feature` frame (role `spectral`), carrying
    `of`/`op`/`op_version`/`params` lineage per the tool contract. The caller is responsible
    for passthrough of the input frame.

    **Memoized** (spec → *Memoization*): keyed on
    ``(op, op_version, input audio hash, canonical params)`` with an empty env fingerprint
    (pure-Python/deterministic). A hit skips the decode and the STFT entirely;
    ``use_cache=False`` forces recompute and refreshes the entry. ``params.cache_hit``
    records which path ran.
    """
    from smplstream import cas, frames as F, memo, memostore

    params = {"n_fft": n_fft, "hop_length": hop_length}
    mkey = memo.memo_key(
        OP, OP_VERSION, [audio_frame["hash"]], params=params, defaults=MEMO_DEFAULTS
    )

    data = memostore.get_json(mkey) if use_cache else None
    cache_hit = data is not None

    if not cache_hit:
        import soundfile as sf

        src = cas.get_path(audio_frame["hash"])
        y, sr = sf.read(str(src), dtype="float32", always_2d=True)
        y = y.T  # (ch, n) for the mono collapse in spectral_shape

        data = spectral_shape(y, sr, n_fft=n_fft, hop_length=hop_length)
        memostore.put_json(mkey, data, op=OP, op_version=OP_VERSION)

    return [
        F.feature_frame(
            data,
            role="spectral",
            of=audio_frame["id"],
            op=OP,
            op_version=OP_VERSION,
            params={**params, "cache_hit": cache_hit},
        )
    ]
