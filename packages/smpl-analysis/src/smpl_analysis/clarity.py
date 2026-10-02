"""Clarity feature family — attune-parity (ticket vault-1fxy).

The "how CLEAR / how MUDDY does this sit in a mix" axis — the spectral-balance and
masking read that a mix engineer hears as mud, boxiness, presence and bite. Where
the spectral-shape family (vault-3uap) describes the distribution's *moments*, these
metrics describe **mix-relevant band relationships**: low-mid buildup masking the
presence region, the mud-to-presence balance, tonal contrast in the low vs high
halves, and how much of the energy is focused / transient in the presence band.

Band energies use the **six standardized band edges** registered in feature-keys.md
(reused from vault-3tuy, not redefined): Sub 20–60 · Bass 60–200 · LoMid 200–500 ·
Mid 500–2k · UpperMid 2–6k · Air 6–20k. "Presence" is the UpperMid band (2–6 kHz).

Like the movement family, clarity is **duration-gated** (see feature-keys.md →
*Duration / role gating*): the metrics need enough material for a stable spectrum
and are degenerate on tiny fragments, so material shorter than :data:`MIN_DURATION_S`
emits every key as ``None``.

Metrics (all off a single STFT — band powers time-averaged, the transient read per-frame):

  ``low_mid_masking_db``        LoMid vs Mid energy (200–500 / 500–2k), in dB — the
                                boxy low-mid buildup that masks the core midrange.
  ``mud_presence_ratio``        LoMid / UpperMid POWER (200–500 vs 2–6k), linear —
                                the headline mud-vs-presence balance (>1 = mud-heavy).
  ``band_contrast_lo_db``       spectral peak-to-valley contrast in the LOW half
                                (20–500 Hz), dB — tonal (peaky) vs washy (flat).
  ``band_contrast_hi_db``       the same in the HIGH half (500 Hz–20 kHz), dB.
  ``presence_focus_ratio``      UpperMid / total energy (0–1) — how presence-focused.
  ``presence_transient_ratio``  temporal crest of the presence band's per-frame energy —
                                transient/percussive presence vs sustained presence.

Layering rule (spec): this op EMITS the measurements only. What counts as "too
muddy" lives in profiles / docs, never as a threshold here.

Pure functions returning frame dicts. Heavy imports live inside the functions.
"""

from __future__ import annotations

OP = "clarity"
OP_VERSION = "clarity@1"

# --- pinned recipe constants ---------------------------------------------------------------
MIN_DURATION_S = 0.5        # duration gate — below this the spectrum is too unstable (emit null)
N_FFT = 2048
HOP_LENGTH = 512
PRESENCE_LO = 2000.0        # presence band = registry UpperMid (2–6 kHz), reused
PRESENCE_HI = 6000.0
LOW_HALF_HZ = (20.0, 500.0)     # low-half contrast window (Sub+Bass+LoMid)
HIGH_HALF_HZ = (500.0, 20000.0) # high-half contrast window (Mid+UpperMid+Air)
CONTRAST_LO_PCT = 15.0
CONTRAST_HI_PCT = 85.0
CONTRAST_FLOOR_REL = 1e-6   # valley floor = region-peak × this (caps contrast at ~60 dB)
BAND_FLOOR_REL = 1e-6       # per-band power floor = total × this (bounds ratios to ±60 dB)

_EPS = 1e-12

KEYS: tuple[str, ...] = (
    "clarity.low_mid_masking_db",
    "clarity.mud_presence_ratio",
    "clarity.band_contrast_lo_db",
    "clarity.band_contrast_hi_db",
    "clarity.presence_focus_ratio",
    "clarity.presence_transient_ratio",
)


# --------------------------------------------------------------------------------------------
# Low-level DSP helpers.
# --------------------------------------------------------------------------------------------


def _to_mono(y):
    import numpy as np

    y = np.asarray(y, dtype="float64")
    if y.ndim > 1:
        y = y.mean(axis=0) if y.shape[0] < y.shape[1] else y.mean(axis=1)
    return np.ascontiguousarray(y)


def _band_edges() -> dict:
    """The six standardized band edges (reused from :mod:`smpl_analysis.width`, minus ``full``)."""
    from . import width as W

    return {name: (lo, hi) for name, lo, hi in W.BANDS if name != "full"}


def _stft_power(y, sr: int):
    """Per-bin/per-frame STFT power ``(freq, frames)`` and the bin frequencies.

    One STFT feeds every clarity metric: band powers (time-averaged) and the presence-band
    transient crest (per-frame). Working in the STFT domain keeps the transient read free of
    the time-domain edge transients a Butterworth band-pass would inject on an abrupt start.
    """
    import librosa
    import numpy as np

    from .spectrogram import pad_short_signal

    y = np.asarray(y, dtype="float32")
    y = pad_short_signal(y, N_FFT)  # short one-shots → full window (no drop, no warning)
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))  # (freq, frames)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    return (S ** 2), freqs


def _band_power(power, freqs, lo: float, hi: float) -> float:
    import numpy as np

    mask = (freqs >= lo) & (freqs < hi)
    return float(power[mask].sum()) if mask.any() else 0.0


def _contrast_db(power, freqs, lo: float, hi: float) -> float:
    """Peak-to-valley contrast over the bins in [lo, hi): 10·log10(p85 / p15), floored/bounded."""
    import numpy as np

    mask = (freqs >= lo) & (freqs < hi)
    bins = power[mask]
    if bins.size < 4:
        return 0.0
    region_peak = float(bins.max())
    if region_peak <= _EPS:
        return 0.0
    floor = region_peak * CONTRAST_FLOOR_REL
    hi_p = max(float(np.percentile(bins, CONTRAST_HI_PCT)), floor)
    lo_p = max(float(np.percentile(bins, CONTRAST_LO_PCT)), floor)
    return float(10.0 * np.log10(hi_p / lo_p))


# --------------------------------------------------------------------------------------------
# Scalar pipeline.
# --------------------------------------------------------------------------------------------


def null_scalars() -> dict:
    """The clarity keys, all ``None`` — emitted for duration-gated (too-short) material."""
    return {k: None for k in KEYS}


def clarity_scalars(y, sr: int) -> dict:
    """The six clarity scalars for a mono/multichannel signal (assumes duration OK)."""
    import numpy as np

    mono = _to_mono(y)
    Sp, freqs = _stft_power(mono, sr)     # (freq, frames) power, bin freqs
    power = Sp.mean(axis=1)               # time-averaged power per bin
    edges = _band_edges()

    bp = {name: _band_power(power, freqs, lo, hi) for name, (lo, hi) in edges.items()}
    total = sum(bp.values()) + _EPS
    # Floor each band at −60 dB of the total so a pathologically empty band can't drive a
    # ratio to a numerically meaningless 1e11 — bounds the ratios to a sane ±60 dB range.
    pfloor = total * BAND_FLOOR_REL
    lomid = max(bp.get("lomid", 0.0), pfloor)
    mid = max(bp.get("mid", 0.0), pfloor)
    uppermid = max(bp.get("uppermid", 0.0), pfloor)

    low_mid_masking_db = 10.0 * np.log10(lomid / mid)
    mud_presence_ratio = lomid / uppermid
    contrast_lo = _contrast_db(power, freqs, *LOW_HALF_HZ)
    contrast_hi = _contrast_db(power, freqs, *HIGH_HALF_HZ)
    presence_focus = bp.get("uppermid", 0.0) / total   # raw uppermid fraction (naturally 0–1)

    # presence transient: temporal crest of the presence band's PER-FRAME energy —
    # a percussive/transient presence spikes (high crest); a sustained presence is flat (~1).
    pmask = (freqs >= PRESENCE_LO) & (freqs < PRESENCE_HI)
    pf = Sp[pmask].sum(axis=0) if pmask.any() else np.zeros(1)
    pf_mean = float(pf.mean()) if pf.size else 0.0
    presence_transient = float(np.sqrt(float(pf.max()) / pf_mean)) if pf_mean > _EPS else 0.0

    return {
        "clarity.low_mid_masking_db": round(float(low_mid_masking_db), 3),
        "clarity.mud_presence_ratio": round(float(mud_presence_ratio), 4),
        "clarity.band_contrast_lo_db": round(float(contrast_lo), 3),
        "clarity.band_contrast_hi_db": round(float(contrast_hi), 3),
        "clarity.presence_focus_ratio": round(float(presence_focus), 5),
        "clarity.presence_transient_ratio": round(float(presence_transient), 4),
    }


# --------------------------------------------------------------------------------------------
# Frame-emitting entrypoint (the op behind `smpl clarity` and describe-all).
# --------------------------------------------------------------------------------------------


def clarity_audio_frame(audio_frame: dict) -> list[dict]:
    """Resolve an `audio` frame's PCM and emit its clarity feature frame (role ``clarity``).

    Duration-gated: material shorter than :data:`MIN_DURATION_S` emits every key as ``None``
    (``params.gated = True``). Carries ``of``/``op``/``op_version``/``params`` lineage; the
    caller passes the input frame through.
    """
    import soundfile as sf

    from smplstream import cas, frames as F

    src = cas.get_path(audio_frame["hash"])
    y, sr = sf.read(str(src), dtype="float32", always_2d=True)  # (n, ch)
    dur = y.shape[0] / float(sr) if sr else 0.0
    gated = dur < MIN_DURATION_S

    data = null_scalars() if gated else clarity_scalars(y.T, sr)

    params = {
        "min_duration_s": MIN_DURATION_S, "gated": bool(gated), "dur_s": round(float(dur), 4),
        "n_fft": N_FFT, "hop_length": HOP_LENGTH,
        "presence_band": [PRESENCE_LO, PRESENCE_HI],
        "low_half_hz": list(LOW_HALF_HZ), "high_half_hz": list(HIGH_HALF_HZ),
        "contrast_pct": [CONTRAST_LO_PCT, CONTRAST_HI_PCT],
    }
    return [
        F.feature_frame(
            data, role="clarity", of=audio_frame["id"], op=OP, op_version=OP_VERSION,
            params=params,
        )
    ]
