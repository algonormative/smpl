"""Loudness analysis (research §1; ticket vault-3vau).

Pure functions over a resolved audio path / numpy array, returning smplstream `feature`
(and optional `marker`) frame dicts. Heavy imports (pyloudnorm, scipy, numpy, soundfile)
stay inside the functions so a cold pipe stage starts fast.

Measurements (ITU-R BS.1770 / EBU R128 lineage):

- **Integrated LUFS** — K-weighted, gated perceived loudness, via `pyloudnorm.Meter`
  (BS.1770). Emitted as `loudness.integrated_lufs` (unit LUFS).
- **True-peak dBTP** — peak of the 4x-oversampled signal (per-channel), the inter-sample
  peak that sample-peak misses. Oversampling via `scipy.signal.resample_poly`. Emitted as
  `loudness.true_peak_dbtp` (unit dBTP), and optionally a `marker` frame locating the
  samples (in the ORIGINAL signal) that breach a configurable ceiling.
- **Max short-term LUFS** — the loudest 3 s sliding window (1 s hop), the peak of the
  short-term-loudness curve. Emitted as `loudness.max_short_term_lufs` (unit LUFS).

Feature keys are exactly those registered in feature-keys.md and owned by vault-3vau.
"""

from __future__ import annotations

import math
from typing import Optional

OP = "loudness"
OP_VERSION = "loudness@1"

# Oversampling factor for the true-peak estimate (4x per BS.1770-4 true-peak guidance).
TRUE_PEAK_OVERSAMPLE = 4

# Short-term loudness window/hop (EBU R128: 3 s window, measured on a sliding basis).
SHORT_TERM_WINDOW_S = 3.0
SHORT_TERM_HOP_S = 1.0

# Default ceiling for true-peak "over" markers (the conventional -1 dBTP streaming ceiling).
DEFAULT_OVER_CEILING_DBTP = -1.0

# Declared default table for the memo key (spec → *Parameter canonicalization*): omitted
# params are filled from THIS table for THIS op_version before hashing, so the key stays
# complete and stable even if a later version changes a default.
MEMO_DEFAULTS = {
    "true_peak_oversample": TRUE_PEAK_OVERSAMPLE,
    "short_term_window_s": SHORT_TERM_WINDOW_S,
    "short_term_hop_s": SHORT_TERM_HOP_S,
    "over_ceiling_dbtp": DEFAULT_OVER_CEILING_DBTP,
}


def _to_db(x: float) -> float:
    """Linear amplitude -> dBTP/dBFS. Returns -inf for silence."""
    return 20.0 * math.log10(x) if x > 0.0 else float("-inf")


def _db_round(x: float, ndigits: int = 2) -> Optional[float]:
    """Round a dB/LUFS value; map non-finite (silence) to None for clean JSON."""
    if x is None or not math.isfinite(x):
        return None
    return round(float(x), ndigits)


def _integrated_lufs(data, sr: int):
    """BS.1770 integrated (gated, K-weighted) loudness in LUFS, via pyloudnorm.

    pyloudnorm wants shape (samples,) for mono or (samples, channels) for multichannel.
    Returns a float (which may be -inf for digital silence / too-short signals).
    """
    import numpy as np
    import pyloudnorm as pyln

    meter = pyln.Meter(sr)  # BS.1770 meter at the native rate
    block = data[:, 0] if data.shape[1] == 1 else data
    # pyloudnorm raises if the signal is shorter than the 0.4 s block size; treat as silence.
    if data.shape[0] < int(round(0.4 * sr)) + 1:
        return float("-inf")
    val = meter.integrated_loudness(np.ascontiguousarray(block))
    return float(val)


def _true_peak(data, sr: int, ceiling_dbtp: float):
    """4x-oversampled true peak.

    Returns (true_peak_dbtp, over_points) where over_points is a list of marker dicts for
    samples (in ORIGINAL-rate index space) whose oversampled neighbourhood exceeds the
    ceiling. dBTP is the max over all channels.
    """
    import numpy as np
    from scipy.signal import resample_poly

    n = data.shape[0]
    ch = data.shape[1]
    peak_lin = 0.0
    over_orig_idx: set[int] = set()
    ceiling_lin = 10.0 ** (ceiling_dbtp / 20.0)

    for c in range(ch):
        up = resample_poly(data[:, c], TRUE_PEAK_OVERSAMPLE, 1)
        absup = np.abs(up)
        if absup.size:
            peak_lin = max(peak_lin, float(absup.max()))
        # Map any oversampled sample over the ceiling back to its original-rate index.
        breaches = np.nonzero(absup > ceiling_lin)[0]
        for up_idx in breaches:
            orig_idx = int(up_idx // TRUE_PEAK_OVERSAMPLE)
            if 0 <= orig_idx < n:
                over_orig_idx.add(orig_idx)

    true_peak_dbtp = _to_db(peak_lin)

    over_points = []
    for idx in sorted(over_orig_idx):
        over_points.append({"t": round(idx / sr, 6), "sample": int(idx), "label": "true-peak-over"})
    return true_peak_dbtp, over_points


def _max_short_term_lufs(data, sr: int):
    """Peak of the EBU R128 short-term (3 s window, 1 s hop) loudness curve, in LUFS.

    Returns -inf when the signal is shorter than one short-term window.
    """
    import numpy as np
    import pyloudnorm as pyln

    win = int(round(SHORT_TERM_WINDOW_S * sr))
    hop = int(round(SHORT_TERM_HOP_S * sr))
    n = data.shape[0]
    if win <= 0 or n < win:
        return float("-inf")

    meter = pyln.Meter(sr)
    block_is_mono = data.shape[1] == 1
    best = float("-inf")
    start = 0
    while start + win <= n:
        seg = data[start:start + win, 0] if block_is_mono else data[start:start + win, :]
        val = meter.integrated_loudness(np.ascontiguousarray(seg))
        if math.isfinite(val) and val > best:
            best = float(val)
        start += hop
    return best


def analyze_path(path, *, over_ceiling_dbtp: float = DEFAULT_OVER_CEILING_DBTP):
    """Compute loudness metrics for an audio file path.

    Returns a dict: {integrated_lufs, true_peak_dbtp, max_short_term_lufs, over_points, sr}.
    Values are floats (or None where the signal is too short / silent). `over_points` is a
    list of marker-point dicts in original-rate index space.
    """
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return analyze_array(data, sr, over_ceiling_dbtp=over_ceiling_dbtp)


def analyze_array(data, sr: int, *, over_ceiling_dbtp: float = DEFAULT_OVER_CEILING_DBTP):
    """Compute loudness metrics for a (samples, channels) float array at sample rate `sr`."""
    import numpy as np

    data = np.ascontiguousarray(np.atleast_2d(data))
    if data.ndim == 1:
        data = data[:, None]
    if data.shape[0] < data.shape[1]:
        # Defensive: callers should pass (samples, channels); never (channels, samples).
        pass

    integrated = _integrated_lufs(data, sr)
    true_peak_dbtp, over_points = _true_peak(data, sr, over_ceiling_dbtp)
    max_short_term = _max_short_term_lufs(data, sr)

    return {
        "integrated_lufs": integrated,
        "true_peak_dbtp": true_peak_dbtp,
        "max_short_term_lufs": max_short_term,
        "over_points": over_points,
        "sr": int(sr),
    }


def loudness_frames(audio_frame: dict, *, emit_markers: bool = True,
                    over_ceiling_dbtp: float = DEFAULT_OVER_CEILING_DBTP,
                    use_cache: bool = True) -> list[dict]:
    """Derive loudness frames for one `audio` frame.

    Returns a `feature` frame (always) carrying the three registered keys, plus an optional
    `marker` frame (role "true-peak-over") locating inter-sample overs when any are found.
    The aggregator (`smpl cat`) can call this hook the same way it calls describe.

    **Memoized** (spec → *Memoization*): keyed on
    ``(op, op_version, input audio hash, canonical params)`` with an empty env fingerprint
    (pure-Python/deterministic). A hit emits the cached measurement without decoding the
    audio at all; ``use_cache=False`` forces recompute and refreshes the entry. The feature
    frame's params carry ``cache_hit`` so a caller can see which path ran. ``emit_markers``
    is NOT part of the key — over-points fall out of the same true-peak pass, so they are
    always cached and the flag only gates emission.
    """
    from smplstream import cas, frames as F, memo, memostore

    params = {
        "true_peak_oversample": TRUE_PEAK_OVERSAMPLE,
        "short_term_window_s": SHORT_TERM_WINDOW_S,
        "short_term_hop_s": SHORT_TERM_HOP_S,
        "over_ceiling_dbtp": over_ceiling_dbtp,
    }
    mkey = memo.memo_key(
        OP, OP_VERSION, [audio_frame["hash"]], params=params, defaults=MEMO_DEFAULTS
    )

    payload = memostore.get_json(mkey) if use_cache else None
    cache_hit = payload is not None

    if not cache_hit:
        src = cas.get_path(audio_frame["hash"])
        res = analyze_path(src, over_ceiling_dbtp=over_ceiling_dbtp)
        payload = {
            "feat": {
                "loudness.integrated_lufs": _db_round(res["integrated_lufs"]),
                "loudness.true_peak_dbtp": _db_round(res["true_peak_dbtp"]),
                "loudness.max_short_term_lufs": _db_round(res["max_short_term_lufs"]),
            },
            "over_points": res["over_points"],
        }
        memostore.put_json(mkey, payload, op=OP, op_version=OP_VERSION)

    feat = payload["feat"]
    over_points = payload["over_points"]

    out: list[dict] = [
        F.feature_frame(
            feat,
            role="loudness",
            of=audio_frame["id"],
            op=OP,
            op_version=OP_VERSION,
            params={**params, "cache_hit": cache_hit},
        )
    ]

    if emit_markers and over_points:
        out.append(
            F.marker_frame(
                over_points,
                role="true-peak-over",
                of=audio_frame["id"],
                op=OP,
                op_version=OP_VERSION,
                params={"over_ceiling_dbtp": over_ceiling_dbtp,
                        "true_peak_oversample": TRUE_PEAK_OVERSAMPLE},
            )
        )

    return out
