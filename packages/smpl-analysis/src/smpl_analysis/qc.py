"""Technical QC / defect detection (research §4; ticket vault-1e9a).

The "is this sample technically clean and usable" axis — the deterministic DSP top-6 that
curators and engineers reject for. Pure functions over a resolved audio path / np array
returning smplstream frame dicts:

  1. clipping / true-peak     → `qc.clipping.detected` pass/fail, keyed off an ITU-R BS.1770
                                 true-peak we compute internally; the `loudness.true_peak_dbtp`
                                 dBTP *measurement* is owned by the loudness tier, so we do NOT
                                 re-emit it under a qc.* key (one measurement, one owner)
  2. phase correlation        → `qc.phase.correlation` (−1..1, stereo mono-compatibility)
  3. DC offset                → `qc.dc_offset_dbfs`
  4. noise floor / SNR        → `qc.snr_db`
  5. clicks / gaps            → `marker` frames at defect locations
  6. lossy-origin via cutoff  → `qc.lossy.spectral_cutoff_hz`, `qc.lossy.expected_nyquist_hz`,
                                 `qc.lossy.confidence` (average-FFT brickwall detector)

All measurement keys come from the registry (feature-keys.md, owner vault-1e9a). Heavy
imports (numpy/scipy/soundfile) stay inside functions so cold pipe stages start fast.
"""

from __future__ import annotations

from typing import Any, Optional

OP = "qc"
OP_VERSION = "qc@1"

# A sample at or above this absolute true-peak (in dBTP) is flagged as clipping.
_CLIP_DBTP_THRESHOLD = -0.1
# Fraction of the spectral energy that must fall below the cutoff for it to count as a
# brickwall (LAME/MP3 low-passes hard, so almost all energy sits below the knee).
_LOSSY_ENERGY_FRAC = 0.999
# How far below Nyquist the knee must sit (fraction) before we even consider a brickwall —
# full-band content has its cutoff right at Nyquist and must score ~0.
_LOSSY_MIN_KNEE_FRAC = 0.05
# Confidence saturates once the band well above the knee is this many dB below the in-band
# level (a real LAME/AAC brickwall buries the upper band at the noise floor: −60 dB+).
_LOSSY_FLOOR_DB = 60.0
# Shelf-shape evidence (vault-3t1l): a lossy brickwall must look like a codec cutoff, not
# merely "no highs". Three independent requirements, so a genuinely sub-only synth (a tight
# kick, a 50 Hz sine) — all energy below the knee, dead above it — no longer scores 1.0.
#   1. PLAUSIBILITY: a real MP3/AAC cutoff sits high (128 kbps LAME ≈ 16 kHz; even 96 kbps is
#      ~15 kHz). A "cutoff" down in the bass/mids is band-limited *content*, never a codec.
_LOSSY_MIN_PLAUSIBLE_CUTOFF_HZ = 10000.0
#   2. SLOPE: a brickwall drops hard right at its EDGE — hard relative to the roll-off the
#      spectrum already had below it. Measure the dB step across a third octave centred on
#      the edge, minus the same step a third octave lower; a codec buries it, natural
#      roll-off is gentle (a few dB/third-octave) and gentle in the same way either side of
#      wherever you look. Confidence ramps with that excess and a cliff must clear this.
_LOSSY_SLOPE_DB = 24.0

# Declared default table for the memo key (spec → *Parameter canonicalization*): omitted
# params are filled from THIS table for THIS op_version before hashing. Only true *op*
# params belong here — `sr`/`ch` are properties of the input (already covered by the input
# hash) and ride on the emitted frame, not on the key.
MEMO_DEFAULTS = {"clip_threshold_dbtp": _CLIP_DBTP_THRESHOLD}


def _dbfs(x: float) -> float:
    import math

    return 20.0 * math.log10(x) if x > 0 else float("-inf")


def _load(path: str):
    """Decode to (samples [n, ch] float32, sr). No resampling / normalization."""
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data, int(sr)


def true_peak_dbtp(samples, sr: int) -> float:
    """ITU-R BS.1770 true-peak estimate via 4x polyphase oversampling, in dBTP.

    Per-channel max of the oversampled signal; the overall true-peak is the channel max.
    We compute it here only to key the clipping flag — the canonical `loudness.true_peak_dbtp`
    feature key stays owned by the loudness tier (vault-3vau).
    """
    import numpy as np
    from scipy.signal import resample_poly

    if samples.size == 0:
        return float("-inf")
    peak = 0.0
    for ch in range(samples.shape[1]):
        up = resample_poly(samples[:, ch].astype(np.float64), 4, 1)
        peak = max(peak, float(np.max(np.abs(up))) if up.size else 0.0)
    # also consider the raw sample peak (oversampling only ever raises it, but guard size==1)
    peak = max(peak, float(np.max(np.abs(samples))))
    return _dbfs(peak)


def phase_correlation(samples) -> Optional[float]:
    """Inter-channel (L/R) Pearson correlation in [−1, 1]. None for mono / degenerate.

    +1 = perfectly mono-compatible (in phase); 0 = decorrelated/wide; −1 = anti-phase
    (cancels to silence in mono). Computed on the first two channels.
    """
    import numpy as np

    if samples.shape[1] < 2:
        return None
    left = samples[:, 0].astype(np.float64)
    right = samples[:, 1].astype(np.float64)
    ls = left - left.mean()
    rs = right - right.mean()
    denom = float(np.sqrt(np.sum(ls * ls) * np.sum(rs * rs)))
    if denom <= 0.0:
        return None
    return float(np.clip(np.sum(ls * rs) / denom, -1.0, 1.0))


def dc_offset_dbfs(samples) -> float:
    """Worst-channel DC offset (mean sample value) expressed in dBFS."""
    import numpy as np

    if samples.size == 0:
        return float("-inf")
    means = np.abs(samples.mean(axis=0))
    return _dbfs(float(np.max(means)))


def snr_db(samples, sr: int) -> Optional[float]:
    """Crude dynamic-range SNR: loud-passage RMS vs the quietest-passage RMS (the noise floor).

    Frame the mono mix and take per-frame RMS. The signal level is the 90th-percentile frame
    RMS; the noise floor is the median of the quietest 5% of frames. SNR = signal − noise in
    dB, clamped to a 120 dB ceiling.

    This is the cheap "how far does the body sit above the quiet/silent regions" measure —
    it is most meaningful for material that HAS quiet passages (decay, gaps, head/tail
    silence), where the floor is the actual noise. A perfectly steady continuous signal with
    no quiet region reports a small value by construction (every frame is equally loud — there
    is no quiet passage to contrast against); that is a correct statement about the signal, not
    a hidden failure. Returns None only when the clip is too short to frame or is pure digital
    silence.
    """
    import numpy as np

    if samples.size == 0:
        return None
    mono = samples.mean(axis=1).astype(np.float64)
    win = max(256, int(sr * 0.02))  # ~20 ms frames
    hop = win // 2
    if mono.shape[0] < win:
        return None
    n_frames = 1 + (mono.shape[0] - win) // hop
    rms = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        seg = mono[i * hop : i * hop + win]
        rms[i] = float(np.sqrt(np.mean(seg * seg)))
    # Keep silent (zero-RMS) frames: they ARE the noise floor we want to measure against.
    # Dropping them would measure the floor over the loud body and collapse SNR to ~0.
    if rms.size < 2:
        return None
    signal = float(np.quantile(rms, 0.90))
    if signal <= 0:
        return None  # whole clip is silence
    floor_frames = rms[rms <= np.quantile(rms, 0.05)]
    noise = float(np.median(floor_frames)) if floor_frames.size else float(rms.min())
    if noise <= 0:
        return 120.0  # noise-free floor (digital silence) → clean to the measurement ceiling
    snr = _dbfs(signal) - _dbfs(noise)
    return round(min(snr, 120.0), 2)


def detect_clicks(samples, sr: int, *, max_points: int = 64) -> list[dict]:
    """Click / discontinuity detection via 2nd-difference outliers on the mono mix.

    A click is an isolated sample-to-sample jump far outside the local distribution. Returns
    marker points ({t, sample, label}) at click locations (deduped to one per ~5 ms cluster).
    """
    import numpy as np

    if samples.shape[0] < 3:
        return []
    mono = samples.mean(axis=1).astype(np.float64)
    d2 = np.abs(np.diff(mono, n=2))
    if d2.size == 0:
        return []
    med = float(np.median(d2))
    mad = float(np.median(np.abs(d2 - med))) or 1e-12
    # robust z-score; a true click sits many MADs above the body of the signal
    thresh = med + 12.0 * mad
    # require an absolute jump too, so quiet-but-smooth material doesn't trip
    idx = np.where((d2 > thresh) & (d2 > 0.05))[0]
    if idx.size == 0:
        return []
    points: list[dict] = []
    min_gap = max(1, int(sr * 0.005))  # collapse clusters within 5 ms
    last = -(min_gap + 1)
    for i in idx:
        sample = int(i) + 1  # +1: diff(n=2) index → original sample
        if sample - last < min_gap:
            continue
        last = sample
        points.append({"t": round(sample / sr, 6), "sample": sample, "label": "click"})
        if len(points) >= max_points:
            break
    return points


def detect_gaps(samples, sr: int, *, max_points: int = 64) -> list[dict]:
    """Internal silent-gap / dropout detection.

    Flags runs of >=10 ms of near-digital-silence that are bounded on BOTH sides by signal
    (so leading/trailing silence is not a defect). Returns marker points ({t, sample, dur,
    label}) at each gap onset.
    """
    import numpy as np

    if samples.shape[0] < 3:
        return []
    mono = np.abs(samples).max(axis=1).astype(np.float64)
    silent = mono < 10 ** (-60.0 / 20.0)  # below −60 dBFS
    if not silent.any() or silent.all():
        return []
    min_len = max(1, int(sr * 0.010))  # >=10 ms
    points: list[dict] = []
    n = mono.shape[0]
    i = 0
    while i < n:
        if not silent[i]:
            i += 1
            continue
        j = i
        while j < n and silent[j]:
            j += 1
        run = j - i
        bounded = i > 0 and j < n  # signal on both sides → an internal dropout
        if run >= min_len and bounded:
            points.append(
                {
                    "t": round(i / sr, 6),
                    "sample": int(i),
                    "dur": round(run / sr, 6),
                    "label": "gap",
                }
            )
            if len(points) >= max_points:
                break
        i = j
    return points


def _third_octave_steps_db(power):
    """dB step across a third octave centred on every bin of an averaged power spectrum.

    For bin *i*, compares the mean power of the third octave ABOVE it (bins `i+1 …
    floor(i·2^⅓)`) against the third octave BELOW it (bins `ceil(i/2^⅓) … i`). Negative =
    the spectrum drops there; a codec brickwall buries the step, natural roll-off is gentle.

    Returns a full-length array (NaN where the step is not evaluable — an empty band, or no
    live energy below). Vectorized over a prefix sum, so scanning every candidate is cheap.
    """
    import numpy as np

    third = 2.0 ** (1.0 / 3.0)
    n = len(power)
    steps = np.full(n, np.nan, dtype=np.float64)
    if n < 4:
        return steps
    csum = np.concatenate([[0.0], np.cumsum(power)])
    idx = np.arange(1, n - 1)
    lo = np.ceil(idx / third).astype(int)
    hi = np.minimum(np.floor(idx * third).astype(int), n - 1)
    below_n = idx - lo + 1
    above_n = hi - idx
    below_sum = csum[idx + 1] - csum[lo]
    above_sum = csum[hi + 1] - csum[idx + 1]
    ok = (below_n > 0) & (above_n > 0) & (below_sum > 0)
    if not ok.any():
        return steps
    i_ok = idx[ok]
    above_avg = above_sum[ok] / above_n[ok]
    below_avg = below_sum[ok] / below_n[ok]
    steps[i_ok] = 10.0 * np.log10((above_avg + 1e-30) / below_avg)
    return steps


def _brickwall_edge(power, start_idx: int, stop_idx: int):
    """Locate the brickwall EDGE bin and score how much of a cliff it is.

    The score is the third-octave step at the candidate MINUS the step one third octave
    below it — the *excess* over the roll-off the content already exhibits, not the raw
    steepness. That distinction is what makes scanning safe: a brickwall is a departure from
    a spectrum's own trend, whereas a dark-but-natural source rolls off at a similar rate
    everywhere and so has an excess near zero no matter where you look. The baseline sits a
    full third octave down precisely so its own upper band stops at the candidate and cannot
    already "see" the wall.

    The edge is the LOWEST bin whose excess clears `_LOSSY_SLOPE_DB` — where the spectrum
    first falls off a cliff, not where the fall is deepest (past a hard wall the leakage
    skirt keeps dropping, so an argmax would report an edge well above the real one). Absent
    a cliff there is no wall; the steepest excess is returned so the caller still scores the
    best available evidence, which for gentle roll-off is correctly weak.

    Returns `(idx, excess_db)`, or `(None, None)` when the range is not evaluable.
    """
    import numpy as np

    third = 2.0 ** (1.0 / 3.0)
    n = len(power)
    stop_idx = min(stop_idx, n - 2)
    if start_idx < 1 or stop_idx < start_idx:
        return None, None
    steps = _third_octave_steps_db(power)
    idx = np.arange(start_idx, stop_idx + 1)
    base = np.clip(np.round(idx / third).astype(int), 1, n - 2)
    excess = steps[idx] - steps[base]
    valid = np.isfinite(excess)
    if not valid.any():
        return None, None
    cliff = valid & (excess <= -_LOSSY_SLOPE_DB)
    if cliff.any():
        pick = int(np.argmax(cliff))
    else:
        pick = int(np.argmin(np.where(valid, excess, np.inf)))
    return int(idx[pick]), float(excess[pick])


def lossy_origin(samples, sr: int) -> dict[str, Any]:
    """Average-FFT brickwall detector for lossy (MP3/AAC) origin.

    Averages magnitude spectra over the file, then measures in two stages:

      1. **Knee** — the lowest frequency under which `_LOSSY_ENERGY_FRAC` of the energy
         already lives. This is the cheap *gate*: full-band content keeps its knee at
         Nyquist, and a knee down in the bass/mids is band-limited content, never a codec.
      2. **Edge** — the actual brickwall, located by scanning the third-octave step upward
         from the knee for the first cliff (`_brickwall_edge`). The slope and dead-shelf
         evidence is then measured AT the edge.

    Stage 2 exists because the energy knee is NOT the wall (vault-3fb7b): on content whose
    natural HF decay is steep — a techno loop with most of its energy under 1 kHz — the
    0.999-energy knee lands a kilohertz or more BELOW the encoder ceiling, so evidence
    gathered at the knee straddles live content and reads as a gentle slope even though a
    hard wall sits just above. `qc.lossy.spectral_cutoff_hz` therefore reports the located
    edge once the gates pass (the frequency where the usable band actually ends, which is
    what the key has always meant) and the knee on the gated-out paths, where no wall was
    looked for.

    A hard brickwall well below Nyquist (e.g. ~16 kHz for 128 kbps LAME on 44.1 kHz
    material) flags a likely lossy origin. Returns the three registry keys:
    `qc.lossy.spectral_cutoff_hz`, `qc.lossy.expected_nyquist_hz`, `qc.lossy.confidence`
    (0..1). A high-value FLAG, not proof — natural band-limiting and SBR/AAC+ confound it.
    """
    import numpy as np

    nyquist = sr / 2.0
    out = {
        "qc.lossy.spectral_cutoff_hz": round(nyquist, 1),
        "qc.lossy.expected_nyquist_hz": round(nyquist, 1),
        "qc.lossy.confidence": 0.0,
    }
    mono = samples.mean(axis=1).astype(np.float64)
    n = mono.shape[0]
    nfft = 4096
    if n < nfft:
        return out  # too short to resolve a cutoff reliably

    win = np.hanning(nfft)
    hop = nfft // 2
    n_frames = 1 + (n - nfft) // hop
    acc = np.zeros(nfft // 2 + 1, dtype=np.float64)
    for i in range(n_frames):
        seg = mono[i * hop : i * hop + nfft] * win
        acc += np.abs(np.fft.rfft(seg))
    avg = acc / n_frames
    power = avg * avg
    total = float(power.sum())
    if total <= 0:
        return out

    freqs = np.fft.rfftfreq(nfft, d=1.0 / sr)
    # knee = lowest freq under which `_LOSSY_ENERGY_FRAC` of the energy already lives
    cumfrac = np.cumsum(power) / total
    knee_idx = int(np.searchsorted(cumfrac, _LOSSY_ENERGY_FRAC))
    knee_idx = min(knee_idx, len(freqs) - 1)
    knee = float(freqs[knee_idx])
    out["qc.lossy.spectral_cutoff_hz"] = round(knee, 1)

    # Knee-below-Nyquist is a GATE, not a linear penalty: full-band content puts its cutoff
    # at Nyquist (knee_frac ~0) and is rejected; anything clearly below Nyquist is eligible,
    # and confidence is then driven by how DEAD the band well above the knee is. (Penalizing
    # a 16 kHz brickwall for being "only" 27% below a 22 kHz Nyquist would wrongly suppress
    # the single most valuable forensic flag.)
    knee_frac = (nyquist - knee) / nyquist
    if knee_frac < _LOSSY_MIN_KNEE_FRAC or knee_idx >= len(freqs) - 2:
        return out  # full-band content → not lossy-flagged

    # --- shelf-shape evidence, not mere absence-of-highs (vault-3t1l) -----------------------
    # GATE 1 — plausibility: a codec cutoff sits high. A knee down in the bass/mids is just
    # band-limited content (a sub-only kick keys its 0.999-energy knee at ~140 Hz), never an
    # MP3 brickwall. Report the knee for transparency but do not flag.
    if knee < _LOSSY_MIN_PLAUSIBLE_CUTOFF_HZ:
        return out

    # LOCATE THE EDGE (vault-3fb7b) — the wall sits at or above the energy knee (a wall means
    # no energy above it, so the 0.999 knee cannot outrun it by more than the codec's own
    # leakage). Scan upward from the knee for the first cliff.
    #
    # The scan stops where a full third octave no longer fits below Nyquist. Above that point
    # the comparison band is truncated AND every sample rate's own band edge lives there —
    # anti-alias filters, resampler walls, the Nyquist zero every digital low-pass carries —
    # so the spectrum's collapse accelerates and the steepest, cliff-shaped point in range is
    # ALWAYS up there, whatever the content. Scanning into it turns any steep-but-natural
    # roll-off into a false brickwall. When the knee itself already sits that high (a wall
    # near Nyquist, e.g. a 320 kbps ceiling), the knee is the only candidate and the
    # measurement degenerates to the single-point one this detector has always made there.
    df = sr / nfft
    max_edge_idx = int((nyquist / 2.0 ** (1.0 / 3.0)) // df)
    edge_idx, slope_db = _brickwall_edge(power, knee_idx, max(max_edge_idx, knee_idx))
    if edge_idx is None or slope_db is None:
        return out
    cutoff = float(freqs[edge_idx])
    out["qc.lossy.spectral_cutoff_hz"] = round(cutoff, 1)

    # GATE 2 — slope AT THE EDGE: a brickwall drops hard there, and hard RELATIVE to the
    # spectrum's own roll-off (`_brickwall_edge` scores that excess). A codec buries the
    # step; natural roll-off is gentle and, crucially, gentle in the same way either side of
    # any frequency you pick. This is what distinguishes a hard cutoff from a spectrum that
    # merely tapers (a pad, a filtered synth) or has no highs at all (a sub kick).
    slope_conf = min(1.0, max(0.0, (-slope_db) / _LOSSY_SLOPE_DB))

    in_band = power[: edge_idx + 1]
    # measure deadness in the upper HALF of the edge→Nyquist span (clear of the edge skirt),
    # where a true brickwall is at the noise floor but natural roll-off still carries energy
    hi_start = edge_idx + 1 + (len(freqs) - 1 - edge_idx) // 2
    above = power[hi_start:]
    in_band_avg = float(in_band.mean()) if in_band.size else 0.0
    above_avg = float(above.mean()) if above.size else 0.0
    if in_band_avg <= 0 or above.size == 0:
        return out
    floor_db = float(10.0 * np.log10((above_avg + 1e-30) / in_band_avg))
    # SHELF term — how far below the in-band level the sustained upper region sits.
    shelf_conf = min(1.0, max(0.0, (-floor_db) / _LOSSY_FLOOR_DB))

    # confidence needs ALL THREE: a plausible cutoff (gated above), a steep slope, AND a
    # sustained dead shelf. The product means either a gentle slope or a live upper band
    # pulls it down — it never reaches 1.0 on absence-of-highs alone.
    confidence = round(slope_conf * shelf_conf, 3)
    out["qc.lossy.confidence"] = float(confidence)
    return out


def qc_audio_frame(audio_frame: dict, *, want_markers: bool = True,
                   use_cache: bool = True) -> list[dict]:
    """Run the deterministic QC top-6 over one `audio` frame; return derived frames.

    Emits ONE `feature` frame carrying all the QC scalars (registry `qc.*` keys plus
    `qc.clipping.detected`), plus `marker` frames for click/gap locations.
    Lineage (`of`/`op`/`op_version`/`params`) is set on every derived frame.

    **Memoized** (spec → *Memoization*): keyed on
    ``(op, op_version, input audio hash, canonical params)`` with an empty env fingerprint
    (pure-Python/deterministic). A hit skips the decode and all six measurements;
    ``use_cache=False`` forces recompute and refreshes the entry. ``params.cache_hit``
    records which path ran. ``want_markers`` is NOT part of the key: click/gap detection is
    cheap next to the true-peak and FFT passes, so markers are always computed and cached
    and the flag only gates emission — a ``--no-markers`` run and a full run share one entry.
    """
    from smplstream import cas, frames as F, memo, memostore

    of = audio_frame.get("id")
    params = {"clip_threshold_dbtp": _CLIP_DBTP_THRESHOLD}
    mkey = memo.memo_key(
        OP, OP_VERSION, [audio_frame["hash"]], params=params, defaults=MEMO_DEFAULTS
    )

    payload = memostore.get_json(mkey) if use_cache else None
    cache_hit = payload is not None

    if not cache_hit:
        src = cas.get_path(audio_frame["hash"])
        samples, sr = _load(src)

        tp = true_peak_dbtp(samples, sr)
        clipping = bool(tp >= _CLIP_DBTP_THRESHOLD)
        corr = phase_correlation(samples)
        dc = dc_offset_dbfs(samples)
        snr = snr_db(samples, sr)
        lossy = lossy_origin(samples, sr)

        feat: dict[str, Any] = {
            # Clipping pass/fail ONLY. true-peak is computed internally to decide it, but the
            # dBTP *measurement* is owned solely by the loudness tier (`loudness.true_peak_dbtp`)
            # — "one measurement, one owner" (feature-keys.md). qc does NOT re-emit it under a
            # qc.* key; run `smpl loudness` for the dBTP number.
            "qc.clipping.detected": clipping,
            "qc.dc_offset_dbfs": round(dc, 2) if dc != float("-inf") else None,
            "qc.snr_db": snr,
            **lossy,
        }
        if corr is not None:
            feat["qc.phase.correlation"] = round(corr, 4)

        payload = {
            "feat": feat,
            "sr": sr,
            "ch": int(samples.shape[1]),
            "clicks": detect_clicks(samples, sr),
            "gaps": detect_gaps(samples, sr),
        }
        memostore.put_json(mkey, payload, op=OP, op_version=OP_VERSION)

    feat = payload["feat"]
    frame_params = {
        **params, "sr": payload["sr"], "ch": payload["ch"], "cache_hit": cache_hit
    }
    out: list[dict] = [
        F.feature_frame(feat, role="qc", of=of, op=OP, op_version=OP_VERSION,
                        params=frame_params)
    ]

    if want_markers:
        clicks = payload["clicks"]
        gaps = payload["gaps"]
        if clicks:
            out.append(
                F.marker_frame(clicks, role="defect", of=of, op=OP, op_version=OP_VERSION)
            )
        if gaps:
            out.append(
                F.marker_frame(gaps, role="defect", of=of, op=OP, op_version=OP_VERSION)
            )
    return out
