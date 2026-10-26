"""Full-length annotated listening sheet (vault-16dd.3) — one PNG per clip plus its numbers.

Contact sheets were the single best catch on generated kits (drones, unrouted noise, DC that
every metric passed), and a 4 s window hid all bed evolution. So the sheet covers the WHOLE
clip, every lane time-aligned on one shared x axis:

- **ruler**: bar numbers (from intent, ``--bpm``, or the beat tracker), section letters
- **loudness**: momentary (400 ms) and short-term (3 s) K-weighted loudness (BS.1770), in
  LUFS on a fixed range. Windows are CENTRED (not EBU's trailing windows) so the lane lines up
  with the spectrogram; windows at the clip edges are truncated.
- **CQT**: constant-Q spectrogram calibrated to dBFS of a sinusoid (a full-scale sine reads
  0 dB in its bin), fixed ``db_range`` and colormap, beat lines + bar lines burned in.
- **onsets / sections**: normalized onset strength, measured onsets, the generator's
  intended hits (intent), Foote-novelty section boundaries from :func:`smpl_analysis.arc.segment`.
- **chroma**: per-frame pitch-class share; the intended pitch-class set is marked in the row
  labels when intent exists.
- **six bands**: level per standardized band (feature-keys.md "Standardized band edges",
  reused from :data:`smpl_analysis.width.BANDS`), dBFS of a sinusoid, fixed ``db_range``.
- **stereo correlation**: windowed L/R Pearson correlation, −1..1 (gaps where silent).
- **SSM**: MFCC cosine self-similarity (the matrix arc's segmentation reads), for clips
  ≥ ``ssm_min_s`` (default 30 s).

Vision models read labels and large shapes, not fine pixels: lane labels are large, there is
no colorbar to estimate from (each lane label states its fixed range), and every number that
matters is in the sidecar document — per-bar summaries (loudness, peak, six bands,
correlation, onsets, intended hits, pitch-class in-set share) plus global numbers
(integrated LUFS, peak, DC offset). Cite the sidecar, not pixel readings.

``--ab`` mode (:func:`sheet_ab_frames`) loudness-matches clip B to clip A (integrated LUFS;
the gain is recorded), stacks both on shared axes and adds a CQT difference panel (B − A dB,
diverging colormap, fixed ``diff_range``), with per-bar deltas in the sidecar.

The op measures and draws; it judges nothing (layering rule). Heavy imports stay inside
functions; matplotlib uses the Agg backend.
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path

OP = "sheet"
OP_VERSION = "sheet@1"

# --- Fixed render scales (flags with these defaults; one per run so sheets compare). ---
DEFAULT_DB_RANGE = (-90.0, -10.0)     # CQT + band lanes, dBFS of a sinusoid
DEFAULT_LUFS_RANGE = (-60.0, 0.0)     # loudness lane
DEFAULT_CMAP = "magma"
DEFAULT_DIFF_RANGE_DB = 24.0          # A/B difference panel: ±this many dB
DEFAULT_MAX_WIDTH_PX = 1600           # image width cap (long clips are pooled to fit)
SSM_MIN_S = 30.0                      # SSM panel only for clips at least this long
DPI = 100

# --- Analysis constants. ---
MOMENTARY_S = 0.4                     # BS.1770 / EBU R128 momentary window
SHORT_TERM_S = 3.0                    # EBU R128 short-term window
CQT_FMIN_NOTE = "C1"                  # bin 0 = C, so chroma folds straight off the CQT
CQT_BPO = 12
CQT_MAX_BINS = 96                     # 8 octaves, C1..B8, capped below Nyquist
CQT_TARGET_COLS = 1200                # spectrogram columns before pooling to the image width
CQT_MIN_PAD_S = 1.0                   # zero-pad short clips so the low octaves have support
ONSET_HOP_S = 0.0107                  # onset-strength hop ≈ 512 samples at 48 kHz
BEAT_MIN_S = 4.0                      # never beat-track a clip shorter than this
QUIET_DB = -120.0                     # sidecar levels at/below this are reported as None
SILENT_MS = 1e-7                      # mean square below this (−70 dBFS) is "silent"
MAX_RULER_LABELS = 32
MIN_BEAT_PX = 8.0                     # draw beat lines only when they are this far apart
SECTION_SPAN_S = 20.0                 # ~one section per this many seconds (2..8)
INLINE_BUDGET = 56 * 1024             # sidecar doc inline in the feature frame up to this

_EPS = 1e-12
_FLOOR_DB = -140.0
PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# ---------------------------------------------------------------------------------------------
# Small numeric helpers.
# ---------------------------------------------------------------------------------------------


def _db_pow(ms):
    """Mean square → dB of a sinusoid (a full-scale sine's mean square is 0.5 → 0 dB)."""
    import numpy as np

    return 10.0 * np.log10(np.maximum(2.0 * np.asarray(ms, dtype="float64"), 10 ** (_FLOOR_DB / 10)))


def _lufs_of(ms_sum):
    """Channel-summed K-weighted mean square → LUFS (BS.1770: −0.691 + 10 log10 Σ G·z)."""
    import numpy as np

    return -0.691 + 10.0 * np.log10(np.maximum(np.asarray(ms_sum, dtype="float64"), 1e-14))


def _r(x, nd: int = 2):
    """Round for JSON; non-finite → None."""
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def _block_ms(x, hop: int):
    """Mean square of ``x`` (n,) or (n, ch) in non-overlapping blocks of ``hop`` samples."""
    import numpy as np

    x = np.asarray(x, dtype="float64")
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    nb = max(1, math.ceil(n / hop))
    pad = nb * hop - n
    sq = np.pad(x * x, ((0, pad), (0, 0)))
    counts = np.full(nb, hop, dtype="float64")
    counts[-1] = hop - pad if hop - pad > 0 else hop
    return sq.reshape(nb, hop, -1).sum(axis=1) / counts[:, None]  # (nb, ch)


def _centred_mean(v, width: int):
    """Centred moving mean of a 1-D array; edge windows are truncated to the available data."""
    import numpy as np

    v = np.asarray(v, dtype="float64")
    width = max(1, int(width))
    c = np.concatenate([[0.0], np.cumsum(v)])
    idx = np.arange(v.size)
    lo = np.clip(idx - width // 2, 0, v.size)
    hi = np.clip(idx - width // 2 + width, 0, v.size)
    return (c[hi] - c[lo]) / np.maximum(hi - lo, 1)


def _pool(m, max_n: int, how: str = "mean"):
    """Pool the last axis of ``m`` down to at most ``max_n`` columns. Returns (pooled, factor)."""
    import numpy as np

    m = np.asarray(m, dtype="float64")
    n = m.shape[-1]
    if n <= max_n:
        return m, 1
    f = math.ceil(n / max_n)
    pad = (-n) % f
    if pad:
        widths = [(0, 0)] * (m.ndim - 1) + [(0, pad)]
        m = np.pad(m, widths, mode="edge")
    m = m.reshape(*m.shape[:-1], m.shape[-1] // f, f)
    red = {"mean": np.nanmean, "max": np.nanmax, "min": np.nanmin}[how]
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns stay NaN
        return red(m, axis=-1), f


def _k_weight(samples, sr: int):
    """K-weight every channel with pyloudnorm's BS.1770 filters. Returns (n, ch) float64."""
    import numpy as np
    import pyloudnorm as pyln

    z = np.asarray(samples, dtype="float64").copy()
    meter = pyln.Meter(sr)
    for filt in meter._filters.values():  # high shelf, then high pass (BS.1770 stage order)
        z = filt.apply_filter(z.T).T if z.ndim > 1 else filt.apply_filter(z)
    return z


def integrated_lufs(samples, sr: int) -> float | None:
    """Gated integrated loudness (BS.1770); None when shorter than one 400 ms block."""
    import numpy as np
    import pyloudnorm as pyln

    s = np.asarray(samples, dtype="float64")
    if s.ndim == 1:
        s = s[:, None]
    if s.shape[0] < round(0.4 * sr) + 1:
        return None
    val = pyln.Meter(sr).integrated_loudness(s[:, 0] if s.shape[1] == 1 else s)
    return float(val) if math.isfinite(val) else None


def ungated_lufs(samples, sr: int) -> float | None:
    """Ungated K-weighted loudness of the whole clip (works on one-shots shorter than 400 ms)."""
    import numpy as np

    z = _k_weight(samples, sr)
    if z.ndim == 1:
        z = z[:, None]
    ms = float(np.mean(z * z, axis=0).sum()) if z.size else 0.0
    return float(_lufs_of(ms)) if ms > 0 else None


# ---------------------------------------------------------------------------------------------
# Grid: bars and beats from intent, a --bpm flag, or the beat tracker.
# ---------------------------------------------------------------------------------------------


def make_grid(mono, sr: int, dur: float, *, intent: dict | None = None,
              bpm: float | None = None, beats_per_bar: float | None = None,
              detect: bool = True) -> dict:
    """The bar grid the sheet is drawn and summarized on.

    Priority: explicit ``bpm`` (source ``flag``), intent ``bpm`` (``intent``; bar 1 starts at
    sample 0, as generators place it), the beat tracker on clips ≥ ``BEAT_MIN_S``
    (``detected``), else equal segments (``none``). Returns ``{source, unit, bpm,
    beats_per_bar, beats, bars}`` where ``bars`` is a list of ``(label, t0, t1)``.
    """
    import numpy as np

    intent = intent or {}
    bpb = float(beats_per_bar or intent.get("beats_per_bar") or 4.0)
    source, the_bpm = None, None
    if bpm:
        source, the_bpm = "flag", float(bpm)
    elif intent.get("bpm"):
        source, the_bpm = "intent", float(intent["bpm"])

    if the_bpm:
        beat_s = 60.0 / the_bpm
        bar_s = beat_s * bpb
        n_bars = max(1, math.ceil(dur / bar_s - 1e-6))
        bars = [(str(i + 1), i * bar_s, min((i + 1) * bar_s, dur)) for i in range(n_bars)]
        beats = [float(t) for t in np.arange(0.0, dur, beat_s)]
        return {"source": source, "unit": "bar", "bpm": the_bpm, "beats_per_bar": bpb,
                "beats": beats, "bars": bars}

    if detect and dur >= BEAT_MIN_S:
        from . import beats as B

        try:
            g = B.beat_grid(mono, sr)
        except Exception:
            g = None
        if g and len(g["downbeats"]) >= 2:
            beats = [s / sr for s in g["beats"]]
            downs = [s / sr for s in g["downbeats"]]
            meter = float(str(g["features"].get("rhythm.time_signature", "4/4")).split("/")[0])
            bars = []
            if downs[0] > 0.05:
                bars.append(("0", 0.0, downs[0]))
            for i, t0 in enumerate(downs):
                t1 = downs[i + 1] if i + 1 < len(downs) else dur
                if t1 > t0:
                    bars.append((str(i + 1), t0, t1))
            return {"source": "detected", "unit": "bar", "bpm": float(g["features"]["rhythm.bpm"]),
                    "beats_per_bar": meter, "beats": beats, "bars": bars}

    n_seg = int(min(64, max(8, math.ceil(dur / 2.0))))
    edges = np.linspace(0.0, dur, n_seg + 1)
    bars = [(str(i + 1), float(edges[i]), float(edges[i + 1])) for i in range(n_seg)]
    return {"source": "none", "unit": "segment", "bpm": None, "beats_per_bar": None,
            "beats": [], "bars": bars}


# ---------------------------------------------------------------------------------------------
# Analysis: every lane's series + the per-bar sidecar numbers.
# ---------------------------------------------------------------------------------------------


def _cqt_db(mono, sr: int, hop: int, dur: float):
    """CQT in dB of a sinusoid (calibrated per bin by its filter length) + frame times."""
    import warnings

    import librosa
    import numpy as np

    fmin = float(librosa.note_to_hz(CQT_FMIN_NOTE))
    n_bins = int(min(CQT_MAX_BINS, math.floor(CQT_BPO * math.log2(0.45 * sr / fmin))))
    n_bins -= n_bins % CQT_BPO if n_bins > CQT_BPO else 0
    y = np.asarray(mono, dtype="float32")
    min_n = int(CQT_MIN_PAD_S * sr)
    if y.size < min_n:
        y = np.pad(y, (0, min_n - y.size))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        C = np.abs(librosa.cqt(y, sr=sr, hop_length=hop, fmin=fmin, n_bins=n_bins,
                               bins_per_octave=CQT_BPO))
        freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=CQT_BPO)
        lengths, _ = librosa.filters.wavelet_lengths(freqs=freqs, sr=sr, window="hann",
                                                     filter_scale=1)
    amp = 2.0 * C / np.sqrt(lengths)[:, None]           # sinusoid amplitude per bin
    t = np.arange(C.shape[1]) * hop / float(sr)
    keep = t <= dur + 1e-9
    db = 20.0 * np.log10(np.maximum(amp[:, keep], 10 ** (_FLOOR_DB / 20)))
    return db, t[keep], freqs


def _cqt_hop(n: int) -> int:
    """CQT hop: a multiple of 128 (8-octave early downsampling) giving ~CQT_TARGET_COLS columns."""
    return int(max(128, 128 * math.ceil(n / CQT_TARGET_COLS / 128)))


def _fine_hop(n: int) -> int:
    """Hop for the 1-D lanes: ≤512 samples, finer on short clips (~2000 points minimum)."""
    return int(min(512, max(32, n // 2000)))


def analyze(samples, sr: int, *, grid: dict, intent: dict | None = None,
            dur: float | None = None, cqt_hop: int | None = None,
            sections: bool = True) -> dict:
    """Compute every lane over ``samples`` ((n,) or (n, ch) float) on ``grid``.

    ``dur`` (≥ the clip's own length) zero-pads to a shared axis (A/B mode). Returns a dict
    with numpy ``series`` for rendering and a JSON-ready ``summary`` (global + per-bar).
    """
    import librosa
    import numpy as np

    from . import arc as A
    from .width import BANDS, _bandpass

    s = np.asarray(samples, dtype="float64")
    if s.ndim == 1:
        s = s[:, None]
    n_own = s.shape[0]
    own_dur = n_own / float(sr)
    dur = float(max(dur or own_dur, own_dur))
    n = round(dur * sr)
    if n > n_own:
        s = np.pad(s, ((0, n - n_own), (0, 0)))
    ch = s.shape[1]
    mono = s.mean(axis=1)
    intent = intent or {}

    hf = _fine_hop(n)
    t_fine = (np.arange(math.ceil(n / hf)) + 0.5) * hf / sr

    # --- loudness: K-weighted block energies → centred momentary / short-term. ---
    z = _k_weight(s, sr)
    kms = _block_ms(z, hf).sum(axis=1)                       # channel-summed (G = 1 for L/R)
    mom = _lufs_of(_centred_mean(kms, round(MOMENTARY_S * sr / hf)))
    short = _lufs_of(_centred_mean(kms, round(SHORT_TERM_S * sr / hf))) if dur >= SHORT_TERM_S else None

    # --- six standardized bands (zero-phase band-pass, block mean square, mean over channels). ---
    band_names = [b[0] for b in BANDS if b[0] != "full"]
    band_sig = {}
    band_db = {}
    for name, lo, hi in BANDS:
        if name == "full" or lo >= sr / 2.0:
            continue
        sig = np.stack([_bandpass(s[:, c], sr, lo, hi) for c in range(ch)], axis=1)
        band_sig[name] = sig
        smooth = max(1, round(max(0.005, 1.5 / lo) * sr / hf))  # no per-cycle ripple on low bands
        band_db[name] = _db_pow(_centred_mean(_block_ms(sig, hf).mean(axis=1), smooth))

    # --- stereo correlation in blocks (NaN when silent; None for mono). ---
    corr_t = corr = None
    if ch >= 2:
        cw = max(hf, round((0.05 if dur >= 2.0 else 0.005) * sr))
        nb = math.ceil(n / cw)
        L = np.pad(s[:, 0], (0, nb * cw - n)).reshape(nb, cw)
        R = np.pad(s[:, 1], (0, nb * cw - n)).reshape(nb, cw)
        corr = _pearson_rows(L, R)
        corr_t = (np.arange(nb) + 0.5) * cw / sr

    # --- onsets (measured) at the fine hop. ---
    ymono = mono.astype("float32")
    if ymono.size < 2048:
        ymono = np.pad(ymono, (0, 2048 - ymono.size))
    ho = int(2 ** round(math.log2(ONSET_HOP_S * sr)))
    onset_env = librosa.onset.onset_strength(y=ymono, sr=sr, hop_length=ho)
    onset_t = librosa.frames_to_time(np.arange(onset_env.size), sr=sr, hop_length=ho)
    on_keep = onset_t <= dur
    onset_env, onset_t = onset_env[on_keep], onset_t[on_keep]
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=ho,
                                        units="time") if onset_env.size > 3 else np.array([])
    onsets = [float(t) for t in onsets if t < dur]
    hits = [float(h["t"]) for h in intent.get("hits") or [] if 0.0 <= float(h["t"]) < dur]

    # --- CQT (dB of a sinusoid) and chroma folded off it. ---
    hop2 = cqt_hop or _cqt_hop(n)
    cqt_db, cqt_t, cqt_f = _cqt_db(mono, sr, hop2, dur)
    cqt_pow = 10.0 ** (cqt_db / 10.0)
    nb_full = (cqt_pow.shape[0] // 12) * 12
    chroma_pow = cqt_pow[:nb_full].reshape(-1, 12, cqt_pow.shape[1]).sum(axis=0)   # (12, T)
    frame_peak_db = cqt_db.max(axis=0) if cqt_db.size else np.zeros(0)

    # --- sections (Foote novelty, via arc) + SSM. ---
    bounds: list[float] = []
    if sections and dur >= 2.0 * A.MIN_SECTION_S:
        k = int(min(8, max(2, round(dur / SECTION_SPAN_S))))
        try:
            secs = A.segment(mono, sr, n_sections=k, duration_s=dur)
            bounds = [float(t0) for t0, _t1 in secs[1:]]
        except Exception:
            bounds = []
    ssm = ssm_t = None
    if dur >= SSM_MIN_S:
        ssm, ssm_t = self_similarity(mono, sr)

    pcs = list(intent.get("pitch_classes") or [])
    pc_idx = [PC_NAMES.index(p) for p in pcs if p in PC_NAMES]

    # --- per-bar summaries, computed from samples (not from the pooled pixels). ---
    rows = []
    for label, t0, t1 in grid["bars"]:
        a, b = round(t0 * sr), max(round(t1 * sr), round(t0 * sr) + 1)
        a, b = min(a, n), min(b, n)
        seg = s[a:b]
        zz = z[a:b]
        row = {grid["unit"]: label, "t0": _r(t0, 3), "t1": _r(t1, 3)}
        kseg = float(np.mean(zz * zz, axis=0).sum()) if zz.size else 0.0
        row["lufs"] = _quiet(_lufs_of(kseg)) if kseg > 1e-14 else None
        pk = float(np.max(np.abs(seg))) if seg.size else 0.0
        row["peak_dbfs"] = _r(20 * math.log10(pk)) if pk > 0 else None
        row["bands_db"] = {nm: (_quiet(float(_db_pow(np.mean(band_sig[nm][a:b] ** 2))))
                                if nm in band_sig and b > a else None) for nm in band_names}
        if ch >= 2 and b - a > 2:
            row["corr"] = _r(float(_pearson_rows(seg[None, :, 0], seg[None, :, 1])[0]), 3)
        else:
            row["corr"] = None
        row["onsets"] = sum(1 for t in onsets if t0 <= t < t1)
        if hits:
            row["hits_intended"] = sum(1 for t in hits if t0 <= t < t1)
        fm = (cqt_t >= t0) & (cqt_t < t1)
        cp = chroma_pow[:, fm].sum(axis=1) if fm.any() else np.zeros(12)
        tot = float(cp.sum())
        if tot > 0 and frame_peak_db[fm].max() > -100.0:
            share = cp / tot
            row["pc_top"] = [PC_NAMES[i] for i in np.argsort(share)[::-1][:3]]
            if pc_idx:
                row["pc_in_set_share"] = _r(float(share[pc_idx].sum()), 3)
        else:
            row["pc_top"] = []
            if pc_idx:
                row["pc_in_set_share"] = None
        rows.append(row)

    # --- global numbers. ---
    lufs_i = integrated_lufs(s[:n_own], sr)
    peak = float(np.max(np.abs(s))) if s.size else 0.0
    dc = [float(np.mean(s[:n_own, c])) for c in range(ch)]
    st_vals = np.asarray(short if short is not None else mom)
    st_vals = st_vals[np.isfinite(st_vals) & (st_vals > -70.0)]
    tot_chroma = chroma_pow.sum(axis=1)
    glob = {
        "dur_s": _r(own_dur, 3),
        "sr": int(sr),
        "channels": int(ch),
        "integrated_lufs": _r(lufs_i),
        "ungated_lufs": _r(ungated_lufs(s[:n_own], sr)),
        "max_momentary_lufs": _r(float(np.max(mom))) if np.size(mom) else None,
        "max_short_term_lufs": _r(float(np.max(short))) if short is not None else None,
        "loudness_range_lu": (_r(float(np.percentile(st_vals, 95) - np.percentile(st_vals, 10)))
                              if st_vals.size >= 3 else None),
        "peak_dbfs": _r(20 * math.log10(peak)) if peak > 0 else None,
        "dc_offset": [_r(v, 6) for v in dc],
        "dc_offset_dbfs": [_r(20 * math.log10(abs(v))) if abs(v) > 0 else None for v in dc],
        "corr_mean": (_r(float(np.nanmean(corr)), 3)
                      if corr is not None and np.isfinite(corr).any() else None),
        "corr_min": (_r(float(np.nanmin(corr)), 3)
                     if corr is not None and np.isfinite(corr).any() else None),
        "onsets_measured": len(onsets),
        "hits_intended": len(hits) if intent.get("hits") is not None else None,
        "pc_top": ([PC_NAMES[i] for i in np.argsort(tot_chroma)[::-1][:5]]
                   if tot_chroma.sum() > 0 else []),
        "section_bounds_s": [_r(b, 3) for b in bounds],
        "section_bounds_bar": [_bar_at(grid, b) for b in bounds],
    }
    if pc_idx and tot_chroma.sum() > 0:
        glob["pc_intended"] = pcs
        glob["pc_in_set_share"] = _r(float(tot_chroma[pc_idx].sum() / tot_chroma.sum()), 3)

    return {
        "sr": int(sr), "dur": dur, "own_dur": own_dur, "channels": ch,
        "series": {
            "t_fine": t_fine, "momentary": mom, "short_term": short,
            "band_db": band_db, "corr_t": corr_t, "corr": corr,
            "onset_t": onset_t, "onset_env": onset_env, "onsets": onsets, "hits": hits,
            "cqt_db": cqt_db, "cqt_t": cqt_t, "cqt_f": cqt_f, "chroma_pow": chroma_pow,
            "frame_peak_db": frame_peak_db, "bounds": bounds, "ssm": ssm, "ssm_t": ssm_t,
        },
        "pcs": pcs,
        "grid": grid,
        "summary": {"global": glob, "rows": rows},
        "params": {"fine_hop": hf, "cqt_hop": hop2, "cqt_bins": int(cqt_db.shape[0]),
                   "ssm_panel": ssm is not None},
    }


def _quiet(db) -> float | None:
    """A level below ``QUIET_DB`` is digital silence for the sidecar: None, not −140."""
    v = _r(db)
    return v if v is not None and v > QUIET_DB else None


def _bar_at(grid: dict, t: float) -> str | None:
    for label, t0, t1 in grid["bars"]:
        if t0 <= t < t1:
            return label
    return grid["bars"][-1][0] if grid["bars"] else None


def _pearson_rows(L, R):
    """Row-wise Pearson correlation of two (m, k) arrays; NaN where either row is silent."""
    import numpy as np

    L = np.asarray(L, dtype="float64")
    R = np.asarray(R, dtype="float64")
    Lc = L - L.mean(axis=1, keepdims=True)
    Rc = R - R.mean(axis=1, keepdims=True)
    num = (Lc * Rc).sum(axis=1)
    den = np.sqrt((Lc * Lc).sum(axis=1) * (Rc * Rc).sum(axis=1))
    loud = ((L * L).mean(axis=1) > SILENT_MS) | ((R * R).mean(axis=1) > SILENT_MS)
    out = np.full(L.shape[0], np.nan)
    ok = loud & (den > _EPS)
    out[ok] = np.clip(num[ok] / den[ok], -1.0, 1.0)
    return out


def self_similarity(mono, sr: int):
    """MFCC cosine self-similarity at arc's segmentation rate (the matrix Foote novelty reads)."""
    import librosa
    import numpy as np

    from . import arc as A

    hop = max(1, round(A.SEG_HOP_S * sr))
    mfcc = librosa.feature.mfcc(y=np.asarray(mono, dtype="float32"), sr=sr, n_mfcc=A.N_MFCC,
                                n_fft=A.N_FFT, hop_length=hop)
    F = mfcc - mfcc.mean(axis=1, keepdims=True)
    Fn = F / np.maximum(np.linalg.norm(F, axis=0, keepdims=True), _EPS)
    ssm = np.clip(Fn.T @ Fn, -1.0, 1.0)
    t = librosa.frames_to_time(np.arange(mfcc.shape[1]), sr=sr, hop_length=hop)
    return ssm, t


# ---------------------------------------------------------------------------------------------
# Rendering.
# ---------------------------------------------------------------------------------------------


def _fmt(v, unit: str = "", nd: int = 1) -> str:
    if v is None:
        return "—"
    return f"{v:.{nd}f}{unit}"


def _ruler_step(n: int) -> int:
    k = 1
    while n / k > MAX_RULER_LABELS:
        k *= 2
    return k


def _section_letter(i: int) -> str:
    return chr(ord("A") + i) if i < 26 else str(i + 1)


class _Style:
    def __init__(self, *, db_range, lufs_range, cmap, diff_range, max_width_px):
        self.db_range = tuple(float(v) for v in db_range)
        self.lufs_range = tuple(float(v) for v in lufs_range)
        self.cmap = cmap
        self.diff_range = float(diff_range)
        self.max_width_px = int(max_width_px)
        self.width_in = self.max_width_px / DPI
        self.plot_px = int(self.max_width_px * 0.80)  # columns actually available to a lane

    def params(self) -> dict:
        return {"db_range": list(self.db_range), "lufs_range": list(self.lufs_range),
                "cmap": self.cmap, "diff_range_db": self.diff_range,
                "max_width_px": self.max_width_px}


def _lane_label(ax, text: str, color: str = "black"):
    ax.set_ylabel(text, rotation=0, ha="right", va="center", fontsize=13, fontweight="bold",
                  color=color, labelpad=12)


def _grid_lines(ax, an: dict, style: _Style, *, beats: bool = False, color: str = "#888888"):
    """Bar lines (and optionally beat lines) + section boundaries on one lane."""
    grid = an["grid"]
    if grid["unit"] == "bar":
        for _label, t0, _t1 in grid["bars"]:
            ax.axvline(t0, color=color, lw=0.9, alpha=0.75, zorder=4)
        if beats and grid["beats"]:
            px_per_beat = style.plot_px * (60.0 / grid["bpm"]) / max(an["dur"], 1e-9) if grid["bpm"] else 0
            if px_per_beat >= MIN_BEAT_PX:
                for t in grid["beats"]:
                    ax.axvline(t, color=color, lw=0.5, alpha=0.45, ls=":", zorder=4)
    for t in an["series"]["bounds"]:
        ax.axvline(t, color="#00bcd4", lw=2.0, ls="--", alpha=0.95, zorder=5)


def _draw_ruler(ax, an: dict):
    grid = an["grid"]
    bars = grid["bars"]
    step = _ruler_step(len(bars))
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    for i, (label, t0, _t1) in enumerate(bars):
        if grid["unit"] == "bar":
            ax.axvline(t0, color="#444444", lw=1.0)
        if i % step == 0:
            ax.text(t0, 0.74 if an["series"]["bounds"] else 0.5, f" {label}", fontsize=12,
                    fontweight="bold", va="center", ha="left")
    box = {"boxstyle": "round,pad=0.15", "fc": "#00838f", "ec": "none"}
    for i, t in enumerate(an["series"]["bounds"]):
        ax.axvline(t, ymax=0.5, color="#00bcd4", lw=2.5, ls="--")
        ax.text(t, 0.25, f" {_section_letter(i + 1)} ", fontsize=12, fontweight="bold",
                color="white", va="center", ha="left", bbox=box)
    if an["series"]["bounds"]:
        ax.text(0.0, 0.25, f" {_section_letter(0)} ", fontsize=12, fontweight="bold", color="white",
                va="center", ha="left", bbox=box)
    unit = "bar" if grid["unit"] == "bar" else "segment"
    src = {"intent": "intent", "flag": "--bpm", "detected": "beat tracker", "none": "no grid"}[grid["source"]]
    _lane_label(ax, f"{unit} ({src})" + ("\nsection" if an["series"]["bounds"] else ""))


def _draw_loudness(ax, an: dict, style: _Style, second=None):
    """Momentary (thin) + short-term (thick) loudness; ``second`` is drawn orange (A/B)."""
    for a, col in ((an, "#1565c0"), *(((second, "#e65100"),) if second else ())):
        Sx = a["series"]
        t, _ = _pool(Sx["t_fine"], style.plot_px * 2)
        mom, _ = _pool(Sx["momentary"], style.plot_px * 2)
        short = Sx["short_term"] is not None
        ax.plot(t, mom, color=col, lw=0.9 if short else 2.4, alpha=0.55 if short else 1.0)
        if short:
            st, _ = _pool(Sx["short_term"], style.plot_px * 2)
            ax.plot(t, st, color=col, lw=2.6)
    lo, hi = style.lufs_range
    ax.set_ylim(lo, hi)
    ax.set_yticks([lo, (lo + hi) / 2, hi])
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", alpha=0.3)
    _grid_lines(ax, an, style)
    if an["series"]["short_term"] is not None:
        _lane_label(ax, f"LUFS [{lo:.0f}..{hi:.0f}]\nthin 0.4 s\nthick 3 s")
    else:
        _lane_label(ax, f"LUFS [{lo:.0f}..{hi:.0f}]\n0.4 s window")


def _draw_cqt(ax, an: dict, style: _Style, title: str = "CQT"):
    import numpy as np

    S = an["series"]
    db, _f = _pool(S["cqt_db"], style.plot_px, "max")
    t, _ = _pool(S["cqt_t"], style.plot_px)
    lo, hi = style.db_range
    nb = db.shape[0]
    dt = (t[1] - t[0]) if t.size > 1 else an["dur"]
    ax.imshow(np.clip(db, lo, hi), origin="lower", aspect="auto", cmap=style.cmap, vmin=lo, vmax=hi,
              extent=(t[0] - dt / 2 if t.size else 0, (t[-1] + dt / 2) if t.size else an["dur"], 0, nb),
              interpolation="nearest")
    ticks = list(range(0, nb, 12))
    ax.set_yticks([k + 0.5 for k in ticks])
    ax.set_yticklabels([f"C{1 + k // 12}" for k in ticks], fontsize=11)
    _grid_lines(ax, an, style, beats=True, color="#dddddd")
    _lane_label(ax, f"{title}\ndBFS [{lo:.0f}..{hi:.0f}]\n{style.cmap}")


def _draw_onsets(ax, an: dict, style: _Style):
    import numpy as np

    S = an["series"]
    env = S["onset_env"]
    m = float(np.max(env)) if env.size else 0.0
    envn = env / m if m > 0 else env
    e, _ = _pool(envn, style.plot_px * 2, "max")
    t, _ = _pool(S["onset_t"], style.plot_px * 2)
    ax.fill_between(t, 0, e, color="#78909c", alpha=0.6, lw=0)
    dense = len(S["onsets"]) > style.plot_px / 6
    if dense:  # a wall of ticks reads as nothing: draw onsets per bar instead
        counts = [(t0, t1, r["onsets"]) for (_l, t0, t1), r in
                  zip(an["grid"]["bars"], an["summary"]["rows"])]
        top = max(c for _a, _b, c in counts) or 1
        for t0, t1, c in counts:
            ax.add_patch(_rect(t0, 0.0, t1 - t0, 0.8 * c / top, "#1565c0", 0.55))
    else:
        ax.vlines(S["onsets"], 0.0, 0.35, color="#1565c0", lw=1.4)
    if S["hits"]:
        ax.plot(S["hits"], [0.92] * len(S["hits"]), "v", color="#e65100", ms=8,
                label="intended hits")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([])
    _grid_lines(ax, an, style)
    lab = (f"onsets\nblue = per {an['grid']['unit']}\n(max {top})" if dense
           else "onsets\nblue | measured")
    if S["hits"]:
        lab += "\norange ▼ intended"
    _lane_label(ax, lab)


def _rect(x, y, w, h, color, alpha):
    from matplotlib.patches import Rectangle

    return Rectangle((x, y), w, h, facecolor=color, alpha=alpha, lw=0, zorder=3)


def _draw_chroma(ax, an: dict, style: _Style):
    import numpy as np

    S = an["series"]
    cp = S["chroma_pow"]
    share = cp / np.maximum(cp.max(axis=0, keepdims=True), _EPS)
    share[:, S["frame_peak_db"] < style.db_range[0]] = 0.0     # silence stays dark
    sh, _ = _pool(share, style.plot_px)
    t, _ = _pool(S["cqt_t"], style.plot_px)
    dt = (t[1] - t[0]) if t.size > 1 else an["dur"]
    ax.imshow(sh, origin="lower", aspect="auto", cmap=style.cmap, vmin=0, vmax=1,
              extent=(t[0] - dt / 2 if t.size else 0, (t[-1] + dt / 2) if t.size else an["dur"], 0, 12),
              interpolation="nearest")
    pcs = set(an["pcs"])
    ax.set_yticks([k + 0.5 for k in range(12)])
    labels = [f"● {p}" if p in pcs else p for p in PC_NAMES]
    ax.set_yticklabels(labels, fontsize=10)
    for tl, p in zip(ax.get_yticklabels(), PC_NAMES):
        if p in pcs:
            tl.set_color("#2e7d32")
            tl.set_fontweight("bold")
    _grid_lines(ax, an, style, color="#bbbbbb")
    lab = "chroma\nshare per frame"
    if pcs:
        lab += "\n● = intended set"
    _lane_label(ax, lab)


def _draw_band(ax, an: dict, name: str, lo_hz: float, hi_hz: float, style: _Style,
               second=None, color="#1565c0"):
    lo, hi = style.db_range
    S = an["series"]
    if name in S["band_db"]:
        v, _ = _pool(S["band_db"][name], style.plot_px * 2)
        t, _ = _pool(S["t_fine"], style.plot_px * 2)
        ax.fill_between(t, lo, v, color=color, alpha=0.55 if second else 0.8, lw=0)
    if second is not None and name in second["series"]["band_db"]:
        v2, _ = _pool(second["series"]["band_db"][name], style.plot_px * 2)
        t2, _ = _pool(second["series"]["t_fine"], style.plot_px * 2)
        ax.plot(t2, v2, color="#e65100", lw=1.4)
    if name in S["band_db"] and second is None:
        import numpy as np

        live = S["band_db"][name][S["band_db"][name] > lo]
        med = f"{np.median(live):.0f} dB" if live.size else "silent"
        ax.text(1.006, 0.5, f"med\n{med}", transform=ax.transAxes, fontsize=11, va="center",
                ha="left")
    ax.set_ylim(lo, hi)
    ax.set_yticks([])
    ax.grid(axis="y", alpha=0.3)
    _grid_lines(ax, an, style)
    hz = (lambda f: f"{f / 1000:g}k" if f >= 1000 else f"{f:g}")
    _lane_label(ax, f"{name} {hz(lo_hz)}–{hz(hi_hz)}")


def _draw_corr(ax, an: dict, style: _Style, second=None):
    S = an["series"]
    ax.axhline(0.0, color="#999999", lw=0.8)
    drew = False
    for a, col in ((an, "#1565c0"), *(((second, "#e65100"),) if second else ())):
        Sx = a["series"]
        if Sx["corr"] is None:
            continue
        c, _ = _pool(Sx["corr"], style.plot_px * 2)
        t, _ = _pool(Sx["corr_t"], style.plot_px * 2)
        ax.plot(t, c, color=col, lw=2.2)
        drew = True
    if not drew:
        ax.text(0.5, 0.5, "mono (1 channel): no stereo lane", transform=ax.transAxes,
                ha="center", va="center", fontsize=13)
    del S
    ax.set_ylim(-1.15, 1.15)
    ax.set_yticks([-1, 0, 1])
    ax.tick_params(axis="y", labelsize=11)
    _grid_lines(ax, an, style)
    _lane_label(ax, "stereo corr\nL/R [-1..1]")


def _draw_ssm(ax, an: dict, style: _Style):
    S = an["series"]
    ssm, t = S["ssm"], S["ssm_t"]
    end = an["dur"]
    ax.imshow(ssm, origin="lower", cmap=style.cmap, vmin=0, vmax=1, extent=(0, end, 0, end),
              interpolation="nearest")
    for b in S["bounds"]:
        ax.axvline(b, color="#00bcd4", lw=2, ls="--")
        ax.axhline(b, color="#00bcd4", lw=2, ls="--")
    ax.set_title("self-similarity (MFCC cosine), time × time\nbright blocks = sections "
                 "that repeat / hold", fontsize=12)
    ax.set_xlabel("time (s)", fontsize=12)
    ax.tick_params(labelsize=11)
    del t


def _header_text(name: str, an: dict, extra: str = "") -> str:
    g = an["summary"]["global"]
    grid = an["grid"]
    bits = [f"{g['dur_s']:.2f} s", f"{g['channels']} ch", f"{g['sr']} Hz"]
    if grid["bpm"]:
        bits.append(f"{grid['bpm']:.1f} BPM ({grid['source']}), {len(grid['bars'])} bars")
    line2 = [f"integrated {_fmt(g['integrated_lufs'], ' LUFS')}",
             f"ungated {_fmt(g['ungated_lufs'], ' LUFS')}",
             f"peak {_fmt(g['peak_dbfs'], ' dBFS')}",
             "DC " + "/".join(_fmt(v, "", 0) for v in g["dc_offset_dbfs"]) + " dBFS"]
    line3 = []
    if g["corr_mean"] is not None:
        line3.append(f"corr mean {g['corr_mean']:+.2f} min {g['corr_min']:+.2f}")
    else:
        line3.append("mono")
    line3.append(f"onsets {g['onsets_measured']}"
                 + (f" (intended hits {g['hits_intended']})" if g["hits_intended"] is not None else ""))
    if g.get("pc_intended"):
        line3.append(f"intended {' '.join(g['pc_intended'])}: in-set share {g['pc_in_set_share']:.2f}")
    lines = [f"{name}   ·   " + " · ".join(bits), " · ".join(line2), " · ".join(line3)]
    if extra:
        lines.append(extra)
    return "\n".join(lines)


def _new_fig(style: _Style, ratios: list[float]):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    height = sum(ratios)
    fig = plt.figure(figsize=(style.width_in, height), dpi=DPI)
    gs = fig.add_gridspec(len(ratios), 1, height_ratios=ratios, hspace=0.12,
                          left=0.17, right=0.94, top=1 - 0.2 / height, bottom=0.65 / height)
    return fig, gs, plt


def _finish_time_axes(axes, an: dict):
    dur = an["dur"]
    for ax in axes:
        ax.set_xlim(0.0, dur)
        ax.tick_params(axis="x", labelbottom=False)
    axes[-1].tick_params(axis="x", labelbottom=True, labelsize=11)
    axes[-1].set_xlabel("time (s)", fontsize=12)


def _png(fig, plt) -> bytes:
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=DPI)
    finally:
        plt.close(fig)
    return buf.getvalue()


def render(an: dict, style: _Style, name: str) -> bytes:
    """Render the single-clip sheet. Returns PNG bytes."""
    from .width import BANDS

    bands = [b for b in BANDS if b[0] != "full"]
    has_ssm = an["series"]["ssm"] is not None
    ruler = 0.75 if an["series"]["bounds"] else 0.45
    ratios = [1.1, ruler, 1.5, 3.4, 0.95, 2.0] + [0.6] * len(bands) + [0.95]
    if has_ssm:
        ratios += [1.0, 5.2]
    fig, gs, plt = _new_fig(style, ratios)

    head = fig.add_subplot(gs[0])
    head.axis("off")
    head.text(0.0, 0.5, _header_text(name, an), fontsize=13, va="center", ha="left",
              family="monospace", transform=head.transAxes)

    lanes = []
    first = fig.add_subplot(gs[1])
    lanes.append(first)
    _draw_ruler(first, an)

    def lane(i):
        ax = fig.add_subplot(gs[i], sharex=first)
        lanes.append(ax)
        return ax

    _draw_loudness(lane(2), an, style)
    _draw_cqt(lane(3), an, style)
    _draw_onsets(lane(4), an, style)
    _draw_chroma(lane(5), an, style)
    for k, (nm, lo, hi) in enumerate(bands):
        _draw_band(lane(6 + k), an, nm, lo, hi, style)
    _draw_corr(lane(6 + len(bands)), an, style)
    _finish_time_axes(lanes, an)

    if has_ssm:
        sub = gs[len(ratios) - 1].subgridspec(1, 2, width_ratios=[1, 1.6], wspace=0.15)
        sax = fig.add_subplot(sub[0])
        sax.set_box_aspect(1)
        _draw_ssm(sax, an, style)
        tax = fig.add_subplot(sub[1])
        tax.axis("off")
        tax.text(0.0, 1.0, _section_table(an), fontsize=12, family="monospace", va="top",
                 ha="left", transform=tax.transAxes)
    return _png(fig, plt)


def _section_table(an: dict) -> str:
    """Per-section numbers burned next to the SSM (the sidecar carries them per bar too)."""
    import numpy as np

    rows = an["summary"]["rows"]
    edges = [0.0, *an["series"]["bounds"], an["dur"]]
    lines = ["section  start    LUFS   sub   bass  mid   air"]
    for i in range(len(edges) - 1):
        t0, t1 = edges[i], edges[i + 1]
        sel = [r for r in rows if r["t0"] is not None and t0 - 1e-6 <= r["t0"] < t1]
        if not sel:
            continue

        def med(key, sub=None, sel=sel):
            vals = [(r[key][sub] if sub else r[key]) for r in sel]
            vals = [v for v in vals if v is not None]
            return float(np.median(vals)) if vals else None

        lines.append(f"   {_section_letter(i)}    {t0:6.1f}s {_fmt(med('lufs'), '', 1):>6} "
                     f"{_fmt(med('bands_db', 'sub'), '', 0):>5} {_fmt(med('bands_db', 'bass'), '', 0):>5} "
                     f"{_fmt(med('bands_db', 'mid'), '', 0):>5} {_fmt(med('bands_db', 'air'), '', 0):>5}")
    lines.append("")
    lines.append("(medians of the per-bar rows; dB of a sinusoid)")
    return "\n".join(lines)


def render_ab(a: dict, b: dict, style: _Style, names: tuple[str, str], gain_a: float, gain_b: float) -> bytes:
    """Render the A/B sheet: shared axes, A blue / B orange, CQT difference panel. PNG bytes."""
    import numpy as np

    from .width import BANDS

    bands = [x for x in BANDS if x[0] != "full"]
    ratios = [1.2, 0.45, 1.5, 2.6, 2.6, 2.6] + [0.6] * len(bands) + [0.95]
    fig, gs, plt = _new_fig(style, ratios)

    head = fig.add_subplot(gs[0])
    head.axis("off")
    ga, gb = a["summary"]["global"], b["summary"]["global"]
    gain_a, gain_b = (g + 0.0 if abs(g) >= 0.005 else 0.0 for g in (gain_a, gain_b))  # never "-0.00"
    extra = (f"A = blue, B = orange · gain A {gain_a:+.2f} dB, B {gain_b:+.2f} dB (louder one down) · integrated "
             f"A {_fmt(ga['integrated_lufs'], ' LUFS')}, B {_fmt(gb['integrated_lufs'], ' LUFS')} "
             f"(after gain)")
    head.text(0.0, 0.5, "A: " + _header_text(names[0], a).split("\n")[0] + "\n"
              + "B: " + _header_text(names[1], b).split("\n")[0] + "\n" + extra,
              fontsize=12, va="center", ha="left", family="monospace", transform=head.transAxes)

    lanes = []
    first = fig.add_subplot(gs[1])
    lanes.append(first)
    _draw_ruler(first, a)

    def lane(i):
        ax = fig.add_subplot(gs[i], sharex=first)
        lanes.append(ax)
        return ax

    _draw_loudness(lane(2), a, style, second=b)
    _draw_cqt(lane(3), a, style, title="CQT  A")
    _draw_cqt(lane(4), b, style, title="CQT  B")

    dax = lane(5)
    lo, hi = style.db_range
    da = np.clip(a["series"]["cqt_db"], lo, hi)
    db_ = np.clip(b["series"]["cqt_db"], lo, hi)
    m = min(da.shape[1], db_.shape[1])
    nb = min(da.shape[0], db_.shape[0])
    diff = db_[:nb, :m] - da[:nb, :m]
    diff, _ = _pool(diff, style.plot_px)
    t, _ = _pool(a["series"]["cqt_t"][:m], style.plot_px)
    dt = (t[1] - t[0]) if t.size > 1 else a["dur"]
    r = style.diff_range
    dax.imshow(diff, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-r, vmax=r,
               extent=(t[0] - dt / 2, t[-1] + dt / 2, 0, nb), interpolation="nearest")
    ticks = list(range(0, nb, 12))
    dax.set_yticks([k + 0.5 for k in ticks])
    dax.set_yticklabels([f"C{1 + k // 12}" for k in ticks], fontsize=11)
    _grid_lines(dax, a, style)
    _lane_label(dax, f"B − A dB\n[±{r:.0f}] red = B louder\nblue = A louder")

    for k, (nm, lo_hz, hi_hz) in enumerate(bands):
        _draw_band(lane(6 + k), a, nm, lo_hz, hi_hz, style, second=b)
    _draw_corr(lane(6 + len(bands)), a, style, second=b)
    _finish_time_axes(lanes, a)
    return _png(fig, plt)


# ---------------------------------------------------------------------------------------------
# Sidecar documents.
# ---------------------------------------------------------------------------------------------


def sidecar_doc(an: dict, *, name: str, intent: dict | None, style: _Style) -> dict:
    g = an["grid"]
    return {
        "v": 1,
        "op_version": OP_VERSION,
        "name": name,
        "grid": {"source": g["source"], "unit": g["unit"], "bpm": _r(g["bpm"], 3),
                 "beats_per_bar": g["beats_per_bar"], "n": len(g["bars"])},
        "intent_source": (intent or {}).get("source"),
        "global": an["summary"]["global"],
        "rows": an["summary"]["rows"],
        "render": {**style.params(), **an["params"]},
        "units": {"lufs": "LUFS (K-weighted, ungated per row)", "peak_dbfs": "dBFS sample peak",
                  "bands_db": "dB of a sinusoid (full-scale sine = 0)", "corr": "-1..1",
                  "pc_in_set_share": "0..1 share of chroma energy in the intended set"},
    }


def sidecar_markdown(doc: dict) -> str:
    g = doc["global"]
    grid = doc["grid"]
    unit = grid["unit"]
    out = [f"# sheet: {doc['name']}", ""]
    gl = [f"dur {g['dur_s']} s", f"{g['channels']} ch", f"{g['sr']} Hz"]
    if grid["bpm"]:
        gl.append(f"{grid['bpm']} BPM ({grid['source']}), {grid['n']} bars")
    else:
        gl.append(f"no bar grid ({grid['n']} equal segments)")
    out.append("- " + " · ".join(gl))
    out.append(f"- integrated {g['integrated_lufs']} LUFS · ungated {g['ungated_lufs']} LUFS · "
               f"max momentary {g['max_momentary_lufs']} · max short-term {g['max_short_term_lufs']} · "
               f"range {g['loudness_range_lu']} LU")
    out.append(f"- peak {g['peak_dbfs']} dBFS · DC offset {g['dc_offset']} "
               f"({g['dc_offset_dbfs']} dBFS) · corr mean {g['corr_mean']} min {g['corr_min']}")
    pc = f"- top pitch classes {' '.join(g['pc_top']) or '—'}"
    if g.get("pc_intended"):
        pc += f" · intended {' '.join(g['pc_intended'])} · in-set share {g['pc_in_set_share']}"
    out.append(pc)
    out.append(f"- onsets measured {g['onsets_measured']} · intended hits {g['hits_intended']}")
    if g["section_bounds_s"]:
        out.append(f"- section boundaries (s): {g['section_bounds_s']} (at {unit}s "
                   f"{g['section_bounds_bar']})")
    if doc.get("ab"):
        ab = doc["ab"]
        out.append(f"- A/B: gain A {ab['gain_db_a']:+.2f} dB, B {ab['gain_db_b']:+.2f} dB ({ab['match']}, "
                   f"louder one down); A {ab['lufs_a']} vs B {ab['lufs_b']} LUFS, both {ab['lufs_matched']} after")
    out.append("")
    has_pc = any("pc_in_set_share" in r for r in doc["rows"])
    has_hits = any("hits_intended" in r for r in doc["rows"])
    if doc.get("ab"):
        out.append(f"| {unit} | t0 | LUFS A | LUFS B | Δ | sub Δ | bass Δ | lomid Δ | mid Δ | "
                   "upmid Δ | air Δ | corr A | corr B |")
        out.append("|---" * 13 + "|")
        for r in doc["rows"]:
            d = r["bands_delta_db"]
            out.append(f"| {r[unit]} | {r['t0']} | {r['lufs_a']} | {r['lufs_b']} | {r['lufs_delta']} | "
                       + " | ".join(str(d[k]) for k in ("sub", "bass", "lomid", "mid", "uppermid", "air"))
                       + f" | {r['corr_a']} | {r['corr_b']} |")
        return "\n".join(out) + "\n"
    head = [unit, "t0", "LUFS", "peak", "sub", "bass", "lomid", "mid", "upmid", "air", "corr",
            "onsets"]
    if has_hits:
        head.append("hits")
    if has_pc:
        head.append("in-set")
    head.append("top PCs")
    out.append("| " + " | ".join(head) + " |")
    out.append("|---" * len(head) + "|")
    for r in doc["rows"]:
        b = r["bands_db"]
        cells = [r[unit], r["t0"], r["lufs"], r["peak_dbfs"], b["sub"], b["bass"], b["lomid"],
                 b["mid"], b["uppermid"], b["air"], r["corr"], r["onsets"]]
        if has_hits:
            cells.append(r.get("hits_intended"))
        if has_pc:
            cells.append(r.get("pc_in_set_share"))
        cells.append(" ".join(r["pc_top"]))
        out.append("| " + " | ".join("—" if c is None else str(c) for c in cells) + " |")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------------------------
# Frame entrypoints.
# ---------------------------------------------------------------------------------------------


def _load(audio_frame: dict):
    import soundfile as sf
    from smplstream import cas

    src = cas.get_path(audio_frame["hash"])
    y, sr = sf.read(str(src), dtype="float32", always_2d=True)
    return y, int(sr)


def frame_name(audio_frame: dict) -> str:
    src = (audio_frame.get("params") or {}).get("source")
    if src:
        return Path(src).name
    return f"{audio_frame.get('role', 'audio')}:{str(audio_frame.get('id', '?'))[:12]}"


def _emit(doc: dict, png: bytes, *, role: str, of: str, lineage: list[str], params: dict,
          size: tuple[int, int]) -> list[dict]:
    from smplstream import cas
    from smplstream import frames as F

    img_h = cas.put_blob(png, "image/png")
    body = json.dumps(doc, sort_keys=True)
    json_h = cas.put_blob(body.encode("utf-8"), "application/json")
    md = sidecar_markdown(doc)
    md_h = cas.put_blob(md.encode("utf-8"), "text/plain")
    p = {**params, "json_hash": json_h, "markdown_hash": md_h, "image_hash": img_h}
    data = doc
    if len(body.encode("utf-8")) > INLINE_BUDGET:
        data = {k: v for k, v in doc.items() if k != "rows"}
        data["rows_truncated"] = True  # full per-bar rows live in the CAS JSON (params.json_hash)
    image = F.image_frame(img_h, media="image/png", role=role, of=of, lineage=lineage,
                          op=OP, op_version=OP_VERSION, params=p,
                          meta={"width_px": size[0], "height_px": size[1]})
    feat = F.feature_frame(data, role=role, of=of, lineage=lineage, op=OP, op_version=OP_VERSION,
                           params=p)
    from smplstream.frames import mint_id

    text = mint_id({"kind": "text", "role": f"{role}:summary", "hash": md_h, "media": "text/plain",
                    "of": of, "lineage": lineage, "op": OP, "op_version": OP_VERSION, "params": p})
    return [image, feat, text]


def _png_size(png: bytes) -> tuple[int, int]:
    import struct

    w, h = struct.unpack(">II", png[16:24])
    return int(w), int(h)


def _style(db_range=None, lufs_range=None, cmap=None, diff_range=None, max_width_px=None) -> _Style:
    return _Style(db_range=db_range or DEFAULT_DB_RANGE, lufs_range=lufs_range or DEFAULT_LUFS_RANGE,
                  cmap=cmap or DEFAULT_CMAP, diff_range=diff_range or DEFAULT_DIFF_RANGE_DB,
                  max_width_px=max_width_px or DEFAULT_MAX_WIDTH_PX)


def sheet_array(samples, sr: int, *, name: str = "clip", intent: dict | None = None,
                bpm: float | None = None, beats_per_bar: float | None = None,
                detect_beats: bool = True, **style_kw) -> tuple[bytes, dict]:
    """Analyze + render one clip from an array. Returns ``(png_bytes, sidecar_doc)``."""
    import numpy as np

    st = _style(**style_kw)
    s = np.asarray(samples, dtype="float64")
    s2 = s[:, None] if s.ndim == 1 else s
    dur = s2.shape[0] / float(sr)
    grid = make_grid(s2.mean(axis=1), sr, dur, intent=intent, bpm=bpm,
                     beats_per_bar=beats_per_bar, detect=detect_beats)
    an = analyze(s2, sr, grid=grid, intent=intent)
    png = render(an, st, name)
    return png, sidecar_doc(an, name=name, intent=intent, style=st)


def sheet_audio_frame(audio_frame: dict, *, intent_path: str | None = None,
                      bpm: float | None = None, beats_per_bar: float | None = None,
                      detect_beats: bool = True, **style_kw) -> list[dict]:
    """One `audio` frame → ``[image (role sheet), feature (role sheet), text (sheet:summary)]``."""
    from . import intent as I

    y, sr = _load(audio_frame)
    intent = I.for_audio_frame(audio_frame, explicit=intent_path)
    name = frame_name(audio_frame)
    png, doc = sheet_array(y, sr, name=name, intent=intent, bpm=bpm, beats_per_bar=beats_per_bar,
                           detect_beats=detect_beats, **style_kw)
    params = {**doc["render"], "grid_source": doc["grid"]["source"],
              "intent": bool(intent), "bpm": bpm, "beats_per_bar": beats_per_bar}
    return _emit(doc, png, role="sheet", of=audio_frame["id"], lineage=[audio_frame["id"]],
                 params=params, size=_png_size(png))


def ab_arrays(sa, sra: int, sb, srb: int, *, names=("A", "B"), intent: dict | None = None,
              bpm: float | None = None, beats_per_bar: float | None = None,
              detect_beats: bool = True, **style_kw) -> tuple[bytes, dict]:
    """A/B sheet from two arrays: B resampled to A's rate, the louder clip turned down to the other's
    loudness (never up, so matching can't push a clip over 0 dBFS)."""
    import numpy as np

    st = _style(**style_kw)
    a = np.asarray(sa, dtype="float64")
    b = np.asarray(sb, dtype="float64")
    a = a[:, None] if a.ndim == 1 else a
    b = b[:, None] if b.ndim == 1 else b
    if srb != sra:
        import librosa

        b = librosa.resample(b.T, orig_sr=srb, target_sr=sra).T
    if a.shape[1] != b.shape[1]:  # mono vs stereo: compare as mono
        a = a.mean(axis=1, keepdims=True)
        b = b.mean(axis=1, keepdims=True)
    la, lb = integrated_lufs(a, sra), integrated_lufs(b, sra)
    match = "integrated"
    if la is None or lb is None:
        la, lb, match = ungated_lufs(a, sra), ungated_lufs(b, sra), "ungated"
    diff = float(la - lb) if la is not None and lb is not None else 0.0
    gain_a, gain = min(0.0, -diff), min(0.0, diff)
    a = a * (10.0 ** (gain_a / 20.0))
    b = b * (10.0 ** (gain / 20.0))

    dur = max(a.shape[0], b.shape[0]) / float(sra)
    grid = make_grid(a.mean(axis=1), sra, dur, intent=intent, bpm=bpm,
                     beats_per_bar=beats_per_bar, detect=detect_beats)
    hop2 = _cqt_hop(round(dur * sra))
    an_a = analyze(a, sra, grid=grid, intent=intent, dur=dur, cqt_hop=hop2, sections=False)
    an_b = analyze(b, sra, grid=grid, intent=None, dur=dur, cqt_hop=hop2, sections=False)
    png = render_ab(an_a, an_b, st, names, gain_a, gain)

    unit = grid["unit"]
    rows = []
    for ra, rb in zip(an_a["summary"]["rows"], an_b["summary"]["rows"]):
        def d(x, y):
            return _r(y - x) if x is not None and y is not None else None
        rows.append({
            unit: ra[unit], "t0": ra["t0"], "t1": ra["t1"],
            "lufs_a": ra["lufs"], "lufs_b": rb["lufs"], "lufs_delta": d(ra["lufs"], rb["lufs"]),
            "bands_a_db": ra["bands_db"], "bands_b_db": rb["bands_db"],
            "bands_delta_db": {k: d(ra["bands_db"][k], rb["bands_db"][k]) for k in ra["bands_db"]},
            "corr_a": ra["corr"], "corr_b": rb["corr"],
            "onsets_a": ra["onsets"], "onsets_b": rb["onsets"],
        })
    doc = {
        "v": 1, "op_version": OP_VERSION, "name": f"{names[0]} vs {names[1]}",
        "grid": {"source": grid["source"], "unit": unit, "bpm": _r(grid["bpm"], 3),
                 "beats_per_bar": grid["beats_per_bar"], "n": len(grid["bars"])},
        "ab": {"a": names[0], "b": names[1], "match": match, "gain_db_a": _r(gain_a, 3),
               "gain_db_b": _r(gain, 3), "lufs_a": _r(la), "lufs_b": _r(lb),
               "lufs_matched": _r(la + gain_a if la is not None else None)},
        "global": an_a["summary"]["global"],
        "global_b": an_b["summary"]["global"],
        "rows": rows,
        "render": {**st.params(), **an_a["params"]},
    }
    return png, doc


def sheet_ab_frames(frame_a: dict, frame_b: dict, *, intent_path: str | None = None,
                    bpm: float | None = None, beats_per_bar: float | None = None,
                    detect_beats: bool = True, **style_kw) -> list[dict]:
    """Two `audio` frames → A/B ``[image (sheet:ab), feature (sheet:ab), text (sheet:ab:summary)]``."""
    from . import intent as I

    ya, sra = _load(frame_a)
    yb, srb = _load(frame_b)
    intent = I.for_audio_frame(frame_a, explicit=intent_path)
    names = (frame_name(frame_a), frame_name(frame_b))
    png, doc = ab_arrays(ya, sra, yb, srb, names=names, intent=intent, bpm=bpm,
                         beats_per_bar=beats_per_bar, detect_beats=detect_beats, **style_kw)
    params = {**doc["render"], "grid_source": doc["grid"]["source"], "gain_db_a": doc["ab"]["gain_db_a"],
              "gain_db_b": doc["ab"]["gain_db_b"]}
    return _emit(doc, png, role="sheet:ab", of=frame_a["id"],
                 lineage=[frame_a["id"], frame_b["id"]], params=params, size=_png_size(png))
