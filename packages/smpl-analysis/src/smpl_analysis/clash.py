"""Clash — how layers interact, and whether a loop closes on itself (vault-16dd.4).

Layer problems were caught by hand: a drone holding 58 % of its energy in the kick's sub
range, a +2 transpose putting F♯ into two beds, a loop whose seam is only inferred. This op
is multi-input: every selected `audio` frame is a LAYER (a stem, a kit lane) and every pair
of layers is measured. It reports measured values only; what counts as "too much" is a
flag with a documented default (layering rule), never a keep/cut verdict.

Per pair (layers aligned at t=0, both mixed to mono, resampled to :data:`SR`)::

    erb        per-ERB-band co-activity. A band-frame is "active" for a layer when its power
               is within ``active_db`` of that band's own peak (and within ``floor_db`` of the
               layer's loudest band-frame, so a noise floor never counts). ``overlap`` is
               cells both layers occupy / cells either occupies; ``worst_bands`` rank by the
               share of the clip a band is contested.
    sub        energy below ``crossover_hz`` as a share of each layer's total energy, in the
               frames where both layers are active (frame power within ``active_db`` of the
               layer's peak frame). ``collision`` is the smaller of the two shares: both
               layers must live in the sub for it to collide.
    pitch      pitch-class profile of each layer built from its detected partials; the pitch
               classes one layer plays that the other's dominant set lacks, those that sit a
               semitone from a dominant one, and the sensory dissonance of the pair.
    dissonance Sethares (1993), after Plomp & Levelt (1965): for every pair of simultaneous
               partials (one from each layer, frequencies f1 <= f2, amplitudes a1, a2)
               ``d = a1 a2 (exp(-3.5 s (f2-f1)) - exp(-5.75 s (f2-f1)))`` with
               ``s = 0.24 / (0.0207 f1 + 18.96)``. The score is sum d / sum a1 a2 pooled over
               the frames where both are active: the amplitude-weighted mean roughness of a
               cross-layer partial pair, scale-free (0 for a unison or octave of pure
               tones, about 0.25 at the worst spacing). Partial amplitudes are relative to
               each layer's own loudest partial.

Per layer (``seam``; also run on a single input): the clip is rolled so the wrap sits
mid-clip and each wrap number is set against the same quantity around it, so a drone and a
drum loop are each judged against their own signal (a loop that opens on a hit reads as a hit,
because the reference region holds the hit too)::

    jump        |x[0] - x[-1]|, and its ratio to the rms sample step within ``local_ms``
    deriv       slope mismatch across the wrap over ``slope_ms``, as a ratio to the rms
                mismatch at nearby boundaries
    rms         RMS of the last vs first ``window_ms``, in dB, and its excess over the largest
                such step anywhere else in the clip
    flux        spectral flux of the frames that straddle the wrap, over the clip's median
                flux (``flux_ratio``) and over the largest flux elsewhere (``flux_vs_max``)

``flagged`` is any ratio over its flag; the flux flag also needs the wrap to be the clip's
most abrupt spectral change (``flux_vs_max`` over 1). Pitch classes the layers' sidecars
declare (`intent`) ride along as ``intent_*`` next to the measured ones.

Heavy imports (numpy/scipy/librosa) stay inside the functions.
"""

from __future__ import annotations

import math
from itertools import combinations

OP = "clash"
OP_VERSION = "clash@1"
SEAM_ROLE = "clash-seam"

# --- pinned recipe constants (memo params stay stable per op_version) ---------------------
SR = 32000                  # analysis rate: every layer is resampled here
N_FFT = 4096
HOP = 1024
ERB_LO_HZ = 30.0
ERB_HI_HZ = 15000.0
ACTIVE_DB = 20.0            # active = within this of the peak
FLOOR_DB = 60.0             # ... and within this of the layer's loudest band-frame
CROSSOVER_HZ = 80.0
PARTIALS_PER_FRAME = 10
PARTIAL_FLOOR_DB = 40.0     # partials below this (re the layer's loudest) are ignored
PARTIAL_LO_HZ = 40.0
PARTIAL_HI_HZ = 5000.0
PC_PLAYED = 0.25            # a pitch class is "played" at this fraction of the layer's peak
PC_DOMINANT = 0.5           # ... and "dominant" at this fraction
SPAN_S = 2.0                # marker window length
SPAN_COUNT = 3              # markers per problem kind per pair
SEAM_WINDOW_MS = 20.0
SEAM_SLOPE_MS = 1.0
SEAM_LOCAL_MS = 5.0
PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Sethares (1993) fit of the Plomp-Levelt curve.
_B1, _B2, _S1, _S2 = 3.5, 5.75, 0.0207, 18.96
_DSTAR = 0.24

_EPS = 1e-20

PAIR_KEYS: tuple[str, ...] = (
    "clash.erb_overlap",
    "clash.sub_collision",
    "clash.sub_share_a",
    "clash.sub_share_b",
    "clash.pc_clash_a",
    "clash.pc_clash_b",
    "clash.dissonance",
)
SEAM_KEYS: tuple[str, ...] = (
    "clash.seam_jump",
    "clash.seam_jump_ratio",
    "clash.seam_deriv_ratio",
    "clash.seam_rms_jump_db",
    "clash.seam_rms_excess_db",
    "clash.seam_flux_ratio",
    "clash.seam_flagged",
)


# ---------------------------------------------------------------------------
# Loading + per-layer analysis (done once per layer, reused by every pair).
# ---------------------------------------------------------------------------
def _load_mono(frame: dict, sr_out: int | None = None):
    """An audio frame's PCM as ``(mono float64, sr)``; resampled when ``sr_out`` is given."""
    import numpy as np
    import soundfile as sf
    from smplstream import cas

    data, sr = sf.read(str(cas.get_path(frame["hash"])), dtype="float64", always_2d=True)
    y = np.ascontiguousarray(data.mean(axis=1))
    sr = int(sr)
    if sr_out and sr != sr_out:
        from scipy.signal import resample_poly

        g = math.gcd(sr, sr_out)
        y = resample_poly(y, sr_out // g, sr // g)
        sr = sr_out
    return y, sr


def erb_edges(lo: float = ERB_LO_HZ, hi: float = ERB_HI_HZ) -> list[float]:
    """Band edges one ERB apart (Glasberg & Moore 1990 ERB-rate), ``lo``..``hi`` Hz."""
    def to_erb(f):
        return 21.4 * math.log10(1.0 + 0.00437 * f)

    def from_erb(e):
        return (10.0 ** (e / 21.4) - 1.0) / 0.00437

    n = math.floor(to_erb(hi) - to_erb(lo))
    edges = [from_erb(to_erb(lo) + i) for i in range(n + 1)]
    return edges + ([hi] if edges[-1] < hi * 0.999 else [])


def _partials(mag, freqs, ref: float):
    """Top ``PARTIALS_PER_FRAME`` spectral peaks per frame: ``(freq_hz, amp)`` arrays (T, N).

    Peaks are local maxima of the magnitude within the partial band, interpolated by a
    parabola through the three bins; amplitudes are relative to ``ref`` (the layer's loudest
    peak). Absent partials are amplitude 0.
    """
    import numpy as np

    lo = np.searchsorted(freqs, PARTIAL_LO_HZ).item()
    hi = np.searchsorted(freqs, PARTIAL_HI_HZ).item()
    m = mag[:, lo - 1:hi + 1]                       # one guard bin each side
    c = m[:, 1:-1]
    left, right = m[:, :-2], m[:, 2:]
    floor = ref * 10.0 ** (-PARTIAL_FLOOR_DB / 20.0)
    is_peak = (c > left) & (c >= right) & (c >= floor)
    score = np.where(is_peak, c, 0.0)
    n = min(PARTIALS_PER_FRAME, score.shape[1])
    idx = np.argpartition(-score, n - 1, axis=1)[:, :n]
    rows = np.arange(score.shape[0])[:, None]
    amp = score[rows, idx]
    a, b, g = left[rows, idx], c[rows, idx], right[rows, idx]
    den = a - 2.0 * b + g
    delta = np.where(np.abs(den) > 1e-30, 0.5 * (a - g) / np.where(den == 0, 1, den), 0.0)
    delta = np.clip(delta, -0.5, 0.5)
    df = freqs[1] - freqs[0]
    f = (lo + idx + delta) * df
    f = np.where(amp > 0, f, 0.0)
    return f, amp / max(ref, _EPS)


def analyse_layer(y, sr: int, *, crossover_hz: float = CROSSOVER_HZ) -> dict:
    """Everything a pair needs from one layer: band powers, sub energy, partials, activity."""
    import librosa
    import numpy as np

    S = librosa.stft(np.asarray(y, dtype="float32"), n_fft=N_FFT, hop_length=HOP,
                     window="hann", center=True)
    P = (np.abs(S) ** 2).astype("float64").T              # (T, bins)
    mag = np.sqrt(P)
    freqs = np.linspace(0.0, sr / 2.0, P.shape[1])
    edges = erb_edges()
    band_idx = np.clip(np.searchsorted(edges, freqs, side="right") - 1, -1, len(edges) - 1)
    bands = np.zeros((P.shape[0], len(edges) - 1))
    for b in range(len(edges) - 1):
        sel = band_idx == b
        if sel.any():
            bands[:, b] = P[:, sel].sum(axis=1)
    total = P.sum(axis=1)
    sub = P[:, freqs < crossover_hz].sum(axis=1)
    pf, pa = _partials(mag, freqs, float(mag.max()) if mag.size else 1.0)
    return {"sr": sr, "hop_s": HOP / sr, "edges": edges, "bands": bands, "total": total,
            "sub": sub, "pf": pf, "pa": pa, "n": P.shape[0], "crossover_hz": crossover_hz}


def _frame_active(total, active_db: float):
    peak = float(total.max()) if total.size else 0.0
    return (total > 0) & (total >= peak * 10.0 ** (-active_db / 10.0))


def _band_active(bands, active_db: float, floor_db: float):
    import numpy as np

    peak = bands.max(axis=0, keepdims=True)
    gpeak = float(bands.max()) if bands.size else 0.0
    thr = np.maximum(peak * 10.0 ** (-active_db / 10.0), gpeak * 10.0 ** (-floor_db / 10.0))
    return (bands > 0) & (bands >= thr)


def _pad(a, n: int):
    import numpy as np

    if a.shape[0] >= n:
        return a[:n]
    pad = np.zeros((n - a.shape[0],) + a.shape[1:], dtype=a.dtype)
    return np.concatenate([a, pad], axis=0)


def _top_spans(score, hop_s: float, k: int = SPAN_COUNT, span_s: float = SPAN_S):
    """The ``k`` highest-scoring non-overlapping windows: ``[(t, dur, mean_score), ...]``."""
    import numpy as np

    n = len(score)
    if n == 0:
        return []
    w = max(1, min(n, round(span_s / hop_s)))
    csum = np.concatenate([[0.0], np.cumsum(score)])
    means = (csum[w:] - csum[:-w]) / w                    # window starting at i
    out = []
    taken = means.copy()
    for _ in range(k):
        i = int(np.argmax(taken))
        if taken[i] <= 0:
            break
        out.append((round(i * hop_s, 4), round(w * hop_s, 4), float(means[i])))
        taken[max(0, i - w + 1):i + w] = -1.0
    return out


# ---------------------------------------------------------------------------
# Pair measurements.
# ---------------------------------------------------------------------------
def _pc_profile(f, a, mask):
    """12-bin pitch-class profile (sums to 1, or all 0) from partial freqs/amps in masked frames."""
    import numpy as np

    f, a = f[mask], a[mask]
    ok = (a > 0) & (f > 0)
    prof = np.zeros(12)
    if ok.any():
        pc = np.rint(69.0 + 12.0 * np.log2(f[ok] / 440.0)).astype(int) % 12
        np.add.at(prof, pc, a[ok] ** 2)
    s = prof.sum()
    return prof / s if s > 0 else prof


def _dissonance(fa, aa, fb, ab, mask):
    """Pooled Sethares dissonance + the per-frame raw sum (for markers)."""
    import numpy as np

    f1, f2 = fa[:, :, None], fb[:, None, :]
    w = aa[:, :, None] * ab[:, None, :]
    lo, df = np.minimum(f1, f2), np.abs(f1 - f2)
    s = _DSTAR / (_S1 * lo + _S2)
    d = w * (np.exp(-_B1 * s * df) - np.exp(-_B2 * s * df))
    d = np.where(w > 0, d, 0.0)
    per_frame = d.sum(axis=(1, 2)) * mask
    weight = (w.sum(axis=(1, 2)) * mask)
    tot = float(weight.sum())
    return (float(per_frame.sum()) / tot if tot > 0 else None), per_frame


def measure_pair(A: dict, B: dict, *, active_db: float = ACTIVE_DB, floor_db: float = FLOOR_DB,
                 pc_played: float = PC_PLAYED, pc_dominant: float = PC_DOMINANT) -> dict:
    """All pair measurements from two :func:`analyse_layer` results. Pure numpy."""
    import numpy as np

    n = max(A["n"], B["n"])
    hop_s = A["hop_s"]
    tot_a, tot_b = _pad(A["total"], n), _pad(B["total"], n)
    sub_a, sub_b = _pad(A["sub"], n), _pad(B["sub"], n)
    both = _frame_active(tot_a, active_db) & _frame_active(tot_b, active_db)

    # --- ERB overlap ---------------------------------------------------------------
    act_a = _band_active(_pad(A["bands"], n), active_db, floor_db)
    act_b = _band_active(_pad(B["bands"], n), active_db, floor_db)
    co, either = act_a & act_b, act_a | act_b
    n_either = int(either.sum())
    nb = co.shape[1]
    edges = A["edges"]
    per_band = []
    for b in range(nb):
        e = int(either[:, b].sum())
        per_band.append({
            "lo_hz": round(edges[b], 1), "hi_hz": round(edges[b + 1], 1),
            "overlap": round(float(co[:, b].sum()) / e, 4) if e else 0.0,
            "co_active": round(float(co[:, b].sum()) / max(n, 1), 4),
        })
    worst = sorted(per_band, key=lambda r: (-r["co_active"], -r["overlap"]))[:5]
    worst = [r for r in worst if r["co_active"] > 0]
    erb_score = co.sum(axis=1) / max(nb, 1)
    erb = {
        "overlap": round(float(co.sum()) / n_either, 4) if n_either else 0.0,
        "co_active_frac": round(float(co.sum()) / max(co.size, 1), 4),
        "worst_bands": worst,
    }

    # --- sub collision -------------------------------------------------------------
    nboth = int(both.sum())
    share_a = float(sub_a[both].sum() / max(tot_a[both].sum(), _EPS)) if nboth else 0.0
    share_b = float(sub_b[both].sum() / max(tot_b[both].sum(), _EPS)) if nboth else 0.0
    fa_sh = np.divide(sub_a, tot_a, out=np.zeros(n), where=tot_a > 0)
    fb_sh = np.divide(sub_b, tot_b, out=np.zeros(n), where=tot_b > 0)
    sub_score = np.minimum(fa_sh, fb_sh) * both
    sub = {
        "crossover_hz": A["crossover_hz"],
        "share_a": round(share_a, 4), "share_b": round(share_b, 4),
        "collision": round(min(share_a, share_b), 4),
        "both_active_frac": round(nboth / max(n, 1), 4),
    }

    # --- pitch classes + dissonance --------------------------------------------------
    fa, aa = _pad(A["pf"], n), _pad(A["pa"], n)
    fb, ab = _pad(B["pf"], n), _pad(B["pa"], n)
    pa_prof, pb_prof = _pc_profile(fa, aa, both), _pc_profile(fb, ab, both)

    def sets(prof):
        m = prof.max()
        if m <= 0:
            return set(), set()
        return ({i for i in range(12) if prof[i] >= pc_played * m},
                {i for i in range(12) if prof[i] >= pc_dominant * m})

    played_a, dom_a = sets(pa_prof)
    played_b, dom_b = sets(pb_prof)
    foreign_a = sorted(i for i in played_a if i not in dom_b) if dom_b else []
    foreign_b = sorted(i for i in played_b if i not in dom_a) if dom_a else []

    def semis(foreign, other_dom):
        return [[PC_NAMES[i], PC_NAMES[j]] for i in foreign for j in sorted(other_dom)
                if (i - j) % 12 in (1, 11)]

    diss, diss_frames = _dissonance(fa, aa, fb, ab, both)
    pitch = {
        "profile_a": [round(float(v), 4) for v in pa_prof],
        "profile_b": [round(float(v), 4) for v in pb_prof],
        "dominant_a": [PC_NAMES[i] for i in sorted(dom_a)],
        "dominant_b": [PC_NAMES[i] for i in sorted(dom_b)],
        "foreign_a": [PC_NAMES[i] for i in foreign_a],
        "foreign_b": [PC_NAMES[i] for i in foreign_b],
        "foreign_share_a": round(float(sum(pa_prof[i] for i in foreign_a)), 4),
        "foreign_share_b": round(float(sum(pb_prof[i] for i in foreign_b)), 4),
        "semitone_a": semis(foreign_a, dom_b),
        "semitone_b": semis(foreign_b, dom_a),
        "dissonance": None if diss is None else round(diss, 5),
    }

    spans = {
        "erb-overlap": _top_spans(erb_score, hop_s),
        "sub-collision": _top_spans(sub_score, hop_s),
        "dissonance": _top_spans(diss_frames, hop_s),
    }
    return {"erb": erb, "sub": sub, "pitch": pitch, "spans": spans}


def pair_features(m: dict) -> dict:
    """The registered flat ``clash.*`` scalars from a :func:`measure_pair` result."""
    return {
        "clash.erb_overlap": m["erb"]["overlap"],
        "clash.sub_collision": m["sub"]["collision"],
        "clash.sub_share_a": m["sub"]["share_a"],
        "clash.sub_share_b": m["sub"]["share_b"],
        "clash.pc_clash_a": m["pitch"]["foreign_share_a"],
        "clash.pc_clash_b": m["pitch"]["foreign_share_b"],
        "clash.dissonance": m["pitch"]["dissonance"],
    }


# ---------------------------------------------------------------------------
# Loop seam.
# ---------------------------------------------------------------------------
def _interior_p(v, q: float):
    import numpy as np

    return float(np.percentile(v, q)) if len(v) else 0.0


def measure_seam(y, sr: int, *, window_ms: float = SEAM_WINDOW_MS, slope_ms: float = SEAM_SLOPE_MS,
                 local_ms: float = SEAM_LOCAL_MS, jump_ratio: float = 4.0,
                 rms_excess_db: float = 6.0, flux_ratio: float = 8.0) -> dict:
    """How badly the clip's last sample meets its first when it loops.

    The clip is rolled so the wrap sits mid-clip, then every wrap number is set against the
    same quantity measured around it (``local_ms`` either side) or across the rest of the
    clip, so a drone and a drum loop are each judged against their own signal rather than an
    absolute scale. A loop that opens on a hit reads as a hit, not a click, because the
    reference region contains the hit too. ``flagged`` is any ratio over its flag threshold
    (flags, not verdicts: the numbers are the report).
    """
    import librosa
    import numpy as np

    y = np.asarray(y, dtype="float64")
    N = len(y)
    out: dict = {"samples": N, "sr_hz": sr, "dur_s": round(N / sr, 4)}
    if N < 256:
        raise ValueError("clip too short to check a loop seam")
    mid = N // 2
    r = np.roll(y, mid)                     # r[mid] == y[0]; the wrap is the step r[mid-1] -> r[mid]
    peak = float(np.max(np.abs(y)))
    L = max(8, min(round(local_ms * 1e-3 * sr), mid // 2))

    # 1. sample jump: the step across the wrap vs the steps around it.
    d = np.diff(r)                          # d[i] = r[i+1] - r[i]; the wrap step is d[mid-1]
    w_step = abs(float(d[mid - 1]))
    near = np.concatenate([d[mid - 1 - L:mid - 1], d[mid:mid + L]])
    ref = float(np.sqrt(np.mean(near ** 2))) if near.size else 0.0
    out["jump"] = round(w_step, 8)
    out["jump_ratio"] = round(w_step / ref, 3) if ref > 0 else 0.0

    # 2. slope mismatch over ~slope_ms either side, vs the same mismatch at nearby boundaries.
    k = max(4, min(round(slope_ms * 1e-3 * sr), mid // 4))
    lo, hi = mid - L - k, mid + L + k
    seg = r[lo - k:hi + k]
    # slope of seg[i-k..i-1] vs seg[i..i+k-1] at every boundary i in [k, len-k]
    idx = np.arange(k, len(seg) - k + 1)
    before = (seg[idx - 1] - seg[idx - k]) / (k - 1)
    after = (seg[idx + k - 1] - seg[idx]) / (k - 1)
    mis = np.abs(after - before)
    centre = mid - (lo - k)
    at_wrap = float(mis[centre - k])
    far = np.abs(idx - centre) >= k
    ref = float(np.sqrt(np.mean(mis[far] ** 2))) if far.any() else 0.0
    out["deriv_jump"] = round(at_wrap, 8)
    out["deriv_ratio"] = round(at_wrap / ref, 3) if ref > 0 else 0.0

    # 3. short-window RMS step at the wrap vs the largest step anywhere else in the clip.
    w = max(8, min(round(window_ms * 1e-3 * sr), N // 8))
    floor = max(peak * 1e-4, 1e-9)          # -80 dB re peak: digital silence is not -inf
    c2 = np.concatenate([[0.0], np.cumsum(r * r)])
    rb = np.sqrt((c2[w:-w] - c2[:-2 * w]) / w)           # rms of r[n-w..n-1], n = w..N-w
    ra = np.sqrt((c2[2 * w:] - c2[w:-w]) / w)
    n_r = min(len(rb), len(ra))
    db = np.abs(20.0 * np.log10((ra[:n_r] + floor) / (rb[:n_r] + floor)))
    centre = mid - w
    wrap_db = float(db[centre])
    away = np.abs(np.arange(n_r) - centre) >= w
    out["rms_jump_db"] = round(wrap_db, 3)
    out["rms_excess_db"] = round(wrap_db - float(db[away].max()), 3) if away.any() else 0.0

    # 4. spectral flux of the frames straddling the wrap, vs the median and the largest flux
    #    elsewhere in the clip.
    n_fft = min(2048, 2 ** math.floor(math.log2(max(N // 8, 64))))
    hop = max(1, n_fft // 8)
    S = np.abs(librosa.stft(r.astype("float32"), n_fft=n_fft, hop_length=hop,
                            window="hann", center=False))
    Lg = np.log1p(100.0 * S / max(float(S.max()), _EPS))
    flux = np.sqrt((np.maximum(Lg[:, 1:] - Lg[:, :-1], 0.0) ** 2).sum(axis=0))  # flux[i]: frame i+1
    starts = (np.arange(flux.shape[0]) + 1) * hop
    touch = (starts < mid) & (starts + n_fft > mid)
    clear = (starts + n_fft <= mid) | (starts >= mid + hop)   # frames (and their predecessor) off the wrap
    if touch.any() and clear.any():
        wrap_flux = float(flux[touch].max())
        med = max(float(np.median(flux)), 1e-6)
        out["flux_wrap"] = round(wrap_flux, 4)
        out["flux_median"] = round(med, 4)
        out["flux_ratio"] = round(wrap_flux / med, 3)
        top = float(flux[clear].max())
        out["flux_vs_max"] = round(wrap_flux / top, 3) if top > 0 else None
    else:
        out["flux_wrap"] = out["flux_median"] = out["flux_ratio"] = out["flux_vs_max"] = None

    reasons = []
    if out["jump_ratio"] > jump_ratio:
        reasons.append("sample-jump")
    if out["deriv_ratio"] > jump_ratio:
        reasons.append("slope-jump")
    if out["rms_excess_db"] > rms_excess_db:
        reasons.append("rms-jump")
    if (out["flux_ratio"] is not None and out["flux_ratio"] > flux_ratio
            and out["flux_vs_max"] is not None and out["flux_vs_max"] > 1.0):
        reasons.append("flux")
    out["flagged"] = bool(reasons)
    out["reasons"] = reasons
    out["flags"] = {"jump_ratio": jump_ratio, "rms_excess_db": rms_excess_db,
                    "flux_ratio": flux_ratio}
    return out


def seam_features(s: dict) -> dict:
    return {
        "clash.seam_jump": s["jump"],
        "clash.seam_jump_ratio": s["jump_ratio"],
        "clash.seam_deriv_ratio": s["deriv_ratio"],
        "clash.seam_rms_jump_db": s["rms_jump_db"],
        "clash.seam_rms_excess_db": s["rms_excess_db"],
        "clash.seam_flux_ratio": s["flux_ratio"],
        "clash.seam_flagged": s["flagged"],
    }


# ---------------------------------------------------------------------------
# Frame emission.
# ---------------------------------------------------------------------------
def _layer_names(frames: list[dict]) -> list[str]:
    """A short unique label per layer: its role, else its source file stem, else its id."""
    from pathlib import Path

    def uniq(names):
        return len(set(names)) == len(names)

    roles = [f.get("role") or "audio" for f in frames]
    if uniq(roles):
        return roles
    stems = [Path((f.get("params") or {}).get("source") or "").name.split(".")[0] or None
             for f in frames]
    if all(stems) and uniq(stems):
        return stems
    return [f.get("id") or f"layer{i}" for i, f in enumerate(frames)]


def _intent_pcs(frame: dict):
    """The pitch classes a layer's sidecar says it plays, or None (no sidecar / none stated)."""
    from smpl_analysis import intent

    try:
        doc = intent.for_audio_frame(frame)
    except (OSError, ValueError):
        return None
    return (doc or {}).get("pitch_classes") or None


def _layer_doc(frame: dict, name: str) -> dict:
    return {"id": frame.get("id"), "role": frame.get("role"), "name": name}


def clash_audio_frames(
    audio_frames: list[dict],
    *,
    pairs: bool = True,
    seam: bool = False,
    crossover_hz: float = CROSSOVER_HZ,
    active_db: float = ACTIVE_DB,
    floor_db: float = FLOOR_DB,
    pc_played: float = PC_PLAYED,
    pc_dominant: float = PC_DOMINANT,
    window_ms: float = SEAM_WINDOW_MS,
    jump_ratio: float = 4.0,
    rms_excess_db: float = 6.0,
    flux_ratio: float = 8.0,
) -> list[dict]:
    """Resolve each audio frame, measure every pair (and optionally every seam).

    Returns, per pair, a `feature` frame (role ``clash``; ``data`` is the document naming
    both layers), a `feature` frame (role ``clash-features``; the registered scalars) and a
    `marker` frame (role ``clash``; where each problem peaks); per seam, a `feature` frame
    (role ``clash-seam``) and, when flagged, a `marker` frame at the wrap. The caller
    passes the inputs through.
    """
    from smplstream import frames as F

    names = _layer_names(audio_frames)
    params = {
        "crossover_hz": crossover_hz, "active_db": active_db, "floor_db": floor_db,
        "pc_played": pc_played, "pc_dominant": pc_dominant, "analysis_sr_hz": SR,
        "n_fft": N_FFT, "hop": HOP, "layers": len(audio_frames),
    }
    out: list[dict] = []

    if pairs and len(audio_frames) >= 2:
        layers = []
        for fr in audio_frames:
            y, sr = _load_mono(fr, SR)
            layers.append(analyse_layer(y, sr, crossover_hz=crossover_hz))
        intent_pcs = [_intent_pcs(fr) for fr in audio_frames]
        for (i, a), (j, b) in combinations(list(enumerate(audio_frames)), 2):
            m = measure_pair(layers[i], layers[j], active_db=active_db, floor_db=floor_db,
                             pc_played=pc_played, pc_dominant=pc_dominant)
            ids = [x for x in (a.get("id"), b.get("id")) if x]
            common = {"of": a.get("id"), "op": OP, "op_version": OP_VERSION,
                      "lineage": ids or None, "params": {**params, "pair": [names[i], names[j]]}}
            doc = {
                "a": _layer_doc(a, names[i]), "b": _layer_doc(b, names[j]),
                "pair": f"{names[i]}|{names[j]}",
                "erb": m["erb"], "sub": m["sub"], "pitch": m["pitch"],
            }
            ia, ib = intent_pcs[i], intent_pcs[j]
            if ia is not None and ib is not None:   # what the generators said they played
                doc["pitch"]["intent_a"], doc["pitch"]["intent_b"] = ia, ib
                doc["pitch"]["intent_foreign_a"] = [p for p in ia if p not in ib]
                doc["pitch"]["intent_foreign_b"] = [p for p in ib if p not in ia]
            points = []
            for kind, spans in m["spans"].items():
                for t, dur, score in spans:
                    points.append({"t": t, "dur": dur, "label": f"{kind} {names[i]}|{names[j]}",
                                   "value": round(score, 5)})
            points.sort(key=lambda p: p["t"])
            out.append(F.feature_frame(doc, role="clash", **common))
            out.append(F.feature_frame(pair_features(m), role="clash-features", **common))
            out.append(F.marker_frame(points, role="clash", **common))

    if seam or len(audio_frames) == 1:
        for i, fr in enumerate(audio_frames):
            y, sr = _load_mono(fr)
            s = measure_seam(y, sr, window_ms=window_ms, jump_ratio=jump_ratio,
                             rms_excess_db=rms_excess_db, flux_ratio=flux_ratio)
            common = {"of": fr.get("id"), "op": OP, "op_version": OP_VERSION,
                      "lineage": [fr["id"]] if fr.get("id") else None,
                      "params": {"window_ms": window_ms, "layer": names[i]}}
            doc = {"layer": _layer_doc(fr, names[i]), **s, "features": seam_features(s)}
            out.append(F.feature_frame(doc, role=SEAM_ROLE, **common))
            if s["flagged"]:
                out.append(F.marker_frame(
                    [{"t": round(s["dur_s"], 4), "label": "seam " + "+".join(s["reasons"]),
                      "value": s["jump_ratio"]}], role=SEAM_ROLE, **common))
    return out
