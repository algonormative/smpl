"""Conform: measured vs intended (vault-16dd.2).

`conform` compares a clip to what its generator meant it to be, read from the normalized
intent document (:mod:`smpl_analysis.intent`). It is the checks of the Assembly run's
hand-written gate (``assembly-serum-01/gate.py``) that held up, lifted into an op and
stripped of their keep/cut policy: every check reports **measured**, **intended** and
**delta**; tolerances are flags with documented defaults, and the only thing a tolerance
does is decide which hits get a marker and whether a check's ``over_tol`` boolean is set.
A check with no stated intent reports ``None`` for the intended value and the delta; it is
never guessed.

Checks
------
length
    Measured duration vs intent ``dur_s``. Default tolerance 1 ms (gate.py's).
timing
    Per intended hit, how far the measured onset sits from where the recipe puts it
    (:func:`timing_errors`, a port of gate.py's ``timing_ms``). Median / p90 of the absolute
    error, the signed mean (negative = early, positive = late), and the count measured.
pitch classes
    Intent ``pitch_classes`` vs chroma. See :func:`pitch_fit` for why this is a rotation
    test and not a plain in-set threshold.
loudness
    Integrated LUFS and 4x-oversampled true peak, vs intent ``lufs`` / ``true_peak_db``
    (or the ``lufs`` / ``true_peak_db`` arguments, which win).

Why intent and not a detector
-----------------------------
Recipe hit times beat onset detectors: librosa onsets read 15-49 ms medians on plain
four-on-the-floor kick stems (a floor-tom kick's noise burst fires extra onsets), and a
chroma "white share" read 0.60-0.70 on real voices from harmonics alone (A's 5th partial,
C#, lands off the set). So the reference is the generator's own record of what it played,
and the measurement is how far the audio sits from it.

Pure functions returning dicts / frame dicts; numpy, librosa and soundfile are imported
inside the functions so a cold pipe stage starts fast.
"""

from __future__ import annotations

OP = "conform"
OP_VERSION = "conform@1"

# gate.py's thresholds, now flags.
LENGTH_TOL_MS = 1.0
TIMING_TOL_MS = 8.0
ACCENT_VELOCITY = 0.6          # hits quieter than this (ghosts) have no onset of their own
ISOLATION_S = 0.06             # a hit with another within 60 ms is inside a ratchet / under a tail
MIN_TIMED_HITS = 3             # fewer measurable hits than this and the timing median is None
PC_FLAG_SHARE = 0.20           # see pitch_fit

MEMO_DEFAULTS = {
    "length_tol_ms": LENGTH_TOL_MS,
    "timing_tol_ms": TIMING_TOL_MS,
    "accent_velocity": ACCENT_VELOCITY,
    "isolation_s": ISOLATION_S,
    "pc_flag_share": PC_FLAG_SHARE,
}

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Keys are registered in feature-keys.md (vault-16dd.2).
FEATURE_KEYS = (
    "conform.length_s", "conform.length_intended_s", "conform.length_delta_ms",
    "conform.timing_median_ms", "conform.timing_p90_ms", "conform.timing_mean_ms",
    "conform.timing_n", "conform.timing_intended_n",
    "conform.pc_in_set_share", "conform.pc_fit_margin", "conform.pc_best_rotation", "conform.pc_concentration",
    "conform.pc_shares", "conform.pc_offset_flagged", "conform.pc_outside_allowed",
    "conform.lufs", "conform.lufs_intended", "conform.lufs_delta",
    "conform.true_peak_dbtp", "conform.true_peak_intended_dbtp", "conform.true_peak_delta_db",
    "conform.over_tol",
)


def _r(x, nd=3):
    return None if x is None else round(float(x), nd)


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------
def timing_errors(m, sr: int, hits, length_s: float, *,
                  accent: float = ACCENT_VELOCITY, isolation: float = ISOLATION_S):
    """Signed onset error in seconds (+ late) for every measurable hit, as ``[(t, err_s)]``.

    Ported from gate.py's ``timing_ms`` (Assembly run), which returned the median of the
    absolute errors; this returns each one signed so the caller can report spread and bias.

    Measurable = accented (velocity >= ``accent``) with no other hit within ``isolation``: a
    ghost under a tail or a 32nd inside a ratchet has no onset of its own to find. For each,
    the 1 ms RMS envelope's quietest point in the 30 ms before the hit is the floor; the onset
    is where it first climbs 20 % of the way from there to the peak of the next 30 ms. The
    loop is tiled twice so a hit on sample 0 has the loop's own tail before it.

    History: librosa onsets missed hits at t = 0 and put ringing bells 14 ms early
    (backtracking); "steepest rise" put a floor-tom kick 14 ms late (its body swells after
    the click); a fixed window floor read dense grooves as 30 ms off. Controls: kick stems
    ~0.3 ms, the same files shifted 20 ms late ~20 ms.

    ``hits`` are dicts with ``t`` and optional ``velocity`` (the intent shape). The error is
    capped at the 30 ms window: a hit with no rise at all reads +30 ms.
    """
    import numpy as np

    times = sorted(float(h["t"]) for h in hits)
    keep = [float(h["t"]) for h in hits
            if float(h.get("velocity", 1.0)) >= accent and 0 <= float(h["t"]) < length_s
            and not any(0 < abs(o - float(h["t"])) < isolation for o in times)]
    n = len(m)
    half = int(0.03 * sr)
    if not keep or n <= half:     # nothing to measure, or shorter than the 30 ms search window
        return []
    y = np.concatenate([m, m]).astype(np.float64)
    w = max(1, int(0.001 * sr))
    env = np.sqrt(np.convolve(y * y, np.ones(w) / w, mode="same"))
    out = []
    for t in keep:
        c = n + round(t * sr)                 # the hit in the second copy
        s = c - half + int(np.argmin(env[c - half:c + 1]))
        seg = env[s:c + half]
        above = np.nonzero(seg > seg[0] + 0.2 * (seg.max() - seg[0]))[0]
        out.append((t, (s + int(above[0]) - c) / sr if len(above) else half / sr))
    return out


def timing_summary(errs, tol_ms: float = TIMING_TOL_MS, min_hits: int = MIN_TIMED_HITS):
    """Median / p90 of |error| (ms), signed mean (ms) and the count; Nones below ``min_hits``."""
    import numpy as np

    if len(errs) < min_hits:
        return {"median_ms": None, "p90_ms": None, "mean_ms": None, "n": len(errs)}
    e = np.array([x for _, x in errs]) * 1000.0
    return {"median_ms": _r(np.median(np.abs(e))), "p90_ms": _r(np.percentile(np.abs(e), 90)),
            "mean_ms": _r(e.mean()), "n": len(errs)}


# ---------------------------------------------------------------------------
# pitch classes
# ---------------------------------------------------------------------------
def chroma_shares(m, sr: int):
    """Share of chroma energy per pitch class (index 0 = C), summed over frames, sum 1.

    ``librosa.feature.chroma_cqt`` (default hop, tuning estimated) with ``norm=None`` so loud
    frames count for more than the noise floor between notes. (gate.py normalised each frame
    to its max, which gives silence the same vote as a note; on 63 tonal kit files the
    unnormalised form put 98 % of originals on rotation 0 vs 94 %, and 0 % of +2 semitone
    copies vs 3 %.) Returns None for a silent clip.
    """
    import librosa
    import numpy as np

    c = librosa.feature.chroma_cqt(y=m, sr=sr, norm=None).sum(axis=1)
    tot = float(c.sum())
    return None if not np.isfinite(tot) or tot <= 0 else c / tot


def pc_concentration(shares):
    """1 - normalized entropy of the chroma shares: 0 = flat (noise, drums), 1 = one class.

    Reported, never gated: it tells a reader when a pitch verdict rests on near-flat chroma.
    It does not separate drums from tonal material. On 25 files per Assembly lane the medians
    read kick 0.020, metal 0.012, groove 0.014, but mass pads 0.048 and sub 0.090, with drum
    p90s up to 0.14, so which lanes get the pitch check is the caller's call (``checks``).
    """
    import numpy as np

    if shares is None:
        return None
    p = np.asarray(shares, dtype=float)
    return _r(1.0 + float((p * np.log(p + 1e-12)).sum()) / float(np.log(12)))


def pitch_fit(shares, intended, flag_share: float = PC_FLAG_SHARE):
    """How well chroma ``shares`` fit the intended pitch-class set.

    Returns ``in_set_share`` (chroma share inside the intended set), ``best_rotation`` and
    ``fit_margin``, and ``offset_flagged`` (the pitch classes outside the set whose share
    exceeds ``flag_share``, ``{name: share}``).

    A plain threshold on the in-set share cannot work: real voices and pads read 0.60-0.70
    inside the *white* set from harmonics alone, and a seven-note set captures 7/12 = 0.58 of
    flat noise. So the intended set is also scored at every semitone rotation. If the audio
    really plays the set, rotation 0 scores best (``best_rotation`` 0, ``fit_margin`` > 0:
    the in-set share minus the best other rotation's). If it was transposed, another
    rotation wins (``best_rotation`` +2 for a clip two semitones sharp) and the margin goes
    negative. A set that maps to itself under rotation (the chromatic set) has no meaningful
    margin and reads None.

    ``flag_share`` is for the per-class view: a harmonic leaks a few percent into a foreign
    class (the 5th partial of A is C#), a note actually played off the set carries a note's
    weight. Measured on 63 tonal Assembly files (mass / voice / bed / sub, real intent from
    provenance): the largest off-set class share reads median 0.084, p90 0.115, max 0.165 on
    the originals and median 0.22, p90 0.53 on the same files pitched +2 semitones. The default
    0.20 flags none of the originals and about half of the shifted copies; the rotation test
    catches all of them (100 % of shifted copies off rotation 0, vs 2 % of originals). The pitch check only means
    something on pitched material: unpitched drums carry sample-bank pitch classes in their
    intent, but their chroma is noise.
    """
    import numpy as np

    idx = sorted(PITCH_CLASSES.index(p) for p in intended)
    if shares is None or not idx:
        return None
    s = np.asarray(shares, dtype=float)
    rot = [float(sum(s[(i + k) % 12] for i in idx)) for k in range(12)]
    same = [k for k in range(1, 12) if sorted((i + k) % 12 for i in idx) == idx]
    others = [rot[k] for k in range(1, 12) if k not in same]
    in_set = rot[0]
    best = int(np.argmax(rot))
    if rot[best] - in_set < 1e-9:   # ties go to "as intended"
        best = 0
    flagged = {PITCH_CLASSES[i]: _r(s[i]) for i in range(12)
               if i not in idx and s[i] > flag_share}
    return {
        "in_set_share": _r(in_set),
        "best_rotation": best if best <= 6 else best - 12,
        "fit_margin": _r(in_set - max(others)) if others and not same else None,
        "offset_flagged": flagged,
    }


# ---------------------------------------------------------------------------
# loudness
# ---------------------------------------------------------------------------
def loudness_values(x, sr: int):
    """(integrated LUFS, true peak dBTP) of ``x`` shaped (n, channels).

    Reuses :mod:`smpl_analysis.loudness` (pyloudnorm BS.1770 + 4x ``resample_poly`` true
    peak), the same measurements gate.py made, with gate.py's one convention on top: a clip
    under 1 s is zero-padded to 1 s before the integrated measurement (pyloudnorm needs a
    400 ms block; loudness reports -inf below it). The true peak is read on the unpadded clip.
    """
    import math

    import numpy as np

    from smpl_analysis import loudness as L

    pad = x if x.shape[0] >= sr else np.concatenate([x, np.zeros((sr - x.shape[0], x.shape[1]))])
    lufs = L._integrated_lufs(pad, sr)
    tp, _ = L._true_peak(x, sr, ceiling_dbtp=200.0)
    return (None if not math.isfinite(lufs) else lufs, None if not math.isfinite(tp) else tp)


# ---------------------------------------------------------------------------
# the op
# ---------------------------------------------------------------------------
def conform_signal(x, sr: int, intent: dict | None, *,
                   length_tol_ms: float = LENGTH_TOL_MS, timing_tol_ms: float = TIMING_TOL_MS,
                   pc_flag_share: float = PC_FLAG_SHARE, allowed_pcs=None,
                   lufs: float | None = None, true_peak_db: float | None = None,
                   accent_velocity: float = ACCENT_VELOCITY, isolation_s: float = ISOLATION_S,
                   checks=("length", "timing", "pitch", "loudness")) -> dict:
    """All checks on a decoded clip ``x`` (n, channels) against ``intent``.

    Returns ``{"features": {...conform.* keys...}, "markers": [...], "timing_errors": [...]}``.
    In-process entry point: loads nothing, so a caller can run it over many files without
    paying CLI start-up. ``allowed_pcs`` is an iterable of pitch-class names; the intended
    classes outside it are listed in ``conform.pc_outside_allowed`` (gate.py's "floor").
    """
    import numpy as np

    intent = intent or {}
    n = x.shape[0]
    dur = n / sr
    m = x.mean(axis=1).astype(np.float32)
    f: dict = {k: None for k in FEATURE_KEYS}
    over: dict = {}
    markers: list = []
    errs: list = []

    if "length" in checks:
        f["conform.length_s"] = _r(dur, 6)
        want = intent.get("dur_s")
        if want is not None:
            f["conform.length_intended_s"] = _r(want, 6)
            f["conform.length_delta_ms"] = _r((dur - want) * 1000.0)
            over["length"] = abs(dur - want) * 1000.0 > length_tol_ms

    hits = intent.get("hits")
    if "timing" in checks and hits:
        errs = timing_errors(m, sr, hits, dur, accent=accent_velocity, isolation=isolation_s)
        s = timing_summary(errs)
        f["conform.timing_median_ms"], f["conform.timing_p90_ms"] = s["median_ms"], s["p90_ms"]
        f["conform.timing_mean_ms"], f["conform.timing_n"] = s["mean_ms"], s["n"]
        f["conform.timing_intended_n"] = len(hits)
        if s["median_ms"] is not None:
            over["timing"] = s["median_ms"] > timing_tol_ms
        by_t = {round(float(h["t"]), 6): h for h in hits}
        for t, e in errs:
            if abs(e) * 1000.0 > timing_tol_ms:
                h = by_t.get(round(t, 6), {})
                markers.append({"t": round(t, 6), "sample": round(t * sr),
                                "label": f"hit {e * 1000.0:+.1f} ms"
                                         + (f" ({h['track']})" if h.get("track") else ""),
                                "error_ms": _r(e * 1000.0)})

    pcs = intent.get("pitch_classes")
    if "pitch" in checks:
        if allowed_pcs is not None and pcs:
            allowed = set(allowed_pcs)
            f["conform.pc_outside_allowed"] = [p for p in pcs if p not in allowed]
            over["allowed"] = bool(f["conform.pc_outside_allowed"])
        if pcs:
            shares = chroma_shares(m, sr)
            fit = pitch_fit(shares, pcs, pc_flag_share)
            if fit:
                f["conform.pc_in_set_share"] = fit["in_set_share"]
                f["conform.pc_fit_margin"] = fit["fit_margin"]
                f["conform.pc_best_rotation"] = fit["best_rotation"]
                f["conform.pc_concentration"] = pc_concentration(shares)
                f["conform.pc_shares"] = {PITCH_CLASSES[i]: _r(shares[i]) for i in range(12)}
                f["conform.pc_offset_flagged"] = fit["offset_flagged"]
                over["pitch"] = bool(fit["offset_flagged"]) or fit["best_rotation"] != 0

    if "loudness" in checks:
        lu, tp = loudness_values(x.astype("float64"), sr)
        want_lu = lufs if lufs is not None else intent.get("lufs")
        want_tp = true_peak_db if true_peak_db is not None else intent.get("true_peak_db")
        f["conform.lufs"] = _r(lu, 2)
        f["conform.true_peak_dbtp"] = _r(tp, 2)
        if want_lu is not None:
            f["conform.lufs_intended"] = float(want_lu)
            f["conform.lufs_delta"] = None if lu is None else _r(lu - want_lu, 2)
        if want_tp is not None:
            f["conform.true_peak_intended_dbtp"] = float(want_tp)
            f["conform.true_peak_delta_db"] = None if tp is None else _r(tp - want_tp, 2)

    f["conform.over_tol"] = sorted(k for k, v in over.items() if v)
    return {"features": f, "markers": markers, "timing_errors": errs}


def conform_file(path, intent_path: str | None = None, **kw) -> dict:
    """:func:`conform_signal` on an audio file, intent from ``intent_path`` or its sidecar."""
    import soundfile as sf

    from smpl_analysis import intent as _intent

    x, sr = sf.read(str(path), dtype="float64", always_2d=True)
    if intent_path:
        it = _intent.load(intent_path)
    else:
        side = _intent.sidecar_for(path)
        it = _intent.load(side) if side else None
    return conform_signal(x, int(sr), it, **kw)


def conform_audio_frame(audio_frame: dict, *, intent_path: str | None = None, **kw) -> list[dict]:
    """Resolve an `audio` frame's PCM and emit its conform frames.

    Returns a `feature` frame (role ``conform``) holding the ``conform.*`` keys and, when any
    intended hit missed by more than the timing tolerance, a `marker` frame (role ``conform``)
    with one point per miss. Intent comes from ``intent_path`` or the sidecar next to the
    frame's read source; with neither, every ``intended`` / ``delta`` key is None. The caller
    passes the input frame through.
    """
    import soundfile as sf
    from smplstream import cas
    from smplstream import frames as F

    from smpl_analysis import intent as _intent

    src = cas.get_path(audio_frame["hash"])
    x, sr = sf.read(str(src), dtype="float64", always_2d=True)
    sr = int(sr)
    it = _intent.for_audio_frame(audio_frame, intent_path)
    res = conform_signal(x, sr, it, **kw)

    of = audio_frame.get("id")
    params = {
        "length_tol_ms": kw.get("length_tol_ms", LENGTH_TOL_MS),
        "timing_tol_ms": kw.get("timing_tol_ms", TIMING_TOL_MS),
        "pc_flag_share": kw.get("pc_flag_share", PC_FLAG_SHARE),
        "allowed_pcs": sorted(kw["allowed_pcs"]) if kw.get("allowed_pcs") else None,
        "lufs": kw.get("lufs"), "true_peak_db": kw.get("true_peak_db"),
        "intent_source": (it or {}).get("source"),
        "sr_hz": sr,
    }
    common = {"of": of, "op": OP, "op_version": OP_VERSION,
              "lineage": [of] if of else None, "params": params}
    out = [F.feature_frame(res["features"], role="conform", **common)]
    if res["markers"]:
        out.append(F.marker_frame(res["markers"], role="conform", **common))
    return out
