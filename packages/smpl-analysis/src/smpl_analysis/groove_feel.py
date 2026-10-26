"""Groove feel — what a rhythm *feels* like, as numbers and as text an LLM can read
(vault-16dd.7).

`smpl groove` answers "is it on the grid, and how far off": one swing value plus a per-step
nudge. It does not say whether the pattern is syncopated, whether the hat rushes while the
snare drags, or which steps carry the accent. This op reports those, and — the point —
exports the hits as compact symbolic text. Language models reason well over symbolic rhythm
and badly over audio, so a step grid with velocities and millisecond offsets is a better
thing to hand Claude than a spectrogram.

Hits
----
* **Intent exists** (an ``intent`` dict from :mod:`smpl_analysis.intent`): the generator's
  hit list is the reference. Each hit's cell (bar, step) is the one the generator named, or,
  when only times are known, the nearest cell of the *swung* grid (the same assignment
  ``groove`` uses — a 0.62-swing offbeat is nearer the next straight step than its own).
  Then the audio is consulted: the hit's **actual onset** is the first point where the 1 ms RMS
  envelope has climbed 20 % of the way from the quietest point just before the hit to the
  peak just after it (the attack start; librosa onsets read 15-49 ms late on plain kick stems
  and miss hits at t=0 — see `music/sample-generation/smplvst/assembly-serum-01/gate.py`),
  and its **level** is the peak around that onset. A hit with another hit within 60 ms, or no
  clear rise, is marked ``ok: false`` and left out of the timing statistics: a ghost under a
  tail has no onset of its own to find.
* **No intent**: onsets are detected and fitted to the grid by ``groove.extract_groove`` (not
  re-implemented here), one anonymous track ``all``. That fit aligns the grid origin on the
  odd-step hits, so a *uniform* push or drag of the whole loop is invisible to it — give an
  intent (an absolute grid) or compare against ``--ref`` when that matters.

Offsets are signed milliseconds against the **straight** grid (so swing shows up as a
positive offset on even steps, a hat that rushes as a negative one).

Syncopation
-----------
Longuet-Higgins & Lee (1984), as formalised for rhythm patterns by Fitch & Rosenfeld
(2007): every step gets a metrical weight (bar 0, then 0 > -1 > -2 ...  down the metrical
hierarchy; for 16 steps of 4/4: 0, -4, -3, -4, -2, -4, -3, -4, -1 ...). An onset followed by
silence on a *stronger* step, before the next onset, is a syncopation worth the weight
difference (largest silent step in the gap). Summed per bar. Each bar is treated as a loop of
its own (the gap after the last onset runs through the bar line to the first onset). The raw
index is divided by the maximum any pattern can score on the grid to give a 0-1
``syncopation_norm`` comparable across meters. Grids other than 4/4-by-16 get a hierarchy from
the prime factors of the beats per bar then of the steps per beat (12 steps of 4/4: 2,2,3;
20 steps of 5/4: 5,2,2).

Level
-----
Per grid step, the mean peak level of the hits that fall on it, in dB relative to the loudest
step: the accent profile. It is the level *of the mix* at that position — simultaneous hits
on several tracks share one number.

Reference grooves
-----------------
``references`` are per-step timing profiles (a ``smpl groove --out`` document, a
groove-feel document, or audio analysed by this same op). The comparison is the per-step
difference ``this - reference``; several references are averaged first.

Pure functions; numpy / soundfile imported lazily so cold pipe stages start fast.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

OP = "groove-feel"
OP_VERSION = "groove-feel@1"

GRID_STEPS = 16
BEATS_PER_BAR = 4.0
SEARCH_MS = 30.0          # attack search window after the intended time
LEVEL_MS = 40.0           # peak window after the onset
NEIGHBOUR_MS = 60.0       # another hit this close makes the onset unmeasurable
RISE_FRACTION = 0.2       # attack start = 20 % of the way floor -> peak
PRE_MS = 10.0             # ... and before it: where the quiet floor is looked for. Dense grooves
                          # (30 ms read p90 19.6 ms off intent, 10 ms 4.9 ms) need it short; widen
                          # both with --search-ms to measure a loop pushed more than 10 ms early
MIN_RISE_RATIO = 3.0      # peak/floor needed for the onset to count as clearly found
FLAG_MS = 10.0            # microtiming is written to the text when |offset| >= this
ACCENT_HI = 0.8
ACCENT_LO = 0.5
GHOST_CHAR, SOFT_CHAR, LOUD_CHAR, EMPTY_CHAR = "o", "x", "X", "."

# max-syncopation search is exhaustive over patterns; beyond this many steps it falls back to
# the "every weakest step sounds" pattern (a lower bound, so norm can then exceed 1).
_EXHAUSTIVE_MAX_STEPS = 20


# --------------------------------------------------------------------------------------
# Metrical weights + Longuet-Higgins/Lee syncopation
# --------------------------------------------------------------------------------------
def _prime_factors(n: int) -> list[int]:
    out, d = [], 2
    while n > 1 and d * d <= n:
        while n % d == 0:
            out.append(d)
            n //= d
        d += 1
    if n > 1:
        out.append(n)
    return out


def metrical_weights(grid_steps: int, beats_per_bar: float = BEATS_PER_BAR) -> list[int]:
    """Metrical weight per step of one bar: 0 at the downbeat, one lower per hierarchy level."""
    n = int(grid_steps)
    if abs(beats_per_bar - round(beats_per_bar)) < 1e-9 and n % round(beats_per_bar) == 0:
        b = round(beats_per_bar)
        factors = _prime_factors(b) + _prime_factors(n // b)
    else:
        factors = _prime_factors(n)
    periods = [n]
    for f in factors:
        periods.append(periods[-1] // f)          # last period is 1 (every step)
    weights = []
    for p in range(n):
        level = next(i for i, per in enumerate(periods) if p % per == 0)
        weights.append(-level)
    return weights


def lhl_bar(onsets, weights: list[int]) -> int:
    """Longuet-Higgins/Lee syncopation of one looped bar. ``onsets``: 0-based step indices."""
    pos = sorted({int(p) for p in onsets})
    n = len(weights)
    total = 0
    for i, p in enumerate(pos):
        nxt = pos[(i + 1) % len(pos)] + (n if i + 1 >= len(pos) else 0)
        gap = [weights[q % n] for q in range(p + 1, nxt)]
        if gap:
            total += max(0, max(gap) - weights[p])
    return total


def max_syncopation(weights: list[int]) -> int:
    """Largest :func:`lhl_bar` any pattern on this grid can score (the normaliser)."""
    import numpy as np

    n = len(weights)
    w = np.asarray(weights)
    if n <= _EXHAUSTIVE_MAX_STEPS:
        masks = np.arange(1, 1 << n, dtype=np.int64)
        bits = ((masks[:, None] >> np.arange(n)) & 1).astype(bool)        # (patterns, n)
        best = np.zeros(len(masks), dtype=np.int64)
        for p in range(n):
            # the nearest onset after p (cyclic) is found by walking forward; vectorised over
            # patterns by tracking the running max weight of the silent gap
            run = np.full(len(masks), -10**6, dtype=np.int64)
            done = np.zeros(len(masks), dtype=bool)
            for d in range(1, n + 1):
                q = (p + d) % n
                active = bits[:, p] & ~done
                sounds = bits[:, q]
                run = np.where(active & ~sounds, np.maximum(run, w[q]), run)
                done |= active & sounds
            val = np.where(bits[:, p] & (run > -10**6), np.maximum(0, run - w[p]), 0)
            best += val
        return int(best.max())
    weakest = min(weights)
    return lhl_bar([p for p in range(n) if weights[p] == weakest], weights)


# --------------------------------------------------------------------------------------
# Measuring hits in audio
# --------------------------------------------------------------------------------------
def _mono(y):
    import numpy as np

    y = np.asarray(y, dtype="float64")
    if y.ndim > 1:
        y = y.mean(axis=0) if y.shape[0] <= y.shape[-1] else y.mean(axis=1)
    return y


def measure_hits(y, sr: int, times: list[float], *, loop: bool,
                 search_ms: float | None = None, level_ms: float = LEVEL_MS,
                 neighbour_ms: float = NEIGHBOUR_MS) -> list[dict]:
    """Actual onset (s), level (dB re full scale) and ``ok`` for each intended hit time.

    ``loop`` tiles the signal once ahead so a hit on sample 0 has the loop's own tail before
    it; otherwise 100 ms of silence is prepended. ``onset`` is None when no rise was found.
    """
    import numpy as np

    mono = _mono(y)
    n = len(mono)
    pad = n if loop else int(0.1 * sr)
    sig = np.concatenate([mono[-pad:] if loop else np.zeros(pad), mono]) if pad else mono
    off = len(sig) - n
    w = max(1, int(0.001 * sr))
    env = np.sqrt(np.convolve(sig * sig, np.ones(w) / w, mode="same"))
    half = max(2, int((search_ms or SEARCH_MS) / 1000 * sr))
    pre = max(2, int((search_ms or PRE_MS) / 1000 * sr))
    lvl = max(2, int(level_ms / 1000 * sr))
    ts = sorted(times)
    out = []
    for t in times:
        c = off + round(t * sr)
        lo = max(0, c - pre)
        hi = min(len(sig), c + half)
        onset, ok = None, False
        if hi - lo > 4 and c - lo > 0:
            s = lo + int(np.argmin(env[lo:c + 1]))
            seg = env[s:hi]
            floor, peak = float(seg[0]), float(seg.max())
            above = np.nonzero(seg > floor + RISE_FRACTION * (peak - floor))[0]
            if len(above) and peak > MIN_RISE_RATIO * max(floor, 1e-9) and peak > 1e-6:
                onset = (s + int(above[0]) - off) / sr
                ok = True
        near = any(1e-6 < abs(o - t) < neighbour_ms / 1000 for o in ts)
        ref = onset if onset is not None else t
        a = max(0, off + round(ref * sr) - int(0.002 * sr))
        peak_abs = float(np.max(np.abs(sig[a:a + lvl]))) if a < len(sig) else 0.0
        out.append({"onset": onset, "ok": bool(ok and not near),
                    "level_db": round(20 * math.log10(max(peak_abs, 1e-9)), 2)})
    return out


# --------------------------------------------------------------------------------------
# Building the hit table
# --------------------------------------------------------------------------------------
def _hits_from_intent(intent: dict, grid_steps: int, bpm: float, beats_per_bar: float) -> list[dict]:
    import numpy as np

    from smpl_analysis import groove

    step_dur = beats_per_bar / grid_steps * 60.0 / bpm
    raw = intent["hits"]
    ts = np.asarray([h["t"] for h in raw], dtype="float64")
    swing = float(intent.get("swing") or 0.0)
    k = groove._assign(ts, 0.0, step_dur, grid_steps, swing) if len(ts) else []
    hits = []
    for i, h in enumerate(raw):
        if h.get("bar") is not None and h.get("step") is not None:
            bar, step = int(h["bar"]), int(h["step"])
        else:
            bar, step = int(k[i]) // grid_steps + 1, int(k[i]) % grid_steps + 1
        grid_t = ((bar - 1) * grid_steps + step - 1) * step_dur
        hits.append({"track": h.get("track") or "all", "bar": bar, "step": step,
                     "t_intent": float(h["t"]), "grid_t": grid_t,
                     "intent_ms": round((float(h["t"]) - grid_t) * 1000, 3),
                     "velocity_intent": float(h.get("velocity", 1.0))})
    return hits


def _hits_from_detection(y, sr, bpm, beats_per_bar, grid_steps) -> tuple[list[dict], float]:
    from smpl_analysis import groove

    res = groove.extract_groove(y, sr, bpm=bpm, beats_per_bar=beats_per_bar, grid_steps=grid_steps)
    bpm = res["groove"]["bpm"]
    step_dur = beats_per_bar / grid_steps * 60.0 / bpm
    bar_dur = step_dur * grid_steps
    origin = res["origin_s"]
    hits = []
    for o in res["onsets"]:
        t = o["sample"] / sr
        step = int(o["step"])
        bar0 = max(0, round((t - origin - (step - 1) * step_dur) / bar_dur))
        grid_t = origin + (bar0 * grid_steps + step - 1) * step_dur
        hits.append({"track": "all", "bar": bar0 + 1, "step": step, "t_measured": t,
                     "grid_t": grid_t, "offset_ms": round((t - grid_t) * 1000, 3)})
    return hits, bpm


def _step_stats(hits: list[dict], grid_steps: int) -> list[dict]:
    """Per grid step: hit count, timing mean/sd (ok hits), mean level and its dB re loudest."""
    import numpy as np

    rows = []
    for s in range(1, grid_steps + 1):
        at = [h for h in hits if h["step"] == s]
        ms = np.asarray([h["offset_ms"] for h in at if h.get("ok") and h.get("offset_ms") is not None])
        lv = [h["level_db"] for h in at if h.get("level_db") is not None]
        rows.append({
            "step": s, "n": len(at), "n_timed": len(ms),
            "mean_ms": round(float(ms.mean()), 2) if len(ms) else None,
            "sd_ms": round(float(ms.std(ddof=1)), 2) if len(ms) > 1 else None,
            "level_db": round(float(np.mean(lv)), 2) if lv else None,
        })
    top = max((r["level_db"] for r in rows if r["level_db"] is not None), default=None)
    for r in rows:
        r["level_rel_db"] = (round(r["level_db"] - top, 2)
                             if r["level_db"] is not None and top is not None else None)
    return rows


def _syncopation(hits: list[dict], weights: list[int], bars: int) -> dict:
    mx = max_syncopation(weights) or 1
    per_bar = []
    for b in range(1, bars + 1):
        steps = [h["step"] - 1 for h in hits if h["bar"] == b]
        per_bar.append(lhl_bar(steps, weights) if steps else 0)
    played = [v for v, b in zip(per_bar, range(1, bars + 1)) if any(h["bar"] == b for h in hits)]
    mean = sum(played) / len(played) if played else 0.0
    return {"per_bar": per_bar, "mean": round(mean, 3), "norm": round(mean / mx, 4), "max": mx}


def analyze(y=None, sr: int | None = None, *, intent: dict | None = None,
            bpm: float | None = None, beats_per_bar: float | None = None,
            grid_steps: int | None = None, bars: int | None = None,
            velocity_source: str = "auto", search_ms: float | None = None,
            level_ms: float = LEVEL_MS, flag_ms: float = FLAG_MS,
            accent_hi: float = ACCENT_HI, accent_lo: float = ACCENT_LO,
            references: list[dict] | None = None) -> dict:
    """The groove-feel document for a clip (``y``/``sr``) and/or an intent.

    With ``y`` None the intent alone is exported (offsets are then the intended ones,
    ``measured`` is False). Returns ``{hits, per_step, tracks, syncopation, accent, ...,
    text, features}``.
    """
    import numpy as np

    has_intent = bool(intent and intent.get("hits"))
    grid_steps = int(grid_steps or (intent or {}).get("grid_steps") or GRID_STEPS)
    beats_per_bar = float(beats_per_bar or (intent or {}).get("beats_per_bar") or BEATS_PER_BAR)
    bpm = bpm or (intent or {}).get("bpm")
    if y is None and not has_intent:
        raise ValueError("need audio or an intent with hits")
    if has_intent and not bpm:
        raise ValueError("intent has hits but no bpm; pass --bpm")

    measured = y is not None
    if has_intent:
        bpm = float(bpm)
        hits = _hits_from_intent(intent, grid_steps, bpm, beats_per_bar)
        source = "intent"
        if measured:
            dur = len(_mono(y)) / sr
            bar_dur = beats_per_bar * 60.0 / bpm
            loop = abs(dur - round(dur / bar_dur) * bar_dur) < bar_dur / grid_steps / 2
            m = measure_hits(y, sr, [h["t_intent"] for h in hits], loop=loop,
                             search_ms=search_ms, level_ms=level_ms)
            for h, r in zip(hits, m):
                h["level_db"] = r["level_db"]
                h["ok"] = r["ok"]
                h["t_measured"] = r["onset"]
                h["offset_ms"] = (round((r["onset"] - h["grid_t"]) * 1000, 3)
                                  if r["onset"] is not None else None)
                h["delta_ms"] = (round(h["offset_ms"] - h["intent_ms"], 3)
                                 if h["offset_ms"] is not None else None)
        else:
            for h in hits:
                h.update(level_db=None, ok=True, offset_ms=h["intent_ms"], t_measured=None,
                         delta_ms=None)
    else:
        hits, bpm = _hits_from_detection(y, sr, bpm, beats_per_bar, grid_steps)
        source = "detected"
        mono = _mono(y)
        m = measure_hits(mono, sr, [h["t_measured"] for h in hits], loop=False,
                         search_ms=search_ms, level_ms=level_ms)
        for h, r in zip(hits, m):
            h["level_db"] = r["level_db"]
            h["ok"] = True            # the onset *is* the detection; no separate attack search
            h["delta_ms"] = None

    bar_dur = beats_per_bar * 60.0 / bpm
    n_bars = max([h["bar"] for h in hits], default=1)
    if measured:
        n_bars = max(n_bars, round(len(_mono(y)) / sr / bar_dur))
    if bars:
        n_bars = max(int(bars), 1)
    elif (intent or {}).get("bars"):
        n_bars = max(n_bars, int(intent["bars"]))

    # velocity as the text will show it
    vs = velocity_source
    if vs == "auto":
        vs = "intent" if source == "intent" else "measured"
    peak = max((10 ** (h["level_db"] / 20) for h in hits if h.get("level_db") is not None), default=None)
    for h in hits:
        if vs == "intent" and "velocity_intent" in h:
            h["vel"] = h["velocity_intent"]
        elif h.get("level_db") is not None and peak:
            h["vel"] = round(10 ** (h["level_db"] / 20) / peak, 4)
        else:
            h["vel"] = h.get("velocity_intent", 1.0)
    hits.sort(key=lambda h: (h["bar"], h["step"], h["track"]))

    weights = metrical_weights(grid_steps, beats_per_bar)
    track_names = []
    for h in hits:
        if h["track"] not in track_names:
            track_names.append(h["track"])
    per_step = _step_stats(hits, grid_steps)
    tracks = {}
    for name in track_names:
        th = [h for h in hits if h["track"] == name]
        tracks[name] = {"hits": len(th), "per_step": _step_stats(th, grid_steps),
                        "syncopation": _syncopation(th, weights, n_bars)}
    sync = _syncopation(hits, weights, n_bars)

    timed = np.asarray([h["offset_ms"] for h in hits if h.get("ok") and h.get("offset_ms") is not None])
    sds = [r["sd_ms"] for r in per_step if r["sd_ms"] is not None]
    lvl_rel = [r["level_rel_db"] for r in per_step if r["level_rel_db"] is not None]
    features = {
        "rhythm.syncopation_lhl": sync["mean"],
        "rhythm.syncopation_norm": sync["norm"],
        # None, never 0.0, when nothing was measured: a silent clip is not a tight groove
        "rhythm.microtiming_abs_ms": round(float(np.mean(np.abs(timed))), 3) if len(timed) else None,
        "rhythm.microtiming_mean_ms": round(float(timed.mean()), 3) if len(timed) else None,
        "rhythm.microtiming_sd_ms": round(float(np.mean(sds)), 3) if sds else None,
        "rhythm.accent_range_db": round(-min(lvl_rel), 2) if lvl_rel else None,
    }

    doc = {
        "v": 1, "source": source, "measured": measured,
        "bpm": round(float(bpm), 3), "beats_per_bar": beats_per_bar, "grid_steps": grid_steps,
        "bars": n_bars, "velocity_source": vs, "flag_ms": flag_ms,
        "timed_hits": len(timed),
        "hits": hits, "per_step": per_step, "tracks": tracks, "syncopation": sync,
        "weights": weights, "features": features,
    }
    if references:
        doc["reference"] = compare_reference(doc, references)
        features["rhythm.groove_ref_delta_ms"] = doc["reference"]["mean_abs_delta_ms"]
    doc["text"] = render_text(doc, accent_hi=accent_hi, accent_lo=accent_lo)
    return doc


# --------------------------------------------------------------------------------------
# Reference grooves
# --------------------------------------------------------------------------------------
def reference_profile(doc: dict, *, grid_steps: int | None = None) -> dict:
    """Reduce a groove document of either kind to ``{step: {"ms", "level_rel_db"}}``.

    Accepts a ``smpl groove --out`` document (``hits[].deviation_ms``, medians per step) or a
    groove-feel document (``per_step[].mean_ms``).
    """
    prof: dict[int, dict] = {}
    if isinstance(doc.get("per_step"), list):
        for r in doc["per_step"]:
            if r.get("mean_ms") is not None or r.get("level_rel_db") is not None:
                prof[int(r["step"])] = {"ms": r.get("mean_ms"), "level_rel_db": r.get("level_rel_db")}
        gs = doc.get("grid_steps")
    elif isinstance(doc.get("hits"), list) and doc["hits"] and "deviation_ms" in doc["hits"][0]:
        for r in doc["hits"]:
            prof[int(r["step"])] = {"ms": r["deviation_ms"], "level_rel_db": None}
        gs = doc.get("grid_steps")
    else:
        raise ValueError("not a groove document (need `hits[].deviation_ms` or `per_step[].mean_ms`)")
    if grid_steps and gs and int(gs) != int(grid_steps):
        raise ValueError(f"reference grid is {gs} steps, this clip's is {grid_steps}")
    return {"grid_steps": gs, "steps": prof}


def load_reference(path: str, *, grid_steps: int, beats_per_bar: float, bpm: float | None) -> dict:
    """A reference profile from a groove JSON or an audio file (analysed with this op)."""
    p = Path(path).expanduser()
    if p.suffix.lower() == ".json":
        prof = reference_profile(json.loads(p.read_text()), grid_steps=grid_steps)
    else:
        import soundfile as sf

        from smpl_analysis import intent as _intent

        y, sr = sf.read(str(p), dtype="float32", always_2d=True)
        side = _intent.sidecar_for(p)
        it = _intent.load(side) if side else None
        d = analyze(y.T, int(sr), intent=it, bpm=bpm or (it or {}).get("bpm"),
                    beats_per_bar=beats_per_bar, grid_steps=grid_steps)
        prof = reference_profile(d, grid_steps=grid_steps)
    prof["path"] = str(p)
    return prof


def compare_reference(doc: dict, references: list[dict]) -> dict:
    """Per-step ``this - reference`` for timing (ms) and accent (dB); references averaged."""
    rows, deltas = [], []
    for r in doc["per_step"]:
        s = r["step"]
        ref_ms = [p["steps"][s]["ms"] for p in references
                  if s in p["steps"] and p["steps"][s]["ms"] is not None]
        ref_lv = [p["steps"][s]["level_rel_db"] for p in references
                  if s in p["steps"] and p["steps"][s].get("level_rel_db") is not None]
        row = {"step": s, "ms": r["mean_ms"],
               "ref_ms": round(sum(ref_ms) / len(ref_ms), 2) if ref_ms else None,
               "level_rel_db": r["level_rel_db"],
               "ref_level_rel_db": round(sum(ref_lv) / len(ref_lv), 2) if ref_lv else None}
        row["delta_ms"] = (round(r["mean_ms"] - row["ref_ms"], 2)
                           if r["mean_ms"] is not None and row["ref_ms"] is not None else None)
        row["delta_level_db"] = (round(r["level_rel_db"] - row["ref_level_rel_db"], 2)
                                 if r["level_rel_db"] is not None and row["ref_level_rel_db"] is not None
                                 else None)
        if row["delta_ms"] is not None:
            deltas.append(abs(row["delta_ms"]))
        rows.append(row)
    return {"files": [p.get("path") for p in references], "per_step": rows,
            "steps_compared": len(deltas),
            "mean_abs_delta_ms": round(sum(deltas) / len(deltas), 3) if deltas else None}


# --------------------------------------------------------------------------------------
# Text export + parser
# --------------------------------------------------------------------------------------
def _bucket(vel: float, hi: float, lo: float) -> str:
    return LOUD_CHAR if vel >= hi else SOFT_CHAR if vel >= lo else GHOST_CHAR


def _safe(name: str) -> str:
    return re.sub(r"\s+", "_", str(name)) or "all"


def render_text(doc: dict, *, accent_hi: float = ACCENT_HI, accent_lo: float = ACCENT_LO) -> str:
    """The compact text Claude reads: step grid per bar per track, flagged ms, hit list.

    ``X`` loud (vel >= accent_hi), ``x`` soft (>= accent_lo), ``o`` ghost (a hit below that),
    ``.`` empty. A space splits the beats inside a bar. ``ms`` lines list the hits whose offset
    from the straight grid is at least ``flag_ms``. Offsets are None-safe: ``-`` means "not
    measurable" (another hit within 60 ms, or no clear attack).
    """
    n, bars = doc["grid_steps"], doc["bars"]
    bpb = doc["beats_per_bar"]
    per_beat = n // int(bpb) if abs(bpb - round(bpb)) < 1e-9 and n % round(bpb) == 0 else 0
    flag = doc["flag_ms"]
    names = list(doc["tracks"])
    width = max(len(_safe(x)) for x in names) if names else 3

    lines = [
        (f"# groove-feel v1 bpm={doc['bpm']:g} meter={bpb:g}beats grid={n} bars={bars} "
        f"source={doc['source']} measured={'yes' if doc['measured'] else 'no'} vel={doc['velocity_source']} "
         f"timed={doc['timed_hits']}/{len(doc['hits'])}"),
        (f"# grid: {LOUD_CHAR} vel>={accent_hi:g}  {SOFT_CHAR} vel>={accent_lo:g}  {GHOST_CHAR} ghost  "
        f"{EMPTY_CHAR} empty | steps are 1..{n}, beats split by a space | ms = signed offset from the "
        f"straight grid, + late - early, listed when |ms|>={flag:g}"),
    ]
    sync = doc["syncopation"]
    lines.append(f"# syncopation (Longuet-Higgins/Lee) per bar: {' '.join(str(v) for v in sync['per_bar'])}"
                 f" | mean {sync['mean']:g} of max {sync['max']} = {sync['norm']:.2f}")
    ps = doc["per_step"]
    def cell(v, f):
        return "-" if v is None else f.format(v)

    lines.append("# step      " + " ".join(f"{r['step']:>5d}" for r in ps))
    lines.append("# mean ms   " + " ".join(f"{cell(r['mean_ms'], '{:+.0f}'):>5}" for r in ps))
    lines.append("# sd ms     " + " ".join(f"{cell(r['sd_ms'], '{:.0f}'):>5}" for r in ps))
    if any(r["level_rel_db"] is not None for r in ps):
        lines.append("# level dB  " + " ".join(f"{cell(r['level_rel_db'], '{:.0f}'):>5}" for r in ps))
    lines.append("")

    for name in names:
        tag = _safe(name)
        th = [h for h in doc["hits"] if h["track"] == name]
        for b in range(1, bars + 1):
            row = [EMPTY_CHAR] * n
            for h in th:
                if h["bar"] == b:
                    ch = _bucket(h["vel"], accent_hi, accent_lo)
                    cur = row[h["step"] - 1]
                    if cur == EMPTY_CHAR or "oxX".index(ch) > "oxX".index(cur):
                        row[h["step"] - 1] = ch
            body = ("".join(row) if not per_beat else
                    " ".join("".join(row[i:i + per_beat]) for i in range(0, n, per_beat)))
            lines.append(f"{tag:<{width}} b{b:<2d} |{body}|")
            flagged = [h for h in th if h["bar"] == b and h.get("offset_ms") is not None
                       and h.get("ok") and abs(h["offset_ms"]) >= flag]
            if flagged:
                lines.append(f"{'':<{width}}     ms " + " ".join(
                    f"{h['step']}:{h['offset_ms']:+.0f}" for h in flagged))
    lines.append("")
    lines.append("# hits: bar step track ms vel level_db   (ms '-' = not measurable)")
    for h in doc["hits"]:
        ms = "-" if h.get("offset_ms") is None or not h.get("ok") else f"{h['offset_ms']:+.1f}"
        lv = "-" if h.get("level_db") is None else f"{h['level_db']:.1f}"
        lines.append(f"{h['bar']} {h['step']} {_safe(h['track'])} {ms} {h['vel']:.2f} {lv}")
    return "\n".join(lines) + "\n"


_GRID_LINE = re.compile(r"^(?P<track>\S+)\s+b(?P<bar>\d+)\s+\|(?P<body>[^|]*)\|\s*$")


def parse_text(text: str) -> dict:
    """Parse :func:`render_text` output back.

    Returns ``{"meta": {...header fields}, "grid": {track: {bar: {step: char}}}, "hits": [...]}``.
    Only the grid lines and the hit list are read; the ``#`` summary lines are for people.
    """
    meta: dict = {}
    grid: dict = {}
    hits: list[dict] = []
    in_hits = False
    for line in text.splitlines():
        if line.startswith("# groove-feel"):
            meta = dict(re.findall(r"(\w+)=(\S+)", line))
            continue
        if line.startswith("# hits:"):
            in_hits = True
            continue
        if line.startswith("#") or not line.strip():
            continue
        if in_hits:
            bar, step, track, ms, vel, lv = line.split()
            hits.append({"bar": int(bar), "step": int(step), "track": track,
                         "ms": None if ms == "-" else float(ms), "vel": float(vel),
                         "level_db": None if lv == "-" else float(lv)})
            continue
        m = _GRID_LINE.match(line)
        if m:
            body = m.group("body").replace(" ", "")
            cells = {i + 1: c for i, c in enumerate(body) if c != EMPTY_CHAR}
            grid.setdefault(m.group("track"), {})[int(m.group("bar"))] = cells
    return {"meta": meta, "grid": grid, "hits": hits}


def steps_of(hits: list[dict]) -> dict[str, dict[int, set[int]]]:
    """``{track: {bar: {step}}}`` of a hit list (for comparing recipes with parsed text)."""
    out: dict = {}
    for h in hits:
        out.setdefault(_safe(h["track"]), {}).setdefault(h["bar"], set()).add(h["step"])
    return out


# --------------------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------------------
def groove_feel_audio_frame(
    audio_frame: dict, *, intent_path: str | None = None, ref_paths: list[str] | None = None,
    bpm: float | None = None, beats_per_bar: float | None = None, grid_steps: int | None = None,
    **kw,
) -> tuple[list[dict], dict]:
    """Resolve an `audio` frame's PCM, find its intent, and emit the groove-feel frames.

    Returns ``(frames, doc)``: a `feature` frame (role ``groove-feel``, ``data`` = the full
    document incl. ``text``) and a `feature` frame (role ``groove-feel-features``) with the
    registered scalar keys. The caller passes the input frame through.
    """
    import soundfile as sf
    from smplstream import cas
    from smplstream import frames as F

    from smpl_analysis import intent as _intent

    src = cas.get_path(audio_frame["hash"])
    y, sr = sf.read(str(src), dtype="float32", always_2d=True)
    sr = int(sr)
    it = _intent.for_audio_frame(audio_frame, intent_path)

    gs = int(grid_steps or (it or {}).get("grid_steps") or GRID_STEPS)
    bpb = float(beats_per_bar or (it or {}).get("beats_per_bar") or BEATS_PER_BAR)
    refs = [load_reference(p, grid_steps=gs, beats_per_bar=bpb, bpm=bpm or (it or {}).get("bpm"))
            for p in (ref_paths or [])]
    doc = analyze(y.T, sr, intent=it, bpm=bpm, beats_per_bar=bpb, grid_steps=gs,
                  references=refs or None, **kw)

    params = {"grid_steps": doc["grid_steps"], "beats_per_bar": doc["beats_per_bar"],
              "bpm": doc["bpm"], "source": doc["source"], "bars": doc["bars"],
              "flag_ms": doc["flag_ms"], "velocity_source": doc["velocity_source"],
              "intent": (it or {}).get("source"), "refs": len(refs), "sr_hz": sr}
    of = audio_frame.get("id")
    common = {"of": of, "op": OP, "op_version": OP_VERSION,
              "lineage": [of] if of else None, "params": params}
    data = {k: v for k, v in doc.items() if k != "features"}
    return [
        F.feature_frame(data, role="groove-feel", **common),
        F.feature_frame(doc["features"], role="groove-feel-features", **common),
    ], doc
