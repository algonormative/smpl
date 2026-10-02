"""Groove extraction — swing + per-step micro-timing lifted off a reference loop
(playbook §6; ticket vault-6m62).

The musical why: a step sequencer places hits on a perfectly even grid, and an even grid
is what makes a programmed loop read as *programmed*. The feel of a reference — the
shuffle of its offbeats, the hat that always rushes, the snare that always drags — lives
entirely in how far each hit sits off that grid. This op measures those deviations on a
reference loop and re-expresses them in the exact vocabulary ``smpl pattern`` consumes, so
a generated loop can borrow a groove instead of inventing one.

The model is `pattern`'s, not a new one
---------------------------------------
``smpl pattern`` places a hit on 1-indexed grid step ``s`` at ``(s-1) × stepGap`` beats
(``stepGap = beats_per_bar / grid_steps``), then applies exactly two timing modifiers:

- **swing** (0..1) delays every *even* step (2, 4, 6, …) by ``swing × stepGap`` beats;
- **nudge** (± beats) offsets one hit by an explicit amount.

So a groove decomposes into one global ``swing`` plus a per-step ``nudge`` residual — and
that is precisely what this op reports. Feed the emitted ``swing`` back as the pattern's
global swing and merge each step's ``nudge`` into that step's hit, and the generated loop
lands where the reference's hits landed.

How it is measured
------------------
1. **Onsets.** ``librosa.onset.onset_detect`` at a *small* hop (128 samples ≈ 2.9 ms at
   44.1 kHz, vs. the 512 the beat grid uses) — micro-timing is the measurement, so hop
   resolution is the error floor. The signal is padded with :data:`LEAD_IN_SEC` of silence
   first: librosa's peak picker needs frames *before* a transient, and without the lead-in
   a hit sitting exactly at t=0 (the downbeat of a trimmed loop — the most important hit
   in the file) is silently missed.
2. **Grid fit.** ``swing`` is found by a **global search** over the 0..0.95 range rather
   than by refining from zero. Refinement cannot work here: at a candidate swing of 0 a
   heavily shuffled offbeat is nearer the *following* odd step than its own, so it is
   assigned there, reads as on-grid, and the estimate is pinned at 0 forever. The search
   scores each candidate by total residual after aligning the grid origin, so a swung
   reference is recognised as swung at any depth (verified to 0.6, the depth `pattern`
   recommends as a ceiling).
3. **Origin alignment.** A loop is not guaranteed to start exactly on its grid, and every
   onset detector carries a small constant bias. Both are one and the same unknown — a
   phase offset — so the origin is aligned to the **odd-step (un-swung) hits**, whose
   position the swing model does not touch. Odd steps are the anchors; even steps are the
   measurement. When a reference has fewer than two odd-step hits there are no anchors, so
   alignment falls back to all onsets and the swing estimate is reported at zero
   confidence.
4. **Residuals.** What the swing model does not explain becomes each step's ``nudge``,
   in beats, aggregated per grid step by median across bars.

Emitted frames (one set per selected audio frame):

===========================  =========================================================
frame                        contents
===========================  =========================================================
``marker`` role ``groove``   every onset used, ``t`` + ``sample``; label ``step-<s>``
                             (the 1-indexed grid step it was assigned to)
``feature`` role ``groove``  ``data`` IS the pattern-consumable groove: ``bpm``,
                             ``beats_per_bar``, ``grid_steps``, ``swing``, and
                             ``hits[]`` of ``{step, nudge, deviation, deviation_ms,
                             count}``
``feature`` role             ``rhythm.swing``, ``rhythm.swing_confidence``,
``groove-features``          ``rhythm.microtiming_beats``
===========================  =========================================================

The groove frame follows `pattern`'s own precedent (a `feature` frame whose ``data`` is a
ready-to-use document); the registered ``rhythm.*`` scalars live in their own frame so a
stats/verdict consumer sees the same flat key→value shape every other feature frame has.

Pure functions returning frame dicts. Heavy imports (librosa/numpy) live inside the
functions so cold pipe stages start fast.
"""

from __future__ import annotations

OP = "groove"
OP_VERSION = "groove@1"

# Analysis defaults (declared so memo params are complete & stable per op_version).
HOP_LENGTH = 128
GRID_STEPS = 16
BEATS_PER_BAR = 4.0
START_BPM = 120.0

# Silence prepended before onset detection so a transient at t=0 has frames to peak
# against. 100 ms is ~34 hops at the default hop — comfortably more than the peak
# picker's pre-max window, and short enough to cost nothing.
LEAD_IN_SEC = 0.1

# Swing search: `pattern` clamps swing to 0.95, so the search covers the same range.
# 0.005 is a fifth of a millisecond-scale step at typical tempi — finer than the onset
# resolution — and the winning candidate is refined to a continuous value afterwards.
SWING_MAX = 0.95
SWING_SEARCH_STEP = 0.005


def onset_times(y, sr: int, *, hop_length: int = HOP_LENGTH) -> "list[float]":
    """Onset times in seconds, measured with a lead-in so a hit at t=0 survives.

    ``backtrack`` is deliberately off: backtracking snaps each onset to the preceding
    energy minimum, which is a *quantization* of the very deviations this op exists to
    measure.
    """
    import librosa
    import numpy as np

    y = np.asarray(y, dtype="float32")
    if y.ndim > 1:  # collapse to mono
        y = np.mean(y, axis=0)
    pad = int(round(LEAD_IN_SEC * sr))
    padded = np.concatenate([np.zeros(pad, dtype="float32"), y])
    times = librosa.onset.onset_detect(
        y=padded, sr=sr, hop_length=hop_length, units="time", backtrack=False
    )
    times = np.asarray(times, dtype="float64").ravel() - pad / sr
    return [float(t) for t in np.maximum(times, 0.0)]


def _assign(ts, origin: float, step_dur: float, grid_steps: int, swing: float):
    """Nearest grid step index per onset under a *swung* grid.

    Returns 0-based global step indices ``k`` (the step counted from the grid origin
    across all bars). The nearest step is not simply ``round((t - origin) / step_dur)``
    once swing is in play — a swung even step sits up to 0.95 of a step late — so the
    three candidates either side of that estimate are scored explicitly.
    """
    import numpy as np

    swing_sec = swing * step_dur
    base = np.round((ts - origin) / step_dur).astype(int)
    best_k = None
    best_err = None
    for delta in (-1, 0, 1):
        k = np.maximum(base + delta, 0)
        step = (k % grid_steps) + 1
        expected = origin + k * step_dur + np.where(step % 2 == 0, swing_sec, 0.0)
        err = np.abs(ts - expected)
        if best_err is None:
            best_k, best_err = k, err
        else:
            better = err < best_err
            best_k = np.where(better, k, best_k)
            best_err = np.where(better, err, best_err)
    return best_k


def _fit(ts, step_dur: float, grid_steps: int, swing: float, *, iters: int = 3):
    """Align the grid origin at a fixed ``swing``; return ``(origin, k, step, residual)``.

    The origin is the median residual of the **odd-step** onsets — the un-swung anchors —
    so the alignment cannot absorb the swing it is meant to leave visible. With fewer than
    two anchors it falls back to all onsets (see the module docstring's step 3).
    """
    import numpy as np

    origin = 0.0
    k = step = residual = None
    for _ in range(iters + 1):
        k = _assign(ts, origin, step_dur, grid_steps, swing)
        step = (k % grid_steps) + 1
        residual = ts - (origin + k * step_dur + np.where(step % 2 == 0, swing * step_dur, 0.0))
        anchors = step % 2 == 1
        reference = residual[anchors] if int(anchors.sum()) >= 2 else residual
        if reference.size:
            origin += float(np.median(reference))
    return origin, k, step, residual


def _search_swing(ts, step_dur: float, grid_steps: int):
    """Global search for the swing that best explains the onset times.

    Scored by total absolute residual (L1) — a wrong swing pushes every offbeat away from
    the step it is assigned to, so the true depth is a clear minimum. Ties keep the lower
    swing, which is the conservative reading (less groove asserted than the evidence
    supports). See the module docstring for why this is a search and not a refinement.
    """
    import numpy as np

    best = None
    swing = 0.0
    while swing <= SWING_MAX + 1e-9:
        origin, k, step, residual = _fit(ts, step_dur, grid_steps, swing)
        cost = float(np.sum(np.abs(residual)))
        if best is None or cost < best[0] - 1e-12:
            best = (cost, swing, origin, k, step, residual)
        swing += SWING_SEARCH_STEP
    return best


def extract_groove(
    y,
    sr: int,
    *,
    bpm: float | None = None,
    beats_per_bar: float = BEATS_PER_BAR,
    grid_steps: int = GRID_STEPS,
    hop_length: int = HOP_LENGTH,
) -> dict:
    """Extract the swing + per-step micro-timing of a reference loop.

    Returns ``{groove, features, onsets, origin_s, bpm_source}`` where ``groove`` is the
    ``smpl pattern``-consumable document (``bpm``/``beats_per_bar``/``grid_steps``/
    ``swing``/``hits``), ``features`` maps the registered ``rhythm.*`` keys to their
    values, and ``onsets`` is one ``{sample, step}`` record per detected onset.

    ``bpm`` is the tempo the grid is measured against; when omitted it is estimated with
    the same ``librosa.beat.beat_track`` call the beat grid uses. **Pass it when known** —
    a one- or two-bar loop is thin evidence for a tempo estimator, and every deviation
    reported here is relative to the grid that tempo defines.
    """
    import numpy as np

    if grid_steps < 1 or beats_per_bar <= 0:
        raise ValueError("grid_steps must be >= 1 and beats_per_bar > 0")

    y = np.asarray(y, dtype="float32")
    if y.ndim > 1:
        y = np.mean(y, axis=0)

    bpm_source = "given"
    if bpm is None or bpm <= 0:
        import librosa

        tempo, _ = librosa.beat.beat_track(
            y=y, sr=sr, hop_length=512, start_bpm=START_BPM, trim=False, units="frames"
        )
        bpm = float(np.atleast_1d(tempo)[0])
        bpm_source = "estimated"
    bpm = float(bpm)
    if bpm <= 0:
        raise ValueError("could not determine a positive bpm; pass one explicitly")

    step_gap = beats_per_bar / grid_steps          # step spacing in beats
    step_dur = step_gap * 60.0 / bpm               # step spacing in seconds

    times = onset_times(y, sr, hop_length=hop_length)
    if not times:
        return {
            "groove": {
                "bpm": round(bpm, 3), "beats_per_bar": beats_per_bar,
                "grid_steps": grid_steps, "swing": 0.0, "hits": [],
            },
            "features": {
                "rhythm.swing": 0.0,
                "rhythm.swing_confidence": 0.0,
                "rhythm.microtiming_beats": 0.0,
            },
            "onsets": [],
            "origin_s": 0.0,
            "bpm_source": bpm_source,
        }

    ts = np.asarray(times, dtype="float64")
    _, swing, origin, k, step, residual = _search_swing(ts, step_dur, grid_steps)

    # Refine the searched swing to a continuous value: the search fixes the *assignment*
    # of onsets to steps, and once that is fixed the depth itself is just the median
    # offbeat deviation. Reporting the grid candidate instead would quantize every groove
    # to SWING_SEARCH_STEP for no reason.
    deviation = ts - (origin + k * step_dur)       # total offset from the un-swung grid
    swung = step % 2 == 0
    if int(swung.sum()) >= 1:
        swing = float(np.median(deviation[swung])) / step_dur
        swing = max(0.0, min(swing, SWING_MAX))
    else:
        swing = 0.0
    residual = deviation - np.where(swung, swing * step_dur, 0.0)

    # Confidence is the *consistency* of the offbeat deviations: a groove all the offbeats
    # agree on is one number that explains them; scattered offbeats are micro-timing, not
    # swing. Half a step is the scale at which an offbeat stops belonging to its step, so
    # it is the natural normalizer. No offbeats at all ⇒ no evidence ⇒ zero.
    if int(swung.sum()) >= 2:
        spread = float(np.median(np.abs(deviation[swung] - np.median(deviation[swung]))))
        confidence = max(0.0, min(1.0, 1.0 - spread / (0.5 * step_dur)))
    else:
        confidence = 0.0
    if int((step % 2 == 1).sum()) < 2:
        confidence = 0.0  # no un-swung anchors: the grid phase itself is a guess

    nudge_beats = residual * bpm / 60.0
    deviation_beats = deviation * bpm / 60.0

    hits = []
    for s in range(1, grid_steps + 1):
        mask = step == s
        count = int(mask.sum())
        if not count:
            continue
        nudge = float(np.median(nudge_beats[mask]))
        dev = float(np.median(deviation_beats[mask]))
        hits.append({
            "step": s,
            "nudge": round(nudge, 6),
            "deviation": round(dev, 6),
            "deviation_ms": round(dev * 60000.0 / bpm, 3),
            "count": count,
        })

    return {
        "groove": {
            "bpm": round(bpm, 3),
            "beats_per_bar": beats_per_bar,
            "grid_steps": grid_steps,
            "swing": round(swing, 6),
            "hits": hits,
        },
        "features": {
            "rhythm.swing": round(swing, 6),
            "rhythm.swing_confidence": round(confidence, 6),
            "rhythm.microtiming_beats": round(float(np.mean(np.abs(nudge_beats))), 6),
        },
        "onsets": [
            {"sample": int(round(t * sr)), "step": int(s)} for t, s in zip(ts, step)
        ],
        "origin_s": round(float(origin), 6),
        "bpm_source": bpm_source,
    }


def groove_audio_frame(
    audio_frame: dict,
    *,
    bpm: float | None = None,
    beats_per_bar: float = BEATS_PER_BAR,
    grid_steps: int = GRID_STEPS,
    hop_length: int = HOP_LENGTH,
) -> list[dict]:
    """Resolve an `audio` frame's PCM from the CAS and emit its groove frames.

    Returns the `marker` frame (role ``groove``), the pattern-consumable `feature` frame
    (role ``groove``) and the registered-key `feature` frame (role ``groove-features``),
    each carrying ``of``/``op``/``op_version``/``params`` lineage per the tool contract.
    The caller passes the input frame through.
    """
    import soundfile as sf

    from smplstream import cas, frames as F

    src = cas.get_path(audio_frame["hash"])
    y, sr = sf.read(str(src), dtype="float32", always_2d=True)
    sr = int(sr)
    y = y.T  # (ch, n) for the mono collapse in extract_groove

    result = extract_groove(
        y, sr, bpm=bpm, beats_per_bar=beats_per_bar,
        grid_steps=grid_steps, hop_length=hop_length,
    )

    params = {
        "grid_steps": grid_steps,
        "beats_per_bar": beats_per_bar,
        "hop_length": hop_length,
        "bpm": result["groove"]["bpm"],
        "bpm_source": result["bpm_source"],
        "origin_s": result["origin_s"],
        "onsets": len(result["onsets"]),
        "sr_hz": sr,
    }
    of = audio_frame.get("id")
    lineage = [audio_frame["id"]] if audio_frame.get("id") else None
    common = {"of": of, "op": OP, "op_version": OP_VERSION,
              "lineage": lineage, "params": params}

    points = [
        {"t": round(o["sample"] / sr, 6), "sample": o["sample"], "label": f"step-{o['step']}"}
        for o in result["onsets"]
    ]
    return [
        F.marker_frame(points, role="groove", **common),
        F.feature_frame(result["groove"], role="groove", **common),
        F.feature_frame(result["features"], role="groove-features", **common),
    ]
