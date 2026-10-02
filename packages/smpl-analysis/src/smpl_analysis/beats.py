"""Downbeat-aware beat grid — beats, downbeats, tempo changes + rhythm features
(research §3, "Build #1"; ticket vault-32n3).

This is the sample-accurate marker source the rest of the MIR tier snaps to (chords
per bar, loop candidates, beat-quantized slices), so every emitted marker point carries
BOTH float-second ``t`` and an integer ``sample`` index at the audio frame's native
``meta.sr`` (spec → *Units & timebase*: float seconds alone can't round-trip to
sample-indexed cue points).

Engine choice — **librosa**, madmom rejected (verified 2026-08-29)
-----------------------------------------------------------------
research §3 offers madmom's DBN downbeat tracker or Beat This! (ISMIR'24), with the
caveat "verify madmom installs cleanly in our runtime (historically lags NumPy)".
It does not:

    $ uv run --with madmom python -c "import madmom"
    ...
    ModuleNotFoundError: No module named 'Cython'
    hint: madmom@0.16.1 depends on `Cython` but doesn't declare it as a build
          dependency

madmom 0.16.1 (2018, sdist-only, no wheels) cannot even build here — it needs Cython
at build time without declaring it, and retrying with Cython present and build
isolation disabled fails the same way. Beyond the build, its runtime uses the
``np.float`` aliases removed in NumPy 1.24 (this stack is NumPy 2.4 / Python 3.11).
**madmom install health: UNHEALTHY — not adopted.**

Beat This! is a torch model: too heavy for the light core tier (multi-hundred-MB
dependency + weights for an op that must start fast in a cold pipe stage).

So the grid runs on **librosa**, already a workspace dependency:

- beats + primary tempo: ``librosa.beat.beat_track`` over an onset-strength envelope.
- ``rhythm.bpm_candidates``: tempogram peaks folded octave-robustly (see
  ``_fold_octave``) so 60/120/240 reinforce ONE hypothesis instead of competing —
  that folding is what makes the top-2 octave-error robust.
- ``rhythm.bpm_confidence``: the winning canonical tempo's share of total peak
  strength, clamped to 0–1.
- downbeats: a deterministic bar-phase estimator. Accent energy is the RMS of a window
  **centred** on each beat (half-width a quarter of the median inter-beat interval —
  centred, because a beat lands within a hop of its transient and a window that starts
  at the beat misses accents that sit just before it). For each meter candidate (4,
  then 3) and each phase within it, score the accent *contrast* — mean accent at the
  candidate downbeats minus mean accent at the other beats — and keep the best.
  Contrast (not raw accent) keeps the comparison fair across meters; ties resolve in
  the declared meter/phase order, so the op is deterministic.
- tempo changes: a windowed tempo track (``librosa.feature.tempo(aggregate=None)``,
  median-smoothed) segmented on sustained relative deviation, each change snapped to
  the nearest beat. **Near-constant-tempo material emits NO tempo-change markers** —
  an empty marker frame, not a spurious point.

Emitted frames (one set per selected audio frame):

===========================  =========================================================
frame                        contents
===========================  =========================================================
``marker`` role ``beat``     every beat; label ``beat-<i>``
``marker`` role ``downbeat`` bar starts (a strict subset of ``beat``); ``downbeat-<b>``
``marker`` role             tempo-change points snapped to beats;
``tempo-change``             label ``tempo-change-<i>`` (empty for steady tempo)
``feature`` role ``beats``   ``rhythm.bpm``, ``rhythm.bpm_confidence``,
                             ``rhythm.bpm_candidates``, ``rhythm.time_signature``
===========================  =========================================================

Three roles rather than one labelled track so a consumer can ``smpl select --role
downbeat`` without filtering point labels, matching how ``slice`` exposes ``onset``.

Pure functions returning frame dicts. Heavy imports (librosa/numpy) live inside the
functions so cold pipe stages start fast.
"""

from __future__ import annotations

OP = "beats"
OP_VERSION = "beats@1"

# Analysis defaults (declared so memo params are complete & stable per op_version).
HOP_LENGTH = 512
START_BPM = 120.0

# Meter candidates for the bar-phase estimator, in preference order (ties keep the first).
METERS = (4, 3)

# Accent window half-width, as a fraction of the median inter-beat interval. Centred on
# the beat so a transient sitting up to a hop early still counts toward that beat.
ACCENT_HALF_WIDTH = 0.25

# Octave folding range for tempo candidates: every peak is halved/doubled into
# [FOLD_MIN, FOLD_MIN * 2) so octave-related peaks aggregate onto one hypothesis.
FOLD_MIN = 70.0

# Two canonical tempi within this relative distance are the same hypothesis measured
# twice (adjacent tempogram bins), not two candidates — merge them.
CANDIDATE_MERGE_TOL = 0.04

# Tempo-change detection: relative deviation from the running segment tempo that must
# persist for TEMPO_MIN_HOLD_SEC before a change point is declared. The thresholds are
# deliberately loose — a windowed tempogram wanders ±10% on non-metrical material
# (verified on white noise), so anything tighter invents changes in audio that has no
# tempo at all. The cost is that modulations under ~12% read as one steady segment.
TEMPO_TOLERANCE = 0.12
TEMPO_MIN_HOLD_SEC = 3.0
TEMPO_SMOOTH_FRAMES = 9
# Frames of the current segment used for its running (median) reference tempo.
TEMPO_REF_FRAMES = 200


# ---------------------------------------------------------------------------
# Tempo helpers.
# ---------------------------------------------------------------------------
def _fold_octave(bpm: float, fold_min: float = FOLD_MIN) -> float:
    """Fold a tempo into ``[fold_min, 2 * fold_min)`` by halving/doubling.

    This is the octave-error-robust step: 60, 120 and 240 BPM all fold to the same
    canonical tempo, so tempogram peaks an octave apart reinforce a single hypothesis
    instead of splitting the vote between two.
    """
    if not (bpm > 0):
        return 0.0
    while bpm < fold_min:
        bpm *= 2.0
    while bpm >= fold_min * 2.0:
        bpm /= 2.0
    return bpm


def _unfold_near(canonical: float, reference: float) -> float:
    """Octave-shift ``canonical`` to the multiple/divisor nearest ``reference``.

    Candidates are reported in the register the primary tempo actually lives in, so a
    128 BPM track reads ``[128.0, 85.3]`` rather than the folded ``[85.3, 85.3]``.
    """
    if not (canonical > 0 and reference > 0):
        return round(canonical, 3)
    best = canonical
    best_err = abs(canonical - reference)
    value = canonical
    for _ in range(4):  # up to 4 octaves either way covers any musical tempo
        value *= 2.0
        err = abs(value - reference)
        if err < best_err:
            best, best_err = value, err
    value = canonical
    for _ in range(4):
        value /= 2.0
        err = abs(value - reference)
        if err < best_err:
            best, best_err = value, err
    return round(best, 3)


def tempo_candidates(onset_env, sr: int, *, hop_length: int = HOP_LENGTH) -> tuple[list[float], float]:
    """Top-2 octave-robust tempo hypotheses and the winner's confidence.

    Returns ``(candidates, confidence)`` where ``candidates`` always has length 2 (the
    primary first) and ``confidence`` is the winning canonical tempo's share of the
    total tempogram peak strength, in 0–1.
    """
    import librosa
    import numpy as np
    from scipy.signal import find_peaks

    tg = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    strength = np.mean(tg, axis=1)
    freqs = librosa.tempo_frequencies(len(strength), sr=sr, hop_length=hop_length)

    finite = np.isfinite(freqs) & (freqs > 0) & np.isfinite(strength)
    freqs, strength = freqs[finite], np.maximum(strength[finite], 0.0)
    if freqs.size == 0:
        return [0.0, 0.0], 0.0

    # tempo_frequencies is descending; ascending order makes peak-picking read naturally.
    order = np.argsort(freqs)
    freqs, strength = freqs[order], strength[order]

    peak_idx, _ = find_peaks(strength)
    if peak_idx.size == 0:
        peak_idx = np.array([int(np.argmax(strength))])

    # Aggregate peak strength onto octave-folded canonical tempi (rounded to 0.5 BPM
    # so neighbouring bins of one peak collapse together).
    buckets: dict[float, float] = {}
    for i in peak_idx:
        canonical = round(_fold_octave(float(freqs[i])) * 2.0) / 2.0
        if canonical <= 0:
            continue
        buckets[canonical] = buckets.get(canonical, 0.0) + float(strength[i])
    if not buckets:
        return [0.0, 0.0], 0.0

    # Sort by strength desc, then tempo asc — deterministic under ties.
    ranked = sorted(buckets.items(), key=lambda kv: (-kv[1], kv[0]))

    # Merge near-identical canonical tempi (adjacent bins of one peak) into the stronger
    # one, so the two reported candidates are genuinely competing hypotheses.
    merged: list[list[float]] = []
    for tempo, strength in ranked:
        for entry in merged:
            if abs(tempo - entry[0]) / entry[0] <= CANDIDATE_MERGE_TOL:
                entry[1] += strength
                break
        else:
            merged.append([tempo, strength])

    total = sum(s for _, s in merged)
    confidence = float(merged[0][1] / total) if total > 0 else 0.0
    candidates = [t for t, _ in merged[:2]]
    if len(candidates) < 2:
        # No second hypothesis in the tempogram: report the metrical alternative, which
        # is the ambiguity a listener would actually have (half-time vs double-time).
        primary = candidates[0]
        candidates.append(primary / 2.0 if primary >= 100.0 else primary * 2.0)
    return candidates, max(0.0, min(1.0, confidence))


# ---------------------------------------------------------------------------
# Bar phase (downbeats).
# ---------------------------------------------------------------------------
def beat_accents(y, beat_samples, sr: int) -> "list[float]":
    """Accent energy per beat: RMS over a window centred on each beat sample.

    Centred rather than forward-looking — a tracked beat sits within a hop of its
    transient in either direction, and a forward-only window silently reads ~0 for every
    accent that lands just before its beat.
    """
    import numpy as np

    y = np.abs(np.asarray(y, dtype="float64").ravel())
    samples = [int(s) for s in beat_samples]
    if len(samples) < 2:
        return [0.0] * len(samples)
    ibi = float(np.median(np.diff(samples)))
    half = max(1, int(ACCENT_HALF_WIDTH * ibi))
    out = []
    for s in samples:
        seg = y[max(0, s - half) : s + half]
        out.append(float(np.sqrt(np.mean(seg ** 2))) if seg.size else 0.0)
    return out


def bar_phase(accent, *, meters: tuple[int, ...] = METERS) -> tuple[int, int]:
    """Pick ``(meter, phase)`` maximizing downbeat accent contrast over a beat accent array.

    ``accent[i]`` is the accent energy at beat ``i``. The score for a (meter, phase) pair
    is ``mean(accent at downbeats) - mean(accent elsewhere)``; the mean-difference form
    keeps meters with different downbeat counts comparable. Ties keep the earlier meter
    in ``meters`` and the lower phase, so the result is deterministic.
    """
    import numpy as np

    accent = np.asarray(accent, dtype="float64").ravel()
    best = (meters[0], 0)
    best_score = -np.inf
    for meter in meters:
        if accent.size < meter:
            continue
        for phase in range(meter):
            mask = np.zeros(accent.size, dtype=bool)
            mask[phase::meter] = True
            if not mask.any() or mask.all():
                continue
            score = float(np.mean(accent[mask]) - np.mean(accent[~mask]))
            if score > best_score:
                best_score, best = score, (meter, phase)
    return best


# ---------------------------------------------------------------------------
# Tempo-change points.
# ---------------------------------------------------------------------------
def tempo_change_times(
    onset_env,
    sr: int,
    *,
    hop_length: int = HOP_LENGTH,
    tolerance: float = TEMPO_TOLERANCE,
    min_hold_sec: float = TEMPO_MIN_HOLD_SEC,
) -> list[float]:
    """Times (seconds) where the windowed tempo track leaves its running segment tempo.

    The track is median-smoothed and **octave-folded** before segmentation: an unfolded
    windowed tempogram flips between a tempo and its double at a section boundary, which
    would otherwise read as a burst of changes rather than one. The running reference is
    the median of the current segment's opening ``TEMPO_REF_FRAMES`` (not its first frame,
    which one outlying window would otherwise define — and not a sliding window, which
    creeps onto the new tempo and dissolves the very jump it is meant to detect).

    Steady material returns ``[]`` — no spurious markers on constant-tempo audio.
    """
    import librosa
    import numpy as np
    from scipy.ndimage import median_filter

    track = librosa.feature.tempo(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length, aggregate=None
    )
    track = np.asarray(track, dtype="float64").ravel()
    min_hold = max(2, int(round(min_hold_sec * sr / hop_length)))
    if track.size < min_hold * 2:
        return []
    track = median_filter(track, size=TEMPO_SMOOTH_FRAMES, mode="nearest")
    track = np.array([_fold_octave(float(v)) for v in track])

    times: list[float] = []
    seg_start = 0
    run_start = -1
    for i in range(1, track.size):
        window = track[seg_start : min(i, seg_start + TEMPO_REF_FRAMES)]
        reference = float(np.median(window)) if window.size else float(track[i])
        deviates = reference > 0 and abs(track[i] - reference) / reference > tolerance
        if not deviates:
            run_start = -1
            continue
        if run_start < 0:
            run_start = i
        elif i - run_start + 1 >= min_hold:
            times.append(float(librosa.frames_to_time(run_start, sr=sr, hop_length=hop_length)))
            seg_start, run_start = run_start, -1
    return times


# ---------------------------------------------------------------------------
# The grid.
# ---------------------------------------------------------------------------
def beat_grid(y, sr: int, *, hop_length: int = HOP_LENGTH, start_bpm: float = START_BPM) -> dict:
    """Compute the downbeat-aware beat grid over a signal.

    Returns ``{beats, downbeats, tempo_changes, features}`` where the first three are
    lists of sample indices (ints, ascending, within ``[0, len(y)]``) and ``features``
    maps the registered ``rhythm.*`` keys to their values.
    """
    import librosa
    import numpy as np

    y = np.asarray(y, dtype="float32")
    if y.ndim > 1:  # collapse to mono
        y = np.mean(y, axis=0)
    n_samples = int(y.shape[0])

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length,
        start_bpm=start_bpm, trim=False, units="frames",
    )
    bpm = float(np.atleast_1d(tempo)[0])
    beat_frames = np.asarray(beat_frames, dtype=int).ravel()

    def _clamp(sample: int) -> int:
        return int(min(max(sample, 0), n_samples))

    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    beat_samples = [_clamp(round(float(t) * sr)) for t in beat_times]

    # --- downbeats: best (meter, phase) by accent contrast over the beat grid. ---
    meter, phase = bar_phase(beat_accents(y, beat_samples, sr))
    downbeat_samples = beat_samples[phase::meter]

    # --- tempo changes, snapped to the beat grid so they stay on the sample grid. ---
    change_samples: list[int] = []
    for t in tempo_change_times(onset_env, sr, hop_length=hop_length):
        target = _clamp(round(t * sr))
        if beat_samples:
            target = min(beat_samples, key=lambda s: abs(s - target))
        if target not in change_samples:
            change_samples.append(target)
    change_samples.sort()

    candidates, confidence = tempo_candidates(onset_env, sr, hop_length=hop_length)
    candidates = [_unfold_near(c, bpm) for c in candidates]

    return {
        "beats": beat_samples,
        "downbeats": list(downbeat_samples),
        "tempo_changes": change_samples,
        "features": {
            "rhythm.bpm": round(bpm, 3),
            "rhythm.bpm_confidence": round(confidence, 6),
            "rhythm.bpm_candidates": candidates,
            "rhythm.time_signature": f"{meter}/4",
        },
    }


def _points(samples, sr: int, label_prefix: str) -> list[dict]:
    """Marker points carrying float-second ``t`` AND the integer ``sample`` index."""
    return [
        {"t": round(s / sr, 6), "sample": int(s), "label": f"{label_prefix}-{i}"}
        for i, s in enumerate(samples)
    ]


def beats_audio_frame(
    audio_frame: dict,
    *,
    hop_length: int = HOP_LENGTH,
    start_bpm: float = START_BPM,
) -> list[dict]:
    """Resolve an `audio` frame's PCM from the CAS and emit its beat grid + rhythm features.

    Returns the three `marker` frames (roles ``beat``, ``downbeat``, ``tempo-change``) and
    the `feature` frame (role ``beats``), each carrying ``of``/``op``/``op_version``/
    ``params`` lineage per the tool contract. The caller passes the input frame through.
    """
    import soundfile as sf

    from smplstream import cas, frames as F

    src = cas.get_path(audio_frame["hash"])
    y, sr = sf.read(str(src), dtype="float32", always_2d=True)
    sr = int(sr)
    y = y.T  # (ch, n) for the mono collapse in beat_grid

    grid = beat_grid(y, sr, hop_length=hop_length, start_bpm=start_bpm)
    params = {"hop_length": hop_length, "start_bpm": start_bpm, "sr_hz": sr}
    of = audio_frame.get("id")
    lineage = [audio_frame["id"]] if audio_frame.get("id") else None

    out = [
        F.marker_frame(
            _points(grid[key], sr, prefix),
            role=role,
            of=of,
            op=OP,
            op_version=OP_VERSION,
            lineage=lineage,
            params=params,
        )
        for key, role, prefix in (
            ("beats", "beat", "beat"),
            ("downbeats", "downbeat", "downbeat"),
            ("tempo_changes", "tempo-change", "tempo-change"),
        )
    ]
    out.append(
        F.feature_frame(
            grid["features"],
            role="beats",
            of=of,
            op=OP,
            op_version=OP_VERSION,
            lineage=lineage,
            params=params,
        )
    )
    return out
