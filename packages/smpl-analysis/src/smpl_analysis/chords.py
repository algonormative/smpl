"""Chord timeline + key/tuning estimation (research §3; ticket vault-379o).

Emits two derived frames per audio frame:

* a `marker` frame (role ``chord``) whose points use the SPEC marker shape —
  ``{t, dur, label, sample}``. A chord occupies a *range*, so ``t`` is the span start in
  float seconds, ``dur`` its length in seconds, ``label`` the chord symbol, and ``sample``
  the integer start-sample index at the audio's native ``meta.sr`` (spec → *Units &
  timebase*: markers destined for sample-exact export MUST carry ``sample``, which is what
  lets these chords round-trip to WAV ``cue`` / Octatrack ``.ot`` export).
* a `feature` frame (role ``key``) carrying the three registered keys
  ``tonal.key_key``, ``tonal.key_scale``, ``tonal.tuning_frequency`` (feature-keys.md).

Because both frames are ``of`` the same audio frame and the chord points are on the same
``t``/``sample`` timebase as beat/downbeat markers, chord-per-bar is derivable downstream
by joining chord spans against downbeat markers — no coupling between the two ops.

Engine choice (NOTE)
--------------------
The ticket names madmom (DeepChroma + CRF) and Chordino. Neither is installable on this
stack, so this op is a compact **librosa/numpy chroma + template-matching** path instead:

* **madmom** — verified unbuildable here 2026-08-29: 0.16.1 is sdist-only, does not declare
  its Cython build dependency, and does not build against NumPy 2.4. It is deliberately NOT
  a workspace dependency.
* **Chordino** — a Vamp plugin; needs a vamp host plus platform plugin binaries, which are
  not pip-installable.

librosa is already a workspace dependency, so the chroma path adds nothing to the install.
Should a buildable madmom/Chordino path appear, it replaces the internals behind the same
frame shapes and bumps ``OP_VERSION``.

Conventions
-----------
* **Label format** ``<pitch-class>:maj`` / ``<pitch-class>:min`` (e.g. ``C:maj``,
  ``A:min``), over the 24 major/minor triads. Pitch classes are spelled with **sharps
  only** (``C C# D D# E F F# G G# A A# B``) — no enharmonic/key-aware respelling.
* **No-chord** spans are labelled ``N`` and ARE emitted, for frames whose RMS is below
  :data:`SILENCE_DB` dBFS. A fully silent input therefore yields one ``N`` span rather than
  an empty timeline; a zero-length input yields an empty marker frame (not an error).
* **Smoothing** is a deterministic median filter over the 24×frames similarity matrix along
  time (no HMM/Viterbi), after which consecutive identical labels merge into spans.
* **Key** is Krumhansl-Schmuckler correlation of the mean chroma against the 24 key
  profiles. With no tonal energy at all (digital silence) ``tonal.key_key`` and
  ``tonal.key_scale`` are ``None``; ``tonal.tuning_frequency`` is always a float.

Pure functions returning frame dicts. Heavy imports (librosa/numpy/scipy) live inside the
functions so cold pipe stages start fast.
"""

from __future__ import annotations

OP = "chords"
OP_VERSION = "chords@1"

# Analysis defaults (declared so memo params are complete & stable per op_version).
HOP_LENGTH = 512
SMOOTH_FRAMES = 9          # median-filter width over the similarity matrix, in frames (odd)
SILENCE_DB = -60.0         # frame RMS below this (dBFS, ref=1.0) is labelled `N`
RMS_FRAME_LENGTH = 2048    # window for the per-frame RMS used by the `N` gate

# Sharp-only pitch-class spelling (see *Conventions* above).
PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

# Binary triad templates on C, rotated per root to build the 24-chord dictionary.
MAJOR_TRIAD = (1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0)  # root, major 3rd, 5th
MINOR_TRIAD = (1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0)  # root, minor 3rd, 5th

NO_CHORD = "N"

# Krumhansl-Schmuckler key profiles (C major / C minor), rotated per tonic.
KS_MAJOR = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
KS_MINOR = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)


def chord_templates():
    """Return ``(labels, T)`` — the 24 chord labels and their L2-normalized templates."""
    import numpy as np

    labels: list[str] = []
    rows = []
    for quality, base in (("maj", MAJOR_TRIAD), ("min", MINOR_TRIAD)):
        for i, pc in enumerate(PITCH_CLASSES):
            labels.append(f"{pc}:{quality}")
            rows.append(np.roll(np.asarray(base, dtype="float64"), i))
    T = np.asarray(rows)
    return labels, T / np.linalg.norm(T, axis=1, keepdims=True)


def _to_mono(y):
    import numpy as np

    y = np.asarray(y, dtype="float32")
    if y.ndim > 1:
        y = np.mean(y, axis=0)
    return np.ascontiguousarray(y)


def estimate_key(chroma, tuning: float) -> dict:
    """Krumhansl-Schmuckler key + tuning frequency from a chroma matrix.

    ``tuning`` is librosa's fractional-semitone offset from A440; the reported
    ``tonal.tuning_frequency`` is ``440 * 2**(tuning/12)`` Hz. Returns the three registered
    ``tonal.*`` keys.
    """
    import numpy as np

    tuning_hz = round(440.0 * 2.0 ** (float(tuning) / 12.0), 3)
    mean = np.nan_to_num(np.asarray(chroma, dtype="float64")).mean(axis=1) if np.size(chroma) else np.zeros(12)

    # No tonal energy (digital silence) → the correlation is undefined; report no key
    # rather than an arbitrary one.
    if not np.isfinite(mean).all() or float(mean.sum()) <= 1e-9 or float(mean.std()) <= 1e-9:
        return {
            "tonal.key_key": None,
            "tonal.key_scale": None,
            "tonal.tuning_frequency": tuning_hz,
        }

    best_r, best_key, best_scale = None, None, None
    for profile, scale in ((KS_MAJOR, "major"), (KS_MINOR, "minor")):
        prof = np.asarray(profile, dtype="float64")
        for i, pc in enumerate(PITCH_CLASSES):
            r = float(np.corrcoef(mean, np.roll(prof, i))[0, 1])
            if np.isfinite(r) and (best_r is None or r > best_r):
                best_r, best_key, best_scale = r, pc, scale

    return {
        "tonal.key_key": best_key,
        "tonal.key_scale": best_scale,
        "tonal.tuning_frequency": tuning_hz,
    }


def chord_timeline(
    y,
    sr: int,
    *,
    hop_length: int = HOP_LENGTH,
    smooth_frames: int = SMOOTH_FRAMES,
    silence_db: float = SILENCE_DB,
) -> dict:
    """Analyze a signal into chord spans + key/tuning.

    Returns ``{"markers": [{t, dur, label, sample}, ...], "tonal": {tonal.* keys}}``.
    Spans are contiguous, ascending, non-overlapping, each with ``dur > 0``, and the last
    span's ``t + dur`` is clamped to the signal duration. An empty signal yields no spans.
    """
    import librosa
    import numpy as np
    from scipy.ndimage import median_filter

    y = _to_mono(y)
    n_samples = int(y.shape[-1])

    if n_samples == 0:
        return {"markers": [], "tonal": estimate_key(np.zeros((12, 0)), 0.0)}

    tuning = float(librosa.estimate_tuning(y=y, sr=sr))
    chroma = np.nan_to_num(
        librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length, tuning=tuning)
    )

    labels, T = chord_templates()
    norms = np.linalg.norm(chroma, axis=0, keepdims=True)
    sim = T @ (chroma / np.maximum(norms, 1e-12))          # cosine similarity, (24, frames)
    sim = median_filter(sim, size=(1, max(1, int(smooth_frames))), mode="nearest")
    best = np.argmax(sim, axis=0)

    rms = librosa.feature.rms(y=y, frame_length=RMS_FRAME_LENGTH, hop_length=hop_length)[0]
    quiet = librosa.amplitude_to_db(rms, ref=1.0) < float(silence_db)

    n_frames = min(best.shape[0], quiet.shape[0])
    per_frame = [NO_CHORD if quiet[i] else labels[int(best[i])] for i in range(n_frames)]

    # Merge runs of identical labels into spans; frame i covers [i*hop, (i+1)*hop).
    points: list[dict] = []
    dur_s = n_samples / float(sr)
    start = 0
    for i in range(1, n_frames + 1):
        if i < n_frames and per_frame[i] == per_frame[start]:
            continue
        t0 = min(start * hop_length / float(sr), dur_s)
        t1 = min(i * hop_length / float(sr), dur_s)
        if t1 > t0:
            points.append({
                "t": round(t0, 6),
                "dur": round(t1 - t0, 6),
                "label": per_frame[start],
                "sample": min(int(round(t0 * sr)), n_samples),
            })
        start = i

    return {"markers": points, "tonal": estimate_key(chroma, tuning)}


def chords_audio_frame(
    audio_frame: dict,
    *,
    hop_length: int = HOP_LENGTH,
    smooth_frames: int = SMOOTH_FRAMES,
    silence_db: float = SILENCE_DB,
) -> list[dict]:
    """Resolve an `audio` frame's PCM from the CAS and emit its chord + key frames.

    Returns ``[marker_frame(role="chord"), feature_frame(role="key")]``, both carrying
    ``of``/``op``/``op_version``/``params`` lineage per the tool contract. The caller is
    responsible for passthrough of the input frame.
    """
    import soundfile as sf

    from smplstream import cas, frames as F

    src = cas.get_path(audio_frame["hash"])
    y, sr = sf.read(str(src), dtype="float32", always_2d=True)
    sr = int(sr)

    result = chord_timeline(
        y.T, sr, hop_length=hop_length, smooth_frames=smooth_frames, silence_db=silence_db
    )

    fid = audio_frame.get("id")
    lineage = [fid] if fid else None
    params = {
        "hop_length": hop_length,
        "smooth_frames": smooth_frames,
        "silence_db": silence_db,
        "sr_hz": sr,
    }
    return [
        F.marker_frame(
            result["markers"],
            role="chord",
            of=fid,
            lineage=lineage,
            op=OP,
            op_version=OP_VERSION,
            params=params,
        ),
        F.feature_frame(
            result["tonal"],
            role="key",
            of=fid,
            lineage=lineage,
            op=OP,
            op_version=OP_VERSION,
            params=params,
        ),
    ]
