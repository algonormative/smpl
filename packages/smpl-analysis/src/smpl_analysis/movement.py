"""Movement feature family — attune-parity (ticket vault-1fxy).

The "how does this loop MOVE in time" axis. Where the aligned-hit envelope
(vault-3tuy) describes a single hit's shape, these metrics describe the whole
signal's ENERGY TRAJECTORY over time — pump depth, per-band modulation depth,
high-frequency activity, and end-of-loop tail decay. They are meaningful on
**loops and set recordings** and mostly **degenerate on one-shots / very short
material** (a single hit has no modulation cycle to measure).

Following the LUFS-integrated precedent (``loudness.py`` returns ``None`` when a
signal is shorter than the span the metric needs), the whole family is
**duration-gated**: material shorter than :data:`MIN_DURATION_S` emits every key
as ``None`` rather than a misleading number. See feature-keys.md → *Duration /
role gating*.

Metrics (measured off short-window RMS envelopes; unit dB unless noted):

  ``sidechain_db``          pump depth — the p90−p10 swing of the BROADBAND RMS
                            envelope (dB). A sidechained/pumping loop swings wide;
                            a static drone barely moves.
  ``bass_mod_depth_db``     the same swing on the BASS band (60–200 Hz, the
                            registry Bass edges): a gated/plucked bassline vs a
                            held sub.
  ``hf_mod_depth_db``       the same swing on the HIGH band (≥6 kHz, the registry
                            Air lower edge): shimmer/hats opening and closing vs a
                            steady top end.
  ``hf_silence_pct``        % of frames whose HF energy sits ≥60 dB below the
                            full-band peak (HF effectively absent). Sparse hats
                            read high, a continuous hiss ~0, a bass-only drone ~100.
  ``tail_decay_200ms_db``   dB the broadband envelope falls across the FINAL 200 ms
                            (first-half vs last-half mean level). A decaying/faded
                            tail reads positive; a loop that runs level into its end
                            ~0. (A loop-continuity signal for the QC gate.)

Layering rule (spec): this op EMITS the measurements only. What counts as "enough
pump" or "too much tail decay" lives in profiles / docs, never as a threshold here.

Pure functions returning frame dicts. Heavy imports (librosa/scipy/numpy) live
inside the functions so cold pipe stages start fast.
"""

from __future__ import annotations

from typing import Optional

OP = "movement"
OP_VERSION = "movement@1"

# --- pinned recipe constants (declared so memo params are complete & stable per op_version) --
MIN_DURATION_S = 1.0        # duration gate — below this the family is degenerate (emit null)
FRAME_LENGTH = 2048         # RMS-envelope window (~46 ms @ 44.1k): smooths audio-rate, passes beat-rate
HOP_LENGTH = 512            # RMS-envelope hop (~12 ms @ 44.1k → ~86 Hz envelope rate)
BASS_LO = 60.0              # bass-band low edge (registry Bass 60–200, reused)
BASS_HI = 200.0            # bass-band high edge
HF_HZ = 6000.0             # HF high-pass = registry Air lower edge (reused)
HF_SILENCE_REL_DB = 60.0   # a frame's HF is "silent" if ≥ this far below the full-band peak
TAIL_MS = 200.0            # tail-decay measurement window at the end of the signal
SWING_LO_PCT = 10.0        # robust envelope-swing low percentile
SWING_HI_PCT = 90.0        # robust envelope-swing high percentile
ACTIVE_REL_DB = 60.0       # frames below (peak − this) are silence, excluded from the swing

_FLOOR_DB = -140.0
_EPS = 1e-12

KEYS: tuple[str, ...] = (
    "movement.sidechain_db",
    "movement.bass_mod_depth_db",
    "movement.hf_mod_depth_db",
    "movement.hf_silence_pct",
    "movement.tail_decay_200ms_db",
)


# --------------------------------------------------------------------------------------------
# Low-level DSP helpers (pure numpy/scipy/librosa).
# --------------------------------------------------------------------------------------------


def _to_mono(y):
    import numpy as np

    y = np.asarray(y, dtype="float64")
    if y.ndim > 1:  # accept (ch, n) or (n, ch); collapse the channel axis
        y = y.mean(axis=0) if y.shape[0] < y.shape[1] else y.mean(axis=1)
    return np.ascontiguousarray(y)


def _sosfilt(x, sr: int, *, lo: Optional[float] = None, hi: Optional[float] = None,
             btype: str = "band", order: int = 4):
    """Zero-phase Butterworth filter (matches the sosfiltfilt style used across smpl-analysis)."""
    import numpy as np
    from scipy.signal import butter, sosfiltfilt

    x = np.asarray(x, dtype="float64")
    if x.shape[0] <= 3 * order * 2:  # sosfiltfilt needs more samples than its pad length
        return x
    nyq = sr / 2.0
    if btype == "band":
        lo_w = max((lo or 0.0) / nyq, 1e-4)
        hi_w = min((hi or nyq) / nyq, 0.999)
        if hi_w <= lo_w:
            return x
        sos = butter(order, [lo_w, hi_w], btype="bandpass", output="sos")
    elif btype == "high":
        w = min(max((lo or 0.0) / nyq, 1e-4), 0.999)
        sos = butter(order, w, btype="high", output="sos")
    else:  # low
        w = min(max((hi or nyq) / nyq, 1e-4), 0.999)
        sos = butter(order, w, btype="low", output="sos")
    return sosfiltfilt(sos, x)


def _rms_env(x, sr: int):
    """Per-frame RMS amplitude envelope (linear). Robust to signals shorter than one window."""
    import librosa
    import numpy as np

    from .spectrogram import pad_short_signal

    x = np.asarray(x, dtype="float32")
    if x.size == 0:
        return np.asarray([0.0], dtype="float64")
    x = pad_short_signal(x, FRAME_LENGTH)  # short one-shots → full window (no drop, no warning)
    rms = librosa.feature.rms(y=x, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH, center=True)[0]
    return np.asarray(rms, dtype="float64")


def _to_db(lin):
    import numpy as np

    floor = 10.0 ** (_FLOOR_DB / 20.0)
    return 20.0 * np.log10(np.maximum(np.asarray(lin, dtype="float64"), floor))


def _swing_db(env_lin) -> float:
    """p90−p10 dB swing over the ACTIVE frames of an RMS envelope (the modulation depth)."""
    import numpy as np

    env_db = _to_db(env_lin)
    if env_db.size < 2:
        return 0.0
    peak = float(env_db.max())
    active = env_db[env_db > peak - ACTIVE_REL_DB]
    if active.size < 2:
        return 0.0
    return float(np.percentile(active, SWING_HI_PCT) - np.percentile(active, SWING_LO_PCT))


def _hf_silence_pct(hf_env_lin, fullband_peak_db: float) -> float:
    """% of frames whose HF energy sits ≥ HF_SILENCE_REL_DB below the full-band peak."""
    import numpy as np

    hf_db = _to_db(hf_env_lin)
    if hf_db.size == 0:
        return 100.0
    silent = hf_db < (fullband_peak_db - HF_SILENCE_REL_DB)
    return 100.0 * float(np.mean(silent))


def _tail_decay_db(env_lin, sr: int) -> float:
    """dB drop across the final TAIL_MS window (first-half mean − last-half mean, in dB)."""
    import numpy as np

    env_db = _to_db(env_lin)
    if env_db.size < 2:
        return 0.0
    env_rate = sr / float(HOP_LENGTH)  # envelope frames per second
    tail_frames = max(2, int(round(TAIL_MS / 1000.0 * env_rate)))
    tail = env_db[-min(tail_frames, env_db.size):]
    if tail.size < 2:
        return 0.0
    half = tail.size // 2
    first = tail[:half] if half else tail[:1]
    last = tail[half:] if half else tail[-1:]
    return float(np.mean(first) - np.mean(last))


# --------------------------------------------------------------------------------------------
# Scalar pipeline.
# --------------------------------------------------------------------------------------------


def null_scalars() -> dict:
    """The movement keys, all ``None`` — emitted for duration-gated (too-short) material."""
    return {k: None for k in KEYS}


def movement_scalars(y, sr: int) -> dict:
    """The five movement scalars for a mono/multichannel signal (assumes duration OK).

    ``y`` may be 1-D or ``(ch, n)`` / ``(n, ch)``; it is collapsed to mono.
    """
    import numpy as np

    mono = _to_mono(y)

    broadband_env = _rms_env(mono, sr)
    fullband_peak_db = float(_to_db(broadband_env).max()) if broadband_env.size else _FLOOR_DB

    sidechain = _swing_db(broadband_env)

    bass = _sosfilt(mono, sr, lo=BASS_LO, hi=BASS_HI, btype="band")
    bass_mod = _swing_db(_rms_env(bass, sr))

    hf = _sosfilt(mono, sr, lo=HF_HZ, btype="high")
    hf_env = _rms_env(hf, sr)
    hf_mod = _swing_db(hf_env)
    hf_silence = _hf_silence_pct(hf_env, fullband_peak_db)

    tail = _tail_decay_db(broadband_env, sr)

    return {
        "movement.sidechain_db": round(float(sidechain), 3),
        "movement.bass_mod_depth_db": round(float(bass_mod), 3),
        "movement.hf_mod_depth_db": round(float(hf_mod), 3),
        "movement.hf_silence_pct": round(float(hf_silence), 2),
        "movement.tail_decay_200ms_db": round(float(tail), 3),
    }


# --------------------------------------------------------------------------------------------
# Frame-emitting entrypoint (the op behind `smpl movement` and describe-all).
# --------------------------------------------------------------------------------------------


def movement_audio_frame(audio_frame: dict) -> list[dict]:
    """Resolve an `audio` frame's PCM and emit its movement feature frame (role ``movement``).

    Duration-gated: material shorter than :data:`MIN_DURATION_S` emits every key as ``None``
    (``params.gated = True``) rather than a misleading number. Carries ``of``/``op``/
    ``op_version``/``params`` lineage; the caller passes the input frame through.
    """
    import soundfile as sf

    from smplstream import cas, frames as F

    src = cas.get_path(audio_frame["hash"])
    y, sr = sf.read(str(src), dtype="float32", always_2d=True)  # (n, ch)
    dur = y.shape[0] / float(sr) if sr else 0.0
    gated = dur < MIN_DURATION_S

    data = null_scalars() if gated else movement_scalars(y.T, sr)

    params = {
        "min_duration_s": MIN_DURATION_S, "gated": bool(gated), "dur_s": round(float(dur), 4),
        "frame_length": FRAME_LENGTH, "hop_length": HOP_LENGTH,
        "bass_band": [BASS_LO, BASS_HI], "hf_hz": HF_HZ,
        "hf_silence_rel_db": HF_SILENCE_REL_DB, "tail_ms": TAIL_MS,
        "swing_pct": [SWING_LO_PCT, SWING_HI_PCT],
    }
    return [
        F.feature_frame(
            data, role="movement", of=audio_frame["id"], op=OP, op_version=OP_VERSION,
            params=params,
        )
    ]
