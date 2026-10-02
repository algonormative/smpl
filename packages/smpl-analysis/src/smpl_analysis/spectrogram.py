"""Spectrogram / waveform image rendering (research §2 — "Images for the LLM").

Render annotated PNGs for an `audio` frame and CAS each one, returning `image`
frames the rest of the pipe can resolve to a path (for the LLM to view):

- ``spectrogram:mel``  — annotated mel-spectrogram (always available; the default)
- ``spectrogram:cqt``  — constant-Q / log-frequency (chords, bass, detuning legible)
- ``spectrogram:hpss`` — harmonic vs percussive two-panel (pad vs stab)
- ``waveform``         — time-domain amplitude

Pure functions over a resolved audio frame; the thin `smpl spectrogram` subcommand
calls :func:`render_audio_frame`. Heavy imports (librosa, matplotlib) stay inside
functions so a cold pipe stage that never renders pays nothing. Matplotlib uses the
non-interactive **Agg** backend (no display, thread-safe, deterministic file output).
"""

from __future__ import annotations

import io
from typing import Optional

OP = "spectrogram"
OP_VERSION = "spectrogram@1"

# The render kinds this op knows how to produce, mapped to their image-frame role.
KINDS = ("mel", "cqt", "hpss", "waveform")
_ROLE = {
    "mel": "spectrogram:mel",
    "cqt": "spectrogram:cqt",
    "hpss": "spectrogram:hpss",
    "waveform": "waveform",
}

# STFT / mel defaults — modest sizes keep the PNGs small and the render fast.
_N_FFT = 2048
_HOP = 512
_N_MELS = 128

# Declared default table for the memo key (spec → *Parameter canonicalization*): omitted
# params are filled from THIS table for THIS op_version before hashing. `kind` has no
# default — each render kind is memoized separately, so asking for mel+cqt after a
# mel-only run hits the cached mel and renders only the cqt.
MEMO_DEFAULTS = {"n_fft": _N_FFT, "hop_length": _HOP, "n_mels": _N_MELS}


def pad_short_signal(y, n_fft: int):
    """Zero-pad a signal shorter than the FFT window up to ``n_fft`` samples.

    Short one-shots (single-cycle subs, tiny percussion) are shorter than the
    2048-sample window. librosa's STFT then warns and returns a near-empty
    (degenerate) transform — and on older librosa raised ``ParameterError``,
    which propagated up and caused the mel image frame to be dropped entirely
    during ``describe-all`` / contact-sheet builds. Padding to one full window
    guarantees a valid, full-resolution mel across librosa versions and never
    silently drops a frame; longer signals are returned unchanged.
    """
    import numpy as np

    if y.shape[-1] < n_fft:
        y = np.pad(y, (0, n_fft - y.shape[-1]))
    return y


def _load_mono(path: str) -> tuple["object", int]:
    """Decode an audio file to a mono float32 array + its native sample rate.

    No resampling (sr=None) — librosa display handles the native rate; keeping the
    native rate means the rendered frequency axis matches the source.
    """
    import librosa

    y, sr = librosa.load(path, sr=None, mono=True)
    return y, int(sr)


def _fig_to_png_bytes(fig) -> bytes:
    """Render a matplotlib figure to in-memory PNG bytes (no temp file)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def _render_mel(y, sr) -> bytes:
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    import numpy as np

    y = pad_short_signal(y, _N_FFT)  # short one-shots → full window (no drop, no warning)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=_N_FFT, hop_length=_HOP, n_mels=_N_MELS)
    S_db = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        S_db, sr=sr, hop_length=_HOP, x_axis="time", y_axis="mel", ax=ax, cmap="magma"
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Mel spectrogram")
    try:
        return _fig_to_png_bytes(fig)
    finally:
        plt.close(fig)


def _render_cqt(y, sr) -> bytes:
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    import numpy as np

    C = np.abs(librosa.cqt(y=y, sr=sr, hop_length=_HOP))
    C_db = librosa.amplitude_to_db(C, ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        C_db, sr=sr, hop_length=_HOP, x_axis="time", y_axis="cqt_note", ax=ax, cmap="magma"
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Constant-Q (log-frequency)")
    try:
        return _fig_to_png_bytes(fig)
    finally:
        plt.close(fig)


def _render_hpss(y, sr) -> bytes:
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    import numpy as np

    y = pad_short_signal(y, _N_FFT)  # short one-shots → full window (no drop, no warning)
    y_harm, y_perc = librosa.effects.hpss(y)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, comp, title in (
        (axes[0], y_harm, "Harmonic"),
        (axes[1], y_perc, "Percussive"),
    ):
        D = librosa.amplitude_to_db(np.abs(librosa.stft(comp, n_fft=_N_FFT, hop_length=_HOP)), ref=np.max)
        img = librosa.display.specshow(
            D, sr=sr, hop_length=_HOP, x_axis="time", y_axis="log", ax=ax, cmap="magma"
        )
        ax.set_title(title)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.suptitle("HPSS")
    try:
        return _fig_to_png_bytes(fig)
    finally:
        plt.close(fig)


def _render_waveform(y, sr) -> bytes:
    import librosa.display
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    try:
        return _fig_to_png_bytes(fig)
    finally:
        plt.close(fig)


_RENDER = {
    "mel": _render_mel,
    "cqt": _render_cqt,
    "hpss": _render_hpss,
    "waveform": _render_waveform,
}


def render_array(y, sr: int, kind: str) -> bytes:
    """Render one PNG of the given kind from a mono float32 array. Returns PNG bytes."""
    if kind not in _RENDER:
        raise ValueError(f"unknown spectrogram kind {kind!r}; choose from {KINDS}")
    # Force the Agg backend BEFORE pyplot is imported anywhere (non-interactive, headless,
    # deterministic file output — required by the hard rules).
    import matplotlib

    matplotlib.use("Agg")
    return _RENDER[kind](y, sr)


def _source_sr(audio_frame: dict) -> Optional[int]:
    """Native sample rate of the source, without decoding it.

    ``_load_mono`` loads at ``sr=None`` (native rate), so the CAS meta sidecar — written at
    ingest from the same file — carries the exact rate a render would have used. Reading it
    is what lets a cache hit skip the decode entirely; the frame's own ``meta`` is the
    fallback when no sidecar rate is recorded.
    """
    from smplstream import cas

    meta = cas.read_meta(audio_frame["hash"]) or {}
    sr = meta.get("sr") or (audio_frame.get("meta") or {}).get("sr")
    return int(sr) if sr else None


def render_audio_frame(audio_frame: dict, *, kinds: Optional[list[str]] = None,
                       use_cache: bool = True) -> list[dict]:
    """Render the requested image kinds for one `audio` frame; return `image` frames.

    Each PNG is stored in the CAS (``cas.put_blob(png, "image/png")``) and referenced by
    an `image` frame whose ``of`` is the audio frame id, with ``op``/``op_version``/``params``
    set per the tool contract. ``kinds`` defaults to ``["mel"]``.

    **Memoized per kind** (spec → *Memoization*): each render is keyed on
    ``(op, op_version, input audio hash, canonical params incl. kind)`` with an empty env
    fingerprint (matplotlib's Agg backend is deterministic file output). A hit reuses the
    PNG already in the CAS — no decode, no render — and, when every requested kind hits, the
    audio is never loaded at all. ``use_cache=False`` forces re-render and refreshes the
    entries. ``params.cache_hit`` records which path ran.
    """
    from smplstream import cas, frames as F, memo, memostore

    kinds = list(kinds) if kinds else ["mel"]

    plan: list[tuple[str, dict, str, Optional[str]]] = []  # (kind, params, memo key, hit hash)
    for kind in kinds:
        params = {"kind": kind, "n_fft": _N_FFT, "hop_length": _HOP, "n_mels": _N_MELS}
        mkey = memo.memo_key(
            OP, OP_VERSION, [audio_frame["hash"]], params=params, defaults=MEMO_DEFAULTS
        )
        entry = memostore.lookup(mkey) if use_cache else None
        plan.append((kind, params, mkey, entry["hash"] if entry else None))

    sr = _source_sr(audio_frame)
    y = None
    if any(hit is None for _, _, _, hit in plan) or sr is None:
        # At least one kind must actually render (or the sidecar has no rate) → decode once.
        src = cas.get_path(audio_frame["hash"])
        y, sr = _load_mono(str(src))

    out: list[dict] = []
    for kind, params, mkey, hit in plan:
        if hit is not None:
            h, cache_hit = hit, True
        else:
            png = render_array(y, sr, kind)
            h = cas.put_blob(png, "image/png")
            memostore.record(mkey, h, media="image/png", op=OP, op_version=OP_VERSION)
            cache_hit = False
        out.append(
            F.image_frame(
                h,
                media="image/png",
                role=_ROLE[kind],
                of=audio_frame["id"],
                op=OP,
                op_version=OP_VERSION,
                params={**params, "cache_hit": cache_hit},
                meta={"sr": sr},
            )
        )
    return out
