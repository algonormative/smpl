"""Recording arc vs intended arc — energy-curve overlay + anchor segmentation (vault-3te4).

Overlays the **measured** arc of a session recording against the **intended** tension curve
in a set's ``narrative.yaml``: one annotated PNG plus one `feature` frame per detected
section, so an agent reads *callouts*, not two lines to eyeball.

Framing (non-negotiable, and it shapes the vocabulary in here): a measured-vs-intended
mismatch is a **difference** — very often the improvised departure that WORKED. Nothing in
this module calls it an error, a failure, or a miss.

**Why not just an LUFS curve.** A live set's LUFS trajectory is FLATTENED by the master
limiter: an LUFS-only "energy curve" draws a wall where the ear hears an arc. So the arc is
a composite of several trajectories, loudness one voice among them — ``rms_db`` (the
limiter-flattened loudness line), ``crest_db`` (peak-to-loudness: a dense limited climax
reads LOW, a sparse intro HIGH, so it enters the composite **inverted** — this is the line
that still moves when loudness does not), ``hi_sub_ratio_db`` (Air ≥6 k over Sub+Bass
20–200) and ``brightness_hz`` (spectral centroid). Each is normalized to the recording's own
p5–p95 span, so the overlay compares **shape**, not level; :data:`WEIGHTS` sums them.

**Sectioning.** Foote self-similarity novelty over an MFCC feature matrix, asked for roughly
as many sections as the narrative has **anchors** (consecutive beats sharing an ``anchor:``
are one anchor run). ``--sections`` overrides.

**Beat → time distribution (v1 rule).** Narrative beats carry no timestamps — the narrative
is *proportional*. v1 distributes beats **equally** across the recording's duration; a set
whose beats hold for wildly uneven lengths will therefore show a phase difference that is an
artifact of this rule, not of the performance.

Pure functions over a resolved audio frame; the thin `smpl arc` subcommand calls
:func:`arc_audio_frame`. Heavy imports (librosa, matplotlib, yaml) stay inside functions so a
cold pipe stage that never renders pays nothing. Matplotlib uses the **Agg** backend.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional

OP = "arc"
OP_VERSION = "arc@1"

# --- pinned recipe constants (declared so memo params are complete & stable per op_version) --
WINDOW_S = 4.0          # trajectory analysis window (long enough to average past a bar)
HOP_S = 1.0             # trajectory hop
N_FFT = 2048
HOP_LENGTH = 512
N_MFCC = 20
SEG_HOP_S = 0.5         # feature-matrix rate for the self-similarity segmentation
MIN_SECTION_S = 10.0    # minimum spacing between accepted novelty peaks
NORM_LO_PCT = 5.0       # trajectory normalization span (robust to a single stray frame)
NORM_HI_PCT = 95.0
DIVERGENCE_THRESHOLD = 0.20   # |measured − intended| above this is called out on the plot
MAX_SECTIONS = 24       # bound on the segmentation request (a narrative can't ask for 400)

# Composite weights. Loudness is deliberately NOT dominant — see the module docstring.
WEIGHTS: dict[str, float] = {
    "rms_db": 0.35,
    "crest_db": 0.25,        # inverted (low crest = dense/limited = high energy)
    "hi_sub_ratio_db": 0.25,
    "brightness_hz": 0.15,
}
INVERTED = frozenset({"crest_db"})

TRAJECTORIES: tuple[str, ...] = ("rms_db", "crest_db", "hi_sub_ratio_db", "brightness_hz")

_EPS = 1e-12
_FLOOR_DB = -140.0


# --- Narrative parsing (schema: narrative/v1).
@dataclass(frozen=True)
class Beat:
    id: str
    scene: str
    tension: float
    anchor: str


@dataclass(frozen=True)
class Narrative:
    set_name: str
    structure: str
    beats: tuple[Beat, ...]

    @property
    def anchors(self) -> tuple[str, ...]:
        """Anchor RUNS in order — consecutive beats sharing an anchor collapse to one entry.

        A narrative may legitimately return to an anchor later; that is a second run, and it
        is counted separately (it is a separate stretch of the set).
        """
        out: list[str] = []
        for b in self.beats:
            if not out or out[-1] != b.anchor:
                out.append(b.anchor)
        return tuple(out)


def parse_narrative(path: str) -> Narrative:
    """Read a ``narrative.yaml`` and return its :class:`Narrative` (beats in file order)."""
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: narrative must be a YAML mapping")
    schema = str(doc.get("schema") or "")
    if schema and not schema.startswith("narrative/"):
        raise ValueError(f"{path}: unsupported schema {schema!r} (expected narrative/v1)")
    raw = doc.get("beats")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: narrative has no `beats` list")
    beats: list[Beat] = []
    for i, b in enumerate(raw):
        if not isinstance(b, dict):
            raise ValueError(f"{path}: beat {i} is not a mapping")
        if "tension" not in b:
            raise ValueError(f"{path}: beat {b.get('id', i)!r} has no `tension`")
        beats.append(
            Beat(
                id=str(b.get("id") or f"beat-{i + 1}"),
                scene=str(b.get("scene") or ""),
                tension=float(b["tension"]),
                anchor=str(b.get("anchor") or "unanchored"),
            )
        )
    return Narrative(
        set_name=str(doc.get("set") or "untitled"),
        structure=str(doc.get("structure") or ""),
        beats=tuple(beats),
    )


def beat_spans(narrative: Narrative, duration_s: float) -> list[tuple[float, float]]:
    """Distribute beats over ``[0, duration_s)`` — v1 rule: **equal time per beat**."""
    n = len(narrative.beats)
    step = float(duration_s) / n if n else 0.0
    return [(i * step, (i + 1) * step) for i in range(n)]


def intended_at(narrative: Narrative, duration_s: float, times):
    """Sample the intended tension curve at ``times`` (piecewise-constant per beat)."""
    import numpy as np

    t = np.asarray(times, dtype="float64")
    n = len(narrative.beats)
    if n == 0 or duration_s <= 0:
        return np.zeros_like(t)
    step = duration_s / n
    idx = np.clip((t / step).astype(int), 0, n - 1)
    tension = np.asarray([b.tension for b in narrative.beats], dtype="float64")
    return tension[idx]


# --- Time-resolved trajectories.
def _to_mono(y):
    import numpy as np

    y = np.asarray(y, dtype="float64")
    if y.ndim > 1:
        y = y.mean(axis=0) if y.shape[0] < y.shape[1] else y.mean(axis=1)
    return np.ascontiguousarray(y)


def _band_edges() -> dict:
    """The six standardized band edges, reused from :mod:`smpl_analysis.width` (not redefined)."""
    from . import width as W

    return {name: (lo, hi) for name, lo, hi in W.BANDS if name != "full"}


def _db(x):
    import numpy as np

    floor = 10.0 ** (_FLOOR_DB / 20.0)
    return 20.0 * np.log10(np.maximum(np.asarray(x, dtype="float64"), floor))


def trajectories(y, sr: int, *, window_s: float = WINDOW_S, hop_s: float = HOP_S) -> dict:
    """Short-window trajectories over a mono signal.

    Returns ``{"t": times, "rms_db": …, "crest_db": …, "hi_sub_ratio_db": …,
    "brightness_hz": …}`` — one value per analysis window, all 1-D float arrays of equal
    length. One STFT feeds the spectral lines; the crest peak comes from the raw samples in
    the window (a peak is a time-domain fact).
    """
    import librosa
    import numpy as np

    mono = _to_mono(y)
    n = mono.size
    dur = n / float(sr) if sr else 0.0

    S = np.abs(librosa.stft(mono.astype("float32"), n_fft=N_FFT, hop_length=HOP_LENGTH))
    power = S ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    edges = _band_edges()
    sub_lo = edges["sub"][0]
    bass_hi = edges["bass"][1]
    air_lo = edges["air"][0]
    low_mask = (freqs >= sub_lo) & (freqs < bass_hi)
    hi_mask = freqs >= air_lo
    low_pf = power[low_mask].sum(axis=0) if low_mask.any() else np.zeros(power.shape[1])
    hi_pf = power[hi_mask].sum(axis=0) if hi_mask.any() else np.zeros(power.shape[1])
    tot_pf = power.sum(axis=0)
    centroid_pf = (freqs[:, None] * power).sum(axis=0) / np.maximum(tot_pf, _EPS)
    frame_t = librosa.frames_to_time(np.arange(power.shape[1]), sr=sr, hop_length=HOP_LENGTH)

    # Window grid: at least one window, last window clipped to the end of the signal.
    step = max(hop_s, 1e-3)
    n_win = max(1, int(np.floor(max(dur - window_s, 0.0) / step)) + 1)
    starts = np.arange(n_win) * step
    out = {k: np.zeros(n_win, dtype="float64") for k in TRAJECTORIES}
    times = np.zeros(n_win, dtype="float64")

    for i, t0 in enumerate(starts):
        t1 = min(t0 + window_s, dur)
        times[i] = 0.5 * (t0 + t1)
        s0, s1 = int(t0 * sr), max(int(t1 * sr), int(t0 * sr) + 1)
        seg = mono[s0:s1]
        rms = float(np.sqrt(np.mean(seg * seg))) if seg.size else 0.0
        peak = float(np.max(np.abs(seg))) if seg.size else 0.0
        out["rms_db"][i] = _db(rms)
        out["crest_db"][i] = float(_db(peak) - _db(max(rms, 10.0 ** (_FLOOR_DB / 20.0))))
        fm = (frame_t >= t0) & (frame_t < t1)
        if not fm.any():
            fm = np.zeros_like(frame_t, dtype=bool)
            fm[min(int(t0 * sr / HOP_LENGTH), fm.size - 1)] = True
        out["hi_sub_ratio_db"][i] = float(
            10.0 * np.log10((hi_pf[fm].mean() + _EPS) / (low_pf[fm].mean() + _EPS))
        )
        out["brightness_hz"][i] = float(centroid_pf[fm].mean())

    out["t"] = times
    return out


def normalize_curve(v):
    """Map a trajectory onto 0..1 over its own robust p5–p95 span (clamped).

    Shape, not absolute level: the overlay asks whether the recording RISES where the
    narrative asks it to rise, not whether it hit a particular dBFS.
    """
    import numpy as np

    v = np.asarray(v, dtype="float64")
    if v.size == 0:
        return v
    lo = float(np.percentile(v, NORM_LO_PCT))
    hi = float(np.percentile(v, NORM_HI_PCT))
    if hi - lo <= _EPS:
        return np.full_like(v, 0.5)
    return np.clip((v - lo) / (hi - lo), 0.0, 1.0)


def composite_energy(traj: dict, weights: Optional[dict] = None):
    """Weighted composite of the normalized trajectories, itself renormalized to 0..1.

    ``crest_db`` enters INVERTED: a limited, dense passage has a low peak-to-loudness ratio
    and is high-energy; a sparse passage has a high one and is low-energy.
    """
    import numpy as np

    w = dict(WEIGHTS if weights is None else weights)
    total = sum(w.values()) or 1.0
    acc = None
    for key, weight in w.items():
        n = normalize_curve(traj[key])
        if key in INVERTED:
            n = 1.0 - n
        acc = n * weight if acc is None else acc + n * weight
    acc = np.asarray(acc, dtype="float64") / total
    return normalize_curve(acc)


# --- Foote novelty segmentation.
def _checkerboard(size: int):
    import numpy as np

    half = size // 2
    g = np.arange(-half, half)
    taper = np.exp(-0.5 * (g / (half / 2.0 + _EPS)) ** 2)
    sign = np.outer(np.sign(g + 0.5), np.sign(g + 0.5))
    return sign * np.outer(taper, taper)


def foote_novelty(feat, kernel_size: int):
    """Foote novelty along the diagonal of a cosine self-similarity matrix.

    ``feat`` is ``(d, n)``; returns an ``(n,)`` novelty curve (edges are zero — a boundary
    cannot be asserted where half the checkerboard hangs off the recording).
    """
    import numpy as np

    F = np.asarray(feat, dtype="float64")
    norms = np.linalg.norm(F, axis=0, keepdims=True)
    Fn = F / np.maximum(norms, _EPS)
    ssm = Fn.T @ Fn                      # (n, n) cosine self-similarity
    n = ssm.shape[0]
    size = max(4, min(kernel_size - kernel_size % 2, n - (n % 2)))
    half = size // 2
    kern = _checkerboard(size)
    nov = np.zeros(n, dtype="float64")
    for i in range(half, n - half):
        nov[i] = float((ssm[i - half:i + half, i - half:i + half] * kern).sum())
    if nov.max() > _EPS:
        nov = nov / nov.max()
    return nov


def _pick_boundaries(nov, times, k: int, min_sep_s: float):
    """Greedy top-k−1 novelty peaks with a minimum spacing; returned in time order."""
    import numpy as np

    if k <= 1 or nov.size == 0:
        return []
    order = np.argsort(nov)[::-1]
    picked: list[int] = []
    for i in order:
        if nov[i] <= 0:
            break
        if all(abs(times[i] - times[j]) >= min_sep_s for j in picked):
            picked.append(int(i))
        if len(picked) >= k - 1:
            break
    return sorted(float(times[i]) for i in picked)


def segment(y, sr: int, *, n_sections: int, duration_s: float) -> list[tuple[float, float]]:
    """Segment the recording into approximately ``n_sections`` sections (Foote novelty).

    "Approximately" is honest: novelty may not offer enough well-separated peaks, in which
    case fewer sections come back. Never more.
    """
    import librosa
    import numpy as np

    n_sections = max(1, min(int(n_sections), MAX_SECTIONS))
    if n_sections == 1 or duration_s <= MIN_SECTION_S:
        return [(0.0, float(duration_s))]

    mono = _to_mono(y).astype("float32")
    hop = max(1, int(round(SEG_HOP_S * sr)))
    mfcc = librosa.feature.mfcc(y=mono, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=hop)
    times = librosa.frames_to_time(np.arange(mfcc.shape[1]), sr=sr, hop_length=hop)
    if mfcc.shape[1] < 8:
        return [(0.0, float(duration_s))]

    kernel = max(8, min(64, mfcc.shape[1] // max(2, n_sections)))
    nov = foote_novelty(mfcc, kernel)
    min_sep = max(MIN_SECTION_S, duration_s / (2.0 * n_sections))
    bounds = _pick_boundaries(nov, times, n_sections, min_sep)

    edges = [0.0, *bounds, float(duration_s)]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1) if edges[i + 1] > edges[i]]


# --- Per-section stats + divergence.
def _section_anchor(narrative: Narrative, duration_s: float, t0: float, t1: float) -> tuple:
    """The anchor + beat ids whose spans overlap ``[t0, t1)`` most (anchor by max overlap)."""
    spans = beat_spans(narrative, duration_s)
    overlap: dict[str, float] = {}
    ids: list[str] = []
    for beat, (b0, b1) in zip(narrative.beats, spans):
        ov = min(t1, b1) - max(t0, b0)
        if ov > 0:
            overlap[beat.anchor] = overlap.get(beat.anchor, 0.0) + ov
            ids.append(beat.id)
    if not overlap:
        return ("unanchored", [])
    return (max(overlap.items(), key=lambda kv: kv[1])[0], ids)


def _section_stats(y, sr: int, t0: float, t1: float) -> dict:
    """Scalar stats for one section: crest, band ratios, stereo width, tempo."""
    import librosa
    import numpy as np

    arr = np.asarray(y, dtype="float64")
    if arr.ndim == 1:
        arr = arr[None, :]
    s0, s1 = int(t0 * sr), max(int(t1 * sr), int(t0 * sr) + 1)
    seg = arr[:, s0:s1]
    mono = seg.mean(axis=0)
    rms = float(np.sqrt(np.mean(mono * mono))) if mono.size else 0.0
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0

    S = np.abs(librosa.stft(mono.astype("float32"), n_fft=N_FFT, hop_length=HOP_LENGTH)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    edges = _band_edges()
    bp = {n: float(S[(freqs >= lo) & (freqs < hi)].sum()) for n, (lo, hi) in edges.items()}
    low = bp.get("sub", 0.0) + bp.get("bass", 0.0)
    total = sum(bp.values()) + _EPS
    centroid = float((freqs[:, None] * S).sum() / max(S.sum(), _EPS))

    if seg.shape[0] >= 2:
        mid = 0.5 * (seg[0] + seg[1])
        side = 0.5 * (seg[0] - seg[1])
        m = float(np.sqrt(np.mean(mid * mid)))
        width = float(np.sqrt(np.mean(side * side)) / m) if m > _EPS else 0.0
    else:
        width = 0.0

    try:
        tempo = float(np.atleast_1d(librosa.feature.tempo(y=mono.astype("float32"), sr=sr))[0])
    except Exception:
        tempo = 0.0

    return {
        "arc.section.rms_db": round(float(_db(rms)), 3),
        "arc.section.crest_db": round(float(_db(peak) - _db(max(rms, 1e-7))), 3),
        "arc.section.hi_sub_ratio_db":
            round(float(10.0 * np.log10((bp.get("air", 0.0) + _EPS) / (low + _EPS))), 3),
        "arc.section.brightness_hz": round(centroid, 1),
        "arc.section.low_energy_fraction": round(float(low / total), 4),
        "arc.section.side_mid_ratio": round(width, 4),
        "arc.section.tempo_bpm": round(tempo, 2),
    }


def _difference_label(idx: int, anchor: str, divergence: float, threshold: float) -> Optional[str]:
    """A burned-in callout for a notable difference — never phrased as an error."""
    if abs(divergence) < threshold:
        return None
    direction = "over" if divergence > 0 else "under"
    return f"section {idx} ({anchor}): energy well {direction} intended ({divergence:+.2f})"


def analyze(y, sr: int, narrative: Narrative, *, n_sections: Optional[int] = None,
            window_s: float = WINDOW_S, hop_s: float = HOP_S,
            threshold: float = DIVERGENCE_THRESHOLD) -> dict:
    """The whole measurement: trajectories, sections, per-section stats, divergences.

    ``y`` is 1-D or ``(ch, n)`` (``(n, ch)`` is transposed for you). Returns a plain dict
    (no frames, no plotting) so it is testable and re-renderable.
    """
    import numpy as np

    arr = np.asarray(y, dtype="float64")
    if arr.ndim > 1 and arr.shape[0] > arr.shape[1]:
        arr = arr.T
    mono = _to_mono(arr)
    duration = mono.size / float(sr) if sr else 0.0

    traj = trajectories(mono, sr, window_s=window_s, hop_s=hop_s)
    measured = composite_energy(traj)
    intended = intended_at(narrative, duration, traj["t"])

    want = n_sections if n_sections else len(narrative.anchors)
    spans = segment(mono, sr, n_sections=want, duration_s=duration)

    sections = []
    for i, (t0, t1) in enumerate(spans, start=1):
        mask = (traj["t"] >= t0) & (traj["t"] < t1)
        if not mask.any():
            mask = np.zeros_like(traj["t"], dtype=bool)
            mask[min(int(np.searchsorted(traj["t"], t0)), mask.size - 1)] = True
        m = float(measured[mask].mean())
        it = float(intended[mask].mean())
        anchor, beat_ids = _section_anchor(narrative, duration, t0, t1)
        div = m - it
        sections.append({
            "index": i,
            "t0": round(float(t0), 3),
            "t1": round(float(t1), 3),
            "anchor": anchor,
            "beats": beat_ids,
            "measured_energy": round(m, 4),
            "intended_tension": round(it, 4),
            "divergence": round(div, 4),
            "notable_difference": abs(div) >= threshold,
            "difference": _difference_label(i, anchor, div, threshold),
            **_section_stats(arr, sr, t0, t1),
        })

    return {
        "set": narrative.set_name,
        "structure": narrative.structure,
        "duration_s": round(duration, 3),
        "anchors": list(narrative.anchors),
        "beat_distribution": "equal-per-beat",
        "t": traj["t"],
        "measured": measured,
        "intended": intended,
        "trajectories": {k: traj[k] for k in TRAJECTORIES},
        "sections": sections,
        "threshold": threshold,
    }


# --- Render.
def render(result: dict) -> bytes:
    """Render the annotated overlay PNG (annotations BURNED IN). Returns PNG bytes."""
    import matplotlib

    matplotlib.use("Agg")  # non-interactive, headless, deterministic file output
    import matplotlib.pyplot as plt

    t = result["t"]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 2]})

    ax.plot(t, result["intended"], color="#7a7a7a", lw=2.0, ls="--",
            drawstyle="steps-post", label="intended (narrative tension)")
    ax.plot(t, result["measured"], color="#1f77b4", lw=2.2, label="measured (composite energy)")

    callouts: list[str] = []
    for sec in result["sections"]:
        ax.axvline(sec["t0"], color="#cccccc", lw=0.8, zorder=0)
        mid = 0.5 * (sec["t0"] + sec["t1"])
        ax.text(mid, 1.24, f"{sec['index']}. {sec['anchor']}", ha="center", va="center",
                fontsize=8, color="#444444")
        if sec["notable_difference"]:
            ax.axvspan(sec["t0"], sec["t1"], color="#ff7f0e", alpha=0.13, zorder=0)
            # Fixed y so the callout never lands on top of a curve, whatever the shape.
            ax.text(mid, 1.13, f"{sec['divergence']:+.2f}", ha="center", va="center",
                    fontsize=9, color="#b35400", fontweight="bold")
            callouts.append(sec["difference"])

    ax.set_ylim(-0.08, 1.32)
    ax.set_ylabel("normalized 0–1")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.90), fontsize=9, framealpha=0.9)
    ax.set_title(f"Recording arc vs intended arc — {result['set']}\nshaded = notable "
                 "DIFFERENCE from the intended shape (a departure, not an error)", fontsize=12)

    for key, color in zip(TRAJECTORIES, ("#1f77b4", "#d62728", "#2ca02c", "#9467bd")):
        label = key + (" (inverted)" if key in INVERTED else "")
        series = normalize_curve(result["trajectories"][key])
        if key in INVERTED:
            series = 1.0 - series
        ax2.plot(t, series, lw=1.3, color=color, alpha=0.85, label=label)
    ax2.set_ylabel("normalized 0–1")
    ax2.set_xlabel("time (s)")
    ax2.legend(loc="upper right", fontsize=8, ncol=2)
    ax2.set_title("trajectories behind the composite — loudness is one line among several "
                  "(a limiter flattens it; crest still moves)", fontsize=9)

    body = "\n".join(callouts) if callouts else (
        "no section differs from the intended shape by more than "
        f"{result['threshold']:.2f} — the arc tracks the narrative"
    )
    fig.text(0.01, -0.01, "differences:\n" + body, ha="left", va="top", fontsize=9,
             family="monospace", color="#b35400" if callouts else "#2f6f2f")
    fig.tight_layout()

    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    finally:
        plt.close(fig)
    return buf.getvalue()


# --- Frame-emitting entrypoint (the op behind `smpl arc`).
def arc_audio_frame(audio_frame: dict, narrative_path: str, *,
                    n_sections: Optional[int] = None,
                    window_s: float = WINDOW_S, hop_s: float = HOP_S,
                    threshold: float = DIVERGENCE_THRESHOLD) -> list[dict]:
    """Resolve an `audio` frame, overlay it against a narrative, emit image + section frames.

    Emits, in order: one `image` frame (role ``arc:overlay``, PNG in the CAS), one `feature`
    frame per section (role ``arc:section`` — this IS the per-section NDJSON, since frames
    are NDJSON), and one `feature` summary (role ``arc:summary``).
    """
    import soundfile as sf

    from smplstream import cas, frames as F

    narrative = parse_narrative(narrative_path)
    src = cas.get_path(audio_frame["hash"])
    y, sr = sf.read(str(src), dtype="float32", always_2d=True)  # (n, ch)

    result = analyze(y.T, int(sr), narrative, n_sections=n_sections,
                     window_s=window_s, hop_s=hop_s, threshold=threshold)
    png = render(result)
    h = cas.put_blob(png, "image/png")

    # Complete + stable per op_version, so the memo key can't collide across recipes.
    params = {
        "narrative": narrative_path, "set": result["set"], "window_s": window_s,
        "hop_s": hop_s, "n_sections": len(result["sections"]),
        "requested_sections": int(n_sections) if n_sections else len(narrative.anchors),
        "beat_distribution": result["beat_distribution"], "divergence_threshold": threshold,
        "weights": dict(WEIGHTS), "n_fft": N_FFT, "hop_length": HOP_LENGTH,
        "seg_hop_s": SEG_HOP_S, "n_mfcc": N_MFCC,
    }
    aid = audio_frame["id"]
    kw = {"of": aid, "op": OP, "op_version": OP_VERSION, "params": params}
    out: list[dict] = [
        F.image_frame(h, media="image/png", role="arc:overlay",
                      meta={"sr": int(sr), "duration_s": result["duration_s"]}, **kw)
    ]
    out += [F.feature_frame(dict(sec), role="arc:section", **kw) for sec in result["sections"]]
    out.append(F.feature_frame({
        "set": result["set"], "structure": result["structure"],
        "duration_s": result["duration_s"], "anchors": result["anchors"],
        "n_sections": len(result["sections"]),
        "beat_distribution": result["beat_distribution"],
        "differences": [s["difference"] for s in result["sections"] if s["difference"]],
    }, role="arc:summary", **kw))
    return out
