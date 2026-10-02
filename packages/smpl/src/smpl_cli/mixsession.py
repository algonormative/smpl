"""The `smpl mix` session model + render engine (the suite's **combinator** shape).

`mix` is the third frame-flow shape: N audio frames in → **one** rendered `audio` frame
out, carrying `lineage` over every input, `op: mix`, `op_version`, and the resolved
arrangement in `params`.

**Agent-as-UI split.** The *session* is canonical data on disk (a `.smplset.json` file);
`smpl mix` is a **stateless control plane** over it (`init` / `add-clip` / `set-gain` /
`set-pan` / `rm-clip` / `show` / `render`, cf. `mpc` over `mpd`). No daemon, no hidden
state: every verb is read-file → mutate → write-file, and `render` is a pure function of
(session ∪ CAS content). See ``docs/mix.md`` for the decision record.

Timebase is sample-accurate (spec → *Units & timebase*): a clip's ``at`` resolves to an
integer sample index at the session's ``sr``, and may be written directly as a sample
index, as seconds, as ``bar.beat.frac`` (needs ``bpm``), or as a **marker reference**
(``marker:beat#3``) which reads ``marker.data[i].sample`` at the marker's native rate.
"""

from __future__ import annotations

import io
import json
import math
import os
from pathlib import Path
from typing import Any, Optional

OP = "mix"
OP_VERSION = "mix@1"
SESSION_VERSION = 1

#: Clip-guard ceiling. A summed mix that exceeds this is scaled by ONE factor across the
#: whole bus (relative balance preserved) rather than clipped or per-clip normalized.
DEFAULT_CEILING_DBFS = -0.3

#: Op default table — filled into `params` BEFORE hashing (spec → *Parameter
#: canonicalization*: do NOT drop defaults).
DEFAULTS: dict[str, Any] = {"guard": "peak", "ceiling_dbfs": DEFAULT_CEILING_DBFS}


class MixError(Exception):
    """A fatal, user-facing mix error (bad session, unresolvable ref)."""


# ---------------------------------------------------------------------------
# Memo index (self-contained; ports to the shared store when it lands).
# ---------------------------------------------------------------------------
def mix_dir() -> Path:
    """Resolve the mix state dir (reads ``SMPL_MIX_DIR`` each call so tests override)."""
    return Path(os.environ.get("SMPL_MIX_DIR", "~/.smpl/mix")).expanduser()


def _memo_index_path() -> Path:
    return mix_dir() / "index.json"


def _read_memo_index() -> dict:
    p = _memo_index_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _write_memo_index(index: dict) -> None:
    p = _memo_index_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(index, sort_keys=True, indent=2))


# ---------------------------------------------------------------------------
# Session model — canonical data on disk.
# ---------------------------------------------------------------------------
def new_session(
    *,
    sr: Optional[int] = None,
    ch: Optional[int] = None,
    bpm: Optional[float] = None,
    beats_per_bar: float = 4.0,
) -> dict:
    s: dict[str, Any] = {"smplset": SESSION_VERSION, "tracks": []}
    if sr:
        s["sr"] = int(sr)
    if ch:
        s["ch"] = int(ch)
    if bpm:
        s["bpm"] = float(bpm)
        s["beats_per_bar"] = float(beats_per_bar)
    s["master"] = {"guard": "peak", "ceiling_dbfs": DEFAULT_CEILING_DBFS}
    return s


def load_session(path: str | Path) -> dict:
    try:
        raw = json.loads(Path(path).read_text())
    except OSError as exc:
        raise MixError(f"cannot read session {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise MixError(f"session {path} is not valid JSON: {exc}") from exc
    return normalize_session(raw)


def save_session(path: str | Path, session: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(session, indent=2, sort_keys=True) + "\n")


def normalize_session(raw: dict) -> dict:
    """Accept a hand-written session **or** a `smpl pattern` ``.smplset.json``.

    `smpl pattern` already emits the legacy smplmix session shape (``bpm`` /
    ``beats_per_bar`` / ``tracks[].clips[]`` with ``at: "bar.beat.frac"`` and a
    ``source: {path|hash|gen}``); that is the format this reads, so
    ``smpl pattern | smpl mix`` composes with no adapter.
    """
    if not isinstance(raw, dict):
        raise MixError("session must be a JSON object")
    s: dict[str, Any] = dict(raw)
    s.setdefault("smplset", SESSION_VERSION)
    tracks = s.get("tracks")
    if not isinstance(tracks, list):
        raise MixError("session needs a `tracks` array")
    norm_tracks = []
    for i, t in enumerate(tracks):
        if not isinstance(t, dict):
            raise MixError(f"track {i} must be an object")
        clips = t.get("clips") or []
        if not isinstance(clips, list):
            raise MixError(f"track {t.get('name', i)}: `clips` must be an array")
        # Preserve any field this version doesn't know about — a session is the user's
        # document, and a round-trip through `mix` must not quietly delete half of it.
        norm = {k: v for k, v in t.items() if k not in ("name", "gain_db", "pan", "clips")}
        norm.update({
            "name": str(t.get("name") or f"track{i + 1}"),
            "gain_db": float(t.get("gain_db", 0.0) or 0.0),
            "pan": float(t.get("pan", 0.0) or 0.0),
            "clips": [_normalize_clip(c) for c in clips],
        })
        norm_tracks.append(norm)
    s["tracks"] = norm_tracks
    master = s.get("master") if isinstance(s.get("master"), dict) else {}
    s["master"] = {
        "guard": str(master.get("guard", DEFAULTS["guard"])),
        "ceiling_dbfs": float(master.get("ceiling_dbfs", DEFAULTS["ceiling_dbfs"])),
    }
    return s


def _normalize_clip(c: Any) -> dict:
    if not isinstance(c, dict):
        raise MixError("every clip must be an object")
    src = c.get("source")
    if isinstance(src, str):
        src = parse_source_ref(src)
    if not isinstance(src, dict) or not src:
        raise MixError("every clip needs a `source` (role/id/hash/path)")
    at = c.get("at", {"sample": 0})
    if isinstance(at, bool):
        raise MixError(f"clip `at` must be a position, got {at!r}")
    if isinstance(at, (int, float)):
        at = {"sample": max(0, int(round(float(at))))}  # a bare JSON number is a sample index
    elif isinstance(at, str):
        at = parse_at(at)
    if not isinstance(at, dict):
        raise MixError(f"clip `at` must be a string or object, got {at!r}")
    out = {k: v for k, v in c.items() if k not in ("source", "at", "gain_db", "pan")}
    out.update({
        "source": src,
        "at": at,
        "gain_db": float(c.get("gain_db", 0.0) or 0.0),
        "pan": float(c.get("pan", 0.0) or 0.0),
    })
    # Legacy fields v1 does not render are KEPT on the clip and flagged — never silently
    # honored, never silently dropped.
    unsupported = [k for k in ("len", "transform") if c.get(k) is not None]
    if unsupported:
        out["unsupported"] = sorted(unsupported)
    return out


# ---------------------------------------------------------------------------
# Reference + position parsing.
# ---------------------------------------------------------------------------
def parse_source_ref(text: str) -> dict:
    """``role:<r>`` | ``id:<i>`` | ``blake3:<hex>`` | ``path:<p>`` | bare (role, else path)."""
    text = str(text).strip()
    if not text:
        raise MixError("empty source ref")
    if text.startswith("blake3:"):
        return {"hash": text}
    for prefix, key in (("role:", "role"), ("id:", "id"), ("path:", "path"), ("hash:", "hash")):
        if text.startswith(prefix):
            return {key: text[len(prefix):]}
    if os.sep in text or text.lower().endswith((".wav", ".aiff", ".aif", ".flac")):
        return {"path": text}
    return {"role": text}


def parse_at(text: str) -> dict:
    """``sample:N`` | ``sec:F`` / ``Fs`` | ``bar:B.b.f`` | ``marker:<role>#<i>``.

    Unprefixed forms follow the **legacy smplmix session** reading, which is what
    `smpl pattern` emits: a bare integer is a sample index, and a bare *dotted* numeral
    (``"1.3"``, ``"2.1.25"``) is ``bar.beat.frac`` — never seconds. Seconds always carry a
    marker (``sec:1.5`` or ``1.5s``), because ``"1.5"`` reading as 1.5 s in one place and
    bar 1 beat 5 in another is exactly the ambiguity that silently misplaces clips.
    """
    text = str(text).strip()
    if not text:
        return {"sample": 0}
    if text.startswith("sample:"):
        return {"sample": int(text[len("sample:"):])}
    if text.startswith("sec:"):
        return {"sec": float(text[len("sec:"):])}
    if text.startswith("bar:"):
        return {"bar": text[len("bar:"):]}
    if text.startswith("marker:"):
        body = text[len("marker:"):]
        role, _, idx = body.partition("#")
        return {"marker": {"role": role, "index": int(idx or 0)}}
    if text.endswith("s") and _is_number(text[:-1]):
        return {"sec": float(text[:-1])}
    if text.isdigit():
        return {"sample": int(text)}
    parts = text.split(".")
    if 2 <= len(parts) <= 3 and all(p.isdigit() for p in parts):
        return {"bar": text}
    raise MixError(f"cannot parse position {text!r} (use sample:N | sec:F | bar:B.b.f | marker:r#i)")


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _bar_to_beats(spec: str, beats_per_bar: float) -> float:
    """``"2.1.5"`` (bar.beat.frac, 1-indexed) → total beats from the start."""
    parts = str(spec).split(".")
    bar = int(parts[0] or 1)
    beat = int(parts[1]) if len(parts) > 1 and parts[1] != "" else 1
    frac = float("0." + parts[2]) if len(parts) > 2 and parts[2] != "" else 0.0
    return (bar - 1) * beats_per_bar + (beat - 1) + frac


# ---------------------------------------------------------------------------
# Verbs — stateless mutations over the session dict.
# ---------------------------------------------------------------------------
def _track(session: dict, name: str, *, create: bool = True) -> dict:
    for t in session["tracks"]:
        if t["name"] == name:
            return t
    if not create:
        raise MixError(f"no track named {name!r}")
    t = {"name": name, "gain_db": 0.0, "pan": 0.0, "clips": []}
    session["tracks"].append(t)
    return t


def add_clip(session: dict, *, source: str, at: str = "sample:0", gain_db: float = 0.0,
             pan: float = 0.0, track: str = "main") -> dict:
    clip = _normalize_clip({
        "source": parse_source_ref(source), "at": parse_at(at),
        "gain_db": gain_db, "pan": pan,
    })
    _track(session, track)["clips"].append(clip)
    return session


def set_gain(session: dict, *, db: float, track: Optional[str] = None,
             clip: Optional[int] = None) -> dict:
    if track is None:
        raise MixError("set-gain needs --track (optionally with --clip)")
    t = _track(session, track, create=False)
    if clip is None:
        t["gain_db"] = float(db)
    else:
        _clip_at(t, clip)["gain_db"] = float(db)
    return session


def set_pan(session: dict, *, pan: float, track: Optional[str] = None,
            clip: Optional[int] = None) -> dict:
    if track is None:
        raise MixError("set-pan needs --track (optionally with --clip)")
    t = _track(session, track, create=False)
    if clip is None:
        t["pan"] = float(pan)
    else:
        _clip_at(t, clip)["pan"] = float(pan)
    return session


def rm_clip(session: dict, *, track: str, clip: int) -> dict:
    t = _track(session, track, create=False)
    _clip_at(t, clip)
    t["clips"].pop(clip)
    return session


def _clip_at(track: dict, index: int) -> dict:
    if index < 0 or index >= len(track["clips"]):
        raise MixError(f"track {track['name']!r} has no clip {index} (0..{len(track['clips']) - 1})")
    return track["clips"][index]


# ---------------------------------------------------------------------------
# Resolution — session + stream → a fully-resolved, hashable arrangement plan.
# ---------------------------------------------------------------------------
def resolve_arrangement(session: dict, inframes: list[dict]) -> tuple[dict, list[dict]]:
    """Resolve every clip against the stream/CAS. Returns ``(plan, extra_frames)``.

    ``extra_frames`` are `audio` frames minted for ``path:`` sources ingested at render
    time — emitted BEFORE the mix frame so ``lineage`` never dangles (spec → *Tool
    contract / Lineage*).
    """
    from smplstream import cas, frames as F, select as S

    beats_per_bar = float(session.get("beats_per_bar", 4.0) or 4.0)
    bpm = float(session.get("bpm", 0.0) or 0.0)

    resolved: list[dict] = []
    extra: list[dict] = []
    lineage: list[str] = []
    srs: list[int] = []
    chs: list[int] = []

    for t in session["tracks"]:
        for ci, clip in enumerate(t["clips"]):
            frame = _resolve_source(clip["source"], inframes, extra, S, cas, F)
            # Trust the CAS sidecar over the frame's `meta` for the three fields the plan
            # is built from: the plan preallocates the bus, so a frame whose meta disagrees
            # with its own bytes would silently truncate the clip's tail at render time.
            cmeta = cas.read_meta(frame["hash"]) or {}
            meta = {**(frame.get("meta") or {}),
                    **{k: v for k, v in cmeta.items() if k in ("sr", "ch", "dur")}}
            srs.append(int(meta.get("sr") or 0))
            chs.append(int(meta.get("ch") or 1))
            fid = frame.get("id")
            if fid and fid not in lineage:
                lineage.append(fid)
            resolved.append({
                "track": t["name"], "index": ci,
                "hash": frame["hash"], "frame_id": fid,
                "sr": int(meta.get("sr") or 0), "ch": int(meta.get("ch") or 1),
                "dur": float(meta.get("dur") or 0.0),
                "at": clip["at"],
                "gain_db": round(float(clip["gain_db"]) + float(t["gain_db"]), 6),
                "pan": _clamp(float(clip["pan"]) + float(t["pan"]), -1.0, 1.0),
                "unsupported": clip.get("unsupported", []),
            })

    if not resolved:
        raise MixError("nothing to mix: the session has no clips")

    sr = int(session.get("sr") or 0) or (srs[0] if srs else 0)
    if sr <= 0:
        raise MixError("cannot determine session sample rate (set `sr` on the session)")
    want_pan = any(abs(c["pan"]) > 1e-9 for c in resolved)
    ch = int(session.get("ch") or 0) or (2 if (want_pan or max(chs) >= 2) else 1)

    for c in resolved:
        c["start_sample"] = _at_to_sample(c["at"], sr=sr, bpm=bpm, beats_per_bar=beats_per_bar,
                                          inframes=inframes)
        c["n_samples"] = int(round(c["dur"] * c["sr"])) if c["sr"] else 0
        c.pop("at", None)
        c.pop("dur", None)

    length = max(c["start_sample"] + c["n_samples"] for c in resolved)
    master = session.get("master") or {}
    plan = {
        "session_version": int(session.get("smplset", SESSION_VERSION)),
        "sr": sr,
        "ch": ch,
        "guard": str(master.get("guard", DEFAULTS["guard"])),
        "ceiling_dbfs": float(master.get("ceiling_dbfs", DEFAULTS["ceiling_dbfs"])),
        "length_samples": int(length),
        "clips": resolved,
        "lineage": lineage,
    }
    return plan, extra


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _resolve_source(src: dict, inframes: list[dict], extra: list[dict], S, cas, F) -> dict:
    """A clip source ref → the `audio` frame it names (ingesting a ``path:`` if needed)."""
    if src.get("id"):
        for f in inframes:
            if f.get("id") == src["id"] and f.get("kind") == "audio" and f.get("hash"):
                return f
        raise MixError(f"no audio frame with id {src['id']}")
    if src.get("role"):
        got = S.select(inframes, kind="audio", role=src["role"],
                       predicate=lambda f: bool(f.get("hash")), mode="last")
        if not got:
            raise MixError(f"no audio frame with role {src['role']!r}")
        return got[0]
    if src.get("hash"):
        h = src["hash"]
        for f in inframes:
            if f.get("kind") == "audio" and f.get("hash") == h:
                return f
        meta = cas.read_meta(h)
        if meta is None:
            raise MixError(f"hash {h} is not in the stream and not in the CAS")
        return _mint_input_frame(h, meta, role="mix.input", extra=extra, F=F)
    if src.get("path"):
        p = src["path"]
        if not Path(p).exists():
            raise MixError(f"clip source path not found: {p}")
        h = cas.put_audio_file(p)
        meta = cas.read_meta(h) or {}
        return _mint_input_frame(h, meta, role="mix.input", extra=extra, F=F,
                                 params={"source": p})
    if src.get("gen"):
        raise MixError("`gen:` clip sources are not rendered by mix v1 — generate upstream "
                       "(`smpl gen … | smpl mix --clip source=role:<r>`)")
    raise MixError(f"unrecognized clip source {src!r}")


def _mint_input_frame(h: str, meta: dict, *, role: str, extra: list[dict], F,
                      params: Optional[dict] = None) -> dict:
    for f in extra:
        if f.get("hash") == h:
            return f
    frame = F.audio_frame(
        h, sr=meta.get("sr", 0), ch=meta.get("ch", 1), dur=meta.get("dur", 0.0),
        role=role, op="read", op_version="read@1", fmt=meta.get("fmt"),
        params=params or {"hash": h},
    )
    extra.append(frame)
    return frame


def _at_to_sample(at: dict, *, sr: int, bpm: float, beats_per_bar: float,
                  inframes: list[dict]) -> int:
    """Resolve a clip position to an integer sample index at the session rate."""
    if "sample" in at:
        return max(0, int(at["sample"]))
    if "sec" in at:
        return max(0, int(round(float(at["sec"]) * sr)))
    if "bar" in at:
        if bpm <= 0:
            raise MixError("a bar.beat.frac position needs a positive `bpm` on the session")
        beats = _bar_to_beats(at["bar"], beats_per_bar)
        return max(0, int(round(beats * 60.0 / bpm * sr)))
    if "marker" in at:
        return _marker_to_sample(at["marker"], sr=sr, inframes=inframes)
    raise MixError(f"unrecognized position {at!r}")


def _marker_to_sample(ref: dict, *, sr: int, inframes: list[dict]) -> int:
    """``marker:<role>#<i>`` → ``data[i].sample``, rescaled from the marker's native rate.

    Sample-accurate by construction (spec → *Units & timebase*): markers destined for
    sample-exact placement carry ``sample: int`` at the frame's native ``meta.sr``, so a
    clip lands exactly on a detected onset/beat rather than on a rounded float second.
    """
    from smplstream import select as S

    role = ref.get("role")
    got = S.select(inframes, kind="marker", role=role, mode="last")
    if not got:
        raise MixError(f"no marker frame with role {role!r}")
    marker = got[0]
    points = marker.get("data") or []
    idx = int(ref.get("index", 0))
    if idx < 0 or idx >= len(points):
        raise MixError(f"marker {role!r} has no point {idx} (0..{len(points) - 1})")
    point = points[idx]
    native_sr = int((marker.get("params") or {}).get("sr_hz") or 0)
    if not native_sr:
        native_sr = _sr_of_ancestor(marker, inframes) or sr
    if point.get("sample") is None:
        # Float seconds only — usable, but not sample-exact; round at the session rate.
        return max(0, int(round(float(point.get("t", 0.0)) * sr)))
    return max(0, int(round(int(point["sample"]) * (sr / native_sr))))


def _sr_of_ancestor(frame: dict, inframes: list[dict]) -> int:
    parent = frame.get("of") or (frame.get("lineage") or [None])[0]
    for f in inframes:
        if f.get("id") == parent:
            return int((f.get("meta") or {}).get("sr") or 0)
    return 0


# ---------------------------------------------------------------------------
# Render — pure numpy summation. Same plan + same CAS content ⇒ same output hash.
# ---------------------------------------------------------------------------
def plan_memo_key(plan: dict) -> str:
    """memo_key over the **full input-hash set** + the canonical arrangement params."""
    from smplstream import memo

    return memo.memo_key(
        OP, OP_VERSION,
        sorted({c["hash"] for c in plan["clips"]}),
        params=plan_params(plan),
        defaults=DEFAULTS,
    )


def plan_params(plan: dict) -> dict:
    """The arrangement as memo-hashable params (frame ids excluded — content only).

    ``clips`` is a **sequence** (order preserved); the memo key's input-hash list is the
    order-insensitive set, so re-ordering two clips that land at the same sample changes
    nothing, while moving one clip changes the key.
    """
    return {
        "session_version": plan["session_version"],
        "sr": plan["sr"], "ch": plan["ch"],
        "guard": plan["guard"], "ceiling_dbfs": plan["ceiling_dbfs"],
        "length_samples": plan["length_samples"],
        "clips": [
            {"track": c["track"], "index": c["index"], "hash": c["hash"],
             "start_sample": c["start_sample"], "n_samples": c["n_samples"],
             "gain_db": c["gain_db"], "pan": c["pan"]}
            for c in plan["clips"]
        ],
    }


def _pan_gains(pan: float, ch: int):
    """Constant-power pan law: centre is -3 dB per leg (√2/2), edges are full on one leg."""
    if ch < 2:
        return (1.0,)
    angle = (_clamp(pan, -1.0, 1.0) + 1.0) * (math.pi / 4.0)
    return (math.cos(angle), math.sin(angle))


def _fit_channels(block, ch: int):
    import numpy as np

    if block.ndim == 1:
        block = block[:, None]
    if block.shape[1] == ch:
        return block
    if block.shape[1] > ch:  # downmix
        return block.mean(axis=1, keepdims=True) if ch == 1 else block[:, :ch]
    return np.repeat(block[:, :1], ch, axis=1)  # mono → N


def render_plan(plan: dict):
    """Sum the arrangement into a float32 ``(n, ch)`` buffer + a guard report."""
    import numpy as np
    import soundfile as sf

    from smplstream import cas

    sr, ch = plan["sr"], plan["ch"]
    bus = np.zeros((plan["length_samples"], ch), dtype="float64")

    for c in plan["clips"]:
        if c["sr"] != sr:
            raise MixError(
                f"clip {c['track']}[{c['index']}] is {c['sr']} Hz but the session is {sr} Hz — "
                f"mix v1 does not resample; convert it first (`smpl convert`)"
            )
        data, _ = sf.read(str(cas.get_path(c["hash"])), dtype="float32", always_2d=True)
        block = _fit_channels(data.astype("float64"), ch)
        block = block * (10.0 ** (float(c["gain_db"]) / 20.0))
        gains = _pan_gains(float(c["pan"]), ch)
        if ch >= 2:
            block = block * np.array([gains[0], gains[1]] + [1.0] * (ch - 2))
        start = c["start_sample"]
        end = min(start + block.shape[0], bus.shape[0])
        if end > start:
            bus[start:end] += block[: end - start]

    peak = float(np.max(np.abs(bus))) if bus.size else 0.0
    guard_db = 0.0
    ceiling = 10.0 ** (float(plan["ceiling_dbfs"]) / 20.0)
    if plan["guard"] == "peak" and peak > ceiling:
        guard_db = round(20.0 * math.log10(ceiling / peak), 6)
        bus = bus * (ceiling / peak)
    return bus.astype("float32"), {"peak_before": round(peak, 6), "guard_gain_db": guard_db}


def render(plan: dict, *, role: str = "mix") -> dict:
    """Render (or serve from the memo index) and return the one output `audio` frame."""
    import numpy as np
    import soundfile as sf

    from smplstream import cas, frames as F

    mkey = plan_memo_key(plan)
    index = _read_memo_index()
    cached = index.get(mkey)
    cache_hit = bool(cached and cas.exists(cached))

    if cache_hit:
        out_hash = cached
        report = {"peak_before": None, "guard_gain_db": None}
    else:
        bus, report = render_plan(plan)
        buf = io.BytesIO()
        sf.write(buf, np.ascontiguousarray(bus), plan["sr"], format="WAV", subtype="FLOAT")
        out_hash = cas.put_audio_bytes(buf.getvalue())
        index[mkey] = out_hash
        _write_memo_index(index)

    meta = cas.read_meta(out_hash) or {}
    params = {
        **plan_params(plan),
        "memo_key": mkey,
        "cache_hit": cache_hit,
        **{k: v for k, v in report.items() if v is not None},
    }
    return F.audio_frame(
        out_hash,
        sr=meta.get("sr", plan["sr"]),
        ch=meta.get("ch", plan["ch"]),
        dur=meta.get("dur", plan["length_samples"] / plan["sr"] if plan["sr"] else 0.0),
        role=role,
        lineage=plan["lineage"] or None,
        op=OP,
        op_version=OP_VERSION,
        params=params,
        fmt=meta.get("fmt"),
    )


# ---------------------------------------------------------------------------
# Data frames (control/session) — inline, or CAS-backed when over the inline limit.
# ---------------------------------------------------------------------------
def data_frame(kind: str, role: str, data: Any, *, params: Optional[dict] = None) -> dict:
    """Mint a `control` frame carrying ``data`` inline, spilling to CAS past 64 KiB."""
    from smplstream import cas, ndjson
    from smplstream.frames import MAX_INLINE_BYTES
    from smplstream.ids import mint_id

    frame: dict[str, Any] = {"kind": kind, "role": role, "op": OP, "op_version": OP_VERSION}
    blob = ndjson.dumps(data)
    if len(blob) > MAX_INLINE_BYTES:
        frame["hash"] = cas.put_blob(blob, "application/json")
        frame["media"] = "application/json"
    else:
        frame["data"] = data
    if params:
        frame["params"] = params
    return mint_id(frame)
