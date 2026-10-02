"""Embedded metadata on export — WAV `smpl`/`cue `/`acid`/`bext`, FLAC Vorbis comments.

spec → *Standards alignment* marks this **[v1, export-side SHOULD]**: a tool that writes
audio SHOULD carry the stream's analysis out *inside the file*, so the sample auto-warps in
a DAW and loops in a sampler with no smplstream present. This module is that carry-out; it
is pure byte surgery over an already-written file (soundfile writes none of these chunks),
so nothing here decodes or re-encodes audio.

What lands where
----------------
===========  =====================================================================
`smpl`       root note (from ``tonal.key_key``) + loop points (sample-indexed)
`cue `       slice/transient points, one cue per `marker` point (sample-indexed)
`acid`       tempo (``rhythm.bpm``), root note, beat count, meter
             (``rhythm.time_signature``), one-shot|loop flag
`bext`       BWF originator + origination date/time + description (a `text` caption)
Vorbis       FLAC mirror of the same tempo/key/loop: ``BPM``, ``KEY``,
             ``LOOP_START``, ``LOOP_END``
===========  =====================================================================

**Only what the stream actually carries is embedded.** No bpm on the wire → no acid tempo;
no key → no root note; no markers → no `cue ` chunk. Nothing is invented or inferred from
the audio here — that is the analysis ops' job, upstream. A stream carrying no metadata at
all writes no chunks, so a bare ``smpl read x.wav | smpl write y.wav`` stays byte-faithful.

One-shot or loop? (the flag that decides whether a DAW warps at all)
--------------------------------------------------------------------
ACID-honouring hosts do **not** tempo-stretch a file flagged one-shot — the tempo and beat
count are inert on it. So flagging the canonical ``smpl read loop.wav | smpl beats | smpl
write out.wav`` as a one-shot would make the acid chunk useless in exactly the case it
exists for. A file is written as a **loop** when either:

- it carries explicit loop evidence — ``loop``-role markers, or the ``loopify`` op; or
- the stream carries a **rhythmic grid for this frame**: a known ``rhythm.bpm``, a
  beat/downbeat marker frame referencing the written audio, and a written length that is a
  whole number of beats (``>= MIN_GRID_BEATS``, within ``GRID_TOLERANCE``). The tolerance
  is relative to the beat count because tempo-estimate error accumulates over length.

Everything else stays a one-shot: a bare hit with no beat grid, and a beat-tracked file
whose length is not grid-cut (a full track, a tail-trimmed take). The root note and loop
points still travel on those; only the warp flag differs.

Sample positions come from `marker.sample`, never from `t`
----------------------------------------------------------
Cue and loop positions are **sample-indexed by definition**, so this module REQUIRES the
integer ``sample`` field the marker producers emit (`beats`, `chords`, `slice`, `qc`) and
raises :class:`MarkerPrecisionError` when a point lacks it. Rounding float-second ``t``
back to samples here would write cue points that are quietly a hop or two off — silently
wrong slice points are worse than a refused export, and the caller surfaces the refusal as
an `error` frame before any bytes are written.

Manual verification (the DAW/sampler check this exists for)
-----------------------------------------------------------
The tests below re-parse the written container byte-level (round-trip), which proves the
chunks are *well-formed and carry the right numbers* — it cannot prove a host reads them.
That half is a **manual check**, run when the chunk writers change:

1. ``smpl read loop.wav | smpl beats | smpl chords | smpl write out.wav``
2. **Auto-warp (acid):** drag ``out.wav`` into Ableton Live (or Bitwig/FL). The clip must
   arrive already warped at the analyzed BPM with Warp enabled — Live reads the `acid`
   chunk's tempo + beat count. A clip that lands unwarped, or warped to the project tempo
   with the wrong stretch, means the `acid` tempo/beats disagree with the audio length.
3. **Loop + root note (smpl):** load ``out.wav`` into a hardware/software sampler that
   honours the `smpl` chunk (Kontakt, TAL-Sampler, Renoise, an MPC). Held notes must loop
   at the written loop points with no re-trigger click, and the sample must play back at
   unity pitch on the root-note key.
4. **Slices (cue):** open in Recycle/Serato/Renoise (or Audacity → *File ▸ Import ▸ Labels*
   is not it — Audacity reads cue points on import directly) and confirm one marker per
   analyzed onset, landing on the transients.
5. **bext:** ``ffprobe -show_format out.wav`` (or BWF MetaEdit) lists the originator,
   origination date, and description.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Optional

# Pitch-class order (matches smpl_analysis.chords.PITCH_CLASSES — the producer of
# `tonal.key_key`); index into it is the semitone offset from C.
PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

# Marker roles eligible for the `cue ` chunk, most slice-like first — the first role
# present in the stream wins, so an onset-sliced sample does not also get a beat grid
# written over its cue list.
CUE_ROLE_PREFERENCE = ("slice", "onset", "beat", "downbeat")

# Marker role carrying explicit loop points (start[, end]).
LOOP_ROLE = "loop"

# Marker roles that constitute a rhythmic grid for the written frame (see the module
# docstring's one-shot|loop rules).
GRID_ROLES = ("beat", "downbeat")

# A written length counts as grid-cut when its beat count is within this fraction of a
# whole number of beats. Relative, not absolute: a tempo estimate off by 0.15% is 0.01
# beats out over 8 beats and 0.1 out over 64, so a fixed beats tolerance would call long
# loops one-shots purely for being long.
GRID_TOLERANCE = 0.03

# Below this, "a whole number of beats" is not evidence of anything (a one-beat hit is
# trivially grid-cut).
MIN_GRID_BEATS = 2

# MIDI note for C in the sampler-default octave (60 = middle C). `tonal.key_key`'s pitch
# class is added to this to get the `smpl`/`acid` root note.
ROOT_NOTE_C = 60

# BWF originator: the producing entity, not a person.
ORIGINATOR = "smpl"

# acid flag bits (Acidizer).
ACID_ONE_SHOT = 0x01
ACID_ROOT_SET = 0x02

# Chunks this module owns: any pre-existing copy is REPLACED, never duplicated.
_OWNED_CHUNKS = (b"bext", b"acid", b"smpl", b"cue ")

_VORBIS_COMMENT_BLOCK = 4


class MarkerPrecisionError(Exception):
    """A marker point bound for a sample-indexed chunk lacks the REQUIRED int ``sample``."""


# ---------------------------------------------------------------------------
# Plan: what the stream actually carries.
# ---------------------------------------------------------------------------
def _references(frame: dict, fid: str) -> bool:
    return frame.get("of") == fid or fid in (frame.get("lineage") or [])


def _parse_time_signature(value: Any) -> Optional[tuple[int, int]]:
    """``"4/4"`` → ``(4, 4)``; anything unparseable → None (never a guessed meter)."""
    if not isinstance(value, str) or "/" not in value:
        return None
    num, _, den = value.partition("/")
    try:
        n, d = int(num), int(den)
    except ValueError:
        return None
    return (n, d) if n > 0 and d > 0 else None


def _sample_of(point: Any, *, role: str, index: int) -> int:
    sample = point.get("sample") if isinstance(point, dict) else None
    if not isinstance(sample, int) or isinstance(sample, bool):
        raise MarkerPrecisionError(
            f"marker role {role!r} point {index} carries no integer `sample`; refusing to "
            "round float-second `t` into sample-indexed chunk positions"
        )
    return int(sample)


def _cue_samples(markers: dict[str, list]) -> list[int]:
    for role in CUE_ROLE_PREFERENCE:
        points = markers.get(role)
        if not points:
            continue
        return [_sample_of(p, role=role, index=i) for i, p in enumerate(points)]
    return []


def _loop_points(markers: dict[str, list], audio: dict) -> Optional[tuple[int, Optional[int]]]:
    """``(start, end)`` in samples; ``end is None`` means "to the end of the file".

    Explicit ``loop``-role markers win. Failing that, audio produced by ``smpl loop``
    (op ``loopify``) IS a loop by construction — the op exists to make the whole file
    tile-safe — so the whole file is the loop.
    """
    points = markers.get(LOOP_ROLE)
    if points:
        samples = [_sample_of(p, role=LOOP_ROLE, index=i) for i, p in enumerate(points)]
        return (samples[0], samples[-1] if len(samples) > 1 else None)
    if audio.get("op") == "loopify":
        return (0, None)
    return None


def collect(frames: list[dict], audio: dict) -> dict:
    """Build the embed plan for ``audio`` from the stream. Raises :class:`MarkerPrecisionError`.

    Call this BEFORE writing bytes: the marker precondition must fail without leaving a
    file carrying wrong cue points on disk.

    **Markers must reference the written frame; scalars need not.** A tempo, a key and a
    caption survive a downstream edit, so they are read from the whole stream (last-wins).
    Sample positions do not: markers computed on the source are in the source's
    coordinates, and a trimming op like ``loopify`` shifts every one of them. So a marker
    frame is used only when it points at the frame being written (``of``/``lineage``) —
    embedding nothing beats embedding cue points that are silently off by a trim.
    """
    fid = audio.get("id")
    related = [f for f in frames if fid and _references(f, fid)]
    scalars = related or list(frames)
    positional = related if fid else list(frames)

    features: dict[str, Any] = {}
    caption: Optional[str] = None
    for frame in scalars:
        kind, data = frame.get("kind"), frame.get("data")
        if kind == "feature" and isinstance(data, dict):
            features.update(data)  # later frames win
        elif kind == "text" and frame.get("role") == "caption" and isinstance(data, str):
            caption = data

    markers: dict[str, list] = {}
    for frame in positional:
        if frame.get("kind") == "marker" and isinstance(frame.get("data"), list):
            markers[frame.get("role") or "marker"] = frame["data"]

    plan: dict[str, Any] = {}
    bpm = features.get("rhythm.bpm")
    if isinstance(bpm, (int, float)) and not isinstance(bpm, bool) and bpm > 0:
        plan["bpm"] = float(bpm)
    meter = _parse_time_signature(features.get("rhythm.time_signature"))
    if meter:
        plan["time_signature"] = meter
    key = features.get("tonal.key_key")
    if isinstance(key, str) and key in PITCH_CLASSES:
        plan["key"] = key
        plan["root_note"] = ROOT_NOTE_C + PITCH_CLASSES.index(key)
        scale = features.get("tonal.key_scale")
        if isinstance(scale, str):
            plan["scale"] = scale
    cues = _cue_samples(markers)
    if cues:
        plan["cue"] = cues
    loop = _loop_points(markers, audio)
    if loop is not None:
        plan["loop"] = loop
    if any(markers.get(role) for role in GRID_ROLES):
        # Not positions — just "a beat grid was tracked for THIS frame", which is half the
        # evidence for the acid loop flag (the other half is the grid-cut length check,
        # which needs the written file's sample count).
        plan["beat_grid"] = True
    if caption:
        plan["description"] = caption
    return plan


def beat_count(plan: dict, *, sr: int, n_samples: int) -> Optional[float]:
    """The written length in beats at the planned tempo, or None without one."""
    bpm = plan.get("bpm")
    if not bpm or sr <= 0 or n_samples <= 0:
        return None
    return n_samples / sr * bpm / 60.0


def is_grid_cut(plan: dict, *, sr: int, n_samples: int) -> bool:
    """True when a tracked grid says the written length is a whole number of beats.

    The loop half of the acid flag for material with no explicit loop points: a
    beat-tracked file cut to the grid is a loop a DAW should warp, while the same
    analysis over a length that lands mid-beat is a take, not a loop.
    """
    if not plan.get("beat_grid"):
        return False
    beats = beat_count(plan, sr=sr, n_samples=n_samples)
    if beats is None:
        return False
    whole = round(beats)
    return whole >= MIN_GRID_BEATS and abs(beats - whole) <= GRID_TOLERANCE * whole


def _resolve_loop(plan: dict, n_samples: int) -> Optional[tuple[int, int]]:
    """Clamp the planned loop to the written file. ``end`` is the LAST sample played."""
    loop = plan.get("loop")
    if loop is None or n_samples <= 0:
        return None
    last = n_samples - 1
    start, end = loop
    start = max(0, min(int(start), last))
    end = last if end is None else max(start, min(int(end), last))
    return (start, end)


# ---------------------------------------------------------------------------
# RIFF/WAVE surgery.
# ---------------------------------------------------------------------------
def _riff_chunk(cid: bytes, payload: bytes) -> bytes:
    """One RIFF chunk: id + LE size + payload, word-aligned (pad byte excluded from size)."""
    out = cid + struct.pack("<I", len(payload)) + payload
    return out + b"\x00" if len(payload) % 2 else out


def parse_riff(raw: bytes) -> list[tuple[bytes, bytes]]:
    """``[(chunk_id, payload), ...]`` from a RIFF/WAVE file. Also the tests' reader."""
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE container")
    chunks: list[tuple[bytes, bytes]] = []
    pos = 12
    while pos + 8 <= len(raw):
        cid = raw[pos : pos + 4]
        size = struct.unpack_from("<I", raw, pos + 4)[0]
        chunks.append((cid, raw[pos + 8 : pos + 8 + size]))
        pos += 8 + size + (size & 1)
    return chunks


def _build_riff(chunks: list[tuple[bytes, bytes]]) -> bytes:
    body = b"WAVE" + b"".join(_riff_chunk(cid, payload) for cid, payload in chunks)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def smpl_payload(*, sr: int, root_note: int, loop: Optional[tuple[int, int]]) -> bytes:
    """The `smpl` chunk: sampler root note + (0 or 1) forward loop, sample-indexed."""
    sample_period = int(round(1e9 / sr)) if sr > 0 else 0
    loops = [] if loop is None else [loop]
    payload = struct.pack(
        "<9I", 0, 0, sample_period, root_note, 0, 0, 0, len(loops), 0
    )
    for i, (start, end) in enumerate(loops):
        payload += struct.pack("<6I", i, 0, start, end, 0, 0)  # type 0 = forward
    return payload


def cue_payload(samples: list[int]) -> bytes:
    """The `cue ` chunk: one 24-byte cue point per marker, positioned by sample index."""
    payload = struct.pack("<I", len(samples))
    for i, sample in enumerate(samples):
        payload += struct.pack("<II4sIII", i + 1, sample, b"data", 0, 0, sample)
    return payload


def acid_payload(
    *,
    bpm: Optional[float],
    root_note: Optional[int],
    num_beats: int,
    meter: Optional[tuple[int, int]],
    looping: bool,
) -> bytes:
    """The 24-byte Acidizer chunk — what a DAW reads to auto-warp."""
    flags = 0 if looping else ACID_ONE_SHOT
    if root_note is not None:
        flags |= ACID_ROOT_SET
    num, den = meter or (4, 4)
    return struct.pack(
        "<IHHfIHHf",
        flags,
        root_note or 0,
        0x8000,          # constant, per the format
        0.0,             # unused float, per the format
        num_beats,
        den,
        num,
        float(bpm or 0.0),
    )


def bext_payload(*, description: str, originator: str, date: str, time: str) -> bytes:
    """The 602-byte BWF `bext` chunk (version 1, no coding history)."""

    def fixed(value: str, size: int) -> bytes:
        return value.encode("utf-8", "replace")[:size].ljust(size, b"\x00")

    return (
        fixed(description, 256)
        + fixed(originator, 32)
        + fixed("", 32)              # OriginatorReference
        + fixed(date, 10)
        + fixed(time, 8)
        + struct.pack("<IIH", 0, 0, 1)  # TimeReference lo/hi, Version
        + b"\x00" * 64                  # UMID
        + b"\x00" * 10                  # loudness fields (v2 only; zero here)
        + b"\x00" * 180                 # Reserved
    )


def embed_wav(path: Path, plan: dict, *, sr: int, n_samples: int) -> list[str]:
    """Splice the planned chunks into a written WAV. Returns the chunk names written."""
    from datetime import datetime, timezone

    loop = _resolve_loop(plan, n_samples)
    bpm = plan.get("bpm")
    root_note = plan.get("root_note")
    new: list[tuple[bytes, bytes]] = []

    now = datetime.now(timezone.utc)
    new.append((
        b"bext",
        bext_payload(
            description=plan.get("description", ""),
            originator=ORIGINATOR,
            date=now.strftime("%Y-%m-%d"),
            time=now.strftime("%H:%M:%S"),
        ),
    ))
    if bpm is not None or root_note is not None or loop is not None:
        beats = beat_count(plan, sr=sr, n_samples=n_samples)
        new.append((
            b"acid",
            acid_payload(
                bpm=bpm,
                root_note=root_note,
                num_beats=int(round(beats)) if beats is not None else 0,
                meter=plan.get("time_signature"),
                # Explicit loop points, or a grid-cut length on a tracked grid. A one-shot
                # flag would make the tempo above inert — hosts don't warp one-shots.
                looping=loop is not None or is_grid_cut(plan, sr=sr, n_samples=n_samples),
            ),
        ))
    if root_note is not None or loop is not None:
        new.append((b"smpl", smpl_payload(sr=sr, root_note=root_note or ROOT_NOTE_C, loop=loop)))
    if plan.get("cue"):
        new.append((b"cue ", cue_payload(plan["cue"])))

    chunks = [(cid, data) for cid, data in parse_riff(path.read_bytes())
              if cid not in _OWNED_CHUNKS]
    at = next((i for i, (cid, _) in enumerate(chunks) if cid == b"data"), len(chunks))
    path.write_bytes(_build_riff(chunks[:at] + new + chunks[at:]))
    return [cid.decode().strip() for cid, _ in new]


# ---------------------------------------------------------------------------
# FLAC metadata-block surgery (the Vorbis-comment mirror).
# ---------------------------------------------------------------------------
def parse_flac(raw: bytes) -> tuple[list[tuple[int, bytes]], bytes]:
    """``([(block_type, payload), ...], audio_tail)``. Also the tests' reader."""
    if raw[:4] != b"fLaC":
        raise ValueError("not a FLAC container")
    blocks: list[tuple[int, bytes]] = []
    pos = 4
    while pos + 4 <= len(raw):
        header = raw[pos]
        size = int.from_bytes(raw[pos + 1 : pos + 4], "big")
        blocks.append((header & 0x7F, raw[pos + 4 : pos + 4 + size]))
        pos += 4 + size
        if header & 0x80:  # last-metadata-block flag
            break
    return blocks, raw[pos:]


def _build_flac(blocks: list[tuple[int, bytes]], tail: bytes) -> bytes:
    out = b"fLaC"
    for i, (btype, payload) in enumerate(blocks):
        last = 0x80 if i == len(blocks) - 1 else 0
        out += bytes([last | btype]) + len(payload).to_bytes(3, "big") + payload
    return out + tail


def parse_vorbis_comment(payload: bytes) -> tuple[bytes, list[str]]:
    """``(vendor, ["KEY=value", ...])`` from a VORBIS_COMMENT block. Also the tests' reader."""
    vendor_len = struct.unpack_from("<I", payload, 0)[0]
    vendor = payload[4 : 4 + vendor_len]
    pos = 4 + vendor_len
    count = struct.unpack_from("<I", payload, pos)[0]
    pos += 4
    items: list[str] = []
    for _ in range(count):
        size = struct.unpack_from("<I", payload, pos)[0]
        pos += 4
        items.append(payload[pos : pos + size].decode("utf-8", "replace"))
        pos += size
    return vendor, items


def _build_vorbis_comment(vendor: bytes, items: list[str]) -> bytes:
    payload = struct.pack("<I", len(vendor)) + vendor + struct.pack("<I", len(items))
    for item in items:
        raw = item.encode("utf-8")
        payload += struct.pack("<I", len(raw)) + raw
    return payload


def vorbis_tags(plan: dict, *, n_samples: int) -> list[tuple[str, str]]:
    """The tempo/key/loop mirror for FLAC — only what the stream carries."""
    tags: list[tuple[str, str]] = []
    if plan.get("bpm") is not None:
        tags.append(("BPM", f"{plan['bpm']:g}"))
    if plan.get("key"):
        scale = plan.get("scale")
        tags.append(("KEY", f"{plan['key']} {scale}" if scale else plan["key"]))
    loop = _resolve_loop(plan, n_samples)
    if loop is not None:
        tags.append(("LOOP_START", str(loop[0])))
        tags.append(("LOOP_END", str(loop[1])))
    return tags


def embed_flac(path: Path, plan: dict, *, sr: int, n_samples: int) -> list[str]:
    """Merge the tempo/key/loop tags into the FLAC's VORBIS_COMMENT block."""
    tags = vorbis_tags(plan, n_samples=n_samples)
    if not tags:
        return []
    blocks, tail = parse_flac(path.read_bytes())

    at = next((i for i, (btype, _) in enumerate(blocks) if btype == _VORBIS_COMMENT_BLOCK), None)
    if at is None:
        vendor, items = ORIGINATOR.encode(), []
        at = min(1, len(blocks))  # right after STREAMINFO
        blocks.insert(at, (_VORBIS_COMMENT_BLOCK, b""))
    else:
        vendor, items = parse_vorbis_comment(blocks[at][1])

    ours = {name for name, _ in tags}
    kept = [i for i in items if i.split("=", 1)[0].upper() not in ours]
    merged = kept + [f"{name}={value}" for name, value in tags]
    blocks[at] = (_VORBIS_COMMENT_BLOCK, _build_vorbis_comment(vendor, merged))
    path.write_bytes(_build_flac(blocks, tail))
    return [name for name, _ in tags]


def embed(path: Path, plan: dict, *, sr: int, n_samples: int) -> list[str]:
    """Embed ``plan`` into the written file. Containers we can't carry metadata in → ``[]``."""
    if not plan:
        return []
    suffix = path.suffix.lower()
    if suffix in (".wav", ".wave"):
        return embed_wav(path, plan, sr=sr, n_samples=n_samples)
    if suffix == ".flac":
        return embed_flac(path, plan, sr=sr, n_samples=n_samples)
    return []
