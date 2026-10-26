"""Intent: what a generator meant a clip to be (vault-16dd.2).

The checks that held up on generated kits compared output to the generator's own intent, not
to detectors: recipe hit times beat onset detection (detector medians read 15-49 ms on plain
kick stems), and the pitch classes a build actually played beat chroma (real voices read
0.60-0.70 white share from harmonics alone). This module turns whatever a generator left
behind into one normalized ``intent`` dict that `conform`, `sheet`, `clash` and `groove-feel`
read. It measures nothing.

Sources, first found wins:

1. an explicit file (``--intent FILE``),
2. ``<audio>.intent.json`` next to the source file,
3. ``<audio>.provenance.json`` (the smplvst / Assembly sidecar: ``recipe``, ``bpm``, ``bars``,
   ``pitch_classes``).

The normalized shape (every field optional; absent means "no intent stated")::

    {
      "v": 1,
      "bpm": 120.0, "beats_per_bar": 4.0, "bars": 4, "grid_steps": 16, "swing": 0.0,
      "dur_s": 8.0,                        # bars x beats_per_bar x 60 / bpm when all three are stated
      "hits": [{"t": 0.0, "velocity": 1.0, "track": "kick", "bar": 1, "step": 1}],  # seconds from
                                           # sample 0; bar/step (1-based) only when the source states them
      "pitch_classes": ["C", "D", "E"],    # sharps spelled "#"
      "key": "A minor",
      "lufs": -14.0, "true_peak_db": -1.0,
      "source": "/path/to/the/sidecar.json"
    }

An ``.intent.json`` file may give ``hits`` directly (seconds) or a ``recipe`` in the
`smpl pattern` DSL, which is expanded the way `smpl pattern` places hits: even steps late by
``swing x step``, plus ``nudge`` in beats.
"""

from __future__ import annotations

import json
from pathlib import Path

INTENT_VERSION = 1

_FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#",
                  "C♯": "C#", "D♯": "D#", "F♯": "F#", "G♯": "G#", "A♯": "A#",
                  "D♭": "C#", "E♭": "D#", "G♭": "F#", "A♭": "G#", "B♭": "A#"}
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def pitch_class(name: str) -> str:
    """Spell a pitch-class name with sharps ("Bb" -> "A#"); raises on anything else."""
    pc = _FLAT_TO_SHARP.get(name.strip(), name.strip())
    if pc not in PITCH_CLASSES:
        raise ValueError(f"not a pitch class: {name!r}")
    return pc


def recipe_hits(recipe: dict, bpm: float | None = None) -> list[dict]:
    """Every hit a `smpl pattern` recipe places, in seconds, sorted by time."""
    bpm = float(bpm or recipe["bpm"])
    beats = float(recipe.get("beats_per_bar", 4))
    beat_s, step_b = 60.0 / bpm, beats / recipe["grid_steps"]
    out = []
    for track in recipe.get("tracks", []):
        for h in track.get("hits", []):
            swing = h.get("swing", track.get("swing", recipe.get("swing", 0.0)))
            b = (h["bar"] - 1) * beats + (h["step"] - 1) * step_b
            b += swing * step_b if h["step"] % 2 == 0 else 0.0
            out.append({"t": round((b + h.get("nudge", 0.0)) * beat_s, 6),
                        "velocity": float(h.get("velocity", 1.0)),
                        "track": track.get("name"), "bar": int(h["bar"]), "step": int(h["step"])})
    return sorted(out, key=lambda h: h["t"])


def normalize(doc: dict, source: str | None = None) -> dict:
    """Normalize an ``.intent.json`` or ``.provenance.json`` document to the intent shape."""
    recipe = doc.get("recipe") if isinstance(doc.get("recipe"), dict) else {}

    def pick(key):  # the document's own value, else the recipe's (provenance stores null for "none")
        return doc[key] if doc.get(key) is not None else recipe.get(key)

    out: dict = {"v": INTENT_VERSION}
    bpm = pick("bpm")
    if bpm is not None:
        out["bpm"] = float(bpm)
    beats = pick("beats_per_bar")
    if beats is None and "5/4" in (doc.get("note") or ""):
        beats = 5  # Assembly provenance states its one odd meter only in the note
    if beats is not None:  # never assumed: a 5/4 clip read as 4/4 would fail conform's length check
        out["beats_per_bar"] = float(beats)
    bars = pick("bars")
    if bars is not None:
        out["bars"] = int(bars)
    for key in ("grid_steps", "swing"):
        val = pick(key)
        if val is not None:
            out[key] = val
    if doc.get("dur_s") is not None:
        out["dur_s"] = float(doc["dur_s"])
    elif "bpm" in out and "bars" in out and "beats_per_bar" in out:
        out["dur_s"] = round(out["bars"] * out["beats_per_bar"] * 60.0 / out["bpm"], 6)
    if isinstance(doc.get("hits"), list):
        out["hits"] = sorted(({"t": float(h["t"]), "velocity": float(h.get("velocity", 1.0)),
                               "track": h.get("track"),
                               **{k: int(h[k]) for k in ("bar", "step") if h.get(k) is not None}}
                              for h in doc["hits"]), key=lambda h: h["t"])
    elif recipe and "grid_steps" in recipe and recipe.get("tracks"):
        out["hits"] = recipe_hits(recipe, out.get("bpm"))
    if doc.get("pitch_classes"):
        out["pitch_classes"] = sorted({pitch_class(p) for p in doc["pitch_classes"]},
                                      key=PITCH_CLASSES.index)
    if doc.get("key"):
        out["key"] = doc["key"]
    for key in ("lufs", "true_peak_db"):
        if doc.get(key) is not None:
            out[key] = float(doc[key])
    if source:
        out["source"] = str(source)
    return out


def load(path: str | Path) -> dict:
    p = Path(path).expanduser()
    return normalize(json.loads(p.read_text()), source=str(p))


def sidecar_for(audio_path: str | Path) -> Path | None:
    """The ``.intent.json`` or ``.provenance.json`` next to an audio file, if any."""
    for suffix in (".intent.json", ".provenance.json"):
        cand = Path(str(audio_path) + suffix)
        if cand.is_file():
            return cand
    return None


def for_audio_frame(frame: dict, explicit: str | None = None) -> dict | None:
    """Intent for an `audio` frame: ``explicit`` if given, else a sidecar of its read source.

    Only frames straight from ``smpl read`` carry ``params.source``; a derived frame (filtered,
    sliced) has no sidecar, and gets intent only through ``explicit``.
    """
    if explicit:
        return load(explicit)
    src = (frame.get("params") or {}).get("source")
    side = sidecar_for(src) if src else None
    return load(side) if side else None
