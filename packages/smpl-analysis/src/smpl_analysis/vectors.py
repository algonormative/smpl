"""Embedding vectors from a frame stream: the shared substrate of `cohesion`, `coverage`, `probe`.

`smpl embed` (tools/smpl-embed, the heavy tier) emits `vector` frames: role
``ml.<backend>.embedding`` for audio (``of`` → the audio frame) and
``ml.<backend>.text-embedding`` for a text prompt (``params.text``), with the model id in
``meta.model`` / ``params.model``. This module turns those frames into numpy arrays with
labels, so the pack ops stay deterministic numpy with no torch anywhere:

- :func:`vector_values` reads an inline ``data`` vector or a CAS ``.npy`` (never pickle).
- :func:`audio_items` / :func:`text_vectors` split the stream into the two kinds, one model at
  a time (vectors from different models live in different spaces and never mix).
- :func:`text_vectors` falls back to running ``smpl-embed text`` when a prompt has no vector in
  the stream yet; a missing heavy tool raises :class:`TextVectorsUnavailable`.
- :func:`pole` pools a prompt set the vault-3hvg way: L2-normalize each, mean, L2-normalize.

Every vector is L2-normalized on the way in, so dot products are cosines.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


class AmbiguousModel(ValueError):
    """The stream carries vectors from more than one model and none was chosen."""


class TextVectorsUnavailable(RuntimeError):
    """A prompt has no text vector in the stream and `smpl-embed text` could not supply one."""


@dataclass
class Item:
    """One audio embedding: the item a pack op scores."""

    id: str          # the item's id: the audio frame the vector embeds (`of`), else the vector's
    vector_id: str   # the vector frame's own id (lineage)
    label: str       # source file stem, else role, else a short id
    model: str
    vec: np.ndarray  # L2-normalized float64


def l2(v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(a))
    if not np.isfinite(n) or n == 0.0:
        raise ValueError("degenerate vector (zero or non-finite norm)")
    return a / n


def vector_values(frame: dict) -> np.ndarray:
    """A `vector` frame's values as float64: inline ``data``, or the CAS ``.npy`` by ``hash``."""
    if frame.get("data") is not None:
        return np.asarray(frame["data"], dtype=np.float64).reshape(-1)
    from smplstream import cas

    path = cas.get_path(frame["hash"])  # FileNotFoundError if the blob is not in this CAS
    return np.asarray(np.load(str(path), allow_pickle=False), dtype=np.float64).reshape(-1)


def model_of(frame: dict) -> str | None:
    return (frame.get("meta") or {}).get("model") or (frame.get("params") or {}).get("model")


def is_text_vector(frame: dict) -> bool:
    return (frame.get("params") or {}).get("text") is not None or str(
        frame.get("role") or "").endswith("text-embedding")


def _vectors(frames: Iterable[dict]) -> list[dict]:
    return [f for f in frames if f.get("kind") == "vector"]


def pick_model(frames: Iterable[dict], model: str | None = None) -> str:
    """The one model the audio vectors in ``frames`` come from (or ``model`` if given)."""
    models = sorted({model_of(f) for f in _vectors(frames) if not is_text_vector(f)} - {None})
    if model:
        if model not in models:
            raise ValueError(f"no audio vectors from model {model!r} (stream has {models})")
        return model
    if not models:
        raise ValueError("no audio vector frames in the stream (pipe through `smpl embed` first)")
    if len(models) > 1:
        raise AmbiguousModel(f"vectors from {len(models)} models {models}; choose one with --model")
    return models[0]


def _label(frame: dict, by_id: dict) -> str:
    src = by_id.get(frame.get("of"))
    path = ((src or {}).get("params") or {}).get("source")
    if path:
        return os.path.splitext(os.path.basename(str(path)))[0]
    if src and src.get("role"):
        return str(src["role"])
    return str(frame.get("of") or frame.get("id") or "?").removeprefix("blake3:")[:16]


def audio_items(frames: list[dict], model: str) -> list[Item]:
    """Every audio vector from ``model``, labelled, de-duplicated by the audio it embeds."""
    by_id = {f.get("id"): f for f in frames if f.get("id")}
    items: dict[str, Item] = {}
    for f in _vectors(frames):
        if is_text_vector(f) or model_of(f) != model:
            continue
        key = f.get("of") or f.get("id")
        items[key] = Item(id=key, vector_id=f.get("id"), label=_label(f, by_id), model=model,
                          vec=l2(vector_values(f)))
    return list(items.values())


def matrix(items: list[Item]) -> np.ndarray:
    return np.stack([it.vec for it in items]) if items else np.zeros((0, 0))


def _embed_text_subprocess(prompts: list[str], model: str) -> list[dict]:
    """Run ``smpl-embed text`` for the prompts; its vector frames land in the same CAS."""
    import io
    import shutil
    import subprocess

    from smplstream import ndjson

    exe = shutil.which("smpl-embed")
    if exe is None:
        raise TextVectorsUnavailable(
            "smpl-embed is not on PATH (install tools/smpl-embed with the [torch] extra), and the "
            "stream carries no text vectors for: " + "; ".join(prompts))
    proc = subprocess.run([exe, "text", *prompts, "--model", model], stdin=subprocess.DEVNULL,
                          capture_output=True, check=False)
    frames = list(ndjson.read_frames(io.BytesIO(proc.stdout)))
    errors = [f for f in frames if f.get("kind") == "error"]
    if proc.returncode != 0 or errors:
        msg = (errors[0].get("data") or {}).get("message") if errors else proc.stderr.decode()[-400:]
        raise TextVectorsUnavailable(f"smpl-embed text failed for model {model!r}: {msg}")
    return frames


def text_vectors(prompts: list[str], model: str, frames: list[dict], *,
                 embed_missing: bool = True) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Map each prompt → its L2-normalized text vector from ``model``.

    Looks in the stream first (`ml.<backend>.text-embedding` frames, matched on
    ``params.text``); prompts still missing are embedded with ``smpl-embed text`` when
    ``embed_missing``. Returns the map plus the newly minted text-vector frames, which the
    caller appends to its output so the stream records what the scores were computed against.
    """
    have: dict[str, np.ndarray] = {}
    for f in _vectors(frames):
        text = (f.get("params") or {}).get("text")
        if text is not None and model_of(f) == model and text in prompts:
            have[text] = l2(vector_values(f))
    missing = [p for p in dict.fromkeys(prompts) if p not in have]
    minted: list[dict] = []
    if missing:
        if not embed_missing:
            raise TextVectorsUnavailable("no text vectors in the stream for: " + "; ".join(missing))
        minted = [f for f in _embed_text_subprocess(missing, model)
                  if f.get("kind") == "vector" and is_text_vector(f)]
        for f in minted:
            have[f["params"]["text"]] = l2(vector_values(f))
    still = [p for p in prompts if p not in have]
    if still:
        raise TextVectorsUnavailable("smpl-embed returned no vector for: " + "; ".join(still))
    return have, minted


def pole(vectors: list[np.ndarray]) -> np.ndarray:
    """Pool a prompt set into one pole (vault-3hvg): L2-normalize each, mean, L2-normalize."""
    return l2(np.mean([l2(v) for v in vectors], axis=0))


def as_prompts(value) -> list[str]:
    """A brief's ``text`` field: one prompt or a list of paraphrases."""
    if value is None:
        return []
    return [value] if isinstance(value, str) else [str(v) for v in value]


def tag_match(pattern: str, label: str) -> bool:
    return re.search(pattern, label, flags=re.IGNORECASE) is not None
