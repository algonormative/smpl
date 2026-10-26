"""Shared plumbing for the pack ops over embedding vectors (`cohesion`, `coverage`, `probe`).

Each op reads a stream that already went through `smpl embed`, passes it through, and scores
the audio vectors of ONE model. These helpers resolve that model, collect the items (optionally
only those whose embedded audio frame has ``--role``), and turn the known failure modes into
error frames instead of tracebacks.
"""

from __future__ import annotations


def add_vector_args(parser) -> None:
    parser.add_argument("--model", default=None,
                        help="embedding model whose vectors to use (required when the stream "
                             "carries vectors from several models)")
    parser.add_argument("--role", default=None,
                        help="only score vectors whose embedded audio frame has this role")


def collect(frames: list[dict], args):
    """→ (model, items). Raises ValueError (incl. AmbiguousModel) with a usable message."""
    from smpl_analysis import vectors as V

    model = V.pick_model(frames, getattr(args, "model", None))
    items = V.audio_items(frames, model)
    role = getattr(args, "role", None)
    if role is not None:
        by_id = {f.get("id"): f for f in frames}
        items = [it for it in items if (by_id.get(it.id) or {}).get("role") == role]
    if not items:
        raise ValueError(f"no audio vectors from model {model!r}"
                         + (f" with role {role!r}" if role else ""))
    return model, items


def read_ndjson(path: str) -> list[dict]:
    from smplstream import ndjson

    with open(path, "rb") as fh:
        return list(ndjson.read_frames(fh))
