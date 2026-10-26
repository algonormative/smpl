"""`smpl probe "text"` — score samples on a calibrated text axis (vault-16dd.5, vault-3hvg way).

    smpl read a.wav b.wav | smpl embed --model clap \\
      | smpl probe "distorted, clipped, harsh" --neg "clean, pristine, undistorted"

The axis is a contrast, not a raw cosine: positive prompts (the positional text plus any
``--pos``) and negative prompts (``--neg``, default the generic "a sound") are each pooled
(L2-normalize, mean, L2-normalize), the axis is ``normalize(P_pos − P_neg)``, and each item's
score is the projection of its unit audio vector on it (``probe.score_cos``), plus a z within
the scored set (``probe.z``, ≥ 3 items). Only differences between items are meaningful.
Recipe and caveats: `smpl_analysis.probe`.

Text vectors come from the stream (`smpl embed text …`) or from running ``smpl-embed text``
(the heavy tier); without it the op degrades to an ``unsupported`` error frame.

Passes every input frame through, then any text-vector frames it minted, a `feature` frame per
item (role ``probe``) and a ranked summary (role ``probe-pack``).
"""

from __future__ import annotations

from .._common import emit, eprint, read_stdin_frames
from .._vectors import add_vector_args, collect

HELP = "score samples on a contrastive text axis (positive vs negative prompts) in CLAP/MuLan space"


def add_arguments(parser):
    parser.add_argument("text", help="the positive pole prompt")
    parser.add_argument("--pos", action="append", default=[],
                        help="extra positive paraphrase (repeatable; pooled with TEXT)")
    parser.add_argument("--neg", action="append", default=[],
                        help='negative pole prompt (repeatable; default "a sound")')
    add_vector_args(parser)


def run(args) -> int:
    from smpl_analysis import probe as P
    from smpl_analysis import vectors as V
    from smplstream import error_frame

    inframes = read_stdin_frames()
    out = list(inframes)
    try:
        model, items = collect(inframes, args)
        pos = [args.text, *args.pos]
        neg = list(args.neg) or list(P.DEFAULT_NEG)
        text_vecs, minted = V.text_vectors(pos + neg, model, inframes)
        out.extend(minted)
        out.extend(P.probe_frames(items, pos=pos, neg=neg, text_vecs=text_vecs, model=model))
        rc = 0
    except V.TextVectorsUnavailable as exc:
        eprint(f"probe: {exc}")
        out.append(error_frame("unsupported", str(exc), op="probe"))
        rc = 0
    except Exception as exc:
        eprint(f"probe: {exc}")
        out.append(error_frame("op_failed", str(exc), op="probe"))
        rc = 1
    emit(out)
    return rc
