"""`smpl cohesion` — how tightly a pack hangs together in an embedding space (vault-16dd.5).

Reads a stream that went through `smpl embed` and scores the audio vectors of one model:
per item, cosine distance to the pack centroid and its robust z (outlier flag above
``--z-threshold``); per pack, the mean pairwise cosine distance and, with ``--ref``, the
Kernel Audio Distance (KAD, Chung et al. 2025) against a reference set of vectors.

    smpl read pack/*.wav | smpl embed --model clap | smpl cohesion
    smpl read new/*.wav | smpl embed --model clap \\
      | smpl cohesion --ref old-kicks.ndjson          # KAD vs a reference pack

``--ref`` is an NDJSON file of frames carrying vectors from the same model (e.g. the saved
output of `smpl embed` over the reference pack); its CAS blobs must be in the local CAS.

Passes every input frame through, then appends a `feature` frame per item (role
``cohesion``) and one pack summary (role ``cohesion-pack``). The math lives in
`smpl_analysis.cohesion`; no torch here.
"""

from __future__ import annotations

from .._common import emit, eprint, read_stdin_frames
from .._vectors import add_vector_args, collect, read_ndjson

HELP = "pack cohesion over embedding vectors: centroid distance, outlier z, KAD vs a reference"


def add_arguments(parser):
    add_vector_args(parser)
    parser.add_argument("--ref", default=None,
                        help="NDJSON file of reference vector frames (same model) for KAD")
    parser.add_argument("--z-threshold", type=float, default=3.5,
                        help="flag items whose robust z exceeds this (default 3.5)")


def run(args) -> int:
    from smplstream import error_frame

    inframes = read_stdin_frames()
    out = list(inframes)
    try:
        from smpl_analysis import cohesion as C
        from smpl_analysis import vectors as V

        model, items = collect(inframes, args)
        ref_items = None
        if args.ref:
            ref_frames = read_ndjson(args.ref)
            ref_items = V.audio_items(ref_frames, model)
            if len(ref_items) < 2:
                raise ValueError(f"--ref {args.ref}: {len(ref_items)} vectors from {model!r} "
                                 "(KAD needs >= 2)")
        out.extend(C.cohesion_frames(items, model=model, ref_items=ref_items,
                                     z_threshold=args.z_threshold, ref_label=args.ref))
        rc = 0
    except Exception as exc:
        eprint(f"cohesion: {exc}")
        out.append(error_frame("op_failed", str(exc), op="cohesion"))
        rc = 1
    emit(out)
    return rc
