"""`smpl coverage` — does a pack hold the families a brief requires, in quota? (vault-16dd.5)

    smpl read pack/*.wav | smpl embed --model clap | smpl coverage --brief kit.coverage.json

The brief names families with a ``quota`` (minimum), optional ``max``, and how to recognise
them: ``text`` (one prompt or a list, pooled) for embedding assignment, ``tag`` (regex over
the source file stem) for label assignment. ``--mode text`` assigns each item to its nearest
family text pole by cosine; ``--mode tag`` by the first matching tag. Default: text when every
family has text, else tag. Format and semantics: `smpl_analysis.coverage`.

Text vectors come from the stream (`smpl embed text …` upstream) or, for prompts the stream
lacks, from running ``smpl-embed text`` (the heavy tier); without it, text mode degrades to an
``unsupported`` error frame.

Passes every input frame through, then any text-vector frames it minted, a `feature` frame per
item (role ``coverage``: family, score, margin) and one pack summary (role ``coverage-pack``:
count / quota / max / delta per family, missing, over, unassigned).
"""

from __future__ import annotations

from .._common import emit, eprint, read_stdin_frames
from .._vectors import add_vector_args, collect

HELP = "count pack items per required family (text or tag assignment) against brief quotas"


def add_arguments(parser):
    add_vector_args(parser)
    parser.add_argument("--brief", required=True, help="coverage brief JSON ({families: {…}})")
    parser.add_argument("--mode", choices=["text", "tag"], default=None,
                        help="assignment: nearest family text pole, or first matching tag")


def run(args) -> int:
    from smpl_analysis import coverage as CV
    from smpl_analysis import vectors as V
    from smplstream import error_frame

    inframes = read_stdin_frames()
    out = list(inframes)
    try:
        brief = CV.load_brief(args.brief)
        if args.model is None and brief.get("model"):
            args.model = brief["model"]
        model, items = collect(inframes, args)
        mode = args.mode or CV.default_mode(brief)
        text_vecs = None
        if mode == "text":
            prompts = [p for f in brief["families"].values() for p in f["text"]]
            if not all(f["text"] for f in brief["families"].values()):
                raise ValueError("text mode needs `text` on every family")
            text_vecs, minted = V.text_vectors(prompts, model, inframes)
            out.extend(minted)
        out.extend(CV.coverage_frames(items, brief, model=model, mode=mode,
                                      text_vecs=text_vecs, brief_label=args.brief))
        rc = 0
    except V.TextVectorsUnavailable as exc:
        eprint(f"coverage: {exc}")
        out.append(error_frame("unsupported", str(exc), op="coverage"))
        rc = 0
    except Exception as exc:
        eprint(f"coverage: {exc}")
        out.append(error_frame("op_failed", str(exc), op="coverage"))
        rc = 1
    emit(out)
    return rc
