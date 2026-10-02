"""`smpl groove` — lift the swing + micro-timing off a reference loop (vault-6m62).

The counterpart to ``smpl pattern``: where `pattern` *writes* a groove (global ``swing``
plus per-hit ``nudge``), this *reads* one off a reference. Onsets are detected, fitted to
the same step grid `pattern` places hits on, and the deviation of each hit is reported —
decomposed into the one global ``swing`` value `pattern` accepts and the per-step ``nudge``
residuals it cannot express. That is how a generated loop borrows a reference's feel
instead of landing on a dead grid.

Passes every input frame through unchanged, then for each selected `audio` frame appends
a `marker` frame (role ``groove``, one point per onset, labelled with the grid step it was
assigned to) plus two `feature` frames: role ``groove``, whose ``data`` IS the
pattern-consumable groove document, and role ``groove-features`` with ``rhythm.swing``,
``rhythm.swing_confidence`` and ``rhythm.microtiming_beats``.

Round-trip::

    smpl read ref.wav | smpl groove --bpm 134 --out groove.json
    # merge groove.json's `swing` + per-step `nudge` into your pattern DSL, then
    smpl pattern --pattern-file loop.json --out loop.smplset.json

Thin wrapper: the analysis lives in `smpl_analysis.groove` (which documents the swing
search and the origin alignment). Heavy imports stay inside `run()`.
"""

from __future__ import annotations

import json

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "extract swing + per-step nudge (groove) from a reference loop"


def add_arguments(parser):
    add_selection_args(parser)
    parser.add_argument("--bpm", type=float, default=None,
                        help="tempo the grid is measured against (default: estimate it)")
    parser.add_argument("--grid-steps", type=int, default=16,
                        help="steps per bar, matching the pattern grid (default 16)")
    parser.add_argument("--beats-per-bar", type=float, default=4.0,
                        help="beats per bar (default 4)")
    parser.add_argument("--hop-length", type=int, default=128,
                        help="onset-detection hop in samples; the timing floor (default 128)")
    parser.add_argument("--out",
                        help="also write the groove document here as JSON (last frame wins)")


def run(args) -> int:
    from smplstream import error_frame, select as S

    inframes = read_stdin_frames()
    out = list(inframes)  # passthrough first (spec: passthrough before derived)

    audios = S.select(inframes, kind="audio", role=args.role, mode=selection_mode(args))
    if not audios and inframes:
        audios = S.select(inframes, kind="audio", mode="all")

    from smpl_analysis import groove as _groove

    rc = 0
    document = None
    for audio in audios:
        try:
            derived = _groove.groove_audio_frame(
                audio,
                bpm=args.bpm,
                beats_per_bar=args.beats_per_bar,
                grid_steps=args.grid_steps,
                hop_length=args.hop_length,
            )
            out.extend(derived)
            document = next(
                (f["data"] for f in derived
                 if f.get("kind") == "feature" and f.get("role") == "groove"),
                document,
            )
        except Exception as exc:
            eprint(f"groove: {audio.get('id')}: {exc}")
            out.append(error_frame("op_failed", str(exc), of=audio.get("id"), op="groove"))
            rc = 1

    if args.out and document is not None:
        try:
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump(document, fh, indent=2)
                fh.write("\n")
            eprint(f"groove: wrote {args.out}")
        except Exception as exc:
            eprint(f"groove: could not write {args.out}: {exc}")
            rc = 1

    emit(out)
    return rc
