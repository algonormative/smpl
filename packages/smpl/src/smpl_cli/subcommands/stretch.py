"""`smpl stretch` — time-stretch at constant pitch (phase vocoder) or extreme paulstretch.

Two modes, one subcommand:
  - default: ``--ratio`` phase-vocoder stretch (transient-preserving, musical ratios).
  - ``--paul``: ``--factor`` paulstretch (phase-randomised) for extreme stretches into
    ambience, where a phase vocoder would only produce metallic warble.
"""

from __future__ import annotations

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "time-stretch by --ratio (phase vocoder) or --paul --factor (paulstretch); emits <role>.wet"


def add_arguments(parser):
    add_selection_args(parser)
    parser.add_argument("--ratio", type=float, help="stretch factor (>1 slower); required unless --paul")
    parser.add_argument("--paul", action="store_true",
                        help="paulstretch mode (phase-randomised, for extreme stretches)")
    parser.add_argument("--factor", type=float,
                        help="paulstretch length multiplier (>1 longer); required with --paul")
    parser.add_argument("--window-s", type=float, default=0.28,
                        help="paulstretch window in seconds (default 0.28)")
    parser.add_argument("--stereo-decorrelate", action="store_true",
                        help="paulstretch: skip the partial L/R re-blend, leaving the channels fully decorrelated")


def run(args) -> int:
    from smplstream import error_frame, select as S

    if args.paul:
        if args.factor is None:
            eprint("stretch: --paul requires --factor"); return 2
        if args.ratio is not None:
            eprint("stretch: --ratio is for the phase-vocoder mode; use --factor with --paul"); return 2
    else:
        if args.factor is not None:
            eprint("stretch: --factor is paulstretch-only; pass --paul (or use --ratio)"); return 2
        if args.ratio is None:
            eprint("stretch: --ratio is required (or use --paul --factor)"); return 2

    inframes = read_stdin_frames()
    out = list(inframes)
    audios = S.select(inframes, kind="audio", role=args.role, mode=selection_mode(args))
    if not audios and inframes:
        audios = S.select(inframes, kind="audio", mode="all")

    from smpl_analysis import edit

    rc = 0
    for audio in audios:
        try:
            if args.paul:
                out.append(edit.apply_paulstretch(
                    audio, factor=args.factor, window_s=args.window_s,
                    stereo_decorrelate=args.stereo_decorrelate,
                ))
            else:
                out.append(edit.apply_stretch(audio, ratio=args.ratio))
        except Exception as exc:
            eprint(f"stretch: {audio.get('id')}: {exc}")
            out.append(error_frame("op_failed", str(exc), of=audio.get("id"), op="stretch"))
            rc = 1
    emit(out)
    return rc
