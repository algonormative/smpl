"""`smpl spectrogram` — render annotated spectrogram / waveform PNGs as a filter.

Passes every input frame through unchanged, then appends `image` frames (one per
selected audio frame per requested render kind). Each PNG is CAS-stored and referenced
by role ``spectrogram:mel`` / ``spectrogram:cqt`` / ``spectrogram:hpss`` / ``waveform``.
The rendering lives in ``smpl_analysis.spectrogram``; this module is a thin shim.
"""

from __future__ import annotations

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "render mel/cqt/hpss spectrogram + waveform PNG image frames for audio frame(s)"

_KINDS = ("mel", "cqt", "hpss", "waveform")


def add_arguments(parser):
    add_selection_args(parser)
    parser.add_argument(
        "--kind",
        action="append",
        choices=_KINDS,
        help="which render(s) to produce; repeatable (default: mel)",
    )
    parser.add_argument(
        "--all-kinds",
        action="store_true",
        help="render every kind (mel, cqt, hpss, waveform)",
    )
    parser.add_argument("--no-cache", action="store_true",
                        help="bypass the memo cache: force recompute and refresh the entry")


def run(args) -> int:
    from smplstream import error_frame, select as S

    inframes = read_stdin_frames()
    out = list(inframes)  # passthrough first (spec: passthrough before derived)

    audios = S.select(inframes, kind="audio", role=args.role, mode=selection_mode(args))
    if not audios and inframes:
        audios = S.select(inframes, kind="audio", mode="all")

    if args.all_kinds:
        kinds = list(_KINDS)
    else:
        kinds = args.kind or ["mel"]

    try:
        from smpl_analysis import spectrogram as _spec
    except Exception as exc:  # analysis tier not installed
        eprint(f"spectrogram: analysis library unavailable: {exc}")
        emit(out)
        return 1

    rc = 0
    for audio in audios:
        try:
            out.extend(
                _spec.render_audio_frame(audio, kinds=kinds, use_cache=not args.no_cache)
            )
        except Exception as exc:
            eprint(f"spectrogram: {audio.get('id')}: {exc}")
            out.append(error_frame("op_failed", str(exc), of=audio.get("id"), op="spectrogram"))
            rc = 1
    emit(out)
    return rc
