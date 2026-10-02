"""`smpl qc` — technical QC / defect detection as a filter (research §4; ticket vault-1e9a).

Passes every input frame through, then appends derived QC frames (one `feature` with the
deterministic top-6 scalars + `marker` frames for click/gap locations) for each audio frame.
The DSP lives in `smpl_analysis.qc`; this module is a thin selection-and-passthrough shell.
"""

from __future__ import annotations

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "QC audio frame(s) as a filter: passthrough + clipping/phase/DC/SNR/lossy + defect markers"


def add_arguments(parser):
    add_selection_args(parser)
    parser.add_argument(
        "--no-markers",
        action="store_true",
        help="skip click/gap defect marker frames (emit only the feature frame)",
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

    try:
        from smpl_analysis import qc as _qc
    except Exception as exc:  # the analysis tier must be installed for qc
        eprint(f"qc: analysis library unavailable: {exc}")
        emit(out)
        return 1

    rc = 0
    for audio in audios:
        try:
            out.extend(
                _qc.qc_audio_frame(
                    audio,
                    want_markers=not args.no_markers,
                    use_cache=not args.no_cache,
                )
            )
        except Exception as exc:
            eprint(f"qc: {audio.get('id')}: {exc}")
            out.append(error_frame("op_failed", str(exc), of=audio.get("id"), op="qc"))
            rc = 1
    emit(out)
    return rc
