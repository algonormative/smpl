"""`smpl loudness` — BS.1770 loudness as a filter (research §1; ticket vault-3vau).

Passes every input frame through unchanged, then appends, for each selected audio frame,
a `feature` frame carrying:

    loudness.integrated_lufs      (LUFS)   — BS.1770 integrated, K-weighted, gated
    loudness.true_peak_dbtp       (dBTP)   — 4x-oversampled inter-sample true peak
    loudness.max_short_term_lufs  (LUFS)   — peak of the 3 s/1 s short-term curve

and, when inter-sample overs are found above the ceiling, a `marker` frame
(role "true-peak-over") locating them (each point carries `sample:` index).

The work lives in `smpl_analysis.loudness`; this module is a thin shell.
"""

from __future__ import annotations

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "measure BS.1770 loudness (integrated LUFS, true-peak dBTP, max short-term LUFS)"


def add_arguments(parser):
    add_selection_args(parser)
    parser.add_argument("--no-markers", action="store_true",
                        help="skip true-peak-over marker frames")
    parser.add_argument("--over-ceiling-dbtp", type=float, default=None,
                        help="true-peak ceiling for over markers (default: -1.0 dBTP)")
    parser.add_argument("--no-cache", action="store_true",
                        help="bypass the memo cache: force recompute and refresh the entry")


def run(args) -> int:
    from smplstream import error_frame, select as S
    from smpl_analysis import loudness as L

    inframes = read_stdin_frames()
    out = list(inframes)  # passthrough first (spec: passthrough before derived)

    audios = S.select(inframes, kind="audio", role=args.role, mode=selection_mode(args))
    if not audios and inframes:
        audios = S.select(inframes, kind="audio", mode="all")

    ceiling = args.over_ceiling_dbtp
    if ceiling is None:
        ceiling = L.DEFAULT_OVER_CEILING_DBTP

    rc = 0
    for audio in audios:
        try:
            out.extend(
                L.loudness_frames(
                    audio,
                    emit_markers=not args.no_markers,
                    over_ceiling_dbtp=ceiling,
                    use_cache=not args.no_cache,
                )
            )
        except Exception as exc:
            eprint(f"loudness: {audio.get('id')}: {exc}")
            out.append(error_frame("op_failed", str(exc), of=audio.get("id"), op="loudness"))
            rc = 1

    emit(out)
    return rc
