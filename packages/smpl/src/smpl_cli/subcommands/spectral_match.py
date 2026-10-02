"""`smpl spectral-match` — EQ the selected audio toward a reference's spectral balance.

The reference-replication linchpin: measure ``--reference``'s average per-band spectrum,
measure the source's, and apply a bounded corrective peaking-EQ chain that moves the source's
*balance* toward the reference's. Matches SHAPE, not level (both curves are mean-normalized
first — leave loudness to ``smpl normalize``; ``--anchor-hz`` pivots them on one band instead,
so a dark source's low end is held rather than cut). Passthrough every input frame first, then append
one wet `audio` frame (role ``<role>.wet``, ``op: spectral-match``) per selected audio frame.
The full corrective curve is recorded in the wet frame's ``params``, alongside the A/B spectra
(``spectrum_db``: source before, reference, matched after) and the ``residual_db`` distance to
the reference before vs after. DSP in ``smpl_analysis.edit``.

  smpl read loop.wav | smpl spectral-match --reference ref.wav --strength 0.8 | smpl normalize --lufs -12 | smpl write matched.wav
"""

from __future__ import annotations

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "EQ the selected audio toward a reference's spectral balance (emits <role>.wet)"


def add_arguments(parser):
    add_selection_args(parser)
    parser.add_argument("--reference", required=True,
                        help="reference audio file whose spectral balance to match toward")
    parser.add_argument("--strength", type=float, default=1.0,
                        help="0..1 fraction of the correction to apply (default 1.0)")
    parser.add_argument("--max-correction", type=float, default=6.0,
                        help="per-band gain clamp in dB (default 6)")
    parser.add_argument("--bands", type=int, default=12,
                        help="number of log-spaced match bands (default 12)")
    parser.add_argument("--lo", type=float, default=30.0, help="low edge of the band grid, Hz (default 30)")
    parser.add_argument("--hi", type=float, default=16000.0, help="high edge of the band grid, Hz (default 16000)")
    parser.add_argument("--protect-below", type=float, default=60.0,
                        help="leave bands centered below this Hz at 0 dB — protect the sub (default 60)")
    parser.add_argument("--anchor-hz", type=float, default=None,
                        help="pivot-align both curves at the band containing this Hz (reads 0 dB "
                             "in both) instead of subtracting each curve's mean — holds a dark "
                             "source's low end and lifts only upward (default: mean)")


def run(args) -> int:
    from smplstream import error_frame, select as S

    inframes = read_stdin_frames()
    out = list(inframes)

    audios = S.select(inframes, kind="audio", role=args.role, mode=selection_mode(args))
    if not audios and inframes:
        audios = S.select(inframes, kind="audio", mode="all")

    from smpl_analysis import edit

    rc = 0
    for audio in audios:
        try:
            out.append(edit.apply_spectral_match(
                audio,
                reference_path=args.reference,
                strength=args.strength,
                max_correction_db=args.max_correction,
                n_bands=args.bands,
                lo_hz=args.lo,
                hi_hz=args.hi,
                protect_below_hz=args.protect_below,
                anchor_hz=args.anchor_hz,
            ))
        except Exception as exc:
            eprint(f"spectral-match: {audio.get('id')}: {exc}")
            out.append(error_frame("op_failed", str(exc), of=audio.get("id"), op="spectral-match"))
            rc = 1
    emit(out)
    return rc
