"""`smpl conform` — measured vs intended (vault-16dd.2).

Compares each audio frame to what its generator meant it to be and reports, per check, the
measured value, the intended value and the delta: duration, per-hit timing against the
recipe's hit times, played pitch classes against chroma, integrated LUFS and true peak.
Intent is the normalized document of ``smpl_analysis.intent``: ``--intent FILE``, else
``<audio>.intent.json`` or ``<audio>.provenance.json`` next to the file ``smpl read`` read.
A check with no stated intent reports ``None``; nothing is guessed.

Tolerances are flags, and they decide nothing but which hits get a marker and which checks
are listed in ``conform.over_tol``. No keep/cut call is made here.

Passes every input frame through unchanged, then for each selected `audio` frame appends a
`feature` frame (role ``conform``, the ``conform.*`` keys) and, if any intended hit missed by
more than ``--timing-tol-ms``, a `marker` frame (role ``conform``, one point per miss).

    smpl read kick.wav | smpl conform --allowed-pcs A,B,C,D,E,F,G

Thin wrapper: the checks, and why each is built the way it is, live in
`smpl_analysis.conform`. Heavy imports stay inside `run()`.
"""

from __future__ import annotations

from .._common import (
    add_selection_args,
    emit,
    eprint,
    read_stdin_frames,
    selection_mode,
)

HELP = "measured vs intended: length, hit timing, pitch classes, loudness"


def add_arguments(parser):
    add_selection_args(parser)
    parser.add_argument("--intent", help="intent file (default: <audio>.intent.json / "
                                         ".provenance.json next to the source)")
    parser.add_argument("--length-tol-ms", type=float, default=1.0,
                        help="length delta above which 'length' is listed in over_tol (default 1)")
    parser.add_argument("--timing-tol-ms", type=float, default=8.0,
                        help="hit error above which a hit gets a marker; a median above it lists "
                             "'timing' in over_tol (default 8, the drum-stem threshold)")
    parser.add_argument("--pc-flag-share", type=float, default=0.20,
                        help="chroma share above which an off-set pitch class is flagged (default 0.20)")
    parser.add_argument("--allowed-pcs",
                        help="comma list of allowed pitch classes; intended classes outside it are "
                             "listed in conform.pc_outside_allowed (e.g. A,B,C,D,E,F,G)")
    parser.add_argument("--lufs", type=float, default=None,
                        help="intended integrated LUFS (overrides the intent's)")
    parser.add_argument("--true-peak-db", type=float, default=None,
                        help="intended true peak in dBTP (overrides the intent's)")
    parser.add_argument("--checks", default="length,timing,pitch,loudness",
                        help="comma list of checks to run (default: all four)")


def run(args) -> int:
    from smplstream import error_frame
    from smplstream import select as S

    inframes = read_stdin_frames()
    out = list(inframes)  # passthrough first (spec: passthrough before derived)

    audios = S.select(inframes, kind="audio", role=args.role, mode=selection_mode(args))
    if not audios and inframes:
        audios = S.select(inframes, kind="audio", mode="all")

    from smpl_analysis import conform as _conform
    from smpl_analysis.intent import pitch_class

    rc = 0
    try:
        allowed = ({pitch_class(p) for p in args.allowed_pcs.split(",") if p.strip()}
                   if args.allowed_pcs else None)
        checks = tuple(c.strip() for c in args.checks.split(",") if c.strip())
        bad = [c for c in checks if c not in ("length", "timing", "pitch", "loudness")]
        if bad:
            raise ValueError(f"unknown check(s): {', '.join(bad)}")
    except ValueError as exc:
        eprint(f"conform: {exc}")
        out.append(error_frame("unsupported", str(exc), op="conform"))
        emit(out)
        return 2

    for audio in audios:
        try:
            out.extend(_conform.conform_audio_frame(
                audio, intent_path=args.intent, length_tol_ms=args.length_tol_ms,
                timing_tol_ms=args.timing_tol_ms, pc_flag_share=args.pc_flag_share,
                allowed_pcs=allowed, lufs=args.lufs, true_peak_db=args.true_peak_db,
                checks=checks))
        except Exception as exc:
            eprint(f"conform: {audio.get('id')}: {exc}")
            out.append(error_frame("op_failed", str(exc), of=audio.get("id"), op="conform"))
            rc = 1

    emit(out)
    return rc
