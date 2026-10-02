"""`smpl chords` — chord timeline + key/tuning as a filter (research §3; vault-379o).

Passes every input frame through unchanged, then for each selected `audio` frame appends
a `marker` frame (role ``chord``, points ``{t, dur, label, sample}``) and a `feature` frame
(role ``key``) carrying ``tonal.key_key`` / ``tonal.key_scale`` / ``tonal.tuning_frequency``.

Thin wrapper: the analysis lives in `smpl_analysis.chords` (which also documents why this
runs on a librosa chroma path rather than madmom/Chordino). Heavy imports stay inside
`run()`.
"""

from __future__ import annotations

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "chord-span marker frame + key/tuning feature frame per audio frame"


def add_arguments(parser):
    add_selection_args(parser)
    parser.add_argument("--hop-length", type=int, default=512,
                        help="chroma analysis hop in samples (default 512)")
    parser.add_argument("--smooth-frames", type=int, default=9,
                        help="median-filter width over the chord similarity matrix (default 9)")
    parser.add_argument("--silence-db", type=float, default=-60.0,
                        help="frame RMS below this dBFS is labelled N (no chord); default -60")


def run(args) -> int:
    from smplstream import error_frame, select as S

    inframes = read_stdin_frames()
    out = list(inframes)  # passthrough first (spec: passthrough before derived)

    audios = S.select(inframes, kind="audio", role=args.role, mode=selection_mode(args))
    if not audios and inframes:
        audios = S.select(inframes, kind="audio", mode="all")

    from smpl_analysis import chords as _chords

    rc = 0
    for audio in audios:
        try:
            out.extend(
                _chords.chords_audio_frame(
                    audio,
                    hop_length=args.hop_length,
                    smooth_frames=args.smooth_frames,
                    silence_db=args.silence_db,
                )
            )
        except Exception as exc:
            eprint(f"chords: {audio.get('id')}: {exc}")
            out.append(error_frame("op_failed", str(exc), of=audio.get("id"), op="chords"))
            rc = 1
    emit(out)
    return rc
