"""`smpl beats` — downbeat-aware beat grid as a filter (research §3; vault-32n3).

Passes every input frame through unchanged, then for each selected `audio` frame appends
three `marker` frames — roles ``beat``, ``downbeat`` (bar starts) and ``tempo-change``,
every point carrying float-second ``t`` AND the integer ``sample`` index at the frame's
native ``meta.sr`` — plus a `feature` frame (role ``beats``) with ``rhythm.bpm``,
``rhythm.bpm_confidence``, ``rhythm.bpm_candidates`` and ``rhythm.time_signature``.

Thin wrapper: the analysis lives in `smpl_analysis.beats` (which also documents the
librosa-over-madmom engine choice). Heavy imports stay inside `run()`.
"""

from __future__ import annotations

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "beat/downbeat/tempo-change markers + rhythm.* features per audio frame"


def add_arguments(parser):
    add_selection_args(parser)
    parser.add_argument("--hop-length", type=int, default=512,
                        help="onset-envelope hop in samples (default 512)")
    parser.add_argument("--start-bpm", type=float, default=120.0,
                        help="tempo prior for beat tracking (default 120)")


def run(args) -> int:
    from smplstream import error_frame, select as S

    inframes = read_stdin_frames()
    out = list(inframes)  # passthrough first (spec: passthrough before derived)

    audios = S.select(inframes, kind="audio", role=args.role, mode=selection_mode(args))
    if not audios and inframes:
        audios = S.select(inframes, kind="audio", mode="all")

    from smpl_analysis import beats as _beats

    rc = 0
    for audio in audios:
        try:
            out.extend(
                _beats.beats_audio_frame(
                    audio, hop_length=args.hop_length, start_bpm=args.start_bpm
                )
            )
        except Exception as exc:
            eprint(f"beats: {audio.get('id')}: {exc}")
            out.append(error_frame("op_failed", str(exc), of=audio.get("id"), op="beats"))
            rc = 1
    emit(out)
    return rc
