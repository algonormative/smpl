"""`smpl timbre` — AudioCommons timbral descriptors as a filter (research §6; vault-14ia).

Passes every input frame through unchanged, then for each selected `audio` frame appends a
`feature` frame (role ``timbre``) carrying the eight LLM-facing perceptual descriptors —
hardness, depth, brightness, roughness, warmth, sharpness, boominess on 0–100, plus the
binary ``timbre.reverb`` flag — so a sample is natural-language queryable ("dark, warm,
boomy one-shot").

Thin wrapper: the DSP lives in `smpl_analysis.timbre` (a compact reimplementation of the
AudioCommons descriptors, not the unmaintained upstream `timbral_models`). Heavy imports
stay inside `run()`.
"""

from __future__ import annotations

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "AudioCommons timbral feature frame per audio frame (hardness/depth/brightness/...)"


def add_arguments(parser):
    add_selection_args(parser)
    parser.add_argument("--n-fft", type=int, default=2048, help="STFT window size (default 2048)")
    parser.add_argument("--hop-length", type=int, default=512, help="STFT hop (default 512)")


def run(args) -> int:
    from smplstream import error_frame, select as S

    inframes = read_stdin_frames()
    out = list(inframes)  # passthrough first (spec: passthrough before derived)

    audios = S.select(inframes, kind="audio", role=args.role, mode=selection_mode(args))
    if not audios and inframes:
        audios = S.select(inframes, kind="audio", mode="all")

    from smpl_analysis import timbre as _timbre

    rc = 0
    for audio in audios:
        try:
            out.extend(
                _timbre.timbre_audio_frame(audio, n_fft=args.n_fft, hop_length=args.hop_length)
            )
        except Exception as exc:
            eprint(f"timbre: {audio.get('id')}: {exc}")
            out.append(error_frame("op_failed", str(exc), of=audio.get("id"), op="timbre"))
            rc = 1
    emit(out)
    return rc
