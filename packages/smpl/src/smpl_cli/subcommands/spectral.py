"""`smpl spectral` — spectral-shape feature family as a filter (research §2; vault-3uap).

Passes every input frame through unchanged, then for each selected `audio` frame appends a
`feature` frame carrying the spectral distribution-shape descriptors (flatness, crest,
spread, rolloff, contrast, slope, skewness, kurtosis), each as a frame-aggregated
`{mean, stdev}` object under its registered `lowlevel.spectral_*` key.

Thin wrapper: the analysis lives in `smpl_analysis.spectral`. Heavy imports stay inside
`run()`.
"""

from __future__ import annotations

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "spectral-shape feature frame per audio frame (flatness/crest/spread/rolloff/...)"


def add_arguments(parser):
    add_selection_args(parser)
    parser.add_argument("--n-fft", type=int, default=2048, help="STFT window size (default 2048)")
    parser.add_argument("--hop-length", type=int, default=512, help="STFT hop (default 512)")
    parser.add_argument("--no-cache", action="store_true",
                        help="bypass the memo cache: force recompute and refresh the entry")


def run(args) -> int:
    from smplstream import error_frame, select as S

    inframes = read_stdin_frames()
    out = list(inframes)  # passthrough first (spec: passthrough before derived)

    audios = S.select(inframes, kind="audio", role=args.role, mode=selection_mode(args))
    if not audios and inframes:
        audios = S.select(inframes, kind="audio", mode="all")

    from smpl_analysis import spectral as _spectral

    rc = 0
    for audio in audios:
        try:
            out.extend(
                _spectral.spectral_audio_frame(
                    audio, n_fft=args.n_fft, hop_length=args.hop_length,
                    use_cache=not args.no_cache,
                )
            )
        except Exception as exc:
            eprint(f"spectral: {audio.get('id')}: {exc}")
            out.append(error_frame("op_failed", str(exc), of=audio.get("id"), op="spectral"))
            rc = 1
    emit(out)
    return rc
