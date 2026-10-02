"""`smpl write <out>` — materialize the selected audio frame to a file (a tee-style sink).

Passes frames through to stdout by default so `… | smpl write out.wav | …` keeps composing;
`--quiet` suppresses the passthrough.

The sink is also where the stream's analysis leaves the pipe *inside the file*: whatever the
frames carry (tempo, key, markers, loop, caption) is embedded as WAV `smpl`/`cue `/`acid`/
`bext` chunks, or mirrored into FLAC Vorbis comments, so the sample auto-warps and loops
with no smplstream present (spec → *Standards alignment*, export-side SHOULD). See
``smpl_cli._chunks`` for what lands where — and for the manual DAW/sampler check.
`--no-embed` restores a bytes-faithful copy.
"""

from __future__ import annotations

from pathlib import Path

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "write the selected audio frame to a file"


def add_arguments(parser):
    parser.add_argument("out", help="output file path")
    add_selection_args(parser)
    parser.add_argument("--format", help="output subtype (e.g. PCM_16, PCM_24, FLOAT); default: faithful copy")
    parser.add_argument("--no-embed", dest="embed", action="store_false",
                        help="do not embed analysis chunks (smpl/cue/acid/bext, Vorbis comments)")
    parser.add_argument("--quiet", action="store_true", help="do not pass frames through to stdout")


def run(args) -> int:
    from smplstream import cas, error_frame, select as S

    from .. import _chunks

    inframes = read_stdin_frames()
    audio = S.resolve_single_audio(inframes, role=args.role, strict=args.strict)

    # Plan BEFORE writing bytes: a marker point with no integer `sample` is a refusal, not
    # a rounding — and a refused export must not leave a file with wrong cue points behind.
    plan = None
    if args.embed:
        try:
            plan = _chunks.collect(inframes, audio)
        except _chunks.MarkerPrecisionError as exc:
            eprint(f"write: {exc}")
            err = error_frame("op_failed", str(exc), of=audio.get("id"), op="write")
            emit(([] if args.quiet else list(inframes)) + [err])
            return 1

    src = cas.get_path(audio["hash"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    same_container = src.suffix.lower() == out.suffix.lower()
    if same_container and not args.format:
        import shutil

        shutil.copyfile(src, out)
    else:
        import soundfile as sf

        data, sr = sf.read(str(src), dtype="float32", always_2d=False)
        subtype = args.format.upper() if args.format else None
        sf.write(str(out), data, sr, subtype=subtype)
    eprint(f"write: {out} ({audio['meta'].get('dur', 0):.3f}s)")

    if plan:
        import soundfile as sf

        info = sf.info(str(out))
        try:
            written = _chunks.embed(out, plan, sr=int(info.samplerate), n_samples=int(info.frames))
        except Exception as exc:  # a container we can't splice — the audio is still valid
            eprint(f"write: embed skipped ({exc})")
        else:
            if written:
                eprint(f"write: embedded {', '.join(written)}")

    if not args.quiet:
        emit(inframes)
    return 0
