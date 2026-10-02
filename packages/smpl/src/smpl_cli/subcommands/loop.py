"""`smpl loop` — make a rendered loop tile-safe (trim render-offset, exact bars, de-click seam).

A render is not a loop: smplmix puts bar-1-beat-1 at sample ~249 and cuts the tail where the
render ended, so naive repeats drift late and click at the seam. This shifts the downbeat to 0,
forces the exact bar length for ``--bpm``/``--bars``, and fades the seam to zero. Passthrough
every input frame first, then append one wet `audio` frame (role ``<role>.wet``, ``op: loopify``)
per selected audio frame. DSP in ``smpl_analysis.edit``.

  smpl read render.wav | smpl loop --bpm 132 --bars 2 | smpl write loop.wav
"""

from __future__ import annotations

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "make the selected audio tile-safe: downbeat→0, exact bar length, de-clicked seam"


def add_arguments(parser):
    add_selection_args(parser)
    parser.add_argument("--bpm", type=float, required=True, help="loop tempo in BPM (required)")
    parser.add_argument("--bars", type=int, default=2, help="loop length in bars (default 2)")
    parser.add_argument("--beats", type=int, default=4, help="beats per bar (default 4)")
    parser.add_argument("--declick-ms", type=float, default=5.0,
                        help="seam fade-to-zero length in ms (default 5)")
    parser.add_argument("--max-trim-ms", type=float, default=12.0,
                        help="longest lead treated as render-offset; longer = musical fade-in, "
                             "left intact (default 12)")


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
            out.append(edit.apply_loopify(
                audio, bpm=args.bpm, bars=args.bars, beats_per_bar=args.beats,
                declick_ms=args.declick_ms, max_trim_ms=args.max_trim_ms))
        except Exception as exc:
            eprint(f"loop: {audio.get('id')}: {exc}")
            out.append(error_frame("op_failed", str(exc), of=audio.get("id"), op="loopify"))
            rc = 1
    emit(out)
    return rc
