"""`smpl sheet` — full-length annotated listening sheet + its numbers (vault-16dd.3).

One PNG per clip covering the WHOLE duration, every lane time-aligned: bar ruler, loudness
(momentary + short-term), CQT with beat/bar grid, onsets vs intended hits + section
boundaries, chroma with the intended pitch-class set marked, six band-energy lanes, stereo
correlation, and a self-similarity panel for clips ≥ 30 s. Built for a reader that sees but
cannot hear: big labels, fixed scales per run, and every number in the sidecar.

    smpl read kick.wav | smpl sheet --out sheets/
    smpl sheet loop.wav --out sheets/                 # path-in, intent from loop.wav.provenance.json
    smpl read a.wav b.wav | smpl sheet --ab --out sheets/

Every input frame passes through unchanged, then per selected audio frame this appends an
`image` frame (role ``sheet``), a `feature` frame (role ``sheet``) whose ``data`` is the
sidecar document (global numbers + per-bar rows), and a `text` frame (role
``sheet:summary``, markdown in the CAS). ``--ab`` takes the last two selected audio frames
(A = earlier, B = later), gains B to A's integrated loudness and emits the same three frames
with role ``sheet:ab``. ``--out DIR`` also writes ``<name>.sheet.png|json|md``. The analysis
and rendering live in ``smpl_analysis.sheet``; this module is glue.
"""

from __future__ import annotations

from .._common import (
    add_selection_args,
    emit,
    eprint,
    read_stdin_frames,
    selection_mode,
)

HELP = "render a full-length annotated listening sheet (PNG + per-bar sidecar), or an A/B sheet"

OP = "sheet"


def add_arguments(parser):
    parser.add_argument("audio", nargs="*", default=None,
                        help="audio file(s) to ingest and render (omit when piping frames in)")
    add_selection_args(parser)
    parser.add_argument("--ab", action="store_true",
                        help="A/B mode: the last two selected audio frames, loudness-matched, "
                             "stacked on shared axes with a B−A difference panel")
    parser.add_argument("--intent", default=None,
                        help="intent file (default: <audio>.intent.json / .provenance.json "
                             "next to the read source); in --ab it applies to clip A")
    parser.add_argument("--bpm", type=float, default=None,
                        help="draw the bar grid at this tempo (overrides intent and the tracker)")
    parser.add_argument("--beats-per-bar", type=float, default=None,
                        help="beats per bar for the grid (default: intent's, else 4)")
    parser.add_argument("--no-beat-track", action="store_true",
                        help="with no intent/--bpm, use equal segments instead of the beat tracker")
    parser.add_argument("--db-range", type=float, nargs=2, metavar=("MIN", "MAX"), default=None,
                        help="fixed dB range for the CQT and band lanes, dB of a sinusoid "
                             "(default: -90 -10)")
    parser.add_argument("--lufs-range", type=float, nargs=2, metavar=("MIN", "MAX"), default=None,
                        help="fixed loudness-lane range in LUFS (default: -60 0)")
    parser.add_argument("--cmap", default=None, help="colormap for every heatmap (default: magma)")
    parser.add_argument("--diff-range", type=float, default=None,
                        help="--ab difference panel spans ±this many dB (default: 24)")
    parser.add_argument("--max-width", type=int, default=None,
                        help="image width cap in pixels; long clips are pooled to fit (default: 1600)")
    parser.add_argument("--out", default=None,
                        help="also write <name>.sheet.png, .sheet.json and .sheet.md into this directory")


def _ingest(path: str) -> dict:
    from smplstream import cas
    from smplstream import frames as F

    h = cas.put_audio_file(path)
    meta = cas.read_meta(h) or {}
    return F.audio_frame(
        h, sr=meta.get("sr", 0), ch=meta.get("ch", 1), dur=meta.get("dur", 0.0),
        role="source", op="read", op_version="read@1", fmt=meta.get("fmt"),
        params={"source": path},
    )


def _write_out(out_dir: str, stem: str, derived: list[dict]) -> list[str]:
    import json
    from pathlib import Path

    from smplstream import cas

    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    written = []
    for f in derived:
        if f.get("kind") == "image":
            p = d / f"{stem}.png"
            p.write_bytes(cas.get_path(f["hash"]).read_bytes())
            written.append(str(p))
            params = f.get("params") or {}
            pj = d / f"{stem}.json"
            pj.write_text(json.dumps(json.loads(cas.get_path(params["json_hash"]).read_text()),
                                     indent=2) + "\n")
            pm = d / f"{stem}.md"
            pm.write_bytes(cas.get_path(params["markdown_hash"]).read_bytes())
            written += [str(pj), str(pm)]
    return written


def _stem(name: str) -> str:
    import re

    base = name.rsplit(".", 1)[0] if "." in name else name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base)


def run(args) -> int:
    from smplstream import error_frame
    from smplstream import select as S

    inframes = read_stdin_frames()
    out = list(inframes)  # passthrough first (spec: passthrough before derived)

    if args.audio:
        audios = []
        for path in args.audio:
            try:
                fr = _ingest(path)
            except Exception as exc:
                eprint(f"sheet: {path}: {exc}")
                out.append(error_frame("decode_failed", f"{path}: {exc}", op=OP))
                emit(out)
                return 1
            out.append(fr)
            audios.append(fr)
    else:
        mode = "all" if args.ab else selection_mode(args)
        audios = S.select(inframes, kind="audio", role=args.role, mode=mode)
        if not audios and inframes:
            audios = S.select(inframes, kind="audio", mode="all" if args.ab else "last")

    if args.ab and len(audios) < 2:
        eprint(f"sheet: --ab needs two audio frames, got {len(audios)}")
        out.append(error_frame("op_failed", "--ab needs two audio frames", op=OP))
        emit(out)
        return 1
    if not audios:
        eprint("sheet: no audio frame to render (pipe one in, or pass a path)")
        emit(out)
        return 1

    try:
        from smpl_analysis import sheet as _sheet
    except Exception as exc:  # analysis tier not installed
        eprint(f"sheet: analysis library unavailable: {exc}")
        emit(out)
        return 1

    kw = {"bpm": args.bpm, "beats_per_bar": args.beats_per_bar,
          "detect_beats": not args.no_beat_track, "db_range": args.db_range,
          "lufs_range": args.lufs_range, "cmap": args.cmap, "diff_range": args.diff_range,
          "max_width_px": args.max_width}

    jobs = [("ab", audios[-2], audios[-1])] if args.ab else [("one", a, None) for a in audios]
    rc = 0
    for kind, a, b in jobs:
        try:
            if kind == "ab":
                derived = _sheet.sheet_ab_frames(a, b, intent_path=args.intent, **kw)
                stem = f"{_stem(_sheet.frame_name(a))}__vs__{_stem(_sheet.frame_name(b))}.sheet-ab"
            else:
                derived = _sheet.sheet_audio_frame(a, intent_path=args.intent, **kw)
                stem = f"{_stem(_sheet.frame_name(a))}.sheet"
        except Exception as exc:
            eprint(f"sheet: {a.get('id')}: {exc}")
            out.append(error_frame("op_failed", str(exc), of=a.get("id"), op=OP))
            rc = 1
            continue
        out.extend(derived)
        if args.out:
            try:
                for p in _write_out(args.out, stem, derived):
                    eprint(f"sheet: wrote {p}")
            except Exception as exc:
                eprint(f"sheet: cannot write into {args.out!r}: {exc}")
                rc = 1
    emit(out)
    return rc
