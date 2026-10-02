"""`smpl arc` — recording arc vs intended arc overlay (ticket vault-3te4).

Overlays the MEASURED arc of a session recording against the INTENDED tension curve in a
set's ``narrative.yaml``, with the differences burned into the PNG. A mismatch is rendered
as a **difference** — often the improvised departure that worked — never as an error.

Two ways in, both fine::

    smpl read set.wav | smpl arc --narrative sets/transmission/narrative.yaml
    smpl arc set.wav --narrative sets/transmission/narrative.yaml

Every input frame passes through unchanged, then this appends an `image` frame (role
``arc:overlay``), one `feature` frame per section (role ``arc:section`` — the per-section
NDJSON), and an ``arc:summary`` feature frame. ``--out PREFIX`` additionally writes
``PREFIX.png`` + ``PREFIX.sections.ndjson`` sidecars. The measurement and rendering live
in ``smpl_analysis.arc``; this module is glue.
"""

from __future__ import annotations

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "overlay a recording's measured energy arc against a narrative.yaml intended tension curve"

OP = "arc"


def add_arguments(parser):
    parser.add_argument("audio", nargs="?", default=None,
                        help="audio file to ingest and analyze (omit when piping frames in)")
    parser.add_argument("--narrative", required=True,
                        help="path to the set's narrative.yaml (schema: narrative/v1)")
    add_selection_args(parser)
    parser.add_argument("--sections", type=int, default=None,
                        help="segment into ~N sections (default: the narrative's anchor count)")
    parser.add_argument("--window", type=float, default=None,
                        help="trajectory analysis window in seconds (default: 4.0)")
    parser.add_argument("--hop", type=float, default=None,
                        help="trajectory hop in seconds (default: 1.0)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="|measured−intended| called out as a notable difference (default: 0.20)")
    parser.add_argument("--out", default=None,
                        help="also write PREFIX.png and PREFIX.sections.ndjson sidecars")


def _ingest(path: str) -> dict:
    from smplstream import cas, frames as F

    h = cas.put_audio_file(path)
    meta = cas.read_meta(h) or {}
    return F.audio_frame(
        h, sr=meta.get("sr", 0), ch=meta.get("ch", 1), dur=meta.get("dur", 0.0),
        role="source", op="read", op_version="read@1", fmt=meta.get("fmt"),
        params={"source": path},
    )


def _write_sidecars(prefix: str, derived: list[dict]) -> None:
    from smplstream import cas, ndjson

    for f in derived:
        if f.get("kind") == "image" and f.get("role") == "arc:overlay":
            with open(f"{prefix}.png", "wb") as fh:
                fh.write(cas.get_path(f["hash"]).read_bytes())
    with open(f"{prefix}.sections.ndjson", "wb") as fh:
        for f in derived:
            if f.get("role") == "arc:section":
                fh.write(ndjson.dumps(f) + b"\n")


def run(args) -> int:
    from smplstream import error_frame, select as S

    inframes = read_stdin_frames()
    out = list(inframes)  # passthrough first (spec: passthrough before derived)

    if args.audio:
        try:
            ingested = _ingest(args.audio)
        except Exception as exc:
            eprint(f"arc: {args.audio}: {exc}")
            out.append(error_frame("decode_failed", f"{args.audio}: {exc}", op=OP))
            emit(out)
            return 1
        out.append(ingested)
        audios = [ingested]
    else:
        audios = S.select(inframes, kind="audio", role=args.role, mode=selection_mode(args))
        if not audios and inframes:
            audios = S.select(inframes, kind="audio", mode="last")

    if not audios:
        eprint("arc: no audio frame to analyze (pipe one in, or pass a path)")
        emit(out)
        return 1

    try:
        from smpl_analysis import arc as _arc
    except Exception as exc:  # analysis tier not installed
        eprint(f"arc: analysis library unavailable: {exc}")
        emit(out)
        return 1

    kw = {}
    if args.sections is not None:
        kw["n_sections"] = args.sections
    if args.window is not None:
        kw["window_s"] = args.window
    if args.hop is not None:
        kw["hop_s"] = args.hop
    if args.threshold is not None:
        kw["threshold"] = args.threshold

    rc = 0
    for audio in audios:
        try:
            derived = _arc.arc_audio_frame(audio, args.narrative, **kw)
        except Exception as exc:
            eprint(f"arc: {audio.get('id')}: {exc}")
            out.append(error_frame("op_failed", str(exc), of=audio.get("id"), op=OP))
            rc = 1
            continue
        out.extend(derived)
        for d in derived:
            if d.get("role") == "arc:section" and d["data"].get("difference"):
                eprint(d["data"]["difference"])
        if args.out:
            try:
                _write_sidecars(args.out, derived)
            except Exception as exc:
                eprint(f"arc: cannot write sidecars at {args.out!r}: {exc}")
                rc = 1
    emit(out)
    return rc
