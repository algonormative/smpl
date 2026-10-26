"""`smpl groove-feel` — how a rhythm feels, as numbers and as text Claude can read
(vault-16dd.7).

`smpl groove` says where hits sit relative to a grid (swing + per-step nudge). This adds what
a pattern *feels* like — a syncopation index (Longuet-Higgins/Lee), the microtiming
distribution per grid step, the accent profile (level per step) — and exports the hits as a
compact step grid with velocity buckets plus a MIDI-like hit list, because language models
reason over symbolic rhythm far better than over audio::

    kick  b1 |X... .x.. ..x. X...|
          b2 |X... .... ..x. X.x.|
          ms 11:+31 15:-18

Hits come from the clip's intent (``--intent FILE``, or a ``.intent.json`` /
``.provenance.json`` sidecar next to the source; the real onset and level are then measured
against it) and otherwise from detected onsets fitted to the grid by `groove`'s own fit.
``--ref FILE...`` compares the per-step timing and accent against reference grooves: loops,
or groove JSON from ``smpl groove --out``.

Passes every input frame through unchanged, then appends two `feature` frames per selected
`audio` frame: role ``groove-feel`` (``data`` = the full document including ``text``) and
role ``groove-feel-features`` (the registered scalar keys).

Thin wrapper: the analysis lives in `smpl_analysis.groove_feel`. Heavy imports stay inside
`run()`.
"""

from __future__ import annotations

import json

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "syncopation, per-step microtiming, accent profile + hits as text, vs intent or a reference"


def add_arguments(parser):
    add_selection_args(parser)
    parser.add_argument("--intent", help="intent file (default: sidecar next to the source)")
    parser.add_argument("--ref", nargs="+", metavar="FILE",
                        help="reference grooves: audio loops or `smpl groove --out` JSON; "
                             "emits per-step timing / accent deltas (averaged over the set)")
    parser.add_argument("--bpm", type=float, default=None,
                        help="tempo the grid is measured against (default: intent's, else estimate)")
    parser.add_argument("--grid-steps", type=int, default=None,
                        help="steps per bar (default: intent's, else 16)")
    parser.add_argument("--beats-per-bar", type=float, default=None,
                        help="beats per bar (default: intent's, else 4)")
    parser.add_argument("--flag-ms", type=float, default=10.0,
                        help="list a hit's ms offset in the text when |ms| >= this (default 10)")
    parser.add_argument("--accent-hi", type=float, default=0.8,
                        help="velocity at or above which a hit is X (default 0.8)")
    parser.add_argument("--accent-lo", type=float, default=0.5,
                        help="velocity at or above which a hit is x; below it o (default 0.5)")
    parser.add_argument("--velocity-source", choices=("auto", "intent", "measured"), default="auto",
                        help="velocity shown in the text: intent's, or measured level relative to "
                             "the loudest hit (auto: intent when there is one)")
    parser.add_argument("--search-ms", type=float, default=None,
                        help="attack-search window each side of an intended hit; default 30 after "
                             "/ 10 before. Widen to measure a loop pushed >10 ms early")
    parser.add_argument("--text-out",
                        help="also write the text export here ('-' for stderr)")
    parser.add_argument("--out", help="also write the full document here as JSON")


def run(args) -> int:
    from smplstream import error_frame, select as S

    inframes = read_stdin_frames()
    out = list(inframes)  # passthrough first (spec: passthrough before derived)

    audios = S.select(inframes, kind="audio", role=args.role, mode=selection_mode(args))
    if not audios and inframes:
        audios = S.select(inframes, kind="audio", mode="all")

    from smpl_analysis import groove_feel as _gf

    rc = 0
    document = None
    for audio in audios:
        try:
            derived, document = _gf.groove_feel_audio_frame(
                audio,
                intent_path=args.intent,
                ref_paths=args.ref,
                bpm=args.bpm,
                beats_per_bar=args.beats_per_bar,
                grid_steps=args.grid_steps,
                velocity_source=args.velocity_source,
                search_ms=args.search_ms,
                flag_ms=args.flag_ms,
                accent_hi=args.accent_hi,
                accent_lo=args.accent_lo,
            )
            out.extend(derived)
        except Exception as exc:
            eprint(f"groove-feel: {audio.get('id')}: {exc}")
            out.append(error_frame("op_failed", str(exc), of=audio.get("id"), op="groove-feel"))
            rc = 1

    if document is not None:
        try:
            if args.text_out == "-":
                eprint(document["text"], end="")
            elif args.text_out:
                with open(args.text_out, "w", encoding="utf-8") as fh:
                    fh.write(document["text"])
                eprint(f"groove-feel: wrote {args.text_out}")
            if args.out:
                with open(args.out, "w", encoding="utf-8") as fh:
                    json.dump(document, fh, indent=2)
                    fh.write("\n")
                eprint(f"groove-feel: wrote {args.out}")
        except Exception as exc:
            eprint(f"groove-feel: could not write output: {exc}")
            rc = 1

    emit(out)
    return rc
