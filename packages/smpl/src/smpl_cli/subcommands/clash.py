"""`smpl clash` — how layers interact, and whether a loop closes on itself (vault-16dd.4).

A multi-input op: every selected `audio` frame is a LAYER (a stem, a kit lane) and every
pair of layers is measured. ``--role`` narrows the layers; without it every audio frame in
the stream is one. Passes every input frame through unchanged, then per pair appends

* a `feature` frame, role ``clash``: the document naming both layers (id/role/name) with
  the ERB-band overlap (worst bands), the sub collision, the pitch-class clash and the
  Sethares sensory dissonance;
* a `feature` frame, role ``clash-features``: the flat ``clash.*`` scalars;
* a `marker` frame, role ``clash``: the spans where each problem peaks.

``--seam`` (and a single input) adds the loop-seam check per layer: a `feature` frame, role
``clash-seam``, with the tail/head sample jump, slope mismatch, short-window RMS step and the
spectral flux across the wrap, each against the same quantity at the clip's interior; plus a
`marker` frame at the wrap when a ratio is over its flag.

Measures only: the thresholds are flags with documented defaults, not verdicts.

    smpl read drone.wav kick.wav | smpl clash --crossover-hz 80 --out clash.json
    smpl read loop.wav | smpl clash --seam

Thin wrapper: the analysis lives in `smpl_analysis.clash`. Heavy imports stay in `run()`.
"""

from __future__ import annotations

import json

from .._common import add_selection_args, emit, eprint, read_stdin_frames

HELP = "layer interaction: ERB overlap, sub collision, pitch-class clash + dissonance, loop seam"


def add_arguments(parser):
    add_selection_args(parser)
    parser.add_argument("--crossover-hz", type=float, default=80.0,
                        help="sub boundary for the collision check, Hz (default 80)")
    parser.add_argument("--active-db", type=float, default=20.0,
                        help="a frame/band is active within this many dB of the layer's peak "
                             "(default 20)")
    parser.add_argument("--floor-db", type=float, default=60.0,
                        help="... and within this many dB of the layer's loudest band-frame "
                             "(default 60)")
    parser.add_argument("--pc-played", type=float, default=0.25,
                        help="pitch class counts as played above this fraction of the layer's "
                             "peak class (default 0.25)")
    parser.add_argument("--pc-dominant", type=float, default=0.5,
                        help="... and as dominant above this fraction (default 0.5)")
    parser.add_argument("--seam", action="store_true",
                        help="also check each layer's loop seam (always on for a single input)")
    parser.add_argument("--seam-window-ms", type=float, default=20.0,
                        help="RMS window either side of the wrap, ms (default 20)")
    parser.add_argument("--seam-jump-ratio", type=float, default=4.0,
                        help="flag a sample/slope jump over this multiple of the local step "
                             "(default 4)")
    parser.add_argument("--seam-rms-db", type=float, default=6.0,
                        help="flag an RMS step this many dB over the largest elsewhere in the clip "
                             "(default 6)")
    parser.add_argument("--seam-flux-ratio", type=float, default=8.0,
                        help="flag wrap spectral flux over this multiple of the median "
                             "(default 8)")
    parser.add_argument("--out",
                        help="also write the per-pair / per-layer documents here as JSON")


def run(args) -> int:
    from smplstream import error_frame, select as S

    inframes = read_stdin_frames()
    out = list(inframes)  # passthrough first (spec: passthrough before derived)

    audios = S.select(inframes, kind="audio", role=args.role,
                      predicate=lambda f: bool(f.get("hash")), mode="all")
    if not audios:
        eprint("clash: no audio frame in the stream")
        out.append(error_frame("not_found", "clash: no audio frame in the stream", op="clash"))
        emit(out)
        return 1

    from smpl_analysis import clash as _clash

    rc = 0
    try:
        derived = _clash.clash_audio_frames(
            audios,
            seam=args.seam,
            crossover_hz=args.crossover_hz,
            active_db=args.active_db,
            floor_db=args.floor_db,
            pc_played=args.pc_played,
            pc_dominant=args.pc_dominant,
            window_ms=args.seam_window_ms,
            jump_ratio=args.seam_jump_ratio,
            rms_excess_db=args.seam_rms_db,
            flux_ratio=args.seam_flux_ratio,
        )
        out.extend(derived)
    except Exception as exc:
        eprint(f"clash: {exc}")
        out.append(error_frame("op_failed", str(exc), op="clash"))
        rc = 1
        derived = []

    if args.out and derived:
        try:
            docs = [f["data"] for f in derived
                    if f.get("kind") == "feature" and f.get("role") in ("clash", "clash-seam")]
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump(docs, fh, indent=2)
                fh.write("\n")
            eprint(f"clash: wrote {args.out}")
        except Exception as exc:
            eprint(f"clash: could not write {args.out}: {exc}")
            rc = 1

    emit(out)
    return rc
