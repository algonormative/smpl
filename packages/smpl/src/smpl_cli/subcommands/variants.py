"""`smpl variants` — bake N timbral variants of one source across a cutoff range (vault-5yeb).

A static one-shot palette reads dead: every hit is the same timbre, and with no live filter
under your hands there is no closed→open, dark→bright gesture to play. `smpl automate --target
cutoff` bakes that movement WITHIN one render; this bakes it ACROSS renders — the same sweep
frozen at N log-spaced points, giving the closed / muted / half / open / bright palette a sound
designer would print by hand. Each variant is a STATIC resonant low-pass render.

Passthrough every input frame first, then append one wet `audio` frame per step per selected
audio frame (role ``<role>.variant:<k>``, ``op: variants``, k = 1..steps, closed → open) with
full lineage. DSP lives in ``smpl_analysis.edit``; this module is thin.

Examples:
  # 5-step palette from a dark pad: 200 Hz (closed) → 8 kHz (open)
  smpl read pad.wav | smpl variants --lo 200 --hi 8000 --steps 5
  # resonant, vocal-edged variants, then materialize the most open one
  smpl read pad.wav | smpl variants --steps 4 --resonance 6 | smpl write --role source.variant:4 open.wav
"""

from __future__ import annotations

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "render N static filter variants across a cutoff range (emits <role>.variant:<k>)"


def add_arguments(parser):
    add_selection_args(parser)
    parser.add_argument("--lo", type=float, default=200.0,
                        help="closed end of the cutoff range in Hz (default 200)")
    parser.add_argument("--hi", type=float, default=8000.0,
                        help="open end of the cutoff range in Hz (default 8000)")
    parser.add_argument("--steps", type=int, default=5,
                        help="number of variants, log-spaced lo..hi inclusive (default 5)")
    parser.add_argument("--resonance", type=float, default=0.707,
                        help="low-pass Q (0.707 flat; 3-10 = resonant edge at the cutoff)")


def run(args) -> int:
    from smplstream import error_frame, select as S

    if args.steps < 1:
        eprint("variants: --steps must be >= 1"); return 2
    if args.resonance <= 0:
        eprint("variants: --resonance must be > 0"); return 2
    if args.steps > 1 and args.lo >= args.hi:
        eprint("variants: --lo must be below --hi"); return 2

    inframes = read_stdin_frames()
    out = list(inframes)  # passthrough first (spec: passthrough before derived)

    audios = S.select(inframes, kind="audio", role=args.role, mode=selection_mode(args))
    if not audios and inframes:
        audios = S.select(inframes, kind="audio", mode="all")

    from smpl_analysis import edit

    rc = 0
    for audio in audios:
        try:
            out.extend(edit.render_cutoff_variants(
                audio, lo_hz=args.lo, hi_hz=args.hi, steps=args.steps,
                resonance=args.resonance))
        except Exception as exc:
            eprint(f"variants: {audio.get('id')}: {exc}")
            out.append(error_frame("op_failed", str(exc), of=audio.get("id"), op="variants"))
            rc = 1
    emit(out)
    return rc
