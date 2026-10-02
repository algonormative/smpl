"""`smpl sidechain` — duck the selected TARGET under a TRIGGER's transients (2-audio-input op).

The Birmingham rumble + glue move: a kick (TRIGGER) pushes a sub, rumble, or full mix (TARGET)
out of the way so the transient lands in cleared space, then the target recovers on the release.

  target  — the last-wins selected `audio` frame from stdin (``--role`` narrows the pick).
  trigger — ``--with <PATH-or-role>``: an existing file path is CAS-ingested as a `source`
            frame AND emitted (lineage closure); otherwise it resolves as a role in the stream.

Passthrough every input frame first, then append the ingested trigger (if a file) and one wet
`audio` frame (role ``<target_role>.wet``, ``op: sidechain``, ``lineage: [target.id, trigger.id]``).
The wet frame's ``params`` carry the MEASURED duck (``gr_max_db``, ``ducked_fraction``) — the A/B
evidence. DSP lives in ``smpl_analysis.duo``. Follows the ``vocode`` subcommand structure.

  smpl read sub.wav | smpl sidechain --with kick.wav --depth 12 --release 120 | smpl write duck.wav
"""

from __future__ import annotations

import os

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "duck the selected target under a trigger's transients (--with PATH|role); emits <role>.wet"


def add_arguments(parser):
    add_selection_args(parser)
    parser.add_argument("--with", dest="trigger", required=True,
                        help="trigger: an audio file PATH (ingested + emitted) or a role in the stream")
    parser.add_argument("--attack", type=float, default=5.0,
                        help="duck attack, ms — how fast the target gets out of the way (default 5)")
    parser.add_argument("--release", type=float, default=120.0,
                        help="duck release, ms — how fast the target recovers (default 120)")
    parser.add_argument("--depth", type=float, default=12.0,
                        help="maximum gain reduction, dB (default 12; 0 disables)")
    parser.add_argument("--threshold", type=float, default=-30.0,
                        help="trigger knee, dBFS — below it the target is untouched (default -30)")


def run(args) -> int:
    from smplstream import cas, error_frame, frames as F, select as S

    inframes = read_stdin_frames()
    out = list(inframes)

    # Target: last-wins selected audio frame from the stream (--role narrows).
    targets = S.select(inframes, kind="audio", role=args.role, mode=selection_mode(args))
    if not targets and inframes:
        targets = S.select(inframes, kind="audio", mode="last")
    if not targets:
        eprint("sidechain: no target audio frame in the stream")
        out.append(error_frame("not_found", "sidechain: no target audio frame in the stream",
                               op="sidechain"))
        emit(out)
        return 1
    target = targets[-1]

    # Trigger: --with is a file PATH (ingest + emit for lineage closure) or a role in the stream.
    trigger = None
    spec = args.trigger
    if os.path.isfile(spec):
        try:
            h = cas.put_audio_file(spec)
            meta = cas.read_meta(h) or {}
            trigger = F.audio_frame(
                h,
                sr=meta.get("sr", 0),
                ch=meta.get("ch", 1),
                dur=meta.get("dur", 0.0),
                role="source",
                op="read",
                op_version="read@1",
                fmt=meta.get("fmt"),
                params={"source": spec},
            )
            out.append(trigger)  # emit the ingested trigger so its lineage is in-stream
        except Exception as exc:
            eprint(f"sidechain: failed to ingest trigger {spec!r}: {exc}")
            out.append(error_frame("decode_failed", f"{spec}: {exc}", op="sidechain"))
            emit(out)
            return 1
    else:
        trigs = S.select(inframes, kind="audio", role=spec, mode="last")
        if not trigs:
            eprint(f"sidechain: trigger role {spec!r} not found in the stream (and not a file path)")
            out.append(error_frame("not_found",
                                    f"sidechain: trigger {spec!r} is neither a file path nor a stream role",
                                    op="sidechain"))
            emit(out)
            return 1
        trigger = trigs[-1]

    from smpl_analysis import duo

    rc = 0
    try:
        out.append(duo.apply_sidechain(
            target,
            trigger,
            attack_ms=args.attack,
            release_ms=args.release,
            depth_db=args.depth,
            threshold_db=args.threshold,
        ))
    except Exception as exc:
        eprint(f"sidechain: {target.get('id')}: {exc}")
        out.append(error_frame("op_failed", str(exc), of=target.get("id"), op="sidechain"))
        rc = 1
    emit(out)
    return rc
