"""`smpl crosscheck` — referee an audio LLM's claims against smpl's own measurements (vault-16dd.6).

Reads `text` frames with ``role: claim`` (written by ``smpl ask``) plus the `feature` / `marker`
frames already in the stream (``beats`` / ``groove`` / ``loudness`` / ``qc`` / ``chords`` / ``width``)
and appends, per claim, a `feature` frame (role ``crosscheck``) marking it ``supported`` /
``contradicted`` / ``unverifiable`` with the frame values compared, plus one count summary frame.
Stereo-image and fidelity questions come back ``route: smpl`` (the model hears a mono downmix).

    smpl read loop.wav | smpl beats | smpl groove | smpl ask --q tempo_bpm | smpl crosscheck

Passes every input frame through; ``--drop-contradicted`` omits contradicted claim frames from
the passthrough (their content is copied into the verdict frame so lineage never dangles).
Thin wrapper: the logic and the tolerance defaults live in `smpl_analysis.crosscheck`.
"""

from __future__ import annotations

from .._common import emit, eprint, read_stdin_frames

HELP = "mark audio-LLM claims supported / contradicted / unverifiable against smpl measurements"


def add_arguments(parser):
    parser.add_argument("--drop-contradicted", action="store_true",
                        help="drop contradicted claim frames from the output stream")
    parser.add_argument("--tol", action="append", default=[], metavar="KEY=VALUE",
                        help="override a tolerance (repeatable); keys are listed in the module docs")
    parser.add_argument("--report", action="store_true",
                        help="also print a one-line-per-claim table on stderr")


def run(args) -> int:
    from smpl_analysis import crosscheck as _cc

    tol = {}
    for item in args.tol:
        key, sep, val = item.partition("=")
        if not sep or key not in _cc.DEFAULTS:
            eprint(f"crosscheck: bad --tol {item!r}; keys: {sorted(_cc.DEFAULTS)}")
            return 2
        try:
            tol[key] = float(val)
        except ValueError:
            eprint(f"crosscheck: --tol {key}: {val!r} is not a number")
            return 2

    frames = read_stdin_frames()
    out = _cc.crosscheck(frames, tol=tol, drop_contradicted=args.drop_contradicted)
    emit(out)
    if args.report:
        for f in out:
            if f.get("role") == "crosscheck":
                d = f["data"]
                route = f" [route: {d['route']}]" if d.get("route") else ""
                eprint(f"{d['status']:<13} {d['question_id']:<16}{route} {d['reason']}")
    return 0
