"""`smpl render-stems` — a pattern recipe → full mix + grouped stems (perc/bass/atmos).

The native form of the v2 loop pipeline's ``render-stems.py`` wrapper (ticket vault-3i88).
Reads a ``smpl pattern`` recipe, groups its tracks by their ``stem`` tag (missing tags are
inferred from the track name), and renders:

  <name>.full.wav    every track, mastered   (audition + A/B reference)
  <name>.perc.wav    kick/clap/snare/hats/…  (→ drum machine / sampler track)
  <name>.bass.wav    sub/bass                (→ sub track, mono)
  <name>.atmos.wav   stabs/texture/pads/…    (→ modular / sampler track)

  smpl render-stems --pattern-file loop.pattern.json --outdir renders
  cat loop.pattern.json | smpl render-stems --outdir renders --no-master

**The ladder is the point.** Every sub-session is rendered with ``master.loudnorm`` off and
the per-stem post chain is level-neutral, so the recipe's per-track ``gain_db`` balance
survives into the stems and they still sum to the mix. Only the full mix is mastered (widen
above 220 Hz → presence bell + air shelf → normalize to −14 LUFS); ``--no-master`` leaves
even that raw. Bus glue is applied in-process through the same DSP the `gain` / `filter` /
`widen` subcommands use — no sox, no nested pipes. **Delta vs the sox wrapper:** the perc
bus's ``overdrive 2 0`` saturation has no in-suite equivalent, so perc glue is the −3 dB
pad alone (the one deliberate level change in the whole op).

Composition still runs on **smplmix**, an external binary: it is resolved from ``--smplmix``,
then ``$SMPL_SMPLMIX_BIN``, then PATH, and a missing one is a hard error (never a partial
render). Emits one `audio` frame per output plus a `feature` manifest, so the op composes
with the rest of the pipe world.
"""

from __future__ import annotations

import os

from .._common import emit, eprint

HELP = "render a pattern recipe to a mastered full mix + ladder-preserving stems"

OP_VERSION = "render-stems@1"


def add_arguments(parser):
    parser.add_argument("--pattern-file", help="pattern recipe JSON file (default: read stdin)")
    parser.add_argument("--outdir", default=".", help="output directory (default: .)")
    parser.add_argument("--stems", default="all",
                        help="comma-separated stems to render (default: all = perc,bass,atmos)")
    parser.add_argument("--no-master", dest="no_master", action="store_true",
                        help="skip the full mix's master chain (render it raw, like the stems)")
    parser.add_argument("--no-widen", dest="no_widen", action="store_true",
                        help="skip the per-stem stereo widen (glue only)")
    parser.add_argument("--smplmix", help="path to the smplmix binary (default: $SMPL_SMPLMIX_BIN or PATH)")
    parser.add_argument("--role", default="stems", help="role prefix for emitted frames (default: stems)")


def _apply_chain(frame: dict, chain: list) -> dict:
    """Run a ``[(op, kwargs), ...]`` post chain over an audio frame, in-process."""
    from smpl_analysis import edit

    ops = {
        "gain": edit.apply_gain,
        "mono": edit.apply_mono,
        "filter": edit.apply_filter,
        "widen": edit.apply_widen,
        "eq": edit.apply_eq,
        "normalize": edit.apply_normalize,
    }
    for op, kwargs in chain:
        fn = ops.get(op)
        if fn is None:
            raise ValueError(f"unknown post-chain op {op!r}")
        frame = fn(frame, **kwargs)
    return frame


def _ingest(path: str, role: str) -> dict:
    """CAS a rendered file and return an `audio` frame the edit ops can consume."""
    from smplstream import cas, frames as F

    h = cas.put_audio_file(path)
    meta = cas.read_meta(h) or {}
    return F.audio_frame(h, sr=meta.get("sr", 0), ch=meta.get("ch", 1), dur=meta.get("dur", 0.0),
                         role=role, op="render-stems", op_version=OP_VERSION,
                         fmt=meta.get("fmt"), params={"source": path})


def _render(session: dict, out_wav: str, *, binary: str, kind: str) -> None:
    """Write the expanded session to a temp file and let smplmix render it to ``out_wav``."""
    import json
    import subprocess
    import tempfile

    from .. import _stems as ST

    with tempfile.TemporaryDirectory(prefix="smpl-render-stems-") as tmp:
        sess = os.path.join(tmp, f"{kind}.smplset.json")
        with open(sess, "w", encoding="utf-8") as fh:
            json.dump(session, fh)
        proc = subprocess.run(ST.smplmix_command(binary, sess, out_wav),
                              capture_output=True, text=True)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[:400]
        raise RuntimeError(f"smplmix render failed for {kind} (exit {proc.returncode}): {detail}")
    if not os.path.exists(out_wav):
        raise RuntimeError(f"smplmix produced no output for {kind} ({out_wav})")


def _measure(path: str) -> dict:
    """Integrated LUFS + true peak of a finished output (reported, never acted on)."""
    import soundfile as sf

    from smpl_analysis import loudness

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    res = loudness.analyze_array(data, int(sr))
    return {"integrated_lufs": res.get("integrated_lufs"),
            "true_peak_dbtp": res.get("true_peak_dbtp")}


def run(args) -> int:
    import shutil
    import tempfile

    from smplstream import cas, frames as F

    from .. import _stems as ST
    from .pattern import _expand, _load_pattern

    try:
        recipe = _load_pattern(args)
        stems = ST.parse_stems(args.stems)
        layout = ST.plan(recipe, outdir=args.outdir, stems=stems, master=not args.no_master,
                         widen=not args.no_widen, pattern_path=args.pattern_file)
    except Exception as exc:
        eprint(f"render-stems: {exc}")
        return 1

    binary = ST.resolve_smplmix(args.smplmix)
    if binary is None:
        eprint(ST.missing_smplmix_message(args.smplmix))
        return 1

    out_frames: list = []
    outputs: dict = {}
    try:
        os.makedirs(layout["outdir"], exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="smpl-stems-raw-") as raw_dir:
            for job in layout["jobs"]:
                kind = job["kind"]
                # Render RAW into a scratch dir — smplmix's sidecar manifest never litters
                # the user's outdir, and a failed job leaves no half-finished stem behind.
                raw = os.path.join(raw_dir, f"{layout['name']}.{kind}.raw.wav")
                session = _expand(job["recipe"])
                _render(session, raw, binary=binary, kind=kind)

                frame = _apply_chain(_ingest(raw, f"{args.role}.{kind}"), job["chain"])
                shutil.copyfile(cas.get_path(frame["hash"]), job["out"])

                loud = _measure(job["out"])
                meta = frame.get("meta", {})
                out_frames.append(F.audio_frame(
                    frame["hash"], sr=meta.get("sr", 0), ch=meta.get("ch", 1),
                    dur=meta.get("dur", 0.0), role=f"{args.role}.{kind}",
                    of=frame.get("id"), lineage=[frame["id"]] if frame.get("id") else None,
                    op="render-stems", op_version=OP_VERSION, fmt=meta.get("fmt"),
                    params={"kind": kind, "path": job["out"],
                            "tracks": [str(t.get("name") or "") for t in job["tracks"]],
                            "chain": [op for op, _ in job["chain"]],
                            "loudnorm": kind == "full" and not args.no_master,
                            "loudness": loud},
                ))
                outputs[kind] = job["out"]
                eprint(f"render-stems: {kind:<6} {os.path.basename(job['out']):<32} "
                       f"LUFS {loud['integrated_lufs']}  dBTP {loud['true_peak_dbtp']}")
    except Exception as exc:
        eprint(f"render-stems: {exc}")
        return 1

    out_frames.append(F.feature_frame(
        {"name": layout["name"], "set": layout["set"], "outdir": layout["outdir"],
         "outputs": outputs, "groups": layout["groups"],
         "mastered": not args.no_master, "ladder_preserved": True},
        role=args.role, op="render-stems", op_version=OP_VERSION,
        params={"outputs": len(outputs), "stems": list(stems)},
    ))
    emit(out_frames)
    return 0
