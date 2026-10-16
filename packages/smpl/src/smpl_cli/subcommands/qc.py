"""`smpl qc` — technical QC / defect detection as a filter (research §4; ticket vault-1e9a).

Passes every input frame through, then appends derived QC frames (one `feature` with the
deterministic top-6 scalars + `marker` frames for click/gap locations) for each audio frame.
The DSP lives in `smpl_analysis.qc`; this module is a thin selection-and-passthrough shell.

Two opt-in modes for gating generated one-shots (`smpl read a.wav b.wav c.wav | smpl qc ...`):

  --audiobox  also score each file with Audiobox-Aesthetics (CE/CU/PC/PQ; needs the
              `smpl-analysis[audiobox]` extra) and append a `qc.audiobox.*` feature frame.
  --report    instead of NDJSON frames, print ONE JSON document on stdout:
              {"files": [{"id", "file", "attack_time_s", "late_energy_share",
                          "audiobox": {"CE","CU","PC","PQ"} | null, "error"?}, ...]}
              One entry per input audio frame (implies --all), in input order. Thresholds
              are the caller's business; this only measures.
"""

from __future__ import annotations

import json
import sys

from .._common import add_selection_args, emit, eprint, read_stdin_frames, selection_mode

HELP = "QC audio frame(s) as a filter: passthrough + clipping/phase/DC/SNR/lossy + defect markers"


def add_arguments(parser):
    add_selection_args(parser)
    parser.add_argument(
        "--no-markers",
        action="store_true",
        help="skip click/gap defect marker frames (emit only the feature frame)",
    )
    parser.add_argument("--audiobox", action="store_true",
                        help="also score with Audiobox-Aesthetics CE/CU/PC/PQ "
                             "(requires: pip install 'smpl-analysis[audiobox]')")
    parser.add_argument("--report", action="store_true",
                        help="emit one JSON report (one entry per input file with "
                             "attack_time_s, late_energy_share, audiobox) instead of frames; "
                             "implies --all")
    parser.add_argument("--no-cache", action="store_true",
                        help="bypass the memo cache: force recompute and refresh the entry")


def run(args) -> int:
    from smplstream import error_frame, select as S

    inframes = read_stdin_frames()
    out = list(inframes)  # passthrough first (spec: passthrough before derived)

    mode = "all" if args.report else selection_mode(args)
    audios = S.select(inframes, kind="audio", role=args.role, mode=mode)
    if not audios and inframes:
        audios = S.select(inframes, kind="audio", mode="all")

    try:
        from smpl_analysis import qc as _qc
    except Exception as exc:  # the analysis tier must be installed for qc
        eprint(f"qc: analysis library unavailable: {exc}")
        emit(out)
        return 1

    ab = None
    if args.audiobox:
        try:
            from smpl_analysis import audiobox as ab

            ab.check_available()
        except Exception as exc:  # missing extra: one clear line, no traceback
            eprint(f"qc: --audiobox unavailable: {exc}")
            return 2

    rc = 0
    entries = []
    for audio in audios:
        entry = {"id": audio.get("id"), "file": (audio.get("params") or {}).get("source"),
                 "attack_time_s": None, "late_energy_share": None,
                 "audiobox": None}
        try:
            derived = _qc.qc_audio_frame(
                audio,
                want_markers=not args.no_markers,
                use_cache=not args.no_cache,
            )
            feat = next(f for f in derived if f["kind"] == "feature")["data"]
            entry["attack_time_s"] = feat.get("qc.shape.attack_time_s")
            entry["late_energy_share"] = feat.get("qc.shape.late_energy_share")
            out.extend(derived)
            if ab is not None:
                from smplstream import cas, frames as F

                scores = ab.score_path(str(cas.get_path(audio["hash"])))
                entry["audiobox"] = scores
                out.append(F.feature_frame(
                    {ab.FEATURE_KEYS[k]: v for k, v in scores.items()},
                    role="audiobox", of=audio.get("id"), op="audiobox", op_version="audiobox@1",
                    params={}))
        except Exception as exc:
            eprint(f"qc: {audio.get('id')}: {exc}")
            out.append(error_frame("op_failed", str(exc), of=audio.get("id"), op="qc"))
            entry["error"] = str(exc)
            rc = 1
        entries.append(entry)

    if args.report:
        json.dump({"files": entries}, sys.stdout, indent=2)
        sys.stdout.write("\n")
        sys.stdout.flush()
        return rc
    emit(out)
    return rc
