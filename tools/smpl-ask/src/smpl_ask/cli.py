"""`smpl ask` / `smpl-ask` CLI.

Ask:        smpl ask --q <question-id> [--backend gemini] [--model M] [--role R | --a RA --b RB]
                     [--dry-run] [--allow-routed] [--ledger FILE]
Questions:  smpl ask questions [--json]
Calibrate:  smpl-ask calibrate [--model M] [--only q1,q2] [--include-routed] [--estimate | --yes]
                     [--workdir DIR] [--out report.json]
Ledger:     smpl ask export-ledger [--out FILE]      (claims, optionally crosschecked, on stdin)

`ask` passes every input frame through, then appends one `text` frame (role `claim`) per question.
Without a key or backend it degrades to a single `unsupported` error frame plus a stderr hint,
exit 0, like the other heavy tools. Stdout is frames; stderr is for humans (and the cost line).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import questions as Q


def _eprint(*a) -> None:
    print(*a, file=sys.stderr)


def _unsupported(frames: list[dict], message: str, hint: str, *, of=None) -> int:
    from smplstream import error_frame, ndjson

    ndjson.write_frames([*frames, error_frame("unsupported", message, of=of, op="ask")])
    sys.stdout.buffer.flush()
    _eprint(f"smpl ask: {message}\n  hint: {hint}")
    return 0


def _read_stdin() -> list[dict]:
    from smplstream import ndjson

    return [] if sys.stdin.isatty() else list(ndjson.read_frames())


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="smpl ask", description="ask an audio LLM one narrow question; emit claim frames")
    p.add_argument("--q", dest="qid", required=True, metavar="QUESTION_ID", help="a question id (see `smpl ask questions`)")
    p.add_argument("--backend", default="gemini", help="gemini (default) | local (stub, not built)")
    p.add_argument("--model", help="model id (default $SMPL_ASK_MODEL or gemini-3.8-flash)")
    p.add_argument("--role", help="clip questions: select the audio frame with this role (default: the last audio frame)")
    p.add_argument("--a", dest="role_a", help="A/B questions: role of clip A (default: second-to-last audio frame)")
    p.add_argument("--b", dest="role_b", help="A/B questions: role of clip B (default: last audio frame)")
    p.add_argument("--max-seconds", type=float, default=120.0, help="crop each clip to this length (default 120)")
    p.add_argument("--thinking-level", help="pass generationConfig.thinkingConfig.thinkingLevel (model-dependent)")
    p.add_argument("--allow-routed", action="store_true",
                   help="ask a smpl-routed (stereo / fidelity) question anyway; the model hears a mono downmix")
    p.add_argument("--dry-run", action="store_true", help="build the request and print the token / cost estimate; no call, no key")
    p.add_argument("--ledger", help="also write the claims as ear-ledger questions (JSON) to this file")
    return p


def _pick_audio(frames: list[dict], q: Q.Question, args):
    from smplstream import select as S

    audios = [f for f in frames if f.get("kind") == "audio"]
    if q.kind == "clip":
        if args.role:
            sel = S.select(frames, kind="audio", role=args.role, mode="last")
            return sel or None
        return audios[-1:] or None
    if args.role_a or args.role_b:
        a = S.select(frames, kind="audio", role=args.role_a, mode="last") if args.role_a else audios[-2:-1]
        b = S.select(frames, kind="audio", role=args.role_b, mode="last") if args.role_b else audios[-1:]
        return [a[0], b[0]] if a and b else None
    return audios[-2:] if len(audios) >= 2 else None


def _ask(argv: list[str]) -> int:
    from smplstream import cas, error_frame, ndjson

    from . import ask as A
    from . import backends as B
    from . import ledger as L

    args = _build_parser().parse_args(argv)
    try:
        q = Q.get(args.qid)
    except KeyError as exc:
        _eprint(f"smpl ask: {exc.args[0]}")
        return 2
    frames = _read_stdin()
    out = list(frames)

    chosen = _pick_audio(frames, q, args)
    if not chosen:
        need = "two audio frames (or --a/--b)" if q.kind == "ab" else "an audio frame (or --role)"
        _eprint(f"smpl ask: {q.id} needs {need} on stdin")
        return 2

    try:
        backend = B.get_backend(args.backend, args.model, **({"thinking_level": args.thinking_level} if args.backend == "gemini" else {}))
    except B.Unsupported as exc:
        return _unsupported(frames, str(exc), exc.hint, of=chosen[0].get("id"))

    if q.route and q.route != "ear" and not args.allow_routed:
        _eprint(f"smpl ask: {q.id} is routed to smpl {q.smpl_op}: the model hears a mono 16 kHz downmix, "
                f"so use `smpl {q.smpl_op}` (--allow-routed asks anyway)")
        out.append(A.routed_claim_frame(q, chosen[0]["id"], backend.model))
        ndjson.write_frames(out)
        sys.stdout.buffer.flush()
        return 0

    try:
        wavs = [A.prepare_audio(cas.get_path(f["hash"]), max_seconds=args.max_seconds)[0] for f in chosen]
    except Exception as exc:
        _eprint(f"smpl ask: could not read audio: {exc}")
        out.append(error_frame("op_failed", f"could not read audio: {exc}", of=chosen[0].get("id"), op="ask"))
        ndjson.write_frames(out)
        sys.stdout.buffer.flush()
        return 1

    if args.dry_run:
        import io

        import soundfile as sf

        from .backends import AUDIO_TOKENS_PER_SECOND, Usage, cost_usd

        secs = sum(sf.info(io.BytesIO(w)).duration for w in wavs)
        calls = 2 if q.kind == "ab" else 1
        u = Usage(text_tokens=175 * calls, audio_tokens=int(secs * AUDIO_TOKENS_PER_SECOND) * calls,
                  output_tokens=950 * calls)
        c = cost_usd(backend.model, u)
        _eprint(f"smpl ask --dry-run: {q.id} ({q.kind}) {calls} call(s) to {backend.model}; ~{u.audio_tokens} audio + "
                f"{u.text_tokens} text tokens in, ~{u.output_tokens} out (incl. assumed thinking); "
                + ("price unknown for this model" if c is None else f"~${c:.4f}"))
        ndjson.write_frames(out)
        sys.stdout.buffer.flush()
        return 0

    try:
        claim = A.ask(backend, q, wavs)
    except B.Unsupported as exc:
        return _unsupported(frames, str(exc), exc.hint, of=chosen[0].get("id"))
    claim.sent = {"sr_hz": A.SEND_SR, "channels": 1, "max_seconds": args.max_seconds}
    _eprint(A.log_line(claim))
    for w in claim.warnings:
        _eprint(f"smpl ask: warning: {w}")
    out.append(A.claim_frame(claim, [f["id"] for f in chosen]))
    ndjson.write_frames(out)
    sys.stdout.buffer.flush()
    if args.ledger:
        Path(args.ledger).write_text(json.dumps(L.export(out), indent=2) + "\n", encoding="utf-8")
        _eprint(f"smpl ask: wrote ear-ledger questions to {args.ledger}")
    return 0


def _questions(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="smpl ask questions")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    rows = Q.bank().values()
    if args.json:
        print(json.dumps([{"id": q.id, "kind": q.kind, "route": q.route, "smpl_op": q.smpl_op,
                           "answer": q.answer, "timestamps": q.timestamps, "prompt": q.prompt} for q in rows], indent=2))
        return 0
    for q in rows:
        route = {None: "model", "ear": "model (ear-only)", "smpl": f"smpl {q.smpl_op} (routed)"}[q.route]
        print(f"{q.id:<18} {q.kind:<5} {route:<22} {q.prompt}")
    return 0


def _export_ledger(argv: list[str]) -> int:
    from . import ledger as L

    p = argparse.ArgumentParser(prog="smpl ask export-ledger")
    p.add_argument("--out", help="write here instead of stdout")
    args = p.parse_args(argv)
    doc = L.export(_read_stdin())
    text = json.dumps(doc, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        _eprint(f"smpl ask: {len(doc['items'])} ledger question(s) -> {args.out}")
    else:
        sys.stdout.write(text)
    return 0


def _calibrate(argv: list[str]) -> int:
    import tempfile

    from . import backends as B
    from . import calibration as C

    p = argparse.ArgumentParser(prog="smpl-ask calibrate", description="score the model on synthetic clips with known answers")
    p.add_argument("--backend", default="gemini")
    p.add_argument("--model")
    p.add_argument("--only", help="comma-separated question ids")
    p.add_argument("--include-routed", action="store_true", help="also ask the smpl-routed stereo / phase questions (measures the downmix blindness)")
    p.add_argument("--workdir", help="where the clips are written (default: a temp dir)")
    p.add_argument("--estimate", action="store_true", help="print the token / cost estimate and exit; no call, no key")
    p.add_argument("--yes", action="store_true", help="required to spend money on a live run")
    p.add_argument("--think-tokens", type=int, default=C.EST_THINK_TOKENS, help="assumed hidden thinking tokens per call, for the estimate")
    p.add_argument("--thinking-level")
    p.add_argument("--max-seconds", type=float, default=120.0)
    p.add_argument("--out", help="write the full report JSON here")
    args = p.parse_args(argv)

    work = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="smpl-ask-cal-"))
    cases = C.build_cases(work, include_routed=args.include_routed)
    if args.only:
        want = set(args.only.split(","))
        unknown = want - {c.question_id for c in cases}
        if unknown:
            _eprint(f"smpl-ask calibrate: no cases for {sorted(unknown)} (routed ones need --include-routed)")
            return 2
        cases = [c for c in cases if c.question_id in want]

    try:
        backend = B.get_backend(args.backend, args.model, **({"thinking_level": args.thinking_level} if args.backend == "gemini" else {}))
    except B.Unsupported as exc:
        _eprint(f"smpl-ask calibrate: {exc}\n  hint: {exc.hint}")
        return 2
    est = C.estimate(cases, backend.model, think_tokens=args.think_tokens, max_seconds=args.max_seconds)
    _eprint(f"calibration set: {len(cases)} cases, {est['calls']} calls to {backend.model}, clips in {work}\n"
            f"  estimate: {est['audio_tokens']} audio + {est['text_tokens']} text tokens in; "
            f"cost ${est['cost_usd']} assuming {args.think_tokens} thinking tokens/call "
            f"(${est['cost_usd_without_thinking']} with none)")
    if args.estimate:
        print(json.dumps(est, indent=2))
        return 0
    if not args.yes:
        _eprint("smpl-ask calibrate: a live run spends money; re-run with --yes (or --estimate to just price it)")
        return 2
    try:
        backend.require_key()
        outcomes = C.run(backend, cases, max_seconds=args.max_seconds, log=_eprint)
    except B.Unsupported as exc:
        _eprint(f"smpl-ask calibrate: {exc}\n  hint: {exc.hint}")
        return 2
    rep = C.report(outcomes, model=backend.model, skipped_routed=not args.include_routed,
                   extra={"price_source": B.PRICE_SOURCE, "estimate": est})
    print(C.format_report(rep))
    if args.out:
        C.write_report(rep, args.out)
        _eprint(f"smpl-ask calibrate: report written to {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    sub = {"questions": _questions, "calibrate": _calibrate, "export-ledger": _export_ledger}
    if argv and argv[0] in sub:
        return sub[argv[0]](argv[1:])
    return _ask(argv)


if __name__ == "__main__":
    raise SystemExit(main())
