"""`smpl mix` — multi-input arrangement/render: the suite's **combinator** (N audio → 1).

The third frame-flow shape. `read`/`gen` are *sources*, the analysis/edit ops are
*filters* (1→1), and `mix` is the *combinator*: it selects N input `audio` frames from the
accumulated stream by role/id/hash, places them on a sample-accurate timeline, sums them,
and emits **one** rendered `audio` frame carrying ``lineage`` over every input plus
``op: mix`` / ``op_version`` / ``params`` (the resolved arrangement)::

    smpl gen "kick" | smpl gen "hat" | smpl mix --arrange loop.smplset.json | smpl write out.wav

**Session = canonical data on disk; `smpl mix` = a stateless control plane over it**
(agent-as-UI, cf. `mpc` over `mpd`). Verbs read the session file, mutate it, write it
back — no daemon, no hidden state — and `render` is a pure function of
(session ∪ CAS content), so a written session round-trips to a byte-identical render::

    smpl mix init      --session set.smplset.json --sr 48000 --bpm 120
    smpl mix add-clip  --session set.smplset.json --source role:stem:drums --at sample:0
    smpl mix add-clip  --session set.smplset.json --source role:stem:bass  --at marker:beat#4
    smpl mix set-gain  --session set.smplset.json --track main --index 1 --db -3
    smpl mix render    --session set.smplset.json --dry-run     # plan only, no bytes
    smpl mix render    --session set.smplset.json               # memoized render

``--dry-run`` emits the resolved arrangement as a `control` frame (role ``mix.plan``) and
renders nothing — the spec's *lazy frames* instinct: bytes are computed only when someone
asks for them. See ``docs/mix.md`` for the format, the verb set, and the decision record.
"""

from __future__ import annotations

from .._common import emit, eprint, read_stdin_frames

HELP = "arrange + render N audio frames into one mixed audio frame (session control plane)"

VERBS = ("render", "init", "add-clip", "set-gain", "set-pan", "rm-clip", "show")

SESSION_ROLE = "mix.session"
PLAN_ROLE = "mix.plan"


def add_arguments(parser):
    parser.add_argument("verb", nargs="?", default="render", choices=VERBS,
                        help="control-plane verb (default: render)")
    parser.add_argument("--session", "-s", help="session file (canonical arrangement on disk)")
    parser.add_argument("--arrange", help="render this session file (read-only alias of --session)")
    parser.add_argument("--stream", action="store_true",
                        help="frame mode (implied when stdin is a pipe)")
    parser.add_argument("--clip", action="append", default=[], metavar="SPEC",
                        help="ad-hoc clip: 'source=role:stem:drums,at=sample:0,gain_db=-3,pan=-0.2'")
    parser.add_argument("--input", action="append", default=[], metavar="REF",
                        help="ad-hoc clip at sample 0 (role|id|blake3:hash|path); repeatable")
    parser.add_argument("--source", help="clip source ref (add-clip)")
    parser.add_argument("--at", default="sample:0",
                        help="position: sample:N | sec:F | bar:B.b.f | marker:<role>#<i>")
    parser.add_argument("--track", default="main", help="track name (default: main)")
    parser.add_argument("--index", type=int, help="clip index within the track")
    parser.add_argument("--gain-db", dest="gain_db", type=float, default=0.0, help="clip gain (dB)")
    parser.add_argument("--db", type=float, help="gain in dB (set-gain)")
    parser.add_argument("--pan", type=float, default=0.0, help="pan -1 (L) .. 1 (R)")
    parser.add_argument("--sr", type=int, help="session sample rate (default: first input's)")
    parser.add_argument("--ch", type=int, help="session channels (default: 2 if panned/stereo)")
    parser.add_argument("--bpm", type=float, help="session tempo (needed for bar.beat.frac)")
    parser.add_argument("--beats-per-bar", dest="beats_per_bar", type=float, default=4.0)
    parser.add_argument("--role-out", dest="role_out", default="mix",
                        help="role for the rendered frame (default: mix)")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true",
                        help="emit the resolved arrangement only; render nothing")


def _parse_clip_spec(spec: str) -> dict:
    """``source=role:x,at=sample:0,gain_db=-3,pan=-0.2,track=drums`` → `add_clip` kwargs."""
    from .. import mixsession as M

    fields: dict = {}
    for part in spec.split(","):
        if not part.strip():
            continue
        key, _, value = part.partition("=")
        fields[key.strip()] = value.strip()
    unknown = set(fields) - {"source", "at", "gain_db", "pan", "track"}
    if unknown:
        raise M.MixError(f"unknown clip-spec field(s) {sorted(unknown)} in {spec!r}")
    if "source" not in fields:
        raise M.MixError(f"clip spec needs source=…: {spec!r}")
    return {
        "source": fields["source"],
        "at": fields.get("at", "sample:0"),
        "gain_db": float(fields.get("gain_db", 0.0)),
        "pan": float(fields.get("pan", 0.0)),
        "track": fields.get("track", "main"),
    }


def _session_from_stream(inframes: list[dict]) -> dict | None:
    """A session carried on the stream: `control` role ``mix.session`` or `smpl pattern`'s
    `feature` role ``smplset`` (last-wins)."""
    from smplstream import cas, select as S

    from .. import mixsession as M

    for kind, role in (("control", SESSION_ROLE), ("feature", "smplset")):
        got = S.select(inframes, kind=kind, role=role, mode="last")
        if got:
            frame = got[0]
            data = frame.get("data")
            if data is None and frame.get("hash"):
                import json

                data = json.loads(cas.get_path(frame["hash"]).read_text())
            return M.normalize_session(data)
    return None


def _build_session(args, inframes: list[dict]):
    """Resolve the session for a render: file > stream > ad-hoc clips > every audio frame."""
    from .. import mixsession as M

    path = args.session or args.arrange
    session = M.load_session(path) if path else _session_from_stream(inframes)
    explicit = session is not None
    if session is None:
        session = M.new_session(sr=args.sr, ch=args.ch, bpm=args.bpm,
                                beats_per_bar=args.beats_per_bar)
    for key, value in (("sr", args.sr), ("ch", args.ch), ("bpm", args.bpm)):
        if value:
            session[key] = value

    for spec in args.clip:
        M.add_clip(session, **_parse_clip_spec(spec))
    for ref in args.input:
        M.add_clip(session, source=ref, at="sample:0", track=args.track)

    if not explicit and not any(t["clips"] for t in session["tracks"]):
        # Bare `… | smpl mix`: sum every audio frame in the stream from sample 0.
        for frame in inframes:
            if frame.get("kind") == "audio" and frame.get("hash"):
                M.add_clip(session, source=f"id:{frame['id']}", at="sample:0", track=args.track)
    return session


def _mutate(args, session):
    """Apply one control-plane verb to the session (pure; caller persists)."""
    from .. import mixsession as M

    if args.verb == "add-clip":
        if not args.source:
            raise M.MixError("add-clip needs --source")
        return M.add_clip(session, source=args.source, at=args.at, gain_db=args.gain_db,
                          pan=args.pan, track=args.track)
    if args.verb == "set-gain":
        if args.db is None:
            raise M.MixError("set-gain needs --db")
        return M.set_gain(session, db=args.db, track=args.track, clip=args.index)
    if args.verb == "set-pan":
        return M.set_pan(session, pan=args.pan, track=args.track, clip=args.index)
    if args.verb == "rm-clip":
        if args.index is None:
            raise M.MixError("rm-clip needs --index")
        return M.rm_clip(session, track=args.track, clip=args.index)
    raise M.MixError(f"unhandled verb {args.verb!r}")


def run(args) -> int:
    from smplstream import error_frame

    from .. import mixsession as M

    inframes = read_stdin_frames()
    out = list(inframes)  # passthrough first (spec → Stream ordering)

    try:
        if args.verb == "init":
            path = args.session or args.arrange
            if not path:
                raise M.MixError("init needs --session <path>")
            session = M.new_session(sr=args.sr, ch=args.ch, bpm=args.bpm,
                                    beats_per_bar=args.beats_per_bar)
            M.save_session(path, session)
            eprint(f"mix: wrote {path}")
            out.append(M.data_frame("control", SESSION_ROLE, session, params={"path": str(path)}))

        elif args.verb in ("add-clip", "set-gain", "set-pan", "rm-clip"):
            path = args.session or args.arrange
            if not path:
                raise M.MixError(f"{args.verb} needs --session <path> (the canonical session)")
            session = _mutate(args, M.load_session(path))
            M.save_session(path, session)
            out.append(M.data_frame("control", SESSION_ROLE, session, params={"path": str(path)}))

        elif args.verb == "show":
            path = args.session or args.arrange
            session = M.load_session(path) if path else _session_from_stream(inframes)
            if session is None:
                raise M.MixError("show needs --session <path> or a session frame on the stream")
            out.append(M.data_frame("control", SESSION_ROLE, session,
                                    params={"path": str(path)} if path else None))

        else:  # render
            session = _build_session(args, inframes)
            plan, extra = M.resolve_arrangement(session, inframes)
            out.extend(extra)  # ingested path sources, before the frame that cites them
            if args.dry_run:
                # Lazy: publish the resolved arrangement + the memo key it WOULD render to.
                out.append(M.data_frame(
                    "control", PLAN_ROLE, plan,
                    params={"memo_key": M.plan_memo_key(plan), "clips": len(plan["clips"]),
                            "length_samples": plan["length_samples"], "rendered": False},
                ))
                eprint(f"mix: dry-run — {len(plan['clips'])} clip(s), "
                       f"{plan['length_samples'] / plan['sr']:.3f}s, not rendered")
            else:
                frame = M.render(plan, role=args.role_out)
                out.append(frame)
                eprint(f"mix: {len(plan['clips'])} clip(s) → {frame['meta']['dur']:.3f}s"
                       f"{' (cache hit)' if frame['params']['cache_hit'] else ''}")

    except M.MixError as exc:
        eprint(f"mix: {exc}")
        out.append(error_frame("op_failed", str(exc), op="mix"))
        emit(out)
        return 1
    except Exception as exc:  # noqa: BLE001 — a render failure is one frame, one failure
        eprint(f"mix: {exc}")
        out.append(error_frame("op_failed", str(exc), op="mix"))
        emit(out)
        return 1

    emit(out)
    return 0
