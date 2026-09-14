"""`smpl plugin` CLI + the effect op.

    smpl plugin list [--json]
    smpl plugin describe <Effect> [--json] [--schema] [--param k=v ...]
    smpl read dry.wav | smpl plugin --name Compressor --param threshold_db=-20 | smpl write wet.wav

The bare form reads frames from stdin, passes every input frame through unchanged (tool
contract → *Passthrough*), then appends the wet `audio` frame. `pedalboard` is this tool's one
heavy dep, in its OWN venv; if it is missing the CLI emits passthrough + a clean `unsupported`
error frame on stdout and an install hint on stderr, exit 0 — never a traceback."""

from __future__ import annotations

import io
import sys
from typing import Optional

from . import descriptor as D, hostbuiltin as host

# --- the op: an upstream `audio` frame → a wet `audio` frame, memoized -------------------
# memo_key inputs (spec → *Memoization*): op "plugin"; op_version "plugin@1+pedalboard:<lib
# version>+<EffectName>" (library version folded in); inputs [upstream audio hash]; params the
# EFFECTIVE dict (defaults filled before hashing, per the spec); env_fp "" — pedalboard is an
# in-process pip dep, already pinned by op_version. A hit re-emits the cached output hash
# WITHOUT rendering and marks the wet frame `memo: hit`; a hit whose blob was GC'd degrades to
# a miss in `memostore.lookup`, so a dangling hash is never served. `cacheable: false` skips
# the memo on both sides.
ENV_FINGERPRINT = ""


def memo_key(effect: str, input_hash: str, effective: dict) -> str:
    from smplstream import memo

    return memo.memo_key(host.OP, host.op_version(effect), [input_hash],
                         params=effective, env_fingerprint=ENV_FINGERPRINT)


def _emit(hash_: str, *, src: dict, effect: str, params: dict, role: Optional[str]) -> dict:
    from smplstream import cas, frames as F

    meta = cas.read_meta(hash_) or {}
    base = src.get("role") or "audio"
    return F.audio_frame(
        hash_, sr=meta.get("sr", 0), ch=meta.get("ch", 0), dur=meta.get("dur", 0.0),
        role=role or (base if base.endswith(".wet") else f"{base}.wet"),
        of=src.get("id"), lineage=[src["id"]] if src.get("id") else None,
        op=host.OP, op_version=host.op_version(effect), params=params, fmt=meta.get("fmt"))


def apply_effect(src: dict, effect: str, params: Optional[dict] = None, *,
                 role: Optional[str] = None, use_cache: bool = True) -> dict:
    """Run one built-in effect over an upstream `audio` frame → the wet `audio` frame.

    Raises :class:`hostbuiltin.EffectError` for an unknown effect or a bad parameter.
    """
    import numpy as np
    import soundfile as sf

    from smplstream import cas, memostore

    effective = host.coerce_params(effect, params)
    desc = D.describe(effect, effective)
    out_params = {"effect": effect, "params": effective, "state_hash": desc["state_hash"],
                  "descriptor_hash": desc["descriptor_hash"], "cacheable": desc["cacheable"]}

    mkey = memo_key(effect, src["hash"], effective) if desc["cacheable"] else None
    if mkey is not None and use_cache:
        cached = memostore.get_json(mkey)
        if cached and cas.exists(cached.get("hash", "")):
            return _emit(cached["hash"], src=src, effect=effect, role=role,
                         params={**out_params, "memo": "hit"})

    samples, sr = sf.read(str(cas.get_path(src["hash"])), dtype="float32", always_2d=True)
    wet = host.render(effect, samples, sr, effective)
    buf = io.BytesIO()  # WAV back-patches its RIFF header → needs a seekable sink
    sf.write(buf, np.ascontiguousarray(wet), int(sr), format="WAV", subtype="FLOAT")
    h = cas.put_audio_bytes(buf.getvalue())
    if mkey is not None:
        memostore.put_json(mkey, {"hash": h}, op=host.OP, op_version=host.op_version(effect))
    return _emit(h, src=src, effect=effect, role=role, params={**out_params, "memo": "miss"})


# --- CLI ---------------------------------------------------------------------------------

def _build_parser():
    import argparse

    p = argparse.ArgumentParser(prog="smpl plugin",
                                description="host a built-in audio effect as an smpl op")
    sub = p.add_subparsers(dest="cmd")
    lst = sub.add_parser("list", help="list the built-in effects this host can run")
    lst.add_argument("--json", action="store_true", help="emit one JSON object per effect")
    desc = sub.add_parser("describe", help="print a plugin descriptor")
    desc.add_argument("name", nargs="?", help="effect name (e.g. Compressor)")
    desc.add_argument("--json", action="store_true", help="machine-readable descriptor")
    desc.add_argument("--schema", action="store_true", help="print the descriptor JSON schema")
    desc.add_argument("--param", action="append", default=[], metavar="K=V")
    p.add_argument("--name", help="effect to apply to the upstream audio frame")
    p.add_argument("--param", action="append", default=[], metavar="K=V",
                   help="effect parameter (repeatable), e.g. --param threshold_db=-20")
    p.add_argument("--state", metavar="blake3:…", help="load params from a CAS state blob")
    p.add_argument("--role", help="output role (default: <upstream role>.wet)")
    p.add_argument("--no-cache", action="store_true",
                   help="bypass the memo lookup (still records the result)")
    return p


def _parse_params(items: list[str]) -> dict:
    """`--param k=v` → dict. Values stay strings; the host coerces against declared types."""
    out: dict = {}
    for item in items:
        if "=" not in item:
            sys.stderr.write(f"smpl plugin: ignoring malformed --param {item!r} (need k=v)\n")
            continue
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _read_input_frames() -> list[dict]:
    from smplstream import ndjson

    raw = b"" if sys.stdin.isatty() else sys.stdin.buffer.read()
    return list(ndjson.read_frames(io.BytesIO(raw))) if raw.lstrip()[:1] == b"{" else []


def _write(frames: list[dict]) -> None:
    from smplstream import ndjson

    ndjson.write_frames(frames)
    sys.stdout.buffer.flush()


def _fail(frames: list[dict], code: str, msg: str, of: Optional[str] = None, rc: int = 1) -> int:
    """Passthrough, then one error frame on stdout and the message on stderr."""
    from smplstream import error_frame

    sys.stderr.write(f"smpl plugin: {msg}\n")
    _write([*frames, error_frame(code, msg, of=of, op=host.OP)])
    return rc


def _emit_unsupported(frames: list[dict]) -> int:
    """Degrade path: the pedalboard native extension is absent or unimportable. Exit 0 — an
    unsupported op is one frame, one failure, so the pipe stays resilient."""
    sys.stderr.write(f"smpl plugin: install with: {host.INSTALL_HINT}\n")
    return _fail(frames, "unsupported", "plugin host unavailable: the `pedalboard` package "
                 f"failed to import. Install with `{host.INSTALL_HINT}`.", rc=0)


def _cmd_list(args) -> int:
    import json

    if not host.available():
        return _emit_unsupported([])
    for name in host.list_effects():
        print(json.dumps(dict(name=name, format="builtin", vendor="pedalboard", kind="effect"))
              if args.json else name)
    return 0


def _cmd_describe(args) -> int:
    import json

    if args.schema:
        print(json.dumps(D.SCHEMA, indent=2, sort_keys=True))
        return 0
    if not args.name:
        sys.stderr.write("smpl plugin describe: need an effect name (or --schema)\n")
        return 2
    if not host.available():
        return _emit_unsupported([])
    try:
        desc = D.describe(args.name, _parse_params(args.param))
    except host.EffectError as exc:
        sys.stderr.write(f"smpl plugin: {exc}\n")
        return 2
    print(json.dumps(desc, sort_keys=True, indent=None if args.json else 2))
    return 0


def _cmd_apply(args) -> int:
    from smplstream.errors import ResolutionError
    from smplstream.select import select

    input_frames = _read_input_frames()
    if not host.available():
        return _emit_unsupported(input_frames)
    if not args.name:
        return _fail(input_frames, "op_failed", "need --name <Effect> (see `smpl plugin list`)")
    audio = select(input_frames, kind="audio", predicate=lambda f: bool(f.get("hash")), mode="last")
    if not audio:
        return _fail(input_frames, "not_found", "no upstream `audio` frame on stdin")
    src = audio[0]

    params: dict = {}
    if args.state:
        try:
            params.update(D.load_state(args.state))
        except Exception as exc:
            return _fail(input_frames, "not_found",
                         f"cannot load --state {args.state}: {exc}", of=src.get("id"))
    params.update(_parse_params(args.param))  # explicit --param overrides the saved state
    try:
        wet = apply_effect(src, args.name, params, role=args.role, use_cache=not args.no_cache)
    except (host.EffectError, ResolutionError, FileNotFoundError, OSError) as exc:
        return _fail(input_frames, "op_failed", str(exc), of=src.get("id"))
    _write([*input_frames, wet])
    return 0


def main(argv: list[str] | None = None) -> int:
    # SIGPIPE hygiene: a downstream `head` closing the pipe must not raise a traceback onto
    # stdout (which would emit a truncated final NDJSON line — a fatal read error).
    try:
        import signal

        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except (ImportError, ValueError, AttributeError):
        pass  # not POSIX — best effort

    args = _build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.cmd == "list":
        return _cmd_list(args)
    return _cmd_describe(args) if args.cmd == "describe" else _cmd_apply(args)


if __name__ == "__main__":
    raise SystemExit(main())
