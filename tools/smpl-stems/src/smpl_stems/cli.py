"""`smpl stems` CLI — a smplstream 1→many *filter* tool + model management.

Separation:  smpl stems [--model M] [--role-filter ROLE] [input.wav]
             … | smpl stems        (frame stream on stdin)
Models:      smpl stems models list | install <id> | update <id> | rm <id>

A filter that consumes ONE audio frame and emits N audio frames, one per stem
(role ``stem:drums|bass|vocals|other|guitar|piano|instrumental``, op ``demucs``).
Input resolves from (in priority order):
  1. a positional ``input.wav`` path argument, or
  2. the single (last-wins) audio frame on a stdin frame stream.

Models (``--model`` / ``$SMPL_STEMS_MODEL``; see ``backends.KNOWN_MODELS``):
  * ``htdemucs_6s`` — 6-stem drums/bass/vocals/other/guitar/piano (default), **local** tier.
  * ``htdemucs``    — 4-stem drums/bass/vocals/other, **local** tier.
  * ``bs-roformer`` — 2-stem vocals/instrumental (UVR BS-RoFormer, ~3 dB SDR over Demucs on
    vocals); heavier — route to the **mac-mini batch tier**. Routing is guidance, not
    enforced here.

Two-tier degrade path (HARD RULE): the heavy separator (Demucs via
``python-audio-separator`` → torch) is lazy-imported inside ``run()`` via
``backends``. When it is absent (the light default install), this CLI does NOT
crash with an ImportError and does NOT hang — it emits a clean ``error`` frame
(code ``unsupported``) to stdout AND writes the exact install command to stderr,
then exits 0 (a per-frame failure, not a fatal/usage error).

Cacheable note (spec → *Memoization*, NORMATIVE): Demucs is GPU/MPS-capable and
its inference is **not** guaranteed bit-deterministic across devices/seeds. The
emitted stem frames therefore carry ``params.cacheable = false`` so memoization
is skipped for these ops unless determinism is pinned. ``op_version`` still folds
the weights identity (see ``backends.op_version_for``) so that, were determinism
pinned, the cache key would correctly invalidate on a weights swap.
"""

from __future__ import annotations

import io
import sys

from . import backends


def _build_parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="smpl stems",
        description=(
            "separate one audio frame into N stem frames "
            "(drums/bass/vocals/other/guitar/piano, or vocals/instrumental for bs-roformer)"
        ),
    )
    p.add_argument("input", nargs="?", help="input audio file (else read a frame stream from stdin)")
    p.add_argument(
        "--model",
        help=(
            "separator model: htdemucs_6s (default, local) | htdemucs (local) | "
            "bs-roformer (2-stem vocals/instrumental, mac-mini batch tier) "
            "[default: $SMPL_STEMS_MODEL or htdemucs_6s]"
        ),
    )
    p.add_argument("--role", help="role of the input audio frame to separate (default: last-wins audio)")
    return p


def _read_stdin_bytes() -> bytes:
    return b"" if sys.stdin.isatty() else sys.stdin.buffer.read()


def _resolve_input(args, stdin_bytes: bytes):
    """Return (input_path, input_frame_id, passthrough_frames).

    Form 1: an explicit positional path is CAS'd and used directly (no stdin frames).
    Form 2: a frame stream on stdin → resolve the single (last-wins) audio frame, hand its
            CAS blob path to the separator, and pass every input frame through unchanged.
    """
    from smplstream import cas, ndjson, select as sel

    # Form 1: explicit path argument wins.
    if args.input:
        h = cas.put_audio_file(args.input)
        path = str(cas.get_path(h))
        meta = cas.read_meta(h) or {}
        frame = _input_frame_from_meta(h, meta)
        return path, frame, [frame]

    # Form 2: a frame stream on stdin.
    stripped = stdin_bytes.lstrip()
    if stripped[:1] != b"{":
        return None, None, []

    frames = list(ndjson.read_frames(io.BytesIO(stdin_bytes)))
    audio = sel.select(frames, kind="audio", role=args.role,
                       predicate=lambda f: bool(f.get("hash")), mode="last")
    if not audio:
        return None, None, frames
    target = audio[0]
    path = str(cas.get_path(target["hash"]))
    return path, target, frames


def _input_frame_from_meta(h: str, meta: dict) -> dict:
    """Build the source audio frame for a path-in invocation (so lineage stays resolvable)."""
    from smplstream import frames as F

    return F.audio_frame(
        h,
        sr=meta.get("sr", 0),
        ch=meta.get("ch", 1),
        dur=meta.get("dur", 0.0),
        role="source",
        fmt=meta.get("fmt"),
    )


def _emit_unsupported(install_hint: str, message: str, of: str | None) -> int:
    """Two-tier degrade: clean `unsupported` error frame to stdout + install hint to stderr."""
    from smplstream import error_frame, ndjson

    ndjson.write_frame(error_frame("unsupported", message, of=of, op="demucs"))
    sys.stdout.buffer.flush()
    sys.stderr.write(f"smpl stems: {message}\n")
    sys.stderr.write(f"smpl stems: install the separator with:\n  {install_hint}\n")
    return 0  # per-frame failure, not a fatal/usage error (spec → *Error model*)


def _separate(args) -> int:
    import soundfile as sf

    from smplstream import cas, error_frame, frames as F, ndjson

    stdin_bytes = _read_stdin_bytes()
    input_path, input_frame, passthrough = _resolve_input(args, stdin_bytes)

    if input_path is None:
        sys.stderr.write(
            "smpl stems: no input audio (pass a file path or pipe a frame stream with an audio frame)\n"
        )
        return 2  # usage error → non-zero (no resolvable input at all)

    of_id = input_frame.get("id") if input_frame else None

    # Pass input frames through FIRST so derived stem frames reference earlier ids (ordering).
    out = list(passthrough)

    backend = backends.get_backend(args.model)
    try:
        stems = backend.separate(input_path)
    except backends.UnsupportedBackend as exc:
        # Emit passthrough (so the pipe keeps the original), then the unsupported error frame.
        if out:
            ndjson.write_frames(out)
        return _emit_unsupported(exc.install_hint, str(exc), of_id)

    op_version = backend.op_version
    for stem_name, wav_path in stems:
        h = cas.put_audio_file(wav_path)
        meta = cas.read_meta(h) or {}
        out.append(
            F.audio_frame(
                h,
                sr=meta.get("sr", 0),
                ch=meta.get("ch", 1),
                dur=meta.get("dur", 0.0),
                role=backends.STEM_ROLES.get(stem_name, f"stem:{stem_name}"),
                op="demucs",
                op_version=op_version,
                of=of_id,
                lineage=[of_id] if of_id else None,
                params={
                    "model": backend.model,
                    "stem": stem_name,
                    # GPU/MPS Demucs inference is not guaranteed bit-deterministic across
                    # devices; skip memoization unless determinism is pinned (spec →
                    # *Memoization*: non-deterministic ops declare cacheable:false).
                    "cacheable": False,
                },
                fmt=meta.get("fmt"),
            )
        )
    ndjson.write_frames(out)
    sys.stdout.buffer.flush()
    return 0


def _models(argv: list[str]) -> int:
    import json

    action = argv[0] if argv else "list"
    if action == "list":
        for row in backends.list_models():
            print(json.dumps(row))
        return 0
    if action in ("install", "update") and len(argv) > 1:
        info = backends.install_model(argv[1])
        sys.stderr.write(f"smpl stems models: {action}ed {argv[1]} → {info['weights']}\n")
        return 0
    if action == "rm" and len(argv) > 1:
        ok = backends.remove_model(argv[1])
        sys.stderr.write(
            f"smpl stems models: {'removed ' + argv[1] if ok else 'no such model ' + argv[1]}\n"
        )
        return 0 if ok else 1
    sys.stderr.write("usage: smpl stems models list|install <id>|update <id>|rm <id>\n")
    return 2


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "models":
        return _models(argv[1:])
    args = _build_parser().parse_args(argv)
    try:
        return _separate(args)
    except BrokenPipeError:  # pipe hygiene (spec → *Error model / Pipe hygiene*)
        try:
            sys.stdout.close()
        except Exception:
            pass
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
