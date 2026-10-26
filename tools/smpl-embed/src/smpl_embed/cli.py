"""`smpl embed` / `smpl index` CLI — vector frames + a FAISS similarity index.

Surfaces TWO verbs via argv dispatch (like smpl-gen's `models` subcommand):

  smpl embed  [--model M] [--role R] [--strict]      filter: audio frames → vector frames
  smpl index  build <name> [--all]                   sink:   build FAISS index over vectors
  smpl index  query <name> [--k N] [--model M]        sink:   query nearest ids
  smpl embed  text "prompt" ["prompt" …] [--model M]  source: text → text-embedding vectors
  smpl embed  models list | install <id> | update <id> | rm <id>

Two-tier discipline: the heavy ML stack (torch + transformers) and faiss are lazy-imported
inside `backends`; a missing dep/model degrades to a clean `error` frame (code
`unsupported`) on STDOUT plus a stderr line with the exact install command. Nothing heavy is
imported at module top — the default light install runs and degrades, never hangs.

embed contract:
  - Reads `audio` frames from stdin, embeds each, emits a `vector` frame per audio frame.
  - Passes EVERY input frame through unchanged first (tool contract / passthrough).
  - dim > 64 → the vector goes to CAS as a binary `.npy` blob referenced by `hash`
    (`media: application/x-npy`), NEVER pickle. dim <= 64 inlines as `data`.
  - `meta` carries `model`, `dim`, `dtype` (spec → *Frame kinds / vector*).
  - `op_version` incorporates the model's weights identity (spec → *Memoization*).
"""

from __future__ import annotations

import io
import sys

from . import backends


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _read_stdin_frames() -> list[dict]:
    """Read NDJSON frames from stdin (empty list if a tty / no input)."""
    from smplstream import ndjson

    if sys.stdin.isatty():
        return []
    data = sys.stdin.buffer.read()
    if not data.strip():
        return []
    return list(ndjson.read_frames(io.BytesIO(data)))


def _emit_unsupported(exc: "backends.UnsupportedError", *, op: str, of: str | None = None) -> None:
    """Emit a clean `unsupported` error frame (stdout) + the exact install hint (stderr)."""
    from smplstream import error_frame, ndjson

    ndjson.write_frame(error_frame("unsupported", str(exc), op=op, of=of))
    sys.stdout.buffer.flush()
    sys.stderr.write(f"smpl {op}: unsupported — install with: {exc.install_hint}\n")


def _vector_frame_for(
    vec, *, model_id: str, spec: dict, op_version: str, of: str | None, text: str | None = None
):
    """Build a `vector` frame from a 1-D float32 numpy vector.

    dim > 64 → store the binary `.npy` in CAS (`cas.put_blob(npy_bytes,'application/x-npy')`)
    and reference it by `hash`, NEVER pickle. dim <= 64 inlines as `data`. `meta` always
    carries model + dim + dtype; `params` carries the model id, dim, backend and HF repo (and
    the prompt for a text vector), and the role is ``ml.<backend>.embedding`` /
    ``ml.<backend>.text-embedding``.
    """
    import numpy as np

    from smplstream import cas, frames as F
    from smplstream.frames import VECTOR_INLINE_MAX_DIM

    vec = np.ascontiguousarray(np.asarray(vec, dtype=np.float32)).reshape(-1)
    dim = int(vec.shape[0])
    dtype = "float32"
    params = {"model": model_id, "dim": dim, "backend": spec["backend"], "hf_repo": spec["hf_repo"]}
    if text is not None:
        params["text"] = text
    role = backends.vector_role(spec, text=text is not None)

    if dim > VECTOR_INLINE_MAX_DIM:
        buf = io.BytesIO()
        np.save(buf, vec, allow_pickle=False)  # binary .npy, NEVER pickle
        h = cas.put_blob(buf.getvalue(), "application/x-npy")
        return F.vector_frame(
            model=model_id, dim=dim, dtype=dtype, hash=h, media="application/x-npy",
            role=role, of=of, op="embed", op_version=op_version, params=params,
        )
    return F.vector_frame(
        model=model_id, dim=dim, dtype=dtype, data=[float(x) for x in vec.tolist()],
        role=role, of=of, op="embed", op_version=op_version, params=params,
    )


def _materialize_pcm(frame: dict):
    """Resolve an audio frame's CAS blob → mono float32 PCM + sample rate.

    Returns (pcm_f32_mono, sr) or raises backends.UnsupportedError if the blob is missing
    (soundfile isn't a heavy dep — it's a default dependency — but a missing blob is still
    an `unsupported` outcome for the embed op on that frame).
    """
    import numpy as np
    import soundfile as sf

    from smplstream import cas

    h = frame.get("hash")
    if not h:
        raise backends.UnsupportedError(
            "audio frame has no CAS hash to embed", install_hint="(produce audio first)"
        )
    try:
        path = cas.get_path(h)
    except FileNotFoundError as exc:
        raise backends.UnsupportedError(
            f"audio blob not in CAS for {h}: {exc}", install_hint="(re-run the producing op)"
        ) from exc
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = np.asarray(data, dtype=np.float32).mean(axis=1)  # downmix for the encoder
    return mono, int(sr)


# ---------------------------------------------------------------------------
# `smpl embed` — filter: audio frames → vector frames
# ---------------------------------------------------------------------------


def _embed(argv: list[str]) -> int:
    import argparse

    from smplstream import ndjson

    p = argparse.ArgumentParser(prog="smpl embed", description="embed audio frames → vectors")
    p.add_argument("--model", help="embedding model (default: $SMPL_EMBED_MODEL or clap)")
    p.add_argument("--role", default=None, help="only embed audio frames with this role")
    args = p.parse_args(argv)

    frames = _read_stdin_frames()

    # Pass every input frame through FIRST (tool contract / passthrough + stream ordering).
    out: list[dict] = list(frames)

    # Resolve the model id up front so `op_version` (and a model-resolution failure) is one
    # decision, not per-frame. A bad model id degrades to a single `unsupported` frame.
    try:
        model_id, spec = backends.resolve_model(args.model)
    except backends.UnsupportedError as exc:
        ndjson.write_frames(out)
        _emit_unsupported(exc, op="embed")
        return 0  # one-frame-one-failure: degrade, don't crash the pipe
    op_version = backends.op_version(model_id, spec)

    targets = [
        f for f in frames
        if f.get("kind") == "audio" and f.get("hash")
        and (args.role is None or f.get("role") == args.role)
    ]

    import time

    times: list[float] = []
    for f in targets:
        of = f.get("id")
        try:
            pcm, sr = _materialize_pcm(f)
            t0 = time.perf_counter()
            vec = backends.embed(pcm, sr, model_id=model_id)
            times.append(time.perf_counter() - t0)
            out.append(_vector_frame_for(
                vec, model_id=model_id, spec=spec, op_version=op_version, of=of))
        except backends.UnsupportedError as exc:
            # One frame, one failure: emit a per-frame `unsupported` error and keep going.
            from smplstream import error_frame

            out.append(error_frame("unsupported", str(exc), op="embed", of=of))
            sys.stderr.write(f"smpl embed: unsupported — install with: {exc.install_hint}\n")

    if times:
        # The first call pays the model load; report it apart from the steady per-item cost.
        steady = times[1:] or times
        sys.stderr.write(
            f"smpl embed: {model_id} {len(times)} item(s), first {times[0]:.2f} s (incl. load), "
            f"then {sum(steady) / len(steady):.3f} s/item\n"
        )
    ndjson.write_frames(out)
    sys.stdout.buffer.flush()
    return 0


# ---------------------------------------------------------------------------
# `smpl embed text` — text prompts → vectors in the model's joint audio-text space
# ---------------------------------------------------------------------------


def _text(argv: list[str]) -> int:
    """Embed text prompts (CLAP / MuQ-MuLan text towers) → `ml.<backend>.text-embedding`
    vector frames, one per prompt, with the prompt in ``params.text``. Inbound frames pass
    through first, so `… | smpl embed text "a kick" | smpl probe …` keeps the audio vectors."""
    import argparse

    from smplstream import ndjson

    p = argparse.ArgumentParser(prog="smpl embed text", description="embed text prompts → vectors")
    p.add_argument("prompts", nargs="+", help="text prompt(s)")
    p.add_argument("--model", help="model with a text tower (default: $SMPL_EMBED_MODEL or clap)")
    args = p.parse_args(argv)

    out: list[dict] = list(_read_stdin_frames())
    try:
        model_id, spec = backends.resolve_model(args.model)
        vecs = backends.embed_text(args.prompts, model_id=model_id)
    except backends.UnsupportedError as exc:
        ndjson.write_frames(out)
        _emit_unsupported(exc, op="embed")
        return 0
    op_version = backends.op_version(model_id, spec)
    for prompt, vec in zip(args.prompts, vecs):
        out.append(_vector_frame_for(
            vec, model_id=model_id, spec=spec, op_version=op_version, of=None, text=prompt))
    ndjson.write_frames(out)
    sys.stdout.buffer.flush()
    return 0


# ---------------------------------------------------------------------------
# `smpl index` — sink: build / query a FAISS index over vector frames
# ---------------------------------------------------------------------------


def _vector_values(frame: dict):
    """Resolve a `vector` frame to a 1-D float32 numpy vector (inline `data` or CAS `.npy`)."""
    import numpy as np

    from smplstream import cas

    if frame.get("data") is not None:
        return np.asarray(frame["data"], dtype=np.float32)
    h = frame.get("hash")
    path = cas.get_path(h)  # raises FileNotFoundError if missing → surfaced by caller
    arr = np.load(str(path), allow_pickle=False)  # binary .npy only, NEVER pickle
    return np.asarray(arr, dtype=np.float32).reshape(-1)


def _index(argv: list[str]) -> int:
    import argparse
    import json

    from smplstream import ndjson

    p = argparse.ArgumentParser(prog="smpl index", description="build/query a FAISS vector index")
    p.add_argument("action", choices=["build", "query"], nargs="?", default="build")
    p.add_argument("name", nargs="?", default="default", help="index name")
    p.add_argument("--k", type=int, default=10, help="query: number of neighbors")
    p.add_argument("--model", help="query: embed the single audio frame on stdin with this model")
    args = p.parse_args(argv)

    frames = _read_stdin_frames()
    vector_frames = [f for f in frames if f.get("kind") == "vector"]

    if args.action == "build":
        vectors, ids = [], []
        for vf in vector_frames:
            try:
                vectors.append(_vector_values(vf))
                ids.append(vf.get("id"))
            except FileNotFoundError as exc:
                sys.stderr.write(f"smpl index: skipping vector with missing blob: {exc}\n")
        try:
            info = backends.index_build(args.name, vectors, ids)
        except backends.UnsupportedError as exc:
            # Pass through inbound frames so the pipe's lineage survives, then the error frame.
            ndjson.write_frames(frames)
            _emit_unsupported(exc, op="index")
            return 0
        # A sink reports a `feature` frame (structured) summarizing the build.
        from smplstream import frames as F

        ndjson.write_frames(frames)
        ndjson.write_frame(
            F.feature_frame(
                {"index.name": info["name"], "index.count": info["count"], "index.dim": info["dim"]},
                role="index", op="index", op_version="index:faiss-flat-ip@1",
                params={"action": "build", "name": args.name},
            )
        )
        sys.stdout.buffer.flush()
        sys.stderr.write(f"smpl index: built {info['name']} ({info['count']} vectors, dim {info['dim']})\n")
        return 0

    # query
    query_vec = None
    if vector_frames:
        try:
            query_vec = _vector_values(vector_frames[-1])  # last-wins
        except FileNotFoundError as exc:
            sys.stderr.write(f"smpl index: query vector blob missing: {exc}\n")
    if query_vec is None:
        # No vector on stdin: optionally embed the single audio frame (degrades if no model).
        audio = [f for f in frames if f.get("kind") == "audio" and f.get("hash")]
        if audio:
            try:
                model_id, _ = backends.resolve_model(args.model)
                pcm, sr = _materialize_pcm(audio[-1])
                query_vec = backends.embed(pcm, sr, model_id=model_id)
            except backends.UnsupportedError as exc:
                _emit_unsupported(exc, op="index")
                return 0
    if query_vec is None:
        sys.stderr.write("smpl index query: no vector or audio frame on stdin to query with\n")
        return 2

    try:
        hits = backends.index_query(args.name, query_vec, k=args.k)
    except backends.UnsupportedError as exc:
        _emit_unsupported(exc, op="index")
        return 0
    for hit in hits:
        print(json.dumps(hit))
    return 0


# ---------------------------------------------------------------------------
# `smpl embed models …` — registry management
# ---------------------------------------------------------------------------


def _models(argv: list[str]) -> int:
    import json

    action = argv[0] if argv else "list"
    if action == "list":
        for row in backends.list_models():
            print(json.dumps(row))
        return 0
    if action in ("install", "update") and len(argv) > 1:
        try:
            info = backends.install_model(argv[1])
        except backends.UnsupportedError as exc:
            sys.stderr.write(f"smpl embed models: {exc}\n")
            return 1
        sys.stderr.write(f"smpl embed models: {action}ed {argv[1]} → {info['path']}\n")
        return 0
    if action == "rm" and len(argv) > 1:
        ok = backends.remove_model(argv[1])
        sys.stderr.write(
            f"smpl embed models: {'removed ' + argv[1] if ok else 'no such model ' + argv[1]}\n"
        )
        return 0 if ok else 1
    sys.stderr.write("usage: smpl embed models list|install <id>|update <id>|rm <id>\n")
    return 2


# ---------------------------------------------------------------------------
# entrypoint — argv dispatch across the two verbs (`embed` default, `index`)
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # `smpl-embed` is reachable as both `smpl embed` and `smpl index`. The `smpl` launcher
    # strips the verb, so we sniff it back: an `index` first token routes to the sink; a
    # `models` first token routes to the registry; everything else is the `embed` filter.
    if argv and argv[0] == "index":
        return _index(argv[1:])
    if argv and argv[0] == "embed":
        argv = argv[1:]
    if argv and argv[0] == "models":
        return _models(argv[1:])
    if argv and argv[0] == "text":
        return _text(argv[1:])
    return _embed(argv)


if __name__ == "__main__":
    raise SystemExit(main())
