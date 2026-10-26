"""Embedding backends + a thin model registry, and the FAISS index manager.

Backends are a customizable install surface:

- ``clap`` — LAION-CLAP music checkpoint (``laion/larger_clap_music``) through
  ``transformers.ClapModel`` + ``ClapProcessor``: ``get_audio_features`` for audio,
  ``get_text_features`` for text, one shared 512-d space (timbre-leaning).
- ``muq-mulan`` — ``OpenMuQ/MuQ-MuLan-large`` through the ``muq`` package: a music-text
  joint space (512-d) trained on music, for musical similarity and text queries.
- ``mert-v1-*`` — MERT self-supervised music features (audio only, mean-pooled hidden state).

The model store path + default model resolve from env vars (`SMPL_EMBED_HOME`,
`SMPL_EMBED_MODEL`). Heavy deps (`torch` + `transformers` / `muq`, and `faiss`) are isolated
in THIS tool's own venv behind extras and are **lazy-imported inside `embed()` / index ops**
— never at module top — so the default install runs light. A missing dep/model raises
:class:`UnsupportedError`, which the CLI renders as a clean `unsupported` error frame + a
stderr install hint (two-tier model).

Weights are managed under `SMPL_EMBED_HOME` (ollama-style list / install / update / rm),
never as a pip dependency. `op_version` incorporates the **weights identity** (registry
id + revision, or the on-disk weights blake3) so a model upgrade invalidates the memo
cache instead of silently serving stale vectors (spec → *Memoization*).

Every vector this module returns is L2-normalized: the joint audio-text spaces score by
cosine, and normalizing once here means consumers (cohesion, coverage, probe) never have to
guess whether a vector is unit length.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Optional


class UnsupportedError(Exception):
    """A required heavy dep / model / binary is missing.

    Carries the exact install command so the CLI can emit a clean `unsupported` error frame
    (stdout) plus a human-facing stderr install hint, and never hang or import torch at top.
    """

    def __init__(self, message: str, *, install_hint: str):
        super().__init__(message)
        self.install_hint = install_hint


def embed_home() -> Path:
    return Path(os.environ.get("SMPL_EMBED_HOME", "~/.smpl/embed")).expanduser()


def default_model() -> str:
    return os.environ.get("SMPL_EMBED_MODEL", "clap")


# Hugging Face repo ids for the supported encoders. The pinned `revision` is part of the
# weights identity (and thus `op_version`) so a re-pin re-keys the cache. `backend` names the
# vector role (`ml.<backend>.embedding`); `text` says whether the model has a text tower.
_MODEL_CATALOG = {
    # LAION-CLAP music (HTSAT-base). The hub's transformers port, `laion/larger_clap_music`,
    # ships INITIALIZATION weights (BatchNorm identity, projection std 0.0064, logit scales
    # ~0): every input embeds to the same vector (pairwise cosine 1.000 on a kick, hat, bass
    # and snare). The original LAION checkpoint is fine, so we convert it once into a
    # `ClapModel` dir under SMPL_EMBED_HOME (see `_convert_laion_clap`), taking config,
    # tokenizer and feature extractor from `larger_clap_general` (same HTSAT-base shape).
    "clap": {
        "kind": "clap",
        "backend": "clap",
        "hf_repo": "lukewys/laion_clap",
        "revision": "b3708341862f581175dba5c356a4ebf74a9b6651",
        "laion_file": "music_audioset_epoch_15_esc_90.14.pt",
        "base_repo": "laion/larger_clap_general",
        "base_revision": "ada0c23a36c4e8582805bb38fec3905903f18b41",
        "dim": 512,
        "sr": 48000,
        "clip_s": 10.0,
        "text": True,
    },
    "clap-general": {
        "kind": "clap",
        "backend": "clap",
        "hf_repo": "laion/larger_clap_general",
        "revision": "ada0c23a36c4e8582805bb38fec3905903f18b41",
        "dim": 512,
        "sr": 48000,
        "clip_s": 10.0,
        "text": True,
    },
    "clap-htsat-unfused": {
        "kind": "clap",
        "backend": "clap",
        "hf_repo": "laion/clap-htsat-unfused",
        "revision": "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a",
        "dim": 512,
        "sr": 48000,
        "clip_s": 10.0,
        "text": True,
    },
    "muq-mulan": {
        "kind": "mulan",
        "backend": "mulan",
        "hf_repo": "OpenMuQ/MuQ-MuLan-large",
        "revision": "2e01c796b71dca71b45251384c04cd7b237c9020",
        "dim": 512,
        "sr": 24000,
        "text": True,
    },
    "mert-v1-95m": {
        "kind": "mert",
        "backend": "mert",
        "hf_repo": "m-a-p/MERT-v1-95M",
        "revision": "main",
        "dim": 768,
        "sr": 24000,
        "text": False,
    },
    "mert-v1-330m": {
        "kind": "mert",
        "backend": "mert",
        "hf_repo": "m-a-p/MERT-v1-330M",
        "revision": "main",
        "dim": 1024,
        "sr": 24000,
        "text": False,
    },
}

_INSTALL_HINT = (
    "uv tool install 'smpl-embed[torch]' && smpl embed models install <model-id>"
)
_MUQ_HINT = "uv tool install 'smpl-embed[muq]'"
_FAISS_HINT = "uv tool install 'smpl-embed[faiss]'"


def model_catalog() -> dict:
    return _MODEL_CATALOG


def resolve_model(model_id: Optional[str]) -> tuple[str, dict]:
    """Resolve a model id to its catalog spec (raises UnsupportedError for unknown ids)."""
    model_id = model_id or default_model()
    spec = _MODEL_CATALOG.get(model_id)
    if spec is None:
        raise UnsupportedError(
            f"unknown embedding model {model_id!r}; known: {sorted(_MODEL_CATALOG)}",
            install_hint=_INSTALL_HINT,
        )
    return model_id, spec


def vector_role(spec: dict, *, text: bool = False) -> str:
    """The role a vector frame carries: ``ml.<backend>.embedding`` (audio) or
    ``ml.<backend>.text-embedding`` (a text prompt in the same joint space)."""
    return f"ml.{spec['backend']}.{'text-embedding' if text else 'embedding'}"


def _weights_identity(model_id: str, spec: dict) -> str:
    """Weights identity for `op_version` (spec → *Memoization*, ML weights rule).

    Prefer the blake3 of the on-disk weights if the model is materialized; otherwise fall
    back to the pinned registry `repo@revision`. Either way a model upgrade re-keys the memo
    cache instead of silently serving stale vectors from the old weights.
    """
    info = _registry().get(model_id)
    if info and info.get("weights_blake3"):
        return f"blake3:{info['weights_blake3']}"
    return f"{spec['hf_repo']}@{spec.get('revision', 'main')}"


def op_version(model_id: str, spec: dict) -> str:
    """`op_version` = op tag + model id + weights identity (NOT just the friendly name).

    ``@2``: CLAP moved to ``ClapModel.get_*_features`` with deterministic 10 s windows, and
    every vector is L2-normalized.
    """
    return f"embed@2:{spec['kind']}:{model_id}:{_weights_identity(model_id, spec)}"


# ---------------------------------------------------------------------------
# Embedding (heavy path — lazy-imported, guarded → UnsupportedError)
# ---------------------------------------------------------------------------

# Loaded models, keyed by model id: a batch over a pack pays the load once per process.
_LOADED: dict = {}


def _l2(vec):
    import numpy as np

    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n == 0.0:
        raise ValueError("degenerate embedding (zero or non-finite norm)")
    return (v / n).astype(np.float32)


def _require_torch(model_id: str, *, extra: str = "torch", hint: str = _INSTALL_HINT):
    try:  # pragma: no cover - heavy dep, not installed in the light default venv
        import torch  # noqa: F401
    except ImportError as exc:
        raise UnsupportedError(
            f"embedding model {model_id!r} needs torch (the [{extra}] extra; not installed): {exc}",
            install_hint=hint,
        ) from exc


def _resample(pcm_f32, sr: int, target_sr: int):
    import numpy as np

    audio = np.asarray(pcm_f32, dtype=np.float32).reshape(-1)
    if sr == target_sr:
        return audio
    import torch  # pragma: no cover - heavy path
    import torchaudio  # pragma: no cover - heavy path

    return torchaudio.functional.resample(torch.from_numpy(audio), sr, target_sr).numpy()


def _windows(audio, sr: int, clip_s: float):
    """Deterministic ≤``clip_s`` windows covering the file.

    CLAP's feature extractor crops audio longer than its 10 s input at RANDOM (``rand_trunc``
    / ``fusion``), which would make the same file embed differently on every run. Feeding it
    consecutive non-overlapping windows (each ≤ 10 s, so no crop) and averaging the
    normalized window embeddings keeps the op deterministic and covers the whole file.
    """
    n = max(1, round(clip_s * sr))
    if len(audio) <= n:
        return [audio]
    out = [audio[i:i + n] for i in range(0, len(audio), n)]
    # A tail shorter than a quarter window is mostly padding to the encoder: fold it away.
    if len(out) > 1 and len(out[-1]) < n // 4:
        out.pop()
    return out


# Original LAION-CLAP → transformers key renames (transformers' own
# convert_clap_original_pytorch_to_hf.py, applied in this order).
_LAION_RENAMES = (
    ("text_branch", "text_model"),
    ("audio_branch", "audio_model.audio_encoder"),
    ("attn", "attention.self"),
    ("self.proj", "output.dense"),
    ("attention.self_mask", "attn_mask"),
    ("mlp.fc1", "intermediate.dense"),
    ("mlp.fc2", "output.dense"),
    ("norm1", "layernorm_before"),
    ("norm2", "layernorm_after"),
    ("bn0", "batch_norm"),
)


def _rename_laion_state(sd: dict) -> dict:
    """Rename an original LAION-CLAP state dict to `ClapModel` keys.

    Strips the DDP ``module.`` prefix first — upstream's converter did not, and loading with
    ``strict=False`` then silently kept the random init (the broken hub port). Drops what the
    transformers model does not have: the in-model STFT/mel front end (ClapFeatureExtractor
    does that) and the training-only ``*_transform`` heads.
    """
    import re

    out = {}
    for key, value in sd.items():
        key = key.removeprefix("module.")
        if key.startswith(("text_transform.", "audio_transform.")) or (
            "spectrogram_extractor" in key or "logmel_extractor" in key
        ):
            continue
        for a, b in _LAION_RENAMES:
            key = key.replace(a, b)
        seq = re.match(r".*sequential.(\d+).*", key)
        proj = re.match(r".*_projection.(\d+).*", key)
        if seq:
            n = int(seq.group(1))
            key = key.replace(f"sequential.{n}.", f"layers.{n // 3}.linear.")
        elif proj:
            n = int(proj.group(1))
            key = key.replace(f"_projection.{n}.", f"_projection.linear{1 if n == 0 else 2}.")
        if "qkv" in key:  # fused qkv → query / key / value
            d = value.size(0) // 3
            for i, name in enumerate(("query", "key", "value")):
                out[key.replace("qkv", name)] = value[i * d:(i + 1) * d]
        else:
            out[key] = value
    return out


def _convert_laion_clap(model_id: str, spec: dict) -> Path:  # pragma: no cover - heavy path
    """Convert the original LAION-CLAP checkpoint into a `ClapModel` dir, once.

    Lives at ``SMPL_EMBED_HOME/<model_id>/`` (the registry's weights home), so the second
    process just loads it. Refuses to save a model with any parameter left at its init
    value: every ``ClapModel`` parameter must come from the checkpoint.
    """
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import ClapConfig, ClapModel, ClapProcessor

    dest = embed_home() / model_id
    if (dest / "config.json").is_file():
        return dest
    sys.stderr.write(f"smpl embed: converting {spec['hf_repo']}/{spec['laion_file']} → {dest}\n")
    ckpt = hf_hub_download(spec["hf_repo"], spec["laion_file"], revision=spec["revision"])
    # The checkpoint pickles a few numpy scalars (epoch bookkeeping); allow exactly those,
    # never a blanket weights_only=False.
    try:
        from numpy._core.multiarray import scalar
    except ImportError:  # numpy < 2
        from numpy.core.multiarray import scalar

    allowed = [(scalar, "numpy.core.multiarray.scalar"), np.dtype,
               type(np.dtype(np.float64)), type(np.dtype(np.int64)), type(np.dtype(np.float32))]
    with torch.serialization.safe_globals(allowed):
        raw = torch.load(ckpt, map_location="cpu", weights_only=True)
    state = _rename_laion_state(raw.get("state_dict", raw))

    base, base_rev = spec["base_repo"], spec.get("base_revision")
    model = ClapModel(ClapConfig.from_pretrained(base, revision=base_rev))
    missing, unexpected = model.load_state_dict(state, strict=False)
    params = {name for name, _ in model.named_parameters()}
    missing_params = [k for k in missing if k in params]
    # HTSAT's AudioSet classifier (tscam_conv / head) is unused by the embedding path, and
    # the Swin shifted-window attn_mask buffers are recomputed by transformers.
    unexpected = [k for k in unexpected
                  if ".tscam_conv." not in k and ".head." not in k and not k.endswith(".attn_mask")]
    if missing_params or unexpected:
        raise UnsupportedError(
            f"LAION-CLAP conversion for {model_id!r} does not line up: "
            f"{len(missing_params)} parameters missing (e.g. {missing_params[:3]}), "
            f"{len(unexpected)} unexpected keys (e.g. {list(unexpected)[:3]})",
            install_hint=_INSTALL_HINT,
        )
    dest.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(dest))
    ClapProcessor.from_pretrained(base, revision=base_rev).save_pretrained(str(dest))
    return dest


def _load_clap(model_id: str, spec: dict):  # pragma: no cover - heavy path
    if model_id in _LOADED:
        return _LOADED[model_id]
    _require_torch(model_id)
    try:
        from transformers import ClapModel, ClapProcessor
    except ImportError as exc:
        raise UnsupportedError(
            f"CLAP needs transformers (the [torch] extra; not installed): {exc}",
            install_hint=_INSTALL_HINT,
        ) from exc
    try:
        if spec.get("laion_file"):
            src, rev = str(_convert_laion_clap(model_id, spec)), None
        else:
            src, rev = spec["hf_repo"], spec.get("revision", "main")
        model = ClapModel.from_pretrained(src, revision=rev)
        processor = ClapProcessor.from_pretrained(src, revision=rev)
    except UnsupportedError:
        raise
    except Exception as exc:  # download/weights problem → unsupported, not a crash
        raise UnsupportedError(
            f"could not load weights for {model_id!r} ({spec['hf_repo']}): {exc}",
            install_hint=_INSTALL_HINT,
        ) from exc
    model.eval()
    _LOADED[model_id] = (model, processor)
    return _LOADED[model_id]


def _pooled(out):
    """transformers ≥5 returns ``BaseModelOutputWithPooling`` from ``get_*_features``
    (the projected embedding is ``.pooler_output``); ≤4 returned the tensor itself."""
    return out.pooler_output if hasattr(out, "pooler_output") else out


def _clap_audio(audio, model_id: str, spec: dict):  # pragma: no cover - heavy path
    import numpy as np
    import torch

    model, processor = _load_clap(model_id, spec)
    sr = spec["sr"]
    vecs = []
    with torch.no_grad():
        for win in _windows(audio, sr, spec.get("clip_s", 10.0)):
            inputs = processor(audio=[win], sampling_rate=sr, return_tensors="pt")
            vecs.append(_l2(_pooled(model.get_audio_features(**inputs))[0].cpu().numpy()))
    return _l2(np.mean(np.stack(vecs), axis=0))


def _clap_text(texts: list[str], model_id: str, spec: dict):  # pragma: no cover - heavy path
    import torch

    model, processor = _load_clap(model_id, spec)
    with torch.no_grad():
        inputs = processor(text=list(texts), return_tensors="pt", padding=True)
        out = _pooled(model.get_text_features(**inputs)).cpu().numpy()
    return [_l2(row) for row in out]


def _load_mulan(model_id: str, spec: dict):  # pragma: no cover - heavy path
    if model_id in _LOADED:
        return _LOADED[model_id]
    _require_torch(model_id, extra="muq", hint=_MUQ_HINT)
    try:
        from muq import MuQMuLan
    except ImportError as exc:
        raise UnsupportedError(
            f"MuQ-MuLan needs the `muq` package (the [muq] extra; not installed): {exc}",
            install_hint=_MUQ_HINT,
        ) from exc
    try:
        model = MuQMuLan.from_pretrained(spec["hf_repo"], revision=spec.get("revision", "main"))
    except Exception as exc:
        raise UnsupportedError(
            f"could not load weights for {model_id!r} ({spec['hf_repo']}): {exc}",
            install_hint=_MUQ_HINT,
        ) from exc
    model = model.to("cpu").eval()
    _LOADED[model_id] = model
    return model


def _mulan_audio(audio, model_id: str, spec: dict):  # pragma: no cover - heavy path
    import torch

    model = _load_mulan(model_id, spec)
    with torch.no_grad():
        # MuQMuLan splits > 10 s audio into consecutive clips itself and averages the latents.
        out = model(wavs=torch.from_numpy(audio).unsqueeze(0))
    return _l2(out[0].cpu().numpy())


def _mulan_text(texts: list[str], model_id: str, spec: dict):  # pragma: no cover - heavy path
    import torch

    model = _load_mulan(model_id, spec)
    with torch.no_grad():
        out = model(texts=list(texts)).cpu().numpy()
    return [_l2(row) for row in out]


def _mert_audio(audio, model_id: str, spec: dict):  # pragma: no cover - heavy path
    import torch

    if model_id not in _LOADED:
        _require_torch(model_id)
        try:
            from transformers import AutoModel, Wav2Vec2FeatureExtractor

            rev = spec.get("revision", "main")
            processor = Wav2Vec2FeatureExtractor.from_pretrained(
                spec["hf_repo"], revision=rev, trust_remote_code=True)
            model = AutoModel.from_pretrained(spec["hf_repo"], revision=rev, trust_remote_code=True)
        except Exception as exc:
            raise UnsupportedError(
                f"could not load weights for {model_id!r} ({spec['hf_repo']}): {exc}",
                install_hint=_INSTALL_HINT,
            ) from exc
        _LOADED[model_id] = (model.eval(), processor)
    model, processor = _LOADED[model_id]
    with torch.no_grad():
        inputs = processor(audio, sampling_rate=spec["sr"], return_tensors="pt")
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.last_hidden_state  # mean-pool over time → one deterministic vector
        return _l2(hidden.mean(dim=1).squeeze(0).to(torch.float32).cpu().numpy())


def embed(pcm_f32, sr: int, *, model_id: Optional[str] = None):
    """Embed mono float32 PCM → a 1-D, L2-normalized float32 numpy vector.

    Lazy-imports torch + the model package INSIDE this call (never at module top). On a
    missing heavy dep / model weights it raises :class:`UnsupportedError` (the CLI turns that
    into a clean `unsupported` error frame + stderr install hint) — the light default path.

    Runs on CPU in eval mode with deterministic windowing (see :func:`_windows`), so the
    vectors are *cacheable*; a GPU/MPS path with nondeterministic kernels would have to set
    ``cacheable: false`` in its params (spec → *Memoization*, non-deterministic ops).
    """
    model_id, spec = resolve_model(model_id)
    _require_torch(model_id, extra="muq" if spec["kind"] == "mulan" else "torch",
                   hint=_MUQ_HINT if spec["kind"] == "mulan" else _INSTALL_HINT)
    audio = _resample(pcm_f32, sr, spec["sr"])  # pragma: no cover - heavy path below
    if spec["kind"] == "clap":
        return _clap_audio(audio, model_id, spec)
    if spec["kind"] == "mulan":
        return _mulan_audio(audio, model_id, spec)
    return _mert_audio(audio, model_id, spec)


def embed_text(texts: list[str], *, model_id: str | None = None) -> list:
    """Embed text prompts into the model's joint audio-text space (L2-normalized vectors).

    Only models with a text tower (``clap``, ``muq-mulan``) accept text; others raise
    :class:`UnsupportedError`. Same lazy-import / degrade contract as :func:`embed`.
    """
    model_id, spec = resolve_model(model_id)
    if not spec.get("text"):
        raise UnsupportedError(
            f"model {model_id!r} has no text tower; use clap or muq-mulan",
            install_hint=_INSTALL_HINT,
        )
    _require_torch(model_id, extra="muq" if spec["kind"] == "mulan" else "torch",
                   hint=_MUQ_HINT if spec["kind"] == "mulan" else _INSTALL_HINT)
    if spec["kind"] == "clap":  # pragma: no cover - heavy path
        return _clap_text(texts, model_id, spec)
    return _mulan_text(texts, model_id, spec)  # pragma: no cover - heavy path

# ---------------------------------------------------------------------------
# Model registry (ollama-style: list / install / update / rm) under SMPL_EMBED_HOME
# ---------------------------------------------------------------------------


def _registry_file() -> Path:
    return embed_home() / "models.json"


def _registry() -> dict:
    f = _registry_file()
    return json.loads(f.read_text()) if f.exists() else {}


def list_models() -> list[dict]:
    installed = _registry()
    rows = []
    for mid, spec in _MODEL_CATALOG.items():
        info = installed.get(mid)
        rows.append(
            {
                "id": mid,
                "kind": spec["kind"],
                "hf_repo": spec["hf_repo"],
                "revision": spec.get("revision", "main"),
                "dim": spec["dim"],
                "installed": bool(info),
                "op_version": op_version(mid, spec),
                **({"path": info["path"]} if info and info.get("path") else {}),
            }
        )
    return rows


def install_model(model_id: str) -> dict:
    """Register a model as installed. Real weight download lands with the heavy extra; the
    registry + path management (and weights-identity capture) is the v1 surface."""
    model_id, spec = resolve_model(model_id)
    embed_home().mkdir(parents=True, exist_ok=True)
    f = _registry_file()
    reg = _registry()
    path = embed_home() / model_id
    entry: dict = {
        "kind": spec["kind"],
        "hf_repo": spec["hf_repo"],
        "revision": spec.get("revision", "main"),
        "path": str(path),
    }
    # If weights are materialized on disk, capture their blake3 so op_version pins identity.
    wfile = path / "model.safetensors"
    if wfile.exists():
        entry["weights_blake3"] = hashlib.blake2b(wfile.read_bytes(), digest_size=16).hexdigest()
    reg[model_id] = entry
    f.write_text(json.dumps(reg, indent=2))
    return entry


def remove_model(model_id: str) -> bool:
    f = _registry_file()
    if not f.exists():
        return False
    reg = json.loads(f.read_text())
    if model_id not in reg:
        return False
    del reg[model_id]
    f.write_text(json.dumps(reg, indent=2))
    return True


# ---------------------------------------------------------------------------
# FAISS index manager (heavy path — lazy-imported faiss, guarded → UnsupportedError)
# ---------------------------------------------------------------------------


def index_home() -> Path:
    return Path(os.environ.get("SMPL_INDEX_HOME", "~/.smpl/index")).expanduser()


def _index_paths(name: str) -> tuple[Path, Path]:
    safe = "".join(c for c in name if c.isalnum() or c in "-_.") or "default"
    base = index_home() / safe
    return base.with_suffix(".faiss"), base.with_suffix(".ids.json")


def _require_faiss():
    try:  # pragma: no cover - heavy dep, not installed in the light default venv
        import faiss  # noqa: F401
    except ImportError as exc:
        raise UnsupportedError(
            f"`smpl index` needs faiss (not installed): {exc}", install_hint=_FAISS_HINT
        ) from exc
    import faiss  # noqa: F811

    return faiss


def index_build(name: str, vectors, ids: list[str]) -> dict:
    """Build / overwrite a FAISS index `name` over `vectors` (N×dim float32), keyed by ids.

    Lazy-imports faiss; a missing faiss raises :class:`UnsupportedError` (the CLI emits an
    `unsupported` frame + stderr hint). Uses inner-product over L2-normalized vectors, so the
    score is cosine similarity.
    """
    import numpy as np

    faiss = _require_faiss()  # pragma: no cover - requires the faiss extra
    mat = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
    if mat.ndim != 2 or mat.shape[0] == 0:
        raise UnsupportedError("no vectors to index", install_hint=_FAISS_HINT)
    faiss.normalize_L2(mat)
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)
    faiss_path, ids_path = _index_paths(name)
    faiss_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(faiss_path))
    ids_path.write_text(json.dumps({"ids": ids, "dim": int(mat.shape[1])}))
    return {"name": name, "count": int(mat.shape[0]), "dim": int(mat.shape[1]), "path": str(faiss_path)}


def index_query(name: str, vector, k: int = 10) -> list[dict]:
    """Query FAISS index `name` for the `k` nearest ids to `vector`. → [{id, score}]."""
    import numpy as np

    faiss = _require_faiss()  # pragma: no cover - requires the faiss extra
    faiss_path, ids_path = _index_paths(name)
    if not faiss_path.exists():
        raise UnsupportedError(
            f"no index named {name!r} at {faiss_path}", install_hint=_FAISS_HINT
        )
    index = faiss.read_index(str(faiss_path))
    ids = json.loads(ids_path.read_text())["ids"]
    q = np.ascontiguousarray(np.asarray([vector], dtype=np.float32))
    faiss.normalize_L2(q)
    scores, idxs = index.search(q, min(k, len(ids)))
    out = []
    for score, i in zip(scores[0].tolist(), idxs[0].tolist()):
        if 0 <= i < len(ids):
            out.append({"id": ids[i], "score": float(score)})
    return out
