"""Separator backend + a thin model registry (plan.md → External engines, §5 ML).

The separator (Demucs, run through ``python-audio-separator``) is a heavy,
torch-backed dep that lives behind the ``[torch]`` extra and is imported lazily
inside :func:`SeparatorBackend.separate` — NEVER at module top. The model store
path + default model resolve from env vars (``SMPL_STEMS_HOME``,
``SMPL_STEMS_MODEL``); weights are managed under ``SMPL_STEMS_HOME`` (ollama-style
list/install/update/rm), never as a pip dependency.

The crucial wire-protocol concern here is **op_version**: for an ML op it MUST
incorporate the *weights identity*, not just a friendly model name — otherwise a
Demucs upgrade silently serves stale cached results from the old weights (spec →
*Memoization*). We hash the resolved weights file when present, falling back to
the registry id+version, and fold that into ``op_version``.

## Model routing (guidance, not runtime enforcement)

Three models are known (see :data:`KNOWN_MODELS`), and they are NOT meant to run
in the same place:

* ``htdemucs`` (4-stem) and ``htdemucs_6s`` (6-stem, the default) are **local**
  models — Demucs is MPS-capable, so they run fine on the laptop/desktop tier.
* ``bs-roformer`` (2-stem vocals/instrumental, ~3 dB SDR over Demucs on vocals)
  is heavier and is intended for the **mac-mini batch tier**: queue batch jobs
  there rather than running it interactively on a laptop.

Nothing here schedules anything — routing is an operator decision; this module
only documents the intent and pins the exact checkpoint each id resolves to.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional


def stems_home() -> Path:
    return Path(os.environ.get("SMPL_STEMS_HOME", "~/.smpl/stems")).expanduser()


def default_model() -> str:
    # htdemucs_6s is the 6-stem set (adds guitar + piano) the spec's role table names.
    return os.environ.get("SMPL_STEMS_MODEL", "htdemucs_6s")


# Canonical stem → smplstream role (spec → *Role naming conventions*). audio-separator emits
# Demucs stem names; we map them onto the namespaced `stem:<name>` roles. The 6-stem set
# (htdemucs_6s) adds guitar + piano; 4-stem models simply produce a subset. `instrumental` is
# the RoFormer half — a 2-stem run emits Vocals + Instrumental, and without a row here the
# role filter in `separate()` would silently DROP the instrumental stem.
STEM_ROLES = {
    "drums": "stem:drums",
    "bass": "stem:bass",
    "vocals": "stem:vocals",
    "other": "stem:other",
    "guitar": "stem:guitar",
    "piano": "stem:piano",
    "instrumental": "stem:instrumental",
}


# Known models → the metadata that can't be guessed from the id string.
#
# WHY an explicit table rather than suffix-sniffing: `load_model` takes a *file name*, and the
# extension is model-family-specific — the Demucs family loads by its `.yaml` bundle descriptor,
# while BS-RoFormer loads the UVR `.ckpt` directly. Pinning the exact checkpoint filename also
# keeps `op_version`'s weights identity stable (a different RoFormer epoch is a different file,
# hence a different op_version, hence no stale cache hits — spec → *Memoization*).
KNOWN_MODELS: dict[str, dict] = {
    "htdemucs": {
        "separator_filename": "htdemucs.yaml",
        "stems": "4",
        "description": "Hybrid Transformer Demucs, 4-stem (drums/bass/vocals/other). Local tier.",
    },
    "htdemucs_6s": {
        "separator_filename": "htdemucs_6s.yaml",
        "stems": "6",
        "description": "Demucs 6-stem (adds guitar + piano). Default; local tier.",
    },
    "bs-roformer": {
        "separator_filename": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "stems": "2",
        "description": (
            "UVR BS-RoFormer ep 317 (SDR 12.9755), 2-stem vocals/instrumental. "
            "~3 dB SDR over Demucs on vocals; heavy — route to the mac-mini batch tier."
        ),
    },
}


def separator_filename(model: str) -> str:
    """The file name handed to ``Separator.load_model`` for ``model``.

    Known ids resolve through :data:`KNOWN_MODELS` (yaml for the Demucs family, the pinned
    ckpt for BS-RoFormer); an unknown/custom id keeps the historical ``<id>.yaml`` guess so
    custom Demucs-family models still work unchanged.
    """
    meta = KNOWN_MODELS.get(model)
    if meta:
        return meta["separator_filename"]
    return f"{model}.yaml"


def stems_for(model: str) -> str:
    """Stem count for ``model`` (string, as the registry rows carry it)."""
    meta = KNOWN_MODELS.get(model)
    if meta:
        return meta["stems"]
    return "6" if model.endswith("6s") else "4"


class UnsupportedBackend(Exception):
    """Raised when the heavy separator (or its model) is unavailable.

    Carries the exact install command so the CLI can surface it on stderr while
    emitting the ``unsupported`` error frame to stdout (two-tier degrade path).
    """

    def __init__(self, message: str, *, install_hint: str):
        super().__init__(message)
        self.install_hint = install_hint


# The single source of truth for the stderr install line + the error-frame message.
INSTALL_HINT = (
    "uv tool install 'smpl-stems[torch]' "
    "&& smpl stems models install htdemucs_6s"
)


def _weights_identity(model: str) -> str:
    """Best-effort weights identity for `op_version` (spec → *Memoization*, NORMATIVE).

    Prefer a blake3/blake2 hash of the resolved weights file under SMPL_STEMS_HOME so an
    in-place weights swap invalidates the cache. Fall back to the registry id+version when
    the file isn't materialized yet (light install). Either way the friendly name alone is
    NEVER the identity.
    """
    reg = _load_registry()
    meta = reg.get(model) or {}
    weights_path = meta.get("weights")
    if weights_path:
        p = Path(weights_path).expanduser()
        if p.exists() and p.is_file():
            h = hashlib.blake2b(digest_size=16)
            with p.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            return f"weights-blake2b:{h.hexdigest()}"
    version = meta.get("version", "unpinned")
    return f"registry:{model}@{version}"


def op_version_for(model: str, sep_pkg_version: str = "unknown") -> str:
    """The op_version string bound into the memo key.

    Form: ``audio-separator@<pkg>+<model>:<weights-identity>`` — folds the separator package
    version, the model name, AND the weights identity (file hash or pinned registry version).
    A Demucs/audio-separator upgrade OR a weights swap changes this string, so stale results
    from old weights can't be served from cache (spec → *Memoization*, NORMATIVE).
    """
    return f"audio-separator@{sep_pkg_version}+{model}:{_weights_identity(model)}"


class SeparatorBackend:
    """Demucs source separation via ``python-audio-separator`` (heavy, torch-backed).

    Lazy-imports the separator inside :meth:`separate`; a missing dep/model raises
    :class:`UnsupportedBackend` carrying the install hint (the CLI turns that into the
    ``unsupported`` error frame + stderr line). NEVER imports torch at module top.
    """

    def __init__(self, model: Optional[str] = None):
        self.model = model or default_model()

    def _separator_version(self) -> str:
        try:
            from importlib.metadata import version

            return version("audio-separator")
        except Exception:  # pragma: no cover - light install / metadata absent
            return "unknown"

    @property
    def op_version(self) -> str:
        return op_version_for(self.model, self._separator_version())

    def separate(self, input_path: str):
        """Separate ``input_path`` into stems. Returns ``[(stem_name, wav_path), …]``.

        Lazy-imports the heavy separator HERE (guarded). On ImportError (the light default
        install) or any model/runtime failure, raises :class:`UnsupportedBackend` so the CLI
        degrades to the ``unsupported`` error frame + stderr install hint — it MUST NOT raise
        a bare ImportError nor hang.
        """
        try:  # pragma: no cover - exercised only when the heavy extra is installed
            from audio_separator.separator import Separator  # type: ignore  # noqa: PLC0415
        except ImportError as exc:
            raise UnsupportedBackend(
                f"source separator not installed ({exc})",
                install_hint=INSTALL_HINT,
            ) from exc

        try:  # pragma: no cover - heavy path, not run in the light venv
            out_dir = str(stems_home() / "out")
            stems_home().mkdir(parents=True, exist_ok=True)
            sep = Separator(
                output_dir=out_dir,
                output_format="WAV",
                model_file_dir=str(stems_home() / "models"),
            )
            # Family-specific file name (Demucs → .yaml, BS-RoFormer → the pinned .ckpt);
            # unknown ids fall back to `<id>.yaml`. See `separator_filename`.
            sep.load_model(model_filename=separator_filename(self.model))
            produced = sep.separate(input_path)
        except UnsupportedBackend:
            raise
        except Exception as exc:  # model download blocked / CUDA-OOM / bad input
            raise UnsupportedBackend(
                f"separation failed for model {self.model!r}: {exc}",
                install_hint=INSTALL_HINT,
            ) from exc

        results = []
        for path in produced:
            stem = _infer_stem_name(str(path))
            if stem in STEM_ROLES:
                results.append((stem, str(path)))
        if not results:
            raise UnsupportedBackend(
                f"separator produced no recognized stems for model {self.model!r}",
                install_hint=INSTALL_HINT,
            )
        return results


def _infer_stem_name(path: str) -> str:
    """Map an audio-separator output filename to a canonical stem name.

    audio-separator names outputs ``<track>_(<Stem>)`` and often appends the model name
    (``<track>_(Vocals)_model_bs_roformer_…``), so the parenthesised marker is matched
    anywhere in the name, not only at the end.
    """
    low = Path(path).stem.lower()
    for stem in STEM_ROLES:
        if f"_({stem})" in low or low.endswith(stem) or f"_{stem}" in low:
            return stem
    return low


def get_backend(model: Optional[str] = None) -> SeparatorBackend:
    """Resolve the separator backend (does NOT import torch — that stays inside separate())."""
    return SeparatorBackend(model)


# ---- minimal model registry (ollama-style: list / install / update / rm) ----
# Mirrors smpl-gen/backends.py: the registry + path management is the v1 surface; real
# weight download lands with the heavy extra. Weights live under SMPL_STEMS_HOME, never pip.

def _registry_file() -> Path:
    return stems_home() / "models.json"


def _load_registry() -> dict:
    f = _registry_file()
    return json.loads(f.read_text()) if f.exists() else {}


def list_models() -> list[dict]:
    installed = _load_registry()
    # Every KNOWN_MODELS id (and the default, however it is overridden) is always listed —
    # downloadable on first heavy run even before it is registered — so `models list` shows
    # the user which models `stems` can reach for, including bs-roformer.
    rows = []
    default = default_model()
    known = set(installed) | set(KNOWN_MODELS) | {default}
    for mid in sorted(known):
        meta = installed.get(mid, {})
        rows.append(
            {
                "id": mid,
                "installed": mid in installed,
                "default": mid == default,
                "version": meta.get("version", "unpinned"),
                "weights": meta.get("weights"),
                "stems": meta.get("stems") or stems_for(mid),
            }
        )
    return rows


def install_model(model_id: str, *, version: str = "unpinned") -> dict:
    """Register a model as installed. Real weight download lands with the heavy extra;
    the registry + path management is the v1 surface (weights tracked for op_version)."""
    stems_home().mkdir(parents=True, exist_ok=True)
    f = _registry_file()
    reg = _load_registry()
    # Weights land under SMPL_STEMS_HOME/models under the separator's own file name, so the
    # op_version hash targets the file audio-separator actually downloads (yaml for the Demucs
    # family, the pinned ckpt for BS-RoFormer). Unknown ids keep the historical `<id>.ckpt`.
    weights_name = (
        KNOWN_MODELS[model_id]["separator_filename"]
        if model_id in KNOWN_MODELS
        else f"{model_id}.ckpt"
    )
    reg[model_id] = {
        "version": version,
        "weights": str(stems_home() / "models" / weights_name),
        "stems": stems_for(model_id),
    }
    f.write_text(json.dumps(reg, indent=2))
    return reg[model_id]


def remove_model(model_id: str) -> bool:
    f = _registry_file()
    reg = _load_registry()
    if model_id not in reg:
        return False
    del reg[model_id]
    f.write_text(json.dumps(reg, indent=2))
    return True
