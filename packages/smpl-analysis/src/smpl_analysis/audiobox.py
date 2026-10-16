"""Optional Audiobox-Aesthetics scorer (CE/CU/PC/PQ) for generated one-shots.

Lives behind the `audiobox` extra (`pip install 'smpl-analysis[audiobox]'`) so the base
install stays light: torch and the model are imported lazily, only when scoring is asked for.
We decode the audio ourselves (soundfile -> tensor) because the package's own path loader
needs torchcodec. Scores are on a 1-10 scale: CE content enjoyment, CU content usefulness,
PC production complexity, PQ production quality.
"""

from __future__ import annotations

AXES = ("CE", "CU", "PC", "PQ")
FEATURE_KEYS = {a: f"qc.audiobox.{a.lower()}" for a in AXES}
INSTALL_HINT = (
    "audiobox scoring needs the optional extra: "
    "pip install 'smpl-analysis[audiobox]' (audiobox_aesthetics, requests, soundfile)"
)

_predictor = None


class AudioboxUnavailable(RuntimeError):
    """The audiobox extra (or its model) is not installed / loadable."""


def _get_predictor():
    global _predictor
    if _predictor is None:
        try:
            from audiobox_aesthetics.infer import initialize_predictor
        except ImportError as exc:
            raise AudioboxUnavailable(f"{INSTALL_HINT} ({exc})") from exc
        _predictor = initialize_predictor()
    return _predictor


def check_available() -> None:
    """Raise AudioboxUnavailable early (before any work) if the extra is missing."""
    try:
        import audiobox_aesthetics.infer  # noqa: F401
        import requests  # noqa: F401
        import soundfile  # noqa: F401
    except ImportError as exc:
        raise AudioboxUnavailable(f"{INSTALL_HINT} ({exc})") from exc


def score_path(path: str) -> dict[str, float]:
    """Score one audio file; returns {"CE","CU","PC","PQ"} floats."""
    import soundfile as sf
    import torch

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T.copy())  # (channels, samples)
    res = _get_predictor().forward([{"path": wav, "sample_rate": int(sr)}])[0]
    return {a: round(float(res[a]), 3) for a in AXES}
