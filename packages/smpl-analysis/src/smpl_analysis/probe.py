"""Score samples on a text axis in a joint audio-text space (CLAP / MuQ-MuLan).

Raw audio-text cosines lie: contrastive embeddings sit in a narrow cone, so every cosine is
similar and a single-prompt score mostly measures the prompt (anisotropy + modality gap).
The calibrated recipe (vault-3hvg; research/2026-06-10-musical-axes/clap-landscape.md) is a
**contrast**:

- each pole is a prompt set, pooled: L2-normalize each prompt vector, mean, L2-normalize;
- the axis is ``normalize(P_pos − P_neg)``;
- an item's score is the projection of its unit audio vector on the axis
  (``probe.score_cos``) — only differences between items mean anything;
- ``probe.z`` standardizes the scores within the scored set (mean 0, sample σ 1; needs ≥ 3
  items). A frozen library calibration (μ, σ per axis) is vault-3hvg's snapshot file; until it
  exists the z is set-relative, so it ranks within a pack and does not compare across packs.

With no negative prompt the axis contrasts against the generic ``"a sound"`` — a weak anchor
that still cancels the shared text direction; pass ``--neg`` for a real bipolar axis.
"""

from __future__ import annotations

import numpy as np

OP = "probe"
OP_VERSION = "probe@1"

DEFAULT_NEG = ("a sound",)


def axis(pos_vecs: list, neg_vecs: list) -> np.ndarray:
    """Unit axis from pooled positive and negative poles."""
    from .vectors import l2, pole

    return l2(pole(pos_vecs) - pole(neg_vecs))


def scores(X: np.ndarray, ax: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Projection of each unit row on the axis, plus the in-set z (None below 3 items)."""
    s = np.asarray(X, dtype=np.float64) @ ax
    if len(s) < 3:
        return s, None
    sd = float(np.std(s, ddof=1))
    return s, (s - s.mean()) / sd if sd > 0 else np.zeros_like(s)


def probe_frames(items, *, pos: list[str], neg: list[str], text_vecs: dict,
                 model: str) -> list[dict]:
    """Per-item `feature` frames (role ``probe``) + a ranked summary (role ``probe-pack``)."""
    from smplstream import frames as F

    from .vectors import matrix

    ax = axis([text_vecs[p] for p in pos], [text_vecs[n] for n in neg])
    s, z = scores(matrix(items), ax)
    common = {"op": OP, "op_version": OP_VERSION,
              "params": {"model": model, "pos": list(pos), "neg": list(neg)}}
    out = []
    for i, it in enumerate(items):
        out.append(F.feature_frame({
            "probe.score_cos": round(float(s[i]), 5),
            "probe.z": None if z is None else round(float(z[i]), 3),
        }, role="probe", of=it.id, lineage=[it.vector_id], **common))
    order = np.argsort(-s)
    out.append(F.feature_frame({
        "probe.n": len(items),
        "probe.ranked": [{"label": items[i].label, "score_cos": round(float(s[i]), 5),
                          "z": None if z is None else round(float(z[i]), 3)} for i in order],
    }, role="probe-pack", lineage=[it.vector_id for it in items], **common))
    return out
