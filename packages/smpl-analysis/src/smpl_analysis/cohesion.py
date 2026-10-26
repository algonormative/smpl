"""Pack cohesion over embedding vectors: how tightly a set of samples hangs together.

Inputs are L2-normalized `vector` frames from one model (see :mod:`smpl_analysis.vectors`).
Distances are cosine distances, ``1 − x·y``.

Per item:

- ``cohesion.centroid_dist_cos`` — distance to the pack centroid (the L2-normalized mean).
- ``cohesion.outlier_z`` — robust (modified) z of that distance, Iglewicz & Hoaglin (1993):
  ``0.6745 · (d − median) / MAD``. Signed; large positive = far from the pack.
  A MAD of 0 (more than half the pack at one distance) falls back to the mean absolute
  deviation scaled by 0.7979 (≈ √(2/π), so it estimates σ the way 1.4826·MAD does).
- ``cohesion.outlier`` — flag: ``outlier_z > z_threshold`` (default 3.5, Iglewicz & Hoaglin's
  recommendation). Only a flag: what to do with an outlier is the skill's call.

Per pack:

- ``cohesion.pairwise_dist_cos_mean`` — mean cosine distance over all unordered pairs.
- ``cohesion.kad`` — Kernel Audio Distance against a reference set (Chung et al. 2025,
  arXiv 2502.15602): the unbiased MMD² estimator with a Gaussian RBF kernel
  ``exp(−‖x − y‖² / 2σ²)``, σ = the median pairwise Euclidean distance within the
  reference set, scaled by α = 100 (the paper's default). 0 means the two sets are draws from
  one distribution; the unbiased estimator can dip slightly below 0 for near-identical sets.
  Needs ≥ 2 items on each side. Vectors are unit length here, so ‖x − y‖² = 2(1 − cos).
"""

from __future__ import annotations

import numpy as np

OP = "cohesion"
OP_VERSION = "cohesion@1"

Z_THRESHOLD = 3.5   # Iglewicz & Hoaglin's modified-z outlier cut
KAD_ALPHA = 100.0   # Chung et al. 2025 default scale


def centroid_distances(X: np.ndarray) -> np.ndarray:
    """Cosine distance of each (unit) row to the L2-normalized mean of the rows."""
    c = X.mean(axis=0)
    n = float(np.linalg.norm(c))
    if n == 0.0:
        return np.ones(len(X))
    return 1.0 - X @ (c / n)


def robust_z(d: np.ndarray) -> np.ndarray:
    """Modified z-score (Iglewicz & Hoaglin): 0.6745·(d − median)/MAD, with a mean-AD fallback."""
    d = np.asarray(d, dtype=np.float64)
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    if mad > 1e-12:
        return 0.6745 * (d - med) / mad
    mean_ad = float(np.mean(np.abs(d - med)))
    if mean_ad > 1e-12:
        return (d - med) / (mean_ad / 0.7979)
    return np.zeros_like(d)


def mean_pairwise_distance(X: np.ndarray) -> float | None:
    n = len(X)
    if n < 2:
        return None
    G = X @ X.T
    iu = np.triu_indices(n, k=1)
    return float(np.mean(1.0 - G[iu]))


def _sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    aa = np.sum(A * A, axis=1)[:, None]
    bb = np.sum(B * B, axis=1)[None, :]
    return np.maximum(aa + bb - 2.0 * (A @ B.T), 0.0)


def kad(X: np.ndarray, Y: np.ndarray, *, alpha: float = KAD_ALPHA,
        bandwidth: float | None = None) -> dict:
    """Kernel Audio Distance of eval set ``X`` against reference set ``Y``.

    Returns ``{"kad", "mmd2", "bandwidth", "n_eval", "n_ref"}``. σ defaults to the median
    pairwise distance within the reference set (Chung et al. 2025).
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    m, n = len(X), len(Y)
    if m < 2 or n < 2:
        raise ValueError(f"KAD needs >= 2 items per set (eval {m}, ref {n})")
    dyy = _sqdist(Y, Y)
    if bandwidth is None:
        iu = np.triu_indices(n, k=1)
        bandwidth = float(np.median(np.sqrt(dyy[iu])))
    if not bandwidth > 0:
        raise ValueError("KAD bandwidth is 0 (every reference vector is identical)")
    g = 1.0 / (2.0 * bandwidth * bandwidth)
    kxx = np.exp(-g * _sqdist(X, X))
    kyy = np.exp(-g * dyy)
    kxy = np.exp(-g * _sqdist(X, Y))
    mmd2 = ((kxx.sum() - np.trace(kxx)) / (m * (m - 1))
            + (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
            - 2.0 * kxy.mean())
    return {"kad": float(alpha * mmd2), "mmd2": float(mmd2), "bandwidth": bandwidth,
            "n_eval": m, "n_ref": n}


def cohesion(X: np.ndarray, *, ref: np.ndarray | None = None,
             z_threshold: float = Z_THRESHOLD) -> dict:
    """All cohesion measures for unit rows ``X`` (and KAD vs ``ref`` when given)."""
    X = np.asarray(X, dtype=np.float64)
    d = centroid_distances(X)
    z = robust_z(d)
    out = {
        "n": len(X),
        "centroid_dist": d,
        "outlier_z": z,
        "outlier": z > z_threshold,
        "pairwise_dist_mean": mean_pairwise_distance(X),
        "z_threshold": z_threshold,
    }
    if ref is not None:
        out["kad"] = kad(X, ref)
    return out


def _r(x, nd=4):
    return None if x is None else round(float(x), nd)


def cohesion_frames(items, *, model: str, ref_items=None,
                    z_threshold: float = Z_THRESHOLD, ref_label: str | None = None) -> list[dict]:
    """Derived frames: one per-item `feature` (role ``cohesion``) + one pack summary
    (role ``cohesion-pack``) carrying the pairwise spread, the outlier list and KAD."""
    from smplstream import frames as F

    from .vectors import matrix

    X = matrix(items)
    ref = matrix(ref_items) if ref_items else None
    res = cohesion(X, ref=ref, z_threshold=z_threshold)
    params = {"model": model, "z_threshold": z_threshold, "metric": "cosine"}
    if ref_label:
        params["ref"] = ref_label
    common = {"op": OP, "op_version": OP_VERSION, "params": params}
    out = []
    for i, it in enumerate(items):
        out.append(F.feature_frame({
            "cohesion.centroid_dist_cos": _r(res["centroid_dist"][i]),
            "cohesion.outlier_z": _r(res["outlier_z"][i], 3),
            "cohesion.outlier": bool(res["outlier"][i]),
        }, role="cohesion", of=it.id, lineage=[it.vector_id], **common))
    order = np.argsort(-res["outlier_z"])
    summary = {
        "cohesion.n": res["n"],
        "cohesion.pairwise_dist_cos_mean": _r(res["pairwise_dist_mean"]),
        "cohesion.centroid_dist_cos_median": _r(np.median(res["centroid_dist"])),
        "cohesion.outliers": [items[i].label for i in order if res["outlier"][i]],
        "cohesion.ranked": [{"label": items[i].label, "centroid_dist_cos": _r(res["centroid_dist"][i]),
                             "outlier_z": _r(res["outlier_z"][i], 3)} for i in order],
    }
    if "kad" in res:
        k = res["kad"]
        summary.update({"cohesion.kad": _r(k["kad"]), "cohesion.kad_mmd2": _r(k["mmd2"], 6),
                        "cohesion.kad_bandwidth": _r(k["bandwidth"]),
                        "cohesion.kad_ref_n": k["n_ref"]})
    # The reference set comes from a file, not this stream: it is named in params (`ref`), never
    # in lineage, which must only point at ids the stream can resolve.
    lineage = [it.vector_id for it in items]
    out.append(F.feature_frame(summary, role="cohesion-pack", lineage=lineage, **common))
    return out
