"""Pack coverage: does a set of samples contain the families a brief asks for, in quota?

The brief (JSON) names each required family with a quota and how to recognise it::

    {
      "model": "clap",                       // optional: the embedding model to use
      "families": {
        "kick-driven": {
          "quota": 2,                        // minimum count (default 1)
          "max": 4,                          // optional ceiling (over-representation)
          "text": ["a distorted, overdriven kick drum", "a saturated techno kick"],
          "tag": "driven"                    // optional regex over the item label
        }
      }
    }

Two assignment modes, each item to exactly one family:

- ``text`` — every item goes to the family whose text pole (the family's prompts pooled the
  vault-3hvg way: L2-normalize each, mean, L2-normalize) it is nearest to by cosine. Reports the
  winning score and the margin over the runner-up; a small margin means the assignment is a
  coin toss, which is information, not an error. Audio-text cosines run low and compressed
  (the modality gap), so read margins relative to each other, never as absolute confidences.
- ``tag`` — an item goes to the first family (in brief order) whose ``tag`` regex matches its
  label (the source file stem); items no tag matches are reported as unassigned.

Per family it reports count, quota, max, ``delta = count − quota``, and the flags ``met``
(count ≥ quota) and ``over`` (count > max). Measured vs intended only — whether a missing
family means re-breed or ship is the skill's call.
"""

from __future__ import annotations

import json

import numpy as np

OP = "coverage"
OP_VERSION = "coverage@1"


def load_brief(path_or_doc) -> dict:
    """Read and normalize a coverage brief (path or already-parsed dict)."""
    doc = path_or_doc
    if not isinstance(doc, dict):
        with open(path_or_doc, encoding="utf-8") as fh:
            doc = json.load(fh)
    fams = doc.get("families")
    if not isinstance(fams, dict) or not fams:
        raise ValueError("coverage brief needs a non-empty `families` object")
    from .vectors import as_prompts

    out = {}
    for name, spec in fams.items():
        spec = spec or {}
        quota = int(spec.get("quota", 1))
        mx = spec.get("max")
        out[name] = {"quota": quota, "max": None if mx is None else int(mx),
                     "text": as_prompts(spec.get("text")), "tag": spec.get("tag")}
    return {"model": doc.get("model"), "families": out}


def default_mode(brief: dict) -> str:
    fams = brief["families"].values()
    if all(f["text"] for f in fams):
        return "text"
    if all(f["tag"] for f in fams):
        return "tag"
    raise ValueError("every family needs `text` (text mode) or every family a `tag` (tag mode)")


def assign_text(X: np.ndarray, poles: np.ndarray):
    """Nearest pole per row → (index, score, margin over the runner-up)."""
    S = X @ poles.T
    idx = S.argmax(axis=1)
    top = S[np.arange(len(S)), idx]
    if S.shape[1] > 1:
        second = np.sort(S, axis=1)[:, -2]
    else:
        second = np.full(len(S), np.nan)
    return idx, top, top - second


def tally(names: list[str], assigned: list, brief: dict, labels: list[str]) -> dict:
    """Count assignments against quotas. ``assigned[i]`` is a family name or None."""
    fams = {}
    for name in names:
        spec = brief["families"][name]
        members = [labels[i] for i, a in enumerate(assigned) if a == name]
        c = len(members)
        fams[name] = {"count": c, "quota": spec["quota"], "max": spec["max"],
                      "delta": c - spec["quota"], "met": c >= spec["quota"],
                      "over": spec["max"] is not None and c > spec["max"], "items": members}
    return {
        "families": fams,
        "missing": [n for n in names if not fams[n]["met"]],
        "over": [n for n in names if fams[n]["over"]],
        "unassigned": [labels[i] for i, a in enumerate(assigned) if a is None],
    }


def coverage_frames(items, brief: dict, *, model: str, mode: str, text_vecs=None,
                    brief_label: str | None = None) -> list[dict]:
    """Per-item `feature` frames (role ``coverage``) + one pack summary (role ``coverage-pack``).

    ``text_vecs`` maps prompt → unit text vector (needed in text mode).
    """
    from smplstream import frames as F

    from .vectors import matrix, pole, tag_match

    names = list(brief["families"])
    labels = [it.label for it in items]
    per_item: list[dict] = []
    if mode == "text":
        poles = np.stack([pole([text_vecs[p] for p in brief["families"][n]["text"]]) for n in names])
        idx, top, margin = assign_text(matrix(items), poles)
        assigned = [names[i] for i in idx]
        for k in range(len(items)):
            per_item.append({"coverage.family": assigned[k],
                             "coverage.score_cos": round(float(top[k]), 4),
                             "coverage.margin_cos": None if np.isnan(margin[k])
                             else round(float(margin[k]), 4)})
    elif mode == "tag":
        assigned = []
        for lab in labels:
            hit = next((n for n in names if brief["families"][n]["tag"]
                        and tag_match(brief["families"][n]["tag"], lab)), None)
            assigned.append(hit)
            per_item.append({"coverage.family": hit})
    else:
        raise ValueError(f"unknown coverage mode {mode!r} (text|tag)")

    t = tally(names, assigned, brief, labels)
    params = {"model": model, "mode": mode}
    if brief_label:
        params["brief"] = brief_label
    common = {"op": OP, "op_version": OP_VERSION, "params": params}
    out = [F.feature_frame(d, role="coverage", of=it.id, lineage=[it.vector_id], **common)
           for it, d in zip(items, per_item)]
    out.append(F.feature_frame({
        "coverage.mode": mode,
        "coverage.n": len(items),
        "coverage.families": t["families"],
        "coverage.missing": t["missing"],
        "coverage.over": t["over"],
        "coverage.unassigned": t["unassigned"],
    }, role="coverage-pack", lineage=[it.vector_id for it in items], **common))
    return out
