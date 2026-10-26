"""Export `smpl ask` claims as ear-ledger questions for a listening page (vault-16dd.1).

Nick will not describe a sound unprompted but can say reliably whether it matches a description.
So each claim becomes one forced-choice "matches?" item carrying the exact text being checked and,
where the model gave one, a timestamp to jump to. Claims `smpl crosscheck` marked ``contradicted``
are dropped (the meter already refuted them); ``supported`` and ``unverifiable`` ones go to the ear,
the latter being exactly the ones only an ear can settle. With no crosscheck frames in the stream
every claim is exported.

Document shape (``v: 1``)::

    {"v": 1, "source": "smpl-ask", "items": [
       {"claim_id", "question_id", "audio": [hash, ...], "audio_ids": [...],
        "text": "The tempo is about 128 BPM.", "t": 1.3 | null, "timestamps": [...],
        "kind": "matches" | "choice", "choices": ["yes", "partly", "no"] | ["A", "B", "same"],
        "crosscheck": "supported" | "unverifiable" | null, "confidence": "normal" | "low",
        "model": "..."}]}

A/B claims use ``kind: choice`` (the page asks "A or B?" and compares to the claim); everything
else is ``matches``. The ledger's yes / partly / no is the ground truth for the per-question hit rate.
"""

from __future__ import annotations

KEEP = ("supported", "unverifiable")


def export(frames: list[dict]) -> dict:
    audio_by_id = {f.get("id"): f for f in frames if f.get("kind") == "audio"}
    verdicts = {}
    for f in frames:
        if f.get("kind") == "feature" and f.get("role") == "crosscheck":
            verdicts[(f.get("data") or {}).get("claim_id")] = (f.get("data") or {}).get("status")
    have_cc = bool(verdicts)
    items = []
    for f in frames:
        if not (f.get("kind") == "text" and f.get("role") == "claim"):
            continue
        p = f.get("params") or {}
        if p.get("answer") is None or p.get("route") == "smpl":
            continue  # nothing was claimed, or smpl answers it
        status = verdicts.get(f.get("id"))
        if have_cc and status not in KEEP:
            continue
        ids = list(f.get("lineage") or ([f["of"]] if f.get("of") else []))
        is_ab = len(ids) > 1
        ts = list(p.get("timestamps") or [])
        items.append({
            "claim_id": f.get("id"), "question_id": p.get("question_id"),
            "audio": [(audio_by_id.get(i) or {}).get("hash") for i in ids], "audio_ids": ids,
            "text": f.get("data"), "t": ts[0] if ts else None, "timestamps": ts,
            "kind": "choice" if is_ab else "matches",
            "choices": ["A", "B", "same"] if is_ab else ["yes", "partly", "no"],
            "claimed": p.get("answer") if is_ab else None,
            "crosscheck": status, "confidence": p.get("confidence", "normal"), "model": p.get("model"),
        })
    return {"v": 1, "source": "smpl-ask", "items": items}
