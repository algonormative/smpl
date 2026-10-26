"""Core of `smpl ask`: audio prep, one question, A/B in both orders, the claim record and its frame.

The model is a second ear. Nothing here puts a measurement in a prompt, and an A/B answer only
counts when both presentation orders agree (position bias, AudioJudge): disagreement becomes
``confidence: low`` with the per-order answers kept in ``raw`` and ``warnings``.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field

from . import questions as Q
from .backends import AskBackend, BackendError, Reply

OP = "ask"
OP_VERSION = "ask@1"
SEND_SR = 16000          # Gemini resamples to 16 kHz and downmixes to mono; we send exactly that
DEFAULT_MAX_SECONDS = 120.0


# ---------------------------------------------------------------------------
# Audio prep
# ---------------------------------------------------------------------------
def prepare_audio(path, *, max_seconds: float = DEFAULT_MAX_SECONDS) -> tuple[bytes, float]:
    """Decode ``path``, mix to mono, resample to 16 kHz, crop to ``max_seconds``; return 16-bit WAV bytes.

    Doing the downmix ourselves keeps the request small and makes the model's real input explicit
    (it would downmix anyway, which is why stereo / phase questions are routed to smpl).
    """
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = y.mean(axis=1)
    if int(sr) != SEND_SR:
        g = math.gcd(int(sr), SEND_SR)
        mono = resample_poly(mono, SEND_SR // g, int(sr) // g).astype("float32")
    mono = mono[: int(max_seconds * SEND_SR)]
    buf = io.BytesIO()
    sf.write(buf, np.clip(mono, -1.0, 1.0), SEND_SR, format="WAV", subtype="PCM_16")
    return buf.getvalue(), len(mono) / SEND_SR


# ---------------------------------------------------------------------------
# The claim record
# ---------------------------------------------------------------------------
@dataclass
class Call:
    order: str
    latency_s: float = 0.0
    input_tokens: int = 0
    audio_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    error: str | None = None


@dataclass
class Claim:
    question_id: str
    answer: object = None
    timestamps: list[float] = field(default_factory=list)
    confidence: str = "normal"          # "normal" | "low"
    model: str = ""
    backend: str = ""
    order: str = "single"               # "single" | "AB+BA"
    raw: dict = field(default_factory=dict)
    evidence: str = ""
    warnings: list[str] = field(default_factory=list)
    calls: list[Call] = field(default_factory=list)
    route: str | None = None
    sent: dict = field(default_factory=dict)

    @property
    def cost_usd(self) -> float | None:
        costs = [c.cost_usd for c in self.calls]
        return None if not costs or any(c is None for c in costs) else sum(costs)

    @property
    def latency_s(self) -> float:
        return sum(c.latency_s for c in self.calls)

    def params(self) -> dict:
        return {
            "question_id": self.question_id, "answer": self.answer, "timestamps": self.timestamps,
            "confidence": self.confidence, "model": self.model, "backend": self.backend,
            "order": self.order, "raw": self.raw, "evidence": self.evidence,
            "warnings": self.warnings, "route": self.route, "sent": self.sent,
            "cost_usd": None if self.cost_usd is None else round(self.cost_usd, 6),
            "latency_s": round(self.latency_s, 3),
            "calls": [vars(c) for c in self.calls],
            "cacheable": False,  # a remote model call is not a pure function of its inputs
        }


def _call_log(order: str, reply: Reply | None, err: str | None = None) -> Call:
    if reply is None:
        return Call(order=order, error=err)
    u = reply.usage
    return Call(order=order, latency_s=round(reply.latency_s, 3), input_tokens=u.input_tokens,
                audio_tokens=u.audio_tokens, output_tokens=u.output_tokens, cost_usd=reply.cost_usd)


def log_line(claim: Claim) -> str:
    """The per-call cost line written to stderr."""
    parts = []
    for c in claim.calls:
        cost = "$?" if c.cost_usd is None else f"${c.cost_usd:.5f}"
        parts.append(f"{c.order}: in={c.input_tokens}(audio {c.audio_tokens}) out={c.output_tokens} "
                     f"{cost} {c.latency_s:.1f}s" + (f" ERROR {c.error}" if c.error else ""))
    total = "$?" if claim.cost_usd is None else f"${claim.cost_usd:.5f}"
    return (f"ask: {claim.model} {claim.question_id} -> {claim.answer!r} "
            f"[{claim.confidence}] total {total} | " + " | ".join(parts))


def _flip(ans):
    return {"A": "B", "B": "A"}.get(ans, ans)


# ---------------------------------------------------------------------------
# Asking
# ---------------------------------------------------------------------------
def _one_call(backend: AskBackend, q: Q.Question, wavs: list[bytes], order: str):
    parts, slots = Q.user_parts(q)
    for slot, wav in zip(slots, wavs):
        parts[slot] = ("audio", wav)
    try:
        reply = backend.ask(Q.SYSTEM_PROMPT, parts)
    except BackendError as exc:
        return None, _call_log(order, None, str(exc)), str(exc)
    return reply, _call_log(order, reply), None


def ask(backend: AskBackend, q: Q.Question, wavs: list[bytes]) -> Claim:
    """Ask ``q`` of the prepared clip (kind clip) or pair (kind ab, always both orders)."""
    claim = Claim(question_id=q.id, model=backend.model, backend=backend.name, route=q.route)
    if q.kind == "clip":
        if len(wavs) != 1:
            raise ValueError(f"{q.id} takes exactly one clip")
        reply, call, err = _one_call(backend, q, wavs, "single")
        claim.calls.append(call)
        if reply is None:
            claim.confidence, claim.warnings = "low", [f"call failed: {err}"]
            return claim
        claim.model = reply.model or claim.model
        claim.raw["single"] = reply.text
        p = Q.parse_reply(q, reply.text)
        claim.evidence, claim.warnings = p.evidence, list(p.problems)
        hard = [x for x in p.problems if "no timestamps" not in x]
        if hard:
            claim.confidence = "low"
        else:
            claim.answer, claim.timestamps = p.answer, p.timestamps
        return claim

    if len(wavs) != 2:
        raise ValueError(f"{q.id} takes exactly two clips")
    claim.order = "AB+BA"
    answers = {}
    for order, ws in (("AB", wavs), ("BA", wavs[::-1])):
        reply, call, err = _one_call(backend, q, ws, order)
        claim.calls.append(call)
        if reply is None:
            claim.warnings.append(f"{order}: call failed: {err}")
            answers[order] = None
            continue
        claim.model = reply.model or claim.model
        claim.raw[order] = reply.text
        p = Q.parse_reply(q, reply.text)
        if p.problems:
            claim.warnings.append(f"{order}: " + "; ".join(p.problems))
            answers[order] = None
        else:
            answers[order] = p.answer if order == "AB" else _flip(p.answer)
            claim.evidence = claim.evidence or p.evidence
    ab, ba = answers["AB"], answers["BA"]
    if ab is not None and ab == ba:
        claim.answer = ab
    else:
        claim.confidence = "low"
        claim.warnings.append(f"orders disagree after mapping back: AB={ab!r} BA={ba!r}")
    return claim


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------
def claim_frame(claim: Claim, audio_ids: list[str]):
    from smplstream import frames as F

    q = Q.get(claim.question_id)
    text = Q.render_claim(q, claim.answer, claim.timestamps)
    if claim.confidence == "low":
        text += " [low confidence]"
    return F.text_frame(
        text, role="claim", of=audio_ids[0],
        lineage=list(audio_ids) if len(audio_ids) > 1 else None,
        op=OP, op_version=f"{OP_VERSION}+{claim.model}", params=claim.params(),
    )


def routed_claim_frame(q: Q.Question, audio_id: str, model: str):
    """A claim-less placeholder for a question smpl answers: no model call, nothing to referee."""
    from smplstream import frames as F

    claim = Claim(question_id=q.id, answer=None, model=model, route=q.route)
    claim.warnings.append(f"routed to smpl {q.smpl_op}: the model hears a mono 16 kHz downmix")
    return F.text_frame(
        f"{q.id}: not asked; smpl {q.smpl_op} answers this (the model hears a mono downmix).",
        role="claim", of=audio_id, op=OP, op_version=OP_VERSION, params=claim.params(),
    )
