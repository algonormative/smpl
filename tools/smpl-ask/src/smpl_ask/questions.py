"""The question bank, prompt construction, reply parsing and claim rendering.

The bank is data (``questions.json``); this module is the only code that interprets it. No prompt
ever carries a measurement: the model gets the audio and one narrow question.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources

SYSTEM_PROMPT = (
    "You are a careful listening assistant for electronic music production. You hear each clip only "
    "as a mono, 16 kHz downmix, so you cannot judge stereo image, phase between channels, or content "
    "above 8 kHz. Answer exactly one question about the audio. Reply with ONE JSON object and nothing "
    "else (no prose, no code fence). If you cannot tell, say so with null (or \"unsure\") rather than "
    "guessing. Timestamps are seconds from the start of the clip you are describing."
)


@dataclass(frozen=True)
class Question:
    id: str
    kind: str                 # "clip" | "ab"
    route: str | None         # None | "smpl" | "ear"
    prompt: str
    answer: dict
    timestamps: str           # "none" | "optional" | "required"
    claim: object             # str template or {answer-value: text}
    smpl_op: str | None = None

    @property
    def answer_type(self) -> str:
        return self.answer["type"]

    def answer_format(self) -> str:
        t = self.answer_type
        if t == "number":
            a = f"<number from {self.answer['min']} to {self.answer['max']}, or null>"
        elif t == "bool":
            a = "true or false (or null if you cannot tell)"
        else:
            a = " | ".join(f'"{v}"' for v in self.answer["values"])
        ts = {
            "none": "[]",
            "optional": "[seconds, ...]  (empty list if you cannot place it)",
            "required": "[seconds, ...]  (one entry per instance; empty only if the answer is false)",
        }[self.timestamps]
        return (f'{{"answer": {a}, "timestamps": {ts}, '
                '"evidence": "<one short phrase on what you heard>"}')


@lru_cache(maxsize=1)
def bank() -> dict[str, Question]:
    raw = json.loads(resources.files("smpl_ask").joinpath("questions.json").read_text("utf-8"))
    out = {}
    for q in raw["questions"]:
        out[q["id"]] = Question(
            id=q["id"], kind=q["kind"], route=q.get("route"), prompt=q["prompt"],
            answer=q["answer"], timestamps=q["timestamps"], claim=q["claim"],
            smpl_op=q.get("smpl_op"),
        )
    return out


def get(qid: str) -> Question:
    try:
        return bank()[qid]
    except KeyError:
        raise KeyError(f"unknown question {qid!r}; known: {sorted(bank())}") from None


def user_parts(q: Question) -> tuple[list[tuple[str, object]], list[int]]:
    """Prompt scaffolding with audio slots: returns (parts, audio_slot_indices) where a part is
    ("text", str) or ("audio", None) to be filled by the caller (clip 0, clip 1, ...)."""
    tail = ("text", f"{q.prompt}\n\nReply with exactly this JSON shape:\n{q.answer_format()}")
    if q.kind == "ab":
        parts = [("text", "Clip A:"), ("audio", None), ("text", "Clip B:"), ("audio", None), tail]
    else:
        parts = [("text", "Clip:"), ("audio", None), tail]
    return parts, [i for i, (k, _) in enumerate(parts) if k == "audio"]


# ---------------------------------------------------------------------------
# Reply parsing
# ---------------------------------------------------------------------------
@dataclass
class Parsed:
    answer: object = None
    timestamps: list[float] = field(default_factory=list)
    evidence: str = ""
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_reply(q: Question, text: str) -> Parsed:
    """Parse + validate a model reply against the question's answer type. Never raises."""
    out = Parsed()
    body = _FENCE.sub("", (text or "").strip())
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", body, re.DOTALL)  # tolerate a stray sentence around the object
        try:
            obj = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            obj = None
    if not isinstance(obj, dict) or "answer" not in obj:
        out.problems.append("reply is not a JSON object with an `answer` key")
        return out

    ans = obj["answer"]
    t = q.answer_type
    if ans is None or ans == "unsure":
        out.answer = "unsure" if t == "enum" else None
    elif t == "number":
        if isinstance(ans, bool) or not isinstance(ans, (int, float)) or not math.isfinite(ans):
            out.problems.append(f"answer {ans!r} is not a number")
        elif not q.answer["min"] <= ans <= q.answer["max"]:
            out.problems.append(f"answer {ans} outside [{q.answer['min']}, {q.answer['max']}]")
        else:
            out.answer = float(ans)
    elif t == "bool":
        if isinstance(ans, bool):
            out.answer = ans
        else:
            out.problems.append(f"answer {ans!r} is not true/false")
    else:
        if ans in q.answer["values"]:
            out.answer = ans
        else:
            out.problems.append(f"answer {ans!r} not in {q.answer['values']}")

    ts = obj.get("timestamps", [])
    if not isinstance(ts, list) or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in ts):
        out.problems.append("timestamps is not a list of numbers")
    else:
        out.timestamps = [float(x) for x in ts if x >= 0]
    out.evidence = str(obj.get("evidence", ""))[:300]

    if q.timestamps == "required" and out.answer is True and not out.timestamps:
        out.problems.append("answer is true but no timestamps were given")
    return out


# ---------------------------------------------------------------------------
# Rendering (claim text, used in claim frames and ear-ledger questions)
# ---------------------------------------------------------------------------
def _fmt(x) -> str:
    return f"{x:g}" if isinstance(x, float) else str(x)


def render_claim(q: Question, answer, timestamps=()) -> str:
    if answer is None or answer == "unsure":
        text = "The model could not tell."
    elif isinstance(q.claim, dict):
        key = {True: "true", False: "false"}.get(answer, answer) if isinstance(answer, bool) else answer
        text = q.claim.get(str(key), f"{q.id}: {answer}")
    else:
        text = q.claim.format(answer=_fmt(answer))
    if timestamps:
        text += " (at " + ", ".join(f"{t:.2f} s" for t in timestamps) + ")"
    return text
