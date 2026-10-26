"""Audio-LLM backends, pricing and key handling for `smpl ask`.

* :class:`GeminiBackend` — plain REST (``generateContent``), no SDK. The key is env-var-first
  (``SMPL_ASK_GEMINI_KEY``, then ``GEMINI_API_KEY``, then ``GOOGLE_API_KEY``), never printed or
  written into frame params (:func:`redact`).
* :class:`LocalBackend` — a stub for a later Qwen3-Omni / MOSS-Audio backend. It documents the
  interface a local model must meet and raises :class:`Unsupported`; nothing is built.
* The transport is injectable, so tests stub the network completely.

Cost is logged per call: token counts from the response's ``usageMetadata`` times the model's
published price, kept as :data:`PRICES` (USD per 1M tokens). Prices were read from
ai.google.dev/gemini-api/docs/pricing on 2026-10-26 (through a page summariser — re-check before
trusting a total). A model with no entry reports ``cost_usd: null`` rather than a guess.
"""

from __future__ import annotations

import base64
import datetime as _dt
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field

DEFAULT_MODEL = "gemini-3.8-flash"
API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
AUDIO_TOKENS_PER_SECOND = 32  # Gemini's documented audio tokenisation (countTokens: 10 s -> 320)

# USD per 1M tokens: (valid_until ISO date or None, text_in, audio_in, out incl. thinking).
# Tiers are read in order; the first whose date has not passed applies.
PRICES: dict[str, list[tuple[str | None, float, float, float]]] = {
    "gemini-3.8-flash": [("2026-12-31", 0.75, 0.75, 3.75), (None, 1.50, 1.50, 7.50)],
    "gemini-3.1-pro-preview": [(None, 2.00, 2.00, 12.00)],          # prompts <= 200k tokens
    "gemini-3.1-flash-lite": [(None, 0.25, 0.50, 1.50)],
    "gemini-3.5-flash-lite": [(None, 0.30, 0.30, 2.50)],
}
PRICE_SOURCE = "ai.google.dev/gemini-api/docs/pricing, read 2026-10-26 via a page summariser"


class Unsupported(Exception):
    """The backend cannot run here (no key, not built). Carries the stderr hint."""

    def __init__(self, message: str, *, hint: str):
        super().__init__(message)
        self.hint = hint


class BackendError(Exception):
    """A call reached the backend and failed (HTTP error, blocked prompt, empty reply)."""


def redact(key: str | None) -> str:
    return "<unset>" if not key else f"<redacted:{len(key)}ch>"


@dataclass
class Usage:
    text_tokens: int = 0
    audio_tokens: int = 0
    output_tokens: int = 0      # candidates + thinking
    thinking_tokens: int = 0

    @property
    def input_tokens(self) -> int:
        return self.text_tokens + self.audio_tokens


@dataclass
class Reply:
    text: str
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0
    cost_usd: float | None = None
    model: str = ""


def price_for(model: str, today: _dt.date | None = None):
    today = today or _dt.datetime.now(_dt.timezone.utc).date()
    for until, tin, ain, out in PRICES.get(model, []):
        if until is None or today <= _dt.date.fromisoformat(until):
            return tin, ain, out
    return None


def cost_usd(model: str, usage: Usage, today: _dt.date | None = None) -> float | None:
    p = price_for(model, today)
    if p is None:
        return None
    tin, ain, out = p
    return (usage.text_tokens * tin + usage.audio_tokens * ain + usage.output_tokens * out) / 1e6


# A prompt part: ("text", str) or ("audio", wav_bytes).
Part = tuple[str, object]
Transport = Callable[[str, dict, bytes, float], tuple[int, bytes]]


def _urllib_transport(url: str, headers: dict, body: bytes, timeout: float) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def resolve_key() -> str | None:
    for env in ("SMPL_ASK_GEMINI_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        if os.environ.get(env):
            return os.environ[env]
    return None


class AskBackend:
    """Interface: send audio parts + a question, get a text reply with usage. Nothing else."""

    name = "base"
    model = ""

    def require_key(self) -> str | None:
        """Raise :class:`Unsupported` early (before any audio is prepared) if the backend cannot run."""
        return None

    def ask(self, system: str, parts: list[Part]) -> Reply:  # pragma: no cover - interface
        raise NotImplementedError

    def describe(self) -> dict:
        return {"backend": self.name, "model": self.model}


class GeminiBackend(AskBackend):
    name = "gemini"

    def __init__(self, model: str | None = None, *, key: str | None = None,
                 transport: Transport | None = None, thinking_level: str | None = None,
                 timeout: float = 120.0, retries: int = 3, sleep: Callable[[float], None] = time.sleep):
        self.model = model or os.environ.get("SMPL_ASK_MODEL") or DEFAULT_MODEL
        self._key = key
        self.transport = transport or _urllib_transport
        self.thinking_level = thinking_level
        self.timeout, self.retries, self._sleep = timeout, retries, sleep

    @property
    def key(self) -> str | None:
        return self._key or resolve_key()

    def require_key(self) -> str:
        k = self.key
        if not k:
            raise Unsupported("no Gemini API key",
                              hint="export GEMINI_API_KEY=... (the key lives on the Mac mini; run there)")
        return k

    def build_request(self, system: str, parts: list[Part]) -> dict:
        wire = []
        for kind, payload in parts:
            if kind == "text":
                wire.append({"text": payload})
            else:
                wire.append({"inline_data": {"mime_type": "audio/wav",
                                             "data": base64.b64encode(payload).decode("ascii")}})
        gen = {"temperature": 0, "responseMimeType": "application/json"}
        if self.thinking_level:
            gen["thinkingConfig"] = {"thinkingLevel": self.thinking_level}
        return {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": wire}],
                "generationConfig": gen}

    def ask(self, system: str, parts: list[Part]) -> Reply:
        key = self.require_key()
        url = f"{API_ROOT}/models/{self.model}:generateContent"
        headers = {"x-goog-api-key": key, "content-type": "application/json"}
        body = json.dumps(self.build_request(system, parts)).encode()
        t0 = time.monotonic()
        status, raw = 0, b""
        for attempt in range(self.retries):
            try:
                status, raw = self.transport(url, headers, body, self.timeout)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:  # DNS, refused, timeout
                status, raw = 599, json.dumps({"error": {"message": f"network error: {exc}"}}).encode()
            if status not in (429, 500, 503, 599):
                break
            if attempt + 1 < self.retries:
                self._sleep(2.0 ** attempt)
        latency = time.monotonic() - t0
        try:
            doc = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            doc = {}
        if status != 200:
            msg = (doc.get("error") or {}).get("message", raw[:200].decode("utf-8", "replace"))
            raise BackendError(f"gemini HTTP {status}: {msg}")
        cands = doc.get("candidates") or []
        if not cands:
            raise BackendError(f"gemini returned no candidates (promptFeedback={doc.get('promptFeedback')})")
        text = "".join(p.get("text", "") for p in (cands[0].get("content") or {}).get("parts", [])
                       if not p.get("thought"))
        usage = _usage(doc.get("usageMetadata") or {})
        return Reply(text=text, usage=usage, latency_s=latency,
                     cost_usd=cost_usd(self.model, usage), model=doc.get("modelVersion") or self.model)


def _usage(meta: dict) -> Usage:
    by_mod = {d.get("modality"): int(d.get("tokenCount", 0)) for d in meta.get("promptTokensDetails", [])}
    total_in = int(meta.get("promptTokenCount", sum(by_mod.values())))
    audio = by_mod.get("AUDIO", 0)
    think = int(meta.get("thoughtsTokenCount", 0))
    return Usage(text_tokens=max(total_in - audio, 0), audio_tokens=audio,
                 output_tokens=int(meta.get("candidatesTokenCount", 0)) + think, thinking_tokens=think)


class LocalBackend(AskBackend):
    """Stub for a local audio LLM (Qwen3-Omni, MOSS-Audio; reuse the smplcat AF3/Qwen backends).

    A real one implements :meth:`AskBackend.ask`: take the same ``system`` + ``parts`` (text and
    WAV bytes), run the model, return a :class:`Reply` whose ``text`` is the JSON object. Cost is
    ``0.0``. Not built: the point of this bead is the claim / crosscheck / calibration scaffolding.
    """

    name = "local"

    def __init__(self, model: str | None = None, **_):
        self.model = model or "unset"

    def ask(self, system: str, parts: list[Part]) -> Reply:
        raise Unsupported("the local audio-LLM backend is not built",
                          hint="use --backend gemini; a Qwen3-Omni / MOSS-Audio backend is a follow-up (vault-12j3)")


BACKENDS = {"gemini": GeminiBackend, "local": LocalBackend}


def get_backend(name: str, model: str | None = None, **kw) -> AskBackend:
    try:
        return BACKENDS[name](model, **kw)
    except KeyError:
        raise Unsupported(f"unknown backend {name!r}; known: {sorted(BACKENDS)}",
                          hint="pick --backend gemini") from None
