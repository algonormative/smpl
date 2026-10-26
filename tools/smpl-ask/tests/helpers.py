"""Stub backends for the smpl-ask tests. Nothing here touches the network."""

from __future__ import annotations

import json

from smpl_ask.backends import AskBackend, BackendError, Reply, Usage, cost_usd


class ScriptedBackend(AskBackend):
    """Replies from a callable ``fn(parts) -> str | dict``; records every request."""

    name = "scripted"

    def __init__(self, fn, model="gemini-3.8-flash"):
        self.fn, self.model, self.requests = fn, model, []

    def ask(self, system, parts):
        self.requests.append((system, parts))
        out = self.fn(parts)
        if isinstance(out, Exception):
            raise out
        text = out if isinstance(out, str) else json.dumps(out)
        u = Usage(text_tokens=300, audio_tokens=320 * sum(1 for k, _ in parts if k == "audio"), output_tokens=100)
        return Reply(text=text, usage=u, latency_s=0.5, cost_usd=cost_usd(self.model, u), model=self.model)


def audio_slots(parts):
    return [p for k, p in parts if k == "audio"]


__all__ = ["BackendError", "ScriptedBackend", "audio_slots"]
