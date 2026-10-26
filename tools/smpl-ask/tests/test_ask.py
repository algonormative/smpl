"""`smpl ask` core: bank, prompts, parsing, A/B both orders, Gemini wire + cost, claim frames,
crosscheck hand-off, ear-ledger export. The model is always stubbed (AF-06): no network, no key.

    uv run --group dev pytest tools/smpl-ask/tests -q
"""

from __future__ import annotations

import json
import re

import pytest
from helpers import ScriptedBackend, audio_slots
from smpl_ask import ask as A
from smpl_ask import backends as B
from smpl_ask import ledger as L
from smpl_ask import questions as Q

WAV_A, WAV_B = b"RIFF-A", b"RIFF-B"


# ---- the bank ------------------------------------------------------------------------------
def test_bank_matches_the_crosscheck_registry():
    """The ids smpl-ask can emit are exactly the ids crosscheck can referee."""
    from smpl_analysis import crosscheck as cc

    assert set(Q.bank()) == set(cc.QUESTIONS)


def test_routed_questions_are_the_stereo_and_fidelity_ones():
    routed = {q.id for q in Q.bank().values() if q.route == "smpl"}
    assert routed == {"stereo_width", "phase_problem", "true_peak", "lossy_cutoff"}
    assert all(Q.get(i).smpl_op for i in routed)
    assert Q.get("ab_tight_low_end").route == "ear"


def test_no_measurement_reaches_a_prompt():
    """Prompts are the question and the answer shape only: no bpm / lufs / swing / cents figures."""
    for q in Q.bank().values():
        parts, _ = Q.user_parts(q)
        text = Q.SYSTEM_PROMPT + " ".join(p for k, p in parts if k == "text")
        assert not re.search(r"\b(correlation|rhythm\.|loudness\.|qc\.|width\.|tonal\.)", text, re.IGNORECASE), q.id
        # the only digits allowed are the answer-range bounds and the reference pitch, "0 if in tune", 16 kHz, 8 kHz
        digits = set(re.findall(r"\d+(?:\.\d+)?", text))
        allowed = {str(abs(q.answer.get("min", 0))), str(abs(q.answer.get("max", 0))), "440", "100", "16", "8", "0"}
        assert digits <= allowed, (q.id, digits - allowed)


# ---- parsing -------------------------------------------------------------------------------
def test_parse_reply_variants():
    q = Q.get("tempo_bpm")
    assert Q.parse_reply(q, '{"answer": 128, "timestamps": []}').answer == 128.0
    assert Q.parse_reply(q, '```json\n{"answer": 128.4}\n```').answer == 128.4
    assert Q.parse_reply(q, 'Sure! {"answer": 90, "timestamps": []} hope that helps').answer == 90.0
    assert Q.parse_reply(q, '{"answer": null}').answer is None and Q.parse_reply(q, '{"answer": null}').ok
    for bad in ("not json", '{"foo": 1}', '{"answer": "fast"}', '{"answer": 9999}', '{"answer": true}'):
        assert not Q.parse_reply(q, bad).ok, bad
    d = Q.get("defect_location")
    assert not Q.parse_reply(d, '{"answer": true, "timestamps": []}').ok       # true needs timestamps
    assert Q.parse_reply(d, '{"answer": true, "timestamps": [1.3]}').timestamps == [1.3]
    assert Q.parse_reply(d, '{"answer": false, "timestamps": []}').ok
    e = Q.get("swing_feel")
    assert Q.parse_reply(e, '{"answer": "swung"}').answer == "swung"
    assert Q.parse_reply(e, '{"answer": "bouncy"}').problems


def test_render_claim():
    assert Q.render_claim(Q.get("tempo_bpm"), 128.0) == "The tempo is about 128 BPM."
    assert "at 1.30 s" in Q.render_claim(Q.get("defect_location"), True, [1.3])
    assert Q.render_claim(Q.get("ab_louder"), "B") == "B is louder than A."
    assert "could not tell" in Q.render_claim(Q.get("ab_louder"), None)


# ---- clip questions ------------------------------------------------------------------------
def test_clip_question_makes_one_call_and_a_claim():
    be = ScriptedBackend(lambda parts: {"answer": 128, "timestamps": [], "evidence": "four on the floor"})
    c = A.ask(be, Q.get("tempo_bpm"), [WAV_A])
    assert (c.answer, c.confidence, c.order, len(be.requests)) == (128.0, "normal", "single", 1)
    assert c.calls[0].audio_tokens == 320 and c.cost_usd is not None and c.cost_usd > 0
    system, parts = be.requests[0]
    assert audio_slots(parts) == [WAV_A] and "mono" in system


def test_malformed_reply_is_a_low_confidence_empty_claim():
    be = ScriptedBackend(lambda parts: "I think it is fast")
    c = A.ask(be, Q.get("tempo_bpm"), [WAV_A])
    assert c.answer is None and c.confidence == "low" and c.warnings


def test_backend_error_is_recorded_not_raised():
    be = ScriptedBackend(lambda parts: B.BackendError("gemini HTTP 500: boom"))
    c = A.ask(be, Q.get("tempo_bpm"), [WAV_A])
    assert c.answer is None and c.confidence == "low" and "boom" in c.warnings[0]
    assert c.calls[0].error and "ERROR" in A.log_line(c)


# ---- A/B: both orders, position bias -------------------------------------------------------
def _truthful(parts):
    """Always picks the clip that is WAV_A, whichever slot it is shown in."""
    first = audio_slots(parts)[0]
    return {"answer": "A" if first == WAV_A else "B", "timestamps": []}


def test_ab_runs_both_orders_and_maps_back():
    be = ScriptedBackend(_truthful)
    c = A.ask(be, Q.get("ab_louder"), [WAV_A, WAV_B])
    assert [audio_slots(p) for _, p in be.requests] == [[WAV_A, WAV_B], [WAV_B, WAV_A]]
    assert (c.answer, c.confidence, c.order) == ("A", "normal", "AB+BA")
    assert set(c.raw) == {"AB", "BA"} and len(c.calls) == 2


def test_ab_position_bias_becomes_low_confidence():
    """A model that says "A" whatever it is shown agrees with itself in neither order."""
    be = ScriptedBackend(lambda parts: {"answer": "A"})
    c = A.ask(be, Q.get("ab_louder"), [WAV_A, WAV_B])
    assert c.answer is None and c.confidence == "low"
    assert "orders disagree" in c.warnings[-1] and "AB='A'" in c.warnings[-1] and "BA='B'" in c.warnings[-1]


def test_ab_same_in_both_orders_is_a_normal_claim():
    c = A.ask(ScriptedBackend(lambda parts: {"answer": "same"}), Q.get("ab_louder"), [WAV_A, WAV_B])
    assert (c.answer, c.confidence) == ("same", "normal")


def test_ab_one_order_malformed_is_low():
    replies = iter([{"answer": "A"}, "garbage"])
    c = A.ask(ScriptedBackend(lambda parts: next(replies)), Q.get("ab_louder"), [WAV_A, WAV_B])
    assert c.answer is None and c.confidence == "low"


# ---- Gemini wire + cost --------------------------------------------------------------------
def _gemini_reply(text='{"answer": 128, "timestamps": []}'):
    return json.dumps({
        "candidates": [{"content": {"parts": [{"text": "(thinking)", "thought": True}, {"text": text}]}}],
        "usageMetadata": {"promptTokenCount": 650, "candidatesTokenCount": 40, "thoughtsTokenCount": 460,
                          "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 330},
                                                  {"modality": "AUDIO", "tokenCount": 320}]},
        "modelVersion": "gemini-3.8-flash",
    }).encode()


def test_gemini_request_response_and_cost():
    seen = {}

    def transport(url, headers, body, timeout):
        seen.update(url=url, headers=headers, body=json.loads(body))
        return 200, _gemini_reply()

    be = B.GeminiBackend(key="k-secret", transport=transport)
    c = A.ask(be, Q.get("tempo_bpm"), [WAV_A])
    assert seen["url"].endswith("/models/gemini-3.8-flash:generateContent")
    assert seen["headers"]["x-goog-api-key"] == "k-secret"
    body = seen["body"]
    assert body["generationConfig"] == {"temperature": 0, "responseMimeType": "application/json"}
    inline = [p for p in body["contents"][0]["parts"] if "inline_data" in p]
    assert len(inline) == 1 and inline[0]["inline_data"]["mime_type"] == "audio/wav"
    assert c.answer == 128.0                      # the thought part was ignored
    call = c.calls[0]
    assert (call.audio_tokens, call.input_tokens, call.output_tokens) == (320, 650, 500)
    expect = (330 * 0.75 + 320 * 0.75 + 500 * 3.75) / 1e6 if B.price_for("gemini-3.8-flash")[0] == 0.75 else None
    if expect is not None:
        assert call.cost_usd == pytest.approx(expect)
    assert "key" not in json.dumps(c.params()).lower().replace("keys", "")  # the key is never recorded


def test_gemini_price_tiers_and_unknown_model():
    import datetime as dt

    u = B.Usage(text_tokens=1_000_000, audio_tokens=0, output_tokens=1_000_000)
    assert B.cost_usd("gemini-3.8-flash", u, dt.date(2026, 12, 31)) == pytest.approx(0.75 + 3.75)
    assert B.cost_usd("gemini-3.8-flash", u, dt.date(2027, 1, 1)) == pytest.approx(1.50 + 7.50)
    assert B.cost_usd("gemini-9-imaginary", u) is None


def test_gemini_retries_429_then_succeeds_and_errors_surface():
    calls = []

    def transport(url, headers, body, timeout):
        calls.append(1)
        return (429, b'{"error":{"message":"slow down"}}') if len(calls) < 3 else (200, _gemini_reply())

    sleeps = []
    be = B.GeminiBackend(key="k", transport=transport, sleep=sleeps.append)
    assert A.ask(be, Q.get("tempo_bpm"), [WAV_A]).answer == 128.0 and len(calls) == 3 and sleeps == [1.0, 2.0]

    be = B.GeminiBackend(key="k", transport=lambda *a: (403, b'{"error":{"message":"API key not valid"}}'))
    c = A.ask(be, Q.get("tempo_bpm"), [WAV_A])
    assert c.confidence == "low" and "API key not valid" in c.warnings[0]


def test_network_error_is_retried_then_recorded():
    import urllib.error

    n = []

    def transport(*a):
        n.append(1)
        raise urllib.error.URLError("name resolution failed")

    c = A.ask(B.GeminiBackend(key="k", transport=transport, sleep=lambda s: None), Q.get("tempo_bpm"), [WAV_A])
    assert len(n) == 3 and c.confidence == "low" and "network error" in c.warnings[0]


def test_missing_key_is_unsupported_not_a_crash(monkeypatch):
    for env in ("SMPL_ASK_GEMINI_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    be = B.GeminiBackend(transport=lambda *a: pytest.fail("no key must mean no request"))
    with pytest.raises(B.Unsupported) as e:
        A.ask(be, Q.get("tempo_bpm"), [WAV_A])
    assert "GEMINI_API_KEY" in e.value.hint
    monkeypatch.setenv("GEMINI_API_KEY", "abc")
    assert B.resolve_key() == "abc" and B.redact("abc") == "<redacted:3ch>" and "abc" not in B.redact("abc")


def test_local_backend_is_a_stub():
    with pytest.raises(B.Unsupported):
        B.get_backend("local").ask("s", [])
    with pytest.raises(B.Unsupported):
        B.get_backend("nope")


# ---- frames, crosscheck hand-off, ledger ---------------------------------------------------
def _audio(i):
    from smplstream import frames as F

    return F.audio_frame("blake3:" + str(i) * 64, sr=44100, ch=1, dur=4.0, role="source")


def test_claim_frame_shape_and_crosscheck_roundtrip():
    from smpl_analysis import crosscheck as cc
    from smplstream import frames as F

    a = _audio(1)
    claim = A.ask(ScriptedBackend(lambda p: {"answer": 90, "timestamps": []}), Q.get("tempo_bpm"), [WAV_A])
    f = A.claim_frame(claim, [a["id"]])
    assert (f["kind"], f["role"], f["of"], f["op"]) == ("text", "claim", a["id"], "ask")
    assert f["data"] == "The tempo is about 90 BPM." and f["params"]["question_id"] == "tempo_bpm"
    assert f["params"]["cacheable"] is False and f["params"]["cost_usd"] is not None
    beats = F.feature_frame({"rhythm.bpm": 128.0, "rhythm.bpm_candidates": []}, role="beats", of=a["id"])
    out = cc.crosscheck([a, beats, f], drop_contradicted=True)
    res = next(x["data"] for x in out if x.get("role") == "crosscheck")
    assert res["status"] == "contradicted" and f["id"] not in {x["id"] for x in out}


def test_ab_claim_frame_lineage_and_low_confidence_text():
    a, b = _audio(1), _audio(2)
    claim = A.ask(ScriptedBackend(lambda p: {"answer": "A"}), Q.get("ab_louder"), [WAV_A, WAV_B])
    f = A.claim_frame(claim, [a["id"], b["id"]])
    assert f["of"] == a["id"] and f["lineage"] == [a["id"], b["id"]]
    assert f["params"]["confidence"] == "low" and "[low confidence]" in f["data"]


def test_ledger_export_drops_contradicted_and_routed():
    from smpl_analysis import crosscheck as cc
    from smplstream import frames as F

    a = _audio(1)
    mk = lambda ans, **kw: A.claim_frame(
        A.ask(ScriptedBackend(lambda p: {"answer": ans, "timestamps": kw.get("ts", [])}), Q.get(kw["q"]), [WAV_A]), [a["id"]])
    tempo_bad = mk(90, q="tempo_bpm")
    tempo_ok = mk(128, q="tempo_bpm")
    defect = mk(True, q="defect_location", ts=[1.3])          # no qc frames: unverifiable
    routed = A.routed_claim_frame(Q.get("stereo_width"), a["id"], "m")
    beats = F.feature_frame({"rhythm.bpm": 128.0, "rhythm.bpm_candidates": []}, role="beats", of=a["id"])
    stream = [a, beats, tempo_bad, tempo_ok, defect, routed]

    raw = L.export(stream)                                    # no crosscheck yet: every real claim
    assert [i["question_id"] for i in raw["items"]] == ["tempo_bpm", "tempo_bpm", "defect_location"]

    doc = L.export(cc.crosscheck(stream))
    got = {i["claim_id"]: i for i in doc["items"]}
    assert tempo_bad["id"] not in got and routed["id"] not in got
    assert got[tempo_ok["id"]]["crosscheck"] == "supported"
    d = got[defect["id"]]
    assert d["crosscheck"] == "unverifiable" and d["t"] == 1.3 and d["kind"] == "matches"
    assert d["choices"] == ["yes", "partly", "no"] and d["audio"] == [a["hash"]] and "1.30 s" in d["text"]


def test_ledger_ab_items_are_choices():
    a, b = _audio(1), _audio(2)
    claim = A.ask(ScriptedBackend(_truthful), Q.get("ab_louder"), [WAV_A, WAV_B])
    doc = L.export([a, b, A.claim_frame(claim, [a["id"], b["id"]])])
    item = doc["items"][0]
    assert item["kind"] == "choice" and item["claimed"] == "A" and item["audio"] == [a["hash"], b["hash"]]


# ---- the CLI end to end, stubbed -----------------------------------------------------------
class _Stdio:
    """A stand-in for sys.stdin / sys.stdout exposing ``.buffer`` the way the frame code expects."""

    def __init__(self, data: bytes = b""):
        import io

        self.buffer = io.BytesIO(data)

    def isatty(self):
        return False


def _run_cli(monkeypatch, frames, argv):
    from smpl_ask import cli
    from smplstream import ndjson

    stdin, stdout = _Stdio(b"".join(ndjson.dumps(f) + b"\n" for f in frames)), _Stdio()
    monkeypatch.setattr(cli.sys, "stdin", stdin)
    monkeypatch.setattr(cli.sys, "stdout", stdout)
    rc = cli.main(argv)
    return rc, [json.loads(line) for line in stdout.buffer.getvalue().splitlines()]


@pytest.fixture()
def one_clip(tmp_path, monkeypatch):
    import numpy as np
    import soundfile as sf
    from smplstream import cas
    from smplstream import frames as F

    for env in ("SMPL_ASK_GEMINI_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    wav = tmp_path / "x.wav"
    sf.write(str(wav), (0.3 * np.sin(np.arange(44100) / 20.0)).astype("float32"), 44100)
    return F.audio_frame(cas.put_audio_file(wav), sr=44100, ch=1, dur=1.0, role="source")


def test_cli_without_a_key_degrades_to_an_unsupported_frame(one_clip, monkeypatch, capsys):
    rc, frames = _run_cli(monkeypatch, [one_clip], ["--q", "tempo_bpm"])
    assert rc == 0 and frames[0]["id"] == one_clip["id"]
    assert frames[-1]["kind"] == "error" and frames[-1]["data"]["code"] == "unsupported"
    assert "GEMINI_API_KEY" in capsys.readouterr().err


def test_cli_routed_question_never_reaches_a_backend(one_clip, monkeypatch, capsys):
    rc, frames = _run_cli(monkeypatch, [one_clip], ["--q", "stereo_width"])
    assert rc == 0 and frames[-1]["role"] == "claim"
    assert frames[-1]["params"]["route"] == "smpl" and frames[-1]["params"]["answer"] is None
    assert "routed to smpl width" in capsys.readouterr().err


def test_cli_dry_run_prices_without_a_key(one_clip, monkeypatch, capsys):
    rc, frames = _run_cli(monkeypatch, [one_clip], ["--q", "tempo_bpm", "--dry-run"])
    assert rc == 0 and frames == [one_clip]
    assert "1 call(s) to gemini-3.8-flash" in capsys.readouterr().err


def test_cli_ab_needs_two_clips(one_clip, monkeypatch, capsys):
    rc, _ = _run_cli(monkeypatch, [one_clip], ["--q", "ab_louder"])
    assert rc == 2 and "two audio frames" in capsys.readouterr().err
    assert _run_cli(monkeypatch, [one_clip], ["--q", "nope"])[0] == 2
