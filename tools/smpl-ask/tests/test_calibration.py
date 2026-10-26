"""Known-answer calibration set: determinism, truth validated against smpl's own measurements,
and the runner's per-question arithmetic on STUBBED model answers (no network, no spend).

    uv run --group dev pytest tools/smpl-ask/tests -q
"""

from __future__ import annotations

import json
import math

import pytest
from helpers import ScriptedBackend, audio_slots
from smpl_ask import ask as A
from smpl_ask import calibration as C
from smpl_ask import questions as Q


@pytest.fixture(scope="module")
def cases(tmp_path_factory):
    return C.build_cases(tmp_path_factory.mktemp("cal"), include_routed=True)


def _by_q(cases):
    out = {}
    for c in cases:
        out.setdefault(c.question_id, []).append(c)
    return out


def test_set_shape_and_determinism(cases, tmp_path):
    n = {q: len(v) for q, v in _by_q(cases).items()}
    assert n == {"tempo_bpm": 5, "swing_feel": 6, "timing_drift": 6, "detune": 6, "clipping": 6,
                 "defect_location": 6, "ab_louder": 4, "ab_swingier": 4, "ab_tight_low_end": 4,
                 "stereo_width": 2, "phase_problem": 2}
    default = {c.question_id for c in C.build_cases(tmp_path / "a")}
    assert not default & {"stereo_width", "phase_problem"}      # routed questions are opt-in
    again = C.build_cases(tmp_path / "b", include_routed=True)
    for c1, c2 in zip(cases, again, strict=True):
        assert c1.id == c2.id and c1.truth == c2.truth
        assert [p.read_bytes() for p in c1.files] == [p.read_bytes() for p in c2.files]
    blobs = [f.read_bytes() for c in cases for f in c.files]
    assert len(set(blobs)) == len(blobs)            # no two cases share audio (a stub could not tell them apart)
    # A/B truth is balanced across placements, so "always A" scores 50%, not 100%
    for q in ("ab_louder", "ab_swingier", "ab_tight_low_end"):
        assert sorted(c.truth["answer"] for c in _by_q(cases)[q]) == ["A", "A", "B", "B"]


def test_every_case_answers_a_bank_question(cases):
    for c in cases:
        q = Q.get(c.question_id)
        assert len(c.files) == (2 if q.kind == "ab" else 1) and all(f.exists() for f in c.files)


# ---- the truth of the set, checked against smpl (verify the verifier) -----------------------
def test_calibration_truth_agrees_with_smpl_measurements(cases, tmp_path, monkeypatch):
    """Each synthetic truth is re-measured by the real op that crosscheck would use. A claim that
    matches the truth must therefore also be `supported` there; where the op is known weak
    (beats on 84 / 150 BPM synthetic loops: primary reads 112 / 120), the truth must at least be
    among its tempo candidates, which is what crosscheck's octave rule relies on."""
    from smpl_analysis import beats, chords, groove, loudness, qc
    from smpl_analysis import crosscheck as cc
    from smplstream import cas
    from smplstream import frames as F

    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))

    def audio(path):
        h = cas.put_audio_file(path)
        m = cas.read_meta(h)
        return F.audio_frame(h, sr=m["sr"], ch=m["ch"], dur=m["dur"], role="source")

    for c in cases:
        a = [audio(p) for p in c.files]
        i, t = a[0]["id"], c.truth
        if c.question_id == "tempo_bpm":
            ix = cc.Index([a[0], *beats.beats_audio_frame(a[0])])
            cands = ix.value(i, "beats", "rhythm.bpm_candidates")[0]
            assert any(abs(x - t["answer"]) / t["answer"] <= 0.04 for x in cands), (c.id, cands)
        elif c.question_id == "swing_feel":
            ix = cc.Index([a[0], *groove.groove_audio_frame(a[0], bpm=t["bpm"])])
            sw = ix.value(i, "groove-features", "rhythm.swing")[0]
            assert (sw >= 0.15) if t["answer"] == "swung" else (sw <= 0.10), (c.id, sw)
        elif c.question_id == "timing_drift":
            ix = cc.Index([a[0], *groove.groove_audio_frame(a[0], bpm=t["bpm"])])
            worst = max(abs(d["residual_ms"]) for d in cc.onset_deviations(ix, i))
            assert (worst >= 12.0) if t["answer"] else (worst < 12.0), (c.id, worst)
        elif c.question_id == "detune":
            ix = cc.Index([a[0], *chords.chords_audio_frame(a[0])])
            cents = 1200 * math.log2(ix.value(i, "key", "tonal.tuning_frequency")[0] / 440.0)
            assert abs(cents - t["answer"]) <= 15, (c.id, cents)
        elif c.question_id == "clipping":
            ix = cc.Index([a[0], *qc.qc_audio_frame(a[0])])
            assert ix.value(i, "qc", "qc.clipping.detected")[0] is t["answer"], c.id
        elif c.question_id == "defect_location":
            ix = cc.Index([a[0], *qc.qc_audio_frame(a[0])])
            times = [p["t"] for m in ix.markers(i, "defect") for p in m["data"]]
            if t["answer"]:
                assert any(abs(x - t["t"]) <= 0.05 for x in times), (c.id, times)
            else:
                assert times == [], (c.id, times)
        elif c.question_id == "ab_louder":
            la, lb = (loudness.loudness_frames(x)[0]["data"]["loudness.integrated_lufs"] for x in a)
            assert (la > lb) == (t["answer"] == "A") and abs(la - lb) > 3, (c.id, la, lb)
        elif c.question_id == "ab_swingier":
            sa, sb = (groove.groove_audio_frame(x, bpm=t["bpm"])[2]["data"]["rhythm.swing"] for x in a)
            assert (sa > sb) == (t["answer"] == "A") and abs(sa - sb) > 0.3, (c.id, sa, sb)
        elif c.question_id == "phase_problem":
            corr = qc.qc_audio_frame(a[0])[0]["data"]["qc.phase.correlation"]
            assert (corr < 0) == t["answer"], (c.id, corr)


# ---- the runner on stubbed answers ----------------------------------------------------------
class Oracle:
    """A stub 'model' that knows the truth: identifies each clip by its prepared bytes."""

    def __init__(self, cases, *, wrong=None, always=None):
        self.index = {}
        for c in cases:
            for slot, f in zip("AB", c.files, strict=False):
                self.index[A.prepare_audio(f)[0]] = (c, slot)
        self.prompt_to_q = {q.prompt: q.id for q in Q.bank().values()}
        self.wrong, self.always = wrong or {}, always

    def __call__(self, parts):
        qid = next(self.prompt_to_q[p.split("\n\n")[0]] for k, p in parts if k == "text" and p.split("\n\n")[0] in self.prompt_to_q)
        wavs = audio_slots(parts)
        case, _ = self.index[wavs[0]]
        truth = case.truth["answer"]
        if Q.get(qid).kind == "ab":
            if self.always:
                return {"answer": self.always}
            canon_a_first = self.index[wavs[0]][1] == "A"
            ans = truth if canon_a_first else {"A": "B", "B": "A"}[truth]   # slot labels as the model sees them
            return {"answer": ans, "timestamps": []}
        if qid in self.wrong:
            return {"answer": self.wrong[qid](truth), "timestamps": []}
        if qid in ("timing_drift", "defect_location") and truth is True:
            return {"answer": True, "timestamps": [case.truth["t"] + 0.05]}
        return {"answer": truth, "timestamps": []}


def _run(cases, **kw):
    be = ScriptedBackend(Oracle(cases, **{k: v for k, v in kw.items() if k in ("wrong", "always")}))
    lines = []
    outcomes = C.run(be, cases, log=lines.append)
    return be, outcomes, lines


def test_oracle_scores_everything_and_counts_calls(cases):
    scored = [c for c in cases if c.question_id not in ("stereo_width", "phase_problem")]
    be, outcomes, lines = _run(scored)
    singles = sum(Q.get(c.question_id).kind == "clip" for c in scored)
    abs_ = len(scored) - singles
    assert len(be.requests) == singles + 2 * abs_
    rep = C.report(outcomes, model="gemini-3.8-flash", skipped_routed=True)
    assert rep["calls"] == len(be.requests) and rep["n_cases"] == len(scored)
    for qid, row in rep["questions"].items():
        assert row["hit_rate"] == 1.0, qid
        if row["kind"] == "ab":
            assert row["order_agreement"] == 1.0
    assert rep["questions"]["timing_drift"]["timestamp_hit_rate"] == 1.0
    assert rep["questions"]["defect_location"]["timestamp_hit_rate"] == 1.0
    assert rep["total_cost_usd"] > 0 and all(r["cost_usd"] > 0 for r in rep["questions"].values())
    assert len(lines) == len(scored) and all("total $" in x for x in lines)   # a cost line per case


def test_hit_rates_from_stubbed_wrong_answers(cases):
    scored = [c for c in cases if c.question_id in ("tempo_bpm", "swing_feel", "detune")]
    flip = {"straight": "swung", "swung": "straight"}
    _be, outcomes, _ = _run(scored, wrong={
        "tempo_bpm": lambda t: t * 2,                    # octave error on every clip
        "swing_feel": lambda t: flip[t] if t == "swung" else t,   # misses only the swung clips
        "detune": lambda t: 0.0,                         # always says "in tune"
    })
    q = C.report(outcomes, model="m", skipped_routed=True)["questions"]
    assert q["tempo_bpm"]["hit_rate"] == 0.0 and q["tempo_bpm"]["octave_or_exact_rate"] == 1.0
    assert q["swing_feel"]["hit_rate"] == 0.5 and q["swing_feel"]["n"] == 6 and q["swing_feel"]["chance"] == 0.5
    assert q["detune"]["hit_rate"] == 0.5          # right on the 3 in-tune clips, wrong on the 3 detuned
    assert q["detune"]["mean_abs_err_cents"] == pytest.approx((0 + 0 + 0 + 40 + 35 + 45) / 6, abs=0.1)


def test_position_bias_shows_as_chance_hits_and_zero_agreement(cases):
    ab = [c for c in cases if c.question_id == "ab_louder"]
    _, outcomes, _ = _run(ab, always="A")            # says "A" whatever it hears
    row = C.report(outcomes, model="m", skipped_routed=True)["questions"]["ab_louder"]
    assert row["hit_rate"] == 0.0                   # disagreement is a miss, never a lucky hit
    assert row["order_agreement"] == 0.0 and row["first_slot_share"] == 1.0


def test_report_text_and_estimate(cases, tmp_path):
    scored = [c for c in cases if c.question_id not in ("stereo_width", "phase_problem")]
    _, outcomes, _ = _run(scored)
    rep = C.report(outcomes, model="gemini-3.8-flash", skipped_routed=True)
    text = C.format_report(rep)
    for qid in rep["questions"]:
        assert qid in text
    assert "routed stereo/phase questions skipped" in text
    C.write_report(rep, tmp_path / "r.json")
    assert json.loads((tmp_path / "r.json").read_text())["questions"]["tempo_bpm"]["n"] == 5

    est = C.estimate(scored, "gemini-3.8-flash")
    assert est["calls"] == len(outcomes) + sum(Q.get(c.question_id).kind == "ab" for c in scored)
    assert 0 < est["cost_usd_without_thinking"] < est["cost_usd"] < 1.0
    assert C.estimate(scored, "gemini-9-unpriced")["cost_usd"] is None


def test_calibrate_cli_never_spends_without_yes(tmp_path, monkeypatch, capsys):
    from smpl_ask import cli

    for env in ("SMPL_ASK_GEMINI_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    assert cli.main(["calibrate", "--workdir", str(tmp_path), "--only", "tempo_bpm"]) == 2
    assert "--yes" in capsys.readouterr().err
    assert cli.main(["calibrate", "--workdir", str(tmp_path), "--only", "tempo_bpm", "--yes"]) == 2   # no key
    assert "GEMINI_API_KEY" in capsys.readouterr().err
    assert cli.main(["calibrate", "--workdir", str(tmp_path), "--estimate"]) == 0
    est = json.loads(capsys.readouterr().out)
    assert est["calls"] == 59 and est["model"] == "gemini-3.8-flash"
    assert cli.main(["calibrate", "--workdir", str(tmp_path), "--only", "phase_problem"]) == 2   # routed: opt-in

