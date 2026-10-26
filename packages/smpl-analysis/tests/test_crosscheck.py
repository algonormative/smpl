"""Tests for claim cross-checking (smpl_analysis.crosscheck; vault-16dd.6).

Two layers. The verifier rules are exercised on hand-built measurement frames (every status,
the smpl-routed questions, missing measurements). Then the same machinery is run against the
REAL ``beats`` / ``groove`` / ``qc`` ops on synthetic loops with known answers, so the
"contradicted tempo claim" and "contradicted timing claim" are dropped by genuine measurements,
not by fixtures that agree with the verifier by construction. No network, no model.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
from smpl_analysis import crosscheck as cc
from smplstream import frames as F

SR = 44100
AUDIO = "A1"
AUDIO_B = "B1"


# ---------------------------------------------------------------------------
# Fabricated measurement frames
# ---------------------------------------------------------------------------
def _claim(qid, answer, *, audio=AUDIO, timestamps=None, confidence="normal", lineage=None):
    params = {"question_id": qid, "answer": answer, "timestamps": timestamps or [],
              "confidence": confidence, "model": "stub", "order": "single"}
    return F.text_frame(f"{qid}: {answer}", role="claim", of=audio, lineage=lineage,
                        op="ask", op_version="ask@1", params=params)


def _feat(data, role, audio=AUDIO, **kw):
    return F.feature_frame(data, role=role, of=audio, **kw)


def _status(frames, qid=None):
    out = cc.crosscheck(frames)
    res = [f["data"] for f in out if f.get("role") == "crosscheck"]
    return res[0] if qid is None else next(r for r in res if r["question_id"] == qid)


def test_tempo_supported_contradicted_octave_and_missing():
    beats = _feat({"rhythm.bpm": 128.0, "rhythm.bpm_confidence": 0.9, "rhythm.bpm_candidates": [128.0, 85.3]}, "beats")
    assert _status([beats, _claim("tempo_bpm", 127)])["status"] == "supported"
    bad = _status([beats, _claim("tempo_bpm", 90)])
    assert bad["status"] == "contradicted"
    assert bad["compared"][0] == {"key": "rhythm.bpm", "value": 128.0, "frame": beats["id"], "audio": AUDIO}
    assert _status([beats, _claim("tempo_bpm", 64)])["status"] == "unverifiable"   # half-time: octave
    assert _status([beats, _claim("tempo_bpm", 85)])["status"] == "unverifiable"   # alternate candidate
    miss = _status([_claim("tempo_bpm", 120)])
    assert miss["status"] == "unverifiable" and "smpl beats" in miss["reason"]
    assert _status([beats, _claim("tempo_bpm", None)])["status"] == "unverifiable"


def test_swing_rules_and_ambiguous_band():
    def feats(sw, conf=0.8, src="given"):
        return _feat({"rhythm.swing": sw, "rhythm.swing_confidence": conf}, "groove-features",
                     params={"bpm_source": src})
    assert _status([feats(0.5), _claim("swing_feel", "swung")])["status"] == "supported"
    assert _status([feats(0.5), _claim("swing_feel", "straight")])["status"] == "contradicted"
    assert _status([feats(0.0), _claim("swing_feel", "straight")])["status"] == "supported"
    assert _status([feats(0.11), _claim("swing_feel", "swung")])["status"] == "unverifiable"
    assert _status([feats(0.5, conf=0.0), _claim("swing_feel", "swung")])["status"] == "unverifiable"
    assert _status([feats(0.5), _claim("swing_feel", "unsure")])["status"] == "unverifiable"
    # an estimated tempo can fake swing: a contradiction is downgraded, a support stands
    est = _status([feats(0.5, src="estimated"), _claim("swing_feel", "straight")])
    assert est["status"] == "unverifiable" and "--bpm" in est["reason"]
    assert _status([feats(0.5, src="estimated"), _claim("swing_feel", "swung")])["status"] == "supported"


def test_detune_cents():
    hz = 440.0 * 2 ** (40 / 1200)
    key = _feat({"tonal.tuning_frequency": hz}, "key")
    assert _status([key, _claim("detune", 35)])["status"] == "supported"
    assert _status([key, _claim("detune", 0)])["status"] == "contradicted"
    assert _status([key, _claim("detune", 15)])["status"] == "unverifiable"  # grey zone


def test_clipping_and_defects_asymmetry():
    clip = _feat({"qc.clipping.detected": True}, "qc")
    assert _status([clip, _claim("clipping", True)])["status"] == "supported"
    assert _status([clip, _claim("clipping", False)])["status"] == "contradicted"

    qc = _feat({"qc.clipping.detected": False}, "qc")
    click = F.marker_frame([{"t": 1.30, "sample": 57330, "label": "click"}], role="defect", of=AUDIO)
    assert _status([qc, click, _claim("defect_location", True, timestamps=[1.32])])["status"] == "supported"
    # right idea, wrong place: qc is not exhaustive, so the location is unconfirmed rather than refuted
    assert _status([qc, click, _claim("defect_location", True, timestamps=[3.0])])["status"] == "unverifiable"
    assert _status([qc, click, _claim("defect_location", False)])["status"] == "contradicted"
    # qc finding nothing does NOT refute "there is a click": its detector is not exhaustive.
    nothing = _status([qc, _claim("defect_location", True, timestamps=[1.0])])
    assert nothing["status"] == "unverifiable"
    assert _status([qc, _claim("defect_location", False)])["status"] == "supported"
    assert _status([_claim("defect_location", True)])["status"] == "unverifiable"   # qc never ran


def test_ab_questions_and_order_disagreement():
    la = _feat({"loudness.integrated_lufs": -14.0}, "loudness", AUDIO)
    lb = _feat({"loudness.integrated_lufs": -20.0}, "loudness", AUDIO_B)
    ab = [la, lb]
    mk = lambda ans, **kw: _claim("ab_louder", ans, lineage=[AUDIO, AUDIO_B], **kw)
    assert _status(ab + [mk("A")])["status"] == "supported"
    assert _status(ab + [mk("B")])["status"] == "contradicted"
    assert _status(ab + [mk("same")])["status"] == "contradicted"
    low = _status(ab + [mk("A", confidence="low")])
    assert low["status"] == "unverifiable" and "disagreed" in low["reason"]
    # no objective measure registered for "tight low end": routed to the ear, not guessed
    ear = _status([_claim("ab_tight_low_end", "A", lineage=[AUDIO, AUDIO_B])])
    assert ear["status"] == "unverifiable" and ear["route"] == "ear"


def test_stereo_and_fidelity_questions_route_to_smpl():
    qc = _feat({"qc.phase.correlation": -0.98, "qc.lossy.spectral_cutoff_hz": 16000.0, "qc.lossy.confidence": 0.9}, "qc")
    width = _feat({"width.full.correlation": -0.98, "width.full.side_mid_ratio": 9.0}, "width")
    loud = _feat({"loudness.true_peak_dbtp": 0.8}, "loudness")
    for qid, ans, key in [("phase_problem", False, "qc.phase.correlation"),
                          ("stereo_width", "mono", "width.full.correlation"),
                          ("true_peak", -3.0, "loudness.true_peak_dbtp"),
                          ("lossy_cutoff", 20000, "qc.lossy.spectral_cutoff_hz")]:
        r = _status([qc, width, loud, _claim(qid, ans)])
        assert r["route"] == "smpl" and r["status"] == "unverifiable", qid
        assert "mono" in r["reason"] and any(c["key"] == key for c in r["compared"]), qid


def test_bank_and_registry_agree_on_routes():
    routed = {q for q, fn in cc.QUESTIONS.items() if fn.__name__ == "verify"}
    assert routed == {"stereo_width", "phase_problem", "true_peak", "lossy_cutoff"}


def test_drop_contradicted_keeps_lineage_closed_and_counts():
    beats = _feat({"rhythm.bpm": 128.0, "rhythm.bpm_candidates": []}, "beats")
    good, bad = _claim("tempo_bpm", 128), _claim("tempo_bpm", 90)
    out = cc.crosscheck([beats, good, bad], drop_contradicted=True)
    ids = {f["id"] for f in out}
    assert good["id"] in ids and bad["id"] not in ids
    for f in out:  # lineage closure: nothing references the dropped claim
        assert bad["id"] not in (f.get("lineage") or []) and f.get("of") != bad["id"]
    res = {f["data"]["claim_id"]: f["data"] for f in out if f.get("role") == "crosscheck"}
    assert res[bad["id"]]["dropped"] is True and res[bad["id"]]["claim"]["answer"] == 90
    summ = next(f["data"] for f in out if f.get("role") == "crosscheck-features")
    assert (summ["crosscheck.supported_count"], summ["crosscheck.contradicted_count"]) == (1, 1)
    # default keeps everything
    assert bad["id"] in {f["id"] for f in cc.crosscheck([beats, good, bad])}


def test_no_claims_is_pure_passthrough():
    a = F.audio_frame("blake3:" + "0" * 64, sr=SR, ch=1, dur=1.0)
    assert cc.crosscheck([a]) == [a]


# ---------------------------------------------------------------------------
# Real ops on synthetic loops with known answers
# ---------------------------------------------------------------------------
def _hit(freq, dur=0.05):
    n = int(SR * dur)
    t = np.arange(n) / SR
    return (np.exp(-t / 0.012) * np.sin(2 * np.pi * freq * t)).astype("float32")


def _loop(bpm=128.0, bars=4, swing=0.0, shifts=None):
    """16th-note hats with a kick on every beat. ``shifts``: {(bar, step): +seconds} late hits."""
    shifts = shifts or {}
    step = 60.0 / bpm / 4.0
    n = int((bars * 16 * step + 0.3) * SR)
    y = np.zeros(n, dtype="float32")
    for bar in range(bars):
        for s in range(1, 17):
            t = (bar * 16 + s - 1) * step + (swing * step if s % 2 == 0 else 0.0)
            t += shifts.get((bar, s), 0.0)
            h = _hit(7000.0, 0.02) * 0.4 if s % 4 else _hit(60.0, 0.12) * 0.9
            i = round(t * SR)
            y[i:i + len(h)] += h[: n - i]
    return y


def _frames_for(y, tmp_path, monkeypatch, *, ops=("beats", "groove", "qc")):
    from smpl_analysis import beats, groove, qc
    from smplstream import cas

    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    p = tmp_path / "loop.wav"
    sf.write(str(p), y, SR, subtype="PCM_16")
    h = cas.put_audio_file(p)
    meta = cas.read_meta(h) or {}
    a = F.audio_frame(h, sr=SR, ch=1, dur=meta.get("dur", len(y) / SR), role="source")
    out = [a]
    if "beats" in ops:
        out += beats.beats_audio_frame(a)
    if "groove" in ops:
        out += groove.groove_audio_frame(a, bpm=128.0)
    if "qc" in ops:
        out += qc.qc_audio_frame(a)
    return a, out


def test_real_ops_drop_a_contradicted_tempo_claim(tmp_path, monkeypatch):
    a, frames = _frames_for(_loop(128.0), tmp_path, monkeypatch)
    wrong = F.text_frame("tempo: 90", role="claim", of=a["id"], params={
        "question_id": "tempo_bpm", "answer": 90, "timestamps": [], "confidence": "normal"})
    right = F.text_frame("tempo: 128", role="claim", of=a["id"], params={
        "question_id": "tempo_bpm", "answer": 128, "timestamps": [], "confidence": "normal"})
    out = cc.crosscheck(frames + [wrong, right], drop_contradicted=True)
    by = {f["data"]["claim_id"]: f["data"] for f in out if f.get("role") == "crosscheck"}
    assert by[wrong["id"]]["status"] == "contradicted", by[wrong["id"]]
    assert by[right["id"]]["status"] == "supported", by[right["id"]]
    kept = {f["id"] for f in out}
    assert wrong["id"] not in kept and right["id"] in kept


def test_real_ops_timing_claims_against_a_shifted_hit(tmp_path, monkeypatch):
    late = 0.025  # one hat 25 ms late: bar 2, step 8
    shifts = {(1, 8): late}
    t_late = (1 * 16 + 7) * (60.0 / 128.0 / 4.0) + late
    for y, truth in ((_loop(128.0, shifts=shifts), True), (_loop(128.0), False)):
        a, frames = _frames_for(y, tmp_path, monkeypatch)
        yes = F.text_frame("late hit", role="claim", of=a["id"], params={
            "question_id": "timing_drift", "answer": True, "timestamps": [t_late], "confidence": "normal"})
        no = F.text_frame("on grid", role="claim", of=a["id"], params={
            "question_id": "timing_drift", "answer": False, "timestamps": [], "confidence": "normal"})
        out = cc.crosscheck(frames + [yes, no], drop_contradicted=True)
        by = {f["data"]["claim_id"]: f["data"] for f in out if f.get("role") == "crosscheck"}
        want_yes, want_no = (cc.SUPPORTED, cc.CONTRADICTED) if truth else (cc.CONTRADICTED, cc.SUPPORTED)
        assert by[yes["id"]]["status"] == want_yes, (truth, by[yes["id"]])
        assert by[no["id"]]["status"] == want_no, (truth, by[no["id"]])
        kept = {f["id"] for f in out}
        assert (no["id"] not in kept) == truth and (yes["id"] not in kept) == (not truth)


def test_onset_deviation_floor_on_an_aligned_loop(tmp_path, monkeypatch):
    """Positive control for the 12 ms default: an aligned loop must read well under it."""
    _, frames = _frames_for(_loop(128.0), tmp_path, monkeypatch, ops=("groove",))
    audio = frames[0]["id"]
    devs = cc.onset_deviations(cc.Index(frames), audio)
    assert devs and max(abs(d["residual_ms"]) for d in devs) < 6.0
    _, frames = _frames_for(_loop(128.0, shifts={(1, 8): 0.025}), tmp_path, monkeypatch, ops=("groove",))
    devs = cc.onset_deviations(cc.Index(frames), frames[0]["id"])
    assert max(abs(d["residual_ms"]) for d in devs) > 18.0


@pytest.mark.parametrize("swing,expect", [(0.0, "straight"), (0.5, "swung")])
def test_real_groove_swing_verdicts(tmp_path, monkeypatch, swing, expect):
    a, frames = _frames_for(_loop(128.0, swing=swing), tmp_path, monkeypatch, ops=("groove",))
    other = "swung" if expect == "straight" else "straight"
    ok = F.text_frame("x", role="claim", of=a["id"], params={"question_id": "swing_feel", "answer": expect})
    ko = F.text_frame("x", role="claim", of=a["id"], params={"question_id": "swing_feel", "answer": other})
    by = {f["data"]["claim_id"]: f["data"] for f in cc.crosscheck(frames + [ok, ko]) if f.get("role") == "crosscheck"}
    assert by[ok["id"]]["status"] == "supported", by[ok["id"]]
    assert by[ko["id"]]["status"] == "contradicted", by[ko["id"]]
