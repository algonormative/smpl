"""Tests for groove-feel (smpl_analysis.groove_feel; vault-16dd.7).

Positive and negative controls on synthetic click patterns with known answers: one pattern
rendered straight, swung, pushed (early) and anticipated (a step early) must separate on the
numbers; the text export must round-trip the step grid of two real Assembly kit recipes (a
"drift" groove with nudges up to 0.95 of a step, and a 0.24-swing one).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
from smpl_analysis import groove_feel as gf
from smpl_analysis import intent

SR = 44100
BPM = 120.0
GRID = 16
STEP = 60.0 / BPM * 4 / GRID          # 0.125 s
BAR = STEP * GRID
FIXTURES = Path(__file__).parent / "fixtures"

# One pattern with offbeat steps in it, so swing has something to delay. 1-indexed steps.
PATTERN = [1, 3, 5, 8, 9, 11, 13, 16]


def _click(seconds: float = 0.03) -> np.ndarray:
    n = int(SR * seconds)
    t = np.linspace(0.0, seconds, n, endpoint=False)
    return (np.exp(-np.linspace(0.0, 10.0, n)) * np.sin(2 * np.pi * 1500.0 * t)).astype("float32")


def hit_times(steps, *, bars=2, swing=0.0, push_ms=0.0, shift_steps=0) -> list[tuple[int, int, float]]:
    """``(bar, step, seconds)`` of every hit: swing delays even steps, push moves all early."""
    out = []
    for b in range(1, bars + 1):
        for s in steps:
            s2 = s + shift_steps
            t = ((b - 1) * GRID + s2 - 1) * STEP
            t += swing * STEP if s2 % 2 == 0 else 0.0
            out.append((b, s2, t - push_ms / 1000.0))
    return out


def render(hits, *, bars=2, amp=None) -> np.ndarray:
    click = _click()
    y = np.zeros(round((bars * BAR + 0.05) * SR), dtype="float32")
    for _b, s, t in hits:
        a = (amp or {}).get(s, 0.5)
        i = round(t * SR)
        if i < 0:
            continue
        y[i:i + len(click)] += a * click
    return y


def intent_for(steps, *, bars=2, swing=0.0, shift_steps=0) -> dict:
    """What the generator *meant*: the straight (or declared-swing) hits, as an intent doc."""
    return {"v": 1, "bpm": BPM, "beats_per_bar": 4.0, "bars": bars, "grid_steps": GRID,
            "swing": swing,
            "hits": [{"t": t, "velocity": 1.0, "track": "click"}
                     for _b, _s, t in hit_times(steps, bars=bars, swing=swing, shift_steps=shift_steps)]}


# ---------------------------------------------------------------------------
# Metrical weights + syncopation
# ---------------------------------------------------------------------------
def test_metrical_weights_4_4_by_16():
    w = gf.metrical_weights(16, 4)
    assert w == [0, -4, -3, -4, -2, -4, -3, -4, -1, -4, -3, -4, -2, -4, -3, -4]


def test_metrical_weights_other_meters():
    w12 = gf.metrical_weights(12, 4)         # 4 beats of triplets: bar 2,2 then 3 per beat
    assert w12[0] == 0 and w12[6] == -1 and w12[3] == w12[9] == -2
    assert all(w12[i] == -3 for i in (1, 2, 4, 5, 7, 8, 10, 11))
    w20 = gf.metrical_weights(20, 5)         # 5/4 in sixteenths: 5 beats, then 2,2
    assert w20[0] == 0 and w20[4] == w20[8] == w20[12] == w20[16] == -1
    assert w20[2] == -2 and w20[1] == -3


W16 = gf.metrical_weights(16, 4)


@pytest.mark.parametrize("steps,expected", [
    ([0, 4, 8, 12], 0),            # four on the floor: nothing stronger is ever silent
    ([0], 0),                      # a lone downbeat
    ([2], 3),                      # a lone weak onset: the bar line (0) is silent after it
    ([2, 6, 10, 14], 7),           # off-beat eighths: 1 + 2 + 1 + 3
    ([0, 3, 6, 10, 12], 4),        # tresillo + 12: onset 3 -> silent 4 (2), onset 6 -> silent 8 (2)
])
def test_lhl_known_values(steps, expected):
    assert gf.lhl_bar(steps, W16) == expected


def test_max_syncopation_bounds_every_pattern():
    mx = gf.max_syncopation(W16)
    assert mx == 15
    rng = random.Random(7)
    for _ in range(300):
        steps = [s for s in range(16) if rng.random() < rng.choice((0.15, 0.4, 0.7))]
        if steps:
            assert gf.lhl_bar(steps, W16) <= mx


# ---------------------------------------------------------------------------
# Straight vs swung vs pushed vs anticipated (the acceptance numbers)
# ---------------------------------------------------------------------------
def _features(hits, swing_declared=0.0, *, intent_steps=None, search_ms=45):
    y = render(hits)
    doc = gf.analyze(y, SR, intent=intent_for(intent_steps or PATTERN, swing=swing_declared),
                     search_ms=search_ms)
    return doc


def test_synthetic_pattern_separates_on_microtiming_with_intent():
    straight = _features(hit_times(PATTERN))
    swung = _features(hit_times(PATTERN, swing=0.4), swing_declared=0.4)
    pushed = _features(hit_times(PATTERN, push_ms=40.0))
    f = {k: d["features"] for k, d in
         (("straight", straight), ("swung", swung), ("pushed", pushed))}

    assert abs(f["straight"]["rhythm.microtiming_mean_ms"]) < 1.5
    assert f["straight"]["rhythm.microtiming_abs_ms"] < 1.5
    # even-step hits are 0.4 * 125 ms = 50 ms late; the pattern has 3 of 8 on even steps
    even = [r for r in swung["per_step"] if r["step"] in (8, 16)]
    assert all(45 < r["mean_ms"] < 55 for r in even)
    odd = [r for r in swung["per_step"] if r["step"] in (1, 3, 5, 9, 11, 13)]
    assert all(abs(r["mean_ms"]) < 2 for r in odd)
    # pushed: every step is ~40 ms early, odd steps included - which is what tells it from swing
    assert all(-44 < r["mean_ms"] < -36 for r in pushed["per_step"] if r["n"])
    assert f["pushed"]["rhythm.microtiming_mean_ms"] < -35
    assert f["swung"]["rhythm.microtiming_mean_ms"] > 10
    assert f["pushed"]["rhythm.microtiming_abs_ms"] > f["swung"]["rhythm.microtiming_abs_ms"] > 10 \
        > f["straight"]["rhythm.microtiming_abs_ms"]
    # a pure timing change does not move the syncopation index: it is a function of which steps
    assert (f["straight"]["rhythm.syncopation_lhl"] == f["swung"]["rhythm.syncopation_lhl"]
            == f["pushed"]["rhythm.syncopation_lhl"])


def test_synthetic_pattern_separates_on_syncopation_when_anticipated():
    """Every hit moved one step early: a syncopated version of the same pattern."""
    straight = gf.analyze(render(hit_times([1, 5, 9, 13])), SR,
                          intent=intent_for([1, 5, 9, 13]))
    anticip = gf.analyze(render(hit_times([1, 5, 9, 13], shift_steps=-1)), SR,
                         intent=intent_for([1, 5, 9, 13], shift_steps=-1))
    assert straight["features"]["rhythm.syncopation_lhl"] == 0.0
    assert anticip["features"]["rhythm.syncopation_norm"] > 0.1
    assert (anticip["features"]["rhythm.syncopation_lhl"]
            > straight["features"]["rhythm.syncopation_lhl"] + 5)


def test_detected_path_recovers_swing_and_syncopation_without_intent():
    straight = gf.analyze(render(hit_times(PATTERN)), SR, bpm=BPM)
    swung = gf.analyze(render(hit_times(PATTERN, swing=0.4)), SR, bpm=BPM)
    assert straight["source"] == "detected"
    assert straight["features"]["rhythm.microtiming_abs_ms"] < 3
    assert straight["syncopation"]["per_bar"] == swung["syncopation"]["per_bar"]
    even = {r["step"]: r["mean_ms"] for r in swung["per_step"] if r["step"] in (8, 16)}
    assert all(40 < v < 60 for v in even.values()), even       # 50 ms
    anticip = gf.analyze(render(hit_times([1, 5, 9, 13], shift_steps=-1)), SR, bpm=BPM)
    assert anticip["features"]["rhythm.syncopation_norm"] > 0.1


def test_detected_path_cannot_see_a_uniform_push_but_the_intent_path_can():
    """Documented limit: the detected fit aligns on the odd-step hits, so a uniform shift is
    absorbed; against an intent (an absolute grid) it shows."""
    pushed = render(hit_times(PATTERN, push_ms=40.0))
    assert abs(gf.analyze(pushed, SR, bpm=BPM)["features"]["rhythm.microtiming_mean_ms"]) < 5
    with_intent = gf.analyze(pushed, SR, intent=intent_for(PATTERN), search_ms=45)
    assert with_intent["features"]["rhythm.microtiming_mean_ms"] < -35


def test_measured_delta_against_intent_is_reported_per_hit():
    """Intent says straight; the render is 20 ms late -> delta_ms = +20 on every hit."""
    y = render(hit_times(PATTERN, push_ms=-20.0))
    doc = gf.analyze(y, SR, intent=intent_for(PATTERN), search_ms=45)
    deltas = [h["delta_ms"] for h in doc["hits"] if h["ok"]]
    assert len(deltas) == len(doc["hits"])
    assert all(18 < d < 22 for d in deltas)
    assert all(h["intent_ms"] == 0.0 for h in doc["hits"])


def test_a_hit_with_a_neighbour_inside_60ms_is_not_timed():
    it = intent_for([1, 5])
    it["hits"].append({"t": it["hits"][0]["t"] + 0.03, "velocity": 1.0, "track": "click"})
    doc = gf.analyze(render([(1, 1, 0.0), (1, 5, 4 * STEP), (1, 2, 0.03)], bars=2), SR, intent=it)
    flags = {(h["bar"], h["step"]): h["ok"] for h in doc["hits"]}
    assert flags[(1, 5)] is True
    assert flags[(1, 1)] is False                      # 30 ms from the extra hit


# ---------------------------------------------------------------------------
# Level at grid positions
# ---------------------------------------------------------------------------
def test_accent_profile_reads_the_level_per_step():
    amp = {1: 0.8, 5: 0.4, 9: 0.2, 13: 0.4}                     # 0, -6, -12, -6 dB
    doc = gf.analyze(render(hit_times([1, 5, 9, 13], bars=2), amp=amp), SR,
                     intent=intent_for([1, 5, 9, 13]))
    rel = {r["step"]: r["level_rel_db"] for r in doc["per_step"] if r["n"]}
    assert rel[1] == 0.0
    assert abs(rel[5] - (-6.0)) < 1.0 and abs(rel[13] - (-6.0)) < 1.0
    assert abs(rel[9] - (-12.0)) < 1.0
    assert abs(doc["features"]["rhythm.accent_range_db"] - 12.0) < 1.0
    # velocities shown in the text follow intent by default, measured level on request
    assert doc["velocity_source"] == "intent"
    meas = gf.analyze(render(hit_times([1, 5, 9, 13], bars=2), amp=amp), SR,
                      intent=intent_for([1, 5, 9, 13]), velocity_source="measured")
    v = {h["step"]: h["vel"] for h in meas["hits"] if h["bar"] == 1}
    assert v[1] == 1.0 and abs(v[5] - 0.5) < 0.06 and abs(v[9] - 0.25) < 0.04


# ---------------------------------------------------------------------------
# Text export
# ---------------------------------------------------------------------------
def _recipe_steps(recipe: dict) -> dict:
    out: dict = {}
    for t in recipe["tracks"]:
        for h in t.get("hits", []):
            out.setdefault(t["name"], {}).setdefault(h["bar"], set()).add(h["step"])
    return out


@pytest.mark.parametrize("key", ["drift", "sw62"])
def test_text_export_round_trips_a_real_assembly_recipe(key):
    fx = json.loads((FIXTURES / "assembly_groove_recipes.json").read_text())[key]
    it = intent.normalize({"bpm": fx["bpm"], "bars": fx["bars"], "recipe": fx["recipe"]})
    doc = gf.analyze(None, intent=it)                      # export from intent alone
    parsed = gf.parse_text(doc["text"])

    want = _recipe_steps(fx["recipe"])
    grid_steps = {name: {b: set(cells) for b, cells in bars.items() if cells}
                  for name, bars in parsed["grid"].items()}
    grid_steps = {n: b for n, b in grid_steps.items() if b}
    assert grid_steps == {n.replace(" ", "_"): b for n, b in want.items() if b}
    assert gf.steps_of(parsed["hits"]) == {n: b for n, b in want.items()}

    # velocity buckets agree with the recipe's own velocities (a ratchet - several hits in one
    # cell - shows its loudest; the hit list above keeps every one)
    for t in fx["recipe"]["tracks"]:
        cells: dict = {}
        for h in t.get("hits", []):
            cells[(h["bar"], h["step"])] = max(cells.get((h["bar"], h["step"]), 0.0), h["velocity"])
        for (bar, step), vel in cells.items():
            expect = "X" if vel >= 0.8 else "x" if vel >= 0.5 else "o"
            assert parsed["grid"][t["name"]][bar][step] == expect
    assert parsed["meta"]["grid"] == "16" and parsed["meta"]["source"] == "intent"
    assert parsed["meta"]["measured"] == "no"


def test_text_export_flags_offsets_and_lists_hits():
    fx = json.loads((FIXTURES / "assembly_groove_recipes.json").read_text())["drift"]
    it = intent.normalize({"bpm": fx["bpm"], "bars": fx["bars"], "recipe": fx["recipe"]})
    doc = gf.analyze(None, intent=it, flag_ms=15.0)
    ms_lines = [ln for ln in doc["text"].splitlines() if ln.strip().startswith("ms ")]
    assert ms_lines, "a drift groove has hits 15 ms or more off the grid"
    flagged = [int(tok.split(":")[1]) for ln in ms_lines for tok in ln.split()[1:]]
    assert all(abs(v) >= 15 for v in flagged)
    n_hits = sum(len(t["hits"]) for t in fx["recipe"]["tracks"])
    parsed = gf.parse_text(doc["text"])
    assert len(parsed["hits"]) == n_hits
    biggest = max(parsed["hits"], key=lambda h: abs(h["ms"]))
    assert abs(biggest["ms"]) > 100          # the 0.23-beat nudge: ~115 ms


def test_times_only_intent_assigns_cells_on_the_swung_grid():
    """Without bar/step on the hits, a 0.24-swing offbeat must still land on its own step."""
    fx = json.loads((FIXTURES / "assembly_groove_recipes.json").read_text())["sw62"]
    it = intent.normalize({"bpm": fx["bpm"], "bars": fx["bars"], "recipe": fx["recipe"]})
    for h in it["hits"]:
        h.pop("bar"), h.pop("step")
    doc = gf.analyze(None, intent=it)
    got = gf.steps_of(doc["hits"])
    want = {n: b for n, b in _recipe_steps(fx["recipe"]).items()}
    assert got == want


def test_text_marks_unmeasurable_hits_with_a_dash():
    it = intent_for([1, 5])
    it["hits"].append({"t": it["hits"][0]["t"] + 0.03, "velocity": 1.0, "track": "click"})
    doc = gf.analyze(render([(1, 1, 0.0), (1, 5, 4 * STEP), (1, 2, 0.03)]), SR, intent=it)
    parsed = gf.parse_text(doc["text"])
    ms = {(h["bar"], h["step"]): h["ms"] for h in parsed["hits"]}
    assert ms[(1, 1)] is None and ms[(1, 5)] is not None


# ---------------------------------------------------------------------------
# Reference grooves
# ---------------------------------------------------------------------------
def test_reference_comparison_gives_per_step_deltas():
    ref_straight = gf.analyze(render(hit_times(PATTERN)), SR, intent=intent_for(PATTERN))
    ref = gf.reference_profile(ref_straight)
    swung = gf.analyze(render(hit_times(PATTERN, swing=0.4)), SR,
                       intent=intent_for(PATTERN, swing=0.4), references=[ref])
    rows = {r["step"]: r for r in swung["reference"]["per_step"]}
    assert 45 < rows[8]["delta_ms"] < 55 and 45 < rows[16]["delta_ms"] < 55
    assert abs(rows[1]["delta_ms"]) < 2
    assert swung["reference"]["steps_compared"] == len(PATTERN)
    # same groove as its own reference: nothing to report
    same = gf.analyze(render(hit_times(PATTERN)), SR, intent=intent_for(PATTERN), references=[ref])
    assert same["reference"]["mean_abs_delta_ms"] < 1.0
    assert (swung["features"]["rhythm.groove_ref_delta_ms"]
            > same["features"]["rhythm.groove_ref_delta_ms"] + 10)


def test_reference_accepts_a_groove_json_and_averages_a_set():
    groove_doc = {"bpm": 120, "grid_steps": 16, "swing": 0.2,
                  "hits": [{"step": s, "deviation_ms": 25.0 if s % 2 == 0 else 0.0}
                           for s in range(1, 17)]}
    a = gf.reference_profile(groove_doc)
    b = gf.reference_profile({**groove_doc,
                              "hits": [{"step": s, "deviation_ms": 35.0 if s % 2 == 0 else 0.0}
                                       for s in range(1, 17)]})
    doc = gf.analyze(render(hit_times(PATTERN, swing=0.4)), SR,
                     intent=intent_for(PATTERN, swing=0.4), references=[a, b])
    rows = {r["step"]: r for r in doc["reference"]["per_step"]}
    assert rows[8]["ref_ms"] == 30.0                       # the set's mean
    assert 15 < rows[8]["delta_ms"] < 25                   # 50 - 30


def test_reference_on_a_different_grid_is_rejected():
    with pytest.raises(ValueError, match="grid"):
        gf.reference_profile({"grid_steps": 12, "hits": [{"step": 1, "deviation_ms": 0.0}]},
                             grid_steps=16)
    with pytest.raises(ValueError):
        gf.reference_profile({"nothing": 1})


def test_needs_audio_or_intent():
    with pytest.raises(ValueError):
        gf.analyze(None, None)
