"""Known-answer calibration for `smpl ask`: synthetic clips with a correct answer per question.

The clips are rendered deterministically (seeded, numpy only) so a calibration run is reproducible
bit-for-bit and a hit rate is comparable across models and dates. :func:`build_cases` writes the
clips and returns each case with its truth; :func:`run` asks the model every case through the
same code path as `smpl ask` (A/B pairs in both orders) and :func:`score` turns the claims into a
per-question-type report: hit rate (against chance), A/B order agreement, and total cost.

What the set can and cannot tell you: it measures whether the model hears an *isolated, synthetic*
difference. A high hit rate here is necessary, not sufficient, for trusting it on a real mix; the
real ground truth is Nick's ear on the listening page (claims exported with `export-ledger`).

Routed questions (stereo width, phase) are in the set but skipped unless ``include_routed``: the
model hears a mono downmix, in which anti-phase material cancels to silence. Running them
quantifies that blindness rather than trusting the argument.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import ask as A
from . import questions as Q
from .backends import AskBackend, Usage, cost_usd

SR = 44100
SEED = 20261026
TEMPO_TOL = 0.04          # a tempo answer within 4% of the truth is a hit
TS_TOL_S = 0.25           # a timestamp within this of the truth is a hit
IN_TUNE_CENTS = 10.0      # |cents| <= this counts as "in tune" for the detune hit/miss
TIMING_SHIFT_S = 0.020    # the shifted hit is 20 ms late
EST_TEXT_TOKENS = 175     # measured with countTokens on the real request (system + question + framing)
EST_OUT_TOKENS = 150      # a bare JSON answer
EST_THINK_TOKENS = 800    # allowance for hidden thinking tokens (they bill as output)


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------
def _env(n: int, tau: float) -> np.ndarray:
    return np.exp(-np.arange(n) / (tau * SR)).astype("float32")


def _kick() -> np.ndarray:
    n = int(0.22 * SR)
    t = np.arange(n) / SR
    f = 50 + 110 * np.exp(-t / 0.03)
    return (np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(n, 0.07)).astype("float32")


def _snare(rng) -> np.ndarray:
    n = int(0.16 * SR)
    t = np.arange(n) / SR
    return (0.6 * rng.standard_normal(n) * _env(n, 0.035) + 0.5 * np.sin(2 * np.pi * 190 * t) * _env(n, 0.05)).astype("float32")


def _hat(rng, accent: float) -> np.ndarray:
    n = int(0.03 * SR)
    noise = np.diff(rng.standard_normal(n + 1)).astype("float32")  # crude high-pass
    return noise * _env(n, 0.008) * 0.25 * accent


def _place(y: np.ndarray, hit: np.ndarray, t: float) -> None:
    i = round(t * SR)
    if 0 <= i < len(y):
        seg = hit[: len(y) - i]
        y[i:i + len(seg)] += seg


def drum_loop(bpm: float, *, bars: int = 4, swing: float = 0.0, shifts: dict | None = None,
              seed: int = SEED) -> np.ndarray:
    """Kick on every beat, snare on 2 and 4, a hat on every sixteenth. Even steps are delayed by
    ``swing`` of a step; ``shifts`` maps ``(bar, step)`` -> seconds to push one hit late."""
    rng = np.random.default_rng(seed)
    shifts = shifts or {}
    step = 60.0 / bpm / 4.0
    y = np.zeros(int((bars * 16 * step + 0.4) * SR), dtype="float32")
    kick, snare = _kick(), _snare(rng)
    for bar in range(bars):
        for s in range(1, 17):
            t = (bar * 16 + s - 1) * step + (swing * step if s % 2 == 0 else 0.0) + shifts.get((bar, s), 0.0)
            _place(y, _hat(rng, 1.0 if s % 2 else 0.6), t)
            if s % 4 == 1:
                _place(y, kick, t)
            if s in (5, 13):
                _place(y, snare, t)
    return y


def chord_plucks(cents: float = 0.0, *, root_hz: float = 220.0, intervals=(0, 4, 7), dur: float = 6.0,
                 seed: int = SEED) -> np.ndarray:
    """A repeated plucked triad (4 harmonics per note) tuned ``cents`` away from equal temperament A=440."""
    n = int(dur * SR)
    y = np.zeros(n, dtype="float32")
    t = np.arange(int(0.9 * SR)) / SR
    for k in range(int(dur / 0.75)):
        for iv in intervals:
            f = root_hz * 2 ** (iv / 12.0) * 2 ** (cents / 1200.0)
            note = sum(np.sin(2 * np.pi * f * h * t) / h for h in range(1, 5)) * np.exp(-t / 0.35)
            note = note * np.minimum(1.0, t / 0.006)  # 6 ms attack ramp: a clean pluck has no click of its own
            _place(y, (0.18 * note).astype("float32"), k * 0.75)
    return y


def sub_bass(tau: float, *, bpm: float = 100.0, bars: int = 2) -> np.ndarray:
    """A 55 Hz sub on every beat whose tail length is ``tau`` seconds: short = tight, long = boomy."""
    beat = 60.0 / bpm
    y = np.zeros(int((bars * 4 * beat + 0.8) * SR), dtype="float32")
    n = int(1.0 * SR)
    note = (np.sin(2 * np.pi * 55 * np.arange(n) / SR) * _env(n, tau) * 0.8).astype("float32")
    for b in range(bars * 4):
        _place(y, note, b * beat)
    return y


def _peak_to(y: np.ndarray, peak: float) -> np.ndarray:
    return (y * (peak / max(float(np.abs(y).max()), 1e-9))).astype("float32")


def _write(path: Path, y: np.ndarray) -> None:
    import soundfile as sf

    sf.write(str(path), y, SR, subtype="PCM_16")


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------
@dataclass
class Case:
    id: str
    question_id: str
    files: list[Path]
    truth: dict = field(default_factory=dict)
    note: str = ""


def build_cases(outdir: str | Path, *, include_routed: bool = False) -> list[Case]:
    """Render the calibration clips into ``outdir`` and return the cases (truth included)."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    cases: list[Case] = []

    def add(cid, qid, arrays, truth, note=""):
        files = []
        for i, y in enumerate(arrays):
            p = out / f"{cid}{'' if len(arrays) == 1 else '-' + 'AB'[i]}.wav"
            _write(p, y)
            files.append(p)
        cases.append(Case(cid, qid, files, truth, note))

    # tempo_bpm: five tempos, straight.
    for bpm in (84, 100, 118, 128, 150):
        add(f"tempo-{bpm}", "tempo_bpm", [_peak_to(drum_loop(bpm, seed=SEED + bpm), 0.7)], {"answer": float(bpm), "bpm": float(bpm)})

    # swing_feel: matched straight / swung pairs at three tempos.
    for bpm, sw in ((110, 0.55), (124, 0.60), (138, 0.50)):
        add(f"swing-straight-{bpm}", "swing_feel", [_peak_to(drum_loop(bpm, seed=SEED + bpm), 0.7)], {"answer": "straight", "bpm": float(bpm)})
        add(f"swing-swung-{bpm}", "swing_feel", [_peak_to(drum_loop(bpm, swing=sw, seed=SEED + bpm), 0.7)], {"answer": "swung", "bpm": float(bpm)})

    # timing_drift: aligned vs one snare 20 ms late (bar 2/3/4, step 5 or 13).
    for k, (bpm, bar, s) in enumerate(((120, 1, 13), (128, 2, 5), (112, 3, 13))):
        add(f"timing-aligned-{k}", "timing_drift", [_peak_to(drum_loop(bpm, seed=SEED + k), 0.7)], {"answer": False, "t": None, "bpm": float(bpm)})
        late = drum_loop(bpm, shifts={(bar, s): TIMING_SHIFT_S}, seed=SEED + k)
        t_hit = (bar * 16 + s - 1) * (60.0 / bpm / 4.0) + TIMING_SHIFT_S
        add(f"timing-late-{k}", "timing_drift", [_peak_to(late, 0.7)], {"answer": True, "t": round(t_hit, 4), "bpm": float(bpm)},
            f"snare at bar {bar + 1} step {s} is {TIMING_SHIFT_S * 1000:.0f} ms late")

    # detune: in tune (three roots) vs +/-35-45 cents.
    for k, (root, cents) in enumerate(((220.0, 0), (293.66, 0), (329.63, 0), (220.0, 40), (293.66, -35), (329.63, 45))):
        add(f"detune-{k}-{cents:+d}c", "detune", [chord_plucks(cents, root_hz=root)], {"answer": float(cents)})

    # clipping: clean vs hard-clipped drum+sub mix.
    for k, bpm in enumerate((100, 120, 140)):
        base = drum_loop(bpm, seed=SEED + k)
        add(f"clip-clean-{k}", "clipping", [_peak_to(base, 0.5)], {"answer": False})
        add(f"clip-hot-{k}", "clipping", [np.clip(_peak_to(base, 0.5) * 6.0, -1.0, 1.0).astype("float32")], {"answer": True})

    # defect_location: clean pad vs the same pad with a click at a known time.
    for k, t_click in enumerate((1.3, 2.4, 3.1)):
        pad = chord_plucks(0.0, root_hz=(220.0, 261.63, 196.0)[k], dur=5.5, seed=SEED + k)
        add(f"defect-clean-{k}", "defect_location", [pad], {"answer": False, "t": None})
        clicked = pad.copy()
        i = int(t_click * SR)
        clicked[i:i + 6] += np.array([0.8, -0.8, 0.8, -0.7, 0.6, -0.5], dtype="float32")
        add(f"defect-click-{k}", "defect_location", [np.clip(clicked, -1, 1)], {"answer": True, "t": t_click})

    # ab_louder: the same loop 6 dB apart; the louder one alternates A / B.
    for k, bpm in enumerate((100, 118, 128, 140)):
        loop = drum_loop(bpm, seed=SEED + k)
        loud, quiet = _peak_to(loop, 0.8), _peak_to(loop, 0.4)
        a, b, truth = (loud, quiet, "A") if k % 2 == 0 else (quiet, loud, "B")
        add(f"ab-louder-{k}", "ab_louder", [a, b], {"answer": truth})

    # ab_swingier: swung vs straight, same tempo.
    for k, bpm in enumerate((104, 116, 126, 136)):
        st, sw = drum_loop(bpm, seed=SEED + k), drum_loop(bpm, swing=0.58, seed=SEED + k)
        a, b, truth = (sw, st, "A") if k % 2 == 0 else (st, sw, "B")
        add(f"ab-swingier-{k}", "ab_swingier", [_peak_to(a, 0.7), _peak_to(b, 0.7)], {"answer": truth, "bpm": float(bpm)})

    # ab_tight_low_end: a 55 Hz sub with a short vs a long tail, equal level.
    for k, bpm in enumerate((90, 100, 110, 120)):
        tight, boomy = sub_bass(0.06, bpm=bpm), sub_bass(0.35, bpm=bpm)
        a, b, truth = (tight, boomy, "A") if k % 2 == 0 else (boomy, tight, "B")
        add(f"ab-lowend-{k}", "ab_tight_low_end", [_peak_to(a, 0.7), _peak_to(b, 0.7)], {"answer": truth})

    if include_routed:
        pad = chord_plucks(0.0)
        for k, (w, truth) in enumerate((("mono", "mono"), ("wide", "wide"))):
            if w == "mono":
                st = np.stack([pad, pad], axis=1)
            else:  # decorrelated: opposite micro-detune + 18 ms offset per channel
                st = np.stack([chord_plucks(+6, seed=SEED), np.roll(chord_plucks(-6, seed=SEED), int(0.018 * SR))], axis=1)
            add(f"routed-width-{w}", "stereo_width", [st], {"answer": truth}, "routed to smpl")
        pad = chord_plucks(0.0, root_hz=261.63)
        for name, truth in (("inphase", False), ("antiphase", True)):
            st = np.stack([pad, pad if truth is False else -pad], axis=1)
            add(f"routed-phase-{name}", "phase_problem", [st], {"answer": truth}, "routed to smpl")
    return cases


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
@dataclass
class Outcome:
    case: Case
    claim: A.Claim
    hit: bool
    extras: dict = field(default_factory=dict)


def _near(ts, truth_t) -> bool | None:
    if truth_t is None:
        return None
    return any(abs(t - truth_t) <= TS_TOL_S for t in ts)


def score_case(case: Case, claim: A.Claim) -> Outcome:
    """Compare one claim to the truth. A low-confidence or empty answer is a miss, never a skip."""
    q, ans, truth = case.question_id, claim.answer, case.truth["answer"]
    extras: dict = {}
    if q == "tempo_bpm":
        hit = isinstance(ans, (int, float)) and abs(ans - truth) / truth <= TEMPO_TOL
        if isinstance(ans, (int, float)):
            extras["octave_hit"] = any(abs(ans - truth * m) / (truth * m) <= TEMPO_TOL for m in (0.5, 2.0))
            extras["rel_err"] = round(abs(ans - truth) / truth, 4)
    elif q == "detune":
        hit = isinstance(ans, (int, float)) and (abs(ans) <= IN_TUNE_CENTS) == (abs(truth) <= IN_TUNE_CENTS)
        if isinstance(ans, (int, float)):
            extras["abs_err_cents"] = round(abs(ans - truth), 2)
    elif q == "phase_problem":
        hit = ans is truth
    else:
        hit = ans == truth and claim.confidence != "low"
    if q in ("timing_drift", "defect_location") and truth is True:
        extras["timestamp_hit"] = bool(ans is True and _near(claim.timestamps, case.truth.get("t")))
    if claim.order == "AB+BA":
        extras["order_agree"] = claim.confidence != "low"
        raws = [claim.raw.get("AB"), claim.raw.get("BA")]
        picks = []
        for raw in raws:
            if raw is None:
                continue
            p = Q.parse_reply(Q.get(q), raw)
            if p.ok and p.answer in ("A", "B"):
                picks.append(p.answer)
        extras["first_slot_picks"] = sum(1 for p in picks if p == "A")  # raw replies are in slot labels: A = first-presented clip
        extras["slot_picks_n"] = len(picks)
    return Outcome(case, claim, bool(hit), extras)


CHANCE = {"bool": 0.5, "ab": 0.5, "swing_feel": 0.5, "stereo_width": 1 / 3}


def _chance(qid: str) -> float | None:
    q = Q.get(qid)
    if qid in CHANCE:
        return CHANCE[qid]
    if q.kind == "ab":
        return CHANCE["ab"]
    if q.answer_type == "bool":
        return CHANCE["bool"]
    if qid == "detune":
        return 0.5
    return None  # a free number has no useful chance level


def run(backend: AskBackend, cases: list[Case], *, max_seconds: float = A.DEFAULT_MAX_SECONDS,
        log=None) -> list[Outcome]:
    """Ask every case through the same path as `smpl ask`. ``log(line)`` gets the per-call cost line."""
    outcomes = []
    for case in cases:
        q = Q.get(case.question_id)
        wavs = [A.prepare_audio(p, max_seconds=max_seconds)[0] for p in case.files]
        claim = A.ask(backend, q, wavs)
        if log:
            log(f"{case.id}: " + A.log_line(claim))
        outcomes.append(score_case(case, claim))
    return outcomes


def report(outcomes: list[Outcome], *, model: str, skipped_routed: bool, extra: dict | None = None) -> dict:
    by_q: dict[str, list[Outcome]] = {}
    for o in outcomes:
        by_q.setdefault(o.case.question_id, []).append(o)
    questions = {}
    for qid, outs in by_q.items():
        n, hits = len(outs), sum(o.hit for o in outs)
        costs = [o.claim.cost_usd for o in outs]
        row = {
            "kind": Q.get(qid).kind, "route": Q.get(qid).route, "n": n, "hits": hits,
            "hit_rate": round(hits / n, 3), "chance": _chance(qid),
            "cost_usd": None if any(c is None for c in costs) else round(sum(costs), 5),
            "calls": sum(len(o.claim.calls) for o in outs),
            "mean_latency_s": round(sum(o.claim.latency_s for o in outs) / n, 2),
        }
        if Q.get(qid).kind == "ab":
            row["order_agreement"] = round(sum(o.extras.get("order_agree", False) for o in outs) / n, 3)
            picks, tot = (sum(o.extras.get("first_slot_picks", 0) for o in outs),
                          sum(o.extras.get("slot_picks_n", 0) for o in outs))
            row["first_slot_share"] = None if not tot else round(picks / tot, 3)
        if any("timestamp_hit" in o.extras for o in outs):
            ts = [o.extras["timestamp_hit"] for o in outs if "timestamp_hit" in o.extras]
            row["timestamp_hit_rate"] = round(sum(ts) / len(ts), 3)
        if qid == "tempo_bpm":
            row["octave_or_exact_rate"] = round(sum(o.hit or o.extras.get("octave_hit", False) for o in outs) / n, 3)
        if qid == "detune":
            errs = [o.extras["abs_err_cents"] for o in outs if "abs_err_cents" in o.extras]
            row["mean_abs_err_cents"] = round(sum(errs) / len(errs), 1) if errs else None
        questions[qid] = row
    costs = [o.claim.cost_usd for o in outcomes]
    return {
        "model": model, "date": _dt.datetime.now(_dt.timezone.utc).date().isoformat(),
        "n_cases": len(outcomes), "calls": sum(len(o.claim.calls) for o in outcomes),
        "total_cost_usd": None if any(c is None for c in costs) else round(sum(costs), 4),
        "routed_questions_skipped": skipped_routed, "questions": questions, **(extra or {}),
        "cases": [{"id": o.case.id, "question_id": o.case.question_id, "truth": o.case.truth,
                   "answer": o.claim.answer, "confidence": o.claim.confidence, "hit": o.hit,
                   "timestamps": o.claim.timestamps, **o.extras} for o in outcomes],
    }


def format_report(rep: dict) -> str:
    lines = [f"smpl-ask calibration  model={rep['model']}  date={rep['date']}  "
             f"cases={rep['n_cases']}  calls={rep['calls']}  cost=" +
             ("unknown" if rep["total_cost_usd"] is None else f"${rep['total_cost_usd']:.4f}"),
             "", f"{'question':<18}{'n':>3} {'hit':>5} {'chance':>7} {'order-agree':>12} {'cost':>9}  notes"]
    for qid, r in rep["questions"].items():
        notes = []
        if "timestamp_hit_rate" in r:
            notes.append(f"timestamps {r['timestamp_hit_rate']:.0%}")
        if "octave_or_exact_rate" in r:
            notes.append(f"octave-or-exact {r['octave_or_exact_rate']:.0%}")
        if r.get("mean_abs_err_cents") is not None:
            notes.append(f"mean err {r['mean_abs_err_cents']} c")
        if r.get("first_slot_share") is not None:
            notes.append(f"first-slot share {r['first_slot_share']:.0%}")
        if r["route"]:
            notes.append(f"route: {r['route']}")
        chance = "-" if r["chance"] is None else f"{r['chance']:.0%}"
        agree = "-" if "order_agreement" not in r else f"{r['order_agreement']:.0%}"
        cost = "?" if r["cost_usd"] is None else f"${r['cost_usd']:.4f}"
        lines.append(f"{qid:<18}{r['n']:>3} {r['hit_rate']:>5.0%} {chance:>7} {agree:>12} {cost:>9}  " + "; ".join(notes))
    if rep.get("routed_questions_skipped"):
        lines.append("\n(routed stereo/phase questions skipped: smpl answers them; use --include-routed to measure the downmix blindness)")
    return "\n".join(lines)


def estimate(cases: list[Case], model: str, *, think_tokens: int = EST_THINK_TOKENS,
             max_seconds: float = A.DEFAULT_MAX_SECONDS) -> dict:
    """Token and dollar estimate for a full run with no network: audio at 32 tokens/s, A/B doubled."""
    import soundfile as sf

    from .backends import AUDIO_TOKENS_PER_SECOND

    total = Usage()
    calls = 0
    for case in cases:
        secs = sum(min(sf.info(str(p)).duration, max_seconds) for p in case.files)
        n_calls = 2 if Q.get(case.question_id).kind == "ab" else 1
        for _ in range(n_calls):
            calls += 1
            total.audio_tokens += int(secs * AUDIO_TOKENS_PER_SECOND)
            total.text_tokens += EST_TEXT_TOKENS
            total.output_tokens += EST_OUT_TOKENS + think_tokens
    c = cost_usd(model, total)
    return {"model": model, "calls": calls, "audio_tokens": total.audio_tokens,
            "text_tokens": total.text_tokens, "output_tokens_assumed": total.output_tokens,
            "think_tokens_per_call_assumed": think_tokens,
            "cost_usd": None if c is None else round(c, 4),
            "cost_usd_without_thinking": None if c is None else round(cost_usd(model, Usage(
                total.text_tokens, total.audio_tokens, calls * EST_OUT_TOKENS)), 4)}


def write_report(rep: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
