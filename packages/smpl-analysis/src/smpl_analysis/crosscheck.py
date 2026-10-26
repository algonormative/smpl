"""Claim cross-check — a deterministic referee for what an audio LLM says it heard (vault-16dd.6).

An audio-native model (Gemini through ``smpl ask``) is a *second ear*, never a measurement
source: it is weakest on music (MUSE, AudioCapBench), shows position bias on A/B judgements
(AudioJudge) and hears a mono downmix. This op takes the model's **claims** — ``text`` frames
with ``role: claim`` — and marks each one against the measurements ``smpl`` already put in the
stream (``beats`` / ``groove`` / ``loudness`` / ``qc`` / ``chords`` / ``width`` feature and
marker frames), so a claim is believed only as far as a measurement backs it.

Claim contract (written by ``smpl ask``, read here)
---------------------------------------------------
``text`` frame, ``role: claim``, ``data`` = a one-line human rendering, ``params`` carries::

    question_id   a key of QUESTIONS below
    answer        number | bool | "A"/"B"/"same" | "straight"/"swung" | None
    timestamps    [seconds, ...] in the clip's own timebase (may be empty)
    confidence    "normal" | "low"   (low = A/B orders disagreed or the reply was malformed)
    model, order, raw, ...           provenance, not read here

``of`` is the clip's audio frame id (A/B: ``of`` is A and ``lineage`` is ``[A, B]``).

Result
------
Per claim one ``feature`` frame (role ``crosscheck``) whose ``data`` is::

    {"claim_id", "question_id", "status", "route", "reason", "compared": [...], "claim": {...}}

``status`` is exactly one of ``supported`` / ``contradicted`` / ``unverifiable``;
``compared`` lists every frame value the verdict used (``key``, ``value``, ``frame``, ``audio``);
``route`` is ``"smpl"`` for stereo / fidelity questions (see below), ``"ear"`` for questions
with no registered objective measure (they belong on the listening page), else ``None``.
A second ``feature`` frame (role ``crosscheck-features``) carries the three counts.

Why stereo and fidelity are routed, not checked
-----------------------------------------------
Gemini downmixes every input to a single channel (and resamples to 16 kHz). Anti-phase material
cancels to silence in that downmix, width collapses to nothing, and anything above ~8 kHz (a
lossy cutoff) or between samples (true peak) is gone before the model hears it. So ``stereo_width``,
``phase_problem``, ``true_peak`` and ``lossy_cutoff`` are *smpl's* questions: ``crosscheck`` marks
them ``route: smpl``, ``unverifiable``, and shows the smpl value that answers the question.

Asymmetric trust
----------------
A status of ``contradicted`` needs a *reliable* measurement against the claim. Detectors that are
high-precision but not exhaustive (the ``qc`` click / gap finder) can only contradict a claim of
"no defect"; a claim of "defect at t" they cannot find is ``unverifiable``, not wrong.

Tolerances are flags on the CLI with the defaults in :data:`DEFAULTS`; there is no keep/cut policy here.
"""

from __future__ import annotations

import math
from collections.abc import Callable

OP = "crosscheck"
OP_VERSION = "crosscheck@1"

SUPPORTED = "supported"
CONTRADICTED = "contradicted"
UNVERIFIABLE = "unverifiable"

# Tolerances, all overridable (``smpl crosscheck --tol key=value`` / ``crosscheck(..., tol=...)``).
DEFAULTS: dict[str, float] = {
    "bpm_rel": 0.04,             # |claim-measured| / measured counted as agreement
    "swing_straight_max": 0.08,  # rhythm.swing at/below this is "straight"
    "swing_swung_min": 0.15,     # rhythm.swing at/above this is "swung"; between = ambiguous
    "swing_min_confidence": 0.05,
    "timing_ms": 12.0,           # an onset off the (swung) grid by >= this is "off-grid"; the
                                 # groove detector's hop floor is ~3 ms at 44.1 kHz
    "timing_window_s": 0.12,     # a claimed timestamp is matched to the nearest onset within this
    "defect_window_s": 0.10,     # same, for click / gap markers
    "cents_supported": 15.0,     # claim vs measured tuning agree within this
    "cents_contradicted": 30.0,  # ... and are contradicted beyond this; between = unverifiable
    "ab_lufs": 1.0,              # A/B loudness within this = "same"
    "ab_swing": 0.05,            # A/B swing within this = "same"
}


# ---------------------------------------------------------------------------
# Stream index: find the measurement frames a claim is judged against
# ---------------------------------------------------------------------------
class Index:
    """Feature / marker frames of a stream, looked up by (audio id, role)."""

    def __init__(self, frames: list[dict]):
        self.frames = frames
        self._feat: dict[tuple[str, str], dict] = {}
        self._mark: dict[tuple[str, str], list[dict]] = {}
        for f in frames:
            of, role = f.get("of"), f.get("role")
            if not of or not role:
                continue
            if f.get("kind") == "feature":
                self._feat[(of, role)] = f  # last wins (spec: selection)
            elif f.get("kind") == "marker":
                self._mark.setdefault((of, role), []).append(f)

    def feature(self, audio: str, role: str) -> dict | None:
        return self._feat.get((audio, role))

    def markers(self, audio: str, role: str) -> list[dict]:
        return self._mark.get((audio, role), [])

    def value(self, audio: str, role: str, key: str):
        f = self.feature(audio, role)
        if f is None:
            return None, None
        d = f.get("data")
        if isinstance(d, dict) and key in d:
            return d[key], f.get("id")
        return None, f.get("id")


class Verdict:
    """The outcome of checking one claim."""

    def __init__(self, status: str, reason: str, compared=None, route: str | None = None):
        self.status = status
        self.reason = reason
        self.compared = compared or []
        self.route = route


def _cmp(key: str, value, frame: str | None, audio: str) -> dict:
    return {"key": key, "value": value, "frame": frame, "audio": audio}


def _need(op_hint: str, audio: str) -> Verdict:
    return Verdict(UNVERIFIABLE, f"no {op_hint} measurement for audio {audio}; run `smpl {op_hint.split()[0]}` upstream")


def _num(x) -> float | None:
    if isinstance(x, bool) or x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# ---------------------------------------------------------------------------
# Verifiers: (claim params, audio ids, index, tol) -> Verdict
# ---------------------------------------------------------------------------
def _v_tempo(p, audios, ix, tol):
    a = audios[0]
    claim = _num(p.get("answer"))
    if claim is None or claim <= 0:
        return Verdict(UNVERIFIABLE, "claim carries no tempo number")
    bpm, fid = ix.value(a, "beats", "rhythm.bpm")
    bpm = _num(bpm)
    if bpm is None:
        return _need("beats", a)
    compared = [_cmp("rhythm.bpm", bpm, fid, a)]
    if abs(claim - bpm) / bpm <= tol["bpm_rel"]:
        return Verdict(SUPPORTED, f"claimed {claim:g} BPM within {tol['bpm_rel']:.0%} of measured {bpm:g}", compared)
    cands, _ = ix.value(a, "beats", "rhythm.bpm_candidates")
    compared.append(_cmp("rhythm.bpm_candidates", cands, fid, a))
    others = [c for c in (_num(c) for c in (cands or [])) if c]
    others += [bpm * 2.0, bpm / 2.0]
    if any(abs(claim - c) / c <= tol["bpm_rel"] for c in others):
        return Verdict(UNVERIFIABLE, f"claimed {claim:g} BPM is an octave / alternate candidate of measured {bpm:g}", compared)
    return Verdict(CONTRADICTED, f"claimed {claim:g} BPM vs measured {bpm:g} (tolerance {tol['bpm_rel']:.0%}, no octave match)", compared)


def _grid_trust(v: Verdict, ix: Index, audio: str, role: str) -> Verdict:
    """A contradiction needs a trustworthy grid. ``smpl groove`` fits swing and timing to a tempo;
    when that tempo was ESTIMATED (``bpm_source != "given"``) a wrong tempo makes straight material
    read swung and aligned hits read loose (measured: a 138 BPM straight loop read swing 0.27 with the
    tempo estimated, 0.014 with it given). So an estimated-tempo contradiction is downgraded to
    ``unverifiable``; a supported verdict stands, with the caveat in ``reason``."""
    f = ix.feature(audio, role)
    src = ((f or {}).get("params") or {}).get("bpm_source")
    if src == "given" or v.status == SUPPORTED:
        return v
    if v.status == CONTRADICTED:
        return Verdict(UNVERIFIABLE, v.reason + " [groove tempo was estimated, not given: re-run `smpl groove --bpm N` "
                       "(or supply an intent sidecar) before trusting a contradiction]", v.compared, v.route)
    return v


def _v_swing(p, audios, ix, tol):
    return _grid_trust(_swing(p, audios, ix, tol), ix, audios[0], "groove-features")


def _swing(p, audios, ix, tol):
    a = audios[0]
    claim = p.get("answer")
    if claim not in ("straight", "swung"):
        return Verdict(UNVERIFIABLE, "claim is not a straight/swung call")
    sw, fid = ix.value(a, "groove-features", "rhythm.swing")
    conf, _ = ix.value(a, "groove-features", "rhythm.swing_confidence")
    sw, conf = _num(sw), _num(conf)
    if sw is None:
        return _need("groove", a)
    compared = [_cmp("rhythm.swing", sw, fid, a), _cmp("rhythm.swing_confidence", conf, fid, a)]
    if conf is not None and conf < tol["swing_min_confidence"]:
        return Verdict(UNVERIFIABLE, f"swing confidence {conf:g} below {tol['swing_min_confidence']:g}", compared)
    if sw >= tol["swing_swung_min"]:
        measured = "swung"
    elif sw <= tol["swing_straight_max"]:
        measured = "straight"
    else:
        return Verdict(UNVERIFIABLE, f"measured swing {sw:g} is between straight (<={tol['swing_straight_max']:g}) and swung (>={tol['swing_swung_min']:g})", compared)
    ok = claim == measured
    return Verdict(SUPPORTED if ok else CONTRADICTED, f"claimed {claim}, measured swing {sw:g} reads {measured}", compared)


def onset_deviations(ix: Index, audio: str) -> list[dict] | None:
    """Per-onset timing (ms) against the swung grid the groove op fitted, or None if groove did not run.

    Each entry has ``deviation_ms`` (raw, from the grid) and ``residual_ms`` (against the same
    step's other occurrences; this is what a "loose hit" claim is judged on).

    The groove marker labels each onset ``step-<s>`` (1-indexed grid step within a bar); the groove
    document gives bpm / swing / grid, and ``params.origin_s`` the aligned grid origin. The bar an
    onset belongs to is the one whose expected position is nearest.
    """
    doc_frame = ix.feature(audio, "groove")
    marks = ix.markers(audio, "groove")
    if doc_frame is None or not marks:
        return None
    d = doc_frame.get("data") or {}
    origin = _num((doc_frame.get("params") or {}).get("origin_s"))
    bpm, bpb = _num(d.get("bpm")), _num(d.get("beats_per_bar"))
    grid, swing = d.get("grid_steps"), _num(d.get("swing")) or 0.0
    if None in (origin, bpm, bpb) or not grid or bpm <= 0:
        return None
    gap = bpb / grid  # beats per step
    out = []
    for pt in marks[-1].get("data") or []:
        label, t = str(pt.get("label", "")), _num(pt.get("t"))
        if t is None or not label.startswith("step-"):
            continue
        try:
            step = int(label[5:])
        except ValueError:
            continue
        expected = (step - 1) * gap + (swing * gap if step % 2 == 0 else 0.0)
        t_beats = (t - origin) * bpm / 60.0
        bar = round((t_beats - expected) / bpb)
        dev_beats = t_beats - (bar * bpb + expected)
        out.append({"t": t, "step": step, "bar": bar, "deviation_ms": dev_beats * 60.0 / bpm * 1000.0})
    # Two onsets assigned to one (bar, step) slot: a real hit plus a spurious pre-onset from the
    # detector (seen ~45 ms ahead of a hat). Keep the one nearest the grid; a single loose hit,
    # having no twin, is untouched.
    slot: dict[tuple[int, int], dict] = {}
    for o in out:
        k = (o["bar"], o["step"])
        if k not in slot or abs(o["deviation_ms"]) < abs(slot[k]["deviation_ms"]):
            slot[k] = o
    out = sorted(slot.values(), key=lambda o: o["t"])
    # Residual: each onset against the SAME step in the other bars (leave-one-out median). The
    # onset detector has a per-instrument bias (a kick reads ~10-15 ms late, a stray tail onset
    # can read 40 ms early) that is identical every bar and so cancels here, while one loose hit
    # does not. With fewer than two other bars at that step there is nothing to compare against
    # and the raw deviation stands.
    by_step: dict[int, list[int]] = {}
    for i, o in enumerate(out):
        by_step.setdefault(o["step"], []).append(i)
    for idx in by_step.values():
        for i in idx:
            others = sorted(out[j]["deviation_ms"] for j in idx if j != i)
            if len(others) >= 2:
                mid = len(others) // 2
                med = others[mid] if len(others) % 2 else 0.5 * (others[mid - 1] + others[mid])
                out[i]["residual_ms"] = out[i]["deviation_ms"] - med
            else:
                out[i]["residual_ms"] = out[i]["deviation_ms"]
    return out


def _match_windows(claimed: list[float], points: list[float], window: float):
    """For each claimed time the index of a point within ``window`` (else None), one to one:
    closest pairs first, so one measured event never confirms two claims."""
    pairs = sorted((abs(t - c), ci, pi) for ci, c in enumerate(claimed)
                   for pi, t in enumerate(points) if abs(t - c) <= window)
    out: list = [None] * len(claimed)
    used = set()
    for _, ci, pi in pairs:
        if out[ci] is None and pi not in used:
            out[ci] = pi
            used.add(pi)
    return out


def _roll_up(per: list[str], what: str) -> tuple[str, str]:
    """Combine per-timestamp statuses: all supported -> supported; none supported and some
    contradicted -> contradicted; anything else (mixed, or nothing checkable) -> unverifiable."""
    sup, con = per.count(SUPPORTED), per.count(CONTRADICTED)
    if per and sup == len(per):
        return SUPPORTED, f"every claimed {what} confirmed"
    if con and not sup:
        return CONTRADICTED, f"no claimed {what} confirmed ({con} contradicted)"
    return UNVERIFIABLE, f"{what}s only partly checkable ({sup} supported, {con} contradicted, {len(per) - sup - con} unmatched)"


def _v_timing(p, audios, ix, tol):
    return _grid_trust(_timing(p, audios, ix, tol), ix, audios[0], "groove")


def _timing(p, audios, ix, tol):
    a = audios[0]
    claim = p.get("answer")
    if not isinstance(claim, bool):
        return Verdict(UNVERIFIABLE, "claim is not a yes/no on off-grid timing")
    devs = onset_deviations(ix, a)
    if devs is None:
        return _need("groove", a)
    if not devs:
        return Verdict(UNVERIFIABLE, "no onsets measured, so nothing to check the timing claim against")
    thr = tol["timing_ms"]
    off = [d for d in devs if abs(d["residual_ms"]) >= thr]
    compared = [_cmp("groove.onset_residual_ms", [round(d["residual_ms"], 2) for d in off] or f"all < {thr:g}", None, a)]
    ts = [t for t in (_num(t) for t in (p.get("timestamps") or [])) if t is not None]
    if not claim:
        if off:
            worst = max(off, key=lambda d: abs(d["residual_ms"]))
            return Verdict(CONTRADICTED, f"claimed on-grid but {len(off)} onset(s) are >= {thr:g} ms off (worst {worst['residual_ms']:+.1f} ms at t={worst['t']:g})", compared)
        return Verdict(SUPPORTED, f"claimed on-grid; all {len(devs)} onsets within {thr:g} ms", compared)
    if not ts:
        if off:
            return Verdict(SUPPORTED, f"claimed off-grid; {len(off)} onset(s) >= {thr:g} ms off", compared)
        return Verdict(CONTRADICTED, f"claimed off-grid but all {len(devs)} onsets within {thr:g} ms", compared)
    per = []
    times = [d["t"] for d in devs]
    for c, i in zip(ts, _match_windows(ts, times, tol["timing_window_s"])):
        if i is None:
            per.append("unmatched")
            continue
        per.append(SUPPORTED if abs(devs[i]["residual_ms"]) >= thr else CONTRADICTED)
        compared.append(_cmp(f"groove.residual_ms@{devs[i]['t']:g}", round(devs[i]["residual_ms"], 2), None, a))
    status, why = _roll_up(per, "timestamp")
    return Verdict(status, why, compared)


def _v_detune(p, audios, ix, tol):
    a = audios[0]
    claim = _num(p.get("answer"))
    if claim is None:
        return Verdict(UNVERIFIABLE, "claim carries no cents number")
    hz, fid = ix.value(a, "key", "tonal.tuning_frequency")
    hz = _num(hz)
    if hz is None or hz <= 0:
        return _need("chords", a)
    cents = 1200.0 * math.log2(hz / 440.0)
    compared = [_cmp("tonal.tuning_frequency", hz, fid, a), _cmp("tuning_cents_vs_a440", round(cents, 2), fid, a)]
    diff = abs(claim - cents)
    if diff <= tol["cents_supported"]:
        return Verdict(SUPPORTED, f"claimed {claim:+g} cents within {tol['cents_supported']:g} of measured {cents:+.1f}", compared)
    if diff > tol["cents_contradicted"]:
        return Verdict(CONTRADICTED, f"claimed {claim:+g} cents vs measured {cents:+.1f} (off by {diff:.1f})", compared)
    return Verdict(UNVERIFIABLE, f"claimed {claim:+g} cents vs measured {cents:+.1f}: inside the grey zone", compared)


def _v_clipping(p, audios, ix, tol):
    a = audios[0]
    claim = p.get("answer")
    if not isinstance(claim, bool):
        return Verdict(UNVERIFIABLE, "claim is not a yes/no on clipping")
    det, fid = ix.value(a, "qc", "qc.clipping.detected")
    if det is None:
        return _need("qc", a)
    compared = [_cmp("qc.clipping.detected", det, fid, a)]
    ok = bool(det) == claim
    return Verdict(SUPPORTED if ok else CONTRADICTED, f"claimed clipping={claim}, qc detected={bool(det)}", compared)


def _v_defect(p, audios, ix, tol):
    a = audios[0]
    claim = p.get("answer")
    if not isinstance(claim, bool):
        return Verdict(UNVERIFIABLE, "claim is not a yes/no on a defect")
    qc = ix.feature(a, "qc")
    if qc is None:
        return _need("qc", a)
    pts = [pt for m in ix.markers(a, "defect") for pt in (m.get("data") or [])]
    times = [t for t in (_num(pt.get("t")) for pt in pts) if t is not None]
    compared = [_cmp("qc.defect_markers", [round(t, 3) for t in times], qc.get("id"), a)]
    ts = [t for t in (_num(t) for t in (p.get("timestamps") or [])) if t is not None]
    if not claim:
        if times:
            return Verdict(CONTRADICTED, f"claimed no defect but qc marked {len(times)} click/gap point(s)", compared)
        return Verdict(SUPPORTED, "claimed no defect; qc found none", compared)
    if not times:
        return Verdict(UNVERIFIABLE, "qc found no click/gap; its detector is high-precision, not exhaustive, so a claimed defect is not refuted", compared)
    if not ts:
        return Verdict(SUPPORTED, f"claimed a defect; qc marked {len(times)} point(s)", compared)
    per = ["unmatched" if i is None else SUPPORTED
           for i in _match_windows(ts, times, tol["defect_window_s"])]
    if SUPPORTED not in per:
        # qc found defects, just not where the model said. Its detector is not exhaustive, so the model
        # may have heard a second one: the location is unconfirmed, not refuted.
        return Verdict(UNVERIFIABLE, f"qc marked defects at {[round(t, 3) for t in times]}, none within {tol['defect_window_s']:g}s of the claimed {ts}", compared)
    status, why = _roll_up([s if s == SUPPORTED else "unmatched" for s in per], "timestamp")
    return Verdict(status, why, compared)


def _ab_pick(a_val: float, b_val: float, tol: float) -> str:
    if abs(a_val - b_val) <= tol:
        return "same"
    return "A" if a_val > b_val else "B"


def _ab(p, audios, ix, tol, role, key, tol_key, unit):
    claim = p.get("answer")
    if claim not in ("A", "B", "same"):
        return Verdict(UNVERIFIABLE, "claim is not an A/B/same call")
    if len(audios) < 2:
        return Verdict(UNVERIFIABLE, "A/B claim does not name two audio frames (lineage)")
    if p.get("confidence") == "low":
        return Verdict(UNVERIFIABLE, "the two A/B orders disagreed (position bias); the claim has no standing")
    (va, fa), (vb, fb) = (ix.value(a, role, key) for a in audios[:2])
    va, vb = _num(va), _num(vb)
    if va is None or vb is None:
        return _need(role.split("-")[0], audios[0] if va is None else audios[1])
    compared = [_cmp(key, va, fa, audios[0]), _cmp(key, vb, fb, audios[1])]
    measured = _ab_pick(va, vb, tol[tol_key])
    ok = claim == measured
    return Verdict(SUPPORTED if ok else CONTRADICTED,
                   f"claimed {claim}; measured A={va:g}{unit} B={vb:g}{unit} reads {measured}", compared)


def _v_ab_louder(p, audios, ix, tol):
    return _ab(p, audios, ix, tol, "loudness", "loudness.integrated_lufs", "ab_lufs", " LUFS")


def _v_ab_swing(p, audios, ix, tol):
    return _ab(p, audios, ix, tol, "groove-features", "rhythm.swing", "ab_swing", "")


def _v_ear(p, audios, ix, tol):
    return Verdict(UNVERIFIABLE, "no registered objective smpl measure for this question; it belongs on the listening page", route="ear")


def _routed(op: str, why: str, role: str, keys: tuple[str, ...]):
    def verify(p, audios, ix, tol):
        a = audios[0]
        compared = []
        for k in keys:
            v, fid = ix.value(a, role, k)
            if fid is not None:
                compared.append(_cmp(k, v, fid, a))
        hint = "" if compared else f" (run `smpl {op}` to get it)"
        return Verdict(UNVERIFIABLE, f"{why}; the answer is smpl's, not the model's{hint}", compared, route="smpl")
    return verify


_DOWNMIX = "the model hears a mono 16 kHz downmix"

# question_id -> verifier. The set of ids MUST equal smpl-ask's bank (tools/smpl-ask tests pin it).
QUESTIONS: dict[str, Callable] = {
    "tempo_bpm": _v_tempo,
    "swing_feel": _v_swing,
    "timing_drift": _v_timing,
    "detune": _v_detune,
    "clipping": _v_clipping,
    "defect_location": _v_defect,
    "ab_louder": _v_ab_louder,
    "ab_swingier": _v_ab_swing,
    "ab_tight_low_end": _v_ear,
    "stereo_width": _routed("width", f"stereo image: {_DOWNMIX}", "width", ("width.full.correlation", "width.full.side_mid_ratio")),
    "phase_problem": _routed("qc", f"phase / anti-phase: {_DOWNMIX} and anti-phase cancels in it", "qc", ("qc.phase.correlation",)),
    "true_peak": _routed("loudness", f"true peak: {_DOWNMIX}, with no inter-sample peaks", "loudness", ("loudness.true_peak_dbtp",)),
    "lossy_cutoff": _routed("qc", f"lossy cutoff: {_DOWNMIX} at 16 kHz, so nothing above 8 kHz survives", "qc", ("qc.lossy.spectral_cutoff_hz", "qc.lossy.confidence")),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def is_claim(frame: dict) -> bool:
    return frame.get("kind") == "text" and frame.get("role") == "claim"


def check_claim(claim: dict, ix: Index, tol: dict[str, float]) -> Verdict:
    p = claim.get("params") or {}
    qid = p.get("question_id")
    verify = QUESTIONS.get(qid)
    if verify is None:
        return Verdict(UNVERIFIABLE, f"unknown question_id {qid!r}")
    audios = list(claim.get("lineage") or [])
    if claim.get("of") and claim["of"] not in audios:
        audios.insert(0, claim["of"])
    if not audios:
        return Verdict(UNVERIFIABLE, "claim names no audio frame (`of` / `lineage`)")
    return verify(p, audios, ix, tol)


def crosscheck(frames: list[dict], *, tol: dict[str, float] | None = None,
               drop_contradicted: bool = False) -> list[dict]:
    """Passthrough ``frames`` (minus contradicted claims when asked) plus the verdict frames."""
    from smplstream import frames as F

    t = {**DEFAULTS, **(tol or {})}
    ix = Index(frames)
    claims = [f for f in frames if is_claim(f)]
    results, counts, dropped = [], {SUPPORTED: 0, CONTRADICTED: 0, UNVERIFIABLE: 0}, set()
    for c in claims:
        v = check_claim(c, ix, t)
        counts[v.status] += 1
        p = c.get("params") or {}
        drop = drop_contradicted and v.status == CONTRADICTED
        if drop:
            dropped.add(c.get("id"))
        results.append(F.feature_frame(
            {
                "claim_id": c.get("id"),
                "question_id": p.get("question_id"),
                "status": v.status,
                "route": v.route,
                "reason": v.reason,
                "compared": v.compared,
                "dropped": drop,
                # a dropped claim is not passed through, so its content is copied here (lineage never dangles)
                "claim": {k: p.get(k) for k in ("answer", "timestamps", "confidence", "model", "order")},
            },
            role="crosscheck",
            of=c.get("of") or (c.get("lineage") or [None])[0],
            op=OP, op_version=OP_VERSION,
            lineage=None if drop else ([c["id"]] if c.get("id") else None),
            params={"tolerances": t},
        ))
    feat = F.feature_frame(
        {"crosscheck.supported_count": counts[SUPPORTED],
         "crosscheck.contradicted_count": counts[CONTRADICTED],
         "crosscheck.unverifiable_count": counts[UNVERIFIABLE]},
        role="crosscheck-features", op=OP, op_version=OP_VERSION,
        params={"tolerances": t, "claims": len(claims)},
    )
    kept = [f for f in frames if f.get("id") not in dropped]
    return kept + results + ([feat] if claims else [])
