"""Tests for smpl_analysis.arc — narrative parsing, trajectories, Foote segmentation,
divergence, and the burned-in overlay render.

The audio fixture is SYNTHETIC but deliberately contrastive: four 30 s stretches whose
loudness, crest, brightness and hi/sub balance all move, so the segmentation and the
composite are meaningfully exercised (a constant tone would prove nothing).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from smplstream import cas, frames as F

from smpl_analysis import arc

NARRATIVE = str(Path(__file__).parent / "fixtures" / "arc_narrative.yaml")
SR = 16000
SECTION_S = 30.0


def synth_set(sr: int = SR, section_s: float = SECTION_S) -> np.ndarray:
    """Four contrastive stretches: sparse-quiet · pad+hats · dense-loud-bright · dark tail."""
    rng = np.random.default_rng(7)
    n = int(sr * section_s)
    t = np.arange(n) / sr

    a = 0.03 * np.sin(2 * np.pi * 110 * t)          # 1 — sparse + quiet (high crest, dark)
    blip = int(0.02 * sr)
    for k in range(6):
        i = int(k * section_s / 6 * sr)
        a[i:i + blip] += 0.5 * np.exp(-np.arange(blip) / (0.003 * sr)) * np.sin(
            2 * np.pi * 180 * np.arange(blip) / sr)

    b = 0.15 * (np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 330 * t))  # 2 — pad + hats
    hat = int(0.03 * sr)
    for k in range(int(section_s * 2)):
        i = int(k * 0.5 * sr)
        b[i:i + hat] += 0.2 * np.exp(-np.arange(hat) / (0.005 * sr)) * rng.standard_normal(hat)

    # 3 — the wall: broadband noise soft-clipped (low crest, loud, bright).
    c = 0.9 * np.tanh(3.0 * (0.4 * rng.standard_normal(n) + 0.3 * np.sin(2 * np.pi * 55 * t)))
    # 4 — the tail: a very quiet sub-ish tone fading out (low everything).
    d = 0.02 * np.sin(2 * np.pi * 80 * t) * np.exp(-t / (section_s * 0.6))
    return np.concatenate([a, b, c, d]).astype("float32")


@pytest.fixture(scope="module")
def audio() -> np.ndarray:
    return synth_set()


@pytest.fixture(scope="module")
def narrative() -> arc.Narrative:
    return arc.parse_narrative(NARRATIVE)


@pytest.fixture(scope="module")
def result(audio, narrative) -> dict:
    return arc.analyze(audio, SR, narrative)


@pytest.fixture()
def cas_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    return tmp_path


def _section_at(res: dict, t: float) -> dict:
    return next((s for s in res["sections"] if s["t0"] <= t < s["t1"]), res["sections"][-1])


def _is_png(b: bytes) -> bool:
    return b[:8] == b"\x89PNG\r\n\x1a\n"


# --- Narrative parsing --------------------------------------------------------------------


def test_parse_narrative_reads_beats_and_collapses_anchor_runs(narrative):
    assert (narrative.set_name, narrative.structure) == ("test-arc", "quiet-to-loud")
    assert [b.id for b in narrative.beats] == [
        "open", "settle", "layer", "push", "wall", "hold", "fall"]
    assert [b.tension for b in narrative.beats] == [0.10, 0.20, 0.45, 0.60, 0.95, 0.90, 0.15]
    # 7 beats, 4 anchor runs — consecutive beats sharing an anchor are ONE anchor.
    assert narrative.anchors == ("open", "build", "peak", "fall")


@pytest.mark.parametrize("doc,match", [
    ("schema: narrative/v1\nset: x\n", "no `beats`"),
    ("schema: narrative/v1\nbeats:\n  - id: x\n", "tension"),
    ("schema: patch/v1\nbeats:\n  - {id: x, tension: 0.1}\n", "unsupported schema"),
])
def test_parse_narrative_rejects_bad_documents(tmp_path, doc, match):
    p = tmp_path / "n.yaml"
    p.write_text(doc, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        arc.parse_narrative(str(p))


# --- Beat → time distribution (the documented v1 rule: equal per beat) ---------------------


def test_beat_spans_are_equal_contiguous_and_cover_the_recording(narrative):
    spans = arc.beat_spans(narrative, 120.0)
    widths = [b - a for a, b in spans]
    assert len(spans) == 7 and spans[0][0] == 0.0
    assert spans[-1][1] == pytest.approx(120.0)
    assert all(w == pytest.approx(widths[0]) for w in widths)
    assert all(spans[i][1] == pytest.approx(spans[i + 1][0]) for i in range(6))


def test_intended_curve_is_piecewise_constant_per_beat(narrative):
    mids = np.array([(i + 0.5) * (120.0 / 7) for i in range(7)])
    assert list(arc.intended_at(narrative, 120.0, mids)) == [
        0.10, 0.20, 0.45, 0.60, 0.95, 0.90, 0.15]
    assert arc.intended_at(narrative, 120.0, [0.0])[0] == 0.10       # first beat at t=0
    assert arc.intended_at(narrative, 120.0, [119.99])[0] == 0.15    # last beat at the end


# --- Trajectories + composite -------------------------------------------------------------


def test_trajectories_are_aligned_finite_and_actually_move(result):
    for key in arc.TRAJECTORIES:
        v = result["trajectories"][key]
        assert v.shape == result["t"].shape and np.isfinite(v).all()
        assert float(v.max() - v.min()) > 0.0, f"{key} is flat on a contrastive fixture"


def test_normalize_curve_spans_zero_to_one_and_handles_flat():
    n = arc.normalize_curve(np.linspace(-30, 10, 50))
    assert (n.min(), n.max()) == (pytest.approx(0.0), pytest.approx(1.0))
    assert np.allclose(arc.normalize_curve(np.full(20, -7.0)), 0.5)


def test_composite_inverts_crest():
    """A high peak-to-loudness ratio (sparse) must LOWER composite energy, not raise it."""
    flat = np.zeros(32)
    comp = arc.composite_energy({"rms_db": flat, "hi_sub_ratio_db": flat,
                                 "brightness_hz": flat, "crest_db": np.linspace(0.0, 20.0, 32)})
    assert comp[0] > comp[-1]


def test_loudness_is_not_the_whole_composite():
    """The limiter case: a FLAT loudness line still yields a moving arc off the other lines."""
    comp = arc.composite_energy({
        "rms_db": np.zeros(32),
        "crest_db": np.linspace(20.0, 4.0, 32),
        "hi_sub_ratio_db": np.linspace(-20.0, 6.0, 32),
        "brightness_hz": np.linspace(200.0, 4000.0, 32),
    })
    assert comp[-1] - comp[0] > 0.8


# --- Segmentation -------------------------------------------------------------------------


def test_segmentation_returns_about_anchor_count_and_tiles_the_recording(result, narrative):
    spans = [(s["t0"], s["t1"]) for s in result["sections"]]
    assert 1 < len(spans) <= len(narrative.anchors)
    assert spans[0][0] == 0.0
    assert spans[-1][1] == pytest.approx(result["duration_s"], abs=0.05)
    assert all(spans[i][1] == pytest.approx(spans[i + 1][0]) for i in range(len(spans) - 1))


def test_sections_override_is_honored(audio, narrative):
    assert len(arc.analyze(audio, SR, narrative, n_sections=2)["sections"]) <= 2


def test_foote_novelty_peaks_at_a_planted_change():
    """A feature matrix that switches character halfway must peak in novelty at the seam."""
    rng = np.random.default_rng(3)
    left = np.tile([[1.0], [0.0], [0.0]], (1, 60)) + 0.01 * rng.standard_normal((3, 60))
    right = np.tile([[0.0], [1.0], [0.0]], (1, 60)) + 0.01 * rng.standard_normal((3, 60))
    assert abs(int(np.argmax(arc.foote_novelty(np.hstack([left, right]), 24))) - 60) <= 4


# --- Divergence ---------------------------------------------------------------------------


def test_quiet_tail_under_high_intended_tension_diverges_negative(result):
    """The last stretch is near-silent while the narrative is still holding high — negative."""
    tail, wall = _section_at(result, 110.0), _section_at(result, 75.0)
    assert tail["measured_energy"] < wall["measured_energy"]
    assert tail["intended_tension"] > 0.3
    assert tail["divergence"] < 0.0
    assert tail["notable_difference"] is True


def test_every_section_carries_an_anchor_beats_and_stats(result):
    for sec in result["sections"]:
        assert sec["anchor"] in ("open", "build", "peak", "fall") and sec["beats"]
        for key in ("rms_db", "crest_db", "hi_sub_ratio_db", "brightness_hz",
                    "side_mid_ratio", "tempo_bpm", "low_energy_fraction"):
            assert f"arc.section.{key}" in sec


def test_difference_is_framed_as_a_difference_never_an_error(result, audio, narrative):
    labels = [s["difference"] for s in result["sections"] if s["difference"]]
    assert labels, "the contrastive fixture must produce at least one callout"
    for banned in ("error", "fail", "wrong", "miss", "bad"):
        assert banned not in " ".join(labels).lower()
    assert any("under intended" in x or "over intended" in x for x in labels)
    # …and the threshold is what decides whether anything gets called out at all.
    loose = arc.analyze(audio, SR, narrative, threshold=0.99)
    assert not any(s["notable_difference"] for s in loose["sections"])


# --- Render + frames ----------------------------------------------------------------------


def test_render_emits_a_real_png(result):
    png = arc.render(result)
    assert _is_png(png) and len(png) > 5000  # a real plot, not an empty canvas


def test_arc_audio_frame_emits_image_and_section_frames(tmp_path, cas_dir, audio):
    wav = tmp_path / "set.wav"
    sf.write(str(wav), audio, SR, subtype="FLOAT")
    h = cas.put_audio_file(str(wav))
    meta = cas.read_meta(h) or {}
    af = F.audio_frame(h, sr=meta.get("sr", SR), ch=meta.get("ch", 1),
                       dur=meta.get("dur", 0.0), role="source", op="read", op_version="read@1")

    out = arc.arc_audio_frame(af, NARRATIVE)
    roles = [f.get("role") for f in out]
    assert roles[0] == "arc:overlay" and roles[-1] == "arc:summary"
    assert roles.count("arc:section") >= 2

    img = out[0]
    assert img["kind"] == "image" and img["media"] == "image/png"
    assert img["op"] == "arc" and img["op_version"] == "arc@1"
    assert img["params"]["beat_distribution"] == "equal-per-beat"
    assert img["params"]["narrative"] == NARRATIVE
    assert _is_png(cas.get_path(img["hash"]).read_bytes())

    for f in out:
        assert F.validate_frame(f) == [] and f["of"] == af["id"]

    summary = out[-1]["data"]
    assert summary["set"] == "test-arc"
    assert summary["anchors"] == ["open", "build", "peak", "fall"]
    assert summary["differences"]
