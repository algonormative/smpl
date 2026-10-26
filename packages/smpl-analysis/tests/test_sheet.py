"""Tests for smpl_analysis.sheet — the full-length listening sheet (vault-16dd.3).

Renders on synthetic one-shot / loop / long clips, and checks the sidecar numbers against
signals with known answers (positive AND negative controls): a level step lands in the right
bar's loudness, a band level reads its calibrated dB, correlation sign, DC offset, the
intended pitch-class share, the A/B loudness match, and the frame/`view` wiring.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest
import soundfile as sf
from smpl_analysis import sheet as SH
from smpl_analysis import view as V
from smplstream import cas
from smplstream import frames as F

SR = 22050


def _sine(dur, freq=220.0, amp=0.1, sr=SR):
    t = np.arange(int(sr * dur)) / sr
    return amp * np.sin(2 * np.pi * freq * t)


def _png_wh(png: bytes) -> tuple[int, int]:
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", png[16:24])


@pytest.fixture()
def cas_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    return tmp_path


def _frame(path):
    h = cas.put_audio_file(str(path))
    meta = cas.read_meta(h) or {}
    return F.audio_frame(h, sr=meta["sr"], ch=meta["ch"], dur=meta["dur"], role="source",
                         op="read", op_version="read@1", params={"source": str(path)})


# --- renders on the three clip shapes ---------------------------------------------------------


def test_one_shot_renders_on_segments():
    y = _sine(0.3, 55.0, 0.8) * np.exp(-np.arange(int(SR * 0.3)) / (SR * 0.05))
    png, doc = SH.sheet_array(y, SR, name="kick")
    w, h = _png_wh(png)
    assert w <= SH.DEFAULT_MAX_WIDTH_PX and h > 600
    assert doc["grid"]["source"] == "none" and doc["grid"]["unit"] == "segment"
    assert len(doc["rows"]) == 8
    assert doc["rows"][0]["lufs"] > doc["rows"][-1]["lufs"]          # the decay is in the rows
    assert doc["global"]["integrated_lufs"] is None                  # < 400 ms: no gated value
    assert doc["global"]["ungated_lufs"] is not None
    assert doc["render"]["ssm_panel"] is False


def test_loop_uses_intent_bars_and_hits():
    y = _sine(8.0)
    intent = {"bpm": 120.0, "beats_per_bar": 4.0, "hits": [{"t": 0.0}, {"t": 2.0}, {"t": 2.5}]}
    png, doc = SH.sheet_array(y, SR, name="loop", intent=intent)
    _png_wh(png)
    assert doc["grid"] == {"source": "intent", "unit": "bar", "bpm": 120.0, "beats_per_bar": 4.0,
                           "n": 4}
    assert [r["bar"] for r in doc["rows"]] == ["1", "2", "3", "4"]
    assert [r["hits_intended"] for r in doc["rows"]] == [1, 2, 0, 0]


def test_bpm_flag_overrides_intent():
    _png, doc = SH.sheet_array(_sine(8.0), SR, intent={"bpm": 120.0}, bpm=60.0)
    assert doc["grid"]["source"] == "flag" and doc["grid"]["n"] == 2


def test_long_clip_gets_ssm_and_a_section_boundary_at_the_change():
    rng = np.random.default_rng(3)
    a = 0.1 * rng.standard_normal(SR * 30)                          # 30 s noise
    b = _sine(30.0, 330.0, 0.3)                                      # then 30 s tone
    png, doc = SH.sheet_array(np.concatenate([a, b]), SR, intent={"bpm": 120.0})
    w, _h = _png_wh(png)
    assert w <= SH.DEFAULT_MAX_WIDTH_PX
    assert doc["render"]["ssm_panel"] is True
    bounds = doc["global"]["section_bounds_s"]
    hit = [i for i, t in enumerate(bounds) if abs(t - 30.0) <= 1.0]
    assert hit, bounds
    assert doc["global"]["section_bounds_bar"][hit[0]] in {"15", "16"}


def test_short_clip_has_no_ssm_and_no_sections():
    _png, doc = SH.sheet_array(_sine(12.0), SR, intent={"bpm": 120.0})
    assert doc["render"]["ssm_panel"] is False
    assert doc["global"]["section_bounds_s"] == []


# --- sidecar numbers against known signals ----------------------------------------------------


def test_level_step_shows_in_the_right_bar():
    y = _sine(16.0, 440.0, 0.05)
    y[SR * 8:] *= 10 ** (12 / 20)                                    # +12 dB from bar 5 (t = 8 s)
    _png, doc = SH.sheet_array(y, SR, intent={"bpm": 120.0})
    lufs = [r["lufs"] for r in doc["rows"]]
    assert max(lufs[:4]) - min(lufs[:4]) < 0.1
    assert max(lufs[4:]) - min(lufs[4:]) < 0.1
    assert lufs[4] - lufs[3] == pytest.approx(12.0, abs=0.1)


def test_no_step_means_flat_bars():
    _png, doc = SH.sheet_array(_sine(16.0, 440.0, 0.05), SR, intent={"bpm": 120.0})
    lufs = [r["lufs"] for r in doc["rows"]]
    assert max(lufs) - min(lufs) < 0.1


def test_band_level_is_calibrated_to_a_sinusoid():
    _png, doc = SH.sheet_array(_sine(4.0, 1000.0, 0.5), SR, intent={"bpm": 120.0})
    row = doc["rows"][1]
    assert row["bands_db"]["mid"] == pytest.approx(-6.02, abs=0.3)   # 0.5 amplitude = −6 dB
    assert row["bands_db"]["sub"] is None or row["bands_db"]["sub"] < -60.0


def test_correlation_sign_and_mono():
    y = _sine(4.0, 300.0, 0.3)
    _p, same = SH.sheet_array(np.stack([y, y], 1), SR, intent={"bpm": 120.0})
    _p, flip = SH.sheet_array(np.stack([y, -y], 1), SR, intent={"bpm": 120.0})
    _p, mono = SH.sheet_array(y, SR, intent={"bpm": 120.0})
    assert same["rows"][0]["corr"] == pytest.approx(1.0, abs=1e-3)
    assert flip["rows"][0]["corr"] == pytest.approx(-1.0, abs=1e-3)
    assert flip["global"]["corr_min"] == pytest.approx(-1.0, abs=1e-3)
    assert mono["rows"][0]["corr"] is None and mono["global"]["corr_mean"] is None


def test_dc_offset_reported():
    y = _sine(2.0, 300.0, 0.3)
    _p, clean = SH.sheet_array(y, SR)
    _p, dc = SH.sheet_array(y + 0.01, SR)
    assert abs(clean["global"]["dc_offset"][0]) < 1e-4
    assert dc["global"]["dc_offset"][0] == pytest.approx(0.01, abs=1e-4)
    assert dc["global"]["dc_offset_dbfs"][0] == pytest.approx(-40.0, abs=0.2)


def test_intended_pitch_class_share():
    y = _sine(4.0, 440.0, 0.3)                                       # an A
    _p, hit = SH.sheet_array(y, SR, intent={"bpm": 120.0, "pitch_classes": ["A"]})
    _p, miss = SH.sheet_array(y, SR, intent={"bpm": 120.0, "pitch_classes": ["D#"]})
    assert hit["global"]["pc_in_set_share"] > 0.5
    assert miss["global"]["pc_in_set_share"] < 0.1
    assert hit["rows"][0]["pc_top"][0] == "A"


def test_markdown_sidecar_carries_the_rows():
    _p, doc = SH.sheet_array(_sine(8.0), SR, intent={"bpm": 120.0, "pitch_classes": ["A"]})
    md = SH.sidecar_markdown(doc)
    assert "| bar | t0 | LUFS |" in md and "in-set" in md
    assert md.count("\n| ") >= 5                                     # header + 4 bars


# --- A/B --------------------------------------------------------------------------------------


def test_ab_matches_loudness_and_reports_zero_delta_for_a_gain_copy():
    a = _sine(8.0, 220.0, 0.2) + _sine(8.0, 880.0, 0.05)
    png, doc = SH.ab_arrays(a, SR, 0.5 * a, SR, names=("a", "b"), intent={"bpm": 120.0})
    _png_wh(png)
    assert doc["ab"]["match"] == "integrated"
    assert doc["ab"]["gain_db_a"] == pytest.approx(-6.02, abs=0.05) and doc["ab"]["gain_db_b"] == 0.0
    _png2, doc2 = SH.ab_arrays(0.5 * a, SR, a, SR, names=("a", "b"), intent={"bpm": 120.0})
    assert doc2["ab"]["gain_db_a"] == 0.0 and doc2["ab"]["gain_db_b"] == pytest.approx(-6.02, abs=0.05)
    assert all(abs(r["lufs_delta"]) < 0.05 for r in doc["rows"])


def test_ab_shows_a_real_difference_in_the_right_band():
    a = _sine(8.0, 220.0, 0.2)
    b = a + _sine(8.0, 3000.0, 0.2)                                  # B adds an upper-mid tone
    _png, doc = SH.ab_arrays(a, SR, b, SR, intent={"bpm": 120.0})
    row = doc["rows"][1]
    assert row["bands_a_db"]["uppermid"] is None or row["bands_a_db"]["uppermid"] < -60.0
    assert row["bands_b_db"]["uppermid"] > -25.0                     # 0.2 amp less the match gain
    assert abs(row["bands_delta_db"]["lomid"]) < 6.0                 # only the match gain moves it


# --- frames + view ----------------------------------------------------------------------------


def test_sheet_audio_frame_emits_image_feature_text_and_reads_the_provenance(cas_env):
    wav = cas_env / "loop.wav"
    sf.write(str(wav), _sine(8.0).astype("float32"), SR, subtype="FLOAT")
    (cas_env / "loop.wav.provenance.json").write_text(
        json.dumps({"bpm": 120, "bars": 4, "pitch_classes": ["A", "E"]}))
    audio = _frame(wav)
    out = SH.sheet_audio_frame(audio)
    assert [(f["kind"], f["role"]) for f in out] == [
        ("image", "sheet"), ("feature", "sheet"), ("text", "sheet:summary")]
    img, feat, text = out
    for f in out:
        assert f["of"] == audio["id"] and f["op_version"] == SH.OP_VERSION
    assert cas.get_path(img["hash"]).read_bytes()[:4] == b"\x89PNG"
    assert feat["data"]["grid"]["source"] == "intent"
    assert feat["data"]["global"]["pc_intended"] == ["E", "A"]
    assert "| bar |" in cas.get_path(text["hash"]).read_text()
    assert json.loads(cas.get_path(img["params"]["json_hash"]).read_text())["grid"]["n"] == 4
    assert F.validate_frame(feat) == [] and F.validate_frame(text) == []


def test_view_lists_the_sheet_and_its_sidecars_not_the_rows(cas_env):
    wav = cas_env / "tone.wav"
    sf.write(str(wav), _sine(2.0).astype("float32"), SR, subtype="FLOAT")
    audio = _frame(wav)
    frames = [audio, *SH.sheet_audio_frame(audio)]
    report = V.build_report(frames)
    assert "## Listening sheets (1)" in report
    img = frames[1]
    assert str(cas.get_path(img["params"]["json_hash"])) in report
    assert str(cas.get_path(img["params"]["markdown_hash"])) in report
    assert "## Features" not in report                               # the doc is not flattened
