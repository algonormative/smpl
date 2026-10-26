import json

import pytest
from smpl_analysis import intent

RECIPE = {"bpm": 120, "beats_per_bar": 4, "grid_steps": 16, "swing": 0.5,
          "tracks": [{"name": "kick", "hits": [{"bar": 1, "step": 1}, {"bar": 1, "step": 2, "velocity": 0.5},
                                               {"bar": 2, "step": 1, "nudge": 0.1}]}]}


def test_recipe_hits_place_swing_and_nudge_like_pattern():
    hits = intent.recipe_hits(RECIPE)
    # step 2 is an even step: one 16th (0.125 s) plus half a 16th of swing
    assert [h["t"] for h in hits] == [0.0, 0.1875, 2.05]
    assert hits[1]["velocity"] == 0.5 and hits[0]["track"] == "kick"


def test_provenance_normalizes_and_derives_duration(tmp_path):
    wav = tmp_path / "loop.wav"
    wav.write_bytes(b"")
    (tmp_path / "loop.wav.provenance.json").write_text(json.dumps(
        {"bpm": 120, "bars": 4, "recipe": RECIPE, "pitch_classes": ["Bb", "C", "F♯"], "note": "x"}))
    doc = intent.for_audio_frame({"params": {"source": str(wav)}})
    assert doc["dur_s"] == 8.0 and doc["bars"] == 4 and doc["grid_steps"] == 16
    assert doc["pitch_classes"] == ["C", "F#", "A#"]
    assert len(doc["hits"]) == 3


def test_intent_json_wins_over_provenance_and_null_bpm_falls_back(tmp_path):
    wav = tmp_path / "a.wav"
    (tmp_path / "a.wav.provenance.json").write_text(json.dumps({"bpm": 90}))
    (tmp_path / "a.wav.intent.json").write_text(json.dumps(
        {"bpm": None, "recipe": {"bpm": 100, "grid_steps": 16, "tracks": []}, "lufs": -14,
         "hits": [{"t": 0.5}, {"t": 0.25}]}))
    doc = intent.for_audio_frame({"params": {"source": str(wav)}})
    assert doc["bpm"] == 100.0 and doc["lufs"] == -14.0
    assert [h["t"] for h in doc["hits"]] == [0.25, 0.5]


def test_no_sidecar_means_no_intent_and_bad_pitch_raises(tmp_path):
    assert intent.for_audio_frame({"params": {"source": str(tmp_path / "none.wav")}}) is None
    assert intent.for_audio_frame({"kind": "audio"}) is None
    with pytest.raises(ValueError):
        intent.pitch_class("H")


def test_meter_is_never_assumed():
    doc = intent.normalize({"bpm": 120, "bars": 4})
    assert "beats_per_bar" not in doc and "dur_s" not in doc
    assert intent.normalize({"bpm": 120, "bars": 4, "beats_per_bar": 5})["dur_s"] == 10.0
