"""smpl-stems separation backends: known-model table, stem roles, registry, op_version.

Light-path only — nothing here imports torch or `audio_separator` for real; the one test
that exercises `separate()` injects a fake `audio_separator` module into `sys.modules`.
Run with:

    uv run pytest tools/smpl-stems/tests -q
"""

from __future__ import annotations

import sys
import types

import pytest
from smpl_stems import backends


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Point SMPL_STEMS_HOME at a tmp dir so the registry never touches the real one."""
    monkeypatch.setenv("SMPL_STEMS_HOME", str(tmp_path))
    monkeypatch.delenv("SMPL_STEMS_MODEL", raising=False)
    return tmp_path


# ---- known-model table / filename resolution -------------------------------------------

def test_default_model_is_htdemucs_6s():
    assert backends.default_model() == "htdemucs_6s"


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("htdemucs", "htdemucs.yaml"),
        ("htdemucs_6s", "htdemucs_6s.yaml"),
        ("bs-roformer", "model_bs_roformer_ep_317_sdr_12.9755.ckpt"),
    ],
)
def test_separator_filename_known_models(model, expected):
    """Demucs family loads by yaml; BS-RoFormer by the pinned UVR ckpt."""
    assert backends.separator_filename(model) == expected


def test_separator_filename_unknown_falls_back_to_yaml():
    """Custom/unknown ids keep the historical `<id>.yaml` guess."""
    assert backends.separator_filename("mdx_extra_q") == "mdx_extra_q.yaml"


@pytest.mark.parametrize(
    ("model", "stems"),
    [("htdemucs", "4"), ("htdemucs_6s", "6"), ("bs-roformer", "2")],
)
def test_stems_for_known_models(model, stems):
    assert backends.stems_for(model) == stems


def test_stems_for_unknown_keeps_suffix_guess():
    assert backends.stems_for("htdemucs_ft") == "4"
    assert backends.stems_for("someothermodel_6s") == "6"


# ---- stem roles ------------------------------------------------------------------------

def test_instrumental_role_is_mapped():
    """Without this row the 2-stem RoFormer instrumental output is silently dropped."""
    assert backends.STEM_ROLES["instrumental"] == "stem:instrumental"
    assert backends.STEM_ROLES["vocals"] == "stem:vocals"


@pytest.mark.parametrize(
    ("filename", "stem"),
    [
        # RoFormer-style 2-stem outputs
        ("track_(Vocals).wav", "vocals"),
        ("track_(Instrumental).wav", "instrumental"),
        # audio-separator also appends the model name
        ("track_(Instrumental)_model_bs_roformer_ep_317_sdr_12.9755.wav", "instrumental"),
        ("track_(Vocals)_model_bs_roformer_ep_317_sdr_12.9755.wav", "vocals"),
        # Demucs-style outputs
        ("track_(Drums)_htdemucs_6s.wav", "drums"),
        ("track_(Bass)_htdemucs_6s.wav", "bass"),
        ("track_(Other)_htdemucs.wav", "other"),
        ("track_(Guitar)_htdemucs_6s.wav", "guitar"),
        ("track_(Piano)_htdemucs_6s.wav", "piano"),
    ],
)
def test_infer_stem_name(filename, stem):
    assert backends._infer_stem_name(f"/tmp/out/{filename}") == stem


# ---- registry --------------------------------------------------------------------------

def _rows_by_id():
    return {row["id"]: row for row in backends.list_models()}


def test_list_models_shows_bs_roformer_as_known_but_uninstalled():
    rows = _rows_by_id()
    assert "bs-roformer" in rows
    assert rows["bs-roformer"]["stems"] == "2"
    assert rows["bs-roformer"]["installed"] is False
    assert rows["bs-roformer"]["default"] is False
    assert rows["bs-roformer"]["weights"] is None


def test_list_models_shows_every_known_model_with_default_flag():
    rows = _rows_by_id()
    assert {"htdemucs", "htdemucs_6s", "bs-roformer"} <= set(rows)
    assert rows["htdemucs"]["stems"] == "4"
    assert rows["htdemucs_6s"]["stems"] == "6"
    assert rows["htdemucs_6s"]["default"] is True


def test_install_model_registers_roformer_ckpt_and_stem_count(isolated_home):
    info = backends.install_model("bs-roformer")
    assert info["stems"] == "2"
    assert info["weights"].endswith("model_bs_roformer_ep_317_sdr_12.9755.ckpt")

    row = _rows_by_id()["bs-roformer"]
    assert row["installed"] is True
    assert row["stems"] == "2"

    assert backends.remove_model("bs-roformer") is True
    assert _rows_by_id()["bs-roformer"]["installed"] is False


def test_install_model_unknown_id_keeps_ckpt_fallback():
    info = backends.install_model("mdx_extra_q")
    assert info["weights"].endswith("mdx_extra_q.ckpt")
    assert info["stems"] == "4"


# ---- op_version ------------------------------------------------------------------------

def test_op_version_folds_model_and_registry_version():
    ov = backends.op_version_for("bs-roformer", "0.28.0")
    assert ov == "audio-separator@0.28.0+bs-roformer:registry:bs-roformer@unpinned"
    # distinct models ⇒ distinct op_version (no cross-model cache hits)
    assert ov != backends.op_version_for("htdemucs_6s", "0.28.0")


def test_op_version_hashes_materialized_weights(isolated_home):
    backends.install_model("bs-roformer")
    weights = isolated_home / "models" / "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"fake-checkpoint-v1")
    first = backends.op_version_for("bs-roformer", "0.28.0")
    assert "weights-blake2b:" in first

    weights.write_bytes(b"fake-checkpoint-v2")  # in-place weights swap
    assert backends.op_version_for("bs-roformer", "0.28.0") != first


# ---- separate() role filter (fake separator, no torch) ----------------------------------

class _FakeSeparator:
    """Stand-in for audio_separator.separator.Separator — records the loaded filename."""

    loaded: list[str] = []
    outputs: list[str] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def load_model(self, model_filename):
        type(self).loaded.append(model_filename)

    def separate(self, input_path):
        return list(type(self).outputs)


@pytest.fixture
def fake_separator(monkeypatch):
    _FakeSeparator.loaded = []
    _FakeSeparator.outputs = []
    pkg = types.ModuleType("audio_separator")
    mod = types.ModuleType("audio_separator.separator")
    mod.Separator = _FakeSeparator
    pkg.separator = mod
    monkeypatch.setitem(sys.modules, "audio_separator", pkg)
    monkeypatch.setitem(sys.modules, "audio_separator.separator", mod)
    return _FakeSeparator


def test_separate_keeps_instrumental_stem(fake_separator):
    """The 2-stem RoFormer run must emit BOTH vocals and instrumental (the drop regression)."""
    fake_separator.outputs = [
        "track_(Vocals)_model_bs_roformer_ep_317_sdr_12.9755.wav",
        "track_(Instrumental)_model_bs_roformer_ep_317_sdr_12.9755.wav",
    ]
    got = backends.get_backend("bs-roformer").separate("in.wav")
    assert [stem for stem, _ in got] == ["vocals", "instrumental"]
    assert fake_separator.loaded == ["model_bs_roformer_ep_317_sdr_12.9755.ckpt"]


def test_separate_loads_demucs_by_yaml(fake_separator):
    fake_separator.outputs = [
        "track_(Drums)_htdemucs_6s.wav",
        "track_(Bass)_htdemucs_6s.wav",
        "track_(Vocals)_htdemucs_6s.wav",
        "track_(Other)_htdemucs_6s.wav",
        "track_(Guitar)_htdemucs_6s.wav",
        "track_(Piano)_htdemucs_6s.wav",
    ]
    got = backends.get_backend().separate("in.wav")
    assert [stem for stem, _ in got] == [
        "drums", "bass", "vocals", "other", "guitar", "piano",
    ]
    assert fake_separator.loaded == ["htdemucs_6s.yaml"]


def test_separate_raises_unsupported_when_no_recognized_stems(fake_separator):
    fake_separator.outputs = ["track_(Mystery).wav"]
    with pytest.raises(backends.UnsupportedBackend):
        backends.get_backend("bs-roformer").separate("in.wav")


def test_separate_without_the_heavy_dep_raises_unsupported(monkeypatch):
    """Light install: no audio_separator ⇒ UnsupportedBackend carrying the install hint."""
    monkeypatch.setitem(sys.modules, "audio_separator", None)
    with pytest.raises(backends.UnsupportedBackend) as exc:
        backends.get_backend("bs-roformer").separate("in.wav")
    assert exc.value.install_hint == backends.INSTALL_HINT
