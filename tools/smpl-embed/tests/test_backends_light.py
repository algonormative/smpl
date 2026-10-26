"""smpl-embed light-tier contract: no torch here, so every heavy path must degrade cleanly."""

from __future__ import annotations

import numpy as np
import pytest
from smpl_embed import backends as B


def test_catalog_roles_and_text_towers():
    _, clap = B.resolve_model("clap")
    _, mulan = B.resolve_model("muq-mulan")
    assert B.vector_role(clap) == "ml.clap.embedding"
    assert B.vector_role(mulan, text=True) == "ml.mulan.text-embedding"
    assert clap["text"] and mulan["text"] and not B.resolve_model("mert-v1-95m")[1]["text"]
    assert "lukewys/laion_clap" in B.op_version("clap", clap)


def test_heavy_paths_degrade_to_unsupported_without_torch():
    pcm = np.zeros(4800, dtype=np.float32)
    for mid in ("clap", "muq-mulan"):
        with pytest.raises(B.UnsupportedError) as exc:
            B.embed(pcm, 48000, model_id=mid)
        assert exc.value.install_hint
        with pytest.raises(B.UnsupportedError):
            B.embed_text(["a kick"], model_id=mid)
    with pytest.raises(B.UnsupportedError, match="no text tower"):
        B.embed_text(["a kick"], model_id="mert-v1-95m")
    with pytest.raises(B.UnsupportedError, match="unknown"):
        B.resolve_model("nope")


def test_windows_cover_the_file_deterministically():
    sr = 1000
    assert len(B._windows(np.zeros(5 * sr), sr, 10.0)) == 1
    wins = B._windows(np.zeros(25 * sr), sr, 10.0)
    assert [len(w) for w in wins] == [10 * sr, 10 * sr, 5 * sr]
    assert len(B._windows(np.zeros(21 * sr), sr, 10.0)) == 2  # a 1 s tail folds away


class _T:
    """Minimal tensor stand-in: the renamer only slices rows and asks size(0)."""

    def __init__(self, n):
        self.rows = list(range(n))

    def size(self, dim):
        return len(self.rows)

    def __getitem__(self, sl):
        return self.rows[sl]


def test_laion_rename_strips_ddp_prefix_and_splits_qkv():
    sd = {
        "module.logit_scale_a": 1.0,
        "module.audio_branch.layers.0.blocks.0.attn.qkv.weight": _T(6),
        "module.audio_branch.layers.0.blocks.0.mlp.fc1.weight": 2,
        "module.audio_projection.0.weight": 3,
        "module.audio_projection.2.weight": 4,
        "module.audio_branch.spectrogram_extractor.stft.conv_real.weight": 5,
        "module.text_transform.sequential.0.weight": 6,
        "module.audio_branch.bn0.weight": 7,
    }
    out = B._rename_laion_state(sd)
    p = "audio_model.audio_encoder.layers.0.blocks.0."
    assert out["logit_scale_a"] == 1.0
    assert out[p + "attention.self.query.weight"] == [0, 1]
    assert out[p + "attention.self.value.weight"] == [4, 5]
    assert out[p + "intermediate.dense.weight"] == 2
    assert out["audio_projection.linear1.weight"] == 3 and out["audio_projection.linear2.weight"] == 4
    assert out["audio_model.audio_encoder.batch_norm.weight"] == 7
    assert not any("spectrogram" in k or "transform" in k or k.startswith("module.") for k in out)
