"""CLI tests for `smpl cohesion` / `smpl coverage` / `smpl probe` (vault-16dd.5) — subprocesses.

The streams are hand-built: stub `audio` frames (ids + `params.source`) and inline 16-d
`vector` frames shaped like `smpl embed` output, so no model and no torch ever run. Text
vectors ride in the stream the way `smpl embed text` would put them there; with none and no
`smpl-embed` on PATH, the text ops must degrade to an `unsupported` error frame.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import numpy as np
import pytest

SMPL = shutil.which("smpl")
pytestmark = pytest.mark.skipif(SMPL is None, reason="`smpl` console script not on PATH")
DIM = 16


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def _audio(name):
    return {"v": 1, "kind": "audio", "id": f"blake3:audio-{name}", "hash": f"blake3:h-{name}",
            "media": "audio/wav", "role": "source", "params": {"source": f"/packs/k/{name}.wav"}}


def _vec(of, vec, model="clap", text=None):
    params = {"model": model, "dim": DIM, "backend": model}
    if text is not None:
        params["text"] = text
    tag = text or of
    return {"v": 1, "kind": "vector", "id": f"blake3:vec-{model}-{tag}",
            "role": f"ml.{model}.{'text-embedding' if text else 'embedding'}",
            **({"of": of} if of else {}), "op": "embed", "params": params,
            "meta": {"model": model, "dim": DIM, "dtype": "float32"},
            "data": [round(float(x), 6) for x in vec]}


def _stream(named_vecs, model="clap"):
    frames = []
    for name, v in named_vecs:
        a = _audio(name)
        frames += [a, _vec(a["id"], v, model)]
    return frames


def _run(args, frames, env):
    data = "".join(json.dumps(f) + "\n" for f in frames).encode()
    p = subprocess.run([SMPL, *args], input=data, capture_output=True, env=env, timeout=120,
                       check=False)
    return p, [json.loads(line) for line in p.stdout.splitlines() if line.strip()]


def _one(out, role):
    return next(f for f in out if f.get("role") == role)


@pytest.fixture()
def env(tmp_path):
    e = dict(os.environ)
    e["SMPL_CAS_DIR"] = str(tmp_path / "cas")
    e.pop("VIRTUAL_ENV", None)
    # Keep `smpl` reachable but hide any real `smpl-embed` so nothing heavy can run.
    e["PATH"] = os.path.dirname(SMPL)
    return e


@pytest.fixture()
def pack():
    rng = np.random.default_rng(3)
    a = _unit(rng.standard_normal(DIM))
    named = [(f"kick{i}", _unit(a + 0.15 * rng.standard_normal(DIM))) for i in range(8)]
    named.append(("stranger", _unit(rng.standard_normal(DIM))))
    return named, a, rng


def test_cohesion_flags_the_stranger_and_passes_frames_through(pack, env, tmp_path):
    named, a, rng = pack
    frames = _stream(named)
    p, out = _run(["cohesion"], frames, env)
    assert p.returncode == 0, p.stderr
    assert out[: len(frames)] == frames                       # passthrough first, unchanged
    summary = _one(out, "cohesion-pack")["data"]
    assert summary["cohesion.outliers"] == ["stranger"]
    per = {f["of"]: f["data"] for f in out if f.get("role") == "cohesion"}
    assert per["blake3:audio-stranger"]["cohesion.outlier"] is True
    assert sum(d["cohesion.outlier"] for d in per.values()) == 1

    ref = tmp_path / "ref.ndjson"
    ref.write_text("".join(json.dumps(f) + "\n" for f in _stream(
        [(f"r{i}", _unit(a + 0.15 * rng.standard_normal(DIM))) for i in range(8)])))
    p, out = _run(["cohesion", "--ref", str(ref)], frames, env)
    s = _one(out, "cohesion-pack")
    assert s["data"]["cohesion.kad_ref_n"] == 8 and s["data"]["cohesion.kad"] is not None
    ids = {f["id"] for f in out}
    assert all(x in ids for x in s["lineage"])                # lineage closure (ref not in it)


def test_cohesion_needs_a_model_choice_when_two_models_are_present(pack, env):
    named, _, _ = pack
    frames = _stream(named[:3]) + [_vec(f"blake3:audio-{n}", v, "muq-mulan") for n, v in named[:3]]
    p, out = _run(["cohesion"], frames, env)
    assert p.returncode == 1 and out[-1]["kind"] == "error" and "--model" in out[-1]["data"]["message"]
    p, out = _run(["cohesion", "--model", "muq-mulan"], frames, env)
    assert p.returncode == 0
    assert _one(out, "cohesion-pack")["params"]["model"] == "muq-mulan"


def test_coverage_text_mode_with_stream_text_vectors(env, tmp_path):
    rng = np.random.default_rng(5)
    k, h, s = (_unit(rng.standard_normal(DIM)) for _ in range(3))
    named = [(f"k{i}", _unit(k + 0.2 * rng.standard_normal(DIM))) for i in range(3)]
    named += [("h0", _unit(h + 0.2 * rng.standard_normal(DIM)))]
    brief = tmp_path / "brief.json"
    brief.write_text(json.dumps({"families": {
        "kick": {"quota": 3, "text": "a kick"}, "hat": {"quota": 1, "text": "a hat"},
        "shaker": {"quota": 1, "text": "a shaker"}}}))
    frames = _stream(named) + [_vec(None, k, text="a kick"), _vec(None, h, text="a hat"),
                               _vec(None, s, text="a shaker")]
    p, out = _run(["coverage", "--brief", str(brief)], frames, env)
    assert p.returncode == 0, p.stderr
    summary = _one(out, "coverage-pack")["data"]
    assert summary["coverage.missing"] == ["shaker"]
    assert summary["coverage.families"]["kick"]["count"] == 3
    assert summary["coverage.families"]["kick"]["items"] == ["k0", "k1", "k2"]


def test_coverage_tag_mode_needs_no_text(pack, env, tmp_path):
    named, _, _ = pack
    brief = tmp_path / "brief.json"
    brief.write_text(json.dumps({"families": {"kick": {"quota": 9, "tag": "^kick"},
                                              "odd": {"quota": 1, "tag": "stranger"}}}))
    _p, out = _run(["coverage", "--brief", str(brief)], _stream(named), env)
    fam = _one(out, "coverage-pack")["data"]["coverage.families"]
    assert (fam["kick"]["count"], fam["kick"]["delta"], fam["odd"]["met"]) == (8, -1, True)


def test_probe_orders_on_the_contrast_and_degrades_without_text_vectors(env):
    rng = np.random.default_rng(9)
    c = _unit(rng.standard_normal(DIM))
    d = _unit(rng.standard_normal(DIM))
    named = [("clean", _unit(c + 0.05 * d)), ("dirty", _unit(c + 0.8 * d)), ("mid", _unit(c + 0.3 * d))]
    texts = [_vec(None, _unit(c + d), text="distorted"), _vec(None, _unit(c - d), text="clean")]
    p, out = _run(["probe", "distorted", "--neg", "clean"], _stream(named) + texts, env)
    assert p.returncode == 0, p.stderr
    ranked = _one(out, "probe-pack")["data"]["probe.ranked"]
    assert [r["label"] for r in ranked] == ["dirty", "mid", "clean"]

    p, out = _run(["probe", "distorted", "--neg", "clean"], _stream(named), env)  # no text vectors
    assert p.returncode == 0
    assert out[-1]["kind"] == "error" and out[-1]["data"]["code"] == "unsupported"
