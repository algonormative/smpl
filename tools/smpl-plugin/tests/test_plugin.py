"""`smpl plugin` — built-in effects only; nothing here needs a third-party plugin installed.
Run from the tool's own project: `cd tools/smpl-plugin && uv run pytest`. The `store` fixture
redirects the CAS *and* the memo index at a tmp dir, so `~/.smpl/cas` is never touched."""

from __future__ import annotations

import io
import json
import sys

import numpy as np
import pytest
import soundfile as sf
from smpl_plugin import cli, descriptor as D, hostbuiltin as host

EFFECT = "Compressor"
PARAMS = {"threshold_db": -20.0, "ratio": 4.0}
H1 = "blake3:" + "aa" * 32
NOTE = {"kind": "text", "role": "note", "data": "keep me", "id": "t1"}


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("SMPL_CAS_DIR", str(tmp_path / "cas"))
    monkeypatch.setenv("SMPL_MEMO_DIR", str(tmp_path / "cas" / ".memo"))
    return tmp_path


@pytest.fixture()
def dry_frame(store):
    """Ingest a synthetic dry signal (220 Hz sine + a click) → an `audio` frame factory."""
    from smplstream import cas, frames as F

    def make(ch: int = 1, role: str = "source", sr: int = 44100):
        x = (0.6 * np.sin(2 * np.pi * 220.0 * np.arange(int(sr * 0.5)) / sr)).astype("float32")
        x[int(sr * 0.1)] = 0.99
        buf = io.BytesIO()
        sf.write(buf, np.ascontiguousarray(np.stack(
            [x * (1 - 0.3 * c) for c in range(ch)], axis=1)), sr, format="WAV", subtype="FLOAT")
        m = cas.read_meta(h := cas.put_audio_bytes(buf.getvalue()))
        return F.audio_frame(h, sr=m["sr"], ch=m["ch"], dur=m["dur"], role=role)

    return make


def _run(monkeypatch, capsys, argv, frames=()) -> tuple[int, list[dict]]:
    """Drive `cli.main` over an NDJSON stdin → (exit code, emitted frames)."""
    payload = b"".join(json.dumps(f).encode() + b"\n" for f in frames)
    monkeypatch.setattr(sys, "stdin", type(
        "S", (), {"isatty": lambda self: False, "buffer": io.BytesIO(payload)})())
    return cli.main(argv), [json.loads(ln) for ln in capsys.readouterr().out.splitlines()]


def test_descriptor_validates_and_state_round_trips(store):
    names = host.list_effects()
    # containers, externals and the abstract base are not runnable built-in effects
    assert EFFECT in names and "Reverb" in names and not (set(names) & {
        "Pedalboard", "Chain", "Mix", "VST3Plugin", "IIRFilter", "Plugin", "Convolution"})
    for bad in [("NoSuchEffect", {}), (EFFECT, {"nope": 1}), (EFFECT, {"ratio": 9999})]:
        with pytest.raises(host.EffectError):
            host.coerce_params(*bad)
    desc = D.describe(EFFECT, PARAMS)
    assert D.validate_descriptor(desc) == []
    assert desc["identity"] == {"name": EFFECT, "vendor": "pedalboard", "kind": "effect"}
    assert desc["format"] == "builtin" and desc["cacheable"] is True
    assert desc["version"]["library_version"] == host.pedalboard_version()
    assert desc["bus_layout"]["input_channels"] == "any"
    p = {s["name"]: s for s in desc["parameters"]}
    assert (p["threshold_db"]["value"], p["threshold_db"]["units"]) == (-20.0, "dB")
    assert (p["ratio"]["min"], p["ratio"]["max"]) == (1.0, 50.0)
    assert p["attack_ms"]["value"] == p["attack_ms"]["default"]  # untouched default
    mode = {s["name"]: s for s in D.describe("LadderFilter")["parameters"]}["mode"]
    assert mode["type"] == "enum" and "LPF12" in mode["choices"] and mode["min"] is None
    # the state blob (`--state blake3:…`) round-trips to the same params and the same hash
    effective, sh = host.coerce_params(EFFECT, PARAMS), desc["state_hash"]
    assert D.load_state(sh) == effective
    assert D.state_hash(host.coerce_params(EFFECT, D.load_state(sh))) == sh
    # an explicit --param overrides a loaded state
    assert host.coerce_params(EFFECT, {**effective, "ratio": "8"})["ratio"] == 8.0


@pytest.mark.parametrize(("mutate", "needle"), [
    (lambda d: d.pop("parameters"), "missing required field 'parameters'"),
    (lambda d: d.update(cacheable="yes"), "cacheable: expected type"),
    (lambda d: d["parameters"][0].update(min=10.0, max=-10.0), "min 10.0 > max -10.0"),
    (lambda d: d["identity"].update(kind="synth"), "not one of"),
])
def test_mutated_descriptor_is_rejected(store, mutate, needle):
    desc = D.describe(EFFECT, PARAMS)
    mutate(desc)
    errors = D.validate_descriptor(desc)
    assert errors and any(needle in e for e in errors), errors


def test_cli_describe_json_validates(store, capsys):
    assert cli.main(["describe", EFFECT, "--json", "--param", "ratio=4"]) == 0
    desc = json.loads(capsys.readouterr().out)
    assert D.validate_descriptor(desc) == []
    assert {p["name"]: p["value"] for p in desc["parameters"]}["ratio"] == 4.0
    assert cli.main(["describe", "--schema"]) == 0
    assert json.loads(capsys.readouterr().out)["title"] == "smpl plugin descriptor"


def test_memo_key_derivation(store):
    base = host.coerce_params(EFFECT, PARAMS)
    key = cli.memo_key(EFFECT, H1, base)
    assert key == cli.memo_key(EFFECT, H1, dict(reversed(list(base.items()))))  # order-free
    assert key != cli.memo_key(EFFECT, H1, {**base, "ratio": 8.0})              # param value
    assert key != cli.memo_key(EFFECT, "blake3:" + "bb" * 32, base)            # input hash
    assert key != cli.memo_key("Limiter", H1, host.coerce_params("Limiter", {}))  # op_version
    assert host.pedalboard_version() in host.op_version(EFFECT)  # a lib bump invalidates
    # omitted params must hash as their declared defaults, not as absent keys
    assert key == cli.memo_key(EFFECT, H1, host.coerce_params(EFFECT, dict(
        PARAMS, attack_ms=1.0, release_ms=100.0)))


def test_double_render_is_byte_identical_and_hits_memo(store, dry_frame, monkeypatch):
    """The acceptance: same dry audio + same effect + same params, twice — plus lineage."""
    from smplstream import cas

    src = dry_frame(ch=2, role="drums")
    first = cli.apply_effect(src, EFFECT, PARAMS)
    assert first["params"]["memo"] == "miss"
    first_bytes = cas.get_path(first["hash"]).read_bytes()
    assert first["kind"] == "audio" and first["role"] == "drums.wet" and first["op"] == "plugin"
    assert first["of"] == src["id"] and first["lineage"] == [src["id"]]
    assert first["op_version"] == f"plugin@1+pedalboard:{host.pedalboard_version()}+{EFFECT}"
    assert first["params"]["effect"] == EFFECT
    assert first["params"]["params"] == host.coerce_params(EFFECT, PARAMS)
    # channels preserved; the effect actually changed the audio
    assert first["meta"]["ch"] == 2 and first["hash"] != src["hash"]

    def _boom(*a, **kw):  # a memo hit must not render at all
        raise AssertionError("render() called on a memo hit")

    monkeypatch.setattr(host, "render", _boom)
    second = cli.apply_effect(src, EFFECT, PARAMS)
    assert (second["params"]["memo"], second["hash"]) == ("hit", first["hash"])
    assert cas.get_path(second["hash"]).read_bytes() == first_bytes
    assert second["params"]["state_hash"] == first["params"]["state_hash"]


def test_forced_rerender_is_byte_identical_and_still_records(store, dry_frame):
    """--no-cache skips the LOOKUP and RE-RENDERS, so an identical hash proves byte-for-byte
    determinism is the renderer's property, not the memo's."""
    from smplstream import memostore

    src, effective = dry_frame(ch=2), host.coerce_params(EFFECT, PARAMS)
    first = cli.apply_effect(src, EFFECT, PARAMS)
    again = cli.apply_effect(src, EFFECT, PARAMS, use_cache=False)
    assert again["params"]["memo"] == "miss" and again["hash"] == first["hash"]
    assert memostore.get_json(cli.memo_key(EFFECT, src["hash"], effective))["hash"] == first["hash"]


def test_uncacheable_effect_skips_the_memo_entirely(store, dry_frame, monkeypatch):
    from smplstream import memostore  # built-ins are all cacheable — force the flag off

    real = D.describe
    monkeypatch.setattr(cli.D, "describe",
                        lambda n, p=None, **kw: {**real(n, p, **kw), "cacheable": False})
    wet = cli.apply_effect(src := dry_frame(), EFFECT, PARAMS)
    assert wet["params"]["cacheable"] is False and wet["params"]["memo"] == "miss"
    key = cli.memo_key(EFFECT, src["hash"], host.coerce_params(EFFECT, PARAMS))
    assert memostore.get_json(key) is None


def test_cli_passthrough_and_missing_audio(store, dry_frame, monkeypatch, capsys):
    src = dry_frame()
    code, out = _run(monkeypatch, capsys, ["--name", EFFECT, "--param", "threshold_db=-20"],
                     [NOTE, src])
    assert code == 0 and [f["kind"] for f in out] == ["text", "audio", "audio"]
    assert out[0] == NOTE and out[1]["id"] == src["id"]
    assert out[2]["op"] == "plugin" and out[2]["of"] == src["id"]
    code, out = _run(monkeypatch, capsys, ["--name", EFFECT], [NOTE])
    assert code == 1 and out[-1]["data"]["code"] == "not_found"


def test_missing_pedalboard_degrades_to_unsupported(store, monkeypatch, capsys):
    monkeypatch.setattr(host, "available", lambda: False)
    code, out = _run(monkeypatch, capsys, ["--name", EFFECT], [NOTE])
    assert code == 0 and out[0]["kind"] == "text"  # exit 0, passthrough first
    assert out[-1]["kind"] == "error" and out[-1]["data"]["code"] == "unsupported"
    assert "pedalboard" in out[-1]["data"]["message"]
    monkeypatch.setitem(sys.modules, "pedalboard", None)
    assert host.available() is False
