"""Embedded-chunk export: the plan built from frames, and the bytes written to the file.

Every assertion re-parses the written container (RIFF chunks / FLAC metadata blocks) and
compares against the frames that produced it — a round-trip, not a "did we call it" check.
The DAW/sampler half of the acceptance is a manual procedure, documented in the
``smpl_cli._chunks`` module docstring.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess

import numpy as np
import pytest
import soundfile as sf

from smpl_cli import _chunks

SR = 44100


# ---------------------------------------------------------------------------
# Plan building (what the stream carries → what gets embedded).
# ---------------------------------------------------------------------------
def _audio(fid="a1"):
    return {"kind": "audio", "id": fid, "hash": "blake3:" + "a" * 64, "meta": {"sr": SR}}


def _marker(points, role, of="a1"):
    return {"kind": "marker", "id": f"m-{role}", "role": role, "of": of, "data": points}


def _feature(data, role="beats", of="a1"):
    return {"kind": "feature", "id": f"f-{role}", "role": role, "of": of, "data": data}


def test_collect_reads_tempo_key_markers_and_caption():
    frames = [
        _audio(),
        _feature({"rhythm.bpm": 128.0, "rhythm.time_signature": "3/4"}),
        _feature({"tonal.key_key": "D#", "tonal.key_scale": "minor"}, role="key"),
        _marker([{"t": 0.0, "sample": 0}, {"t": 0.5, "sample": 22050}], "onset"),
        {"kind": "text", "id": "t1", "role": "caption", "of": "a1", "data": "a dark kick"},
    ]
    plan = _chunks.collect(frames, _audio())
    assert plan["bpm"] == 128.0
    assert plan["time_signature"] == (3, 4)
    assert plan["key"] == "D#" and plan["scale"] == "minor"
    assert plan["root_note"] == _chunks.ROOT_NOTE_C + 3
    assert plan["cue"] == [0, 22050]
    assert plan["description"] == "a dark kick"


def test_collect_embeds_only_what_the_stream_carries():
    """No bpm on the wire → no tempo in the plan. Nothing is invented."""
    plan = _chunks.collect([_audio(), _feature({"rhythm.bpm_confidence": 0.4})], _audio())
    assert "bpm" not in plan and "key" not in plan and "cue" not in plan and "loop" not in plan


def test_collect_prefers_slice_markers_over_the_beat_grid():
    frames = [
        _audio(),
        _marker([{"t": 0.0, "sample": 7}], "slice"),
        _marker([{"t": 0.0, "sample": 0}, {"t": 1.0, "sample": SR}], "beat"),
    ]
    assert _chunks.collect(frames, _audio())["cue"] == [7]


def test_loop_from_markers_and_from_loopify():
    frames = [_audio(), _marker([{"t": 0.0, "sample": 100}, {"t": 1.0, "sample": 900}], "loop")]
    assert _chunks.collect(frames, _audio())["loop"] == (100, 900)

    loopified = dict(_audio(), op="loopify")
    assert _chunks.collect([loopified], loopified)["loop"] == (0, None)


def test_marker_without_sample_is_an_error_not_a_rounding():
    """The precondition: cue positions derive from `marker.sample`, never from float `t`."""
    frames = [_audio(), _marker([{"t": 0.25, "label": "onset-0"}], "onset")]
    with pytest.raises(_chunks.MarkerPrecisionError) as excinfo:
        _chunks.collect(frames, _audio())
    assert "sample" in str(excinfo.value)


def test_markers_for_another_frame_are_dropped_not_embedded():
    """Sample positions are only valid for the frame they were computed on.

    A trimming op (loopify) shifts every marker, so a marker set describing an ancestor
    must not become this file's cue points — embedding nothing beats embedding wrong.
    """
    stale = _marker([{"t": 0.0, "sample": 999}], "onset", of="older")
    plan = _chunks.collect([stale, _audio()], _audio())
    assert "cue" not in plan

    fresh = _marker([{"t": 0.0, "sample": 5}], "onset", of="a1")
    assert _chunks.collect([stale, fresh, _audio()], _audio())["cue"] == [5]


def test_scalars_travel_downstream_even_when_markers_do_not():
    """Tempo/key/caption survive an edit; they are read from the stream, not the lineage."""
    frames = [_audio(), _feature({"rhythm.bpm": 140.0}, of="older")]
    plan = _chunks.collect(frames, _audio())
    assert plan["bpm"] == 140.0 and "cue" not in plan


# ---------------------------------------------------------------------------
# WAV: byte-level round trip.
# ---------------------------------------------------------------------------
@pytest.fixture()
def wav(tmp_path):
    path = tmp_path / "out.wav"
    t = np.arange(SR) / SR
    sf.write(str(path), (0.25 * np.sin(2 * np.pi * 220 * t)).astype("float32"), SR,
             subtype="PCM_16")
    return path


def _chunk_map(path):
    return dict(_chunks.parse_riff(path.read_bytes()))


def test_wav_embeds_all_four_chunks_and_stays_readable(wav):
    before, _ = sf.read(str(wav), dtype="float32")
    plan = {
        "bpm": 120.0,
        "time_signature": (4, 4),
        "key": "A",
        "scale": "minor",
        "root_note": _chunks.ROOT_NOTE_C + 9,
        "cue": [0, 11025, 22050],
        "loop": (11025, 33075),
        "description": "test tone",
    }
    written = _chunks.embed(wav, plan, sr=SR, n_samples=SR)
    assert written == ["bext", "acid", "smpl", "cue"]

    chunks = _chunk_map(wav)
    assert {b"bext", b"acid", b"smpl", b"cue "} <= set(chunks)

    # The audio is untouched and the container still parses as a WAV.
    after, sr = sf.read(str(wav), dtype="float32")
    assert sr == SR
    assert np.array_equal(before, after)


def test_wav_cue_positions_are_the_marker_samples(wav):
    samples = [0, 4321, 30000]
    _chunks.embed(wav, {"cue": samples}, sr=SR, n_samples=SR)
    payload = _chunk_map(wav)[b"cue "]

    count = struct.unpack_from("<I", payload, 0)[0]
    assert count == len(samples)
    for i, expected in enumerate(samples):
        ident, position, chunk_id, _, _, offset = struct.unpack_from("<II4sIII", payload, 4 + i * 24)
        assert ident == i + 1
        assert chunk_id == b"data"
        assert position == expected and offset == expected


def test_wav_acid_carries_tempo_meter_and_beat_count(wav):
    plan = {"bpm": 120.0, "time_signature": (3, 4), "root_note": 62, "loop": (0, None)}
    _chunks.embed(wav, plan, sr=SR, n_samples=SR * 2)  # 2 seconds at 120 BPM = 4 beats
    flags, root, _, _, beats, den, num, tempo = struct.unpack("<IHHfIHHf", _chunk_map(wav)[b"acid"])
    assert tempo == pytest.approx(120.0)
    assert beats == 4
    assert (num, den) == (3, 4)
    assert root == 62
    assert flags & _chunks.ACID_ROOT_SET
    assert not flags & _chunks.ACID_ONE_SHOT  # a loop, not a one-shot


def test_wav_acid_marks_a_keyed_sample_with_no_loop_as_one_shot(wav):
    _chunks.embed(wav, {"root_note": 60}, sr=SR, n_samples=SR)
    flags = struct.unpack("<IHHfIHHf", _chunk_map(wav)[b"acid"])[0]
    assert flags & _chunks.ACID_ONE_SHOT


def _acid_flags(path):
    return struct.unpack("<IHHfIHHf", _chunk_map(path)[b"acid"])[0]


def test_wav_acid_flags_a_grid_cut_beat_tracked_file_as_a_loop(wav):
    """Hosts don't warp one-shots — a tracked, grid-cut file must not be flagged one.

    4 s at 120 BPM is exactly 8 beats.
    """
    plan = {"bpm": 120.0, "beat_grid": True}
    _chunks.embed(wav, plan, sr=SR, n_samples=SR * 4)
    assert not _acid_flags(wav) & _chunks.ACID_ONE_SHOT


def test_wav_acid_keeps_a_beat_tracked_take_that_is_not_grid_cut_as_a_one_shot(wav):
    """Same analysis, length landing mid-beat (8.5 beats) — a take, not a loop."""
    plan = {"bpm": 120.0, "beat_grid": True}
    _chunks.embed(wav, plan, sr=SR, n_samples=int(SR * 4.25))
    assert _acid_flags(wav) & _chunks.ACID_ONE_SHOT


def test_wav_acid_keeps_an_untracked_hit_as_a_one_shot(wav):
    """No beat grid for this frame → one-shot, however the length happens to divide."""
    _chunks.embed(wav, {"bpm": 120.0}, sr=SR, n_samples=SR * 4)
    assert _acid_flags(wav) & _chunks.ACID_ONE_SHOT
    # …and a grid with no tempo can't be checked against one either.
    _chunks.embed(wav, {"root_note": 60, "beat_grid": True}, sr=SR, n_samples=SR * 4)
    assert _acid_flags(wav) & _chunks.ACID_ONE_SHOT


def test_grid_cut_needs_at_least_two_beats(wav):
    """A one-beat hit is trivially "a whole number of beats"; that proves nothing."""
    plan = {"bpm": 120.0, "beat_grid": True}
    assert not _chunks.is_grid_cut(plan, sr=SR, n_samples=SR // 2)   # 1 beat
    assert _chunks.is_grid_cut(plan, sr=SR, n_samples=SR)            # 2 beats


def test_grid_tolerance_follows_a_drifting_tempo_estimate():
    """Relative tolerance: a 0.15%-off tempo still reads as grid-cut over 64 beats."""
    plan = {"bpm": 120.0 * 1.0015, "beat_grid": True}
    assert _chunks.is_grid_cut(plan, sr=SR, n_samples=SR * 32)  # 64 beats at true tempo


def test_collect_records_the_beat_grid_only_for_this_frame():
    grid = _marker([{"t": 0.0, "sample": 0}, {"t": 0.5, "sample": 22050}], "beat")
    assert _chunks.collect([_audio(), grid], _audio())["beat_grid"] is True
    # A grid tracked on some other frame is not this frame's grid.
    stale = _marker([{"t": 0.0, "sample": 0}], "beat", of="older")
    assert "beat_grid" not in _chunks.collect([_audio(), stale], _audio())
    # Onsets are transients, not a metrical grid.
    onsets = _marker([{"t": 0.0, "sample": 0}], "onset")
    assert "beat_grid" not in _chunks.collect([_audio(), onsets], _audio())


def test_wav_smpl_carries_root_note_and_loop_points(wav):
    _chunks.embed(wav, {"root_note": 67, "loop": (1000, None)}, sr=SR, n_samples=SR)
    payload = _chunk_map(wav)[b"smpl"]
    (_, _, period, root, _, _, _, n_loops, _) = struct.unpack_from("<9I", payload, 0)
    assert period == int(round(1e9 / SR))
    assert root == 67
    assert n_loops == 1
    _, loop_type, start, end, _, _ = struct.unpack_from("<6I", payload, 36)
    assert loop_type == 0                 # forward
    assert (start, end) == (1000, SR - 1)  # open end → last sample of the file


def test_wav_bext_carries_originator_date_and_description(wav):
    import re

    _chunks.embed(wav, {"description": "a dark kick", "bpm": 90.0}, sr=SR, n_samples=SR)
    payload = _chunk_map(wav)[b"bext"]
    assert len(payload) == 602
    assert payload[0:256].rstrip(b"\x00").decode() == "a dark kick"
    assert payload[256:288].rstrip(b"\x00").decode() == _chunks.ORIGINATOR
    date = payload[320:330].decode()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", date)
    assert re.fullmatch(r"\d{2}:\d{2}:\d{2}", payload[330:338].decode())


def test_wav_embed_replaces_rather_than_duplicates(wav):
    for _ in range(2):
        _chunks.embed(wav, {"cue": [1, 2], "bpm": 100.0}, sr=SR, n_samples=SR)
    ids = [cid for cid, _ in _chunks.parse_riff(wav.read_bytes())]
    for owned in (b"bext", b"acid", b"cue "):
        assert ids.count(owned) == 1
    assert ids.index(b"data") == len(ids) - 1  # metadata sits ahead of the audio


# ---------------------------------------------------------------------------
# FLAC: the Vorbis-comment mirror.
# ---------------------------------------------------------------------------
@pytest.fixture()
def flac(tmp_path):
    path = tmp_path / "out.flac"
    t = np.arange(SR) / SR
    sf.write(str(path), (0.25 * np.sin(2 * np.pi * 220 * t)).astype("float32"), SR)
    return path


def _comments(path):
    blocks, _ = _chunks.parse_flac(path.read_bytes())
    payload = next(data for btype, data in blocks if btype == 4)
    _, items = _chunks.parse_vorbis_comment(payload)
    return dict(item.split("=", 1) for item in items)


def test_flac_mirrors_tempo_key_and_loop(flac):
    before, _ = sf.read(str(flac), dtype="float32")
    plan = {"bpm": 128.0, "key": "F#", "scale": "major", "loop": (100, 4000)}
    assert _chunks.embed(flac, plan, sr=SR, n_samples=SR) == ["BPM", "KEY", "LOOP_START", "LOOP_END"]

    tags = _comments(flac)
    assert tags["BPM"] == "128"
    assert tags["KEY"] == "F# major"
    assert tags["LOOP_START"] == "100" and tags["LOOP_END"] == "4000"

    after, sr = sf.read(str(flac), dtype="float32")  # still decodable, bit-identical
    assert sr == SR and np.array_equal(before, after)


def test_flac_embed_is_idempotent_and_keeps_foreign_tags(flac):
    blocks, tail = _chunks.parse_flac(flac.read_bytes())
    at = next(i for i, (btype, _) in enumerate(blocks) if btype == 4)
    vendor, items = _chunks.parse_vorbis_comment(blocks[at][1])
    blocks[at] = (4, _chunks._build_vorbis_comment(vendor, items + ["ARTIST=someone"]))
    flac.write_bytes(_chunks._build_flac(blocks, tail))

    for _ in range(2):
        _chunks.embed(flac, {"bpm": 90.0}, sr=SR, n_samples=SR)
    tags = _comments(flac)
    assert tags["ARTIST"] == "someone"
    assert tags["BPM"] == "90"
    assert sf.info(str(flac)).frames == SR


def test_flac_without_tempo_key_or_loop_writes_nothing(flac):
    raw = flac.read_bytes()
    assert _chunks.embed(flac, {"cue": [1, 2]}, sr=SR, n_samples=SR) == []
    assert flac.read_bytes() == raw


# ---------------------------------------------------------------------------
# Through the CLI (the real sink).
# ---------------------------------------------------------------------------
SMPL = shutil.which("smpl")
cli = pytest.mark.skipif(SMPL is None, reason="`smpl` console script not on PATH")


@pytest.fixture()
def env(tmp_path):
    e = dict(os.environ)
    e["SMPL_CAS_DIR"] = str(tmp_path / "cas")
    e.pop("VIRTUAL_ENV", None)
    return e


@pytest.fixture()
def tone(tmp_path):
    p = tmp_path / "tone.wav"
    t = np.arange(SR) / SR
    sf.write(str(p), (0.5 * np.sin(2 * np.pi * 220 * t)).astype("float32"), SR, subtype="PCM_16")
    return str(p)


def _read_frames(env, tone):
    r = subprocess.run(["smpl", "read", tone], capture_output=True, env=env, timeout=60)
    assert r.returncode == 0, r.stderr
    return [json.loads(line) for line in r.stdout.splitlines() if line.strip()]


def _pipe(frames, env, out):
    stdin = ("\n".join(json.dumps(f) for f in frames) + "\n").encode()
    return subprocess.run(["smpl", "write", str(out)], input=stdin, capture_output=True,
                          env=env, timeout=60)


@cli
def test_cli_write_embeds_from_the_stream(env, tone, tmp_path):
    from smplstream import frames as F

    source = _read_frames(env, tone)[0]
    fid = source["id"]
    stream = [
        source,
        F.feature_frame({"rhythm.bpm": 96.0, "rhythm.time_signature": "4/4"},
                        role="beats", of=fid, lineage=[fid]),
        F.marker_frame([{"t": 0.0, "sample": 0}, {"t": 0.25, "sample": 11025}],
                       role="onset", of=fid, lineage=[fid]),
    ]
    out = tmp_path / "written.wav"
    r = _pipe(stream, env, out)
    assert r.returncode == 0, r.stderr

    chunks = dict(_chunks.parse_riff(out.read_bytes()))
    assert struct.unpack("<IHHfIHHf", chunks[b"acid"])[7] == pytest.approx(96.0)
    assert struct.unpack_from("<I", chunks[b"cue "], 0)[0] == 2
    assert struct.unpack_from("<II4sIII", chunks[b"cue "], 4 + 24)[1] == 11025


@cli
def test_cli_beat_tracked_grid_cut_export_is_warpable(env, tone, tmp_path):
    """The acceptance case end to end: analyze → write → the DAW is allowed to warp it.

    The 1 s tone is exactly 2 beats at 120 BPM, so the beat grid makes it a loop.
    """
    from smplstream import frames as F

    source = _read_frames(env, tone)[0]
    fid = source["id"]
    stream = [
        source,
        F.feature_frame({"rhythm.bpm": 120.0}, role="beats", of=fid, lineage=[fid]),
        F.marker_frame([{"t": 0.0, "sample": 0}, {"t": 0.5, "sample": SR // 2}],
                       role="beat", of=fid, lineage=[fid]),
    ]
    out = tmp_path / "warpable.wav"
    assert _pipe(stream, env, out).returncode == 0

    acid = dict(_chunks.parse_riff(out.read_bytes()))[b"acid"]
    flags, _, _, _, beats, _, _, tempo = struct.unpack("<IHHfIHHf", acid)
    assert tempo == pytest.approx(120.0) and beats == 2
    assert not flags & _chunks.ACID_ONE_SHOT


@cli
def test_cli_no_embed_is_a_faithful_copy(env, tone, tmp_path):
    from smplstream import frames as F

    source = _read_frames(env, tone)[0]
    fid = source["id"]
    stream = [source, F.feature_frame({"rhythm.bpm": 96.0}, role="beats", of=fid, lineage=[fid])]
    out = tmp_path / "plain.wav"
    stdin = ("\n".join(json.dumps(f) for f in stream) + "\n").encode()
    r = subprocess.run(["smpl", "write", str(out), "--no-embed"], input=stdin,
                       capture_output=True, env=env, timeout=60)
    assert r.returncode == 0, r.stderr
    assert b"acid" not in {cid for cid, _ in _chunks.parse_riff(out.read_bytes())}


@cli
def test_cli_refuses_markers_without_sample(env, tone, tmp_path):
    from smplstream import frames as F

    source = _read_frames(env, tone)[0]
    fid = source["id"]
    stream = [source, F.marker_frame([{"t": 0.25, "label": "onset-0"}], role="onset",
                                     of=fid, lineage=[fid])]
    out = tmp_path / "refused.wav"
    r = _pipe(stream, env, out)

    assert r.returncode == 1
    assert not out.exists()  # nothing written: no file with wrong cue points
    errors = [json.loads(line) for line in r.stdout.splitlines() if line.strip()]
    error = [f for f in errors if f["kind"] == "error"][-1]
    assert error["data"]["code"] == "op_failed"
    assert "sample" in error["data"]["message"]
