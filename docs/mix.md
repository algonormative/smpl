# `smpl mix` — the combinator, and the session it renders

`mix` is the suite's **third frame-flow shape**. `read` / `gen` / `cloud` are *sources*
(0 → 1); the analysis and edit ops are *filters* (1 → 1); `mix` is the **combinator**:

```text
N audio frames  ──►  smpl mix  ──►  ONE audio frame
                                     lineage = [every input id]
                                     op = mix, op_version = mix@1
                                     params = the resolved arrangement
```

No protocol change was needed. Inputs are ordinary `audio` frames already on the stream —
stems from `smpl stems`, `slice:<n>` regions, `smpl gen` / `smpl cloud` output, a
`.wet` subcomponent — selected by role, id, or content hash. The output is one ordinary
`audio` frame. Composition falls out of the existing frame contract:

```bash
smpl gen "kick" | smpl gen "hat" | smpl mix --arrange loop.smplset.json | smpl write out.wav
```

---

## Decision record: the session is data, `mix` is a control plane

**Decided.** A mix session **persists as canonical data on disk** — one JSON file,
`*.smplset.json` — and `smpl mix` is a **stateless control plane** over it: every verb is
read-file → mutate → write-file, no daemon, no in-memory session, no hidden state. The
agent drives the verbs; the file is the source of truth. This is the vault's agent-as-UI
principle applied literally (cf. `mpc` over `mpd`), and it is what makes the round-trip
property below testable rather than aspirational.

Concretely:

- **Format**: JSON (no new dependency — `pyyaml` is not in this workspace) in the
  *existing legacy smplmix session shape*, which `smpl pattern` already emits. That is the
  reconciliation: `smpl pattern`'s `.smplset.json` **is** a mix session, so
  `smpl pattern … | smpl mix` composes with no adapter, and no second, parallel session
  format enters the suite.
- **Location**: wherever the user/agent puts it — it is *their* document, passed by path
  (`--session` / `--arrange`), not hidden in a state dir. The only tool-owned state is the
  memo→CAS index at `~/.smpl/mix/index.json` (`SMPL_MIX_DIR`), which is a **cache**:
  deleting it costs a re-render and changes nothing about the output.
- **Round-trip with the stream**: a session travels as a `control` frame with role
  `mix.session` carrying the session inline (spilling to CAS past the 64 KiB inline
  limit). Every mutating verb emits one; `smpl mix show` emits one on demand; and a render
  with no `--session` picks one up off the stream. So a session is *either* a file path
  *or* a frame, and neither is privileged — which is what lets an agent build an
  arrangement across piped stages without touching the filesystem, then persist it once.
- **Not chosen**: a session as a CAS blob keyed by hash. A session is *mutable working
  state*; the CAS is for immutable content. Content-addressing it would mint a new hash on
  every `add-clip` and give the agent nothing it can edit. The render's *output* is
  content-addressed; the session that produced it is a document.

### Verb set

| Verb | What it does |
| --- | --- |
| `init` | write a new empty session (`--sr` `--ch` `--bpm` `--beats-per-bar`) |
| `add-clip` | append a clip (`--source` `--at` `--gain-db` `--pan` `--track`) |
| `set-gain` | set a track fader, or one clip's gain with `--index` (`--db`) |
| `set-pan` | set a track pan, or one clip's pan with `--index` (`--pan`) |
| `rm-clip` | remove a clip by `--track` + `--index` |
| `show` | emit the session as a `control` / `mix.session` frame |
| `render` *(default)* | resolve, memo-check, render → one `audio` frame |

`render --dry-run` resolves the arrangement and emits it as a `control` / `mix.plan`
frame **without producing bytes**, together with the `memo_key` it *would* render to. That
is the spec's *lazy frames* instinct (§ Lazy frames: "`smplmix --dry-run` is the existing
instinct; here it is universal") — and the dry-run key provably equals the render key, so
a planner can ask "is this already rendered?" for free.

### Reconciliation with the open session tickets

- **vault-3lxh** (derived-source lineage resolver / `.provenance.json`) wanted a mix
  session to surface where its material came from. In the frame protocol this is *free*:
  every input is an `audio` frame that already carries its own `lineage` / `op` /
  `params`, and the mix frame's `lineage` names every one of them, so the full derivation
  graph is reachable from the render without a sidecar file. `params.clips[].hash` pins
  the exact content each clip contributed. A `path:` source (a file with no upstream
  frame) is ingested into the CAS and emitted as its own `audio` frame **before** the mix
  frame, so lineage never dangles — the sidecar's job, done by the stream. *Legacy
  `.provenance.json` import was out of reach here: the legacy smplmix source is not
  present in this repo.*
- **vault-1h5c.2** (vocal-collage assembler) needs "many small clips at exact sample
  positions with per-clip gain". That is the v1 session verbatim: unbounded clips, each
  with an integer `sample` position and its own `gain_db` / `pan`, and a `control` frame
  that spills to the CAS when a large collage exceeds the inline payload limit. The
  assembler writes a session and calls `render`; it does not need a mix-specific API.
- **vault-3kvj** (`smpl compose`: timeline/loop composer, smplmix parity) is the same
  design landing under the name `mix`. There is deliberately **no second command** —
  `arrange` / `loop` are this session's `at` grammar and this verb set.

---

## Session format

```jsonc
{
  "smplset": 1,
  "sr": 44100,                 // session rate; default = the first resolved input's
  "ch": 2,                     // default: 2 if anything is panned or stereo, else 1
  "bpm": 120.0,                // only needed for bar.beat.frac positions
  "beats_per_bar": 4.0,
  "master": { "guard": "peak", "ceiling_dbfs": -0.3 },
  "tracks": [
    {
      "name": "main",
      "gain_db": 0.0,          // track fader; stacks with each clip's gain
      "pan": 0.0,
      "clips": [
        { "source": {"role": "stem:drums"}, "at": {"sample": 0},     "gain_db": 0.0, "pan": 0.0 },
        { "source": {"path": "pad.wav"},    "at": {"bar": "2.1.5"},  "gain_db": -6.0, "pan": -0.5 }
      ]
    }
  ]
}
```

**Clip source** — one of `{"role": …}` (last-wins over the stream), `{"id": …}`,
`{"hash": "blake3:…"}` (stream first, then the CAS), `{"path": …}` (ingested at render
time). On the command line these are written `role:stem:drums`, `id:blake3:…`,
`blake3:…`, `path:pad.wav`; a bare token is a role unless it looks like a path.

**Clip position (`at`)** — the timebase is sample-accurate (spec → *Units & timebase*):

| Form | Meaning |
| --- | --- |
| `{"sample": N}` / `sample:N` / bare integer | sample index at the **session** rate |
| `{"sec": F}` / `sec:F` / `1.5s` | seconds |
| `{"bar": "2.1.5"}` / `bar:2.1.5` / bare `"2.1.5"` | `bar.beat.frac`, 1-indexed (needs `bpm`) |
| `{"marker": {"role": "beat", "index": 3}}` / `marker:beat#3` | that marker point's `sample` |

A **bare dotted numeral is bars, never seconds** — `"1.3"` from `smpl pattern` means bar 1
beat 3. Seconds always carry a marker (`sec:1.5` or `1.5s`), because one string reading as
1.5 s in one place and bar 1 beat 5 in another is exactly the ambiguity that silently
misplaces clips.

A `marker:` position reads `marker.data[i].sample` at the marker frame's **native** rate
(`params.sr_hz`, else the rate of the frame it derives from) and rescales to the session
rate, so a clip lands on the exact detected onset/beat rather than on a rounded float
second. That is why `smpl slice` and friends are required to carry `sample: int`.

## Render semantics (v1)

- **Sum.** Each clip is decoded from the CAS, channel-fitted (mono → N by duplication,
  N → 1 by mean), scaled by `clip.gain_db + track.gain_db`, panned, and added into the bus
  at its start sample. Output length = `max(start + length)` over all clips.
- **Pan** is **constant power**: centre is −3 dB per leg (√2⁄2), hard edges put full
  amplitude on one leg. Only meaningful on a stereo (or wider) bus.
- **Clip guard.** The bus is summed in float64 and is *never* hard-clipped. If the peak
  exceeds `master.ceiling_dbfs` (default **−0.3 dBFS**), **one** gain is applied across the
  whole bus and reported as `params.guard_gain_db`, with the pre-guard peak in
  `params.peak_before`. One bus gain — not per-clip normalization — because the arrangement
  the agent wrote is a statement about *relative* balance, and per-clip rescaling would
  silently rewrite it. Set `master.guard` to anything but `"peak"` to disable and keep the
  raw float sum (`smpl limit` is the delivery-ceiling tool).
- **Sample rates must match.** A clip whose input rate differs from the session rate is
  **refused**, not silently resampled — convert it first (`smpl convert`). Resampling
  inside the combinator would put a quality decision in an op that never announced one.

## Memoization

`memo_key = blake3(mix ‖ mix@1 ‖ sorted(input hashes) ‖ canonicalize(arrangement))`, per
the spec's normative formula, with the op's default table (`guard`, `ceiling_dbfs`) filled
in before hashing. The arrangement is the *resolved* plan — content hashes, integer start
samples, summed gains — so two sessions that describe the same render share a key, and
moving one clip by one sample does not. `params.cache_hit` records which path was taken.

The memo→CAS index lives at `~/.smpl/mix/index.json` (`SMPL_MIX_DIR`), self-contained in
the mix module; it moves to the shared cache store when that lands.

**Determinism.** Rendering is pure float64 numpy plus a float32 WAV encode, so the same
session over the same CAS content yields a byte-identical blob and therefore the same
content hash — with the memo index deleted, a re-render lands on the same hash. That is
the round-trip property `test_session_roundtrips_to_a_deterministic_render` asserts.

## Not in v1

Crossfades, buses/sends, automation curves, sidechain, per-clip `len` trimming,
`transform.transpose_semitones`, master `loudnorm`/`limiter`, `gen:` clip sources, and
resampling. Legacy per-clip fields that v1 does not render (`len`, `transform`) are
preserved in the session and flagged on the resolved clip as `unsupported` rather than
being silently honored or silently dropped.
