---
title: "smplstream v1 — composable media-analysis wire protocol"
created: 2026-06-14
tags: [music, audio, protocol, spec, interop, smplstream, analysis]
description: "The durable interop contract for the smpl* suite: NDJSON frames over a content-addressed store, with hybrid raw-WAV adapters. Audio-first, media-agnostic."
---

# smplstream v1 — wire protocol & interop contract

**This is the canonical contract.** Any new tool in the analysis suite
(`/analysis:audio` today; `/analysis:image`, `/analysis:video` later) MUST
read and write this format to compose with the rest. Treat changes here as
versioned API changes, not edits.

This document is **normative**: MUST / MUST NOT / SHOULD carry RFC-2119 weight.
The sections that pin byte-level invariants — **Canonical PCM**, **Memoization**,
**CAS integrity**, **Units & timebase** — are what make two independent
implementations agree. A conformance suite (golden hashes + lineage closure)
enforces them; see the project README.

The whole point: **pipe self-describing *frames* that reference
content-addressed bytes — never pipe the heavy bytes themselves.** This is
what buys multi-payload streams (audio + image + text + vector in one pipe),
free memoization, and lazy evaluation. See the [README](README.md) for the build sequence and rationale.

## Design principle: be a boring Unix citizen at every seam

The protocol earns composability by being aggressively standard:

- **Stream is NDJSON** → `jq` / `fx` / `mlr` work with zero custom tooling.
- **Heavy bytes live in a CAS** → any external tool gets a real file path.
- **Hybrid raw-WAV adapters** → splice `sox`/`ffmpeg`/any stdin-WAV tool
  mid-pipe without losing lineage or cache.

A tool that is hard to pipe into `jq`, or that can't hand a path to `sox`,
is violating the spec even if its JSON is valid.

## The frame

One frame per line (newline-delimited JSON / JSON Lines), **UTF-8**. Every frame:

```jsonc
{
  "v": 1,                      // protocol version (REQUIRED)
  "kind": "audio",            // frame type (REQUIRED) — see table
  "id": "blake3:9af2…",      // globally-unique frame id (REQUIRED) — see Id assignment
  "seq": 7,                   // monotonic per-stream ordinal (OPTIONAL) — survives reorders
  "role": "stem:drums",       // human/semantic label (OPTIONAL but encouraged)
  "of": "blake3:1c0a…",      // frame id this annotates/derives-from (OPTIONAL)
  "lineage": ["blake3:1c0a…"],// upstream frame ids (OPTIONAL)
  "op": "demucs",             // operation that produced this frame (OPTIONAL)
  "op_version": "audio-separator@0.28+htdemucs:blake3:abcd…", // see Memoization
  "params": {"model": "htdemucs"}, // normalized op params (OPTIONAL)
  "consumed": false,          // true if retained only so its id stays resolvable
  "hash": "blake3:c3d4…",    // CAS key for heavy payloads (kind-dependent)
  "media": "audio/wav",       // MIME type — REQUIRED whenever `hash` is present
  "meta": { "sr": 48000, "dur": 8.0, "ch": 2 }, // kind-specific metadata
  "data": null                // inline payload for small kinds (text/vector/marker)
}
```

Rules:

- Unknown fields MUST be preserved on passthrough (forward-compat).
- Either `hash` (blob in CAS) **or** `data` (inline) carries the payload,
  never both. Heavy kinds use `hash`; small kinds use `data`. `media` is
  REQUIRED whenever `hash` is present (the CAS extension derives from it).
- A `data` payload MUST NOT exceed **64 KiB** serialized; larger payloads MUST
  move to CAS via `hash` (see *Inline payloads & size limits*).
- Implementations MUST accept NDJSON lines up to at least **1 MiB**.

### Id assignment (collision-proof by construction)

A frame's `id` MUST be globally unique **without coordination** — so two
independently-numbered streams can be merged (`cat a.ndjson b.ndjson`,
`smpl gen` appending to an inbound stream) without reference corruption.

- Producers MUST mint ids as either a **content-derived token**
  (`blake3:` prefix over the frame's defining fields — preferred, because the
  *same* frame from two pipelines then shares one id and deduplicates) **or** a
  random token. Producers MUST NOT use a per-stream sequential counter
  (`f1`, `f2`, …) — that is the collision trap.
- On passthrough a tool MUST preserve inbound ids **verbatim**.
- If a tool sees two inbound frames sharing an `id`, it MUST emit an `error`
  frame (`code: id_collision`) and MUST NOT mint references that depend on the
  ambiguous id.

`of` / `lineage` reference ids; they are opaque — a tool passing through a frame
it doesn't understand can still carry lineage to it safely.

### Frame kinds

| kind      | payload          | via         | notes |
|-----------|------------------|-------------|-------|
| `audio`   | PCM/WAV          | `hash`      | the substrate; `meta`: `sr`, `dur`, `ch`, `bits`, `fmt` |
| `image`   | PNG              | `hash`      | spectrogram, waveform, annotated render; `of` → the audio it depicts |
| `video`   | MP4              | `hash`      | reserved (future `/analysis:video`); scrolling spectrogram, etc. |
| `midi`    | .mid / events    | `hash`/`data` | reserved (additive); an **offline** score/arrangement (note/CC events, timestamps in the spec timebase). `.mid` blob in CAS, or small event lists inline. NOT a realtime event stream — see *Out of scope* |
| `text`    | string           | `data`      | caption, lyric transcript, summary; `role` says which; CAS if > 64 KiB |
| `vector`  | float[] or blob  | `data`/`hash` | embedding; **size-split, see below**; `meta`: `model`, `dim`, `dtype` |
| `marker`  | t-list           | `data`      | onsets, beats, slice points, sections; `data`: `[{t, sample?, dur?, label?}]` |
| `feature` | object           | `data`      | scalar/struct analysis (bpm, key, loudness…); CAS if > 64 KiB |
| `control` | object           | `data`      | pipeline hints (dry-run, budget, requested outputs) — not content |
| `error`   | object           | `data`      | non-fatal failure on one frame; `data`: `{code, message, of, op?}` |

`feature` vs `text`: `feature` is structured (machine-consumed, e.g.
`{"bpm":128.0,"key":"Am"}`); `text` is prose for the human/LLM.

**`vector` payload (size-split, in-band discriminator).** Small vectors
(`dim ≤ 64`) MAY inline as `data: float[]`. Vectors with `dim > 64` MUST go to
CAS as a binary blob (`media: "application/x-npy"`, `.npy` or `.safetensors`,
**never pickle**). `meta` carries `model`, `dim`, and `dtype` (e.g. `float32`)
in **both** cases. A consumer picks the location by presence of `hash` vs `data`
(the "never both" rule still holds). **Canonical hashing:** an inline vector's
CAS hash, when later materialized, is computed over the **binary `.npy` of the
declared `dtype`** — not over JSON text — so the inline and CAS forms of the
same vector share one hash (JSON float text is lossy and would diverge).

## Content-addressed store (CAS)

- Location: `~/.smpl/cas/` (override `SMPL_CAS_DIR`).
- Key: `blake3:<hex>` of the **canonical decoded PCM** for audio (see below),
  and of the raw bytes for already-canonical blobs (PNG, MP4, `.npy`).
- Layout: sharded `~/.smpl/cas/<aa>/<aabbcc…>.<ext>` + a sibling
  `<hash>.meta.json` for cheap metadata reads without decoding. `<ext>` derives
  from the frame's `media`.
- Blobs are immutable. Garbage-collected by `smpl gc` (see *CAS integrity*).

### Canonical decoded PCM (the hash basis) — NORMATIVE

The audio hash MUST be reproducible across decoders, library versions, and
machines, or the cache is silently non-portable (two "identical" stems → two
CAS entries → broken dedup + memo thrash, no error). Before hashing, audio is
decoded to a single canonical form with **no discretionary conversion**:

- **Sample format:** IEEE **float32, little-endian**.
- **Channel layout:** **native channel count, interleaved, file-declared
  order.** NO down/upmix — mono stays mono, 5.1 stays 6 channels.
- **Sample rate:** **native rate — NO resampling.** (Resampling is an explicit
  op that produces a *new* frame with its own hash; folding a resampler into
  the hash basis would bind the key to a resampler library version.)
- **NO amplitude normalization, NO dither, NO metadata** — the hash covers the
  PCM samples only. ("Normalized" here means *format-canonicalized*, never
  level-normalized; gain MUST NOT change the hash.)

The key binds format identity so rate/channel/format can't alias:

```text
audio_hash = blake3( canonical_pcm_bytes ‖ u32le(sample_rate) ‖ u8(channels) ‖ u8(format_tag) )
```

Decoders that disagree at the bit level on the same input are a conformance
bug. The conformance suite ships a **golden corpus** of `(input file → expected
hash)`; passing it is required to be v1-conformant. `meta.fmt` records the
source format for humans but does not enter the hash.

### CAS integrity (atomicity, GC, path-safety) — NORMATIVE

- **Atomic writes.** A blob is written to a temp file in the CAS dir and
  atomically `rename()`d into place — a reader never observes a partial blob.
  A write whose recomputed hash ≠ target hash is a fatal integrity error (the
  bad bytes MUST NOT land at the canonical path). This is required because
  memoization + fan-out pipes routinely compute the same blob concurrently.
- **GC safety.** `smpl gc` MUST NOT delete a blob referenced by any frame
  emitted within a grace window, MUST hold a lock excluding concurrent
  producers, and MUST honor producer-held reservations for lazy/promised
  hashes not yet in any ref log. (GC *policy* — TTL, thresholds — may defer to
  v1.1; the *safety rule* "never delete a live or in-flight blob" is v1.)
- **Path safety.** A `hash` MUST match `^blake3:[0-9a-f]{64}$` before being
  mapped to a filesystem path. Any other form is rejected. CAS paths derive
  only from validated hex — never from arbitrary strings — so a hostile
  `blake3:../../etc/…` can't traverse out (CAS paths are handed straight to
  `sox`/`ffmpeg`/Demucs).

## Memoization (the performance story) — NORMATIVE

Every *cacheable* operation is a pure function of its inputs, implementation
version, and environment:

```text
memo_key = blake3( op ‖ op_version ‖ sorted(input_hashes) ‖ canonicalize(params) ‖ env_fingerprint )
```

- `op_version`: a string each op declares, bumped on **any** behavior change.
  For ML ops it MUST incorporate the **weights identity** (model file `blake3`
  or registry id+version), not just a friendly model name — otherwise a
  Demucs/MERT/Whisper upgrade silently serves stale results from the old model.
- `env_fingerprint`: for shell-out ops whose output varies by tool version
  (`sox`/`ffmpeg`, sample-rate-converter quality), the resolved tool version
  (e.g. hash of `ffmpeg -version`). Empty for pure-Python deterministic ops.
- **Non-deterministic ops** (GPU/MPS nondeterminism) MUST either pin
  determinism (fixed seed, `torch.use_deterministic_algorithms`) or declare
  `cacheable: false` and skip memoization. Purity is a **per-op declaration**,
  not a global assumption — the op contract carries a `deterministic` boolean.
- If a blob for `memo_key`'s output already exists, the op is skipped and the
  cached output frame is emitted. This generalizes `smplcat`'s per-file cache
  to the **whole pipeline** — tweak only the tail of a pipe and the head is a
  cache hit.
- The global `SCHEMA` constant gates *protocol-schema* invalidation only;
  per-op correctness rides on `op_version`, so a one-op upgrade doesn't nuke
  the whole cache.

### Parameter canonicalization

`canonicalize(params)` MUST make two spellings of the same request one key:

- Sort object keys; render numbers in a **fixed canonical form** (shortest
  round-trippable decimal; `6.0` ≡ `6`).
- Params that are semantically **sets** are sorted; params that are
  **sequences** preserve order — the op declares which.
- **Do NOT drop defaults.** Instead, fill omitted params from the op's
  declared default table *for that `op_version`* before hashing, so the key is
  complete and stable even when a later version changes a default. (Dropping
  defaults couples the key to the live default table — a hidden version
  dependency.)

### Lazy frames

A frame MAY declare a `hash` whose blob is not yet materialized (a promise).
Bytes are computed only when a downstream stage resolves them (`smpl write`,
`smpl as-wav`, an effect that needs samples). `smplmix --dry-run` is the
existing instinct; here it is universal.

## Stream ordering — NORMATIVE

**Stream order is significant.** Frames carry an implicit causal order:

- A frame's `lineage` / `of` targets MUST appear **earlier** in the stream.
- Tools MUST emit passthrough frames **before** the derived frames that
  reference them, and MUST preserve the relative order of passthrough frames.
- The OPTIONAL `seq` ordinal (monotonic per stream) lets order survive a
  reordering filter; selection tie-breaks on `seq` when present, else on stream
  position.
- `jq`/`mlr` reshapes that **reorder** frames are outside the contract. Safe:
  per-line filters (`select`, field projection). Unsafe: `sort_by`, `group_by`
  that reorder — these can break last-wins selection (below).

## Selection semantics — NORMATIVE

`smpl select` / `as-wav` assume role→frame resolves to one frame, but roles are
not guaranteed unique (a re-filtered `stem:drums.wet` appears twice). Resolution
rules:

- A role/predicate matching multiple frames defaults to **last-wins** (the
  most-recently-emitted match, by `seq` or stream position).
- `--all` passes every match; `--strict` errors on > 1.
- `smpl as-wav` resolves "the single audio frame" by the same last-wins rule by
  default; `--strict` requires exactly one.

## Units & timebase — NORMATIVE

A media protocol with no units convention will not interoperate.

- **Timebase.** Marker `t` and `dur`, and `meta.dur`, are **float seconds**.
  Markers destined for sample-accurate export (Octatrack `.ot`, WAV `cue`)
  MUST also carry `sample: int`, indexed against the frame's **native sample
  rate** (`meta.sr`) — float seconds alone can't round-trip to sample-indexed
  cue points.
- **Units.** Integrated/short-term loudness in **LUFS**; true-peak in **dBTP**;
  sample-peak / gain in **dBFS**; frequency in **Hz**; pitch deviation in
  **cents**. A `feature` value's unit is implied by its namespaced key
  (Essentia namespace, see *Standards*); ad-hoc keys MUST suffix the unit
  (`_db`, `_hz`, `_lufs`).
- **Encoding.** UTF-8 throughout.

## Inline payloads & size limits — NORMATIVE

- A `data` payload MUST NOT exceed **64 KiB** serialized.
- `text`, `feature`, `marker`, `control` normally inline; if one exceeds the
  limit it MUST move to CAS (`application/json` or `text/plain`) referenced by
  `hash`. This bounds NDJSON line length and makes any large payload
  content-addressable (so it can be a memo input by hash).

## Error model — NORMATIVE

"One frame, one failure" keeps a pipe resilient, but a failure MUST be
distinguishable from absence downstream.

- A failed op on one frame emits an `error` frame and continues; only
  fatal/usage errors exit non-zero.
- `error.data` MUST be `{code, message, of}` and SHOULD include `op`. `code` is
  drawn from a standard enum so downstream logic isn't string-matching
  `message`: `decode_failed`, `op_failed`, `resource_exhausted`, `unsupported`,
  `id_collision`, `not_found`.
- **Propagation.** A consumer asked to resolve/select a role that has no
  payload frame but **does** have a matching `error` frame (or whose ancestor
  failed, i.e. `error.of` is an ancestor of the requested role) MUST surface
  that root-cause error and exit non-zero — not report a generic "not found".
  (So a CUDA-OOM in `smpl stems` reaches the user, instead of a downstream
  "≠1 resolvable audio frame".)
- **Pipe hygiene.** Tools MUST handle `SIGPIPE` cleanly and MUST NOT emit a
  truncated final NDJSON line; a partial line is a fatal read error downstream.

## Hybrid raw mode (the sox/ffmpeg bridge)

The frame stream is the default contract, but any single-audio-stream stage
can drop to raw WAV to interoperate with the Unix DSP world:

```bash
smpl read x.wav | smpl select --role stem:drums | smpl as-wav \
  | sox - -t wav - reverb 50 | ffmpeg -i - -af acompressor -f wav - \
  | smpl from-wav --role drums.wet --derives-from stem:drums | smpl describe
```

- `smpl as-wav` — resolve the (last-wins single) selected audio frame to a raw
  WAV byte stream on stdout. Default output is **float32 WAV at the frame's
  native rate and channel count** (no silent bit-depth/rate truncation);
  `--format` overrides. Errors if the stream has no resolvable audio frame.
- `smpl from-wav` — read raw WAV from stdin, CAS it under a fresh
  canonical-PCM hash, emit an `audio` frame. `--derives-from <role|id>`
  reattaches **lineage** so the sox detour doesn't break provenance.

**On memoization across the bridge:** the bytes returning from sox/ffmpeg are
*new* audio (reverb changed them), so they get a new content hash — correctly.
`from-wav` records `op: from-wav` with the **input hash** in `params`, so a
re-run of the *identical* external pipe over the same input is memoizable going
forward. It does **not** dedup against the pre-detour audio (an opaque external
op can't be proven equal). Lineage survives; the cache hit is on the
`from-wav` op, not on the original frame.

Use raw mode only to reach an external effect; stay in frames otherwise.

## Tool contract (every `--stream` tool MUST obey)

1. **Read frames from stdin, write frames to stdout.** A `--stream` flag (or
   reading non-tty stdin) selects frame mode; path-in/path-out stays the
   default so nothing existing breaks.
2. **Passthrough.** Emit every input frame you did not consume, unchanged
   (preserve unknown fields), then your derived frames. The tail of a pipe
   sees the *whole* lineage — original + stems + filtered subcomponent.
3. **Lineage.** Set `of` / `lineage` / `op` / `op_version` / `params` on
   derived frames. A tool that **consumes** an input frame (does not pass it
   through, e.g. a generator eating a `prompt` frame) MUST either retain it with
   `consumed: true` so its id stays resolvable, or copy its defining content
   into the deriving frame's `params` — so `lineage` never dangles. Dangling
   `of`/`lineage` (an id absent from the stream and not CAS-resolvable) is a
   conformance error (the suite checks **lineage closure**).
4. **One frame, one failure.** See *Error model*.
5. **Stderr is for humans, stdout is for frames.** Never write logs to stdout.
6. **Determinism.** Same inputs + `op_version` + params ⇒ same `memo_key` ⇒
   same output, unless the op declares `cacheable: false`.
7. **`--json` legacy.** Existing `--json` (single record) stays for batch
   callers; `--stream` is the composable mode. They may coexist.

### Role naming conventions

`source` · `prompt` · `stem:<name>`
(`stem:drums|bass|vocals|other|guitar|piano` — the 6-stem htdemucs set;
`stem:instrumental` — the 2-stem BS-RoFormer set's non-vocal half) ·
`slice:<n>` · `<name>.wet` / `.dry` · `caption` · `lyrics` ·
`spectrogram[:mel|cqt|hpss]` · `waveform` · `onset` / `beat` / `section`.
Colon-namespaced, kebab inside segments. Roles are NOT guaranteed unique within
a stream — see *Selection*.

## Source tools (generators)

Most tools are filters (consume frames, emit frames). **Generators**
(`smpl gen` local backends, `smpl cloud` provider APIs) are *sources*: they
produce an `audio` frame from a text prompt rather than from an upstream audio
frame. They MUST still be smplstream citizens so they compose —
`… | smpl gen … | smpl cat` and `… | smpl cloud … | smpl cat` just work.

**Prompt input — three equivalent forms (a generator MUST accept all three):**

1. `--prompt "a 4/4 distorted drum loop"` — the explicit flag.
2. **Raw text on stdin** — `--prompt -` (or `--prompt --`) reads the prompt
   from stdin as plain text: `echo 'a 4/4 distorted drum loop' | smpl gen --backend clap --prompt -`.
3. **A `text` frame with `role: prompt`** on stdin — when stdin is a frame
   stream, a `text`/`prompt` frame supplies the prompt and is consumed (and
   retained with `consumed: true` per the tool contract). This is what lets a
   caption flow into a regenerate: `smpl cat | … | smpl gen`.

Disambiguation: if stdin is NDJSON frames, parse as frames (form 3); if
`--prompt -`/`--prompt --` is given, read stdin as raw text (form 2); the
explicit flag (form 1) always wins. A generator with no resolvable prompt is a
usage error (non-zero exit).

**Output:** an `audio` frame with `op: gen`/`op: cloud`, `params` capturing the
backend/provider, model, seed, and the resolved prompt, and `lineage` pointing
at the prompt frame when one was consumed (so provenance survives). Any
non-prompt input frames pass through unchanged.

Backends (local models) and provider keys are an **install/config** concern,
not a wire-protocol one — see the [README](README.md) for model management and env-var-first key handling.

## Interop seams (build for these explicitly)

| Boundary | Mechanism | Example |
|---|---|---|
| Query/transform stream | NDJSON → `jq`/`fx`/`mlr` (per-line only) | `… \| jq 'select(.kind=="text").data'` |
| Byte-level DSP | `as-wav`/`from-wav` ↔ `sox`/`ffmpeg` | reverb/compress detour |
| Hand a blob to anything | `smpl resolve <id\|hash>` → path | feed Demucs, a VST renderer, Audacity |
| Octatrack slices | `marker` (with `sample`) → `ot-slicer` | slice grid → `.ot` |
| DAW / editors | `marker` → Audacity labels, `.cue` | session markers |
| Library search | `vector` frames → `music-rig search --semantic` | similarity |
| Library index | `provenance`/`feature` → `music-rig index` | drain pipe into canonical store |
| Existing sidecars | CAS path ↔ `<file>.analysis.json` | coexist with current world |
| External synthesis/DSP | `smpl resolve`→path / raw-WAV bridge ↔ `sclang`/`scsynth` **NRT** | SuperCollider SynthDef render or effect (offline) |
| MIDI score I/O | `kind:midi` ↔ `basic-pitch` (audio→MIDI), `fluidsynth`/SC NRT (MIDI→audio) | transcribe a sample to notes; render a score |

## Standards alignment (don't invent what exists)

Pro interop comes from reusing established vocabularies. See
the project's research notes for the full rationale. **Maturity is marked** so an
implementer can tell a v1 obligation from an aspiration.

- **[v1] `feature` frame namespacing** — objective MIR features use the
  **Essentia / AcousticBrainz** namespaces (`lowlevel.*`, `rhythm.*`,
  `tonal.*`) with the `{mean, stdev}` statistic convention for
  frame-aggregated values. The concrete key set lands incrementally with each
  analysis op; the registry file `feature-keys.md` (versioned with `SCHEMA`) is
  the source of truth. Perceptual / LLM-facing descriptors use the
  **AudioCommons** 8 timbral names (`hardness`, `depth`, `brightness`,
  `roughness`, `warmth`, `sharpness`, `boominess`, `reverb`; 0–100).
- **[v1] `vector` frame** — tag with producing `model` + version
  (`mert-v1-330m`, `clap`, `chromaprint`) and `dtype`. Large vectors live in
  CAS as `.npy`/`.safetensors`/Arrow (**never pickle** — code-exec on load;
  opt-in import only). See the size-split rule under *Frame kinds*.
- **[v1, export-side SHOULD] `marker` ↔ embedded chunks** — markers round-trip
  to the WAV **`cue`** chunk (slice/transient points, sample-indexed) and
  **`acid`** chunk (tempo, key, beats, time-sig, one-shot/loop); loop markers
  to the **`smpl`** chunk (root note + loop points). Tools that *write audio*
  SHOULD write these so a sample works in any sampler/DAW without smplstream
  present. This is an export obligation, not a wire-protocol one.
- **[future] Provenance** — derived-artifact lineage will serialize to the
  **W3C PROV** shape (`derived_from` + Chromaprint fingerprint, `generated_by`
  tool+version+params, `agent`, `created`, `license`), with an optional
  **C2PA** manifest for AI-generated assets. v1 carries the *structured form*
  on every frame (`lineage`/`op`/`op_version`/`params`); full PROV/C2PA
  serialization and `.asd`/`.ot` *import* are not v1-binding.

## Versioning & evolution — NORMATIVE

- `v` is **per-frame** and bumps **only** on changes that reinterpret existing
  fields (breaking). Additive new `kind`s, new optional fields, and new
  `feature` keys do **NOT** bump `v`.
- A tool MUST **pass through** (unchanged, preserving unknown fields) any frame
  whose `v` is **≤** its own max supported `v` — even kinds/fields it doesn't
  recognize (lineage to it is just an opaque id, so passthrough is safe). A
  tool MUST **reject** (fatal) only a frame whose `v` is **greater** than it
  supports (it can't safely reason about reinterpreted fields). This lets v1
  and v2 tools coexist in one pipe for additive changes, and fail-closed only
  on a true future-breaking frame.
- The `SCHEMA` constant in the `smplstream` package gates cache invalidation at
  the protocol-schema level, mirroring `smplcat`'s `SCHEMA_VERSION` discipline
  (per-op correctness rides on `op_version`, not `SCHEMA`).

## Out of scope (v1) / deferred

- **`control` frame schema** — the kind and namespace are reserved now; v1
  defines no keys. Unknown `control` frames pass through and are ignored by
  tools that don't handle them, so adding keys later is non-breaking.
- **Empty / degenerate streams** — an empty stdin (or a stream of only
  `control` frames) is valid: tools emit nothing relevant and exit 0. Exact
  per-tool behavior is nailed down at implementation time.
- **Cross-machine CAS sync** — the mini batch tiers stay path-based for now.
  Note: *Canonical PCM* + `op_version`-in-key are precisely what make
  cross-machine cache *portability* a future v1.1 feature rather than a v2
  protocol break — getting them right now keeps that door open.
- **Real-time/low-latency streaming** — this is offline analysis, not a live
  DSP graph. Windowed/chunked audio frames are a future extension, not v1.
- **Realtime MIDI / OSC I/O and live-coding** — live `scsynth`, hardware
  controllers, sequencer/transport sync are the live/**daemon** world (a stream
  of timed events, not content-addressed frames) and stay out of the offline
  protocol. The `midi` kind is for offline scores/arrangements only. SuperCollider
  fits as an **NRT** renderer (a pure offline op), not as a live server here.
- **Bidirectional control** — tools are forward filters; no upstream
  backpressure protocol beyond ordinary pipe semantics.
