---
name: smpl-dissect
description: Dissection microscope for audio samples, built on the smpl pipe toolchain. Use when the user wants to isolate and richly describe a subcomponent of a sample ("what's the texture of just the drum bus?", "isolate the bass and describe it", "slice this and tell me about hit 3"), or to understand what is inside an audio file beyond a whole-file summary. Composes smpl pipes (read → stems/slice/select/filter/env → describe → view) and interprets the multimodal report. Requires the smpl CLI on PATH.
---

# smpl-dissect: the sample dissection microscope

You compose **smpl pipes** to isolate any *subcomponent* of an audio
sample, then interpret the resulting multimodal frames. The toolchain
does the measuring; **your job is to interpret**, not to produce
numbers. If a value matters, it comes from a `feature` frame, never
from your own estimate.

Check the toolchain first: `smpl --version`. If it is missing, point
the user at <https://github.com/algonormative/smpl#install> instead of
degrading silently.

## The frame model (what flows through the pipe)

Each stage passes through audio *and* its accumulating metadata as
NDJSON frames referencing content-addressed bytes:

- `audio`: the substrate (in the CAS, referenced by `hash`)
- `feature`: structured measurements (LUFS, spectral shape, QC)
- `marker`: timestamped tracks (onsets, slices, defect locations)
- `image`: spectrograms (mel / CQT / HPSS) and waveforms, as PNGs
- `vector`: embeddings for similarity
- `text`: captions and reports

Never pipe heavy bytes; pipe frames. `smpl resolve <id|hash>` hands
any blob to an external tool. `smpl as-wav | … | smpl from-wav`
splices sox/ffmpeg into the middle of a pipe without losing lineage.

## Compose the right pipe for the intent

Resolve lazily: only materialize what the intent needs. Start narrow,
widen on demand.

| Intent | Pipe |
|---|---|
| describe the whole sample | `smpl read X \| smpl describe-all \| smpl view` |
| isolate + describe a stem | `smpl read X \| smpl stems \| smpl select --role stem:drums \| smpl cat \| smpl view` |
| a filtered subcomponent | `smpl read X \| smpl filter --hp 200 \| smpl env --pluck \| smpl cat \| smpl view` |
| slice + describe one hit | `smpl read X \| smpl slice --emit-audio \| smpl select --role slice:3 \| smpl cat \| smpl view` |
| spectral character | `smpl read X \| smpl spectral \| smpl spectrogram --kind cqt \| smpl view` |
| reach a tool smpl lacks | `smpl read X \| smpl as-wav \| sox - -t wav - <effect> \| smpl from-wav --role x.wet --derives-from source \| smpl cat \| smpl view` |

Heavy stages (`stems`, `transcribe`, `embed`, `gen`, `synth`) are
separate installs discovered on PATH. If one is not installed, the
dispatcher stops and names the missing tool on stderr; give the user
the install command from the repo README rather than working around
it silently. An installed tool missing its optional heavy dependency
emits an `unsupported` error frame with a hint; relay that hint.

## Read the report like an instrument panel

`smpl view` emits a markdown report plus the underlying frames.
Interpret it specifically:

- Open `image` frames (`smpl resolve <hash>` gives the PNG path) and
  describe what you actually see: where energy sits, transient
  density, a brickwall cutoff, stereo width.
- Quote `feature` values **with units** (LUFS, dBTP, Hz, ms) and say
  what each implies for the user's intent. The feature-key registry
  in the repo (`feature-keys.md`) defines every key's unit.
- Tie `marker` times to musical events ("the third slice lands on the
  offbeat at 1.24 s").

## Persisting what you isolated

- Report-only is the default; the pipe writes nothing outside its CAS.
- `… | smpl write out.wav` materializes the isolated subcomponent.
- Marker frames can become editor labels or cue sheets on request.
