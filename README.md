<div align="center">

# smpl

**A composable, content-addressed audio-analysis toolchain you pipe like `jq`.**

*Pipe self-describing **frames** that reference content-addressed bytes — never the
heavy bytes themselves. That one choice buys multi-payload streams, a normative
memoization contract, and lazy evaluation.*

[Site](https://algonormative.github.io/smpl/) · [Quick start](#quick-start) · [Protocol](#the-wire-protocol) · [Install](#install) · [Pipes](#pipes) · [For LLMs](#built-for-llms) · [Skills](#install-the-agent-skills) · [Tools](#the-tools)

</div>

---

```bash
smpl read loop.wav | smpl loudness | smpl view > /dev/null
```

That core-only pipe measures a sample and prints a readable report. Every stage is a
boring Unix citizen: NDJSON on the wire (so `jq` works), real file paths for the heavy
bytes (so `sox` and `ffmpeg` work), content-addressed and memoized (`loudness`, `spectral`,
`qc`, and `spectrogram` consult the memo cache before computing; `--no-cache` forces a
recompute).

## Quick start

This first run installs only the light core, downloads the same four-second loop shipped
in [`docs/assets/loop.wav`](docs/assets/loop.wav), and measures it. It needs no model
download, heavy subcommand, provider account, or API key.

1. Install [`uv`](https://docs.astral.sh/uv/) if needed, then install the core in one
   isolated environment:

   ```bash
   uv tool install git+https://github.com/algonormative/smpl#subdirectory=packages/smpl \
     --with git+https://github.com/algonormative/smpl#subdirectory=packages/smplstream \
     --with git+https://github.com/algonormative/smpl#subdirectory=packages/smpl-analysis
   ```

2. Download the shipped demo loop:

   ```bash
   curl -LO https://algonormative.github.io/smpl/assets/loop.wav
   ```

3. Run the core-only pipe. `view` prints its readable report to the terminal; the
   redirect hides the NDJSON frames kept on stdout for further piping:

   ```bash
   smpl read loop.wav | smpl loudness | smpl view > /dev/null
   ```

Expected output: the Markdown feature table includes integrated loudness around
`-20.79 LUFS` and true peak around `-6.71 dBTP`:

```text
| `loudness.integrated_lufs` | -20.79 | LUFS | loudness | loudness |
| `loudness.true_peak_dbtp` | -6.71 | dBTP | loudness | loudness |
```

Small last-decimal differences across platforms are fine. Seeing those two measured rows
means the install and first pipe worked.

## Why

Most audio CLIs are path-in / path-out batch tools. `smpl` makes the **stream** the
interface: each stage passes through the audio *and* its accumulating metadata, so the
tail of a pipe sees the whole lineage — original → stems → filtered subcomponent — and
any tool (or an LLM) can dissect and describe a piece of it.

- **Composable.** One frame per line of NDJSON. `… | jq 'select(.kind=="feature")'` just works.
- **Content-addressed.** Heavy bytes live in a CAS keyed by the **canonical decoded PCM**,
  so two identical stems share one blob across machines and re-encodes.
- **Built to memoize.** Every cacheable op is a pure function of its inputs, version, and
  environment; the spec defines the memo key and the CAS dedups identical results.
  (Subcommand cache lookups are tracked work — today re-runs recompute.)
- **Hybrid raw mode.** `smpl as-wav | sox … | smpl from-wav` splices the entire Unix DSP
  world into the middle of a pipe without losing lineage.

## The wire protocol

One frame per line, UTF-8 NDJSON. A frame is self-describing and references its payload by
`hash` (in the CAS) or carries it inline as `data`:

```jsonc
{"v":1,"kind":"audio","id":"blake3:9af2…","role":"stem:drums",
 "of":"blake3:1c0a…","op":"demucs","op_version":"audio-separator@0.28+htdemucs:blake3:…",
 "hash":"blake3:c3d4…","media":"audio/wav","meta":{"sr":48000,"dur":8.0,"ch":2}}
```

Kinds: `audio` · `image` (spectrograms/waveforms) · `text` (captions/lyrics) · `vector`
(embeddings) · `marker` (beats/onsets/slices/defects) · `feature` (LUFS/key/QC…) ·
`midi` · `error`. The full normative contract — canonical-PCM hashing, the memo key, CAS
integrity, units & timebase — is in [`spec.md`](spec.md). It is versioned like an API.

## Install

```bash
# the light core (smplstream + smpl-analysis + smpl) — one isolated install
uv tool install git+https://github.com/algonormative/smpl#subdirectory=packages/smpl \
  --with git+https://github.com/algonormative/smpl#subdirectory=packages/smplstream \
  --with git+https://github.com/algonormative/smpl#subdirectory=packages/smpl-analysis

# heavy tools install separately, into their OWN isolated venvs (two-tier):
uv tool install git+https://github.com/algonormative/smpl#subdirectory=tools/smpl-stems
```

`ffmpeg` and `sox` on PATH unlock the raw-WAV bridge and `convert`. The core cold-starts
fast (no torch/librosa on the dispatch path); heavy deps load lazily, per subcommand.

## Pipes

```bash
# Describe a whole sample (passthrough + features + caption + spectrogram)
smpl read pad.wav | smpl describe | smpl view

# Loudness / mastering read
smpl read master.wav | smpl loudness | jq 'select(.kind=="feature").data'

# Level to a LUFS target with a -1 dBTP true-peak ceiling (kit / master prep)
smpl read hot.wav | smpl normalize --lufs -14 | smpl write leveled.wav

# Technical QC + forensics (a lossy origin shows as a brickwall low-pass)
smpl read suspect.wav | smpl qc | smpl spectrogram | smpl view

# Splice the Unix DSP world into the middle of a pipe
smpl read x.wav | smpl as-wav | sox - -t wav - reverb 50 \
  | smpl from-wav --role x.wet --derives-from source | smpl describe

# Generate from a prompt (a source tool) and analyze it — prompt via stdin
echo 'a 4/4 distorted drum loop' | smpl gen --backend synth --prompt - | smpl cat

# Author an explicit drum loop from a step-grid DSL → a smplmix session, then render
smpl pattern --pattern-file loop.json --out loop.smplset.json   # per-hit velocity / pitch / swing
smplmix render loop.smplset.json -o loop.wav                     # composition runs on smplmix
```

`smpl pattern` is the LLM-friendly way to write a drum loop: a 16-step (or N-step) grid,
per-track `steps`/`hits` with **velocity** (→ gain), **pitch** (semitones → transpose),
**swing** (even-step shuffle) and per-hit **nudge** — expanded into a ready-to-render
smplmix session. See `smpl pattern --help` for the full DSL.

## Built for LLMs

`smpl view` is the payoff: a multimodal report for whatever subcomponent you isolated —
feature tables with units, `marker` tracks tied to musical time, and **actual spectrogram
images** an LLM can open and describe. The deterministic tier does the *measuring*; the
model's job is to *interpret*. (*If it doesn't need reasoning, it shouldn't call a model.*)

The companion [`smpl-dissect`](skills/smpl-dissect/SKILL.md) skill drives these pipes
for you: give it a sample and an intent ("isolate the bass and describe its texture") and it
composes the pipe, resolves only what's needed, and reads back the report.

## The tools

| Command | Does |
|---|---|
| `read` / `write` | ingest audio → frames; materialize a selected frame → file |
| `resolve` / `gc` | hash/id/role → CAS path; collect unreferenced blobs |
| `as-wav` / `from-wav` | the raw-WAV bridge to `sox`/`ffmpeg` (lineage-preserving) |
| `cat` / `describe` / `describe-all` | describe-as-filter: passthrough + features + caption + image; `-all` aggregates the whole light tier |
| `loudness` | integrated LUFS, true-peak dBTP, short-term LUFS |
| `spectral` | spectral-shape family (flatness/crest/spread/rolloff/contrast/slope) |
| `chords` | chord-span markers (`{t, dur, label, sample}`) + key/tuning feature |
| `qc` | clipping, phase/mono, DC, SNR, clicks/gaps, lossy-origin cutoff |
| `spectrogram` | annotated mel / CQT / HPSS spectrograms + waveform (PNG) |
| `convert` | format / sample-rate / bit-depth conversion (new frame, own hash) |
| `gain` `normalize` `limit` | level management: dB gain (pure), LUFS-normalize (+ true-peak ceiling), true-peak limit |
| `maximize` `compress` `sidechain` | look-ahead brickwall limiting (drive + cap); downward compression; duck a target under a trigger's transients (kick-vs-sub glue) |
| `filter` `eq` `env` `fx` `slice` `select` | the edit filters + stream selection |
| `automate` `variants` `stereoize` `widen` `spectral-match` | parameter motion over time; N baked closed→open filter variants of one source; mono→wide; M-S width; EQ toward a reference |
| `pattern` | step-grid drum-loop DSL → smplmix session (velocity / pitch / swing / nudge) |
| `view` | the multimodal LLM/human report |
| `gen` · `cloud` · `transcribe` · `stems` · `embed` · `synth` | PATH-discovered heavy tools (own venvs) |

## Install the agent skills

Two agent skills ship in `skills/`, installable with the open-source `skills` CLI
so Codex and Claude Code share one managed copy:

```bash
npx skills add algonormative/smpl --global --agent codex claude-code --yes
```

- **`smpl-dissect`** — isolate a stem, slice, or filtered band and describe exactly
  that piece, numbers cited with units and the spectrogram in front of the model.
- **`smpl-audit`** — the measured bounce check: loudness, true peak, clipping, DC,
  noise, and lossy-origin forensics reported as a verdict.

Both expect the `smpl` CLI on PATH and say so when it is missing.

## Architecture

One `uv` workspace holds the light core; heavy generators are separate `uv tool install`'d
projects discovered on PATH (`smpl gen` execs `smpl-gen`) — so torch never touches the core
lockfile and cold pipe stages stay fast. Optional Rust DSP rides in via pyo3/maturin only
where profiling earns it.

## Development

```bash
uv sync && uv run pytest packages        # build + test the workspace
bash tests/e2e/run_e2e.sh                # end-to-end: clean install + pipes + two-tier
```

## License

MIT.
