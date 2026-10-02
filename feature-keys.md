---
title: "smplstream feature-key registry"
created: 2026-06-14
tags: [music, audio, smplstream, analysis, reference, interop]
description: "The canonical key names, namespaces, and units for smplstream `feature` frames. Shared source of truth so analysis tools don't mint divergent spellings. Versioned with SCHEMA."
---

# smplstream feature-key registry

**The source of truth for `feature`-frame key names** ([spec.md](spec.md) →
*Standards alignment*). Every analysis tool that emits a `feature` frame MUST
use a key registered here; adding a measurement means **adding a row here
first**, then emitting it. This is what stops six tickets from independently
spelling spectral flatness three different ways.

Versioned with the `SCHEMA` constant (a key rename/removal is a `SCHEMA` bump;
adding a row is additive and non-breaking).

## Conventions (from spec → Units & timebase / Standards)

- **Objective MIR features** use the **Essentia / AcousticBrainz** namespaces
  (`lowlevel.*`, `rhythm.*`, `tonal.*`); the unit is implied by the namespaced
  key, and frame-aggregated values carry the `{mean, stdev}` statistic shape.
- **Perceptual / LLM-facing** descriptors use the **AudioCommons** `timbre.*`
  prefix — deliberately **outside** the Essentia namespace (they're perceptual,
  not signal-objective).
- **Domain keys with no Essentia equivalent** (loudness, QC) use a short prefix
  (`loudness.*`, `qc.*`) and, per the spec, **MUST suffix the unit**
  (`_lufs`, `_dbtp`, `_db`, `_dbfs`, `_hz`) so a bare number is never ambiguous.

## Registry

| Key | Namespace | Unit | Stat | Owner |
|---|---|---|---|---|
| `loudness.integrated_lufs` | loudness (unit-suffixed) | LUFS | scalar | vault-3vau |
| `loudness.true_peak_dbtp` | loudness | dBTP | scalar | **vault-3vau (sole owner; QC reuses)** |
| `loudness.max_short_term_lufs` | loudness | LUFS | scalar | vault-3vau |
| `loudness.lra` | loudness | LU | scalar | (P2 dynamics) |
| `lowlevel.spectral_flatness_db` | Essentia | dB (NOT 0–1 ratio) | {mean,stdev} | vault-3uap |
| `lowlevel.spectral_crest` | Essentia | unitless | {mean,stdev} | vault-3uap |
| `lowlevel.spectral_spread` | Essentia | Hz | {mean,stdev} | vault-3uap |
| `lowlevel.spectral_rolloff` | Essentia | Hz | {mean,stdev} | vault-3uap |
| `lowlevel.spectral_contrast` | Essentia | dB | {mean,stdev} | vault-3uap |
| `lowlevel.spectral_slope` | Essentia | unitless | {mean,stdev} | vault-3uap |
| `lowlevel.spectral_skewness` | Essentia | unitless | {mean,stdev} | vault-3uap |
| `lowlevel.spectral_kurtosis` | Essentia | unitless | {mean,stdev} | vault-3uap |
| `qc.clipping.detected` | qc | bool | scalar | vault-1e9a |
| `qc.phase.correlation` | qc | unitless (−1..1) | scalar | vault-1e9a |
| `qc.dc_offset_dbfs` | qc | dBFS | scalar | vault-1e9a |
| `qc.snr_db` | qc | dB | scalar | vault-1e9a |
| `qc.lossy.spectral_cutoff_hz` | qc | Hz | scalar | vault-1e9a |
| `qc.lossy.expected_nyquist_hz` | qc | Hz | scalar | vault-1e9a |
| `qc.lossy.confidence` | qc | 0–1 | scalar | vault-1e9a |
| `timbre.hardness` | AudioCommons (perceptual) | 0–100 | scalar | vault-14ia |
| `timbre.depth` | AudioCommons | 0–100 | scalar | vault-14ia |
| `timbre.brightness` | AudioCommons | 0–100 | scalar | vault-14ia |
| `timbre.roughness` | AudioCommons | 0–100 | scalar | vault-14ia |
| `timbre.warmth` | AudioCommons | 0–100 | scalar | vault-14ia |
| `timbre.sharpness` | AudioCommons | 0–100 | scalar | vault-14ia |
| `timbre.boominess` | AudioCommons | 0–100 | scalar | vault-14ia |
| `timbre.reverb` | AudioCommons | binary (0/1) | scalar | vault-14ia |
| `rhythm.bpm` | Essentia | BPM | scalar | vault-32n3 |
| `rhythm.bpm_confidence` | Essentia | 0–1 | scalar | vault-32n3 |
| `rhythm.bpm_candidates` | Essentia | BPM[] | list | vault-32n3 |
| `rhythm.time_signature` | Essentia | n/d string | scalar | vault-32n3 |
| `rhythm.swing` | Essentia | 0–1 (fraction of a grid step the even steps are delayed) | scalar | vault-6m62 |
| `rhythm.swing_confidence` | Essentia | 0–1 | scalar | vault-6m62 |
| `rhythm.microtiming_beats` | Essentia | beats (mean abs. residual nudge after swing) | scalar | vault-6m62 |
| `tonal.key_key` | Essentia | pitch class | scalar | vault-379o |
| `tonal.key_scale` | Essentia | major/minor | scalar | vault-379o |
| `tonal.tuning_frequency` | Essentia | Hz | scalar | vault-379o |
| `fingerprint.chromaprint` | fingerprint | id (int-array/base64) | scalar | vault-2xro |
| `envelope.peak_db_over_floor` | envelope (unit-suffixed) | dB | scalar | vault-3tuy |
| `envelope.attack_ms_10_90` | envelope | ms | scalar | vault-3tuy |
| `envelope.rise_slope_db_ms` | envelope | dB/ms | scalar | vault-3tuy |
| `envelope.t20_ms` | envelope | ms (None if never −20 dB) | scalar | vault-3tuy |
| `envelope.early_decay_slope` | envelope | dB/ms | scalar | vault-3tuy |
| `envelope.sustain_ratio_150ms` | envelope | ratio (0–1+) | scalar | vault-3tuy |
| `width.sub.correlation` | width | unitless (−1..1) | scalar | vault-3tuy |
| `width.sub.side_mid_ratio` | width | ratio (0=mono) | scalar | vault-3tuy |
| `width.bass.correlation` | width | unitless (−1..1) | scalar | vault-3tuy |
| `width.bass.side_mid_ratio` | width | ratio (0=mono) | scalar | vault-3tuy |
| `width.lomid.correlation` | width | unitless (−1..1) | scalar | vault-3tuy |
| `width.lomid.side_mid_ratio` | width | ratio (0=mono) | scalar | vault-3tuy |
| `width.mid.correlation` | width | unitless (−1..1) | scalar | vault-3tuy |
| `width.mid.side_mid_ratio` | width | ratio (0=mono) | scalar | vault-3tuy |
| `width.uppermid.correlation` | width | unitless (−1..1) | scalar | vault-3tuy |
| `width.uppermid.side_mid_ratio` | width | ratio (0=mono) | scalar | vault-3tuy |
| `width.air.correlation` | width | unitless (−1..1) | scalar | vault-3tuy |
| `width.air.side_mid_ratio` | width | ratio (0=mono) | scalar | vault-3tuy |
| `width.full.correlation` | width | unitless (−1..1) | scalar | vault-3tuy |
| `width.full.side_mid_ratio` | width | ratio (0=mono) | scalar | vault-3tuy |
| `movement.sidechain_db` | movement (unit-suffixed) | dB | scalar (None if gated) | vault-1fxy |
| `movement.bass_mod_depth_db` | movement | dB | scalar (None if gated) | vault-1fxy |
| `movement.hf_mod_depth_db` | movement | dB | scalar (None if gated) | vault-1fxy |
| `movement.hf_silence_pct` | movement | % (0–100) | scalar (None if gated) | vault-1fxy |
| `movement.tail_decay_200ms_db` | movement | dB | scalar (None if gated) | vault-1fxy |
| `clarity.low_mid_masking_db` | clarity (unit-suffixed) | dB | scalar (None if gated) | vault-1fxy |
| `clarity.mud_presence_ratio` | clarity | ratio (>1 = mud-heavy) | scalar (None if gated) | vault-1fxy |
| `clarity.band_contrast_lo_db` | clarity | dB | scalar (None if gated) | vault-1fxy |
| `clarity.band_contrast_hi_db` | clarity | dB | scalar (None if gated) | vault-1fxy |
| `clarity.presence_focus_ratio` | clarity | ratio (0–1) | scalar (None if gated) | vault-1fxy |
| `clarity.presence_transient_ratio` | clarity | ratio (crest) | scalar (None if gated) | vault-1fxy |
| `space.mono_collapse_penalty_db` | space (unit-suffixed) | dB | scalar | vault-1fxy |
| `spectrum.oct6.<hz>` | octave-spectrum | dB (rel. total in-band power, ≤ 0) | scalar per 1/6-octave bin | vault-22oy |
| `spectrum.band.<name>` | octave-spectrum | dB (rel. total in-band power) | scalar per standardized band | vault-22oy |

## Standardized band edges (vault-3tuy)

The `width.*` per-band keys (and any future band-split feature) use these six
**standardized band edges** so every band-aware tool splits the spectrum the same
way. `full` (20 Hz–Nyquist) is the broadband reference kept alongside the split so a
wide-sub mono-collapse a single broadband number would hide stays visible.

| Band | Key stem | Edges (Hz) |
|---|---|---|
| Sub | `width.sub` | 20 – 60 |
| Bass | `width.bass` | 60 – 200 |
| LoMid | `width.lomid` | 200 – 500 |
| Mid | `width.mid` | 500 – 2 000 |
| UpperMid | `width.uppermid` | 2 000 – 6 000 |
| Air | `width.air` | 6 000 – 20 000 |
| (full-band ref) | `width.full` | 20 – 20 000 |

The `movement.*` and `clarity.*` families **reuse these same edges** (they do not
redefine them): movement's bass-modulation read uses the Bass band (60–200), its
HF reads use the Air lower edge (≥6 kHz); clarity's band powers use all six bands,
with "presence" = the UpperMid band (2–6 kHz). The `spectrum.band.*` levels
(octave-spectrum, vault-22oy) also reuse the six edges; its `spectrum.oct6.*` bins are a
finer 1/6-octave split of the same 20 Hz–20 kHz range.

## Duration / role gating (vault-1fxy)

The `movement.*` and `clarity.*` families measure **time-variation and
mix-interaction** — pump depth, per-band modulation, tail decay, mud/presence
balance, masking. On one-shots and very short fragments these are **degenerate**
(a single hit has no modulation cycle; a fragment has no stable spectrum). So both
families are **duration-gated**, following the LUFS-integrated precedent
(`loudness.integrated_lufs` returns `None` below the 0.4 s block it needs): material
shorter than the family's minimum emits **every key as `None`** rather than a
misleading number.

| Family | Gate | Below the gate |
|---|---|---|
| `movement.*` | `dur ≥ 1.0 s` | all five keys emit `None` |
| `clarity.*` | `dur ≥ 0.5 s` | all six keys emit `None` |
| `space.mono_collapse_penalty_db` | **none** | a static stereo relationship is well-defined on any length (same as `width.*`) |

The gate decision is recorded on the emitted frame (`params.gated`, `params.dur_s`,
`params.min_duration_s`), so a consumer can tell "null because gated" from "null
because the measurement failed". Per the layering rule, the gate is a
*support-of-measurement* threshold, not a quality opinion — what counts as "enough
pump" or "too muddy" still lives in profiles / docs, never in the tool.

## Collection-stats frames (vault-3tuy)

`smpl stats build` emits a `feature` frame (role `stats:<role>`) that reuses these
same registered keys, but each value is a **distribution** —
`{median, MAD, p10, p50, p90, n, domain}` reduced over a role corpus in the log
domain — rather than a per-sample scalar. It mints **no new key spellings**: a
role-stats frame is the registered keys carrying corpus statistics.

## Ownership notes (avoid double-emission)

- **True-peak overs** (`loudness.true_peak_dbtp` + over-location markers) are
  owned by **vault-3vau** (loudness tier). The QC ticket (vault-1e9a) **reuses**
  that frame for its clipping pass/fail rather than recomputing under a `qc.*`
  key. One measurement, one owner.
- **Groove** (`rhythm.swing`, `rhythm.swing_confidence`,
  `rhythm.microtiming_beats`; vault-6m62, `smpl groove`) is expressed in
  **`smpl pattern`'s** timing model — swing delays every *even* grid step by
  `swing × stepGap` beats, and the per-step `nudge` residual is in beats — so the
  measurement round-trips into a generated loop with no unit conversion. `smpl
  groove` reports the tempo it measured against on the frame's `params.bpm`, **not**
  as `rhythm.bpm`: that key stays owned by the beat grid (vault-32n3). One
  measurement, one owner.
- `timbre.*` (AudioCommons, perceptual) is intentionally separate from
  `lowlevel.*` (Essentia, objective) even where they sound similar
  (`timbre.sharpness` ≠ any `lowlevel.*` — the former is the perceptual 0–100
  descriptor, MoSQITo/AudioCommons-derived).

## Status

The Essentia-namespaced rows (`lowlevel.*`, `rhythm.*`, `tonal.*`) are
**provisional** until the Essentia-vs-lean-stack spike (vault-tkih) resolves
whether Essentia ships on macOS/ARM or those features come from
librosa/MoSQITo/pyloudnorm. The spike's acceptance includes finalizing these
spellings here. `loudness.*`, `qc.*`, `timbre.*`, `fingerprint.*` do not depend
on Essentia and are stable.
