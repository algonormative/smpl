# `smpl ask` + `smpl crosscheck`: an audio LLM as a second ear

Audio-native models are weakest on music (MUSE, AudioCapBench) and biased by presentation order
(AudioJudge), and Gemini hears a mono 16 kHz downmix. So the model is a **second ear whose claims are
checked**, never a measurement source. `smpl ask` collects claims; `smpl crosscheck` (core, deterministic)
referees them against what smpl measured; the calibration set says how far each question type can be
trusted. See [REFERENCES.md](../REFERENCES.md).

```bash
# one clip
smpl read loop.wav | smpl beats | smpl groove --bpm 128 | smpl qc \
  | smpl ask --q tempo_bpm | smpl ask --q timing_drift | smpl crosscheck --report

# an A/B pair (always asked in both orders); claims out as ear-ledger questions
smpl read a.wav b.wav | smpl loudness --all | smpl ask --q ab_louder --ledger ledger.json | smpl crosscheck

smpl ask questions            # the bank
smpl ask --q tempo_bpm --dry-run   # token / cost estimate, no call, no key
```

`smpl ask` is a heavy-tier tool (`tools/smpl-ask`, own venv, PATH-discovered as `smpl-ask`):
`uv tool install ./tools/smpl-ask`. The key is env-var-first (`SMPL_ASK_GEMINI_KEY`, `GEMINI_API_KEY`,
`GOOGLE_API_KEY`); it lives on the Mac mini, so run there. No key degrades to one `unsupported` error frame
and a stderr hint. Default model `gemini-3.8-flash` (`--model` / `$SMPL_ASK_MODEL`); the `gemini-2.5-*`
ids are closed to new keys. A local backend (Qwen3-Omni, MOSS-Audio) is a stub (`--backend local`).

## The question bank (`tools/smpl-ask/src/smpl_ask/questions.json`)

| Question | Kind | Answer | Checked by crosscheck against |
|---|---|---|---|
| `tempo_bpm` | clip | number | `rhythm.bpm` (beats); an octave / alternate candidate is `unverifiable` |
| `swing_feel` | clip | straight / swung | `rhythm.swing` (groove) |
| `timing_drift` | clip | yes/no + timestamps | per-onset residual vs the same step in other bars (groove) |
| `detune` | clip | cents | `tonal.tuning_frequency` (chords) |
| `clipping` | clip | yes/no | `qc.clipping.detected` |
| `defect_location` | clip | yes/no + timestamps | qc click / gap markers |
| `ab_louder` | A/B | A / B / same | `loudness.integrated_lufs` of each |
| `ab_swingier` | A/B | A / B / same | `rhythm.swing` of each |
| `ab_tight_low_end` | A/B | A / B / same | nothing: `route: ear` (no objective measure; goes to the listening page) |
| `stereo_width`, `phase_problem`, `true_peak`, `lossy_cutoff` | clip | — | **`route: smpl`**, see below |

No measurement ever goes into a prompt (a test pins it): the model gets the audio and one question.
Replies are JSON `{answer, timestamps, evidence}`; timestamps are required when the answer is "yes".

## Why stereo and fidelity questions are routed to smpl

Gemini downmixes every input to one channel and resamples to 16 kHz. In that downmix anti-phase material
cancels to silence and stereo width collapses to nothing; inter-sample (true) peaks do not exist at 16 kHz;
a lossy low-pass above 8 kHz is gone before the model hears it. A model cannot hear what its input no longer
contains, so `stereo_width`, `phase_problem`, `true_peak` and `lossy_cutoff` are smpl's questions
(`smpl width`, `smpl qc`, `smpl loudness`). `smpl ask` refuses them unless `--allow-routed`; `crosscheck`
marks any such claim `route: smpl`, `unverifiable`, and lists the smpl value that answers it.
`smpl-ask calibrate --include-routed` asks them anyway on stereo / anti-phase clips to measure that blindness.

## How crosscheck decides

`supported` / `contradicted` / `unverifiable`, with every frame value compared listed in `compared`. Two trust rules:
a **contradiction needs a reliable measurement** (the qc click finder is precise but not exhaustive, so it can
refute "no click" but never "a click at t"; a groove fitted to an *estimated* tempo can fake swing, so its
contradictions are downgraded until `smpl groove --bpm N` is given); and **an A/B claim whose two orders
disagreed has no standing**. Tolerances are flags (`--tol key=value`, defaults in `smpl_analysis.crosscheck.DEFAULTS`).
`--drop-contradicted` removes contradicted claims from the stream.

Known measurement limits (measured on the calibration clips, not assumed): `smpl beats` reads 84 BPM as 112 and
150 BPM as 120 on dense-hat synthetic loops (the true tempo is among `rhythm.bpm_candidates`, which crosscheck
treats as `unverifiable`, not contradicted); the groove onset detector shows a per-instrument bias of 10-15 ms
and occasional spurious pre-onsets, which is why timing is judged on the *residual* against the same step in
other bars, not the raw grid deviation.

## Calibration

`smpl-ask calibrate` synthesizes 47 deterministic clips with known answers (5 tempos, 6 straight/swung, 6
aligned/one-snare-20-ms-late, 6 in-tune/detuned, 6 clean/clipped, 6 clean/click-at-known-time, 12 A/B pairs;
+4 routed stereo/phase clips with `--include-routed`), asks each through the same code path as `smpl ask`
(A/B in both orders) and reports per question: hit rate against chance, A/B order agreement, first-slot
share (position bias), timestamp hit rate, and cost. Truth is validated against smpl's own measurements in
`tools/smpl-ask/tests/test_calibration.py`.

```bash
smpl-ask calibrate --estimate                       # price it, no call, no key
smpl-ask calibrate --yes --out cal-gemini-3.8-flash.json   # live; on the mini, spends money
```

### Results

Not yet run live. Record the table from `smpl-ask calibrate --yes` here (model, date, per-question hit rate vs
chance, order agreement, cost). Then, as Nick answers the exported "matches?" questions on the listening page,
add the ear-ledger hit rate beside it: the synthetic set measures whether the model hears an isolated
difference, the ledger whether it can be trusted on a real mix.

## Ear-ledger export

`smpl ask --ledger FILE` (or `smpl ask export-ledger`) writes each claim as a forced-choice question:
the exact claim text, the first timestamp to jump to, `kind: matches` (yes / partly / no) or `choice`
(A / B / same for A/B claims). Claims crosscheck marked `contradicted` are dropped; `supported` and
`unverifiable` ones go to the ear. Document shape: `smpl_ask/ledger.py`.
