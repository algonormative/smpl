---
name: smpl-audit
description: Technical audit of a bounce, master, or sample using the smpl toolchain. Use when the user asks whether audio is technically clean or release-ready ("check this master", "did the export clip?", "is this loud enough for streaming?", "is this secretly a re-encoded MP3?"). Runs measured passes (loudness, QC, spectral, spectrogram) and reports a verdict where every claim cites a measured number with its unit. Requires the smpl CLI on PATH.
---

# smpl-audit: the measured bounce check

You audit an audio file with the smpl toolchain and report a verdict.
The rule that makes the audit worth running: **every claim cites a
measured number with its unit.** No number from a frame, no claim.

Check the toolchain first: `smpl --version`. If it is missing, point
the user at <https://github.com/algonormative/smpl#install>.

## The pass

One pipe collects everything the audit needs:

```sh
smpl read MIX.wav | smpl describe-all | smpl view
```

`describe-all` aggregates the light analysis tier (loudness + spectral
+ QC + envelope + a mel spectrogram); `view` renders the frames as a
markdown report with a feature table. For a narrower question, run the
single stage instead: `smpl loudness`, `smpl qc`, or `smpl spectral`.

Open the spectrogram image (`smpl resolve <hash>` on the `image`
frame) and look at it before writing the verdict; several failures
(lossy origin, hum, gaps) are visible before they are legible in
numbers.

## What to check, and against what

| Dimension | Frames to read | Watch for |
|---|---|---|
| Loudness | `loudness.integrated_lufs`, `loudness.max_short_term_lufs` | distance from the user's target; streaming platforms normalize around −14 LUFS, club/DJ material runs hotter; ask what the bounce is *for* rather than assuming |
| True peak | `loudness.true_peak_dbtp` | above −1.0 dBTP risks clipping in lossy encodes |
| Clipping | `qc.clipping.detected` + defect markers | any `true` is a finding; markers say where |
| DC offset | `qc.dc_offset_dbfs` | notable above roughly −40 dBFS |
| Lossy origin | `qc.lossy.confidence`, `qc.lossy.spectral_cutoff_hz` | a cutoff well below Nyquist with high confidence means the "WAV" was once an MP3/AAC |
| Noise | `qc.snr_db` | low SNR on quiet material; confirm on the spectrogram |
| Tonal balance | `lowlevel.spectral_*`, the mel image | compare against the user's reference if one exists, not against taste |

## The verdict

Lead with the verdict in plain words, then a short table: dimension,
measured value (with unit), and pass / watch / fail against the stated
target. Keep interpretation separate from measurement, and say which
is which. If the user gave no target, state the assumption you used
and invite a correction.

If the user wants the fix as well as the finding, the level tools
chain directly onto the same pipe (`smpl normalize --lufs -14`,
`smpl limit`, `smpl gain`) and `smpl write` materializes the result;
re-run the audit on the output and show both numbers.
