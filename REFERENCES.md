# References

Published methods the smpl ops implement or lean on. One entry per technique.

## conform (vault-16dd.2)

- **Integrated loudness and true peak** — ITU-R BS.1770-4, "Algorithms to measure audio programme
  loudness and true-peak audio level", 2015. https://www.itu.int/rec/R-REC-BS.1770 — `conform` reports
  integrated LUFS (via `pyloudnorm`) and a 4x-oversampled true peak against the intent's targets.
- **Chroma from a constant-Q transform** — C. Schörkhuber and A. Klapuri, "Constant-Q transform toolbox
  for music processing", SMC 2010; chroma features after M. Müller, *Fundamentals of Music Processing*,
  Springer 2015, ch. 3. Implemented by `librosa.feature.chroma_cqt` — `conform` sums it over time to
  get the per-pitch-class energy share it compares to the intended pitch-class set.

Not from the literature (a project measurement, documented in `smpl_analysis/conform.py`): the per-hit
onset-error estimate (1 ms RMS envelope, floor-to-20 % rise against the recipe's hit times) and the
rotation test for the intended pitch-class set.

## sheet (vault-16dd.3)

- **ITU-R BS.1770-4, "Algorithms to measure audio programme loudness and true-peak audio
  level"** (ITU, 2015), <https://www.itu.int/rec/R-REC-BS.1770> — K-weighting and the
  −0.691 + 10·log10 Σ z loudness formula behind `smpl sheet`'s loudness lane and per-bar LUFS.
- **EBU R 128 / EBU Tech 3341, "Loudness Metering: 'EBU Mode' metering to supplement
  loudness normalisation"** (EBU, 2016), <https://tech.ebu.ch/publications/tech3341> — the
  400 ms momentary and 3 s short-term windows drawn in `smpl sheet` (centred, not trailing).
- **J. C. Brown, "Calculation of a constant Q spectral transform"** (JASA 89(1), 1991),
  <https://doi.org/10.1121/1.400476> — the constant-Q spectrogram lane in `smpl sheet`
  (via librosa), calibrated per bin to dB of a sinusoid; chroma is folded off the same CQT.
- **J. Foote, "Automatic audio segmentation using a measure of audio novelty"** (ICME 2000),
  <https://doi.org/10.1109/ICME.2000.869637> — the self-similarity panel and the
  checkerboard-novelty section boundaries `smpl sheet` draws (segmentation reused from
  `smpl arc`).

## groove-feel (vault-16dd.7)

- **Longuet-Higgins, H. C. & Lee, C. S. (1984).** "The Rhythmic Interpretation of Monophonic
  Music." *Music Perception* 1(4), 424-441. https://doi.org/10.2307/40285271
  Metrical-weight model of syncopation: each step of the bar carries a weight set by its level
  in the metrical hierarchy (0 at the downbeat, one lower per level); a note followed by a
  silent step of higher weight, before the next note, is a syncopation worth the weight
  difference. Used for `rhythm.syncopation_lhl` and the per-bar index in the `groove-feel`
  text export.
- **Fitch, W. T. & Rosenfeld, A. J. (2007).** "Perception and Production of Syncopated
  Rhythms." *Music Perception* 25(1), 43-58. https://doi.org/10.1525/mp.2007.25.1.43
  The pattern-level formalisation of the Longuet-Higgins/Lee measure used here (sum over the
  notes of a looped bar; the largest silent step in each gap) and the model for normalising by
  the maximum a pattern can score (`rhythm.syncopation_norm`).

## clash (vault-16dd.4)

- **Sensory dissonance of partial pairs.** W. A. Sethares, "Local consonance and the
  relationship between timbre and scale", *J. Acoust. Soc. Am.* 94(3), 1993, 1218-1228.
  https://doi.org/10.1121/1.408175. Used for `clash.dissonance`: the closed-form fit
  `d = a1 a2 (exp(-3.5 s df) - exp(-5.75 s df))`, `s = 0.24 / (0.0207 f1 + 18.96)`, summed over
  simultaneous cross-layer partial pairs.
- **The curve it fits.** R. Plomp and W. J. M. Levelt, "Tonal consonance and critical
  bandwidth", *J. Acoust. Soc. Am.* 38(4), 1965, 548-560. https://doi.org/10.1121/1.1909741.
  Roughness peaks at about a quarter of the critical bandwidth and vanishes at unison and
  beyond a critical band.
- **ERB-rate band spacing.** B. R. Glasberg and B. C. J. Moore, "Derivation of auditory filter
  shapes from notched-noise data", *Hearing Research* 47, 1990, 103-138.
  https://doi.org/10.1016/0378-5955(90)90170-T. Used for the band edges of the per-band
  overlap (`ERB-rate = 21.4 log10(1 + 0.00437 f)`, one band per ERB, 30 Hz to 15 kHz).
- **Spectral flux.** S. Dixon, "Onset detection revisited", *Proc. DAFx-06*, 2006, 133-137.
  Used for the flux across the loop wrap (half-wave-rectified difference of log-compressed
  magnitudes between successive frames).

## ask + crosscheck (vault-16dd.6)

- **MUSE: The MUSE Benchmark: Probing Music Perception and Auditory Relational Reasoning in Audio LLMs**
  — Brandon James Carone, Iran R. Roman, 2025. <https://arxiv.org/abs/2510.19055>.
  Audio-native models sit near chance on core music perception (Qwen2.5-Omni and Audio Flamingo 3 among them).
  Used for: the design rule that a model's musical claim is checked against a measurement before it is believed,
  and the calibration set that measures per-question hit rate against chance.
- **AudioCapBench: Quick Evaluation on Audio Captioning across Sound, Music, and Speech**
  — Jielin Qiu, Jianguo Zhang et al., 2026. <https://arxiv.org/abs/2602.23649>.
  Music is the weakest of the three domains for audio-language models.
  Used for: restricting the bank to narrow, forced-choice or numeric questions rather than open description.
- **AudioJudge: Understanding What Works in Large Audio Model Based Speech Evaluation**
  — Potsawee Manakul, Woody Haosheng Gan et al., 2025. <https://arxiv.org/abs/2507.12705>.
  Large audio models used as pairwise judges show position bias.
  Used for: running every A/B question in both presentation orders and treating disagreement as
  `confidence: low` (`smpl_ask.ask.ask`).

## cohesion / coverage / probe (vault-16dd.5)

- **LAION-CLAP** — Y. Wu, K. Chen, T. Zhang, Y. Hui, M. Nezhurina, T. Berg-Kirkpatrick, S. Dubnov, "Large-scale
  Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation",
  ICASSP 2023. <https://arxiv.org/abs/2211.06687> — `smpl embed --model clap` (the
  `music_audioset_epoch_15_esc_90.14` checkpoint, converted to `transformers.ClapModel`): the joint
  audio-text space `coverage` and `probe` score in.
- **MuQ-MuLan** — H. Zhu, Y. Zhou, H. Chen, J. Yu, Z. Ma, R. Gu, Y. Luo, W. Tan, X. Chen, "MuQ:
  Self-Supervised Music Representation Learning with Mel Residual Vector Quantization", 2025.
  <https://arxiv.org/abs/2501.01108> — `smpl embed --model muq-mulan` (`OpenMuQ/MuQ-MuLan-large`):
  a music-text joint space for musical similarity.
- **MuLan** — Q. Huang, A. Jansen, J. Lee, R. Ganti, J. Y. Li, D. P. W. Ellis, "MuLan: A Joint
  Embedding of Music Audio and Natural Language", ISMIR 2022. <https://arxiv.org/abs/2208.12415> —
  the two-tower contrastive design MuQ-MuLan follows.
- **Kernel Audio Distance (KAD)** — Y. Chung, P. Eu, J. Lee, K. Choi, J. Nam, B. S. Chon, "KAD: No
  More FAD! An Effective and Efficient Evaluation Metric for Audio Generation", 2025.
  <https://arxiv.org/abs/2502.15602> — `smpl cohesion --ref`: unbiased MMD² with a Gaussian RBF
  kernel, σ = median pairwise distance in the reference set, scaled by α = 100.
- **Unbiased MMD estimator** — A. Gretton, K. M. Borgwardt, M. J. Rasch, B. Schölkopf, A. Smola,
  "A Kernel Two-Sample Test", JMLR 13, 2012. <https://jmlr.org/papers/v13/gretton12a.html> — the
  estimator KAD scales.
- **Modified z-score** — B. Iglewicz and D. C. Hoaglin, *How to Detect and Handle Outliers*, ASQC
  Quality Press, 1993 — `smpl cohesion`'s outlier z, 0.6745·(d − median)/MAD, flagged above 3.5.
- **Contrastive text axes in a joint embedding space** — prompt ensembling (L2-normalize, mean,
  L2-normalize) from A. Radford et al., "Learning Transferable Visual Models From Natural Language
  Supervision" (CLIP), ICML 2021, <https://arxiv.org/abs/2103.00020>; the bipolar axis
  normalize(P_pos − P_neg) and per-set standardization as recorded in the vault's
  `research/2026-06-10-musical-axes/clap-landscape.md` (vault-3hvg) — `smpl probe`, and the family
  poles `smpl coverage` assigns to.
