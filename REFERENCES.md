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
