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
