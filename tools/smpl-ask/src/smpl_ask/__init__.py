"""smpl-ask — an audio LLM as a second ear whose claims are checked, never a measurement source.

`smpl ask` puts one narrow question from a fixed bank to Gemini about a clip (or an A/B pair, always
in both orders) and emits `claim` text frames; `smpl crosscheck` (core smpl, deterministic) marks each
claim supported / contradicted / unverifiable against smpl's own measurements. `smpl-ask calibrate`
scores the model on synthetic clips with known answers, per question type.

Why a second ear and not a meter: audio-native models are near chance on music benchmarks (MUSE,
AudioCapBench), show position bias on A/B judgements (AudioJudge) and Gemini downmixes to mono. See
REFERENCES.md.
"""

__version__ = "0.1.0"
