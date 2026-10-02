"""AudioCommons 8 timbral descriptors as LLM-facing features (research §6; ticket vault-14ia).

The "what does this *sound like* in words" axis — eight perceptual descriptors on a 0–100
scale (`timbre.reverb` binary 0/1) so a sample is natural-language queryable ("dark, warm,
boomy one-shot") without a model in the loop. Keys come from the registry
(feature-keys.md, owner vault-14ia), deliberately **outside** the Essentia `lowlevel.*`
namespace because they are perceptual, not signal-objective.

**REIMPLEMENTED, NOT VENDORED.** Upstream `timbral_models` (the AudioCommons reference
implementation) is unmaintained and thousands of lines with its own model pickles; this
module is a compact, dependency-free-beyond-the-workspace (numpy/scipy/librosa)
reimplementation that maps the *published signal correlates* of each descriptor onto the
same 0–100 scale. It is faithful in spirit and direction, NOT numerically identical to the
upstream regressors — treat the values as a calibrated internal scale, not as
AudioCommons ground truth.

Analysis basis
--------------
One magnitude STFT (``N_FFT``/``HOP_LENGTH``) collapsed to a single **active-frame mean
power spectrum** ``P(f)``: frames whose energy is below ``ACTIVE_FLOOR`` × the loudest
frame are dropped so leading/trailing silence cannot drag every band ratio toward zero.
``E(a,b)`` is the summed ``P`` over ``[a, b)`` Hz, ``Etot = E(0, nyquist)``, and
``logmap(x, lo, hi) = clip(log(x/lo) / log(hi/lo), 0, 1)`` is the octave-linear 0–1 ramp
used wherever a frequency is mapped perceptually.

Per-descriptor formulas
-----------------------
``brightness``  0.65·logmap(centroid, 100, 8000) + 0.35·sqrt(E(3k,nyq)/Etot)
                — spectral centroid (log-frequency) plus the high-band energy fraction.
``depth``       0.60·sqrt(E(20,300)/Etot) + 0.40·(1 − logmap(centroid, 50, 4000))
                — low-frequency energy plus the inverse of the centroid.
``warmth``      sqrt(E(60,500)/Etot) · (1 − E(3k,nyq)/Etot)
                — low-mid ("warmth region") energy balance, penalised by high-band energy.
``boominess``   (E(20,200)/E(20,1000))^1 · (E(20,200)/Etot)^0.25
                — the Hatano/Hashimoto-style booming index: low-band dominance inside the
                low-mid range, gated by the low band's share of the whole spectrum, so a
                broadband noise (bass present but not dominant) cannot read as boomy.
``sharpness``   Zwicker/DIN 45692 form: specific-loudness proxy N'(z) = E(z)^0.23 over 24
                Bark bands, weighted g(z) = 1 for z ≤ 15.8 else 0.15·exp(0.42(z−15.8)) +
                0.85; acum = 0.11·Σ N'(z)·g(z)·z / Σ N'(z), scaled by ``SHARP_FULL_ACUM``.
``hardness``    0.5·(1 − logmap(attack_ms, 2, 200)) + 0.5·sqrt(E(2k,nyq)/Etot)
                — attack strength (10 %→90 % rise of the RMS envelope to its peak) plus
                high-frequency content, the two correlates the AudioCommons hardness
                regressor leans on.
``roughness``   Weighted mean, over log-spaced bands, of the amplitude-modulation depth in
                the ``ROUGH_LO``–``ROUGH_HI`` Hz (≈15–75 Hz) beating band: per band,
                sqrt(modulation power in that band) / mean envelope level, band-weighted by
                mean envelope level and saturated through 1 − exp(−x/``ROUGH_SCALE``).
                Modulation *depth* (not raw modulation energy) so a pure tone — whose
                envelope is flat and whose numerator and denominator are both noise —
                cannot read as rough. **Caveat:** a transient has a broadband modulation
                spectrum, so dense percussive material reads high here even without any
                dissonant beating (the upstream model uses Vassilakis peak-pair roughness,
                which separates the two).
``reverb``      Binary decay-tail heuristic on the Schroeder energy-decay curve (backward
                integration of the squared signal): estimate RT60 from the −5 dB→−25 dB
                interval (T20 × 3) and require the dB decay over that span to be *linear*
                (least-squares R² ≥ ``REVERB_MIN_R2``). 1 when RT60 ≥ ``REVERB_RT60_MS``
                with a linear decay, else 0. The linearity gate is what separates a
                reverberant tail (exponential ⇒ straight line in dB) from a steady sustained
                tone, whose EDC also spans 20 dB but curves hard at the cut-off.
                **Known limitation:** a *dry* source with a genuinely long, smooth decay (a
                long 808, a bowed cymbal) is indistinguishable from a room tail by decay
                shape alone — the upstream AudioCommons model uses a trained classifier for
                exactly this reason. The ``REVERB_RT60_MS`` threshold is set well past a
                typical dry one-shot decay so the common cases fall the right way.

Every value is clamped to [0, 100] and rounded, so the output is deterministic for a given
input and ``op_version``. Pure functions returning frame dicts; heavy imports (numpy/
scipy/librosa) live inside the functions so cold pipe stages start fast.
"""

from __future__ import annotations

OP = "timbre"
OP_VERSION = "timbre@1"

# --- pinned analysis constants (declared so memo params are complete & stable per op_version) ---
N_FFT = 2048
HOP_LENGTH = 512
ACTIVE_FLOOR = 1e-6        # frame energy below this × the loudest frame is silence

# Band edges (Hz) — the perceptual regions the descriptors are built from.
LOW_BAND = (20.0, 300.0)       # depth
BOOM_BAND = (20.0, 200.0)      # boominess numerator
BOOM_REF_BAND = (20.0, 1000.0) # boominess denominator (low-mid reference)
WARM_BAND = (60.0, 500.0)      # warmth region
HARD_HF = 2000.0               # hardness high-band floor
BRIGHT_HF = 3000.0             # brightness / warmth high-band floor

CENTROID_LO, CENTROID_HI = 100.0, 8000.0   # brightness centroid ramp
DEPTH_CENTROID_LO, DEPTH_CENTROID_HI = 50.0, 4000.0
ATTACK_LO_MS, ATTACK_HI_MS = 2.0, 200.0    # hardness attack ramp
ATTACK_HOP = 256                            # RMS-envelope hop for the attack estimate

SHARP_FULL_ACUM = 6.0      # acum mapped to 100 (bright hiss ≈ full scale, white noise ≈ half)
ROUGH_LO, ROUGH_HI = 15.0, 75.0   # amplitude-modulation "beating" band (Hz)
ROUGH_ENV_SR = 400.0       # target envelope sample rate for the modulation analysis (Hz)
ROUGH_N_FFT = 1024         # STFT window for the band envelopes
ROUGH_BANDS = 8            # log-spaced bands the modulation depth is averaged over
ROUGH_SCALE = 0.35         # modulation depth mapped through 1 − exp(−x/scale)

REVERB_RT60_MS = 800.0     # RT60 at/above which a linear tail counts as reverberant
REVERB_MIN_R2 = 0.98       # decay must be this linear in dB to be a reverb tail

_EPS = 1e-20


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def _to_mono(y):
    import numpy as np

    y = np.asarray(y, dtype="float64")
    if y.ndim > 1:
        y = y.mean(axis=0) if y.shape[0] < y.shape[1] else y.mean(axis=1)
    return np.ascontiguousarray(y)


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


def _score(x: float) -> float:
    """Map a 0–1 quantity onto the registered 0–100 scale, clamped and rounded."""
    return round(100.0 * _clamp01(x), 3)


def _logmap(x: float, lo: float, hi: float) -> float:
    """Octave-linear 0–1 ramp between ``lo`` and ``hi`` Hz (or ms)."""
    import math

    if x <= lo:
        return 0.0
    if x >= hi:
        return 1.0
    return _clamp01(math.log(x / lo) / math.log(hi / lo))


def mean_power_spectrum(y, sr: int, *, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH):
    """Active-frame mean power spectrum. Returns ``(freqs, P)`` (both 1-D, length n_fft//2+1)."""
    import librosa
    import numpy as np

    from .spectrogram import pad_short_signal

    y = _to_mono(y)
    y = pad_short_signal(np.asarray(y, dtype="float32"), n_fft)
    S = np.abs(librosa.stft(np.asarray(y, dtype="float32"), n_fft=n_fft, hop_length=hop_length))
    power = (S.astype("float64")) ** 2
    frame_energy = power.sum(axis=0)
    peak = float(frame_energy.max()) if frame_energy.size else 0.0
    if peak > 0.0:
        keep = frame_energy >= ACTIVE_FLOOR * peak
        if keep.any():
            power = power[:, keep]
    P = power.mean(axis=1) if power.size else np.zeros(n_fft // 2 + 1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    return freqs, P


def _band_energy(freqs, P, lo: float, hi: float) -> float:
    import numpy as np

    mask = (freqs >= lo) & (freqs < hi)
    return float(np.sum(P[mask])) if mask.any() else 0.0


def _centroid(freqs, P) -> float:
    import numpy as np

    total = float(np.sum(P))
    if total <= _EPS:
        return 0.0
    return float(np.sum(freqs * P) / total)


def _hz_to_bark(f):
    """Traunmüller/Zwicker Bark scale (the standard arctan form)."""
    import numpy as np

    f = np.asarray(f, dtype="float64")
    return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


def attack_ms(y, sr: int, *, hop_length: int = ATTACK_HOP) -> float:
    """10 %→90 %-of-peak rise time of the RMS envelope, up to its global peak (ms).

    A fast rise (percussive hit) is short; a slow swell is long. Returns ``ATTACK_HI_MS``
    when no usable envelope exists (silence), i.e. maximally soft.
    """
    import librosa
    import numpy as np

    y = np.asarray(_to_mono(y), dtype="float32")
    if y.size < hop_length * 2:
        return ATTACK_LO_MS
    env = librosa.feature.rms(y=y, frame_length=hop_length * 4, hop_length=hop_length)[0]
    if env.size == 0 or float(env.max()) <= _EPS:
        return ATTACK_HI_MS
    peak_idx = int(np.argmax(env))
    peak = float(env[peak_idx])
    rise = env[: peak_idx + 1]
    i10 = np.nonzero(rise >= 0.10 * peak)[0]
    i90 = np.nonzero(rise >= 0.90 * peak)[0]
    start = int(i10[0]) if i10.size else 0
    end = int(i90[0]) if i90.size else peak_idx
    frames = max(end - start, 0)
    return max(frames * hop_length / sr * 1000.0, 0.0)


def modulation_depth(y, sr: int) -> float:
    """Energy-weighted amplitude-modulation depth in the 15–75 Hz beating band (0..~1+).

    Band envelopes come from a short-hop STFT (envelope rate ≈ ``ROUGH_ENV_SR``), grouped
    into ``ROUGH_BANDS`` log-spaced bands. Per band the modulation depth is
    ``sqrt(2 · Σ|Env(f)|² over [ROUGH_LO, ROUGH_HI]) / mean(env)`` — scale-invariant, so a
    flat (unmodulated) envelope scores ~0 regardless of level.
    """
    import librosa
    import numpy as np

    y = np.asarray(_to_mono(y), dtype="float32")
    hop = max(1, int(round(sr / ROUGH_ENV_SR)))
    env_sr = sr / hop
    if y.size < ROUGH_N_FFT * 2 or env_sr <= 2 * ROUGH_HI:
        return 0.0

    S = np.abs(librosa.stft(y, n_fft=ROUGH_N_FFT, hop_length=hop)).astype("float64")
    freqs = librosa.fft_frequencies(sr=sr, n_fft=ROUGH_N_FFT)
    nyq = sr / 2.0
    edges = np.geomspace(50.0, max(nyq * 0.95, 100.0), ROUGH_BANDS + 1)

    n = S.shape[1]
    if n < 16:
        return 0.0
    win = np.hanning(n)
    win_gain = float(np.sum(win)) / n or 1.0
    mod_freqs = np.fft.rfftfreq(n, d=1.0 / env_sr)
    band_mask = (mod_freqs >= ROUGH_LO) & (mod_freqs <= ROUGH_HI)
    if not band_mask.any():
        return 0.0

    depths, weights = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (freqs >= lo) & (freqs < hi)
        if not sel.any():
            continue
        env = S[sel].sum(axis=0)
        level = float(np.mean(env))
        if level <= _EPS:
            continue
        spec = np.abs(np.fft.rfft((env - np.mean(env)) * win)) / (n * win_gain)
        depth = float(np.sqrt(2.0 * np.sum(spec[band_mask] ** 2)) / level)
        depths.append(depth)
        weights.append(level)

    if not depths:
        return 0.0
    d = np.asarray(depths)
    w = np.asarray(weights)
    return float(np.sum(d * w) / max(float(np.sum(w)), _EPS))


def decay_estimate(y, sr: int) -> tuple:
    """Schroeder-EDC decay estimate. Returns ``(rt60_ms, r2)``.

    ``rt60_ms`` extrapolates the −5 dB→−25 dB (T20) interval; ``r2`` is the coefficient of
    determination of a straight-line fit to the dB decay over that interval (how
    *exponential* the tail is). Returns ``(0.0, 0.0)`` when the EDC never spans 25 dB.
    """
    import numpy as np

    y = np.asarray(_to_mono(y), dtype="float64")
    if y.size < 32:
        return 0.0, 0.0
    sq = y ** 2
    edc = np.cumsum(sq[::-1])[::-1]          # backward energy integration
    total = float(edc[0])
    if total <= _EPS:
        return 0.0, 0.0
    edc_db = 10.0 * np.log10(np.maximum(edc / total, 1e-12))

    below5 = np.nonzero(edc_db <= -5.0)[0]
    below25 = np.nonzero(edc_db <= -25.0)[0]
    if below5.size == 0 or below25.size == 0:
        return 0.0, 0.0
    i5, i25 = int(below5[0]), int(below25[0])
    if i25 <= i5 + 2:
        return 0.0, 0.0

    t20_ms = (i25 - i5) / sr * 1000.0
    seg = edc_db[i5:i25 + 1]
    t = np.arange(seg.size, dtype="float64")
    slope, intercept = np.polyfit(t, seg, 1)
    resid = seg - (slope * t + intercept)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((seg - np.mean(seg)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > _EPS else 0.0
    return 3.0 * t20_ms, float(r2)


# --------------------------------------------------------------------------------------------
# The eight descriptors
# --------------------------------------------------------------------------------------------


def timbral_descriptors(y, sr: int, *, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH) -> dict:
    """Compute the eight AudioCommons timbral descriptors over a mono/multichannel signal.

    Returns a dict mapping each registered `timbre.*` key to its 0–100 score
    (`timbre.reverb` is the binary 0/1 flag). The caller wraps it in a `feature` frame.
    """
    import numpy as np

    y = _to_mono(y)
    freqs, P = mean_power_spectrum(y, sr, n_fft=n_fft, hop_length=hop_length)
    nyq = sr / 2.0
    etot = float(np.sum(P)) + _EPS

    centroid = _centroid(freqs, P)
    e_low = _band_energy(freqs, P, *LOW_BAND)
    e_boom = _band_energy(freqs, P, *BOOM_BAND)
    e_boom_ref = _band_energy(freqs, P, *BOOM_REF_BAND)
    e_warm = _band_energy(freqs, P, *WARM_BAND)
    e_hf_bright = _band_energy(freqs, P, BRIGHT_HF, nyq + 1.0)
    e_hf_hard = _band_energy(freqs, P, HARD_HF, nyq + 1.0)

    hf_bright_ratio = e_hf_bright / etot

    # --- brightness: log-centroid + high-band share. ---
    brightness = 0.65 * _logmap(centroid, CENTROID_LO, CENTROID_HI) + 0.35 * np.sqrt(hf_bright_ratio)

    # --- depth: low-band share + inverse centroid. ---
    depth = 0.60 * np.sqrt(e_low / etot) + 0.40 * (
        1.0 - _logmap(centroid, DEPTH_CENTROID_LO, DEPTH_CENTROID_HI)
    )

    # --- warmth: low-mid share, penalised by high-band energy. ---
    warmth = np.sqrt(e_warm / etot) * (1.0 - hf_bright_ratio)

    # --- boominess: booming index (low dominance in the low-mids), gated by low share. ---
    boominess = (e_boom / (e_boom_ref + _EPS)) * (e_boom / etot) ** 0.25

    # --- sharpness: Zwicker/DIN 45692 over 24 Bark bands. ---
    bark = _hz_to_bark(freqs)
    idx = np.clip(np.floor(bark).astype(int), 0, 23)
    band_e = np.bincount(idx, weights=P, minlength=24)[:24]
    n_specific = np.power(np.maximum(band_e, 0.0), 0.23)
    z = np.arange(24, dtype="float64") + 0.5
    g = np.where(z <= 15.8, 1.0, 0.15 * np.exp(0.42 * (z - 15.8)) + 0.85)
    denom = float(np.sum(n_specific)) + _EPS
    acum = 0.11 * float(np.sum(n_specific * g * z)) / denom
    sharpness = acum / SHARP_FULL_ACUM

    # --- hardness: attack strength + high-frequency content. ---
    att = attack_ms(y, sr)
    hardness = 0.5 * (1.0 - _logmap(att, ATTACK_LO_MS, ATTACK_HI_MS)) + 0.5 * np.sqrt(
        e_hf_hard / etot
    )

    # --- roughness: 15–75 Hz amplitude-modulation depth, saturated. ---
    md = modulation_depth(y, sr)
    roughness = 1.0 - np.exp(-md / ROUGH_SCALE)

    # --- reverb: binary decay-tail heuristic. ---
    rt60_ms, r2 = decay_estimate(y, sr)
    reverb = 1 if (rt60_ms >= REVERB_RT60_MS and r2 >= REVERB_MIN_R2) else 0

    return {
        "timbre.hardness": _score(hardness),
        "timbre.depth": _score(depth),
        "timbre.brightness": _score(brightness),
        "timbre.roughness": _score(roughness),
        "timbre.warmth": _score(warmth),
        "timbre.sharpness": _score(sharpness),
        "timbre.boominess": _score(boominess),
        "timbre.reverb": reverb,
    }


# --------------------------------------------------------------------------------------------
# Frame-emitting entrypoint (the op behind `smpl timbre`).
# --------------------------------------------------------------------------------------------


def timbre_audio_frame(
    audio_frame: dict,
    *,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> list[dict]:
    """Resolve an `audio` frame's PCM from the CAS and emit its timbral `feature` frame.

    Returns a one-element list with the derived `feature` frame (role `timbre`), carrying
    `of`/`op`/`op_version`/`params` lineage per the tool contract. The caller is responsible
    for passthrough of the input frame.
    """
    import soundfile as sf

    from smplstream import cas, frames as F

    src = cas.get_path(audio_frame["hash"])
    y, sr = sf.read(str(src), dtype="float32", always_2d=True)
    y = y.T  # (ch, n) for the mono collapse

    data = timbral_descriptors(y, sr, n_fft=n_fft, hop_length=hop_length)
    params = {
        "n_fft": n_fft,
        "hop_length": hop_length,
        "sharp_full_acum": SHARP_FULL_ACUM,
        "rough_scale": ROUGH_SCALE,
        "rough_band_hz": [ROUGH_LO, ROUGH_HI],
        "reverb_rt60_ms": REVERB_RT60_MS,
        "reverb_min_r2": REVERB_MIN_R2,
    }
    return [
        F.feature_frame(
            data,
            role="timbre",
            of=audio_frame["id"],
            op=OP,
            op_version=OP_VERSION,
            params=params,
        )
    ]
