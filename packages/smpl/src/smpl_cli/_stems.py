"""Stem grouping + render planning for ``smpl render-stems`` (ticket vault-3i88).

Pure, subprocess-free logic promoted from the vault's ``render-stems.py`` wrapper: read the
per-track ``stem`` tag off a ``smpl pattern`` recipe, group the tracks into
perc / bass / atmos, and lay out the render jobs (one full mix + one per non-empty group)
with their post chains and output paths. Everything here is a plain data transform so the
grouping, the ladder rule, and the glue chains are unit-testable without rendering anything.

The one invariant the whole op exists to protect: **stems are never loudness-normalized**.
Each stem's sub-session is rendered with ``master.loudnorm = false`` and its post chain is
level-neutral (the one documented exception is the perc bus's −3 dB pad), so the per-track
``gain_db`` ladder that makes the mix survives into the individual stems and they still sum
to something recognisable. Only the FULL mix is mastered.
"""

from __future__ import annotations

import copy
import os
import shutil
from typing import Optional

# The three buses. Order is the render order and the report order.
STEMS = ("perc", "bass", "atmos")

# Track-name prefix → stem group, consulted only when a track has no explicit `stem` tag.
# (Verbatim from the v2 wrapper — the names are the ones the recipes actually use.)
STEM_BY_PREFIX = (
    ("kick", "perc"), ("clap", "perc"), ("snare", "perc"), ("rim", "perc"),
    ("hat", "perc"), ("chat", "perc"), ("ohat", "perc"), ("ride", "perc"),
    ("perc", "perc"), ("tom", "perc"), ("cy", "perc"), ("cb", "perc"),
    ("sub", "bass"), ("bass", "bass"),
    ("stab", "atmos"), ("texture", "atmos"), ("drone", "atmos"),
    ("noise", "atmos"), ("pad", "atmos"), ("atmos", "atmos"),
)

# A track that matches nothing lands here (rare; the safest bus to over-fill).
DEFAULT_STEM = "perc"

# ---------------------------------------------------------------------------
# Post chains. A chain is a list of ``(op, kwargs)`` pairs applied in order by the
# subcommand through the in-process ``smpl_analysis.edit`` functions — the same DSP the
# `smpl gain / filter / widen / eq / normalize` subcommands call, no sox, no nested pipes.
# ---------------------------------------------------------------------------

# Per-stem bus glue. LEVEL-NEUTRAL except the documented perc pad.
#   perc  → −3 dB pad for true-peak headroom on the drum bus. The wrapper's sox chain was
#           ``overdrive 2 0 gain -3``; the suite has no saturation op, so the drive half is
#           NOT reproduced (delta noted in the subcommand docstring) and only the pad remains.
#   bass  → mono downmix (average of the channels, sox ``channels 1``): level-neutral, and
#           guarantees the low end is mono before it reaches a sampler track.
#   atmos → 120 Hz high-pass, keeping the room/rumble out of the sub bus.
BUS_GLUE: dict[str, tuple] = {
    "perc": (("gain", {"db": -3.0}),),
    "bass": (("mono", {}),),
    "atmos": (("filter", {"kind": "hp", "freq": 120.0}),),
}

# Stereo-image opening above the crossover (the sub stays mono): side gain in dB per stem.
STEM_WIDEN = {"perc": 2.5, "atmos": 3.0}   # bass is deliberately absent (stays mono)
WIDEN_CROSSOVER_HZ = 220.0

MASTER_LUFS = -14.0

# Full-mix master: widen above the crossover, a mid-presence bell + air shelf, then normalize.
FULL_MASTER: tuple = (
    ("widen", {"side_gain_db": 3.5, "crossover_hz": WIDEN_CROSSOVER_HZ}),
    ("eq", {"bands": [
        {"type": "peaking", "freq": 1300.0, "gain": 5.5, "q": 0.6},
        {"type": "highshelf", "freq": 8000.0, "gain": 2.5},
    ]}),
    ("normalize", {"target_lufs": MASTER_LUFS}),
)


# ---------------------------------------------------------------------------
# Grouping.
# ---------------------------------------------------------------------------
def stem_of(track: dict) -> str:
    """The stem bus a recipe track belongs to: explicit ``stem`` tag, else name prefix."""
    tag = track.get("stem")
    if isinstance(tag, str) and tag.strip().lower() in STEMS:
        return tag.strip().lower()
    name = str(track.get("name") or "").lower()
    for prefix, group in STEM_BY_PREFIX:
        if name.startswith(prefix):
            return group
    return DEFAULT_STEM


def group_tracks(tracks: list) -> dict:
    """``{stem: [track, ...]}`` for every bus (empty lists included, order = ``STEMS``)."""
    groups: dict[str, list] = {s: [] for s in STEMS}
    for track in tracks or []:
        groups[stem_of(track)].append(track)
    return groups


def parse_stems(spec: Optional[str]) -> tuple:
    """``"perc,bass"`` → ``("perc", "bass")`` (``None``/``"all"`` ⇒ every bus)."""
    if spec is None or spec.strip().lower() in ("", "all"):
        return tuple(STEMS)
    wanted = [s.strip().lower() for s in spec.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in STEMS]
    if unknown:
        raise ValueError(f"unknown stem(s) {', '.join(unknown)} (known: {', '.join(STEMS)})")
    # De-duplicate, keep the canonical bus order so output is stable.
    return tuple(s for s in STEMS if s in wanted)


# ---------------------------------------------------------------------------
# Chains + sub-recipes.
# ---------------------------------------------------------------------------
def stem_chain(stem: str, *, widen: bool = True) -> list:
    """Bus glue (+ optional widen) for one stem. Never contains a loudness op."""
    chain = list(BUS_GLUE.get(stem, ()))
    side_db = STEM_WIDEN.get(stem)
    if widen and side_db is not None:
        chain.append(("widen", {"side_gain_db": side_db, "crossover_hz": WIDEN_CROSSOVER_HZ}))
    return chain


def full_chain(*, master: bool = True) -> list:
    """The full mix's post chain — the ONLY chain allowed to touch loudness."""
    return list(FULL_MASTER) if master else []


def sub_recipe(recipe: dict, tracks: list) -> dict:
    """A copy of ``recipe`` restricted to ``tracks`` and rendered RAW.

    ``master.loudnorm`` is forced off for every render (stems must keep the ladder; the full
    mix is mastered afterwards by the post chain instead, so it too is rendered raw), while
    the true-peak limiter stays on.
    """
    out = copy.deepcopy(recipe)
    out["tracks"] = copy.deepcopy(tracks)
    master = dict(out.get("master") or {})
    master["loudnorm"] = False
    master["limiter"] = True
    out["master"] = master
    return out


# ---------------------------------------------------------------------------
# Planning.
# ---------------------------------------------------------------------------
def recipe_name(recipe: dict, pattern_path: Optional[str] = None) -> str:
    """Output basename: the recipe's ``name``, else the pattern file's stem, else ``pattern``."""
    name = recipe.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    if pattern_path:
        base = os.path.basename(pattern_path).split(".")[0]
        if base:
            return base
    return "pattern"


def output_dir(outdir: str, recipe: dict) -> str:
    """Renders nest under the recipe's ``_set`` when it names one (staging-friendly)."""
    setname = recipe.get("_set")
    if isinstance(setname, str) and setname.strip():
        return os.path.join(outdir, setname.strip())
    return outdir


def plan(
    recipe: dict,
    *,
    outdir: str,
    stems: tuple = STEMS,
    master: bool = True,
    widen: bool = True,
    pattern_path: Optional[str] = None,
) -> dict:
    """Lay out every render job for one recipe (pure; touches no disk and no subprocess).

    Returns ``{"name", "set", "outdir", "groups", "jobs": [job, ...]}`` where each job is
    ``{"kind", "tracks", "recipe", "chain", "out"}`` — ``kind`` is ``"full"`` or a stem name.
    Groups with no tracks produce no job (an empty stem is skipped, not an empty file).
    """
    tracks = recipe.get("tracks") or []
    if not tracks:
        raise ValueError("recipe has no tracks")
    name = recipe_name(recipe, pattern_path)
    dest = output_dir(outdir, recipe)
    groups = group_tracks(tracks)

    jobs = [{
        "kind": "full",
        "tracks": list(tracks),
        "recipe": sub_recipe(recipe, tracks),
        "chain": full_chain(master=master),
        "out": os.path.join(dest, f"{name}.full.wav"),
    }]
    for stem in stems:
        group = groups.get(stem) or []
        if not group:
            continue
        jobs.append({
            "kind": stem,
            "tracks": list(group),
            "recipe": sub_recipe(recipe, group),
            "chain": stem_chain(stem, widen=widen),
            "out": os.path.join(dest, f"{name}.{stem}.wav"),
        })
    return {
        "name": name,
        "set": recipe.get("_set"),
        "outdir": dest,
        "groups": {s: [str(t.get("name") or "") for t in groups[s]] for s in STEMS},
        "jobs": jobs,
    }


# ---------------------------------------------------------------------------
# The smplmix seam. smplmix is an EXTERNAL binary (the Rust compositor); this op only ever
# hands it a session file and a destination, so it is trivially stubbable in tests.
# ---------------------------------------------------------------------------
SMPLMIX_ENV = "SMPL_SMPLMIX_BIN"


def resolve_smplmix(explicit: Optional[str] = None) -> Optional[str]:
    """``--smplmix`` → ``$SMPL_SMPLMIX_BIN`` → PATH. A path that is not executable ⇒ None."""
    for candidate in (explicit, os.environ.get(SMPLMIX_ENV)):
        if candidate:
            ok = os.path.isfile(candidate) and os.access(candidate, os.X_OK)
            return candidate if ok else None
    return shutil.which("smplmix")


def missing_smplmix_message(explicit: Optional[str] = None) -> str:
    """The actionable error for an unusable renderer (no silent fallback, no partial output)."""
    if explicit:
        return (f"render-stems: --smplmix {explicit!r} is not an executable file.\n"
                f"      Point it at the smplmix binary, or drop the flag to use PATH.")
    if os.environ.get(SMPLMIX_ENV):
        return (f"render-stems: ${SMPLMIX_ENV} = {os.environ[SMPLMIX_ENV]!r} is not an "
                f"executable file.\n      Fix it, unset it to use PATH, or pass --smplmix PATH.")
    return ("render-stems: `smplmix` not found on PATH — it renders the sessions this op plans.\n"
            f"      Install smplmix, or point at it with --smplmix PATH / ${SMPLMIX_ENV}.")


def smplmix_command(binary: str, session_path: str, out_path: str) -> list:
    """The render invocation: ``smplmix render <session.smplset.json> -o <out.wav>``."""
    return [binary, "render", session_path, "-o", out_path]
