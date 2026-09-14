"""Built-in-effects host over Spotify's pedalboard — no VST3/AU, no instrument path, so the
whole surface is testable with nothing but the pip dep installed. Nothing is hardcoded: the
roster is every `pedalboard.Plugin` subclass that is not a container, not external, is an
effect, and constructs with no arguments (`Convolution` needs an impulse response, `IIRFilter`
is abstract — both fall out); parameter names + types come from the pybind11 `__init__`
signature and defaults from a live instance. pedalboard publishes no ranges for built-ins, so
bounds come from a documented table, flagged by `range_source`. Heavy imports stay INSIDE
functions so `--help` never loads the native extension.
"""

from __future__ import annotations

import re
from typing import Any, Optional

OP = "plugin"
OP_SCHEMA_VERSION = 1
INSTALL_HINT = "uv tool install smpl-plugin  # or: pip install 'pedalboard>=0.9'"


class EffectError(Exception):
    """Unknown effect, bad parameter, or a failed render."""


def available() -> bool:
    """True iff the pedalboard native extension imports (the tool's one heavy dep)."""
    try:
        import pedalboard  # noqa: F401
    except Exception:
        return False
    return True


def pedalboard_version() -> str:
    import pedalboard  # noqa: PLC0415 — lazy by design
    return str(getattr(pedalboard, "__version__", "unknown"))


def op_version(name: str) -> str:
    """Bumped by ANY behavior change — for a host that includes the LIBRARY version: a
    pedalboard upgrade can change a built-in's DSP, and folding the version in is what stops a
    stale memo entry being served after one."""
    return f"{OP}@{OP_SCHEMA_VERSION}+pedalboard:{pedalboard_version()}+{name}"


# --- roster -----------------------------------------------------------------------------

def _is_builtin_effect(obj) -> bool:
    import inspect

    import pedalboard

    bases = (pedalboard.Plugin, pedalboard.PluginContainer, pedalboard.ExternalPlugin)
    if not (inspect.isclass(obj) and issubclass(obj, pedalboard.Plugin)) or obj in bases:
        return False
    if issubclass(obj, bases[1:]):
        return False  # containers compose plugins; externals are VST3/AU (out of scope here)
    try:
        inst = obj()  # no-arg constructible ⇒ a self-contained built-in effect
    except Exception:
        return False
    return bool(getattr(inst, "is_effect", False))


def list_effects() -> list[str]:
    """Sorted names of the built-in effects this host can run."""
    import pedalboard

    return sorted(n for n in dir(pedalboard)
                  if not n.startswith("_") and _is_builtin_effect(getattr(pedalboard, n)))


def effect_class(name: str):
    import pedalboard

    obj = getattr(pedalboard, name, None)
    if obj is None or not _is_builtin_effect(obj):
        raise EffectError(f"unknown built-in effect {name!r}; try `smpl plugin list`")
    return obj


# --- parameters -------------------------------------------------------------------------

_SIG_RE = re.compile(r"^__init__\(([^\n]*)\)\s*->")

# Documented / musically sensible operating ranges (pedalboard API docs + JUCE DSP defaults).
# A parameter not listed reports null bounds with range_source "unbounded" — never an invented one.
_RANGES: dict[str, tuple[float, float]] = {
    "threshold_db": (-100.0, 0.0), "ratio": (1.0, 50.0), "attack_ms": (0.0, 1000.0),
    "release_ms": (0.0, 5000.0), "gain_db": (-60.0, 60.0), "drive_db": (0.0, 100.0),
    "cutoff_frequency_hz": (20.0, 20000.0), "cutoff_hz": (20.0, 20000.0), "q": (0.1, 18.0),
    "centre_frequency_hz": (20.0, 20000.0), "rate_hz": (0.0, 100.0), "depth": (0.0, 1.0),
    "mix": (0.0, 1.0), "feedback": (0.0, 1.0), "resonance": (0.0, 1.0), "drive": (1.0, 100.0),
    "room_size": (0.0, 1.0), "damping": (0.0, 1.0), "wet_level": (0.0, 1.0),
    "dry_level": (0.0, 1.0), "width": (0.0, 1.0), "freeze_mode": (0.0, 1.0),
    "bit_depth": (1.0, 32.0), "semitones": (-72.0, 72.0), "delay_seconds": (0.0, 30.0),
    "centre_delay_ms": (0.0, 100.0), "vbr_quality": (0.0, 9.0),
    "target_sample_rate": (100.0, 192000.0),
}
_NORMALIZED = frozenset({"mix", "depth", "feedback", "resonance", "room_size", "damping",
                         "wet_level", "dry_level", "width", "freeze_mode"})
_UNIT_SUFFIXES = (("_db", "dB"), ("_ms", "ms"), ("_hz", "Hz"), ("_seconds", "s"),
                  ("_sample_rate", "Hz"), ("semitones", "semitones"), ("bit_depth", "bits"))


def _units_for(name: str) -> str:
    for suffix, unit in _UNIT_SUFFIXES:
        if name.endswith(suffix):
            return unit
    return ":1" if name == "ratio" else ("normalized" if name in _NORMALIZED else "")


def _declared_types(cls) -> dict[str, str]:
    """Parameter name → declared type string, from the pybind11 `__init__` signature.

    Enum defaults render as `<Mode.LPF12: 0>` — angle-bracket spans are stripped first so a
    default's inner `:`/`,` can't be mistaken for an argument separator.
    """
    doc = (getattr(cls, "__init__", None).__doc__ or "").strip()
    m = _SIG_RE.match(doc.splitlines()[0] if doc else "")
    types: dict[str, str] = {}
    for arg in re.sub(r"<[^>]*>", "", m.group(1)).split(",") if m else []:
        name, _, rest = arg.partition(":")
        if rest.strip() and name.strip() != "self":
            types[name.strip()] = rest.split("=", 1)[0].strip()
    return types


def _enum_class(cls, typestr: str):
    """Resolve a pybind11 nested enum type (`pedalboard_native.X.Mode`) on `cls`."""
    obj = getattr(cls, typestr.rsplit(".", 1)[-1], None)
    return obj if getattr(obj, "__members__", None) else None


def _jsonable(value) -> Any:
    """Enum members serialize as their NAME; everything else is already JSON-able."""
    if hasattr(value, "name") and hasattr(type(value), "__members__"):
        return value.name
    return value if value is None or isinstance(value, (bool, int, float, str)) else str(value)


def parameter_specs(name: str) -> list[dict]:
    """Descriptor `parameters` entries for an effect, at its declared defaults."""
    cls, specs = effect_class(name), []
    defaults = cls()
    for pname, typestr in _declared_types(cls).items():
        enum_cls = _enum_class(cls, typestr)
        lo, hi = _RANGES.get(pname, (None, None))
        if enum_cls is not None:
            kind, source, lo, hi = "enum", "pedalboard-enum", None, None
        else:
            kind = {"float": "float", "int": "int", "bool": "bool"}.get(typestr, "float")
            source = "documented-range" if lo is not None else "unbounded"
        default = _jsonable(getattr(defaults, pname, None))
        spec = {"name": pname, "type": kind, "min": lo, "max": hi, "default": default,
                "units": _units_for(pname), "range_source": source, "value": default}
        if enum_cls is not None:
            spec["choices"] = list(enum_cls.__members__)
        specs.append(spec)
    return specs


def coerce_params(name: str, params: Optional[dict]) -> dict:
    """Validate + coerce a user param dict → the EFFECTIVE dict: every declared parameter
    present with its default filled in, per the spec's memo rule (omitted params are filled
    from the op's defaults BEFORE hashing)."""
    specs = {s["name"]: s for s in parameter_specs(name)}
    effective = {n: s["default"] for n, s in specs.items()}
    for key, raw in (params or {}).items():
        spec = specs.get(key)
        if spec is None:
            raise EffectError(f"{name}: unknown parameter {key!r}; "
                              f"declared: {', '.join(sorted(specs)) or '(none)'}")
        effective[key] = _coerce_one(name, spec, raw)
    return effective


def _coerce_one(effect: str, spec: dict, raw) -> Any:
    kind = spec["type"]
    if kind == "enum":
        if str(raw) not in spec["choices"]:
            raise EffectError(f"{effect}.{spec['name']}: {raw!r} not one of {spec['choices']}")
        return str(raw)
    if kind == "bool":
        return raw if isinstance(raw, bool) else str(raw).strip().lower() in ("1", "true", "yes")
    try:
        val = int(raw) if kind == "int" else float(raw)
    except (TypeError, ValueError):
        raise EffectError(f"{effect}.{spec['name']}: {raw!r} is not a {kind}") from None
    lo, hi = spec["min"], spec["max"]
    if lo is not None and not (lo <= val <= hi):
        raise EffectError(
            f"{effect}.{spec['name']}: {val} outside [{lo}, {hi}] {spec['units']}".rstrip())
    return val


def render(name: str, samples, sr: int, params: Optional[dict] = None):
    """Process a (frames, ch) float32 array through a built-in effect. Channels preserved."""
    import numpy as np

    cls = effect_class(name)
    arr = np.asarray(samples, dtype="float32")
    arr = np.ascontiguousarray(arr[:, None] if arr.ndim == 1 else arr)
    inst, types = cls(), _declared_types(cls)
    for key, val in coerce_params(name, params).items():
        enum_cls = _enum_class(cls, types.get(key, ""))
        setattr(inst, key, enum_cls.__members__[val] if enum_cls is not None else val)
    try:
        out = np.asarray(inst.process(arr, float(sr), reset=True), dtype="float32")
    except Exception as exc:  # native errors become one `op_failed` frame, not a traceback
        raise EffectError(f"{name}: render failed: {exc}") from exc
    out = out[:, None] if out.ndim == 1 else out
    if out.shape[0] < out.shape[1]:  # pedalboard may hand back (ch, frames)
        out = out.T
    return np.ascontiguousarray(out)
