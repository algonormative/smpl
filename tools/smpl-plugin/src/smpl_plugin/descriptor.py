"""The plugin descriptor: schema, construction, validation. It answers, for one plugin,
everything a caller needs before sending it audio: who it is, what format/version it came
from, its bus layout, its parameters (ranges, units, current values), the content-address of
its state, and whether it may be memoized. `format` is an enum with `vst3`/`au` RESERVED, so
adding an external host later does not reshape the descriptor. Validation is hand-rolled over a
JSON-Schema-shaped dict — one heavy dep is enough for this tool.
"""

from __future__ import annotations

from typing import Any, Optional

SCHEMA_VERSION = 1

_STR, _BOUND = {"type": "string"}, {"type": ["number", "null"]}
_SCALAR = {"type": ["number", "string", "boolean", "null"]}


def _obj(required: list[str], **properties) -> dict:
    return {"type": "object", "required": required, "properties": properties}


SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "smpl plugin descriptor",
    **_obj(
        ["schema_version", "identity", "format", "version", "bus_layout",
         "parameters", "state_hash", "cacheable", "descriptor_hash"],
        schema_version={"type": "integer"},
        identity=_obj(["name", "vendor", "kind"], name=_STR, vendor=_STR,
                      kind={"type": "string", "enum": ["effect", "instrument"]}),
        # `vst3`/`au` are RESERVED — out of scope for this task, in scope for the schema.
        format={"type": "string", "enum": ["builtin", "vst3", "au"]},
        version=_obj(["library", "library_version", "class"],
                     library=_STR, library_version=_STR, **{"class": _STR}),
        # Built-ins are channel-agnostic → the string "any"; an external plugin will report a
        # fixed integer count here.
        bus_layout=_obj(["input_channels", "output_channels", "sample_rate"],
                        input_channels={"type": ["integer", "string"]},
                        output_channels={"type": ["integer", "string"]}, sample_rate=_STR),
        parameters={"type": "array", "items": _obj(
            ["name", "type", "min", "max", "default", "units", "value"],
            name=_STR, min=_BOUND, max=_BOUND, default=_SCALAR, units=_STR, value=_SCALAR,
            range_source=_STR, choices={"type": "array", "items": _STR},
            **{"type": {"type": "string", "enum": ["float", "int", "bool", "enum"]}})},
        state_hash=_STR, cacheable={"type": "boolean"}, descriptor_hash=_STR,
    ),
}

_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool,
          "number": (int, float), "integer": int, "null": type(None)}


def _type_ok(value: Any, spec) -> bool:
    for name in spec if isinstance(spec, list) else [spec]:
        py = _TYPES.get(name)
        if name in ("number", "integer") and isinstance(value, bool):
            continue  # bool is an int in Python; JSON Schema says it is not a number
        if py is not None and isinstance(value, py):
            return True
    return False


def _check(value: Any, spec: dict, path: str, errors: list[str]) -> None:
    if "type" in spec and not _type_ok(value, spec["type"]):
        errors.append(f"{path}: expected type {spec['type']}, got {type(value).__name__}")
        return
    if "enum" in spec and value not in spec["enum"]:
        errors.append(f"{path}: {value!r} not one of {spec['enum']}")
    if isinstance(value, dict):
        for key in spec.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required field {key!r}")
        for key, sub in spec.get("properties", {}).items():
            if key in value:
                _check(value[key], sub, f"{path}.{key}" if path else key, errors)
    if isinstance(value, list) and "items" in spec:
        for i, item in enumerate(value):
            _check(item, spec["items"], f"{path}[{i}]", errors)


def validate_descriptor(desc: Any) -> list[str]:
    """Structural + semantic validation. Empty list ⇒ valid."""
    errors: list[str] = []
    _check(desc, SCHEMA, "", errors)
    if errors or not isinstance(desc, dict):
        return errors
    for i, p in enumerate(desc.get("parameters") or []):
        lo, hi = p.get("min"), p.get("max")
        if (lo is None) != (hi is None):
            errors.append(f"parameters[{i}]: min and max must both be set or both be null")
        elif lo is not None and lo > hi:
            errors.append(f"parameters[{i}]: min {lo} > max {hi}")
        if p.get("type") == "enum" and not p.get("choices"):
            errors.append(f"parameters[{i}]: enum parameter needs a non-empty 'choices'")
    return errors


def state_hash(effective_params: dict, *, store: bool = True) -> str:
    """Content-address the effective parameter dict — the plugin's saved state — as a
    canonical-JSON CAS blob, so a preset made offline is referenceable by hash alone."""
    from smplstream import cas, hashing, memo

    blob = memo.canonical_json(effective_params)
    return cas.put_blob(blob, "application/json") if store else hashing.blob_hash(blob)


def load_state(hash_: str) -> dict:
    """Read a `--state blake3:…` param blob back out of the CAS."""
    import json

    from smplstream import cas

    data = json.loads(cas.get_path(hash_).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"state blob {hash_} is not a parameter object")
    return data


def describe(name: str, params: Optional[dict] = None, *, store: bool = True) -> dict:
    """Build the descriptor for a built-in effect, with `params` applied as current values."""
    from smplstream import hashing, memo

    from . import hostbuiltin as host

    effective = host.coerce_params(name, params)
    specs = host.parameter_specs(name)
    for spec in specs:
        spec["value"] = effective[spec["name"]]
    desc = {
        "schema_version": SCHEMA_VERSION,
        "identity": {"name": name, "vendor": "pedalboard", "kind": "effect"},
        "format": "builtin",
        "version": {"library": "pedalboard", "library_version": host.pedalboard_version(),
                    "class": name},
        # Built-in DSP adapts to whatever it is handed and runs at the host sample rate.
        "bus_layout": {"input_channels": "any", "output_channels": "any", "sample_rate": "host"},
        "parameters": specs,
        "state_hash": state_hash(effective, store=store),
        # Every built-in is pure DSP over (samples, sr, params) — same in, same out.
        "cacheable": True,
    }
    desc["descriptor_hash"] = hashing.blob_hash(memo.canonical_json(desc))
    return desc
