"""Consolidated multimodal report (ticket vault-31r9 — the terminal presentation).

`smpl view` is the *end* of an analysis pipe: it reads the accumulated frame
stream and renders one human/LLM-facing markdown report summarizing everything
the upstream tools produced — every `feature` key/value with units, the
`marker` frames (count + first few times), the `image` frames with their role
**and resolved CAS path** (so an LLM can open the PNG), and any `error` frames
surfaced prominently.

Unlike the other analysis ops, `view` does not derive *per-audio-frame*
artifacts; it derives **one** `text` frame (role ``report``) over the whole
stream. The thin `smpl view` subcommand passes every input frame through first
(so images stay resolvable downstream), then appends the report frame, and also
prints the report to stderr for the human.

Pure functions over a list of frame dicts. The only "heavy" dependency is the
CAS path lookup (``smplstream.cas``); there is no librosa/matplotlib here, so
the import stays light. CAS path resolution is wrapped so a missing/lazy blob
degrades to an annotation in the report rather than crashing the whole view.

Feature-key spellings are NOT minted here — `view` only *reports* whatever keys
the upstream ops emitted, so it is registry-neutral (it owns no rows in
feature-keys.md).
"""

from __future__ import annotations

from typing import Optional

OP = "view"
OP_VERSION = "view@1"

# How many marker times / per-marker-frame points to spell out before eliding.
_MARKER_PREVIEW = 5
# How many image frames to detail; the rest are summarized as a count.
# (Generous — a typical describe pipe emits a handful of spectrograms.)
_IMAGE_PREVIEW = 64


def _fmt_num(v) -> str:
    """Render a number compactly: ints bare, floats trimmed, None/inf legible."""
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v != v:  # NaN
            return "NaN"
        if v in (float("inf"), float("-inf")):
            return "+inf" if v > 0 else "-inf"
        # Trim trailing zeros while staying round-trippable enough for a report.
        return f"{v:.4f}".rstrip("0").rstrip(".") or "0"
    return str(v)


def _flatten_feature(value, prefix: str = "") -> list[tuple[str, str]]:
    """Flatten a feature value into (key, rendered-value) rows.

    Handles the registry's two shapes: scalars (``loudness.integrated_lufs``) and
    the ``{mean, stdev}`` statistic convention used by frame-aggregated Essentia
    features. Nested dicts recurse with a dotted suffix; lists render inline.
    """
    rows: list[tuple[str, str]] = []
    if isinstance(value, dict):
        # Statistic convention: {mean, stdev} renders as "mean (±stdev)".
        if set(value) == {"mean", "stdev"} or set(value) == {"mean", "stdev", "min", "max"}:
            mean = _fmt_num(value.get("mean"))
            stdev = _fmt_num(value.get("stdev"))
            rows.append((prefix.rstrip("."), f"{mean} (±{stdev})"))
            return rows
        for k, v in value.items():
            rows.extend(_flatten_feature(v, f"{prefix}{k}."))
        return rows
    if isinstance(value, (list, tuple)):
        rendered = ", ".join(_fmt_num(x) for x in value[:8])
        if len(value) > 8:
            rendered += f", … (+{len(value) - 8})"
        rows.append((prefix.rstrip("."), f"[{rendered}]"))
        return rows
    rows.append((prefix.rstrip("."), _fmt_num(value) if isinstance(value, (int, float)) or value is None else str(value)))
    return rows


def _unit_for(key: str) -> str:
    """Best-effort unit hint from a feature key's suffix / namespace.

    The spec implies units from the namespaced key; for ad-hoc keys the unit is
    suffixed (``_db``, ``_hz``, ``_lufs``, ``_dbtp``, ``_dbfs``, ``_s``). This is
    a *display* convenience for the report — it never rewrites the key.
    """
    k = key.lower()
    # Envelope / width leaf names carry mixed units the suffix table can't disambiguate
    # (e.g. `rise_slope_db_ms` is dB/ms, not ms) — resolve them by leaf name first.
    leaf = k.rsplit(".", 1)[-1]
    leaf_units = {
        "peak_db_over_floor": "dB",
        "attack_ms_10_90": "ms",
        "rise_slope_db_ms": "dB/ms",
        "t20_ms": "ms",
        "early_decay_slope": "dB/ms",
        "sustain_ratio_150ms": "ratio",
        "correlation": "−1..1",
        "side_mid_ratio": "ratio",
        # movement / clarity (vault-1fxy) — leaf names the suffix table can't disambiguate.
        # (the `*_db` keys below fall through to the `_db` suffix → "dB" automatically.)
        "hf_silence_pct": "%",
        "mud_presence_ratio": "ratio",
        "presence_focus_ratio": "ratio",
        "presence_transient_ratio": "crest",
    }
    if leaf in leaf_units:
        return leaf_units[leaf]
    for suffix, unit in (
        ("_lufs", "LUFS"),
        ("_dbtp", "dBTP"),
        ("_dbfs", "dBFS"),
        ("_db_ms", "dB/ms"),
        ("_db", "dB"),
        ("_hz", "Hz"),
        ("_cents", "cents"),
        ("_lu", "LU"),
        ("_ms", "ms"),
        ("_s", "s"),
    ):
        if k.endswith(suffix):
            return unit
    if k.startswith("timbre."):
        return "0–100"
    if k == "rhythm.bpm" or k.endswith(".bpm"):
        return "BPM"
    if k.endswith("confidence") or k.endswith("_confidence"):
        return "0–1"
    return ""


def _resolve_image_path(hash_: Optional[str]) -> str:
    """Resolve an image frame's CAS hash to a filesystem path string.

    Degrades gracefully: a missing/lazy/malformed blob is annotated in-report
    rather than raising, so one unresolvable image can't sink the whole report.
    """
    from smplstream import cas

    if not hash_:
        return "(no hash)"
    try:
        return str(cas.get_path(hash_))
    except Exception as exc:  # FileNotFoundError, PathSafetyError, lazy promise, …
        return f"(unresolved: {exc})"


def build_report(frames: list[dict]) -> str:
    """Render the accumulated frame stream into a consolidated markdown report.

    Sections (only emitted when non-empty, except the header):
      - **Errors** (first, prominent) — every `error` frame's code/message/of.
      - **Features** — a table of every `feature` frame's key/value with units.
      - **Markers** — per `marker` frame: role, count, and the first few times.
      - **Images** — per `image` frame: role + resolved CAS path (LLM-openable).
      - **Audio / text** — a short inventory so the report is self-describing.

    Pure over the frame list; only touches the CAS for image-path resolution.
    """
    lines: list[str] = []

    kinds: dict[str, int] = {}
    for f in frames:
        kinds[f.get("kind", "?")] = kinds.get(f.get("kind", "?"), 0) + 1

    audios = [f for f in frames if f.get("kind") == "audio"]
    # `sheet` feature frames carry a whole sidecar document (per-bar rows), not registry keys:
    # they are listed under "Listening sheets" with their file paths, not flattened here.
    features = [f for f in frames if f.get("kind") == "feature" and f.get("op") != "sheet"]
    markers = [f for f in frames if f.get("kind") == "marker"]
    images = [f for f in frames if f.get("kind") == "image"]
    texts = [f for f in frames if f.get("kind") == "text" and f.get("op") != "sheet"]
    sheets = [f for f in frames if f.get("kind") == "image" and f.get("op") == "sheet"]
    errors = [f for f in frames if f.get("kind") == "error"]

    lines.append("# smpl analysis report")
    lines.append("")
    inventory = ", ".join(f"{n}× {k}" for k, n in sorted(kinds.items())) or "(empty stream)"
    lines.append(f"**{len(frames)} frame(s):** {inventory}")
    lines.append("")

    # --- Errors first, prominently (spec → Error model: surface root cause). ---
    if errors:
        lines.append(f"## ⚠ Errors ({len(errors)})")
        lines.append("")
        lines.append("| code | message | of | op |")
        lines.append("|---|---|---|---|")
        for e in errors:
            d = e.get("data") or {}
            lines.append(
                f"| `{d.get('code', '?')}` | {d.get('message', '')} "
                f"| `{d.get('of', e.get('of', '—'))}` | {d.get('op', e.get('op', '—'))} |"
            )
        lines.append("")

    # --- Features: every key/value with a unit. ---
    if features:
        lines.append(f"## Features ({len(features)} frame(s))")
        lines.append("")
        lines.append("| key | value | unit | role | op |")
        lines.append("|---|---|---|---|---|")
        for feat in features:
            role = feat.get("role", "—")
            op = feat.get("op", "—")
            data = feat.get("data") or {}
            for key, rendered in _flatten_feature(data):
                lines.append(f"| `{key}` | {rendered} | {_unit_for(key)} | {role} | {op} |")
        lines.append("")

    # --- Markers: count + first few times per frame. ---
    if markers:
        lines.append(f"## Markers ({len(markers)} frame(s))")
        lines.append("")
        for mk in markers:
            pts = mk.get("data") or []
            role = mk.get("role", "marker")
            times = []
            for p in pts[:_MARKER_PREVIEW]:
                t = p.get("t")
                label = p.get("label")
                times.append(f"{_fmt_num(t)}s" + (f" ({label})" if label else ""))
            more = f", … (+{len(pts) - _MARKER_PREVIEW})" if len(pts) > _MARKER_PREVIEW else ""
            preview = ", ".join(times) if times else "(none)"
            lines.append(f"- **{role}** — {len(pts)} point(s): {preview}{more}")
        lines.append("")

    # --- Images: role + resolved CAS path so an LLM can open the file. ---
    if images:
        lines.append(f"## Images ({len(images)} frame(s))")
        lines.append("")
        lines.append("| role | media | path |")
        lines.append("|---|---|---|")
        for img in images[:_IMAGE_PREVIEW]:
            role = img.get("role", "image")
            media = img.get("media", "image/png")
            path = _resolve_image_path(img.get("hash"))
            lines.append(f"| {role} | {media} | `{path}` |")
        if len(images) > _IMAGE_PREVIEW:
            lines.append("")
            lines.append(f"_… and {len(images) - _IMAGE_PREVIEW} more image frame(s)._")
        lines.append("")

    # --- Listening sheets: PNG + the sidecar files a reader should cite instead of pixels. ---
    if sheets:
        lines.append(f"## Listening sheets ({len(sheets)})")
        lines.append("")
        lines.append("Cite the sidecar numbers (per-bar rows), not readings off the PNG.")
        lines.append("")
        lines.append("| role | png | sidecar json | sidecar markdown |")
        lines.append("|---|---|---|---|")
        for img in sheets:
            p = img.get("params") or {}
            lines.append(
                f"| {img.get('role', 'sheet')} | `{_resolve_image_path(img.get('hash'))}` "
                f"| `{_resolve_image_path(p.get('json_hash'))}` "
                f"| `{_resolve_image_path(p.get('markdown_hash'))}` |"
            )
        lines.append("")

    # --- Audio + text inventory (self-describing tail). ---
    if audios:
        lines.append(f"## Audio ({len(audios)} frame(s))")
        lines.append("")
        for a in audios:
            meta = a.get("meta") or {}
            role = a.get("role", "audio")
            dur = _fmt_num(meta.get("dur"))
            sr = meta.get("sr", "?")
            ch = meta.get("ch", "?")
            lines.append(f"- **{role}** — {dur}s · {sr} Hz · {ch}ch · `{a.get('id', '?')}`")
        lines.append("")

    if texts:
        lines.append(f"## Text / captions ({len(texts)} frame(s))")
        lines.append("")
        for t in texts:
            role = t.get("role", "text")
            body = t.get("data", "")
            if isinstance(body, str):
                snippet = body if len(body) <= 200 else body[:200] + "…"
            else:
                snippet = str(body)
            lines.append(f"- **{role}:** {snippet}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def view_frames(frames: list[dict]) -> list[dict]:
    """Build the report and return the derived frame(s) to append after passthrough.

    Returns a single `text` frame (role ``report``) carrying the markdown report.
    Lineage: the report ``of`` points at the first audio frame's id when present
    (the subject of the analysis), and ``lineage`` lists every input frame id, so
    provenance closes and the report is anchored to what it summarizes.

    If the report exceeds the inline `data` ceiling it is moved to the CAS and a
    `text` frame referencing it by `hash` is returned instead (spec → *Inline
    payloads & size limits*: text > 64 KiB MUST move to CAS).
    """
    from smplstream import cas, frames as F
    from smplstream.frames import MAX_INLINE_BYTES, mint_id

    report = build_report(frames)

    of = None
    for f in frames:
        if f.get("kind") == "audio":
            of = f.get("id")
            break
    lineage = [f["id"] for f in frames if f.get("id")]

    params = {"op_version": OP_VERSION, "input_frames": len(frames)}

    # Inline unless oversized. text_frame inlines `data`; if the markdown is huge,
    # store it in the CAS as text/plain and emit a hash-referencing text frame.
    if len(report.encode("utf-8")) <= MAX_INLINE_BYTES - 1024:  # headroom for JSON envelope
        out = F.text_frame(
            report,
            role="report",
            of=of,
            op=OP,
            op_version=OP_VERSION,
            params=params,
            lineage=lineage or None,
        )
        return [out]

    h = cas.put_blob(report.encode("utf-8"), "text/plain")
    frame = {
        "kind": "text",
        "role": "report",
        "hash": h,
        "media": "text/plain",
        "op": OP,
        "op_version": OP_VERSION,
        "params": params,
    }
    if of is not None:
        frame["of"] = of
    if lineage:
        frame["lineage"] = lineage
    return [mint_id(frame)]
