"""smplstream — the composable media-analysis wire protocol.

NDJSON frames over a content-addressed store, with canonical-PCM hashing and pipeline
memoization. See ``spec.md`` (the normative contract) in the repository root.

Two version constants, two jobs (spec → *Versioning*, *Memoization*):
  - ``PROTOCOL_VERSION`` (per-frame ``v``): bumps only on breaking reinterpretation of
    existing fields. Additive kinds/fields/feature-keys do NOT bump it.
  - ``SCHEMA``: gates protocol-schema cache invalidation. Per-op correctness rides on each
    op's ``op_version``, not on ``SCHEMA`` — so a one-op upgrade doesn't nuke the cache.
"""

from __future__ import annotations

from .ids import PROTOCOL_VERSION  # per-frame `v`

# Protocol-schema cache-invalidation gate (mirrors smplcat's SCHEMA_VERSION discipline).
SCHEMA = "smplstream/1"

from . import cas, conformance, errors, frames, hashing, memo, memostore, ndjson, select  # noqa: E402
from .errors import (  # noqa: E402
    ERROR_CODES,
    IntegrityError,
    PathSafetyError,
    ProtocolError,
    ResolutionError,
    SmplError,
    error_frame,
)
from .frames import (  # noqa: E402
    KINDS,
    MAX_INLINE_BYTES,
    VECTOR_INLINE_MAX_DIM,
    audio_frame,
    feature_frame,
    image_frame,
    marker_frame,
    text_frame,
    validate_frame,
    vector_frame,
    verdict_frame,
)
from .hashing import (  # noqa: E402
    FORMAT_TAG_FLOAT32LE,
    audio_hash_bytes,
    audio_hash_file,
    audio_hash_from_pcm,
    blob_hash,
    decode_canonical_bytes,
    decode_canonical_file,
    probe_audio,
)
from .ids import content_id, mint_id, random_id  # noqa: E402
from .memo import canonicalize_params, memo_key, tool_version_fingerprint  # noqa: E402
from .ndjson import read_frames, write_frame, write_frames  # noqa: E402
# NB: export only `resolve_single_audio`, NOT the bare `select` function — that name is the
# submodule (`smplstream.select`), and the CLI/tests use it as the module. Re-binding the
# function over it would shadow the module. Call `smplstream.select.select(...)`.
from .select import resolve_single_audio  # noqa: E402

__all__ = [
    "PROTOCOL_VERSION",
    "SCHEMA",
    "cas",
    "conformance",
    "errors",
    "frames",
    "hashing",
    "memo",
    "memostore",
    "ndjson",
    "select",
    "ERROR_CODES",
    "SmplError",
    "IntegrityError",
    "PathSafetyError",
    "ProtocolError",
    "ResolutionError",
    "error_frame",
    "KINDS",
    "MAX_INLINE_BYTES",
    "VECTOR_INLINE_MAX_DIM",
    "audio_frame",
    "text_frame",
    "feature_frame",
    "marker_frame",
    "image_frame",
    "vector_frame",
    "verdict_frame",
    "validate_frame",
    "FORMAT_TAG_FLOAT32LE",
    "audio_hash_file",
    "audio_hash_bytes",
    "audio_hash_from_pcm",
    "blob_hash",
    "decode_canonical_file",
    "decode_canonical_bytes",
    "probe_audio",
    "content_id",
    "mint_id",
    "random_id",
    "memo_key",
    "canonicalize_params",
    "tool_version_fingerprint",
    "read_frames",
    "write_frame",
    "write_frames",
    "resolve_single_audio",
]
