# `smpl check` — design note (seed)

Part of *The Running Studio* plan (https://claude.ai/code/artifact/9b6a1fd0-ccd8-48e0-866a-02c404cb0150).
Both follow the existing `tools/<name>` pattern: an isolated `uv tool install ./tools/smpl-<name>`, discovered on
PATH as `smpl-<name>`, reachable as `smpl <name>`; glue is PATH discovery only.

## Scope
A YAML check file scoped to a take, stem or passage. The baseline has two halves: known-good captures and a
known-bad control; a metric that fails to separate them by the declared margin is rejected as insensitive
(`validity: invalid`, no pass/fail about the music). Verdicts are observations about the commit and capture
they judged, written through the ledger's writer when `--emit-journal` is given, and kept apart from human
feedback and from invalid measurements.

```
smpl check calibrate checks/ep.yaml
smpl check run checks/ep.yaml --take blake3:… --stem choir --range 41:49 --emit-journal
smpl check explain mud --take blake3:…
```

First port: the Basilica EP battery (`music-hub/projects/basilica/work/ep_qc.py`) as a check file with good
and bad controls. The record contract these verdicts use is `music-hub/contracts/v0`.
