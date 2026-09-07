# `smpl plugin` — design note (seed)

Part of *The Running Studio* plan (https://claude.ai/code/artifact/9b6a1fd0-ccd8-48e0-866a-02c404cb0150).
Both follow the existing `tools/<name>` pattern: an isolated `uv tool install ./tools/smpl-<name>`, discovered on
PATH as `smpl-<name>`, reachable as `smpl <name>`; glue is PATH discovery only.

## Scope
Wraps Spotify's pedalboard (VST3 and AU on macOS) as an smpl op. Two entry paths: an **instrument** accepts note
and control events (a `midi` frame or inline events) plus a preset and renders audio; an **effect** accepts
audio and processes it, with automation or a sidechain input where needed. Both expose the plugin's identity,
format, version, bus layout, parameters (ranges, units, current values) and saved state; state is a
content-addressed blob in the CAS so a preset made offline can be referenced by hash from a live Ginger node.
Deterministic plugins memoize; ones that are not declare `cacheable: false`.

```
smpl plugin list --json
smpl plugin describe "Serum" --json        # params, ranges, units, presets, descriptor hash
smpl read dry.wav | smpl plugin --name "Serum" --state blake3:… --param cutoff=0.4 | smpl write wet.wav
smpl plugin state save "Serum" --from … > blake3:…
```

Acceptance for the first task: a dry sample through a VST3 with a saved state hash renders byte-identically
twice and hits the memo the second time; `describe --json` lists params with ranges. Needs a machine with the
plugin installed; CI covers argument parsing, descriptor schema and memo-key derivation with a stub host.

