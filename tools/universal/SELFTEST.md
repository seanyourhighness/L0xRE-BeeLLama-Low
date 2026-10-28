# Hardware self-test

Close GPU apps, browsers and overlays before testing. Use the same model paths,
steering flags and preset as your normal server command:

```sh
./l0xre selftest -m /models/target.gguf -md /models/draft.gguf --mmproj /models/mmproj.gguf --memory-profile 12gb
```

On native Windows, run `l0xre.cmd selftest` with Windows model paths. The command
owns a separate local server and shuts it down afterward. Existing servers are
not reused or stopped. The normal VRAM preflight selects a validated arena and
can refuse before loading. `--output DIRECTORY` chooses a new receipt directory;
existing output directories are never overwritten.

The protocol records GPU/driver/OS/WSL metadata, free VRAM before load, the chosen
preset/arena/preflight decision and model/runtime/reference hashes. It compares
greedy output and every populated top10 value with a shipped reference, checks
positive live reuse and exact live/cold output, sends one vision request with
known ground truth, then records short narrative/code timings and adapter peak
VRAM. These short samples are not canonical bench.sh results. Peak adapter usage
includes desktop processes; it is not a per-process allocation measurement.

`--deep` adds one short/128K/short depth smoke. It requires a preset with at least
128K context. It is optional and does not silently truncate a smaller preset.

The receipt directory contains `receipt.json`, a readable `SUMMARY.txt` and raw
request/server diagnostics. Submit the JSON using the runtime repository's
**Hardware validation report** issue form; the summary prints its address. A
selftest does not submit an issue or turn a build-only architecture into a
certified one automatically. Reports for custom vector/preset combinations need
a matching shipped reference; an unknown combination refuses instead of using
an unrelated golden output.

# Reference contract for release tooling

The architecture manifest declares:

```json
{"selftest":{"reference":{"path":"architectures/sm86/selftest/reference.json","sha256":"..."}}}
```

The referenced document contains `architecture` and `variants`. Each variant has
`configuration_sha256`, `greedy` (`request` and expected `choices`), and `vision`
(`request`, `printed_text`, `shapes`, and a color-name-to-RGB map). References are
assembled from qualified cold receipts and bundled before running selftest on
the final extracted artifacts. Reference capture is not a public selftest mode.

`configuration_signature()` binds the actual selected arena, all runtime argv
and environment settings, model hashes, vector settings and numerical artifact
hashes. It excludes launcher/reference metadata and machine-specific CPU/GPU
placement, avoiding a hash depending on itself while allowing portable tester
receipts. Launcher integrity checks still verify the complete declared payload.

There is no tolerance for token/top10 differences and no empty-logprob pass.
Both a failure JSON and summary are written when startup, reference selection or
a protocol check fails. Real-card selftest receipts remain a release gate; CPU
HTTP-fixture tests are not evidence of GPU certification.

Selftest always launches with non-interactive mode and closed stdin, including on Windows where a NUL handle may report as a TTY. When full speed does not fit, pass `--accept-slower` or set `L0XRE_ACCEPT_SLOWER=1` to accept a measured slower rung. An explicit validated `--arena-mib` is also consent. Otherwise the selftest fails before loading. `startup.vram_preflight` records `rung`, `arena_mib`, `ubatch`, `free_mib`, `required_mib`, `consent`, and `speed_cost_pct`; reference identity binds the effective arena and ubatch.
