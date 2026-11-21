# L0xRE v0.1.6 "Sparse"

256K context on a 12 GB card, with long-context retrieval close to dense attention in the measured tests.

## What's new

- **D-LSA sparse attention above 8K tokens.** Below 8K the model attends to everything (dense); above it each query
  attends to a sink, a 512-token window and the 4,096 most relevant earlier tokens, selected by a learned map
  (`qwen38-dlsa-d2048.l0xmap`, downloaded by the installer next to the model).
- **Exact-match retrieval links.** While a prompt is read, a rare token sequence that appeared earlier links the current
  block to those earlier blocks. This fixes the main failure of learned sparse attention, picking the wrong one of
  several similar records. Generation uses the plain sparse selection, so decode speed is unchanged.
- **DFlash2 lattice-tree verify and fused verify kernels** (from the v0.1.5 sparse runtime on the RTX 3060).

## Measured (L0xRE-V2-27b-Low + DFlash2, temperature 0)

| | dense attention | v0.1.6 |
| --- | --- | --- |
| Needle recall, 8 records in real agent transcripts, 16K-192K (RTX 5090 and RTX 3060) | 224 of 224 on each card | 223 of 224 on each card |
| Held-out retrieval (code, paths, paraphrased questions), two prompt sets, RTX 5090 | 269 of 288 | 268 of 288 |
| Responses below 8K tokens | | bit-identical to v0.1.6 without the links (logit hashes) |

Speed (prefill, vs the same runtime without the links): RTX 3060 -0.2..+0.1% from 8K to 250K, RTX 4090 -0.4..-0.1%,
RTX 5090 within 1.6%; decode unchanged. RTX 3060 12 GB: about 800 tok/s prefill and 30-50 tok/s decode at 32K-250K.

## Packages

| Package | GPUs | Validation |
| --- | --- | --- |
| `linux-sm86` | RTX 30 series | RTX 3060 12 GB |
| `linux-sm89` | RTX 40 series (the sm86 build) | RTX 4090 |
| `linux-sm120` | RTX 50 series | RTX 5090 |
| `windows-sm86` | RTX 30 series | Build-verified |
| `windows-sm89` | RTX 40 series | Build-verified |
| `windows-sm120` | RTX 50 series | Native RTX 5090 |

All six packages use the 12 GB profile, including on cards with more VRAM.

On Linux, the RTX 4090 and RTX 5090 passed fresh-install and launch smoke tests with token-for-token agreement against the RC5d kits.
The RTX 3060 ran the RC5d kit in production and completed kit qualification. The installer package did not have a separate
RTX 3060 smoke run.

### Native Windows performance

On the same RTX 5090, native Windows and Linux under WSL2 produced identical output tokens in all 14 quicksort and essay comparisons. At 261,120 prompt tokens, Windows decoded quicksort at 224.5 tok/s and the essay at 126.5 tok/s. WSL2 measured 225.7 and 122.4 tok/s respectively. These are single measured samples per case. Windows had prior warmups through 128K.

In a separate repeated cold-prompt benchmark, native Windows prefill trailed WSL2 by 3.0% at 64K, 3.6% at 128K and 4.2% at 261,120 tokens. Matching-card inference testing is pending for the Windows SM86 and SM89 builds.

The shared runtime source is in `L0xRE-v0.1.6-source.tar.gz`. The Windows port, exact host tree, CUDA gather change and incremental build recipes are in `L0xRE-v0.1.6-windows-source.tar.gz`. Each Windows ZIP records the runtime base and backend hashes in `BUILD-INFO.json`.

## Upgrading

`update.sh` (or the installer with `--update`) installs v0.1.6 next to v0.1.5 and downloads the 6 MB map; v0.1.5 is
kept and `--rollback` returns to it. The installed command lines do not change.

On Windows, stop the server, run `update.cmd` and launch with `start.cmd`. Run `update.cmd -Rollback` to restore the previous installation.

Not available with the sparse runtime: `--control-vector`, `--state-cache-dir`, `--arena-mib`, `--measure-arena`
(they are rejected with a clear message). `L0XRE_SPARSE_MATCH=0` turns the retrieval links off.

## Known limits

- One sequence at a time (`-np 1`); auxiliary agent calls share the slot through prompt-cache restores.
- The links can copy a neighbour at counting positions (for example a chapter number after "CHAPTER"). Confident
  disagreements with dense attention stay at the rate of the plain sparse runtime (1 in 862 positions measured).
- Dense mode (`L0XRE_SPARSE=0`) mis-handles some agent transcripts past about 30K tokens; it is not the default above 8K.
- Tree verify runs inside the resident window (up to 8K); deeper verify uses the 8-row chain.
