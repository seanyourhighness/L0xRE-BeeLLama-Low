# L0xRE v0.1.6 "Sparse"

256K context on a 12 GB card, with long-context retrieval at dense-attention parity.

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
| Needle recall, 8 records in real agent transcripts, 16K-192K (RTX 5090 / RTX 4090) | 224 / 224 of 224 | 223 / 223 |
| Held-out retrieval (code, paths, paraphrased questions), two prompt sets, RTX 5090 | 269 of 288 | 268 |
| Responses below 8K tokens | | bit-identical to v0.1.6 without the links (logit hashes) |

Speed (prefill, vs the same runtime without the links): RTX 3060 -0.2..+0.1% from 8K to 250K, RTX 4090 -0.4..-0.1%,
RTX 5090 within 1.6%; decode unchanged. RTX 3060 12 GB: about 800 tok/s prefill and 30-50 tok/s decode at 32K-250K.

## Packages

| Package | GPUs | Qualified on |
| --- | --- | --- |
| `linux-sm86` | RTX 30 series | RTX 3060 12 GB |
| `linux-sm89` | RTX 40 series (the sm86 build) | RTX 4090 |
| `linux-sm120` | RTX 50 series | RTX 5090 |

**Windows stays on v0.1.5** for this release. The 16gb profile runs the qualified 12 GB profile; larger VRAM classes are
not measured yet.

## Upgrading

`update.sh` (or the installer with `--update`) installs v0.1.6 next to v0.1.5 and downloads the 6 MB map; v0.1.5 is
kept and `--rollback` returns to it. The installed command lines do not change.

Not available with the sparse runtime: `--control-vector`, `--state-cache-dir`, `--arena-mib`, `--measure-arena`
(they are rejected with a clear message). `L0XRE_SPARSE_MATCH=0` turns the retrieval links off.

## Known limits

- One sequence at a time (`-np 1`); auxiliary agent calls share the slot through prompt-cache restores.
- The links can copy a neighbour at counting positions (for example a chapter number after "CHAPTER"). Confident
  disagreements with dense attention stay at the rate of the plain sparse runtime (1 in 862 positions measured).
- Dense mode (`L0XRE_SPARSE=0`) mis-handles some agent transcripts past about 30K tokens; it is not the default above 8K.
- Tree verify runs inside the resident window (up to 8K); deeper verify uses the 8-row chain.
