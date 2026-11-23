# L0xRE v0.1.6.1 (RC5d r2)

Runtime update on top of [v0.1.6 "Sparse"](V016-RELEASE.md), consolidating the RC5d r2 decode/sparse optimizations and the verification-consistent server chat template for Linux and native Windows (`sm86`, `sm89`, `sm120`).

## What's new

- **Padded Staged Q3 Output Head** (`BT_HEAD8=staged-padded`): 52-word shared-memory row stride (`head8-tuned.cu`) eliminating shared-memory bank conflicts during 8-row verification.
- **Stable 32-bit D-LSA Radix Top-K** (`BT_RADIX_TOPK=1`, `BT_RADIX_TOPK_TUNE=stable32`): faster 32-bit radix tile selection (`tuned-tile.cuh`), now enabled natively on `sm86`, `sm89`, and `sm120`.
- **Indexed Record-Copy DFlash2 Proposal Hook** (`L0XRE_RECORD_COPY=1`): host-side structured record and source continuation proposals using stock DFlash2 verification and rollback.
- **Loop-Stopping Verification-Consistent Chat Template** (`tool-verification-consistent.jinja`, SHA-256 `8600019b72ecff2b7f39701e3772873fd265cb2b0908b850d0b5d7fbe17d90d0`): shipped in every Linux kit and Windows ZIP and passed as the default `--chat-template-file` by `launch.py` and `start.ps1` (overridable via explicit `--chat-template-file`).
- **Windows gather fix preserved**: retains the RC5d `l0xre_sparse_gather_k1`/`k2` dispatch fix across `sm86`, `sm89`, and `sm120`.

## Packages & Verification

| Package | GPUs | Validation |
| --- | --- | --- |
| `L0xRE-v0.1.6.1-linux-sm86.tar.zst` | RTX 30 series | Build + CUDA symbols verified; `sm86` kit bytes verified live on RTX 4090; RC5d r2 optimizations qualified on RTX 3060 12 GB |
| `L0xRE-v0.1.6.1-linux-sm89.tar.zst` | RTX 40 series (`sm86` kit) | Live smoke on RTX 4090 (SHA256SUMS, short code, tool call, >21K sparse prompt, all 4 runtime markers) |
| `L0xRE-v0.1.6.1-linux-sm120.tar.zst` | RTX 50 series | Live smoke on RTX 5090 (SHA256SUMS, short code, tool call, >21K sparse prompt, all 4 runtime markers) |
| `L0xRE-v0.1.6.1-windows-sm86.zip` | RTX 30 series | Build-verified (CUDA symbols + ZIP SHA256SUMS) |
| `L0xRE-v0.1.6.1-windows-sm89.zip` | RTX 40 series | Build-verified (CUDA symbols + ZIP SHA256SUMS) |
| `L0xRE-v0.1.6.1-windows-sm120.zip` | RTX 50 series | Native Windows live smoke on RTX 5090 (SHA256SUMS, short code, tool call, >21K sparse prompt, all 4 runtime markers) |

Source archives: `L0xRE-v0.1.6.1-source.tar.gz` (commit `d0e4502431026cf0baba178bc85b140637755e6d`, tag `l0xre-v0.1.6.1-source`) and `L0xRE-v0.1.6.1-windows-source.tar.gz`.
