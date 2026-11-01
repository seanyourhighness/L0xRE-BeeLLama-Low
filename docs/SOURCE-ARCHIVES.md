# Historical source archives

The active public distribution branch is `main`. The following retired development branches are preserved at their exact tips by tags. These are historical snapshots, not current runtime recommendations. For current complete source, use the named source release asset linked in the [README](../README.md).

| Former branch | Retained tag | Exact commit |
| --- | --- | --- |
| `codex/bench-quality-evidence-review` | `archive/2026-11-01/codex/bench-quality-evidence-review` | `6952c8762a41677b43d28add422abe9e4ce41702` |
| `feature/windows-bridge-loader` | `archive/2026-11-01/feature/windows-bridge-loader` | `eed82efdd415ad8afc8032e193a598c3172f035a` |
| `release/l0xre-sm89-v0.4.7` | `archive/2026-11-01/release/l0xre-sm89-v0.4.7` | `8ad76c89b8a684a8f9fc5c5dfcdc180f1078beeb` |
| `release/r6-sm120-prefill-sync` | `archive/2026-11-01/release/r6-sm120-prefill-sync` | `c3088e6a6d99c61925942efc8a34de75597b17ff` |
| `release/r6-sm89-prefill-port2` | `beellama-v0.4.7-universal-r6-sm89-linux-prefill-port2` | `414a6e3ca7da22a429a2ece404fb76cedb05c253` |
| `release/r6-windows-prefill-port2` | `beellama-v0.4.7-universal-r6-windows-prefill-port2` | `d9bdf77f37f1cfb2c2616f74b2fc313234012a5f` |
| `release/sm86-b84-universal` | `archive/2026-11-01/release/sm86-b84-universal` | `be302741801080a4fa2ea713fdf9409e2ef599d5` |
| `release/sm86-prefill-80k` | `archive/2026-11-01/release/sm86-prefill-80k` | `5073ba019cbaec387530b64d33e73b7fc6a41338` |

`git clone --branch <retained-tag> <repository-url>` accepts a tag and checks out its exact commit in detached-HEAD mode. Existing release tags, assets and historical receipts are retained unchanged. References to former branches in frozen receipts describe historical provenance.

To restore a former branch when needed, create it at the exact commit above and push that branch without force. Do not move archive tags.
