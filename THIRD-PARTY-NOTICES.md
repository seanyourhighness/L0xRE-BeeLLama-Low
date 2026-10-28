# Third-party notices for v0.1

These notices accompany the components included in the architecture manifests. The original license texts in `licenses/` remain authoritative. Source snapshots preserve their copyright headers; build records identify modifications and exact inputs.

| Component | Attribution and license files |
| --- | --- |
| BeeLLaMA and ggml-derived runtime | The ggml authors and Anbeeld; `licenses/BeeLLaMA-LICENSE`. L0xRE includes modified inference, cache, vision and host integration code. |
| vLLM FLA reference kernels | `licenses/vLLM-Apache-2.0-LICENSE.txt`; pinned source commit `01efc7ef781391e744ed08c3292817a773d654e6`. Generated-source manifests retain the extracted references and modifications. |
| SGLang GDN kernels | `licenses/SGLang-Apache-2.0-LICENSE` and `licenses/GDN-KERNEL-NOTICE.md`. Generated-kernel manifests record source and cache provenance. |
| Flash Linear Attention | Songlin Yang and Yu Zhang; `licenses/Flash-Linear-Attention-MIT-LICENSE`. |
| NVIDIA CUDA runtime | Vendor dependency license in `licenses/nvidia/cuda_cudart/LICENSE`; `licenses/NVIDIA-CUDA-12.8-EULA.txt` and `licenses/NVIDIA-CUDA-13.0-EULA.txt` match the respective Linux payloads; each payload also retains its supplied EULA. |
| NVIDIA cuBLAS / cuBLASLt | `licenses/nvidia/libcublas/LICENSE`. |
| NVIDIA nvJitLink | Where included, `licenses/nvidia/libnvjitlink/LICENSE`. |
| CPython 3.14.3, Windows launcher | `licenses/CPython-LICENSE.txt`, including its Python and bzip2 notices. Linux/WSL use the system Python. |
| OpenSSL 3.0.18 | `licenses/OpenSSL-3.0.18-LICENSE.txt`. |
| zlib-ng 2.2.4 | `licenses/zlib-ng-2.2.4-LICENSE.md`; the runtime uses its zlib compatibility API. |
| libffi 3.4.4 | `licenses/libffi-3.4.4-LICENSE.txt`. |
| SQLite 3.50.4 | `licenses/SQLite-3.50.4-LICENSE.md`, with the upstream public-domain declaration. |
| XZ/liblzma 5.2.5 | `licenses/XZ-5.2.5-COPYING.txt`. The Python extension uses liblzma; no XZ command-line tools are bundled. |
| mpdecimal 4.0.0 | `licenses/mpdecimal-4.0.0-COPYRIGHT.txt`. |
| Expat | `licenses/Expat-COPYING.txt`, from the matching CPython source. |
| Unicode data | `licenses/Unicode-LICENSE.txt`. |
| Zstandard 1.5.7 | `licenses/Zstandard-1.5.7-LICENSE.txt`. |
| Microsoft Visual C++ runtimes | The unmodified `licenses/Microsoft-Visual-C-Runtime-2015-2022-License.docx`; the adjacent readable extraction is provided for convenience. |

The Python dependency versions above are recorded by CPython v3.14.3's build inputs; installed runtime queries also confirm Python 3.14.3, OpenSSL 3.0.18, SQLite 3.50.4 and the zlib-ng compatibility version. `licenses/V01-LICENSE-SOURCES.json` records the source URLs, pinned versions and downloaded notice hashes. The inspected Linux SM86/SM89 payloads bundle CUDA 12.8 libraries, while Linux/WSL SM120 bundles CUDA 13.0 libraries, with their matching `NVIDIA-CUDA-EULA.txt`. Their glibc, libstdc++, libgcc, OpenMP and Linux OpenSSL libraries come from the operating system; these system libraries are not copied into the runtime archives. Per-archive manifests and dependency inventories record this boundary.

## Model files

Model weights and the drafter are acquired separately from their pinned source repository. The Q5 vision projector is published by HermiHg in `HermiHg/Qwen3.8-27B-mmproj-Q5_K-MIX-GGUF`, revision `dbf78ac52891f19cf175abb8813997267adb0e92`. Its model card declares Apache 2.0 and identifies `Qwen/Qwen3.8-27B` as the base. The pinned LFS SHA256 matches the projector used in qualification. The upstream Qwen license is retained as `licenses/Qwen-Apache-2.0-LICENSE.txt`; model provenance is recorded in `docs/V01-PROVENANCE.md`.

Optional control-vector files are user-provided and are not redistributed by these runtime archives. The runtime license does not grant rights to an independently supplied vector. Hardware validation status and source/build verification remain separate from licensing; see the release matrix.
