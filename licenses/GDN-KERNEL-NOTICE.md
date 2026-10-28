# GDN kernel provenance

Architecture payloads may include generated SGLang/Flash Linear Attention GDN cubins and their host wrappers. Each architecture manifest identifies the files actually bundled. The source archive retains the compiled-kernel metadata, source hashes and wrapper build commands; cubins and launch metadata must match their target architecture.

SGLang is licensed under Apache 2.0. Its FLA kernels are adapted from Flash Linear Attention, licensed under MIT and credited to Songlin Yang and Yu Zhang in the source files. The original license texts are retained alongside this notice.

Upstream projects: `https://github.com/sgl-project/sglang` and `https://github.com/fla-org/flash-linear-attention`.

Legacy R6 documentation described these files in its optional fast-8k profile. The v0.1 architecture manifests and sealed profiles define their current inclusion and dispatch; that legacy description is not a v0.1 coverage claim.
