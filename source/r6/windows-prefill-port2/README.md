# R6 Windows prefill port2 candidates

Apply `source/r6/overlay-windows` after preparing the common C07 R6 source. These exact overrides admit SM86/SM89 for native long-context QK16 and masked GDN, retaining the existing shape, stride and rollback guards. The SM89 Q4 MMQ path remains opt-in.

The Windows profile enables INT8 prefill through the native bridge initialization ABI that was already present in the runtime. It also selects the new masked-GDN DLL and matching cubins. The launcher requires `--qualification-probe`; target Windows GPU dispatch, throughput, quality and capacity tests are pending.

`build.cmd` builds the masked-GDN DLL with Windows CUDA 13.3/MSVC. `assemble-cubins.cmd` assembles seven kernels for each architecture from the same PTX as the Linux bridge. `verify.cmd` exercises unsupported-call rejection and output guards without GPU work. The recipes record the build workspace at `C:\work\l0xre-r6-gdn-port-20261007`; adjust that location and toolchain paths when rebuilding elsewhere.

Offline evidence is in `evidence/windows-prefill-port2`. Both native runtime and bridge builds passed, the Windows DLL ABI checks passed, and the fourteen cubin entry names and architectures were checked. None of these checks establishes Windows GPU performance or quality. The existing Linux SM86 certificate retains its original scope and artifact.
