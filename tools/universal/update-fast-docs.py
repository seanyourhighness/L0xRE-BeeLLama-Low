#!/usr/bin/env python3
"""Update release metadata without relabeling inherited architecture payloads."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[2]
tag='beellama-v0.4.7-universal-r5-fast'
artifact='l0xre-beellama-low-v0.4.7-wsl-linux-x86_64-universal-r5-fast'
s=(R/'README.md').read_text()
old='**October 3 SM86 prefill update:**'
banner='''**October 7 fast update:** [S71 fast profile](FAST-RELEASE.md) on RTX3060 measured **53.88 t/s code**, **29.13 t/s narrative**, **567.58 t/s10K /469.42 t/s77K prefill**, and **134/150 pass@3** versus the paired baseline134/150. Pass@1 is127/150 versus131/150; known output failures and guard events remain documented. Use `--profile fast` with the **Q2_K** drafter and `numactl`. This Linux/WSL archive contains the original certified SM86 bytes. SM89/SM120 have [shared source and kernel compilation preparation](PARITY.md); their complete fast runtimes and hardware qualification are pending. Windows remains at r4.

**Earlier October 3 SM86 prefill update:**'''
assert old in s;s=s.replace(old,banner,1)
s=s.replace('**[Universal release candidate](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/releases/tag/beellama-v0.4.7-universal-r4)** · [Archive checksums](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/releases/download/beellama-v0.4.7-universal-r4/CHECKSUMS.txt)',f'**[Linux fast update]'+f'(https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/releases/tag/{tag})** · [Archive checksums](https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/releases/download/{tag}/CHECKSUMS.txt) · [Fast profile instructions](FAST-RELEASE.md)')
s=s.replace('https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/releases/download/beellama-v0.4.7-universal-r4/l0xre-beellama-low-v0.4.7-wsl-linux-x86_64-universal-r4.tar.zst',f'https://github.com/seanyourhighness/L0xRE-BeeLLama-Low/releases/download/{tag}/{artifact}.tar.zst')
s=s.replace('SM86 champion components tested on RTX 3060; inherited SM89 and SM120 receipts included; common launcher validated separately','SM86 S71 fast components certified; SM89/SM120 retained with common source/kernel ports prepared, hardware gates pending')
s=s.replace('`L0xRE-27b-Low.gguf` + `Qwen3.8-27B-DFlash2-Q4_K_M.gguf`','`L0xRE-27b-Low.gguf`; Q2_K for `fast`, Q4_K_M for legacy profiles')
s=s.replace('tar --zstd -xf l0xre-beellama-low-v0.4.7-wsl-linux-x86_64-universal-r4.tar.zst',f'tar --zstd -xf {artifact}.tar.zst').replace('cd l0xre-beellama-low-v0.4.7-wsl-linux-x86_64-universal-r4',f'cd {artifact}')
s=s.replace('Download the Linux archive and `CHECKSUMS.txt` above into one directory.','For the new fast profile, follow [FAST-RELEASE.md](FAST-RELEASE.md). The compatibility-profile example below retains Q4_K_M/N3. Download the Linux archive and `CHECKSUMS.txt` above into one directory.')
s=s.replace('- Linux/WSL: Ubuntu 24.04 or an ABI-compatible x86-64 distribution.','- Linux/WSL: Ubuntu 24.04 or an ABI-compatible x86-64 distribution. The fast launcher requires Python3.12+ and `numactl`, with at least8 available CPU workers.')
(R/'README.md').write_text(s)
ch=(R/'CHANGELOG-UNIVERSAL.md').read_text()
(R/'CHANGELOG-UNIVERSAL.md').write_text('''# Universal r5-fast — October7,2026

- Package exact certified SM86 S71 GDN512/Q2_K/N7 bytes: Bench.sh code53.88/narrative29.13t/s; prefill10K567.58/77K469.42t/s; full150127pass@1/134pass@3 against baseline131/134.
- Add an explicit portable `--profile fast`, runtime hash checks, optional exact model-hash verification, and architecture qualification gates. Retain original guard512 and production sampler; exclude all guard-tuning experiments.
- Prepare one source/profile for SM86/89/120 and compile the transformed bridge, GDN512 bridge, fused-head templates and retargeted PTX kernels on all three. SM89/SM120 full-runtime integration and hardware qualification remain pending, with inherited payloads retained for compatibility profiles.
- Publish source snapshot identity, all quality losses/guard observations, compilation receipts and clean-archive validation. Windows stays at its immutable r4 release.

'''+ch)
m=json.loads((R/'MANIFEST-LINUX-UNIVERSAL.json').read_text());m['version']='0.4.7-universal-r5-fast';m['configuration_release']=tag
m['fast_profile']=json.loads((R/'tools/universal/fast-profile.json').read_text())
m['fast_qualification']={'gpu':'RTX3060 sm86','code_tps':53.88327040707917,'narrative_tps':29.128009464657744,'prefill_10240_tps':567.5764001405505,'prefill_77824_tps':469.419249732974,'pass_at_1':127,'pass_at_3':134,'total':150,'limitations':'evidence/fast/quality-review.json','record':'evidence/fast/certification-final.json'}
m['payloads']['sm86']['fast_profile']='Certified S71 GDN512,Q2_K/N7,8workers,81920KVarN4/4,UB512; explicit fast profile'
for a in ['sm89','sm120']:
 m['payloads'][a]['fast_profile']='Shared source/profile and kernel compilation prepared; complete runtime/hardware qualification pending'
m['qualification_limits']=['Only SM86 S71 fast payload is hardware qualified for the stated Linux/WSL single-slot configuration.','SM89/SM120 inherited runtimes are not relabeled fast; complete fast runtime integration and hardware tests remain pending.','Windows is unchanged at r4.','Known quality failures and guard events are retained; no loop-free-output or global bitwise-logit-equivalence claim.','Original source snapshot plus fast overlay is required; the old pinned source commit alone is insufficient.']
(R/'MANIFEST-LINUX-UNIVERSAL.json').write_text(json.dumps(m,indent=2)+'\n')
g=json.loads((R/'MANIFEST.json').read_text());g.update(status='SM86-fast-qualified; cross-architecture-parity-prepared',qualification='evidence/fast/certification-final.json',fast_release=tag,source_snapshot_sha256='a9dd23bfa3ed38c040ffed6447e7085e083cb6abbb640a5b0cb3cf48f7de7263',source_overlay='source/fast')
(R/'MANIFEST.json').write_text(json.dumps(g,indent=2)+'\n')
p={'model_repo':'YourHighnessLA/L0xRE-27b-Low','revision':'a18987908d220da9c40cf2887913e727cdbf65a4','files':{'L0xRE-27b-Low.gguf':{'bytes':8619127680,'sha256':'b0849250c633aa93853bf119a877dbdafdd7b1a4ebb672bd6b6a1439906f3543'},'Qwen3.8-27B-DFlash2-Q2_K.gguf':{'bytes':705430880,'sha256':'e3eb7705404817cdbcdabe56049a1952b3b37bcc8df6e4d4efaec5d41563fb7e'}}}
(R/'source/fast/MODEL-IDENTITY.json').write_text(json.dumps(p,indent=2)+'\n')
fastsha='e3eb7705404817cdbcdabe56049a1952b3b37bcc8df6e4d4efaec5d41563fb7e'
ms=(R/'MODEL-SHA256SUMS').read_text()
if fastsha not in ms:(R/'MODEL-SHA256SUMS').write_text(ms.rstrip()+'\n'+fastsha+'  models/Qwen3.8-27B-DFlash2-Q2_K.gguf\n')
print('Fast release documentation and manifests updated; inherited architecture/Windows status retained.')
