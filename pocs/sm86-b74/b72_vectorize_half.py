#!/usr/bin/env python3
"""Add a four-half vector load to the bit-exact B71 K3 decode variant."""

from pathlib import Path


root = Path(__file__).resolve().parent
src = root / "k2k3-sm86-k3-f32-v4.ptx"
dst = root / "k2k3-sm86-k3-f32-v4-half-v4.ptx"
whole = src.read_text()
needle = ".visible .entry _ZN12escha_escham21escham_gemv_bw_kernelILi1ELi3"
assert whole.count(needle) == 1
k2, k3 = whole.split(needle, 1)
k3 = needle + k3
changes = {
    "ld.global.nc.u16 %rs66, [%rd71];":
        "ld.global.nc.v4.u16 {%rs66, %rs68, %rs70, %rs72}, [%rd71];",
    "ld.global.nc.u16 %rs68, [%rd71+2];": "",
    "ld.global.nc.u16 %rs70, [%rd71+4];": "",
    "ld.global.nc.u16 %rs72, [%rd71+6];": "",
}
for before, after in changes.items():
    assert k3.count(before) == 1, before
    k3 = k3.replace(before, after, 1)
assert "ld.global.nc.v4.u16" not in k2
dst.write_text(k2 + k3)
print(dst)
