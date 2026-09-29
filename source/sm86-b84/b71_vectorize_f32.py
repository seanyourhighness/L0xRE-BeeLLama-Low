#!/usr/bin/env python3
"""Make an isolated K3 PTX variant with one vector load per four F32 operands."""

from pathlib import Path


root = Path(__file__).resolve().parent
src = root / "k2k3-sm86-baseline.ptx"
dst = root / "k2k3-sm86-k3-f32-v4.ptx"
whole = src.read_text()
needle = ".visible .entry _ZN12escha_escham21escham_gemv_bw_kernelILi1ELi3"
assert whole.count(needle) == 1
k2, k3 = whole.split(needle, 1)
k3 = needle + k3

changes = {
    "ld.global.nc.f32 %f66, [%rd66];":
        "ld.global.nc.v4.f32 {%f66, %f68, %f70, %f72}, [%rd66];",
    "ld.global.nc.f32 %f68, [%rd66+4];": "",
    "ld.global.nc.f32 %f70, [%rd66+8];": "",
    "ld.global.nc.f32 %f72, [%rd66+12];": "",
    "ld.global.nc.f32 %f78, [%rd68];":
        "ld.global.nc.v4.f32 {%f78, %f81, %f84, %f87}, [%rd68];",
    "ld.global.nc.f32 %f81, [%rd68+4];": "",
    "ld.global.nc.f32 %f84, [%rd68+8];": "",
    "ld.global.nc.f32 %f87, [%rd68+12];": "",
}
for before, after in changes.items():
    assert k3.count(before) == 1, before
    k3 = k3.replace(before, after, 1)
assert "ld.global.nc.v4.f32" not in k2
dst.write_text(k2 + k3)
print(dst)
