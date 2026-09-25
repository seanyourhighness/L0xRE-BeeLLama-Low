#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert an Escha 2/3-bit safetensors checkpoint (dense, Qwen3.5 architecture) to
Beellama GGUF with the projections kept in their native escha trellis code.

What lands in the GGUF
----------------------
* every linear projection stays in the packed 2/3-bit code (same bytes as the
  safetensors), plus the per-projection rin/rout rotation vectors and the
  bias-correction vector (loaded, not applied -- matches the reference runtime);
* the shared codec tables: `escha_lut` (codebook A, computed from the closed
  form) and `escha_dep_k2` / `escha_dep_k3` (bit-dependency tables, extracted
  from the escha kernel and checked bit-for-bit against `escham_reconstruct`);
* the token embedding and LM head are dequantized from the LowGPU 3-bit vocab
  to fp16 -- bit-exact to the reference `dequant_full` -- and stored as
  `token_embd.weight` / `output.weight`;
* the remaining tensors (norms, ssm params) are written fp16/fp32 like the
  reference GGUF for this model.

Run:
    python3 convert_escha_to_gguf.py \
        --input-dir weights/escha-w2-lowgpu-mono \
        --outfile escha-w2-lowgpu-mono.gguf

Requires the escha venv (safetensors) and this repo's gguf-py.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

if "NO_LOCAL_GGUF" not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / "gguf-py"))

import gguf
from gguf import GGUFValueType, GGMLQuantizationType


# ---------------------------------------------------------------------------
# escha codec
# ---------------------------------------------------------------------------

def escha_codebook_lut() -> np.ndarray:
    """Codebook A, closed form (recovered from escham_reconstruct_kernel<1, K>)."""
    idx = np.arange(65536, dtype=np.uint32)
    x = ((idx * np.uint32(0xCBAC1FED)) & np.uint32(0x8FFF8FFF)) ^ np.uint32(0x3B603B60)
    lo = (x & np.uint32(0xFFFF)).astype("<u2").view(np.float16)
    hi = (x >> np.uint32(16)).astype("<u2").view(np.float16)
    v = (lo.astype(np.float32) + hi.astype(np.float32)).astype(np.float16)
    return v


def load_dep_table(path: Path) -> np.ndarray:
    dep = np.load(path)
    assert dep.shape == (16, 256), dep.shape
    # writer reverses dims into the file, so store as [weight][bit] -> file [bit][weight]
    return np.ascontiguousarray(dep.T).astype(np.int16)


# ---------------------------------------------------------------------------
# LowGPU vocab dequant (bit-exact to source/lowgpu/format.py::dequant_full)
# ---------------------------------------------------------------------------

def dequant_lowgpu_rows(codes: np.ndarray, scales: np.ndarray, zps: np.ndarray,
                        row0: int, row1: int, group: int = 128) -> np.ndarray:
    """Dequant rows [row0, row1) of the packed 3-bit vocab to fp16."""
    v, kb = codes.shape
    k = kb * 8 // 3
    g = (k + group - 1) // group
    rows = np.arange(row0, row1)
    p = codes[rows].reshape(row1 - row0, kb // 3, 3).astype(np.int32)
    b0, b1, b2 = p[..., 0], p[..., 1], p[..., 2]
    c0 = b0 & 7
    c1 = (b0 >> 3) & 7
    c2 = ((b0 >> 6) | (b1 << 2)) & 7
    c3 = (b1 >> 1) & 7
    c4 = (b1 >> 4) & 7
    c5 = ((b1 >> 7) | (b2 << 1)) & 7
    c6 = (b2 >> 2) & 7
    c7 = (b2 >> 5) & 7
    c = np.stack([c0, c1, c2, c3, c4, c5, c6, c7], axis=-1).reshape(row1 - row0, k)
    sc = scales[rows].reshape(row1 - row0, g, 1).astype(np.float32)
    zp = zps[rows].reshape(row1 - row0, g, 1).astype(np.float32)
    out = ((c.reshape(row1 - row0, g, group).astype(np.float32) - zp) * sc)
    return out.reshape(row1 - row0, k).astype(np.float16)


# ---------------------------------------------------------------------------
# checkpoint -> GGUF mapping
# ---------------------------------------------------------------------------

def escha_gguf_name(blk: str, suffix: str, il: int) -> str:
    return f"blk.{il}.{blk}.{suffix}"


def write_escha_proj(writer: gguf.GGUFWriter, find, ckpt_prefix: str, gguf_blk: str, il: int,
                     expect_oc: int | None = None) -> None:
    """Write one escha projection (code/rin/rout/bias) from the checkpoint."""
    code = find(f"{ckpt_prefix}.escha_code")
    rin  = find(f"{ckpt_prefix}.escha_rin")
    rout = find(f"{ckpt_prefix}.escha_rout")
    s_in = find(f"{ckpt_prefix}.escha_s_in")
    s_out = find(f"{ckpt_prefix}.escha_s_out")
    bias = find(f"{ckpt_prefix}.bias")

    # checkpoint code is (IC/16, OC/16, 16*K); GGUF wants (16*K, OC/16, IC/16)
    assert code.ndim == 3
    ic16, oc16, n_code = code.shape
    ic = ic16 * 16
    oc = oc16 * 16
    assert n_code in (32, 48), n_code
    if expect_oc is not None:
        assert oc == expect_oc, (ckpt_prefix, oc, expect_oc)

    assert rin.shape == (ic,), rin.shape
    assert rout.shape == (oc,), rout.shape
    assert s_in.shape == (ic,), s_in.shape
    assert s_out.shape == (oc,), s_out.shape
    assert bias.shape == (oc,), bias.shape

    # Wscale is already folded into rin; fold the per-channel s_in/s_out on top
    rin_g = (rin.astype(np.float32) * s_in.astype(np.float32)).astype(np.float16)
    rout_g = (rout.astype(np.float32) * s_out.astype(np.float32)).astype(np.float16)
    bias_g = bias.astype(np.float32)

    writer.add_tensor(escha_gguf_name(gguf_blk, "escha_code", il),
                      code, raw_dtype=GGMLQuantizationType.I16)
    writer.add_tensor(escha_gguf_name(gguf_blk, "escha_rin", il),
                      rin_g)
    writer.add_tensor(escha_gguf_name(gguf_blk, "escha_rout", il),
                      rout_g)
    writer.add_tensor(escha_gguf_name(gguf_blk, "bias", il),
                      bias_g)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--outfile", required=True, type=Path)
    parser.add_argument("--metadata-template", type=Path,
                        default=Path(__file__).parent / "conversion" / "escha" / "metadata-template.json")
    parser.add_argument("--dep-k2", type=Path,
                        default=Path(__file__).parent / "conversion" / "escha" / "dep_k2.npy")
    parser.add_argument("--dep-k3", type=Path,
                        default=Path(__file__).parent / "conversion" / "escha" / "dep_k3.npy")
    args = parser.parse_args()

    in_dir = args.input_dir
    files = sorted(p for p in in_dir.iterdir() if p.name.endswith(".safetensors"))
    assert files, f"no safetensors files in {in_dir}"

    # metadata: reuse the reference GGUF's header (tokenizer + hparams), with
    # identity fields pointed at this build
    with open(args.metadata_template, encoding="utf-8") as f:
        meta = json.load(f)

    outfile = args.outfile
    writer = gguf.GGUFWriter(outfile, "qwen35", use_temp_file=True)

    for key, typ, elem, val in meta:
        if key == "general.architecture":
            continue
        if key in ("general.name", "general.basename", "general.finetune"):
            continue
        if typ == 9:
            writer.add_key_value(key, val, GGUFValueType.ARRAY, GGUFValueType(elem))
        else:
            writer.add_key_value(key, val, GGUFValueType(typ))

    writer.add_string("general.name", "Escha W2 x LowGPU hybrid (Beellama port)")
    writer.add_string("general.basename", "escha-w2-lowgpu")
    writer.add_string("general.finetune", "escha-w2-lowgpu")

    dep_k2 = load_dep_table(args.dep_k2)
    dep_k3 = load_dep_table(args.dep_k3)
    lut = escha_codebook_lut()

    # shared codec tables
    writer.add_tensor("escha_lut", lut)
    writer.add_tensor("escha_dep_k2", dep_k2, raw_dtype=GGMLQuantizationType.I16)
    writer.add_tensor("escha_dep_k3", dep_k3, raw_dtype=GGMLQuantizationType.I16)

    layer_prefix = "model.language_model.layers."
    full_attn = {il for il in range(64) if il % 4 == 3}

    # collect per-file tensor name lists
    handles = []
    for p in files:
        sf = safe_open(str(p), framework="np")
        handles.append((p, sf, set(sf.keys())))

    def find(prefix: str):
        for _, sf, keys in handles:
            if prefix in keys:
                return sf.get_tensor(prefix)
        raise KeyError(prefix)

    def in_checkpoint(prefix: str) -> bool:
        return any(prefix in keys for _, _, keys in handles)

    # layer tensors, in load order
    for il in range(64):
        p = f"{layer_prefix}{il}."
        if il in full_attn:
            qkv = "self_attn"
        else:
            qkv = "linear_attn"

        # norms
        attn_norm = find(p + "input_layernorm.weight").astype(np.float32)
        writer.add_tensor(f"blk.{il}.attn_norm.weight", attn_norm)
        post_norm = find(p + "post_attention_layernorm.weight").astype(np.float32)
        writer.add_tensor(f"blk.{il}.post_attention_norm.weight", post_norm)

        if il in full_attn:
            write_escha_proj(writer, find, p + qkv + ".q_proj", "attn_q", il, 12288)
            write_escha_proj(writer, find, p + qkv + ".k_proj", "attn_k", il, 1024)
            write_escha_proj(writer, find, p + qkv + ".v_proj", "attn_v", il, 1024)
            write_escha_proj(writer, find, p + qkv + ".o_proj", "attn_output", il, 5120)
            for name, gguf_blk in (("q_norm", "attn_q_norm"), ("k_norm", "attn_k_norm")):
                n = find(p + qkv + f".{name}.weight").astype(np.float32)
                writer.add_tensor(f"blk.{il}.{gguf_blk}.weight", n)
        else:
            a_log = find(p + qkv + ".A_log").astype(np.float32)
            writer.add_tensor(f"blk.{il}.ssm_a", a_log)
            conv = find(p + qkv + ".conv1d.weight").astype(np.float32)
            conv_g = np.ascontiguousarray(conv.reshape(10240, 4))
            writer.add_tensor(f"blk.{il}.ssm_conv1d.weight", conv_g)
            dt = find(p + qkv + ".dt_bias").astype(np.float32)
            writer.add_tensor(f"blk.{il}.ssm_dt.bias", dt)
            for ckpt, gguf_blk, in_dim, out_dim in (
                    ("in_proj_a", "ssm_beta", 5120, 48),
                    ("in_proj_b", "ssm_alpha", 5120, 48)):
                w = find(p + qkv + f".{ckpt}.weight").astype(np.float16)
                writer.add_tensor(f"blk.{il}.{gguf_blk}.weight", w)
            ssm_norm = find(p + qkv + ".norm.weight").astype(np.float32)
            writer.add_tensor(f"blk.{il}.ssm_norm.weight", ssm_norm)
            write_escha_proj(writer, find, p + qkv + ".in_proj_qkv", "attn_qkv", il, 10240)
            write_escha_proj(writer, find, p + qkv + ".in_proj_z", "attn_gate", il, 6144)
            write_escha_proj(writer, find, p + qkv + ".out_proj", "ssm_out", il, 5120)

        write_escha_proj(writer, find, p + "mlp.gate_proj", "ffn_gate", il, 17408)
        write_escha_proj(writer, find, p + "mlp.up_proj", "ffn_up", il, 17408)
        write_escha_proj(writer, find, p + "mlp.down_proj", "ffn_down", il, 5120)

    output_norm = find("model.language_model.norm.weight").astype(np.float32)
    writer.add_tensor("output_norm.weight", output_norm)

    # vocab: LowGPU 3-bit -> fp16
    embed_codes = find("model.language_model.embed_tokens.weight_lowgpu_codes")
    embed_scales = find("model.language_model.embed_tokens.weight_lowgpu_scales")
    embed_zps = find("model.language_model.embed_tokens.weight_lowgpu_zps")
    head_codes = find("lm_head.weight_lowgpu_codes")
    head_scales = find("lm_head.weight_lowgpu_scales")
    head_zps = find("lm_head.weight_lowgpu_zps")

    v, kb = embed_codes.shape
    k = kb * 8 // 3
    n_chunks = 8
    chunk = (v + n_chunks - 1) // n_chunks

    embed_full = np.empty((v, k), dtype=np.float16)
    for i in range(n_chunks):
        r0, r1 = i * chunk, min((i + 1) * chunk, v)
        embed_full[r0:r1] = dequant_lowgpu_rows(embed_codes, embed_scales, embed_zps, r0, r1)
    writer.add_tensor("token_embd.weight", embed_full)

    head_full = np.empty((v, k), dtype=np.float16)
    for i in range(n_chunks):
        r0, r1 = i * chunk, min((i + 1) * chunk, v)
        head_full[r0:r1] = dequant_lowgpu_rows(head_codes, head_scales, head_zps, r0, r1)
    writer.add_tensor("output.weight", head_full)

    writer.write_header_to_file(path=outfile)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    # verify header round-trips
    r = gguf.GGUFReader(str(outfile))
    print(f"wrote {len(r.tensors)} tensors to {outfile}")


if __name__ == "__main__":
    main()
