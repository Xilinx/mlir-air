# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared head-first FlashAttention wrapper for head_dim=128 LLM prefill.

Why head-first (not seq-first)?
  The seq-first FA kernel (`attn_npu2_seqfirst.py`, used by llama32_1b at
  head_dim=64) enforces dv == lkp. At head_dim=128 the kernel needs
  dv_chunks=2 (lkp=64, dv_tile=64), which seq-first cannot express, and its
  dk_chunks>1 path hangs. So head_dim=128 MUST use the HEAD-FIRST kernel
  `flash_attention/kernel_fusion_based/attn_npu2.py` with host-side transposes
  around it. This module is that host-side plumbing, shared by every
  head_dim=128 GQA model (qwen3_0_6b and 5 siblings).

L3 layouts the head-first kernel expects (dv_chunks = head_dim // lkp = 2):
  Q   L3: [num_heads,             seq, head_dim]              (head-first)
  K   L3: [num_kv_heads,          seq, head_dim]              (head-first)
  V   L3: [num_kv_heads * dv_chunks, seq, dv_tile]            (dv-chunked)
  out L3: [num_heads     * dv_chunks, seq, dv_tile]           (dv-chunked)

The V-pack and output-unpack numpy ops below mirror attn_npu2.py's __main__
exactly (input_v reshape/transpose at ~L1326-1331, output un-transpose at
~L1350-1355) and were verified against a numpy SDPA reference (cos == 1.0).
"""

from __future__ import annotations

import numpy as np
from ml_dtypes import bfloat16


# Backend kwargs MUST be identical between compile and run (the cache keys the
# XRT context on the kernel name, but the BO layout / ELF the load picks up is
# the one compiled under these flags). These mirror the proven-good standalone
# (attn_npu2.py __main__) which PASSES at seq=2048 hd=128 GQA.
def _fa_backend_kwargs(verbose=False):
    return {
        "verbose": verbose,
        # The proven-good standalone (attn_npu2.py __main__, which PASSES at
        # seq=2048 nh=16 nkv=8 hd=128 causal) uses omit_while_true_loop=False.
        # Setting it True at this shape HANGS (ERT_CMD_STATE_TIMEOUT) — keep it
        # aligned with the standalone that passes.
        "omit_while_true_loop": False,
        "omit_pingpong": "all",
        # head-first gp output is 3D [num_heads*dv_chunks, seq, dv_tile] since
        # dv_chunks>1 makes the launch 3D -> tiling must be rank-3.
        "runtime_loop_tiling_sizes": [1, 1, 1],
        "output_format": "elf",
        "instance_name": "attention_bf16",
    }


def compile_headfirst_fa(cache, seq_len, n_heads, n_kv_heads, head_dim, verbose=False):
    """Compile the head-first FlashAttention ELF into `cache` as "flash_attn".

    Only supports head_dim=128 (the case the seq-first kernel can't handle).
    Compiles attn_npu2.o first (so prepare_air_project copies it into
    air_project/ for the ELF link), then the "flash_attn" ELF.
    """
    assert head_dim == 128, (
        f"compile_headfirst_fa is the head_dim=128 path; got head_dim={head_dim}. "
        f"Use the seq-first kernel for head_dim=64."
    )

    from shared.infra.external_kernels import compile_attn_npu2

    lkp = 64
    lqp = 256
    num_q_tiles = 4
    lqp_tile = lqp // num_q_tiles  # tile_size_q = 64

    # Compile the C++ microkernel with the PER-TILE shapes the head-first
    # kernel actually uses (lkp=64, tile_size_q=64, dk_full=dv_full=128). The
    # legacy compile_attn_npu2(head_dim=128) baked lqp=lkp=dk=128 — wrong tile
    # shapes for this config and the kernel hangs. force=True because the same
    # CWD may already hold an hd=64 attn_npu2.o from another model's compile.
    compile_attn_npu2(head_dim=head_dim, lkp=lkp, lqp_tile=lqp_tile, force=True)

    from flash_attention.kernel_fusion_based.attn_npu2 import build_module
    mod = build_module(
        lk=seq_len,
        lkp=lkp,
        lq=seq_len,
        lqp=lqp,
        dk=head_dim,
        dv=head_dim,
        num_q_tiles=num_q_tiles,
        num_cascade_stages=4,
        num_heads=n_heads,
        num_kv_heads=n_kv_heads,
        causal=True,
        num_heads_per_unroll=2,
    )
    cache.compile_and_cache("flash_attn", mod, _fa_backend_kwargs(verbose))


def npu_fa_headfirst(
    cache, q_roped, k_roped, v, n_heads, n_kv_heads, head_dim, seq_len, verbose=False
):
    """Run head-first FlashAttention on NPU and return seq-first bf16 output.

    Args:
        cache: KernelCache with "flash_attn" already compiled.
        q_roped: (seq, n_heads*head_dim)   seq-first, post-QK-norm post-RoPE.
        k_roped: (seq, n_kv_heads*head_dim) seq-first, post-QK-norm post-RoPE.
        v:       (seq, n_kv_heads*head_dim) seq-first, raw V projection.
    Returns:
        (seq, n_heads*head_dim) seq-first bf16 attention output.
    """
    assert head_dim == 128, f"head_dim={head_dim} unsupported (head_dim=128 only)"
    lkp = 64
    dv_chunks = head_dim // lkp  # 2
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim

    q = np.asarray(q_roped, dtype=bfloat16).reshape(seq_len, n_heads, head_dim)
    k = np.asarray(k_roped, dtype=bfloat16).reshape(seq_len, n_kv_heads, head_dim)
    v = np.asarray(v, dtype=bfloat16).reshape(seq_len, n_kv_heads, head_dim)

    # ---- Host transpose seq-first -> head-first ----
    # Q L3: [n_heads, seq, head_dim]
    q_hf = np.ascontiguousarray(q.transpose(1, 0, 2))
    # K L3: [n_kv_heads, seq, head_dim]
    k_hf = np.ascontiguousarray(k.transpose(1, 0, 2))
    # V L3: [n_kv_heads*dv_chunks, seq, dv_tile] -- split head_dim into
    # dv_chunks slices of dv_tile=lkp, dv-chunk axis nests inside the kv-head
    # axis (head*dv_chunks + chunk), matching the kernel's head_v_off.
    v_hf = np.ascontiguousarray(
        v.transpose(1, 0, 2)  # [n_kv, seq, head_dim]
        .reshape(n_kv_heads, seq_len, dv_chunks, lkp)
        .transpose(0, 2, 1, 3)  # [n_kv, dv_chunks, seq, lkp]
        .reshape(n_kv_heads * dv_chunks, seq_len, lkp)
    )

    # Output BO: [n_heads*dv_chunks, seq, dv_tile]
    out_hf = np.zeros((n_heads * dv_chunks, seq_len, lkp), dtype=bfloat16)

    results = cache.load_and_run(
        "flash_attn",
        _fa_backend_kwargs(verbose),
        q_hf,
        k_hf,
        v_hf,
        out_hf,
    )
    gp = results[-1].reshape(n_heads * dv_chunks, seq_len, lkp)

    # ---- Host transpose head-first -> seq-first ----
    # gp [n_heads*dv_chunks, seq, lkp] : axes nest as (head, dv_chunk).
    # Concat the dv_chunks back to head_dim, then move seq to the front.
    attn_out = (
        gp.reshape(n_heads, dv_chunks, seq_len, lkp)
        .transpose(2, 0, 1, 3)  # [seq, n_heads, dv_chunks, lkp]
        .reshape(seq_len, q_dim)
    )
    return np.ascontiguousarray(attn_out).astype(bfloat16)
