# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared head-first FlashAttention wrapper for head_dim>=128 LLM prefill.

Why head-first (not seq-first)?
  The seq-first FA kernel (`attn_npu2_seqfirst.py`, used by llama32_1b at
  head_dim=64) enforces dv == lkp. At head_dim>=128 the kernel needs
  dv_chunks>1, which seq-first cannot express, and its dk_chunks>1 path hangs.
  So head_dim>=128 MUST use the HEAD-FIRST kernel
  `flash_attention/kernel_fusion_based/attn_npu2.py` with host-side transposes
  around it. This module is that host-side plumbing, shared by every
  head_dim=128 GQA model (qwen3_0_6b and 5 siblings) and by Gemma3 at
  head_dim=256.

L3 layouts the head-first kernel expects (dv_chunks = head_dim // dv_tile):
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

# head_dim -> (lkp, lqp, num_q_tiles, num_heads_per_unroll, dv_tile).
#
# Causal masking pins tile_size_q (= lqp // num_q_tiles) == lkp, and the per-core
# L1 working set is dominated by the resident Q tile, dk_chunks * tile_size_q *
# lkp * 2B = head_dim * lkp * 2B, plus the lkp x lkp / lkp x dv_tile companions.
# At head_dim=256 the head_dim=128 tiling (lkp=64) needs 9 x 8 KB = 72 KB and
# aiecc rejects it ("allocated buffers exceeded available memory").
#
# What dominates runtime is L3 traffic, not L1 or FLOPs: the kernel re-streams
# the whole of K per launch iteration, so
#     K bytes = (seq / lqp) * n_heads * (head_dim / dv_tile) * seq * head_dim * 2
# Both remaining knobs therefore matter a lot -- lqp wants to be as large as the
# core budget allows (cores = num_heads_per_unroll * num_q_tiles * 4 cascade
# stages = 32, and lqp = num_q_tiles * lkp, so lqp is maximised at
# num_heads_per_unroll=1), and dv_tile wants to be the full head_dim so the
# dv_chunks launch axis disappears. Physical columns = num_heads_per_unroll *
# num_q_tiles must be <= 8 on NPU2.
_FA_TILING = {
    128: (64, 256, 4, 2, 64),
    # dv_tile=256 (no dv_chunks axis at all) overflows L1 by ~5 KB: v, gp and
    # the V get's second buffer are each dv_tile*lkp*2B, so three of them plus
    # the 16 KB of resident Q tiles does not fit. 128 is the largest that does.
    256: (32, 256, 8, 1, 128),
}


# Backend kwargs MUST be identical between compile and run (the cache keys the
# XRT context on the kernel name, but the BO layout / ELF the load picks up is
# the one compiled under these flags). These mirror the proven-good standalone
# (attn_npu2.py __main__) which PASSES at seq=2048 hd=128 GQA.
def _fa_backend_kwargs(verbose=False, dv_chunks=2):
    return {
        "verbose": verbose,
        # The proven-good standalone (attn_npu2.py __main__, which PASSES at
        # seq=2048 nh=16 nkv=8 hd=128 causal) uses omit_while_true_loop=False.
        # Setting it True at this shape HANGS (ERT_CMD_STATE_TIMEOUT) — keep it
        # aligned with the standalone that passes.
        "omit_while_true_loop": False,
        "omit_pingpong": "all",
        # dv_chunks>1 makes the launch 3D -> tiling must match the launch rank.
        "runtime_loop_tiling_sizes": [1, 1, 1] if dv_chunks > 1 else [1, 1],
        "output_format": "elf",
        "instance_name": "attention_bf16",
    }


def fa_tiling(head_dim):
    """(lkp, lqp, num_q_tiles, num_heads_per_unroll, dv_tile) for a head_dim."""
    if head_dim not in _FA_TILING:
        raise ValueError(
            f"no head-first FA tiling for head_dim={head_dim} "
            f"(have {sorted(_FA_TILING)}); head_dim=64 uses the seq-first kernel"
        )
    return _FA_TILING[head_dim]


def compile_headfirst_fa(
    cache,
    seq_len,
    n_heads,
    n_kv_heads,
    head_dim,
    verbose=False,
    window=None,
    name="flash_attn",
):
    """Compile the head-first FlashAttention ELF into `cache` under `name`.

    Covers the head_dim values the seq-first kernel can't handle (see
    _FA_TILING). Compiles attn_npu2.o first (so prepare_air_project copies it
    into air_project/ for the ELF link), then the ELF.

    `window` (default None = plain causal) enables sliding-window masking: a
    query at position p attends only to keys in (p - window, p]. Callers that
    need both variants must pass distinct `name`s so the ELFs don't collide in
    the cache.
    """
    from shared.infra.external_kernels import compile_attn_npu2

    lkp, lqp, num_q_tiles, num_heads_per_unroll, dv_tile = fa_tiling(head_dim)
    lqp_tile = lqp // num_q_tiles  # tile_size_q == lkp under causal masking

    # Compile the C++ microkernel with the PER-TILE shapes the head-first
    # kernel actually uses (lkp, tile_size_q, dk_full=dv_full=head_dim). The
    # legacy compile_attn_npu2(head_dim=128) baked lqp=lkp=dk=128 — wrong tile
    # shapes for this config and the kernel hangs. force=True because the same
    # CWD may already hold a differently-shaped attn_npu2.o from another
    # model's compile.
    compile_attn_npu2(
        head_dim=head_dim, lkp=lkp, lqp_tile=lqp_tile, dv_tile=dv_tile, force=True
    )

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
        num_heads_per_unroll=num_heads_per_unroll,
        window=window,
        dv_tile=dv_tile,
    )
    cache.compile_and_cache(name, mod, _fa_backend_kwargs(verbose, head_dim // dv_tile))


def npu_fa_headfirst(
    cache,
    q_roped,
    k_roped,
    v,
    n_heads,
    n_kv_heads,
    head_dim,
    seq_len,
    verbose=False,
    name="flash_attn",
):
    """Run head-first FlashAttention on NPU and return seq-first bf16 output.

    Args:
        cache: KernelCache with `name` already compiled (compile_headfirst_fa).
        q_roped: (seq, n_heads*head_dim)   seq-first, post-QK-norm post-RoPE.
        k_roped: (seq, n_kv_heads*head_dim) seq-first, post-QK-norm post-RoPE.
        v:       (seq, n_kv_heads*head_dim) seq-first, raw V projection.
        name: cache key of the ELF to run. Models with per-layer attention
            variants (e.g. Gemma3's alternating sliding-window / global layers)
            compile one ELF per variant and select here.
    Returns:
        (seq, n_heads*head_dim) seq-first bf16 attention output.
    """
    dv_tile = fa_tiling(head_dim)[4]
    dv_chunks = head_dim // dv_tile
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
        .reshape(n_kv_heads, seq_len, dv_chunks, dv_tile)
        .transpose(0, 2, 1, 3)  # [n_kv, dv_chunks, seq, dv_tile]
        .reshape(n_kv_heads * dv_chunks, seq_len, dv_tile)
    )

    # Output BO: [n_heads*dv_chunks, seq, dv_tile]
    out_hf = np.zeros((n_heads * dv_chunks, seq_len, dv_tile), dtype=bfloat16)

    results = cache.load_and_run(
        name,
        _fa_backend_kwargs(verbose, dv_chunks),
        q_hf,
        k_hf,
        v_hf,
        out_hf,
    )
    gp = results[-1].reshape(n_heads * dv_chunks, seq_len, dv_tile)

    # ---- Host transpose head-first -> seq-first ----
    # gp [n_heads*dv_chunks, seq, dv_tile] : axes nest as (head, dv_chunk).
    # Concat the dv_chunks back to head_dim, then move seq to the front.
    attn_out = (
        gp.reshape(n_heads, dv_chunks, seq_len, dv_tile)
        .transpose(2, 0, 1, 3)  # [seq, n_heads, dv_chunks, dv_tile]
        .reshape(seq_len, q_dim)
    )
    return np.ascontiguousarray(attn_out).astype(bfloat16)
