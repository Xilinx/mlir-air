# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Seq-first temporal-causal FlashAttention wrapper (head_dim=128).

The head-first path (``fa_headfirst``) has to transpose Q/K/V into
[head, seq, d] on the host and transpose the result back, because the
head_dim=128 kernel it drives is head-first. ``attn_npu2_temporal_causal`` is
SEQ-FIRST: its L3 operands are exactly the (seq, n_heads*head_dim) /
(seq, n_kv_heads*head_dim) buffers the prefill block already holds, so this
wrapper is a straight pass-through with no host transposes.

Mapping: NB = n_heads/n_kv_heads column-blocks of 2 physical columns, NR = 4
rows, one q-seq-tile per core, so lqp = num_q_tiles * tile_size_q =
8 * 32 = 256 and the launch iterates kv-head groups only.
"""

import numpy as np
from ml_dtypes import bfloat16

# tile_size_q is pinned to 32 at head_dim=128: the kernel's mandatory Q-pair
# staging (2 * tile_size_q * dk * 2 B) cannot alias the K/V slab, so tsq=64
# needs 107 KB of the 64 KB L1.
LKP = 32
NUM_Q_TILES = 8
LQP = NUM_Q_TILES * LKP  # 256


def _fa_backend_kwargs(verbose=False):
    return {
        "verbose": verbose,
        "omit_while_true_loop": False,
        "omit_pingpong": "all",
        # Seq-first output is 2D [seq, n_heads*dv], so the launch is rank-2.
        "runtime_loop_tiling_sizes": [1, 1],
        "output_format": "elf",
        "instance_name": "attention_bf16",
    }


def supports(seq_len, n_heads, n_kv_heads, head_dim):
    """Can this shape run on the temporal-causal kernel?"""
    if head_dim != 128 or seq_len % LQP:
        return False
    # The herd is NB column-blocks x 4 rows over 8 physical columns.
    return n_heads % n_kv_heads == 0 and 2 * (n_heads // n_kv_heads) <= 8


def compile_temporal_fa(cache, seq_len, n_heads, n_kv_heads, head_dim, verbose=False):
    """Compile the seq-first temporal-causal FA ELF into `cache` as "flash_attn"."""
    assert supports(seq_len, n_heads, n_kv_heads, head_dim), (
        f"temporal-causal FA does not support seq_len={seq_len} "
        f"n_heads={n_heads} n_kv_heads={n_kv_heads} head_dim={head_dim}"
    )
    from shared.infra.external_kernels import compile_attn_npu2

    # This kernel keeps d WHOLE (one transfer per head), so the d tile is the
    # full head_dim, not lkp. force=True because the same CWD may hold a
    # differently-shaped attn_npu2.o from another model's compile.
    compile_attn_npu2(
        head_dim=head_dim,
        lkp=LKP,
        lqp_tile=LQP // NUM_Q_TILES,
        dk_tile=head_dim,
        dv_tile=head_dim,
        force=True,
    )

    from flash_attention.kernel_fusion_based.attn_npu2_temporal_causal import (
        build_module,
    )

    mod = build_module(
        lk=seq_len,
        lkp=LKP,
        lq=seq_len,
        lqp=LQP,
        dk=head_dim,
        dv=head_dim,
        num_q_tiles=NUM_Q_TILES,
        num_heads=n_heads,
        num_kv_heads=n_kv_heads,
        causal=True,
        num_heads_per_unroll=1,
    )
    cache.compile_and_cache("flash_attn", mod, _fa_backend_kwargs(verbose))


def npu_fa_temporal(
    cache, q_roped, k_roped, v, n_heads, n_kv_heads, head_dim, seq_len, verbose=False
):
    """Run seq-first temporal-causal FlashAttention; returns seq-first bf16.

    Args:
        q_roped: (seq, n_heads*head_dim)    seq-first, post-RoPE.
        k_roped: (seq, n_kv_heads*head_dim) seq-first, post-RoPE.
        v:       (seq, n_kv_heads*head_dim) seq-first, raw V projection.
    Returns:
        (seq, n_heads*head_dim) seq-first bf16 attention output.
    """
    q = np.ascontiguousarray(q_roped, dtype=bfloat16).reshape(
        seq_len, n_heads * head_dim
    )
    k = np.ascontiguousarray(k_roped, dtype=bfloat16).reshape(
        seq_len, n_kv_heads * head_dim
    )
    v = np.ascontiguousarray(v, dtype=bfloat16).reshape(seq_len, n_kv_heads * head_dim)
    out = np.zeros((seq_len, n_heads * head_dim), dtype=bfloat16)

    results = cache.load_and_run(
        "flash_attn", _fa_backend_kwargs(verbose), q, k, v, out
    )
    return np.ascontiguousarray(
        results[-1].reshape(seq_len, n_heads * head_dim)
    ).astype(bfloat16)
