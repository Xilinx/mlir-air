# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FlashAttention invariant: same standalone ELF + same invocation in every cell.

FA's MLIR builder is at programming_examples/flash_attention/kernel_fusion_based/attn_npu2_seqfirst.py
with kwargs matching Plan 1's compile_all_kernels() in llama32_1b_prefill.py.
"""

import time

import numpy as np
from ml_dtypes import bfloat16

from kernel_builder.cache import KernelCache


def _attn_backend_kwargs():
    return {
        "verbose": False,
        "omit_while_true_loop": False,  # head_dim=64, lkp=64 enables shared buffers
        "omit_pingpong": "all",
        "runtime_loop_tiling_sizes": [1, 1],
        "output_format": "elf",
        "instance_name": "attention_bf16",
    }


def compile_flash_attn(cache: KernelCache, config):
    """Compile FA ELF if not already cached. ~46s first time per profile.md."""
    if "flash_attn" in cache.artifacts:
        return
    from flash_attention.kernel_fusion_based.attn_npu2_seqfirst import (
        build_module as build_attn,
    )

    seq = config["seq_len"]
    head_dim = config["head_dim"]
    n_heads = config["n_heads"]
    n_kv_heads = config["n_kv_heads"]
    mod = build_attn(
        lk=seq,
        lkp=head_dim,
        lq=seq,
        lqp=256,
        dk=head_dim,
        dv=head_dim,
        num_q_tiles=4,
        num_cascade_stages=4,
        num_heads=n_heads,
        num_kv_heads=n_kv_heads,
        causal=True,
    )
    cache.compile_and_cache("flash_attn", mod, _attn_backend_kwargs())
    cache._save_manifest()


def run_flash_attn(cache, q_roped, k_roped, v, layer_idx=0):
    """Run FA on extracted q_roped/k_roped/v from rms_gemms_rope.
    Returns attn_out (extracted to host) ready to feed o_ffn.
    """
    seq = q_roped.shape[0]
    emb = q_roped.shape[1]
    args = [q_roped, k_roped, v, np.zeros((seq, emb), dtype=bfloat16)]
    t0 = time.perf_counter()
    out = cache.load_and_run(
        "flash_attn",
        _attn_backend_kwargs(),
        *args,
        output_indices=[3],
        intermediate_indices={3},
        bo_key=f"FA_L{layer_idx}",
    )
    return {"attn_out": out[3], "_wall_s": time.perf_counter() - t0}
