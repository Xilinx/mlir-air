# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Head-spatial FlashAttention wrapper -- MQA prefill on FLM's CU topology.

Same contract as ``fa_headfirst`` (seq-first in, seq-first out); what differs is
the spatial mapping underneath. ``attn_npu2.py`` iterates heads in the launch
and re-reads all of K and V per head::

    K bytes = (seq/lqp) * n_heads * (head_dim/dv_tile) * seq * head_dim * 2

Under MQA every head wants the SAME K and V, so
``flash_attention/kernel_fusion_based/attn_npu2_headspatial.py`` makes the head
spatial -- one herd per head, all of them fed by ONE K/V broadcast -- and the
``n_heads`` factor leaves the formula. That is FLM's own mapping
(``FLM_Xclbin/Gemma4/attention_DH_256_prefill``): "each CU has 8 CTs and works
on 1 head", four CUs live at once, one K/V send shared by all four.

Four heads are resident per dispatch (four herds of 2x4 = the whole array), so
an 8-head model runs two dispatches per layer and K/V crosses L3 twice instead
of eight times.

L3 layouts this kernel expects (dv_chunks = head_dim // dv_tile):
  Q   L3: [seq, heads_spatial * head_dim]
  KV  L3: flat, seq/lkp records of [K tile | V tile per dv chunk]
  out L3: [seq, heads_spatial * head_dim]

Q and out are SEQ-FIRST, i.e. what the model already holds: the head and dv
chunk shuffles are BD strides, not host transposes.

MQA ONLY. With more than one kv head the single broadcast would feed every head
the wrong K, so `supports()` returns False and the caller keeps head-first.
"""

from __future__ import annotations

import numpy as np
from ml_dtypes import bfloat16

# head_dim -> (lkp, lqp, num_q_tiles, cu_cols, heads_spatial, dv_tile).
#
# One herd per head, cu_cols wide, so heads_spatial * cu_cols <= 8 physical
# columns. num_q_tiles = the herd's core count, and tile_size_q = lqp /
# num_q_tiles must equal lkp under causal masking, which pins lqp.
#
_HS_TILING = {
    256: (32, 256, 8, 2, 4, 128),
    # Gemma4's full-attention layers. lkp halves to 16 so the head_dim*lkp*2B
    # pair (q_saved, qk) stays at 16 KB, and dv_tile doubles to 256 so dv_chunks
    # stays 2 -- four chunks deadlocks.
    512: (16, 128, 8, 2, 4, 256),
}


def supports(head_dim, n_kv_heads):
    """True if the head-spatial kernel covers this shape."""
    return n_kv_heads == 1 and head_dim in _HS_TILING


def hs_tiling(head_dim):
    """(lkp, lqp, num_q_tiles, cu_cols, heads_spatial, dv_tile) for a head_dim."""
    if head_dim not in _HS_TILING:
        raise ValueError(
            f"no head-spatial FA tiling for head_dim={head_dim} "
            f"(have {sorted(_HS_TILING)})"
        )
    return _HS_TILING[head_dim]


def _fa_backend_kwargs(verbose=False):
    # Must be identical between compile and run. The launch is 1-D here (the dv
    # chunking lives inside the core), unlike head-first's 2-D/3-D launch.
    return {
        "verbose": verbose,
        "omit_while_true_loop": False,
        "omit_pingpong": "all",
        "runtime_loop_tiling_sizes": [1],
        "output_format": "elf",
        "instance_name": "attention_bf16",
    }


def compile_headspatial_fa(
    cache,
    seq_len,
    n_heads,
    n_kv_heads,
    head_dim,
    verbose=False,
    window=None,
    name="flash_attn_hs",
    causal_groups=1,
):
    """Compile the head-spatial FlashAttention ELF into `cache` under `name`.

    `causal_groups` cuts the round axis into G launches stitched into ONE ELF.
    Plain causal streams all of K and V on every round; round lx only needs the
    first num_q_tiles*(lx+1) blocks, but a growing extent cannot be one launch
    (air.api needs the size static). Each group gets a constant extent sized for
    its LAST round, so the traffic falls without the per-dispatch cost being
    paid G times. Ignored for the windowed (sliding) layers, which are already
    truncated.
    """
    from shared.infra.external_kernels import compile_attn_npu2

    lkp, lqp, num_q_tiles, cu_cols, heads_spatial, dv_tile = hs_tiling(head_dim)
    if not supports(head_dim, n_kv_heads):
        raise ValueError(f"head-spatial FA is MQA-only; got n_kv_heads={n_kv_heads}")
    if n_heads % heads_spatial:
        raise ValueError(
            f"n_heads ({n_heads}) must be a multiple of the {heads_spatial} "
            f"heads resident per dispatch"
        )
    if seq_len % lqp:
        raise ValueError(f"seq_len ({seq_len}) must be a multiple of lqp ({lqp})")

    # The microkernel is compiled with the PER-TILE shapes, as in head-first:
    # a mismatched object does not fail, it miscomputes.
    compile_attn_npu2(
        head_dim=head_dim,
        lkp=lkp,
        lqp_tile=lqp // num_q_tiles,
        dk_tile=head_dim,
        dv_tile=dv_tile,
        force=True,
    )

    if window is None and causal_groups > 1:
        cache.compile_and_cache(
            name,
            _build_staircase_module(
                seq_len,
                lkp,
                lqp,
                head_dim,
                num_q_tiles,
                heads_spatial,
                cu_cols,
                dv_tile,
                causal_groups,
            ),
            _fa_backend_kwargs(verbose),
        )
        return

    from flash_attention.kernel_fusion_based.attn_npu2_headspatial import build_module

    mod = build_module(
        lk=seq_len,
        lkp=lkp,
        lq=seq_len,
        lqp=lqp,
        dk=head_dim,
        dv=head_dim,
        num_q_tiles=num_q_tiles,
        num_heads=heads_spatial,
        cu_cols=cu_cols,
        num_kv_heads=1,
        causal=True,
        window=window,
        dv_tile=dv_tile,
    )
    cache.compile_and_cache(name, mod, _fa_backend_kwargs(verbose))


_FA_EXTERNS = {
    "@zero_fill_g_bf16",
    "@zero_fill_gp_bf16",
    "@zero_fill_sp_bf16",
    "@neg_inf_fill_up_bf16",
    "@matmul_a_b_bf16",
    "@matmul_g_b_bf16",
    "@fused_softmax",
    "@mul_r_gp",
    "@accum_sp_r_s",
    "@vector_copy_32elems",
    "@div_gp_sp",
    "@apply_causal_mask",
}


def _build_staircase_module(
    seq_len, lkp, lqp, dh, num_q_tiles, heads_spatial, cu_cols, dv_tile, groups
):
    """G causal round-groups, stitched into one ELF over shared Q/KV/out args."""
    from flash_attention.kernel_fusion_based.attn_npu2_headspatial import build_launch
    from shared.infra.stitching import stitch_elf, KernelSlice, FuncArg

    n_rounds = seq_len // lqp
    if n_rounds % groups:
        raise ValueError(f"{n_rounds} rounds not divisible by {groups} groups")
    per = n_rounds // groups
    dv_chunks = dh // dv_tile
    kv_rec = lkp * dh + dv_chunks * lkp * dv_tile

    slices = []
    for g in range(groups):
        ir = str(
            build_launch(
                lk=seq_len,
                lkp=lkp,
                lq=seq_len,
                lqp=lqp,
                dk=dh,
                dv=dh,
                num_q_tiles=num_q_tiles,
                num_heads=heads_spatial,
                cu_cols=cu_cols,
                num_kv_heads=1,
                causal=True,
                window=None,
                dv_tile=dv_tile,
                rounds=per,
                q_round_base=g * per,
                # sized for this group's LAST round
                kv_blocks=num_q_tiles * per * (g + 1),
            ).build(target="npu2")
        )
        slices.append(
            KernelSlice(ir, f"s{g}", {0: 0, 1: 1, 2: 2}, extern_syms=_FA_EXTERNS)
        )

    base_args = [
        FuncArg("%arg0", f"memref<{seq_len}x{heads_spatial * dh}xbf16>"),
        FuncArg("%arg1", f"memref<{(seq_len // lkp) * kv_rec}xbf16>"),
        FuncArg("%arg2", f"memref<{seq_len}x{heads_spatial * dh}xbf16>"),
    ]
    # The func symbol must be the instance name the backend loads, not the
    # cache key -- _fa_backend_kwargs pins "attention_bf16" for both paths.
    return stitch_elf("attention_bf16", base_args, slices)


def pack_kv(k, v, lkp, dv_tile):
    """Lay K and V out as the one interleaved stream the kernel reads.

    `k` and `v` are [seq, head_dim]. The result is flat: seq/lkp records, each a
    K tile followed by that chunk's V tiles. One shim link carries both, which
    is what lets a Q relay share the K/V memtile's column -- see the kernel's
    module docstring. FLM likewise keeps K and V in one BO.
    """
    seq, dk = k.shape
    dv_chunks = dk // dv_tile
    num_chunks = seq // lkp
    rec = lkp * dk + dv_chunks * lkp * dv_tile
    out = np.empty(num_chunks * rec, dtype=bfloat16)
    vc = v.reshape(seq, dv_chunks, dv_tile)
    for c in range(num_chunks):
        base = c * rec
        rows = slice(c * lkp, (c + 1) * lkp)
        out[base : base + lkp * dk] = k[rows, :].reshape(-1)
        base += lkp * dk
        for z in range(dv_chunks):
            n = lkp * dv_tile
            out[base : base + n] = vc[rows, z, :].reshape(-1)
            base += n
    return out


def npu_fa_headspatial(
    cache,
    q_roped,
    k_roped,
    v,
    n_heads,
    n_kv_heads,
    head_dim,
    seq_len,
    verbose=False,
    name="flash_attn_hs",
):
    """Run head-spatial FlashAttention on NPU and return seq-first bf16 output.

    Args mirror ``npu_fa_headfirst``: q/k/v are seq-first
    ([seq, n_heads*head_dim] and [seq, n_kv_heads*head_dim]), the result is
    [seq, n_heads*head_dim]. `n_kv_heads` must be 1.
    """
    lkp, _, _, _, heads_spatial, dv_tile = hs_tiling(head_dim)
    # Same contract compile_headspatial_fa enforces. Unchecked here, a head
    # count that is not a multiple would run the dispatch loop too few times
    # and return the tail head columns of an np.empty buffer.
    if not supports(head_dim, n_kv_heads):
        raise ValueError(f"head-spatial FA is MQA-only; got n_kv_heads={n_kv_heads}")
    if n_heads % heads_spatial:
        raise ValueError(
            f"n_heads ({n_heads}) must be a multiple of the {heads_spatial} "
            f"heads resident per dispatch"
        )
    q_dim = n_heads * head_dim
    hs_dim = heads_spatial * head_dim

    q = np.asarray(q_roped, dtype=bfloat16).reshape(seq_len, q_dim)
    k = np.asarray(k_roped, dtype=bfloat16).reshape(seq_len, head_dim)
    v = np.asarray(v, dtype=bfloat16).reshape(seq_len, head_dim)

    # K and V cross L3 once per dispatch, not once per head.
    kv = pack_kv(k, v, lkp, dv_tile)

    # Both sides are seq-first now, so a dispatch's head group is a contiguous
    # COLUMN BLOCK at either end -- one strided copy each way, where this used
    # to be a full [seq, heads, dh] permute in and a 4-D permute out.
    attn_out = np.empty((seq_len, q_dim), dtype=bfloat16)
    kw = _fa_backend_kwargs(verbose)
    for p in range(n_heads // heads_spatial):
        lo = p * hs_dim
        q_hs = np.ascontiguousarray(q[:, lo : lo + hs_dim])
        out_hs = np.zeros((seq_len, hs_dim), dtype=bfloat16)
        results = cache.load_and_run(name, kw, q_hs, kv, out_hs)
        attn_out[:, lo : lo + hs_dim] = results[-1].reshape(seq_len, hs_dim)

    return attn_out
