# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build-time choices for the SigLIP vision ELFs, read once and shared by the encoder
(what to compile and dispatch) and the runtime (which cache directory to use), so the
two cannot disagree. Every value can be overridden from the environment.

SMOLVLA_LN_EXT (default 1)
    The LayerNorm implementation: the C++ row kernel `layer_norm/layer_norm_rows.cc`
    (about 1.6x faster, same accuracy, not bit-identical), or with 0 the air.api loop.

SMOLVLA_OFFN_TILING, SMOLVLA_LNQKV_TILING (default: computed)
    `runtime_loop_tiling_sizes` of the vit_o_ffn and vit_ln_qkv ELFs: how much of the
    GEMMs' launch-iteration loops is unrolled into the ELF's control stream. The stream
    is what the firmware parses at every dispatch, and time follows its size (vit_o_ffn:
    830 KB and 9.4 ms at the earlier 2,2, 715 KB and 8.4 ms from 6,6 up). By default this
    is computed from the launch grids' own extents (see
    smolvla_vision_builders.vit_o_ffn_runtime_tiling / vit_ln_qkv_runtime_tiling) --
    requesting at least a GEMM's own extent fully unrolls its runtime loop, which is what
    avoids BD-ID recycling (see the comment above _vit_ln_qkv_backend in
    smolvla_vision_encoder.py). The model output is bit-identical to the recycled 2,2.
    These env vars force a specific value instead, e.g. to reproduce 2,2 for comparison.

SMOLVLA_B_STATIONARY (default 1)
    Builds the K=768 vision GEMMs (fused QKV, O, fc1) B-stationary: N stays a launch
    dimension and each launch fills the weight tile once into L2 (tile_k_l2 = K), then
    loops over the M tiles inside the segment streaming only A. The weights are read from
    DRAM once per N block instead of once per (M, N) tile, and the launch grid shrinks
    from (M/256) x N-iterations to N-iterations. Output is bit-identical to 0. fc2 (K=3072,
    B block > one memtile) is unchanged.

SMOLVLA_B_PANEL_LAYOUT (default 1)
    Lays B out offline (host-side repack_gemm_b_panel_layout) so each (K-chunk,
    N-launch-block) slab is one dense contiguous DRAM run instead of tile_k_l2 separate
    tile_n-element bursts: the L3->L2 DMA becomes one burst per panel. QKV-only
    (vit_ln_qkv's 2-launch ELF): O/fc1/fc2 live in vit_o_ffn's 6-launch fused ELF, where
    the same fix measured a wash (launch-count overhead dominates, not DMA burst size --
    see Vu_exp/iron_vs_air-oct-11-2026), so it was reverted there, not shipped. Orthogonal
    to B_STATIONARY (burst size vs reread count) -- bit-identical output either way.
    `make verify` bit-identical to off (cosine 0.9993277192115784) at full production scale
    (M=3072, real HF weights); measured -0.4% to -2.6% vision-stage, 5-18%/rep on
    vit_ln_qkv specifically (2026-10-10/11, noisy -- small ELF, run-to-run variance is
    comparable to the effect size).

SMOLVLA_O_SWAP (default 1)
    Builds vit_o_ffn's O GEMM with tile 64x48x64 on the swapped herd (4 core rows on M, 8 columns
    on N) instead of the registry's 32x96x64 on 8 rows x 4 columns. Its own compiled object
    (mm_m64_n48.o); tile_k_l1 is unchanged, so the packed weights are too. 0 restores the registry.

SMOLVLA_FC1_TILE (default "96x64"; empty = the registry tile, 32x128x32)
    "<tile_n>x<tile_k_l1>": builds vit_o_ffn's fc1 GEMM (bias + GELU) with tile
    32 x tile_n x tile_k_l1. 96x64 is the shape QKV already uses, so fc1 links the same
    compiled object. Changes the fc1 weight packing (tile_k_l1).

SMOLVLA_FC2_SWAP (default 1)
    Builds vit_o_ffn's fc2 GEMM with tile 64x96x64 on the swapped herd. Its own compiled object
    (mm_m64_n96.o); same packed weights. 0 restores the registry.

    The three are part of the ELF cache name, and the action chunk at the gate is identical with
    and without them. Together they take vit_o_ffn from 25.9 to 22.8 ms per image and the vision
    stage from 160.4 to 148.7 ms (3 cameras, NPU2). vit_o_ffn's loop tiling stays the registry one
    (see smolvla_vision_builders.vit_o_ffn_runtime_tiling).

SMOLVLA_ZERO_COPY (default 1)
    Chains the three ELFs of every vision layer (and layer to layer) through shared device
    buffers instead of copying each intermediate to the host and back: Q|K|V, the attention
    output and the layer output stay resident (KernelCache `shared_alias` pools), so only the
    first layer's input is written and the last layer's output read. Host-only change: the
    ELFs and the output are identical, so it is not part of the ELF cache name. Off when
    per-layer hidden states are requested (`return_per_layer`) or with the CPU attention
    diagnostic. 0 restores the per-ELF host round trips.

SMOLVLA_FA_QSEG (default 1)
    Runs the FlashAttention q-block loop inside the segment instead of as a launch-grid
    axis: the same design and microkernels (bit-identical output), but head_groups *
    n_images sequential waves instead of q_blocks * head_groups * n_images. 0 restores
    the launch-grid axis. Read here, once, rather than separately in the encoder
    (compile) and the runtime (cache-dir naming), so the two can never disagree on which
    schedule is baked into the ELF.

SMOLVLA_FA_QK_PRETRANSPOSE (default 1)
    Transposes each 8x8 K block once, in place, before Q@K^T instead of re-transposing it
    for every row-block pair inside the matmul. Bit-identical output; Q@K^T drops from
    ~3000 to ~1650 cycles per 64x64 block at a ~340 cycle transpose pass. Part of the ELF
    cache name (`_kt`). 0 restores the in-matmul transpose.
"""

import os


def _sizes_override(name):
    v = os.environ.get(name)
    return [int(t) for t in v.split(",")] if v else None


LN_EXT = os.environ.get("SMOLVLA_LN_EXT", "1") == "1"
# Rows one LayerNorm kernel call normalizes; the best point on NPU2 (layer_norm_rows.cc).
LN_ROWS = 4
OFFN_TILING_OVERRIDE = _sizes_override("SMOLVLA_OFFN_TILING")
LNQKV_TILING_OVERRIDE = _sizes_override("SMOLVLA_LNQKV_TILING")
FA_Q_IN_SEGMENT = os.environ.get("SMOLVLA_FA_QSEG", "1") == "1"
FA_QK_PRETRANSPOSE = os.environ.get("SMOLVLA_FA_QK_PRETRANSPOSE", "1") == "1"
B_STATIONARY = os.environ.get("SMOLVLA_B_STATIONARY", "1") == "1"
B_PANEL_LAYOUT = os.environ.get("SMOLVLA_B_PANEL_LAYOUT", "1") == "1"
O_SWAP = os.environ.get("SMOLVLA_O_SWAP", "1") == "1"
FC1_TILE = os.environ.get(
    "SMOLVLA_FC1_TILE", "96x64"
)  # "<tile_n>x<tile_k_l1>" or "" (registry)
FC2_SWAP = os.environ.get("SMOLVLA_FC2_SWAP", "1") == "1"
ZERO_COPY = os.environ.get("SMOLVLA_ZERO_COPY", "1") == "1"
