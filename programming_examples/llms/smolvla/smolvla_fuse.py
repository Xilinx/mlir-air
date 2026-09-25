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

SMOLVLA_FA_QSEG (default 0)
    Runs the FlashAttention q-block loop inside the segment instead of as a launch-grid
    axis: the same design and microkernels (bit-identical output), but head_groups *
    n_images sequential waves instead of q_blocks * head_groups * n_images. Read here,
    once, rather than separately in the encoder (compile) and the runtime (cache-dir
    naming), so the two can never disagree on which schedule is baked into the ELF.
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
FA_Q_IN_SEGMENT = os.environ.get("SMOLVLA_FA_QSEG", "0") == "1"
