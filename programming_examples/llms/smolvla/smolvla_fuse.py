# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build-time choices for the SigLIP vision ELFs, read once and shared by the encoder
(what to compile and dispatch) and the runtime (which cache directory to use), so the
two cannot disagree. Every value can be overridden from the environment.

SMOLVLA_LN_EXT (default 1)
    The LayerNorm implementation: the C++ row kernel `layer_norm/layer_norm_rows.cc`
    (about 1.6x faster, same accuracy, not bit-identical), or with 0 the air.api loop.

SMOLVLA_OFFN_TILING, SMOLVLA_LNQKV_TILING (default 12,6)
    `runtime_loop_tiling_sizes` of the vit_o_ffn and vit_ln_qkv ELFs: how much of the
    GEMMs' launch-iteration loops is unrolled into the ELF's control stream. The stream
    is what the firmware parses at every dispatch, and time follows its size (vit_o_ffn:
    830 KB and 9.4 ms at the earlier 2,2, 715 KB and 8.4 ms from 6,6 up). 12,6 is the
    launch grids' own extents. The model output is bit-identical to 2,2.
"""

import os


def _sizes(name, default):
    return [int(t) for t in os.environ.get(name, default).split(",")]


LN_EXT = os.environ.get("SMOLVLA_LN_EXT", "1") == "1"
# Rows one LayerNorm kernel call normalizes; the best point on NPU2 (layer_norm_rows.cc).
LN_ROWS = 4
OFFN_TILING = _sizes("SMOLVLA_OFFN_TILING", "12,6")
LNQKV_TILING = _sizes("SMOLVLA_LNQKV_TILING", "12,6")
