# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AWQ packing helpers used by the int4-AWQ prefill stitchers.

Two kinds of input get bridged to the int4 GEMM micro-kernel's packed BO
layout (matmul_int4_packed.pack_inputs):

* fake_quantize_awq_int4: turn a bf16 weight into (Q, S, Z). Used for
  testing the int4 path against a bf16 HF checkpoint without round-tripping
  through a real AWQ-calibrated quantizer.
* pack_weight_for_int4_gemm: prefill-orientation [K, N] bf16 weight
  -> 3D packed uint8 BO that the GEMM stitcher's weight slots consume.

The real AWQ-checkpoint loader path (HF AutoAWQ) reads (Q, S, Z) directly
from the checkpoint and skips fake_quantize_awq_int4 entirely; it still
goes through matmul_int4_packed.pack_inputs to land on the kernel's
expected layout.
"""

import os
import sys

import numpy as np
from ml_dtypes import bfloat16


_INT4_GEMM_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "matrix_multiplication",
    "int4_awq",
)
if _INT4_GEMM_DIR not in sys.path:
    sys.path.insert(0, _INT4_GEMM_DIR)
from matmul_int4_packed import pack_inputs as pack_inputs_gemm  # noqa: E402


def fake_quantize_awq_int4(W_bf16, gs=128):
    """Asymmetric uint4 per-group quantization of a [M, K] bf16 weight.

    Returns (W_q [M, K/2] uint8 packed, W_s [n_groups, M] bf16,
    W_z [n_groups, M] uint8) matching the layout pack_inputs expects.
    """
    M, K = W_bf16.shape
    assert K % gs == 0, f"K={K} not divisible by gs={gs}"
    n_groups = K // gs
    W_f32 = W_bf16.astype(np.float32)

    W_grp = W_f32.reshape(M, n_groups, gs)
    w_min = W_grp.min(axis=2)
    w_max = W_grp.max(axis=2)
    scale = (w_max - w_min) / 15.0
    scale = np.where(scale == 0, 1e-7, scale)
    zero = np.round(-w_min / scale).clip(0, 15).astype(np.uint8)

    q = np.round(W_f32 / np.repeat(scale, gs, axis=1)
                 + np.repeat(zero.astype(np.float32), gs, axis=1)
                 ).clip(0, 15).astype(np.uint8)
    W_q = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
    W_s = scale.T.astype(bfloat16)
    W_z = zero.T.astype(np.uint8)
    return W_q, W_s, W_z


def pack_weight_for_int4_gemm(W_bf16, M_seq, gs=128, n_tile=16, k_chunk=128):
    """Quantize and pack a prefill-orientation [K, N] bf16 weight for the
    int4-AWQ GEMM kernel (matmul_int4_packed).

    Prefill stores weights as [K, N] (K outer; matches A @ W convention with
    A: [seq, K]). The int4 packer expects the weight in [N, K] output-major,
    so transpose first.

    Returns packed BO of shape [N // n_tile, K // k_chunk, tile_bytes] uint8,
    matching matmul_int4_packed.build_module's arg1 type. M_seq is the
    activation row count the GEMM will be invoked on (pack_inputs uses it
    for stride math; the resulting BO layout doesn't depend on it).
    """
    W_NK = np.ascontiguousarray(W_bf16.T)
    N, K = W_NK.shape
    W_q, W_s, W_z = fake_quantize_awq_int4(W_NK, gs=gs)
    return pack_inputs_gemm(W_q, W_s, W_z, M_seq, K, N, gs, n_tile, k_chunk)
