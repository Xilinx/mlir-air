# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""HuggingFace AutoAWQ checkpoint -> per-tile packed BO layout used by
mlir-air's int4-AWQ GEMV kernels (decode path).

AutoAWQ stores each Linear's quantized weights as three tensors:
    qweight:  [in_features=K, out_features // 8] int32
              (8 uint4 nibbles packed along N per int32, interleaved by AWQ_PACK_ORDER)
    qzeros:   [K // group_size, out_features // 8] int32 (same packing)
    scales:   [K // group_size, out_features] fp16

mlir-air's `matvec_int4_packed.pack_inputs` expects:
    A_q[M=out, K/2] uint8  (col 2i = low nibble, col 2i+1 = high nibble)
    A_s[n_groups, M] bf16
    A_z[n_groups, M] uint8

This module bridges the two formats for the **decode GEMV** path. The
sibling `awq_pack.py` does the analogous bridge for the **prefill GEMM**
path (`matmul_int4_packed.pack_inputs`). A built-in self-test (run via
`python3 awq_repacker.py`) generates synthetic AWQ tensors, repacks,
and verifies that the repacked form dequantizes to exactly the same
bf16 weights as a direct dense dequant.
"""

import argparse
import os
import sys

import numpy as np
from ml_dtypes import bfloat16

# AutoAWQ packs 8 uint4 nibbles into each int32, but the nibble at bit
# position 4*i within the int32 holds the *unpacked* output column
# `8*col_block + AWQ_PACK_ORDER[i]`. See autoawq.utils.packing_utils.pack.
AWQ_PACK_ORDER = np.array([0, 2, 4, 6, 1, 3, 5, 7], dtype=np.int64)
# Inverse: AWQ_UNPACK_PERM[k] == bit position holding output column k.
AWQ_UNPACK_PERM = np.argsort(AWQ_PACK_ORDER)  # = [0, 4, 1, 5, 2, 6, 3, 7]


def unpack_awq_int32(packed: np.ndarray) -> np.ndarray:
    """Unpack AutoAWQ int32 -> uint8 nibbles along the last axis.

    Args:
        packed: int32 array, last axis is the packed-N axis of size N//8.

    Returns:
        uint8 array, last axis size N, values in [0, 16).
    """
    packed64 = packed.astype(np.int64)
    shifts = np.arange(8, dtype=np.int64) * 4
    nibs = ((packed64[..., :, None] >> shifts) & 0xF).astype(np.uint8)
    nibs_reordered = nibs[..., AWQ_UNPACK_PERM]
    return nibs_reordered.reshape(*packed.shape[:-1], packed.shape[-1] * 8)


def dequant_to_bf16(qweight, qzeros, scales, group_size):
    """HF AutoAWQ tensors -> dense bf16 weight matrix [in_features, out_features].

    Matches transformer_block's `wq[in, out]` storage convention so the result
    can be assigned directly to LayerWeights.wq / wk / wv / wo / w_gate /
    w_up / w_down for the bf16-prefill-on-dequant path.

    Dequant formula: w[k, n] = (qweight_u[k, n] - qzeros_u[k//gs, n]) * scales[k//gs, n].
    """
    qweight_u = unpack_awq_int32(qweight)  # [K, N] uint8
    qzeros_u = unpack_awq_int32(qzeros)  # [K/gs, N] uint8
    K, N = qweight_u.shape
    n_groups = K // group_size
    if qzeros_u.shape != (n_groups, N):
        raise ValueError(f"qzeros shape {qzeros_u.shape} vs expected ({n_groups}, {N})")
    if scales.shape != (n_groups, N):
        raise ValueError(f"scales shape {scales.shape} vs expected ({n_groups}, {N})")
    # Round scales to bf16 first so this matches what the NPU kernel actually
    # sees (the packed BO carries bf16 scales). fp16->bf16 loses 3 mantissa
    # bits — intentional precision drift relative to the canonical AWQ fp16
    # dequant.
    s_bf16_as_f32 = scales.astype(bfloat16).astype(np.float32)
    z_per_k = np.repeat(qzeros_u.astype(np.int32), group_size, axis=0)  # [K, N]
    s_per_k = np.repeat(s_bf16_as_f32, group_size, axis=0)  # [K, N]
    w_f32 = (qweight_u.astype(np.int32) - z_per_k) * s_per_k
    return w_f32.astype(bfloat16)


def repack_hf_awq_linear(qweight, qzeros, scales, group_size):
    """HF AutoAWQ tensors -> (A_q, A_s, A_z) in mlir-air `pack_inputs` format.

    Returns:
        A_q: uint8 [M=out_features, K/2], packed nibble pairs (col 2i = low,
             col 2i+1 = high).
        A_s: bf16 [n_groups, M] (lossy fp16->bf16 cast on AWQ's smooth scales).
        A_z: uint8 [n_groups, M], values in [0, 16).
    """
    qweight_u = unpack_awq_int32(qweight)  # [K, N]
    qzeros_u = unpack_awq_int32(qzeros)  # [K/gs, N]
    K, N = qweight_u.shape
    n_groups = K // group_size
    if qzeros_u.shape != (n_groups, N):
        raise ValueError(f"qzeros shape {qzeros_u.shape} vs expected ({n_groups}, {N})")
    if scales.shape != (n_groups, N):
        raise ValueError(f"scales shape {scales.shape} vs expected ({n_groups}, {N})")
    # Transpose K-major (HF) -> M-major: weight[m=n, k] = qweight_u[k, n].
    q_mn = np.ascontiguousarray(qweight_u.T)  # [M, K]
    low = q_mn[:, 0::2] & 0x0F
    high = (q_mn[:, 1::2] & 0x0F) << 4
    A_q = (low | high).astype(np.uint8)  # [M, K/2]
    A_s = np.ascontiguousarray(scales).astype(bfloat16)  # [n_groups, M]
    A_z = np.ascontiguousarray(qzeros_u).astype(np.uint8)  # [n_groups, M]
    return A_q, A_s, A_z


def repack_for_gemv(
    qweight, qzeros, scales, group_size, M_TILE=8, K_CHUNK=2048, N_CORES=8
):
    """HF AutoAWQ -> [total_tiles, tile_bytes] uint8 BO ready for mlir-air decode.

    Calls `matvec_int4_packed.pack_inputs` under the hood. Single-launch
    layout (`M_PER_LAUNCH = M`) — matches the int4 decode ELF builders.
    """
    sys.path.insert(
        0,
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "matrix_vector_multiplication",
            "int4_awq",
        ),
    )
    from matvec_int4_packed import pack_inputs  # type: ignore

    A_q, A_s, A_z = repack_hf_awq_linear(qweight, qzeros, scales, group_size)
    M, K_half = A_q.shape
    K = K_half * 2
    return pack_inputs(A_q, A_s, A_z, M, K, group_size, M_TILE, K_CHUNK, N_CORES, M)


# ---------------------------------------------------------------------------
# Synthetic AWQ generator + self-test
# ---------------------------------------------------------------------------


def _gen_synthetic_awq(K, N, group_size, seed=42):
    """Produce HF-AutoAWQ-shaped tensors from a random nibble matrix.

    Returns (qweight[K, N//8] int32, qzeros[K/gs, N//8] int32,
             scales[K/gs, N] fp16, dense_ref[K, N] uint8) where dense_ref
    is the un-packed nibble matrix used to verify the unpack path.
    """
    rng = np.random.default_rng(seed)
    n_groups = K // group_size
    nibs = rng.integers(0, 16, size=(K, N), dtype=np.uint8)
    z_nibs = rng.integers(0, 16, size=(n_groups, N), dtype=np.uint8)
    scales = rng.uniform(0.005, 0.02, size=(n_groups, N)).astype(np.float16)

    n_blocks = N // 8
    qweight = np.zeros((K, n_blocks), dtype=np.int32)
    qzeros = np.zeros((n_groups, n_blocks), dtype=np.int32)
    for i in range(8):
        col = AWQ_PACK_ORDER[i]
        qweight |= nibs[:, col::8].astype(np.int32) << (4 * i)
        qzeros |= z_nibs[:, col::8].astype(np.int32) << (4 * i)
    return qweight, qzeros, scales, nibs, z_nibs


def self_test(K=512, N=128, group_size=128, seed=42, verbose=True):
    """Round-trip check: pack synthetic AWQ -> repack -> dequant matches
    direct dense dequant. Algebraically identical up to bf16 rounding.
    """
    qweight, qzeros, scales, nibs_ref, z_nibs_ref = _gen_synthetic_awq(
        K, N, group_size, seed=seed
    )

    # (a) unpack round-trip: confirms AWQ_PACK_ORDER handling.
    nibs_unpacked = unpack_awq_int32(qweight)
    if not np.array_equal(nibs_unpacked, nibs_ref):
        wrong = (nibs_unpacked != nibs_ref).sum()
        raise AssertionError(
            f"unpack_awq_int32 mismatch on {wrong} / {nibs_ref.size} nibbles"
        )
    z_unpacked = unpack_awq_int32(qzeros)
    if not np.array_equal(z_unpacked, z_nibs_ref):
        raise AssertionError("qzeros unpack mismatch")
    if verbose:
        print(f"  [a] AWQ_PACK_ORDER unpack: PASS ({K}x{N} nibbles)")

    # (b) dense dequant and our repack agree on every (k, n).
    w_dense = dequant_to_bf16(qweight, qzeros, scales, group_size)
    A_q, A_s, A_z = repack_hf_awq_linear(qweight, qzeros, scales, group_size)
    M = A_q.shape[0]
    K2 = A_q.shape[1] * 2
    if (M, K2) != (N, K):
        raise AssertionError(f"repack shape mismatch: ({M}, {K2}) vs ({N}, {K})")
    nibs_from_packed = np.zeros((M, K2), dtype=np.uint8)
    nibs_from_packed[:, 0::2] = A_q & 0x0F
    nibs_from_packed[:, 1::2] = (A_q >> 4) & 0x0F
    z_per_k = np.repeat(A_z.astype(np.int32), group_size, axis=0)  # [K, M]
    s_per_k = np.repeat(A_s.astype(np.float32), group_size, axis=0)  # [K, M]
    w_repacked_f32 = (nibs_from_packed.astype(np.int32) - z_per_k.T) * s_per_k.T
    w_repacked = w_repacked_f32.astype(bfloat16)
    if not np.array_equal(w_dense, w_repacked.T):
        diff = w_dense.astype(np.float32) - w_repacked.T.astype(np.float32)
        mx = np.max(np.abs(diff))
        raise AssertionError(f"dense vs repacked dequant mismatch: max |Δ| = {mx}")
    if verbose:
        print(f"  [b] dense vs repacked dequant: PASS")

    # (c) end-to-end vs matvec_int4_packed.cpu_reference on a random input.
    sys.path.insert(
        0,
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "matrix_vector_multiplication",
            "int4_awq",
        ),
    )
    from matvec_int4_packed import cpu_reference  # type: ignore

    rng = np.random.default_rng(seed + 1)
    x = rng.standard_normal(K).astype(bfloat16)
    y_repacked = cpu_reference(A_q, A_s, A_z, x)
    y_dense = (w_dense.astype(np.float32).T @ x.astype(np.float32)).astype(bfloat16)
    corr = np.corrcoef(
        y_repacked.astype(np.float32).flatten(),
        y_dense.astype(np.float32).flatten(),
    )[0, 1]
    if not (corr >= 0.9999):
        raise AssertionError(
            f"end-to-end correlation {corr:.6f} below 0.9999 threshold"
        )
    if verbose:
        print(f"  [c] end-to-end (cpu_reference vs dense): PASS (corr={corr:.6f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="awq_repacker.py",
        description="HF AutoAWQ -> mlir-air packed-BO repacker + self-test.",
    )
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--gs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(
        f"AWQ repacker self-test: K={args.k}, N={args.n}, GS={args.gs}, "
        f"seed={args.seed}"
    )
    self_test(K=args.k, N=args.n, group_size=args.gs, seed=args.seed)
    print("All self-tests PASSED.")
