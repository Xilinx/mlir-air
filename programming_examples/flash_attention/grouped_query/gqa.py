# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Grouped Query Attention (GQA) Example

Implements GQA where multiple Q heads share a single K/V head:
  For each KV head group:
    For each Q head in group:
      S = Q[q_head] @ K[kv_head]^T
      P = softmax(S)
      O[q_head] = P @ V[kv_head]

Uses sequential launches, one per Q head. K and V are packed into a
single buffer per KV head to stay within DMA channel limits.

Small dimensions for demonstration: num_q_heads=4, num_kv_heads=2,
lq=16, lk=16, d=16.

Target: AIE2P (NPU2). Requires ELF output format.
"""

import argparse
import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.air import *
from air.dialects import arith
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.func import FuncOp, CallOp
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend


@module_builder
def build_module(num_q_heads, num_kv_heads, lq, lk, d, np_dtype):
    xrt_dtype = type_mapper(np_dtype)
    i32 = IntegerType.get_signless(32)
    index_type = IndexType.get()

    assert num_kv_heads > 0, "num_kv_heads must be positive"
    assert (
        num_q_heads % num_kv_heads == 0
    ), f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
    group_size = num_q_heads // num_kv_heads

    q_head_size = lq * d
    kv_head_size = d * lk + lk * d  # K[d,lk] + V[lk,d] packed
    out_head_size = lq * d

    # L3 types: Q, KV_packed, Output (3 buffers = 2 S2MM + 1 MM2S)
    l3_q_ty = MemRefType.get([num_q_heads * q_head_size], xrt_dtype)
    l3_kv_ty = MemRefType.get([num_kv_heads * kv_head_size], xrt_dtype)
    l3_out_ty = MemRefType.get([num_q_heads * out_head_size], xrt_dtype)

    # L1 types
    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_q_ty = MemRefType.get([q_head_size], xrt_dtype, memory_space=l1_space)
    l1_kv_ty = MemRefType.get([kv_head_size], xrt_dtype, memory_space=l1_space)
    l1_s_ty = MemRefType.get([lq * lk], xrt_dtype, memory_space=l1_space)
    l1_out_ty = MemRefType.get([out_head_size], xrt_dtype, memory_space=l1_space)

    # External kernels
    attn_func = FuncOp(
        "attention_head_bf16",
        ([l1_q_ty, l1_kv_ty, l1_out_ty], []),
        visibility="private",
    )
    attn_func.attributes["link_with"] = StringAttr.get("gqa_kernels.o")
    attn_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    @FuncOp.from_py_func(l3_q_ty, l3_kv_ty, l3_out_ty)
    def gqa(arg_q, arg_kv, arg_out):

        for q_head in range(num_q_heads):
            kv_head = q_head // group_size

            @launch(operands=[arg_q, arg_kv, arg_out])
            def attn_launch(lq_buf, lkv_buf, lo_buf):

                @segment(
                    name=f"seg_{q_head}",
                    operands=[lq_buf, lkv_buf, lo_buf],
                )
                def seg(sq, skv, so):

                    @herd(
                        name=f"h_{q_head}",
                        sizes=[1, 1],
                        operands=[sq, skv, so],
                        link_with="gqa_kernels.o",
                    )
                    def h(_tx, _ty, _sx, _sy, hq, hkv, ho):
                        l1_q = AllocOp(l1_q_ty, [], [])
                        l1_kv = AllocOp(l1_kv_ty, [], [])
                        l1_out = AllocOp(l1_out_ty, [], [])

                        # DMA Q head
                        q_off = ConstantOp(index_type, q_head * q_head_size)
                        dma_memcpy_nd(
                            l1_q,
                            hq,
                            src_offsets=[q_off],
                            src_sizes=[q_head_size],
                            src_strides=[1],
                        )
                        # DMA KV head (K and V packed together)
                        kv_off = ConstantOp(index_type, kv_head * kv_head_size)
                        dma_memcpy_nd(
                            l1_kv,
                            hkv,
                            src_offsets=[kv_off],
                            src_sizes=[kv_head_size],
                            src_strides=[1],
                        )

                        # Single kernel call: Q @ K^T → softmax → P @ V
                        CallOp(attn_func, [l1_q, l1_kv, l1_out])

                        # DMA output
                        out_off = ConstantOp(index_type, q_head * out_head_size)
                        dma_memcpy_nd(
                            ho,
                            l1_out,
                            dst_offsets=[out_off],
                            dst_sizes=[out_head_size],
                            dst_strides=[1],
                        )

                        DeallocOp(l1_q)
                        DeallocOp(l1_kv)
                        DeallocOp(l1_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Grouped Query Attention (GQA) example",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--num-q-heads", type=int, default=4)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--lq", type=int, default=16, help="Query sequence length")
    parser.add_argument("--lk", type=int, default=16, help="Key sequence length")
    parser.add_argument("--d", type=int, default=16, help="Head dimension")
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="elf",
        dest="output_format",
    )
    args = parser.parse_args()

    num_q_heads = args.num_q_heads
    num_kv_heads = args.num_kv_heads
    lq = args.lq
    lk = args.lk
    d = args.d

    if num_kv_heads <= 0:
        parser.error("num_kv_heads must be positive")
    if num_q_heads % num_kv_heads != 0:
        parser.error(
            f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )

    group_size = num_q_heads // num_kv_heads
    INPUT_DATATYPE = bfloat16

    mlir_module = build_module(num_q_heads, num_kv_heads, lq, lk, d, INPUT_DATATYPE)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    np.random.seed(0)
    Q = (np.random.randn(num_q_heads, lq, d) * 0.1).astype(INPUT_DATATYPE)
    K = (np.random.randn(num_kv_heads, d, lk) * 0.1).astype(INPUT_DATATYPE)
    V = (np.random.randn(num_kv_heads, lk, d) * 0.1).astype(INPUT_DATATYPE)

    # Pack K and V per KV head: [K_head | V_head] contiguous
    kv_packed = np.zeros(num_kv_heads * (d * lk + lk * d), dtype=INPUT_DATATYPE)
    for kv_h in range(num_kv_heads):
        base = kv_h * (d * lk + lk * d)
        kv_packed[base : base + d * lk] = K[kv_h].reshape(-1)
        kv_packed[base + d * lk : base + d * lk + lk * d] = V[kv_h].reshape(-1)

    # Reference GQA (f32)
    ref_out = np.zeros((num_q_heads, lq, d), dtype=INPUT_DATATYPE)
    for q_head in range(num_q_heads):
        kv_head = q_head // group_size
        Q_f32 = Q[q_head].astype(np.float32)
        K_f32 = K[kv_head].astype(np.float32)
        V_f32 = V[kv_head].astype(np.float32)
        S = Q_f32 @ K_f32  # [lq, lk]
        S_max = np.max(S, axis=-1, keepdims=True)
        S_exp = np.exp(S - S_max)
        S_sum = np.sum(S_exp, axis=-1, keepdims=True)
        P = S_exp / S_sum
        O = P @ V_f32  # [lq, d]
        ref_out[q_head] = O.astype(INPUT_DATATYPE)

    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        omit_pingpong=True,
        output_format=args.output_format,
        instance_name="gqa",
    )
    exit(
        runner.run_test(
            mlir_module,
            inputs=[Q.reshape(-1), kv_packed],
            expected_outputs=[ref_out.reshape(-1)],
            rtol=1e0,
            atol=0.5,
        )
    )
