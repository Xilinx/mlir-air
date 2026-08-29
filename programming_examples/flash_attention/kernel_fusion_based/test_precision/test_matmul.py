#!/usr/bin/env python3
"""Test 1: Matmul only — verify Q@K^T scores on hardware vs Python reference.

Uses L3→L2→L1 DMA tiling matching the flash attention pipeline to ensure
data arrives at the kernel in the correct tiled format.
"""
import os
import numpy as np
import torch
from ml_dtypes import bfloat16

import air
from air.ir import *
from air.dialects.air import *
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.func import FuncOp, CallOp
from air.dialects.scf import for_ as range_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper

# Tile dimensions matching flash attention kernel
LQP = 64  # tile_size_q (lqp in kernel)
LKP = 64  # K chunk size (lkp in kernel)
DK = 64   # key dimension (dk in kernel)
MMUL_M, MMUL_K, MMUL_N = 8, 8, 8


@module_builder
def build_module():
    bf16 = BF16Type.get()
    i32 = IntegerType.get_signless(32)
    index_type = IndexType.get()
    l1_space = IntegerAttr.get(i32, 2)
    l2_space = IntegerAttr.get(i32, 1)

    # L1 types
    memref_q_l1 = MemRefType.get([LQP, DK], bf16, memory_space=l1_space)
    memref_k_l1 = MemRefType.get([LKP, DK], bf16, memory_space=l1_space)
    memref_g_l1 = MemRefType.get([LQP * LKP], bf16, memory_space=l1_space)

    # L2 types (staging buffers)
    memref_q_l2 = MemRefType.get([LQP, DK], bf16, memory_space=l2_space)
    memref_k_l2 = MemRefType.get([LKP, DK], bf16, memory_space=l2_space)
    memref_g_l2 = MemRefType.get([LQP, LKP], bf16, memory_space=l2_space)

    # L3 types (row-major, no tiling)
    memref_q_l3 = MemRefType.get([LQP, DK], bf16)
    memref_k_l3 = MemRefType.get([LKP, DK], bf16)
    memref_out_l3 = MemRefType.get([LQP, LKP], bf16)

    # Channels
    Channel("ChanQ_L3L2")
    Channel("ChanK_L3L2")
    Channel("ChanQ_L2L1")
    Channel("ChanK_L2L1")
    Channel("ChanG_L1L2")
    Channel("ChanG_L2L3")

    def external_func(name, inputs, outputs=None, link_with=None):
        if outputs is None:
            outputs = []
        func_type = FunctionType.get(inputs, outputs)
        func = FuncOp(name=name, type=func_type, visibility="private")
        func.attributes["llvm.emit_c_interface"] = UnitAttr.get()
        if link_with:
            func.attributes["link_with"] = StringAttr.get(link_with)
        return func

    external_func("zero_fill_g_bf16", [memref_g_l1], link_with="kernels.o")
    external_func(
        "matmul_a_b_bf16",
        [memref_q_l1, memref_k_l1, memref_g_l1],
        link_with="kernels.o",
    )

    @FuncOp.from_py_func(memref_q_l3, memref_k_l3, memref_out_l3)
    def test_matmul(q_in, k_in, g_out):

        @launch(operands=[q_in, k_in, g_out])
        def launch_body(q, k, out):
            # L3↔L2: flat transfers
            ChannelPut("ChanQ_L3L2", q)
            ChannelPut("ChanK_L3L2", k)
            ChannelGet("ChanG_L2L3", out)

            @segment(name="seg")
            def segment_body():
                # L2 staging buffers
                q_l2 = AllocOp(memref_q_l2, [], [])
                k_l2 = AllocOp(memref_k_l2, [], [])
                g_l2 = AllocOp(memref_g_l2, [], [])

                # L3→L2 flat
                ChannelGet("ChanQ_L3L2", q_l2.result)
                ChannelGet("ChanK_L3L2", k_l2.result)

                # L2→L1 with DMA tiling matching flash attention:
                # sizes=[N/8, K/8, 8, 8], strides=[8*K, 8, K, 1]
                ChannelPut(
                    "ChanQ_L2L1",
                    q_l2.result,
                    offsets=[0, 0, 0, 0],
                    sizes=[LQP // MMUL_M, DK // MMUL_K, MMUL_M, MMUL_K],
                    strides=[MMUL_M * DK, MMUL_K, DK, 1],
                )
                ChannelPut(
                    "ChanK_L2L1",
                    k_l2.result,
                    offsets=[0, 0, 0, 0],
                    sizes=[LKP // MMUL_N, DK // MMUL_K, MMUL_N, MMUL_K],
                    strides=[MMUL_N * DK, MMUL_K, DK, 1],
                )

                # L1→L2 with reverse tiling for output G
                ChannelGet(
                    "ChanG_L1L2",
                    g_l2.result,
                    offsets=[0, 0, 0, 0],
                    sizes=[LQP // MMUL_M, LKP // MMUL_N, MMUL_M, MMUL_N],
                    strides=[MMUL_M * LKP, MMUL_N, LKP, 1],
                )

                # L2→L3 flat
                ChannelPut("ChanG_L2L3", g_l2.result)

                @herd(
                    name="herd_0",
                    sizes=[1, 1],
                    link_with="kernels.o",
                )
                def herd_body(tx, ty, sx, sy):
                    q_buf = AllocOp(memref_q_l1, [], [])
                    k_buf = AllocOp(memref_k_l1, [], [])
                    g_buf = AllocOp(memref_g_l1, [], [])

                    # L2→L1 receive (flat at herd level, tiling done at segment)
                    ChannelGet("ChanQ_L2L1", q_buf.result)
                    ChannelGet("ChanK_L2L1", k_buf.result)

                    CallOp([], "zero_fill_g_bf16", [g_buf.result])
                    CallOp(
                        [],
                        "matmul_a_b_bf16",
                        [q_buf.result, k_buf.result, g_buf.result],
                    )

                    # L1→L2 send
                    ChannelPut("ChanG_L1L2", g_buf.result)

                    DeallocOp(q_buf)
                    DeallocOp(k_buf)
                    DeallocOp(g_buf)

                DeallocOp(q_l2)
                DeallocOp(k_l2)
                DeallocOp(g_l2)


if __name__ == "__main__":
    mlir_module = build_module()

    torch.manual_seed(42)
    val_range = float(os.environ.get("VAL_RANGE", "4"))
    inv_scale = 1.0 / np.sqrt(DK)

    Q_torch = torch.rand(LQP, DK, dtype=torch.bfloat16) * val_range
    K_torch = torch.rand(LKP, DK, dtype=torch.bfloat16) * val_range

    use_prescaled = os.environ.get("PRESCALE_Q", "0") == "1"
    Q_input = (Q_torch * inv_scale).to(torch.bfloat16) if use_prescaled else Q_torch
    print(f"{'PRE-SCALED' if use_prescaled else 'UNSCALED'} Q: [{Q_input.min():.4f}, {Q_input.max():.4f}]")
    print(f"K: [{K_torch.min():.4f}, {K_torch.max():.4f}]")

    def torch_bf16_to_np(t):
        return t.contiguous().view(torch.uint16).numpy().view(bfloat16)

    # Send row-major to L3 — DMA handles tiling
    input_q = torch_bf16_to_np(Q_input)
    input_k = torch_bf16_to_np(K_torch)

    # Reference: f32 matmul → bf16, row-major output
    G_ref = torch.matmul(Q_input.float(), K_torch.float().T).to(torch.bfloat16)
    expected_output = torch_bf16_to_np(G_ref)
    print(f"Expected scores: [{G_ref.min():.2f}, {G_ref.max():.2f}], mean={G_ref.float().mean():.2f}")

    runner = XRTRunner(
        verbose=False,
        output_format="elf",
        instance_name="test_matmul",
    )

    rc = runner.run_test(
        mlir_module,
        inputs=[input_q, input_k],
        expected_outputs=[expected_output],
        atol=0.15,
        rtol=0.04,
        error_threshold=0.005,
    )

    if rc != 0:
        print("MATMUL TEST FAILED")
    else:
        print("MATMUL TEST PASSED")
    exit(rc)
