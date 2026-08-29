#!/usr/bin/env python3
"""Test matmul using IRON's mm.cc to compare BFP16 precision.

Uses IRON's matmul_bf16_bf16 function with row-major A tiles and
column-major B tiles (matching IRON's expected layout).
"""
import numpy as np
import torch
from ml_dtypes import bfloat16

import air
from air.ir import *
from air.dialects.air import *
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.func import FuncOp, CallOp
from air.backend.xrt_runner import XRTRunner, type_mapper

LQP = 64  # M
LKP = 64  # N (output cols = K^T cols)
DK = 64   # K (reduction)


@module_builder
def build_module():
    bf16 = BF16Type.get()
    i32 = IntegerType.get_signless(32)
    l1_space = IntegerAttr.get(i32, 2)

    # L1 buffers — flat (pre-tiled in Python for IRON's expected layout)
    memref_a_l1 = MemRefType.get([LQP * DK], bf16, memory_space=l1_space)
    memref_b_l1 = MemRefType.get([LKP * DK], bf16, memory_space=l1_space)
    memref_c_l1 = MemRefType.get([LQP * LKP], bf16, memory_space=l1_space)

    # L3 types
    memref_a_l3 = MemRefType.get([LQP * DK], bf16)
    memref_b_l3 = MemRefType.get([LKP * DK], bf16)
    memref_c_l3 = MemRefType.get([LQP * LKP], bf16)

    Channel("ChanInA")
    Channel("ChanInB")
    Channel("ChanOut")

    def external_func(name, inputs, outputs=None, link_with=None):
        if outputs is None:
            outputs = []
        func_type = FunctionType.get(inputs, outputs)
        func = FuncOp(name=name, type=func_type, visibility="private")
        func.attributes["llvm.emit_c_interface"] = UnitAttr.get()
        if link_with:
            func.attributes["link_with"] = StringAttr.get(link_with)
        return func

    # IRON's zero and matmul functions
    external_func("zero_bf16", [memref_c_l1], link_with="iron_mm.o")
    external_func(
        "matmul_bf16_bf16",
        [memref_a_l1, memref_b_l1, memref_c_l1],
        link_with="iron_mm.o",
    )

    @FuncOp.from_py_func(memref_a_l3, memref_b_l3, memref_c_l3)
    def test_matmul_iron(a_in, b_in, c_out):

        @launch(operands=[a_in, b_in, c_out])
        def launch_body(a, b, out):
            ChannelPut("ChanInA", a)
            ChannelPut("ChanInB", b)
            ChannelGet("ChanOut", out)

            @segment(name="seg")
            def segment_body():

                @herd(
                    name="herd_0",
                    sizes=[1, 1],
                    link_with="iron_mm.o",
                )
                def herd_body(tx, ty, sx, sy):
                    a_buf = AllocOp(memref_a_l1, [], [])
                    b_buf = AllocOp(memref_b_l1, [], [])
                    c_buf = AllocOp(memref_c_l1, [], [])

                    ChannelGet("ChanInA", a_buf.result)
                    ChannelGet("ChanInB", b_buf.result)

                    CallOp([], "zero_bf16", [c_buf.result])
                    CallOp(
                        [],
                        "matmul_bf16_bf16",
                        [a_buf.result, b_buf.result, c_buf.result],
                    )

                    ChannelPut("ChanOut", c_buf.result)

                    DeallocOp(a_buf)
                    DeallocOp(b_buf)
                    DeallocOp(c_buf)


def to_row_major_tiles(mat, rows, cols, tile=8):
    """Row-major tile order (IRON's A layout): row blocks contiguous."""
    tiled = np.zeros(rows * cols, dtype=mat.dtype)
    idx = 0
    for row_blk in range(rows // tile):
        for col_blk in range(cols // tile):
            for row_in in range(tile):
                for col_in in range(tile):
                    tiled[idx] = mat[row_blk * tile + row_in, col_blk * tile + col_in]
                    idx += 1
    return tiled


def to_col_major_tiles(mat, rows, cols, tile=8):
    """Column-major tile order (IRON's B layout with B_COL_MAJ)."""
    tiled = np.zeros(rows * cols, dtype=mat.dtype)
    idx = 0
    for col_blk in range(cols // tile):
        for row_blk in range(rows // tile):
            for row_in in range(tile):
                for col_in in range(tile):
                    tiled[idx] = mat[row_blk * tile + row_in, col_blk * tile + col_in]
                    idx += 1
    return tiled


if __name__ == "__main__":
    mlir_module = build_module()

    torch.manual_seed(42)
    val_range = 4.0
    inv_scale = 1.0 / np.sqrt(DK)

    Q_torch = torch.rand(LQP, DK, dtype=torch.bfloat16) * val_range
    K_torch = torch.rand(LKP, DK, dtype=torch.bfloat16) * val_range

    import os
    use_prescaled = os.environ.get("PRESCALE_Q", "0") == "1"
    Q_input = (Q_torch * inv_scale).to(torch.bfloat16) if use_prescaled else Q_torch
    print(f"Q range: [{Q_input.min():.4f}, {Q_input.max():.4f}]")
    print(f"K range: [{K_torch.min():.4f}, {K_torch.max():.4f}]")

    def torch_bf16_to_np(t):
        return t.contiguous().view(torch.uint16).numpy().view(bfloat16)

    # IRON matmul: A is row-major tiles, B is column-major tiles (B_COL_MAJ)
    # C = A @ B where B is stored column-major.
    # For Q@K^T: A=Q [M,K] row-major tiles, B=K^T [K,N].
    # With B_COL_MAJ: B is stored as column-major tiles of [K, N].
    # But K is originally [N, K] (each row is a K-vector).
    # K^T is [K, N]. In column-major tiles: col_blk iterates N, row_blk iterates K.
    input_a = to_row_major_tiles(torch_bf16_to_np(Q_input), LQP, DK)
    # K^T: transpose K from [LKP, DK] to [DK, LKP], then column-major tile
    K_T = torch_bf16_to_np(K_torch).reshape(LKP, DK).T  # [DK, LKP]
    input_b = to_col_major_tiles(K_T, DK, LKP)

    # Reference: f32 matmul → bf16
    G_ref = torch.matmul(Q_input.float(), K_torch.float().T).to(torch.bfloat16)
    # IRON's C output with B_COL_MAJ: C is column-major tiles (c_col_maj defaults differ)
    # Actually with B_COL_MAJ and no C_COL_MAJ: C is row-major tiles
    expected_output = to_row_major_tiles(torch_bf16_to_np(G_ref), LQP, LKP)

    print(f"Expected score range: [{G_ref.min():.2f}, {G_ref.max():.2f}], mean={G_ref.float().mean():.2f}")

    runner = XRTRunner(
        verbose=False,
        output_format="elf",
        instance_name="test_matmul_iron",
    )

    rc = runner.run_test(
        mlir_module,
        inputs=[input_a, input_b],
        expected_outputs=[expected_output],
        atol=0.15,
        rtol=0.04,
        error_threshold=0.005,
    )

    if rc != 0:
        print("IRON MATMUL TEST: IRON mm.cc diverges from f32→bf16 reference")
    else:
        print("IRON MATMUL TEST: IRON mm.cc matches f32→bf16 reference")

    exit(rc)
