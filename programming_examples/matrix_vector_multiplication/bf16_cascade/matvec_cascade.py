# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Cascade-based matrix-vector multiplication (GEMV): C[M] = A[M,K] @ B[K]
# BF16 input/output, accfloat accumulation in the kernel.
#
# Uses cascade channels to split K-reduction across `n_cascade` tiles in each
# column.  Each column handles `tile_m` independent output rows.  `herd_cols`
# columns run in parallel.
#
# Cascade flows north-to-south within each column:
#   ty == n_cascade-1 (first/northernmost): compute partial dot → cascade put
#   ty == 1..n_cascade-2 (middle): cascade get → add own partial → cascade put
#   ty == 0 (last/southernmost): cascade get → add own partial → write result
#
# L2 (MemTile) staging for A and C; B goes L3→L1 directly (broadcast).
#
# The cascade data transfer is compiler-managed: air.channel.put/get on cascade
# channels are lowered to aie.put_cascade/aie.get_cascade by the compiler.
# The kernels do NOT call put_mcd/get_scd directly.

import argparse
import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects import arith, scf
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.func import FuncOp, CallOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend

range_ = for_


@module_builder
def build_module(
    m, k, tile_m, m_input, herd_cols, n_cascade, np_dtype_in, np_dtype_out
):
    assert (
        n_cascade >= 2
    ), f"n_cascade ({n_cascade}) must be >= 2 for a cascade pipeline"
    k_chunk = k // n_cascade
    assert (
        m % (tile_m * herd_cols) == 0
    ), f"M ({m}) must be divisible by tile_m * herd_cols ({tile_m * herd_cols})"
    assert (
        tile_m % m_input == 0
    ), f"tile_m ({tile_m}) must be divisible by m_input ({m_input})"
    assert k % n_cascade == 0, f"K ({k}) must be divisible by n_cascade ({n_cascade})"
    assert (
        k_chunk % 64 == 0
    ), f"k_chunk ({k_chunk}) must be divisible by 64 (vector width)"

    # L2 capacity guard.
    bytes_per_elem_in = np.dtype(np_dtype_in).itemsize
    bytes_per_elem_out = np.dtype(np_dtype_out).itemsize
    a_l2_bytes = herd_cols * tile_m * k * bytes_per_elem_in
    c_l2_bytes = herd_cols * tile_m * bytes_per_elem_out
    L2_CAPACITY = 512 * 1024
    assert a_l2_bytes + c_l2_bytes <= L2_CAPACITY, (
        f"L2 capacity exceeded: A={a_l2_bytes}B + C={c_l2_bytes}B = "
        f"{a_l2_bytes + c_l2_bytes}B > {L2_CAPACITY}B. "
        f"Reduce herd_cols ({herd_cols}), tile_m ({tile_m}), or k ({k})."
    )

    xrt_dtype_in = type_mapper(np_dtype_in)
    xrt_dtype_out = type_mapper(np_dtype_out)
    f32_type = F32Type.get()

    # L3 MemRefTypes
    memrefTyA = MemRefType.get([m, k], xrt_dtype_in)
    memrefTyB = MemRefType.get([k], xrt_dtype_in)
    memrefTyC = MemRefType.get([m], xrt_dtype_out)

    # L2 MemRefTypes — A staged as [herd_cols, tile_m, k]
    l2_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L2)
    l2MemrefTyA = MemRefType.get(
        shape=[herd_cols, tile_m, k],
        element_type=xrt_dtype_in,
        memory_space=l2_mem_space,
    )
    l2MemrefTyC = MemRefType.get(
        shape=[herd_cols, tile_m],
        element_type=xrt_dtype_out,
        memory_space=l2_mem_space,
    )

    # L1 MemRefTypes
    l1_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    # Each tile sees m_input rows x k_chunk columns of A.
    l1MemrefTyA = MemRefType.get(
        shape=[m_input, k_chunk],
        element_type=xrt_dtype_in,
        memory_space=l1_mem_space,
    )
    # Full B vector slice for this tile's k_chunk.
    l1MemrefTyB = MemRefType.get(
        shape=[k_chunk],
        element_type=xrt_dtype_in,
        memory_space=l1_mem_space,
    )
    # Output buffer (only used by last tile ty==0).
    l1MemrefTyC = MemRefType.get(
        shape=[tile_m],
        element_type=xrt_dtype_out,
        memory_space=l1_mem_space,
    )
    # Cascade bus width on AIE2P is 512 bits = 16 floats.
    # The cascade buffer must be padded to this width.
    CASCADE_WIDTH = 16  # 512 bits / 32 bits per float
    cascade_buf_len = max(tile_m, CASCADE_WIDTH)
    # Round up to multiple of CASCADE_WIDTH for multi-beat transfers.
    cascade_buf_len = (
        (cascade_buf_len + CASCADE_WIDTH - 1) // CASCADE_WIDTH
    ) * CASCADE_WIDTH

    # Float partial sum buffers (used for cascade data transfer and kernel I/O).
    # scratch: kernel output for first/middle tiles; cascade put source
    # recv: cascade get destination; kernel input for middle/last tiles
    # Sized to cascade_buf_len (padded to 512-bit cascade width).
    l1MemrefTyScratch = MemRefType.get(
        shape=[cascade_buf_len],
        element_type=f32_type,
        memory_space=l1_mem_space,
    )

    # Cascade channel: per-column cascade links.
    # n_cascade tiles per column → n_cascade-1 links per column.
    channel("chan_cascade", size=[herd_cols, n_cascade - 1], channel_type="cascade")

    # External kernel declarations.
    # First tile: compute partial sums → scratch
    matvec_first_func = FuncOp(
        "matvec_cascade_first_bf16",
        (
            [T.i32(), T.i32(), l1MemrefTyA, l1MemrefTyB, l1MemrefTyScratch],
            [],
        ),
        visibility="private",
    )
    # Middle tile: recv + own partial → scratch
    matvec_middle_func = FuncOp(
        "matvec_cascade_middle_bf16",
        (
            [
                T.i32(),
                T.i32(),
                l1MemrefTyA,
                l1MemrefTyB,
                l1MemrefTyScratch,  # recv (incoming partial sums)
                l1MemrefTyScratch,  # scratch (output partial sums)
            ],
            [],
        ),
        visibility="private",
    )
    # Last tile: recv + own partial → bf16 output
    matvec_last_func = FuncOp(
        "matvec_cascade_last_bf16",
        (
            [
                T.i32(),
                T.i32(),
                T.i32(),
                l1MemrefTyA,
                l1MemrefTyB,
                l1MemrefTyScratch,  # recv (incoming partial sums)
                l1MemrefTyC,
            ],
            [],
        ),
        visibility="private",
    )
    linalg_fill_func = FuncOp(
        "linalg_fill_bf16",
        ([xrt_dtype_out, l1MemrefTyC], []),
        visibility="private",
    )
    for func in [
        matvec_first_func,
        matvec_middle_func,
        matvec_last_func,
        linalg_fill_func,
    ]:
        func.attributes["link_with"] = StringAttr.get("mv_cascade.o")
        func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    @FuncOp.from_py_func(memrefTyA, memrefTyB, memrefTyC)
    def matvec_cascade_bf16(arg0, arg1, arg2):

        # Each launch handles herd_cols * tile_m output rows.
        launch_size = [m // tile_m // herd_cols, 1]

        @launch(operands=[arg0, arg1, arg2], sizes=launch_size)
        def launch_body(
            launch_ivx,
            launch_ivy,
            launch_sizex,
            launch_sizey,
            l3_a_data,
            l3_b_data,
            l3_c_data,
        ):

            @segment(
                name="matvec_cascade_0",
                operands=[launch_ivx, l3_a_data, l3_b_data, l3_c_data],
            )
            def segment_body(
                launch_ivx_s,
                l3_a_data_s,
                l3_b_data_s,
                l3_c_data_s,
            ):
                # Row offset = launch_ivx * tile_m * herd_cols
                launch_ivx_map = AffineMap.get(
                    0,
                    1,
                    [
                        AffineExpr.get_mul(
                            AffineSymbolExpr.get(0),
                            AffineConstantExpr.get(tile_m * herd_cols),
                        )
                    ],
                )
                launch_offset_m = affine_apply(launch_ivx_map, [launch_ivx_s])

                # Alloc L2 buffers
                l2_a_data = AllocOp(l2MemrefTyA, [], [])
                l2_c_data = AllocOp(l2MemrefTyC, [], [])

                # L3→L2: A (all herd_cols * tile_m rows, full K)
                dma_memcpy_nd(
                    l2_a_data,
                    l3_a_data_s,
                    src_offsets=[0, launch_offset_m, 0],
                    src_sizes=[herd_cols, tile_m, k],
                    src_strides=[tile_m * k, k, 1],
                )

                # Alloc L1 buffers
                l1_a_data = AllocOp(l1MemrefTyA, [], [])
                l1_b_data = AllocOp(l1MemrefTyB, [], [])
                l1_c_data = AllocOp(l1MemrefTyC, [], [])
                l1_scratch = AllocOp(l1MemrefTyScratch, [], [])
                l1_recv = AllocOp(l1MemrefTyScratch, [], [])

                @herd(
                    name="herd_0",
                    sizes=[herd_cols, n_cascade],
                    operands=[
                        l1_a_data,
                        l1_b_data,
                        l1_c_data,
                        l1_scratch,
                        l1_recv,
                        l2_a_data,
                        l3_b_data_s,
                        l2_c_data,
                    ],
                )
                def herd_body(
                    tx,
                    ty,
                    sx,
                    sy,
                    _l1_a,
                    _l1_b,
                    _l1_c,
                    _l1_scratch,
                    _l1_recv,
                    _l2_a,
                    _l3_b,
                    _l2_c,
                ):
                    c0 = arith.ConstantOp.create_index(0)
                    c1_idx = arith.ConstantOp.create_index(1)
                    last_ty = arith.ConstantOp.create_index(n_cascade - 1)

                    m_const = ConstantOp(IntegerAttr.get(T.i32(), m_input), None)
                    k_chunk_const = ConstantOp(IntegerAttr.get(T.i32(), k_chunk), None)

                    # k_offset = ty * k_chunk
                    ty_k_map = AffineMap.get(
                        0,
                        1,
                        [
                            AffineExpr.get_mul(
                                AffineSymbolExpr.get(0),
                                AffineConstantExpr.get(k_chunk),
                            )
                        ],
                    )
                    k_offset = affine_apply(ty_k_map, [ty])

                    # L3→L1: B[k_offset:k_offset+k_chunk] directly
                    dma_memcpy_nd(
                        _l1_b,
                        _l3_b,
                        src_offsets=[k_offset],
                        src_sizes=[k_chunk],
                        src_strides=[1],
                    )

                    # Zero-fill output buffer (only used by last tile)
                    cmp_is_last = arith.CmpIOp(arith.CmpIPredicate.eq, ty, c0)
                    if_zero_fill = scf.IfOp(cmp_is_last)
                    with InsertionPoint(if_zero_fill.then_block):
                        zero = ConstantOp(FloatAttr.get(xrt_dtype_out, 0), None)
                        CallOp(linalg_fill_func, [zero, _l1_c])
                        yield_([])

                    # Loop over m_input rows at a time
                    for j_m in range_(0, tile_m // m_input):
                        j_m_map = AffineMap.get(
                            0,
                            1,
                            [
                                AffineExpr.get_mul(
                                    AffineSymbolExpr.get(0),
                                    AffineConstantExpr.get(m_input),
                                )
                            ],
                        )
                        j_m_offset = affine_apply(j_m_map, [j_m])

                        # L2→L1: A[tx, j_m_offset:j_m_offset+m_input, k_offset:k_offset+k_chunk]
                        dma_memcpy_nd(
                            _l1_a,
                            _l2_a,
                            src_offsets=[tx, j_m_offset, k_offset],
                            src_sizes=[1, m_input, k_chunk],
                            src_strides=[tile_m * k, k, 1],
                        )

                        # === Cascade pipeline logic ===

                        # First tile (ty == n_cascade-1): compute → cascade put
                        cmp_first = arith.CmpIOp(arith.CmpIPredicate.eq, ty, last_ty)
                        if_first = scf.IfOp(cmp_first, has_else=True)
                        with InsertionPoint(if_first.then_block):
                            CallOp(
                                matvec_first_func,
                                [m_const, k_chunk_const, _l1_a, _l1_b, _l1_scratch],
                            )
                            # Send scratch via cascade channel.
                            prev_ty = arith.SubIOp(ty, c1_idx)
                            ChannelPut(
                                "chan_cascade", _l1_scratch, indices=[tx, prev_ty]
                            )
                            yield_([])

                        with InsertionPoint(if_first.else_block):
                            # Last tile (ty == 0): cascade get → compute → write
                            cmp_last = arith.CmpIOp(arith.CmpIPredicate.eq, ty, c0)
                            if_last = scf.IfOp(cmp_last, has_else=True)
                            with InsertionPoint(if_last.then_block):
                                ChannelGet("chan_cascade", _l1_recv, indices=[tx, ty])
                                row_offset_i32 = arith.index_cast(T.i32(), j_m_offset)
                                CallOp(
                                    matvec_last_func,
                                    [
                                        m_const,
                                        k_chunk_const,
                                        row_offset_i32,
                                        _l1_a,
                                        _l1_b,
                                        _l1_recv,
                                        _l1_c,
                                    ],
                                )
                                yield_([])

                            with InsertionPoint(if_last.else_block):
                                # Middle tiles: cascade get → compute → cascade put
                                ChannelGet("chan_cascade", _l1_recv, indices=[tx, ty])
                                CallOp(
                                    matvec_middle_func,
                                    [
                                        m_const,
                                        k_chunk_const,
                                        _l1_a,
                                        _l1_b,
                                        _l1_recv,
                                        _l1_scratch,
                                    ],
                                )
                                prev_ty_mid = arith.SubIOp(ty, c1_idx)
                                ChannelPut(
                                    "chan_cascade",
                                    _l1_scratch,
                                    indices=[tx, prev_ty_mid],
                                )
                                yield_([])

                            yield_([])

                        yield_([])

                    # Only ty==0 tiles write results back: L1→L2
                    cmp_writer = arith.CmpIOp(arith.CmpIPredicate.eq, ty, c0)
                    if_writer = scf.IfOp(cmp_writer)
                    with InsertionPoint(if_writer.then_block):
                        dma_memcpy_nd(
                            _l2_c,
                            _l1_c,
                            dst_offsets=[tx, 0],
                            dst_sizes=[1, tile_m],
                            dst_strides=[tile_m, 1],
                            src_offsets=[],
                            src_sizes=[tile_m],
                            src_strides=[1],
                        )
                        yield_([])

                herd_body.attributes["link_with"] = StringAttr.get("mv_cascade.o")

                # L2→L3: C
                dma_memcpy_nd(
                    l3_c_data_s,
                    l2_c_data,
                    dst_offsets=[launch_offset_m],
                    dst_sizes=[herd_cols * tile_m],
                    dst_strides=[1],
                    src_offsets=[0, 0],
                    src_sizes=[herd_cols, tile_m],
                    src_strides=[tile_m, 1],
                )

                DeallocOp(l2_a_data)
                DeallocOp(l2_c_data)
                DeallocOp(l1_a_data)
                DeallocOp(l1_b_data)
                DeallocOp(l1_c_data)
                DeallocOp(l1_scratch)
                DeallocOp(l1_recv)


if __name__ == "__main__":
    # Default values for NPU2 8-column x 4-row cascade
    # tile_m=2 with 8 cols and K=8192: A_L2 = 8*2*8192*2 = 256KB < 512KB ✓
    M = 2048
    K = 8192
    TILE_M = 2
    M_INPUT = 1
    HERD_COLS = 8
    N_CASCADE = 4
    INPUT_DATATYPE = bfloat16
    OUTPUT_DATATYPE = bfloat16

    parser = argparse.ArgumentParser(
        prog="matvec_cascade.py",
        description="Builds, runs, and tests the cascade bf16 matrix-vector multiplication (GEMV) example",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument(
        "--m", type=int, default=M, help="M dimension (matrix rows / output size)"
    )
    parser.add_argument(
        "--k", type=int, default=K, help="K dimension (matrix columns / vector length)"
    )
    parser.add_argument(
        "--tile-m",
        type=int,
        default=TILE_M,
        dest="tile_m",
        help="Number of output rows per tile per column",
    )
    parser.add_argument(
        "--m-input",
        type=int,
        default=M_INPUT,
        help="Number of matrix rows per kernel call",
    )
    parser.add_argument(
        "--herd-cols",
        type=int,
        default=HERD_COLS,
        dest="herd_cols",
        help="Number of AIE columns (parallel compute tiles along M dimension)",
    )
    parser.add_argument(
        "--n-cascade",
        type=int,
        default=N_CASCADE,
        dest="n_cascade",
        help="Number of cascade tiles per column (K-reduction depth)",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="xclbin",
        dest="output_format",
        help="Output format for the compiled binary (default: xclbin)",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["compile-and-run", "compile-and-xclbin"],
        dest="compile_mode",
        default="compile-and-run",
        help="compile-and-run (default): compile and validate; compile-and-xclbin: generate xclbin only",
    )
    parser.add_argument(
        "--debug-ir",
        action="store_true",
        dest="debug_ir",
        help="Emit IR after each pass into debug_ir/ directory",
    )

    args = parser.parse_args()

    mlir_module = build_module(
        args.m,
        args.k,
        args.tile_m,
        args.m_input,
        args.herd_cols,
        args.n_cascade,
        INPUT_DATATYPE,
        OUTPUT_DATATYPE,
    )
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    if args.compile_mode == "compile-and-run":
        np.random.seed(42)
        input_a = (np.random.randn(args.m, args.k) * 4).astype(INPUT_DATATYPE)
        input_b = (np.random.randn(args.k) * 4).astype(INPUT_DATATYPE)
        output_c = np.dot(
            input_a.astype(np.float32), input_b.astype(np.float32)
        ).astype(OUTPUT_DATATYPE)

        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            omit_pingpong=True,
            runtime_loop_tiling_sizes=[4, 4],
            output_format=args.output_format,
            instance_name="matvec_cascade_bf16",
            debug_ir=args.debug_ir,
            use_lock_race_condition_fix=True,
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_a, input_b],
                expected_outputs=[output_c],
                rtol=0.04,
                atol=1e-3,
            )
        )

    elif args.compile_mode == "compile-and-xclbin":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            omit_pingpong=True,
            runtime_loop_tiling_sizes=[4, 4],
            use_lock_race_condition_fix=True,
        )
        module_function = backend.compile(mlir_module)
        backend.unload()
