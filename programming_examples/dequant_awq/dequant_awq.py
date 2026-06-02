# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AWQ-style int4 to bfloat16 dequantization example.

Dequantizes int4 weights packed in uint8 pairs using per-group
scale (bf16) and zero-point (uint8) parameters:
  output[i] = (int4_weight[i] - zero_point[group]) * scale[group]

Q, S, and Z are concatenated into a single packed L1 BO per tile
(matches the production layout used by matrix_vector_multiplication/int4_awq
and matrix_multiplication/int4_awq), keeping each compute tile within its
2 S2MM + 2 MM2S channel budget while exposing all three pieces of metadata
to a fully vectorized inner loop in dequant.cc.

Uses a 1xHERD_N AIE herd splitting N across compute tiles.
"""

import argparse
import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.func import FuncOp, CallOp
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend


def packed_tile_bytes(n_tile, group_size):
    n_groups_tile = n_tile // group_size
    q_bytes = n_tile // 2
    s_bytes = 2 * n_groups_tile
    z_bytes = n_groups_tile
    raw = q_bytes + s_bytes + z_bytes
    # Pad each tile's L3 row to a 4-byte boundary: aie.dma_bd requires the
    # transfer length to be a multiple of 4 bytes. The kernel only reads
    # [0, raw); the pad bytes are unused.
    tile_bytes = (raw + 3) & ~3
    return q_bytes, s_bytes, z_bytes, tile_bytes


@module_builder
def build_module(n, group_size, herd_n):
    bf16_type = type_mapper(bfloat16)
    u8_type = IntegerType.get_signless(8)

    assert n % herd_n == 0, "n must be divisible by herd_n"
    n_tile = n // herd_n
    assert n_tile % group_size == 0, "n_tile must be divisible by group_size"
    _, _, _, tile_bytes = packed_tile_bytes(n_tile, group_size)

    # L3 types: packed weights+scales+zeros laid out per-tile, dequantized output
    l3_packed_ty = MemRefType.get([herd_n, tile_bytes], u8_type)
    l3_out_ty = MemRefType.get([n], bf16_type)

    # L1 types
    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_packed_ty = MemRefType.get([tile_bytes], u8_type, memory_space=l1_space)
    l1_out_ty = MemRefType.get([n_tile], bf16_type, memory_space=l1_space)

    # External kernel
    dequant_func = FuncOp(
        "dequant_int4_bf16",
        ([l1_packed_ty, l1_out_ty], []),
        visibility="private",
    )
    dequant_func.attributes["link_with"] = StringAttr.get("dequant.o")
    dequant_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    @FuncOp.from_py_func(l3_packed_ty, l3_out_ty)
    def dequant(arg_packed, arg_out):
        @launch(operands=[arg_packed, arg_out])
        def launch_body(l_packed, l_out):
            @segment(name="seg", operands=[l_packed, l_out])
            def segment_body(s_packed, s_out):
                @herd(
                    name="dequant_herd",
                    sizes=[1, herd_n],
                    operands=[s_packed, s_out],
                    link_with="dequant.o",
                )
                def herd_body(_tx, _ty, _sx, _sy, h_packed, h_out):
                    l1_packed = AllocOp(l1_packed_ty, [], [])
                    l1_out = AllocOp(l1_out_ty, [], [])

                    # Each tile pulls one row [_ty, :] of the packed BO.
                    dma_memcpy_nd(
                        l1_packed,
                        h_packed,
                        src_offsets=[_ty, 0],
                        src_sizes=[1, tile_bytes],
                        src_strides=[tile_bytes, 1],
                    )

                    CallOp(dequant_func, [l1_packed, l1_out])

                    # Each tile writes a contiguous output slice
                    # [_ty * n_tile : (_ty + 1) * n_tile].
                    ty_to_off = AffineMap.get(
                        0,
                        1,
                        [
                            AffineExpr.get_mul(
                                AffineSymbolExpr.get(0),
                                AffineConstantExpr.get(n_tile),
                            )
                        ],
                    )
                    out_off = affine_apply(ty_to_off, [_ty])
                    dma_memcpy_nd(
                        h_out,
                        l1_out,
                        dst_offsets=[out_off],
                        dst_sizes=[n_tile],
                        dst_strides=[1],
                    )

                    DeallocOp(l1_packed)
                    DeallocOp(l1_out)


def pack_inputs(int4_vals, scales, zeros, n, group_size, herd_n):
    """Pack Q + S + Z per tile into [herd_n, tile_bytes] uint8."""
    n_tile = n // herd_n
    ng_tile = n_tile // group_size
    q_bytes, s_bytes, z_bytes, tile_bytes = packed_tile_bytes(n_tile, group_size)

    packed_q = (int4_vals[0::2] | (int4_vals[1::2] << 4)).astype(np.uint8)

    packed = np.zeros((herd_n, tile_bytes), dtype=np.uint8)
    for ty in range(herd_n):
        n_off = ty * n_tile
        g_off = ty * ng_tile
        q_tile = packed_q[n_off // 2 : (n_off + n_tile) // 2]
        s_tile = scales[g_off : g_off + ng_tile]
        z_tile = zeros[g_off : g_off + ng_tile]
        bo = packed[ty]
        bo[0:q_bytes] = q_tile
        bo[q_bytes : q_bytes + s_bytes] = s_tile.view(np.uint8)
        bo[q_bytes + s_bytes : q_bytes + s_bytes + z_bytes] = z_tile
    return packed


if __name__ == "__main__":
    N = 1024
    GROUP_SIZE = 128
    HERD_N = 4

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="AWQ-style int4 to bf16 dequantization example",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--n", type=int, default=N, help="Number of elements")
    parser.add_argument(
        "--group-size", type=int, default=GROUP_SIZE, help="Quantization group size"
    )
    parser.add_argument(
        "--herd-n",
        type=int,
        default=HERD_N,
        dest="herd_n",
        help="Number of compute tiles to split N across",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["compile-only", "compile-and-run"],
        dest="compile_mode",
        default="compile-and-run",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="xclbin",
        dest="output_format",
    )
    args = parser.parse_args()

    if args.n <= 0:
        parser.error("N must be positive")
    if args.group_size <= 0:
        parser.error("group_size must be positive")
    if args.herd_n <= 0:
        parser.error("herd_n must be positive")
    if args.n % 2 != 0:
        parser.error("N must be even (2 int4 values per byte)")
    # The vectorized kernel's inner loop processes 32 nibbles per iteration
    # (see GROUP_SIZE static_assert in dequant.cc). Catch the mismatch here
    # with a clear message instead of failing at C++ compile time.
    if args.group_size % 32 != 0:
        parser.error("group_size must be a multiple of 32 (kernel inner vector width)")
    if args.n % args.group_size != 0:
        parser.error("N must be divisible by group_size")
    if args.n % args.herd_n != 0:
        parser.error("N must be divisible by herd_n")
    if (args.n // args.herd_n) % args.group_size != 0:
        parser.error("N / herd_n must be divisible by group_size")

    mlir_module = build_module(args.n, args.group_size, args.herd_n)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    np.random.seed(0)
    n_groups = args.n // args.group_size

    int4_vals = np.random.randint(0, 16, args.n).astype(np.uint8)
    scales = np.random.uniform(0.01, 0.1, n_groups).astype(bfloat16)
    zeros = np.random.randint(7, 10, n_groups).astype(np.uint8)

    packed = pack_inputs(int4_vals, scales, zeros, args.n, args.group_size, args.herd_n)

    ref_output = np.zeros(args.n, dtype=bfloat16)
    for i in range(args.n):
        g = i // args.group_size
        ref_output[i] = bfloat16(
            (float(int4_vals[i]) - float(zeros[g])) * float(scales[g])
        )

    if args.compile_mode == "compile-and-run":
        runner = XRTRunner(
            verbose=args.verbose,
            omit_pingpong=True,
            output_format=args.output_format,
            instance_name="dequant",
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[packed],
                expected_outputs=[ref_output],
                rtol=1e-1,
                atol=5e-2,
            )
        )
    elif args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_pingpong=True,
            output_format=args.output_format,
        )
        module_function = backend.compile(mlir_module)
        backend.unload()
