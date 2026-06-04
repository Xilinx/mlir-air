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

Two codegen paths share the same DMA / packed L1 layout / output shape:

  * Default: the inner kernel body is a CallOp to a hand-written C++
    kernel (dequant.cc, compiled to dequant.o) that the linker resolves.

  * --direct-codegen (AIE2P only): the inner body is authored as standard
    arith/vector/memref ops and lowered through mlir-aie's VectorToAIEVec
    + AIEVecToLLVM pipeline to the AIE2P unpack.I512.I8.I4 intrinsic, the
    magic-number sitofp i16->bf16 sequence, and native bf16 mul. No
    external .o is needed. The inner subgroup processes R=64 nibbles per
    iteration (the natural width of llvm.aie2p.unpack.I512.I8.I4), so
    group_size must be a multiple of 64 in this mode.
"""

import argparse
import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects import arith
from air.dialects.memref import AllocOp, DeallocOp, load as memref_load
from air.dialects.vector import (
    transfer_read,
    transfer_write,
    BroadcastOp,
    bitcast as v_bitcast,
)
from air.dialects.func import FuncOp, CallOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend

range_ = for_

# Inner subgroup width (nibbles per iteration) for --direct-codegen.
# Must match the byte-packed source size of llvm.aie2p.unpack.I512.I8.I4.
R_SUB = 64


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


def _emit_inline_dequant_body(
    l1_packed, l1_out, n_tile, group_size, q_bytes, s_bytes, ng_tile, nsub_per_group
):
    """Per-tile int4 -> bf16 dequant authored as standard MLIR ops.

    Reads Q+S+Z out of l1_packed and writes n_tile bf16 results into l1_out.
    Lowerings (all in upstream mlir-aie):
      * vector.transfer_read + vector.bitcast + arith.extui (int4 path)
        -> aievec.unpack -> llvm.aie2p.unpack.I512.I8.I4
      * arith.extsi (i8 -> i16) + arith.sitofp (i16 -> bf16) -> magic-number
        aievec sequence (UPS + acc-add + cast + sub + SRS)
      * arith.subi (i8) and arith.mulf (bf16, native v32/v64) lower via
        existing aievec patterns.
    """
    i8_type = IntegerType.get_signless(8)
    i16_ty = IntegerType.get_signless(16)
    bf16_type = type_mapper(bfloat16)

    vec_i8 = VectorType.get([R_SUB // 2], i8_type)
    vec_i4 = VectorType.get([R_SUB], IntegerType.get_signless(4))
    vec_i8_unpacked = VectorType.get([R_SUB], i8_type)
    vec_bf16 = VectorType.get([R_SUB], bf16_type)
    id_map = AffineMapAttr.get(AffineMap.get_identity(1))
    c0_i8 = arith.ConstantOp(i8_type, 0)

    # Unroll the per-group loop in Python so each group's z offset and
    # scale offset stay constants. Wrapping it in an scf.for triggers loop
    # strength reduction that rewrites the loop IV in element units and
    # then reuses the same IV for the z/s offsets, silently corrupting
    # them.
    for g_py in range(ng_tile):
        # Scalar zero point: byte at q_bytes + s_bytes + g_py.
        z_off = arith.ConstantOp.create_index(q_bytes + s_bytes + g_py)
        z_byte = memref_load(l1_packed, [z_off])
        zv = BroadcastOp(vec_i8_unpacked, z_byte)

        # Scalar bf16 scale: 2 bytes at q_bytes + g_py*2, reassembled as
        # i16 then bitcast to bf16.
        s_off_lo = arith.ConstantOp.create_index(q_bytes + g_py * 2)
        s_off_hi = arith.ConstantOp.create_index(q_bytes + g_py * 2 + 1)
        s_lo_i8 = memref_load(l1_packed, [s_off_lo])
        s_hi_i8 = memref_load(l1_packed, [s_off_hi])
        s_lo_i16 = arith.extui(i16_ty, s_lo_i8)
        s_hi_i16 = arith.extui(i16_ty, s_hi_i8)
        s_hi_shifted = arith.shli(s_hi_i16, arith.ConstantOp(i16_ty, 8))
        s_bits = arith.ori(s_lo_i16, s_hi_shifted)
        sv = arith.bitcast(bf16_type, s_bits)
        sv_vec = BroadcastOp(vec_bf16, sv)

        # Inner subgroup loop stays as scf.for since its IV only feeds
        # element offsets (no scalar metadata loads keyed on it).
        c_rs = arith.ConstantOp.create_index(R_SUB)
        g_base_elems = arith.ConstantOp.create_index(g_py * group_size)
        for i in range_(0, nsub_per_group):
            sub_off_elems = arith.addi(
                g_base_elems,
                arith.muli(i, c_rs),
            )
            sub_off_bytes = arith.divui(
                sub_off_elems,
                arith.ConstantOp.create_index(2),
            )

            pk_i8 = transfer_read(
                vec_i8, l1_packed, [sub_off_bytes], id_map, c0_i8, [True]
            )
            # Canonical int4 unpack: bitcast to i4 vec then zero-extend to
            # i8. The mlir-aie LowerExtUIOfBitcastI4ToUnpackPattern rewrites
            # to aievec.unpack on the byte-packed source.
            pk_i4 = v_bitcast(vec_i4, pk_i8)
            w_i8 = arith.extui(vec_i8_unpacked, pk_i4)

            wmz_i8 = arith.subi(w_i8, zv)

            # int8 -> int16 -> bf16. LowerVectorSIToFPI16BF16AIE2pPattern
            # picks up the i16 -> bf16 sitofp at v16/v32 widths and lowers
            # via the magic-number trick (UPS + accfloat add/sub + SRS).
            wmz_i16 = arith.extsi(VectorType.get([R_SUB], i16_ty), wmz_i8)
            wmz_bf16 = arith.sitofp(vec_bf16, wmz_i16)

            out_bf16 = arith.mulf(wmz_bf16, sv_vec)

            transfer_write(None, out_bf16, l1_out, [sub_off_elems], id_map, [True])
            yield_([])


@module_builder
def build_module(n, group_size, herd_n, direct_codegen=False):
    bf16_type = type_mapper(bfloat16)
    u8_type = IntegerType.get_signless(8)

    assert n % herd_n == 0, "n must be divisible by herd_n"
    n_tile = n // herd_n
    assert n_tile % group_size == 0, "n_tile must be divisible by group_size"
    if direct_codegen:
        assert (
            group_size % R_SUB == 0
        ), f"--direct-codegen requires group_size multiple of {R_SUB}"
    q_bytes, s_bytes, z_bytes, tile_bytes = packed_tile_bytes(n_tile, group_size)
    ng_tile = n_tile // group_size
    nsub_per_group = group_size // R_SUB if direct_codegen else 0

    # L3 types: packed weights+scales+zeros laid out per-tile, dequantized output
    l3_packed_ty = MemRefType.get([herd_n, tile_bytes], u8_type)
    l3_out_ty = MemRefType.get([n], bf16_type)

    # L1 types
    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_packed_ty = MemRefType.get([tile_bytes], u8_type, memory_space=l1_space)
    l1_out_ty = MemRefType.get([n_tile], bf16_type, memory_space=l1_space)

    # External kernel decl (only declared in the extern path; aircc only
    # links dequant.o when a private @dequant_int4_bf16 FuncOp is present
    # with `link_with = "dequant.o"`).
    dequant_func = None
    if not direct_codegen:
        dequant_func = FuncOp(
            "dequant_int4_bf16",
            ([l1_packed_ty, l1_out_ty], []),
            visibility="private",
        )
        dequant_func.attributes["link_with"] = StringAttr.get("dequant.o")
        dequant_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    herd_kwargs = {"name": "dequant_herd", "sizes": [1, herd_n]}
    if not direct_codegen:
        herd_kwargs["link_with"] = "dequant.o"

    @FuncOp.from_py_func(l3_packed_ty, l3_out_ty)
    def dequant(arg_packed, arg_out):
        @launch(operands=[arg_packed, arg_out])
        def launch_body(l_packed, l_out):
            @segment(name="seg", operands=[l_packed, l_out])
            def segment_body(s_packed, s_out):
                @herd(operands=[s_packed, s_out], **herd_kwargs)
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

                    if direct_codegen:
                        _emit_inline_dequant_body(
                            l1_packed,
                            l1_out,
                            n_tile,
                            group_size,
                            q_bytes,
                            s_bytes,
                            ng_tile,
                            nsub_per_group,
                        )
                    else:
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
        prog="dequant_awq.py",
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
        "--device",
        type=str,
        choices=["npu1", "npu2"],
        default=None,
        dest="device",
        help=(
            "Target NPU device. npu1 = Phoenix (AIE2), npu2 = Strix (AIE2P). "
            "If unset, the XRT backend auto-detects via xrt-smi."
        ),
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
    parser.add_argument(
        "--direct-codegen",
        action="store_true",
        dest="direct_codegen",
        help=(
            "Emit the per-tile dequant body as inline standard MLIR ops "
            "(arith/vector/memref) instead of a CallOp to the hand-written "
            "dequant.o kernel. AIE2P only; requires group_size multiple of "
            f"{R_SUB}."
        ),
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
    if args.direct_codegen:
        if args.group_size % R_SUB != 0:
            parser.error(
                f"--direct-codegen requires group_size multiple of {R_SUB} "
                f"(inline subgroup width)"
            )
        if args.device == "npu1":
            parser.error("--direct-codegen is AIE2P only (no npu1 support)")
    else:
        # The hand-written kernel's inner loop processes 32 nibbles per
        # iteration (see GROUP_SIZE static_assert in dequant.cc). Catch
        # the mismatch here with a clear message instead of failing at
        # C++ compile time.
        if args.group_size % 32 != 0:
            parser.error(
                "group_size must be a multiple of 32 (kernel inner vector width)"
            )
    if args.n % args.group_size != 0:
        parser.error("N must be divisible by group_size")
    if args.n % args.herd_n != 0:
        parser.error("N must be divisible by herd_n")
    if (args.n // args.herd_n) % args.group_size != 0:
        parser.error("N / herd_n must be divisible by group_size")
    if args.device == "npu1" and args.output_format == "elf":
        parser.error("--output-format=elf is not supported on npu1; use xclbin")

    mlir_module = build_module(
        args.n, args.group_size, args.herd_n, direct_codegen=args.direct_codegen
    )
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

    instance_name = "dequant_inline" if args.direct_codegen else "dequant"

    if args.compile_mode == "compile-and-run":
        runner = XRTRunner(
            verbose=args.verbose,
            omit_pingpong=True,
            output_format=args.output_format,
            instance_name=instance_name,
            target_device=args.device,
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
            target_device=args.device,
        )
        module_function = backend.compile(mlir_module)
        backend.unload()
