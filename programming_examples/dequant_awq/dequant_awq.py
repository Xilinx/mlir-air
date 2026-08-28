# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""AWQ-style int4 to bfloat16 dequantization on air.api.

    output[i] = (int4_weight[i] - zero_point[group]) * scale[group]

Q, S and Z are concatenated into one packed L1 buffer per tile -- the layout
``matrix_vector_multiplication/int4_awq`` and ``matrix_multiplication/int4_awq``
use in production -- which keeps each compute tile inside its DMA channel budget
while exposing all three pieces of metadata to one vectorised inner loop. A
1 x HERD_N herd splits N across tiles.

Two codegen paths share that layout, the DMAs and the output shape:

* default: the per-tile body is a call into a hand-written kernel
  (``dequant.cc`` -> ``dequant.o``).

* ``--direct-codegen`` (AIE2P only): the body is written in the DSL and lowered
  through mlir-aie's VectorToAIEVec pipeline to the AIE2P
  ``unpack.I512.I8.I4`` intrinsic, the magic-number sitofp i16->bf16 sequence
  and a native bf16 multiply. No object file.

The interesting line is the unpack::

    ops.cast(ops.bitcast(packed[b : b + R // 2], i4), i8, signed=False)

Both halves of that have to be spelled exactly so. ``ops.bitcast`` reinterprets
32 bytes as 64 half-bytes -- it preserves the representation where ``ops.cast``
preserves the value -- and the widening is ``signed=False`` because these are
quantised magnitudes 0..15, where ``extsi`` would read 0x9 as -7. mlir-aie's
``LowerExtUIOfBitcastI4ToUnpackPattern`` matches that exact pair and rewrites it
into a single ``aievec.unpack``; masking and shifting by hand computes the same
numbers and misses the instruction entirely.

Two departures from the predecessor's IR.

The per-group scale goes through a two-byte L1 buffer. It is assembled from two
bytes with a shift and an or and is a bf16 only once reinterpreted, and the
predecessor keeps it in a register the whole way. In the DSL a value broadcast
across a vector is read at the leaf, so writing the assembly inline would widen
every step of it to 64 lanes; a one-element destination takes the scalar path
instead and produces the predecessor's scalar ops, at the cost of one store and
one load per group.

Offsets are computed in bytes directly -- ``g * (group_size // 2)`` -- rather
than in elements and then halved, so there is no ``arith.divui``. Both are exact
because the group size is even.
"""

import argparse

import numpy as np
from ml_dtypes import bfloat16

from air import api as air
from air.api import ops
from air.api.types import bf16, i4, i8, i16
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner

# Nibbles per inner iteration. Fixed by the byte-packed source width of
# llvm.aie2p.unpack.I512.I8.I4.
R_SUB = 64


def packed_tile_bytes(n_tile, group_size):
    n_groups_tile = n_tile // group_size
    q_bytes = n_tile // 2
    s_bytes = 2 * n_groups_tile
    z_bytes = n_groups_tile
    raw = q_bytes + s_bytes + z_bytes
    # aie.dma_bd needs a transfer length that is a multiple of 4 bytes. The
    # kernel only reads [0, raw); the pad bytes are unused.
    tile_bytes = (raw + 3) & ~3
    return q_bytes, s_bytes, z_bytes, tile_bytes


def build_module(n, group_size, herd_n, direct_codegen=False):
    assert n % herd_n == 0, "n must be divisible by herd_n"
    n_tile = n // herd_n
    assert n_tile % group_size == 0, "n_tile must be divisible by group_size"
    if direct_codegen:
        assert (
            group_size % R_SUB == 0
        ), f"--direct-codegen requires group_size multiple of {R_SUB}"
    q_bytes, s_bytes, _z_bytes, tile_bytes = packed_tile_bytes(n_tile, group_size)
    ng_tile = n_tile // group_size
    nsub_per_group = group_size // R_SUB if direct_codegen else 0

    l3_packed = air.tensor([herd_n, tile_bytes], i8)
    l3_out = air.tensor([n], bf16)

    kernel = (
        None
        if direct_codegen
        else air.extern("dequant_int4_bf16", link_with="dequant.o")
    )

    with air.launch(name="dequant") as launch:

        @launch.body
        def _():

            with air.segment(name="seg") as seg:

                @seg.body
                def _():

                    with air.herd(
                        [range(1), range(herd_n)],
                        name="dequant_herd",
                        shape=(1, herd_n),
                    ) as h:

                        @h.body
                        def _(tx, ty):
                            packed = air.alloc([tile_bytes], i8, scope=h.private())
                            out = air.alloc(
                                [n_tile], bf16, scope=h.private(), vector=R_SUB
                            )

                            # Each tile pulls one row of the packed buffer.
                            ops.load(packed, l3_packed[ty, :])

                            if direct_codegen:
                                scale = air.alloc([1], bf16, scope=h.private())
                                # Unrolled in Python so each group's metadata
                                # offsets stay constants. Inside an scf.for,
                                # loop strength reduction rewrites the
                                # induction variable in element units and then
                                # reuses it for the byte offsets, silently
                                # corrupting them.
                                for g in range(ng_tile):
                                    z = q_bytes + s_bytes + g
                                    lo, hi = q_bytes + g * 2, q_bytes + g * 2 + 1
                                    # Two bytes, little end first, reinterpreted
                                    # as the bf16 they spell.
                                    scale[0:1] = ops.bitcast(
                                        ops.cast(packed[lo : lo + 1], i16, signed=False)
                                        | (
                                            ops.cast(
                                                packed[hi : hi + 1], i16, signed=False
                                            )
                                            << 8
                                        ),
                                        bf16,
                                    )
                                    for i in air.sequential(0, nsub_per_group):
                                        e = g * group_size + i * R_SUB
                                        b = g * (group_size // 2) + i * (R_SUB // 2)
                                        nibbles = ops.cast(
                                            ops.bitcast(packed[b : b + R_SUB // 2], i4),
                                            i8,
                                            signed=False,
                                        )
                                        out[e : e + R_SUB] = (
                                            ops.cast(
                                                ops.cast(
                                                    nibbles - packed[z : z + 1], i16
                                                ),
                                                bf16,
                                            )
                                            * scale[0:1]
                                        )
                            else:
                                kernel(packed, out)

                            # Each tile writes a contiguous output slice.
                            ops.store(out, l3_out[ty * n_tile : ty * n_tile + n_tile])

    return launch


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


def parse_args():
    parser = argparse.ArgumentParser(
        prog="dequant_awq.py",
        description="AWQ-style int4 to bf16 dequantization example",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--n", type=int, default=1024, help="Number of elements")
    parser.add_argument(
        "--group-size", type=int, default=128, help="Quantization group size"
    )
    parser.add_argument(
        "--herd-n",
        type=int,
        default=4,
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
            "Emit the per-tile dequant body in the DSL instead of calling the "
            f"hand-written dequant.o kernel. AIE2P only; requires group_size "
            f"multiple of {R_SUB}."
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
        # iteration (see the GROUP_SIZE static_assert in dequant.cc). Catch the
        # mismatch here instead of at C++ compile time.
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
    return args


def main():
    args = parse_args()

    launch = build_module(
        args.n, args.group_size, args.herd_n, direct_codegen=args.direct_codegen
    )
    mlir_module = launch.build(target=args.device or "auto")
    if args.print_module_only:
        print(mlir_module)
        return 0

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

    if args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_pingpong=True,
            output_format=args.output_format,
            target_device=launch.target,
        )
        backend.compile(mlir_module)
        backend.unload()
        return 0

    # The ELF kernel resolves as main:<instance_name>, which must match @dequant.
    runner = XRTRunner(
        verbose=args.verbose,
        omit_pingpong=True,
        output_format=args.output_format,
        instance_name="dequant",
        target_device=launch.target,
    )
    return runner.run_test(
        mlir_module,
        inputs=[packed],
        expected_outputs=[ref_output],
        rtol=1e-1,
        atol=5e-2,
    )


if __name__ == "__main__":
    exit(main())
