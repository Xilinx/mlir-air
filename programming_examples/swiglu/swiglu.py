# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""SwiGLU (Swish-Gated Linear Unit) on air.api.

Element-wise, on 1-D inputs [N]:

    SwiGLU(x, gate, up) = SiLU(x * gate) * (x * up)

where SiLU(z) = z * sigmoid(z). `ops.silu` is built from the tanh identity,
z * 0.5 * (tanh(z/2) + 1), which is what the predecessor spelled by hand and
what the kernel wants: it avoids exp and division, both of which have precision
and correctness problems on AIE2P, and it lands on the hardware tanh intrinsic.

**The gate and up weights share one buffer**, packed as [2, N] with row 0 gate
and row 1 up, because an AIE2P tile has only 2 S2MM channels and x already
claims one. Reading a row of that buffer is what the whole example turns on:

    out[:] = ops.silu(x[:] * gate_up[0, :]) * (x[:] * gate_up[1, :])

A partial buffer subscript is normally a DMA access pattern -- the thing you
hand to ops.load -- and until now that was all it could be, so packing to save a
DMA channel cost you an unpack to read it. `gate_up[0, :]` is now also an
elementwise operand, spelled the way numpy spells it, and the two rows lower to
two `vector.transfer_read`s at different offsets into the same memref. The
predecessor built the same two reads out of `memref.subview` with a hand-added
`arith.addi(j, tile_n)` on the flat [2 * tile_n] copy.

Because the rows carry their leading axis, the buffers here are [1, tile_n]
rather than [tile_n]: [1, N] and [1, N] broadcast, [1, N] into [N] does not --
numpy's rule, and the same one air.api already applied to reductions. The L3
slices stay rank 1; ops.load and ops.store squeeze the leading unit dimension.
"""

import argparse

import numpy as np
from ml_dtypes import bfloat16

from air import api as air
from air.api import ops
from air.api.types import bf16
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner


def build_module(n, tile_n, dtype=bf16, vector=16):
    if n % tile_n:
        raise ValueError(f"n ({n}) must be a multiple of tile_n ({tile_n})")
    if vector and tile_n % vector:
        raise ValueError(
            f"tile_n ({tile_n}) must be a multiple of the vector width ({vector})"
        )

    x = air.tensor([n], dtype)
    # Row 0 is gate, row 1 is up: one DMA, one shim channel.
    gate_up = air.tensor([2, n], dtype)
    out = air.tensor([n], dtype)

    with air.launch(name="swiglu") as launch:

        @launch.body
        def _():
            with air.herd([range(1)], name="herd_0", shape=(1,)) as herd:

                @herd.body
                def _(tx):
                    # Rank 2 with a leading 1, so a row of l1_gate_up -- which
                    # is [1, tile_n] -- broadcasts against them.
                    l1_x = air.alloc(
                        [1, tile_n], dtype, scope=herd.private(), vector=vector
                    )
                    l1_gate_up = air.alloc(
                        [2, tile_n], dtype, scope=herd.private(), vector=vector
                    )
                    l1_out = air.alloc(
                        [1, tile_n], dtype, scope=herd.private(), vector=vector
                    )

                    for lo in air.sequential(0, n, tile_n):
                        ops.load(l1_x, x[lo : lo + tile_n])
                        ops.load(l1_gate_up, gate_up[:, lo : lo + tile_n])

                        l1_out[:] = ops.silu(l1_x[:] * l1_gate_up[0, :]) * (
                            l1_x[:] * l1_gate_up[1, :]
                        )

                        ops.store(l1_out, out[lo : lo + tile_n])

    return launch


if __name__ == "__main__":
    N = 65536
    TILE_N = 1024
    INPUT_DATATYPE = bfloat16

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the SwiGLU example",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--n", type=int, default=N, help="Total number of elements")
    parser.add_argument("--tile-n", type=int, default=TILE_N, help="Tile size")
    parser.add_argument(
        "--vector-size", type=int, default=16, help="Vector size for SIMD operations"
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
        "--target",
        type=str,
        default="auto",
        help="NPU generation to build for: auto (default, detects the installed "
        "device), npu1 or npu2",
    )
    args = parser.parse_args()

    launch = build_module(args.n, args.tile_n, bf16, args.vector_size)
    # build() resolves --target auto, so it runs before launch.target is read.
    mlir_module = launch.build(target=args.target)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    np.random.seed(0)
    input_x = np.random.uniform(-2.0, 2.0, args.n).astype(INPUT_DATATYPE)
    input_gate = np.random.uniform(-2.0, 2.0, args.n).astype(INPUT_DATATYPE)
    input_up = np.random.uniform(-2.0, 2.0, args.n).astype(INPUT_DATATYPE)

    # Pack gate and up into [2, N]: row 0 = gate, row 1 = up
    input_gate_up = np.stack([input_gate, input_up]).astype(INPUT_DATATYPE)

    if args.compile_mode == "compile-and-run":
        num_samples = 100
        sampled_indices = np.vstack([np.random.randint(0, args.n, num_samples)])

        # SwiGLU reference using tanh-based sigmoid (matches hardware computation)
        def swiglu_ref(x, gate, up):
            x_f32 = x.astype(np.float32)
            g_f32 = gate.astype(np.float32)
            u_f32 = up.astype(np.float32)
            xg = x_f32 * g_f32
            silu_xg = xg * 0.5 * (np.tanh(xg / 2.0) + 1.0)
            return silu_xg * (x_f32 * u_f32)

        sampled_values = np.array(
            [
                swiglu_ref(input_x[i], input_gate[i], input_up[i])
                for i in zip(*sampled_indices)
            ],
            dtype=INPUT_DATATYPE,
        )
        sampled_data = {
            "shape": (args.n,),
            "indices": sampled_indices,
            "values": sampled_values,
        }

        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="swiglu",
            runtime_loop_tiling_sizes=[4, 4],
            target_device=launch.target,
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[input_x, input_gate_up],
                stochastic_expected_outputs=[sampled_data],
                rtol=1e-1,
                atol=5e-2,
            )
        )

    elif args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            runtime_loop_tiling_sizes=[4, 4],
            target_device=launch.target,
        )
        module_function = backend.compile(mlir_module)
        backend.unload()
