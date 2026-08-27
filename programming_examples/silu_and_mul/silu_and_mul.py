# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""SiLU + elementwise multiply, on air.api.

    output[i] = SiLU(gate[i]) * up[i]

The activation stays in C++: `air.extern("silu_and_mul_bf16", link_with="silu_and_mul.o")`
declares the microkernel and stamps link_with on both the declaration and the
herd, so the body here is pure dataflow. This is the twin of `gelu_and_mul/` --
same dataflow, same herd sizing, same argument layout, only the microkernel
differs.

Each tile uses 3 independent shim DMAs (gate in, up in, out) and NPU2 has 8
shim DMA channels, so the herd is capped at `herd_x * herd_y <= 8` tiles; 8x1
and 2x4 place, 8x2 / 4x4 / 8x4 do not. The best config is herd_x=8, herd_y=1,
the full chip width.

**Two entry points over one body.** `build_module` takes a flat [n] interface
and `build_module_2d` takes [rows, cols], which is what the FFN GEMMs produce.
The predecessor wrote the second as a separate builder that inserted three
`memref.collapse_shape` ops at launch scope; here the 2-D tensor region is
reshaped to [n] where it is sliced, so one body serves both and the collapse is
part of the access pattern rather than an op.

Both return the **module**, not the launch, and keep their signatures: the
llms/ FFN builders import them and stitch the result by positional operand
index, so the argument order (gate, up, out) and the `@silu_and_mul_bf16` symbol
are a contract.

The tile offset is ordinary Python arithmetic on the coordinates,

    loop_iv + (tx * herd_y + ty) * tile_n

where the predecessor built the same expression as a three-symbol AffineMap.
"""

import argparse

import numpy as np
from ml_dtypes import bfloat16

from air import api as air
from air.api import ops
from air.api.types import bf16, i32
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner


def build_launch(shape, tile_n, dtype=bf16, herd_x=8, herd_y=1, name=None):
    """One body for both interfaces; `shape` is [n] or [rows, cols].

    `name` is the emitted func's symbol. The two entry points keep the two names
    the predecessor emitted, because the llms/ stitchers slice these modules by
    symbol as well as by operand position.
    """
    n = 1
    for extent in shape:
        n *= int(extent)
    total_tiles = herd_x * herd_y
    if n % (tile_n * total_tiles):
        raise ValueError(
            f"n ({n}) must be divisible by tile_n * herd tiles "
            f"({tile_n * total_tiles}): every tile takes the same number of "
            "elements, and there is no remainder path."
        )
    if total_tiles > 8:
        raise ValueError(
            f"herd_x * herd_y is {total_tiles}: each tile needs 3 shim DMAs "
            "and NPU2 has 8 shim channels, so more than 8 tiles does not place."
        )

    gate = air.tensor(list(shape), dtype)
    up = air.tensor(list(shape), dtype)
    out = air.tensor(list(shape), dtype)

    def flat(t):
        """The tensor as one [n] run, whatever rank it was declared at.

        A rank-2 region reshaped to [n] is the access pattern the predecessor
        built with memref.collapse_shape -- the elements are already contiguous,
        so this renames the walk rather than moving anything.
        """
        whole = t[tuple(slice(0, int(e)) for e in shape)]
        return whole if len(shape) == 1 else whole.reshape(n)

    activation = air.extern(
        "silu_and_mul_bf16", link_with="silu_and_mul.o", scalars=[i32]
    )

    with air.launch(name=name or "silu_and_mul") as launch:

        @launch.body
        def _():
            with air.segment(name="silu_mul_seg") as seg:

                @seg.body
                def _():
                    with air.herd(
                        [range(herd_x), range(herd_y)],
                        name="herd_0",
                        shape=(herd_x, herd_y),
                    ) as herd:

                        @herd.body
                        def _(tx, ty):
                            l1_gate = air.alloc([tile_n], dtype, scope=herd.private())
                            l1_up = air.alloc([tile_n], dtype, scope=herd.private())
                            l1_out = air.alloc([tile_n], dtype, scope=herd.private())

                            for iv in air.sequential(0, n, tile_n * total_tiles):
                                lo = iv + (tx * herd_y + ty) * tile_n

                                ops.load(l1_gate, flat(gate)[lo : lo + tile_n])
                                ops.load(l1_up, flat(up)[lo : lo + tile_n])

                                activation(l1_gate, l1_up, l1_out, tile_n)

                                ops.store(l1_out, flat(out)[lo : lo + tile_n])

    return launch


def build_module(n, tile_n, np_dtype_in=bfloat16, herd_x=8, herd_y=None, target="npu2"):
    """Flat [n] interface. Signature is the llms/ builders' contract."""
    return build_launch(
        [n], tile_n, bf16, herd_x, herd_y or 1, name="silu_and_mul"
    ).build(target=target)


def build_module_2d(
    rows, cols, tile_n, np_dtype_in=bfloat16, herd_x=8, herd_y=1, target="npu2"
):
    """[rows, cols] interface, for gate/up straight out of the FFN GEMMs."""
    return build_launch(
        [rows, cols], tile_n, bf16, herd_x, herd_y, name="silu_and_mul_2d"
    ).build(target=target)


def silu_reference(x):
    """Reference SiLU implementation in F32."""
    x_f32 = x.astype(np.float32)
    return x_f32 * (1.0 / (1.0 + np.exp(-x_f32)))


if __name__ == "__main__":
    N = 65536
    TILE_N = 1024
    INPUT_DATATYPE = bfloat16

    parser = argparse.ArgumentParser(
        prog="silu_and_mul.py",
        description="Builds, runs, and tests the standalone SwiGLU activation kernel",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--n", type=int, default=N, help="Total number of elements")
    parser.add_argument("--tile-n", type=int, default=TILE_N, help="Tile size")
    parser.add_argument(
        "--herd-x",
        type=int,
        default=8,
        help="Herd x dimension (AIE columns, default: 8 — full chip width)",
    )
    parser.add_argument(
        "--herd-y",
        type=int,
        default=None,
        help="Herd y dimension (AIE rows, default: 1). NPU2 caps the herd at "
        "herd_x * herd_y <= 8 tiles (3 shim DMAs/tile, 8 shim channels)",
    )
    parser.add_argument(
        "--perf-iters",
        type=int,
        default=0,
        dest="perf_iters",
        help="If >0, time the kernel over this many iters (after warmup) and "
        "print Latency in addition to the correctness check",
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

    if args.perf_iters < 0:
        parser.error("--perf-iters must be >= 0")

    mlir_module = build_module(
        args.n, args.tile_n, INPUT_DATATYPE, herd_x=args.herd_x, herd_y=args.herd_y
    )
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    # Use N(0,1) (matching the GPU/HF SiLU test standard) so the correctness
    # check sees a realistic signed distribution rather than an all-positive
    # one. Generate in float32 to avoid a large f64 intermediate.
    rng = np.random.default_rng(0)
    gate = rng.standard_normal(args.n, dtype=np.float32).astype(INPUT_DATATYPE)
    up = rng.standard_normal(args.n, dtype=np.float32).astype(INPUT_DATATYPE)

    # Reference: SiLU(gate) * up, full-output FP32 (bf16 inputs upcast to f32,
    # true sigmoid 1/(1+exp(-g)), cast back to bf16) — the bf16-rounded
    # reference a GPU/HF SiLU-and-mul op is verified against. The NPU kernel
    # uses the math-identical 0.5(1+tanh(g/2)) sigmoid; the bf16-tanh path is
    # exactly the error this measures.
    silu_gate = silu_reference(gate)
    expected = (silu_gate * up.astype(np.float32)).astype(INPUT_DATATYPE)

    if args.compile_mode == "compile-and-run":
        # bf16 SiLU-and-mul: rtol = canonical bf16 1.6e-2; atol = 8e-2 sized to
        # the measured worst-case single-element error (NPU2: abs_err max ~0.125,
        # min atol to pass element-wise ~6.7e-2 -> 8e-2 with margin). SiLU is a
        # saturating non-linearity and the hardware aie::tanh<bf16> LUT is
        # coarser than a rounded np.tanh, so the worst element is larger than a
        # pure-rounding op; the mean error (mean_rel_L1 ~1.0e-2) stays inside the
        # bf16 tier. Justified vs the GPU std rtol=1.6e-2/atol=1e-3 in the
        # detail page (02_precision / 03_performance).
        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="silu_and_mul",
            report_precision=True,
            n_perf_iters=args.perf_iters,
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[gate, up],
                expected_outputs=[expected],
                rtol=1.6e-2,
                atol=8e-2,
            )
        )

    elif args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
        )
        module_function = backend.compile(mlir_module)
        backend.unload()
