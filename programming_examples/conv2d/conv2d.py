# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""2D convolution (NHWC input, HWCiCo filter) on one AIE tile, on air.api.

Valid convolution, stride 1::

    output[oh, ow, co] = sum over (kh, kw, ci) of
        input[oh + kh, ow + kw, ci] * filter[kh, kw, ci, co]

Written out, that is six loops with three loads, a multiply, an add and a store
at the bottom, and this is one of the kernels where that *is* the design: no
axis of it is tile-shaped, so there is nothing here for a whole-tile expression
to say. The DSL version is the same six loops::

    for oh, ow, co, kh, kw, ci in nested air.sequential(...):
        out[oh, ow, co] = out[oh, ow, co] + in[oh + kh, ow + kw, ci] * flt[kh, kw, ci, co]

Two pieces of the DSL carry it, and both are ordinary numpy spellings:

* **A fully-integer subscript is a scalar.** ``out[oh, ow, co]`` names one
  element, not a one-element tile, so assigning to it emits a store and no loop
  of its own. That is what keeps the emitted nest six deep rather than twelve.
* **A region's offset may be a loop variable.** ``in[oh + kh, ow + kw, ci]`` is
  the sliding window, and the shift reaches the load as one ``affine.apply`` --
  the same way every index in this DSL is built.

The kernel's own shape is therefore unchanged from the predecessor this
replaces: same nine ``scf.for``, same three ``memref.load``, same two
``memref.store``, same three DMAs, same herd. The one spelling difference is
``affine.apply`` where the predecessor wrote ``arith.addi`` for ``oh + kh``.

The batch axis is dropped on the way into L1 and restored on the way out -- N is
1, so ``[1, Ho, Wo, Co]`` and ``[Ho, Wo, Co]`` are the same buffer.
"""

import argparse

import numpy as np

from air import api as air
from air.api import ops
from air.api.types import i32
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner

# Small default sizes for a simple demonstration.
H_DEFAULT = 8
W_DEFAULT = 8
CI_DEFAULT = 4
CO_DEFAULT = 4
KH = 3
KW = 3


def build_module(H, W, Ci, Co, Kh, Kw, dtype=i32):
    if H < Kh or W < Kw:
        raise ValueError(
            f"invalid convolution: require H >= Kh and W >= Kw, got H={H}, "
            f"W={W}, Kh={Kh}, Kw={Kw}"
        )

    Ho = H - Kh + 1
    Wo = W - Kw + 1

    IN = air.tensor([1, H, W, Ci], dtype)
    FILTER = air.tensor([Kh, Kw, Ci, Co], dtype)
    OUT = air.tensor([1, Ho, Wo, Co], dtype)

    with air.launch(name="conv2d") as launch:

        @launch.body
        def _():
            with air.herd(range(1), name="herd_0", shape=(1,)) as h:

                @h.body
                def _(tx):
                    l1_in = air.alloc([H, W, Ci], dtype, scope=h.private())
                    l1_filter = air.alloc([Kh, Kw, Ci, Co], dtype, scope=h.private())
                    l1_out = air.alloc([Ho, Wo, Co], dtype, scope=h.private())

                    # The batch axis is 1, so it drops on the way in.
                    ops.load(l1_in, IN)
                    ops.load(l1_filter, FILTER)

                    for oh in air.sequential(Ho):
                        for ow in air.sequential(Wo):
                            for co in air.sequential(Co):
                                l1_out[oh, ow, co] = 0

                    for oh in air.sequential(Ho):
                        for ow in air.sequential(Wo):
                            for co in air.sequential(Co):
                                for kh in air.sequential(Kh):
                                    for kw in air.sequential(Kw):
                                        for ci in air.sequential(Ci):
                                            l1_out[oh, ow, co] = (
                                                l1_out[oh, ow, co]
                                                + l1_in[oh + kh, ow + kw, ci]
                                                * l1_filter[kh, kw, ci, co]
                                            )

                    ops.store(l1_out, OUT)

    return launch


def parse_args():
    parser = argparse.ArgumentParser(
        prog="conv2d.py",
        description="Builds, runs, and tests the 2D convolution example",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--H", type=int, default=H_DEFAULT, help="Input height")
    parser.add_argument("--W", type=int, default=W_DEFAULT, help="Input width")
    parser.add_argument("--Ci", type=int, default=CI_DEFAULT, help="Input channels")
    parser.add_argument("--Co", type=int, default=CO_DEFAULT, help="Output channels")
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
    return parser.parse_args()


def main():
    args = parse_args()

    Ho = args.H - KH + 1
    Wo = args.W - KW + 1

    launch = build_module(args.H, args.W, args.Ci, args.Co, KH, KW)
    mlir_module = launch.build(target=args.target)
    if args.print_module_only:
        print(mlir_module)
        return 0

    np.random.seed(0)
    input_data = np.random.randint(0, 4, size=(1, args.H, args.W, args.Ci)).astype(
        np.int32
    )
    filter_data = np.random.randint(0, 4, size=(KH, KW, args.Ci, args.Co)).astype(
        np.int32
    )

    # Reference convolution (NHWC layout).
    output_ref = np.zeros((1, Ho, Wo, args.Co), dtype=np.int32)
    for oh in range(Ho):
        for ow in range(Wo):
            for co in range(args.Co):
                for kh in range(KH):
                    for kw in range(KW):
                        for ci in range(args.Ci):
                            output_ref[0, oh, ow, co] += (
                                input_data[0, oh + kh, ow + kw, ci]
                                * filter_data[kh, kw, ci, co]
                            )

    if args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            runtime_loop_tiling_sizes=[4, 4],
            target_device=launch.target,
        )
        backend.compile(mlir_module)
        backend.unload()
        return 0

    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format=args.output_format,
        instance_name="conv2d",
        runtime_loop_tiling_sizes=[4, 4],
        target_device=launch.target,
    )
    return runner.run_test(
        mlir_module,
        inputs=[input_data, filter_data],
        expected_outputs=[output_ref],
    )


if __name__ == "__main__":
    exit(main())
