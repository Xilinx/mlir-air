# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""A kernel that branches on a parameter, on air.api.

``out = in + 100`` or ``out = in * 100``, chosen by ``param``. The module is
built twice, once per value, and both are run.

This is the third shape of conditional in the tree, and the one that is *not*
about tile coordinates: the predicate here is a **dispatch-time parameter**, so
it is the same for every core. It is still a region and not a Python ``if``,
because the branch is what the example exists to demonstrate -- writing
``if param:`` in Python would pick a body at trace time and hand the compiler a
kernel with no ``scf.if`` in it at all.

``air.symbol`` is the DSL's name for a value of this kind ("known at dispatch
time rather than compile time"). Its *arithmetic* folds to a Python int, because
a symbol used as a tile size has to size a memref; its *comparisons* do not,
because folding one would delete the branch. ``mode != 0`` therefore reaches the
IR as ``arith.constant`` + ``arith.cmpi`` feeding the ``scf.if``, where the
predecessor spelled the same thing as ``arith.index_cast(param) : i1``. Both are
a constant driving a branch, and ``air-to-aie``'s ``SpecializeScfIfPattern``
folds either one away once the herd is unrolled -- so the branch is present for
the compiler to reason about and costs nothing at run time.

The two arms are ``buf[:] = inp[:] + 100`` and ``buf[:] = inp[:] * 100``,
replacing scalar ``memref.load``/``store`` loops over all 48 elements.
"""

import argparse

import numpy as np

from air import api as air
from air.api import ops
from air.api.types import i32
from air.backend.xrt_runner import XRTRunner

N = 48
INOUT_DATATYPE = np.int32


def build_module(n, param):
    src = air.tensor([n], i32)
    dst = air.tensor([n], i32)

    # Not a plain Python int: a comparison on one of those is answered at trace
    # time, and there would be no scf.if left to compile.
    mode = air.symbol(hint=param, name="param")

    with air.launch(name="conditional_branch") as launch:

        @launch.body
        def _():
            with air.segment(name="segment_0") as seg:

                @seg.body
                def _():
                    staged_in = air.alloc([n], i32, scope=seg.private())
                    staged_out = air.alloc([n], i32, scope=seg.private())
                    ops.load(staged_in, src)

                    with air.herd([range(1)], name="herd_0", shape=(1,)) as herd:

                        @herd.body
                        def _(tx):
                            inp = air.alloc([n], i32, scope=herd.private())
                            ops.load(inp, staged_in)
                            buf = air.alloc([n], i32, scope=herd.private())

                            with ops.branch(mode != 0) as chosen:
                                buf[:] = inp[:] + 100
                            with chosen.otherwise():
                                buf[:] = inp[:] * 100

                            ops.store(buf, staged_out)

                    ops.store(staged_out, dst)

    return launch


def parse_args():
    parser = argparse.ArgumentParser(
        prog="single_core.py",
        description="Builds, runs and tests a kernel that branches on a parameter",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--n", type=int, default=N, help="Vector length")
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

    if args.print_module_only:
        print(build_module(args.n, 0).build(target=args.target))
        return 0

    inputs = np.arange(0, args.n, dtype=INOUT_DATATYPE)
    results = []

    # param = 0 takes the else arm (* 100); param = 1 takes the then arm (+ 100).
    for param, expected in ((0, inputs * 100), (1, inputs + 100)):
        launch = build_module(args.n, param)
        # build() is what resolves --target auto to the installed generation, so
        # the module has to exist before launch.target is read. Reading it first
        # passes None to the runner, and the compile fails with the unhelpful
        # "'builtin.module' op Invalid aie.device option".
        mlir_module = launch.build(target=args.target)
        runner = XRTRunner(
            verbose=args.verbose,
            output_format=args.output_format,
            instance_name="conditional_branch",
            runtime_loop_tiling_sizes=[4, 4],
            target_device=launch.target,
        )
        results.append(
            runner.run_test(
                mlir_module,
                inputs=[inputs],
                expected_outputs=[expected.astype(INOUT_DATATYPE)],
            )
        )

    if all(r == 0 for r in results):
        print("Both conditions PASS!")
        return 0
    for i, r in enumerate(results):
        if r != 0:
            print(f"Cond. {i} FAIL!")
    return 1


if __name__ == "__main__":
    exit(main())
