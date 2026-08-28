# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Shared-L1 communication between cores of a single herd, on air.api.

A column of NC cores is one ``air.herd [1, NC]``. Every core runs the same body
and learns its position from ``ty``. Data flows down the column entirely through
shared L1 -- there is no DMA and no hardware cascade between the cores::

    core 0   : v0 = g(in0)                    -> hop[0]
    core k   : vk = g(ink) + hop[k-1]         -> hop[k]        (1 <= k < NC-1)
    core NC-1: out = g(in_{NC-1}) + hop[NC-2] -> @outY

Each ``hop[k]`` is an L1 buffer that neighbouring cores k and k+1 both address:
k writes it, k+1 reads it. In the DSL that is ``<segment>.per_core()`` -- L1
allocated at segment scope and handed to the herd whole. The name says what the
*allocation* is, one buffer per core rather than one slab of a divided one; that
the cores of a single herd can then hand data along it is the compiler's doing,
and it is what this example exists to exercise.

``g()`` adds a per-step constant chosen by ``ops.switch`` on the loop counter --
a value picked by a runtime index, which is ``scf.index_switch`` in its
value-returning form.

Result: out = sum(in[0..NC-1]) + NC * sum(STEP_ADDENDS).

One departure from the predecessor's IR. The role dispatch was a second
``scf.index_switch``, keyed on ``ty`` and run for its effects; here it is a
chain of ``ops.branch``, which is ``scf.if``. The DSL has one N-way construct
and it is the value-returning one, on the grounds that the statement form is
nested branch -- so this is that spelling, not a missing feature. What the
example demonstrated twice it now demonstrates once, in the half where a switch
buys something a branch does not.

The chunk width is pinned with ``vector=VEC`` rather than left to the emitter:
a producer's write to a shared buffer and the consumer's read of it have to
carry matching per-chunk lock acquire/release counts, so the two must be
chunked the same way.
"""

import argparse

import numpy as np
from ml_dtypes import bfloat16

from air import api as air
from air.api import ops
from air.api.types import bf16
from air.backend.xrt_runner import XRTRunner

np.random.seed(42)

NC = 4  # cores in the column; NC-1 shared-L1 hops
T = 64  # elements per core tile
VEC = 16  # vector width

STEP_ADDENDS = [1.0, 10.0]  # per-step constant selected by ops.switch
NSTEP = len(STEP_ADDENDS)


def build_module():
    l3_in = air.tensor([NC, T], bf16)
    l3_out = air.tensor([1, T], bf16)

    in_x = air.channel("inX", size=[NC])  # per-core input feed (L2 -> L1)
    out_y = air.channel("outY", size=[1])  # last core's result (L1 -> L2)

    with air.launch([range(1), range(1)], name="col_relay") as launch:

        @launch.body
        def _(lx, ly):

            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    # Input L3 -> L2 in one transfer, then fanned per core.
                    in_l2 = air.alloc([NC, T], bf16, scope=seg.private())
                    ops.load(in_l2, l3_in)
                    for c in range(NC):
                        in_x.put(in_l2[c, :], indices=[c])

                    # One hop buffer per relay edge: core k writes hop[k],
                    # core k+1 reads it.
                    hop = [
                        air.alloc([T], bf16, scope=seg.per_core(), vector=VEC)
                        for _ in range(NC - 1)
                    ]

                    with air.herd(
                        [range(1), range(NC)], name="col_relay", shape=(1, NC)
                    ) as h:

                        @h.body
                        def _(tx, ty):
                            local = air.alloc([T], bf16, scope=h.private(), vector=VEC)
                            in_x.get(local, indices=[ty])

                            # Every core runs this: local += the step's addend,
                            # which ops.switch picks from the loop counter.
                            for step in air.sequential(0, NSTEP):
                                local[:] = local[:] + ops.switch(step, STEP_ADDENDS)

                            def role(k):
                                if k > 0:
                                    local[:] = local[:] + hop[k - 1][:]
                                if k < NC - 1:
                                    hop[k][:] = local[:]
                                else:
                                    out_y.put(local)

                            # core 0, then 1, ... with the last core in the
                            # final else, matching the predecessor's default arm.
                            def dispatch(k):
                                if k == NC - 1:
                                    role(k)
                                    return
                                with ops.branch(ty == k) as arm:
                                    role(k)
                                with arm.otherwise():
                                    dispatch(k + 1)

                            dispatch(0)

                    # The last core's result, L1 -> L2 -> L3.
                    out_l2 = air.alloc([1, T], bf16, scope=seg.private())
                    out_y.get(out_l2)
                    ops.store(out_l2, l3_out)

    return launch


def parse_args():
    p = argparse.ArgumentParser(description="Single-column shared-L1 relay")
    p.add_argument("-p", "--print-ir", action="store_true", help="Print IR and exit")
    p.add_argument(
        "--output-format",
        choices=["xclbin", "elf"],
        default="xclbin",
        dest="output_format",
    )
    p.add_argument(
        "--target",
        type=str,
        default="auto",
        help="NPU generation to build for: auto (default, detects the installed "
        "device), npu1 or npu2",
    )
    return p.parse_args()


def main():
    args = parse_args()

    launch = build_module()
    mlir_module = launch.build(target=args.target)
    if args.print_ir:
        print(mlir_module)
        return 0

    A = np.random.rand(NC, T).astype(bfloat16)
    s = float(sum(STEP_ADDENDS))
    C = (A.astype(np.float32).sum(axis=0) + NC * s).astype(bfloat16).reshape(1, T)

    runner = XRTRunner(
        omit_while_true_loop=False,
        verbose=False,
        output_format=args.output_format,
        instance_name="col_relay",
        target_device=launch.target,
    )
    return runner.run_test(mlir_module, inputs=[A], expected_outputs=[C], rtol=3e-2)


if __name__ == "__main__":
    exit(main())
