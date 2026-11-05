# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# A shim transfer whose size is a run-time scalar, under an axis that steps.
#
# repeat: the shim sends the left half of each of the first L blocks of X. The
#   block count is the run-time outermost size of a strided pattern: the task
#   repeats L times and the descriptor's iteration steps it a block each time.
# slice: the shim sends blocks b0 .. b0 + n - 1 of every row of X, a slice
#   whose size is known only at run time. Each row's share is one contiguous
#   run of run-time length; the descriptor's iteration steps it row to row.

import argparse
import os
import tempfile

import filelock
import numpy as np

from air import api as air
from air.api.types import i32
from air.backend.xrt import XRTBackend

NB, R, C = 8, 4, 64


def forward(x_in, yout, n, rows, blocks):
    """A core that writes the `rows` x `n` blocks it receives, in order, into a
    `rows` x `blocks` result filled with -1."""
    with air.herd([range(1)], name="fwd", shape=(1,), params=[n]) as f:

        @f.body
        def _(tx):
            blk = air.alloc([C], i32, scope=f.private())
            res = air.alloc([rows, blocks, C], i32, scope=f.private())
            air.ops.fill(res, -1)
            for r in range(rows):
                for b in air.sequential(0, n.as_index()):
                    x_in.get(blk)
                    res[r, b, :] = blk[:]
            yout.put(res)


def build_module(mode):
    if mode == "repeat":
        X = air.tensor([NB, 1, 2 * C], i32)
        Y = air.tensor([1, NB, C], i32)
    else:
        X = air.tensor([R, NB, C], i32)
        Y = air.tensor([R, NB, C], i32)
    L = air.tensor([], i32, name="L")
    B0 = air.tensor([], i32, name="b0")
    x_in = air.channel("XIn")
    yout = air.channel("YOut")

    with air.launch(name="rt_strided", repeat=1) as launch:

        @launch.body
        def _(w):
            n, b0 = air.rtp(L.value), air.rtp(B0.value)
            with air.segment(name="seg", params=[n, b0]) as seg:

                @seg.body
                def _():
                    if mode == "repeat":
                        x_in.put(X[:, :, 0:C], extent=n.as_index())
                        forward(x_in, yout, n, 1, NB)
                    else:
                        first = b0.as_index()
                        x_in.put(X[:, first : first + n.as_index(), :])
                        forward(x_in, yout, n, R, NB)

            yout.get(Y)

    return launch.build(target="npu2")


def expected(mode, x, n, b0):
    if mode == "repeat":
        want = np.full((1, NB, C), -1, np.int32)
        want[0, :n] = x[:n, 0, :C]
    else:
        want = np.full((R, NB, C), -1, np.int32)
        want[:, :n] = x[:, b0 : b0 + n]
    return want


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="run.py")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--mode", choices=["repeat", "slice"], default="repeat")
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="xclbin",
        dest="output_format",
    )
    args = parser.parse_args()

    mlir_module = build_module(args.mode)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    shape = (NB, 1, 2 * C) if args.mode == "repeat" else (R, NB, C)
    x = np.arange(1, np.prod(shape) + 1, dtype=np.int32).reshape(shape)
    out = (1, NB, C) if args.mode == "repeat" else (R, NB, C)
    cases = [(3, 0), (1, 0), (8, 0), (5, 0)]
    if args.mode == "slice":
        cases = [(3, 2), (1, 7), (8, 0), (5, 3)]
    backend = XRTBackend(
        output_format=args.output_format,
        instance_name="rt_strided",
        emit_txn_cpp=args.output_format == "xclbin",
    )
    art = backend.compile(mlir_module)
    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        run = backend.load(art)
        ok = True
        for n, b0 in cases:
            y = run(x, np.zeros(out, np.int32), n, b0)[1].reshape(out)
            good = np.array_equal(y, expected(args.mode, x, n, b0))
            ok &= good
            print(f"n={n} b0={b0}: {'ok' if good else 'MISMATCH'}")
        backend.unload()
    print("PASS!" if ok else "failed.")
    exit(0 if ok else -1)
