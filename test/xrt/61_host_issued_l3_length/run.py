# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# The host sets how many rows a shim sends, at run time, by control packets.
#
# A run-time scalar L says how many rows of X the shim streams to the core. The
# launch asks (air.host_issue_lengths) for the length to be set by control
# packets rather than through the scratchpad: the runtime sequence configures
# the shim's buffer descriptor for all NB rows, then picks the entry for L from
# a table of control packets that set its length and start it.

import argparse
import os
import tempfile

import filelock
import numpy as np

from air import api as air
from air.api.types import i32
from air.backend.xrt import XRTBackend

NB, BLK = 8, 256


def build_module():
    X = air.tensor([NB, BLK], i32)
    Y = air.tensor([NB, BLK], i32)
    L = air.tensor([], i32, name="L")
    x_in = air.channel("XIn")
    yout = air.channel("YOut")

    with air.launch(
        name="host_issue_l3", repeat=1, attrs=["air.host_issue_lengths"]
    ) as launch:

        @launch.body
        def _(w):
            n = air.rtp(L.value)
            with air.segment(name="seg", params=[n]) as seg:

                @seg.body
                def _():
                    x_in.put(X[:, :], extent=n.as_index())

                    with air.herd([range(1)], name="fwd", shape=(1,), params=[n]) as f:

                        @f.body
                        def _(tx):
                            blk = air.alloc([BLK], i32, scope=f.private())
                            res = air.alloc([NB, BLK], i32, scope=f.private())
                            air.ops.fill(res, -1)
                            for b in air.sequential(0, n.as_index()):
                                x_in.get(blk)
                                res[b, :] = blk[:]
                            yout.put(res)

            yout.get(Y)

    return launch.build(target="npu2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="run.py")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--L", type=str, default="3,1,8")
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="xclbin",
        dest="output_format",
    )
    args = parser.parse_args()

    mlir_module = build_module()
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    x = (np.arange(NB)[:, None] * 1000 + np.arange(BLK)[None, :] + 1).astype(np.int32)
    backend = XRTBackend(
        output_format=args.output_format,
        instance_name="host_issue_l3",
        emit_txn_cpp=args.output_format == "xclbin",
    )
    art = backend.compile(mlir_module)
    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        run = backend.load(art)
        ok = True
        for L in [int(v) for v in args.L.split(",")]:
            y = run(x, np.zeros((NB, BLK), np.int32), L)[1].reshape(NB, BLK)
            want = np.full((NB, BLK), -1, np.int32)
            want[:L] = x[:L]
            good = np.array_equal(y, want)
            ok &= good
            print(f"L={L}: {'ok' if good else 'MISMATCH'}")
        backend.unload()
    print("PASS!" if ok else "failed.")
    exit(0 if ok else -1)
