# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# The host sets how many blocks a memtile sends, at run time.
#
# The host fills an L2 buffer with NB blocks; a run-time scalar L says how many
# of them the memtile sends on (a KV cache read back up to the current length).
# The compiler leaves the memtile's buffer descriptor out of its static
# program; the runtime sequence writes it, then picks the entry for L from a
# table of control packets that set its length and start it.

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
    KV = air.tensor([NB, BLK], i32)
    Y = air.tensor([NB, BLK], i32)
    L = air.tensor([], i32, name="L")
    kv_in = air.channel("KvIn")
    kv_out = air.channel("KvOut")
    yout = air.channel("YOut")

    with air.launch(name="host_issue_l2", repeat=1) as launch:

        @launch.body
        def _(w):
            n = air.rtp(L.value)
            with air.segment(name="seg", params=[n]) as seg:

                @seg.body
                def _():
                    kv = air.alloc([NB, BLK], i32, scope=seg.private())
                    kv_in.put(KV)
                    kv_in.get(kv)
                    kv_out.put(kv, extent=n.as_index())

                    with air.herd([range(1)], name="fwd", shape=(1,), params=[n]) as f:

                        @f.body
                        def _(tx):
                            blk = air.alloc([BLK], i32, scope=f.private())
                            o = air.alloc([BLK], i32, scope=f.private())
                            for _b in air.sequential(0, n.as_index()):
                                kv_out.get(blk)
                                o[:] = blk[:]
                                yout.put(o)

            yout.get(Y[:, :], extent=L)

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

    kv = (np.arange(NB)[:, None] * 1000 + np.arange(BLK)[None, :] + 1).astype(np.int32)
    backend = XRTBackend(
        output_format=args.output_format,
        instance_name="host_issue_l2",
        emit_txn_cpp=args.output_format == "xclbin",
    )
    art = backend.compile(mlir_module)
    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        run = backend.load(art)
        ok = True
        for L in [int(v) for v in args.L.split(",")]:
            y = run(kv, np.zeros((NB, BLK), np.int32), L)[1].reshape(NB, BLK)
            want = np.zeros((NB, BLK), np.int32)
            want[:L] = kv[:L]
            good = np.array_equal(y, want)
            ok &= good
            print(f"L={L}: {'ok' if good else 'MISMATCH'}")
        backend.unload()
    print("PASS!" if ok else "failed.")
    exit(0 if ok else -1)
