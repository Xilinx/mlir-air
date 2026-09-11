#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Python equivalent of elfver_run.cpp: replay one decode step through the
# full-ELF build, taking the context length from the scratchpad. Needs an XRT
# whose pyxrt exposes run.get_ctrl_scratchpad_bo() (>= 2026-05-19); before that
# the C++ harness was the only way in.
#
#   python3 elfver_run.py --dir /tmp/elfver --elf decode.elf --l 1000

import argparse
from pathlib import Path

import numpy as np
import pyxrt as xrt
from ml_dtypes import bfloat16
from aie.utils.hostruntime.xrtruntime.parameter_scratchpad import ParameterScratchpad

KERNEL = "main:q4nx_decode"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/tmp/elfver")
    ap.add_argument("--elf", default="decode.elf")
    ap.add_argument("--params", default="params.txt")
    ap.add_argument("--l", type=int, default=None)
    a = ap.parse_args()

    meta = {}
    for line in open(f"{a.dir}/meta.txt"):
        k, v = line.split()
        meta[k] = int(v)
    L = a.l if a.l is not None else meta["L"]

    dev = xrt.device(0)
    kern = xrt.ext.kernel(xrt.hw_context(dev, xrt.elf(a.elf)), KERNEL)
    TO = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    FROM = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE

    specs = [
        ("x.bin", meta["k"]),
        ("W.bin", meta["w_elems"]),
        ("rms.bin", meta["rms_size"]),
        (None, meta["ny"]),
        ("kv.bin", meta["kv_elems"]),
    ]
    bos, maps = [], []
    for name, n in specs:
        b = xrt.ext.bo(dev, n * 2)
        m = np.frombuffer(b.map(), dtype=np.int16)
        m[:] = 0 if name is None else np.fromfile(f"{a.dir}/{name}", np.int16)
        b.sync(TO)
        bos.append(b)
        maps.append(m)

    run = xrt.run(kern)
    for i, b in enumerate(bos):
        run.set_arg(i, b)
    run.set_arg(5, L)

    params = ParameterScratchpad(run, f"{a.params}")
    params.write("__air_param_argoff_5_x256_m256", np.int32((L - 1) * meta["region_w"]))
    params.write("__air_param_attn_blk_0", np.int32(L))
    params.sync()

    run.start()
    run.wait2()
    assert "COMPLETED" in str(run.state()), run.state()

    bos[3].sync(FROM)
    dy, vn, vs = meta["decode_y"], meta["voc_n"], meta["vocab"]
    got = maps[3].view(bfloat16)[dy : dy + vn].astype(np.float32)[:vs]
    ref = np.fromfile(f"{a.dir}/ref.bin", np.float32)
    print(
        f"[py-elf] L={L} argmax={int(np.argmax(got))} top5={np.argsort(got)[-5:][::-1].tolist()}"
    )
    print(
        f"[py-elf] bit-identical to xclbin ref: {bool(np.array_equal(ref, got))}"
        f"  max|diff|={float(np.max(np.abs(ref - got))):.6g}"
    )


if __name__ == "__main__":
    main()
