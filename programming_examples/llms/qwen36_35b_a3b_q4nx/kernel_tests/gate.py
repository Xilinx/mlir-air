# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# The mixer output's gate on one core: sigmoid(gate) * o for a full-attention
# layer, RMSNorm(o) * w * SiLU(z) per value head for a Gated DeltaNet layer.
import os
import tempfile

import filelock
import numpy as np
import torch
import torch.nn.functional as F
from ml_dtypes import bfloat16

import common
import qwen36_35b_a3b_q4nx_weights as W
from air import api as air
from air.api import bf16, i32
from air.backend.xrt import XRTBackend

# a check's bound on the relative L2 error
TOL = 0.03

N = W.NVH * W.DV  # = NQ * DH


def build_module():
    sig = air.extern("gate_sigmoid", link_with="gate.o", scalars=[i32])
    slu = air.extern("gate_silu", link_with="gate.o", scalars=[i32])
    nrm = air.extern("dn_norm", link_with="gate.o", scalars=[i32])
    IN = air.tensor([4, N], bf16)
    WN = air.tensor([W.DV], bf16)
    OUT = air.tensor([2, N], bf16)
    x_in, w_in, y_out = (air.channel(n) for n in ("XIn", "WIn", "YOut"))

    with air.launch(name="gate") as launch:

        @launch.body
        def _():
            x_in.put(IN)
            w_in.put(WN)
            y_out.get(OUT)
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    with air.herd([range(1)], name="gate", shape=(1,)) as he:

                        @he.body
                        def _(tx):
                            p = he.private()
                            a = air.alloc([N], bf16, scope=p)
                            b = air.alloc([N], bf16, scope=p)
                            y = air.alloc([N], bf16, scope=p)
                            w = air.alloc([W.DV], bf16, scope=p)
                            x_in.get(a)
                            x_in.get(b)
                            sig(y, a, b, N)
                            y_out.put(y)
                            w_in.get(w)
                            x_in.get(a)
                            x_in.get(b)
                            nrm(b, b, w, N)
                            slu(y, a, b, N)
                            y_out.put(y)

    return launch.build(target="npu2")


if __name__ == "__main__":
    common.compile_kernel("gate")
    w = W.Qwen36Q4nx()
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, N)).astype(bfloat16)  # gate, o, z, o
    nw = w.gdn(0)[6].astype(bfloat16)
    be = XRTBackend(instance_name="gate")
    art = be.compile(build_module())
    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        run = be.load(art)
        out = run(x, nw, np.zeros((2, N), bfloat16))[2].reshape(2, N)
        be.unload()
    t = torch.from_numpy(x.astype(np.float32))
    want_a = torch.sigmoid(t[0]) * t[1]
    o = t[3].view(-1, W.DV)
    normed = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + W.EPS)
    want_g = (normed * torch.from_numpy(nw.astype(np.float32))).flatten() * F.silu(t[2])
    ok = True
    for n, g, wnt in (("attention", out[0], want_a), ("gdn", out[1], want_g)):
        c, l2, e = common.compare(g, wnt.double().numpy())
        good = c > 0.999 and l2 < TOL
        ok &= good
        print(
            f"{n}: corr {c:.6f} rel L2 {l2:.2e} max rel err {e:.2e}"
            f" {'ok' if good else 'BAD'}"
        )
    print("PASS!" if ok else "failed.")
    raise SystemExit(0 if ok else 1)
