# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# The Gated DeltaNet decode step on one column of four cores, as FLM runs it:
# core c owns values 32c..32c+31 of every value head, takes each head's q, k
# (alpha after it) and v (beta after it) by broadcast, and streams that head's
# chunk of the state in and out.
import argparse
import os
import tempfile

import filelock
import numpy as np
from ml_dtypes import bfloat16

import common
import qwen38_27b_q4nx_weights as W
from air import api as air
from air.api import bf16, f32, i32
from air.backend.xrt import XRTBackend

# the output's element-wise atol, over max |want|: twice the worst measured
# over seeds 0-2 (4.2e-3)
ATOL = 1e-2

H, DK, DV, R = W.NVH, W.DK, W.DV, 4
VC = DV // R
SLOT = DV + 32  # a vector and its float32 scalar, padded


def build_module():
    step = air.extern("gdn_step", link_with="gdn_decode.o", scalars=[i32])
    KVQ = air.tensor([H, 3 * SLOT], bf16)
    S = air.tensor([H, R, DK * VC], f32)
    SO = air.tensor([H, R, DK * VC], f32)
    O = air.tensor([H, R, VC], bf16)
    kvq_in = air.channel("KVQ", size=[1, 1], broadcast_shape=[1, R])
    s_in = air.channel("SIn", size=[1, R])
    s_out = air.channel("SOut", size=[1, R])
    o_out = air.channel("OOut", size=[1, R])

    with air.launch(name="gdn") as launch:

        @launch.body
        def _():
            kvq_in.put(KVQ)
            for r in range(R):
                s_in.put(S[:, r, :], indices=[0, r])
                s_out.get(SO[:, r, :], indices=[0, r])
                o_out.get(O[:, r, :], indices=[0, r])
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    with air.herd([range(1), range(R)], name="gdn", shape=(1, R)) as he:

                        @he.body
                        def _(tx, ty):
                            p = he.private()
                            kvq = air.alloc([3 * SLOT], bf16, scope=p)
                            si = air.alloc([DK * VC], f32, scope=p)
                            so = air.alloc([DK * VC], f32, scope=p)
                            o = air.alloc([VC], bf16, scope=p)
                            for _h in air.sequential(0, H):
                                kvq_in.get(kvq, indices=[tx, ty])
                                s_in.get(si, indices=[tx, ty])
                                step(si, so, kvq, o, ty)
                                s_out.put(so, indices=[tx, ty])
                                o_out.put(o, indices=[tx, ty])

    return launch.build(target="npu2")


def pack(k, alpha, v, beta, q):
    """Per head [k, alpha | v, beta | q], each slot SLOT bf16, a scalar as a
    float32 in the two bf16 after its vector."""
    out = np.zeros((H, 3, SLOT), bfloat16)
    for i, (vec, s) in enumerate(((k, alpha), (v, beta), (q, None))):
        out[:, i, : vec.shape[1]] = vec
        if s is not None:
            out.view(np.uint16)[:, i, DK : DK + 2] = (
                np.asarray(s, np.float32).view(np.uint16).reshape(H, 2)
            )
    return out.reshape(H, 3 * SLOT)


def reference(q, k, v, alpha, beta, S):
    """q, k, v [H, 128] (bf16 values), S [H, DK, DV] -> (S', o [H, DV])."""
    q, k, v, S = (a.astype(np.float64) for a in (q, k, v, S))
    sk = np.einsum("hkv,hk->hv", S, k)
    u = beta[:, None] * (v - alpha[:, None] * sk)
    S2 = alpha[:, None, None] * S + k[:, :, None] * u[:, None, :]
    o = np.einsum("hkv,hk->hv", S2, q)
    return S2, o


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--print-module-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    module = build_module()
    if a.print_module_only:
        print(module)
        raise SystemExit(0)
    common.compile_kernel("gdn_decode")
    rng = np.random.default_rng(a.seed)
    unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)  # noqa: E731
    q = (unit(rng.standard_normal((H, DK))) / np.sqrt(DK)).astype(bfloat16)
    k = unit(rng.standard_normal((H, DK))).astype(bfloat16)
    v = rng.standard_normal((H, DV)).astype(bfloat16)
    alpha = rng.uniform(0.5, 1.0, H).astype(np.float32)
    beta = rng.uniform(0.0, 1.0, H).astype(np.float32)
    S = (0.3 * rng.standard_normal((H, DK, DV))).astype(np.float32)
    # state chunks: [head][chunk][key][32 values]
    Sd = np.ascontiguousarray(S.reshape(H, DK, R, VC).transpose(0, 2, 1, 3))
    be = XRTBackend(instance_name="gdn")
    art = be.compile(module)
    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        run = be.load(art)
        out = run(
            pack(k, alpha, v, beta, q),
            Sd.reshape(H, R, DK * VC),
            np.zeros((H, R, DK * VC), np.float32),
            np.zeros((H, R, VC), bfloat16),
        )
        be.unload()
    S2 = out[2].reshape(H, R, DK, VC).transpose(0, 2, 1, 3).reshape(H, DK, DV)
    o = out[3].reshape(H, DV).astype(np.float64)
    S2w, ow = reference(q, k, v, alpha, beta, S)
    # the float32 state's rounding: 1.0e-5 to 1.4e-5 of max |state| over
    # seeds, at 32 heads (Qwen3.6) as at 48
    ok = common.check("state", S2, S2w, 2e-5, rtol=0.0)
    ok &= common.check("out", o, ow, ATOL)
    print("PASS!" if ok else "failed.")
    raise SystemExit(0 if ok else 1)
