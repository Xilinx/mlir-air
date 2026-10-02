# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The decode builder and the weight packer agree on every layer's slab.

Weight-free: synthetic weights of each layer class's real shapes. Checks that
a slab has the size the builder streams for its wave, and that a narrow or
KV-shared slab is the padded slab with whole rounds (or down-proj column steps)
left out -- the order the projection cores read weights in.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gemma4_e2b_q4nx_requant as rq  # noqa: E402
import gemma4_e2b_q4nx_weights as gw  # noqa: E402
import validate_layer_npu as vln  # noqa: E402

UNI = 35
fd = vln._load_fd(UNI, 64, kv_src=[gw.kv_source_layer(i) for i in range(UNI)])
assert fd.PER_CLASS
K, BLK, NCX, NPH = fd.K, fd.BLOCK_BF16, fd.NCX, fd.NPH
GP, DP = fd.GLU_PHASE, fd.DOWN_PHASE
rng = np.random.default_rng(0)


def weights(L):
    dh, inter = gw.head_dim(L), fd.MODEL["INTER_NARROW"]
    if L >= fd.MODEL["FIRST_KV_SHARED"]:
        inter *= 2
    r = lambda *s: rng.standard_normal(s).astype(np.float32)  # noqa: E731
    w = dict(
        q=r(fd.NUM_Q_HEADS * dh, K),
        o=r(K, fd.NUM_Q_HEADS * dh),
        up=r(inter, K),
        gate=r(inter, K),
        down=r(K, inter),
    )
    if gw.owns_kv(L):
        w["k"], w["v"] = r(dh, K), r(dh, K)
    return w


def phases(slab, per_col):
    """slab -> per phase, per column: the two channel halves."""
    out, o = [], 0
    for p in range(NPH):
        n = per_col[p] * BLK
        cols = [slab[o + c * n : o + (c + 1) * n] for c in range(NCX)]
        out.append([np.split(c, 2) for c in cols])
        o += NCX * n
    return out


def padded(L, w):
    """The same layer packed at the build's full geometry."""
    w = dict(w)
    full = 2 * fd.MODEL["INTER_NARROW"]
    w["up"], w["gate"] = rq._pad_rows(w["up"], full), rq._pad_rows(w["gate"], full)
    w["down"] = rq._pad_cols(w["down"], full)
    if "k" not in w:
        dh = gw.head_dim(L)
        w["k"], w["v"] = np.zeros((dh, K), np.float32), np.zeros((dh, K), np.float32)
    saved = fd.PER_CLASS, fd.w_layer_of
    fd.PER_CLASS, fd.w_layer_of = False, lambda _l: fd.W_LAYER
    try:
        return rq.pack_layer_weights(fd, L, w)
    finally:
        fd.PER_CLASS, fd.w_layer_of = saved


print(f"W_DEC {fd.W_DEC} = sum of slabs: {fd.W_DEC == sum(fd.W_SLABS)}")
print(f"offsets cumulative: {fd.W_OFFS == list(np.cumsum([0] + fd.W_SLABS[:-1]))}")
for L in (0, 19):
    w = weights(L)
    native = rq.pack_layer_weights(fd, L, w)
    narrow = L < fd.MODEL["FIRST_KV_SHARED"]
    i2, j2 = (fd.I2P_N, fd.J2P_N) if narrow else (fd.I2P_W, fd.J2P_W)
    pc = fd.PER_COL_PH_N if narrow else fd.PER_COL_PH_W
    got, ref = phases(native, pc), phases(padded(L, w), fd.PER_COL_PH)
    ok = True
    for p in range(NPH):
        for c in range(NCX):
            for h in range(2):
                a = got[p][c][h].reshape(i2[p], 2 * j2[p], -1)
                b = ref[p][c][h].reshape(fd.I2P[p], 2 * fd.J2P[p], -1)
                ok &= np.array_equal(a, b[: i2[p], : 2 * j2[p]])
    kind = "narrow" if narrow else "kv-shared"
    print(f"layer {L} ({kind}): slab {native.size} == w_layer_of, rounds match: {ok}")
