# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The decode builder and the weight packer agree on every layer's slab.

Weight-free: synthetic weights of each layer class's real shapes. The reference
is the full-geometry slab of the weights as the device computes with them
(a sliding layer's q/k/v rows compact, its o-proj columns at the front of each
head slot). A layer's slab must be exactly the blocks of it the projection
cores read for that layer's arm, in their order, and every block left out
must be zero weights.
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
DUAL = bool(getattr(fd, "W_DUAL_CHAN", 0))
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


def device_slab(L, w):
    """Layer L's weights in the device's layout, packed at the full geometry."""
    dh, DH, NQ, NKV = gw.head_dim(L), fd.DH_A, fd.NUM_Q_HEADS, fd.NUM_KV_HEADS
    swa = dh != DH
    head = (
        (lambda a, n, ax=0: a)
        if swa
        else (lambda a, n, ax=0: rq._pad_head(a, n, dh, DH, ax))
    )
    qkv = [head(w["q"], NQ)]
    if "k" in w:
        qkv += [head(np.tile(w[x], (NKV, 1)), NKV) for x in "kv"]
    qkv = rq._pad_rows(np.concatenate(qkv), fd.M)
    o = np.zeros((K, NQ * DH), np.float32)
    for h in range(NQ):  # [head | 0] per slot; identity on a full layer
        o[:, h * DH : h * DH + dh] = w["o"][:, h * dh : (h + 1) * dh]
    full = 2 * fd.MODEL["INTER_NARROW"]
    up, gate = rq._pad_rows(w["up"], full), rq._pad_rows(w["gate"], full)
    q = [None] * NPH
    q[0], q[fd.OPROJ_PHASE] = rq._requant_q4k(qkv, fd.GROUP), rq._requant_q4k(
        o, fd.GROUP
    )
    q[GP] = rq._interleave512(
        rq._requant_q4k(up, fd.GROUP), rq._requant_q4k(gate, fd.GROUP), fd.GLU_CHUNK
    )
    q[DP] = rq._requant_q4k(rq._pad_cols(w["down"], full), fd.GROUP)
    return np.concatenate(
        [
            fd.pack_q4k_cascade(*q[p], NCX, fd.NCY, iter_major=True, dual_chan=DUAL)
            for p in range(NPH)
        ]
    )


# one packed block of zero weights; every block of one is identical
_z = rq._requant_q4k(
    np.zeros((NCX * fd.NCY * fd.ROW_BLOCK, fd.COL_BLOCK), np.float32), fd.GROUP
)
ZERO_BLOCK = fd.pack_q4k_cascade(*_z, NCX, fd.NCY, iter_major=True)[:BLK]

print(f"W_DEC {fd.W_DEC} = sum of slabs: {fd.W_DEC == sum(fd.W_SLABS)}")
print(f"offsets cumulative: {fd.W_OFFS == list(np.cumsum([0] + fd.W_SLABS[:-1]))}")
for L in (0, 4, 15, 19):
    w = weights(L)
    native = rq.pack_layer_weights(fd, L, w)
    arm = fd.arm_of_layer(L)
    i2, j2, xs = fd.CLASS_GEOM[arm]
    got = phases(native, fd.class_per_col(arm))
    ref = phases(device_slab(L, w), fd.PER_COL_PH)
    ok = True
    for p in range(NPH):
        for c in range(NCX):
            for h in range(2):
                # [round][X block][this X block's weight blocks]
                a = got[p][c][h].reshape(i2[p], 2 * j2[p], -1)
                b = ref[p][c][h].reshape(fd.I2P[p], 2 * fd.J2P[p], -1)
                read = np.zeros(b.shape[:2], bool)
                read[: i2[p], : 2 * j2[p] * xs[p] : xs[p]] = True
                ok &= np.array_equal(a.reshape(-1, b.shape[2]), b[read])
                ok &= bool(np.all(b[~read].reshape(-1, BLK) == ZERO_BLOCK))
    print(
        f"layer {L} (arm {arm}): slab {native.size} == w_layer_of, blocks match: {ok}"
    )
