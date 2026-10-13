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
K, G, BLK, NCX, NPH = fd.K, fd.GROUP, fd.BLOCK_BF16, fd.NCX, fd.NPH
GP, DP = fd.GLU_PHASE, fd.DOWN_PHASE
DUAL = bool(getattr(fd, "W_DUAL_CHAN", 0))
rng = np.random.default_rng(0)


def weights(L):
    dh, inter = gw.head_dim(L), fd.MODEL["INTER_NARROW"]
    if L >= fd.MODEL["FIRST_KV_SHARED"]:
        inter *= 2

    def r(m, k):  # (q, scale, min)
        return (
            rng.integers(0, 16, (m, k), dtype=np.uint8),
            rng.standard_normal((m, k // G)).astype(np.float32),
            rng.standard_normal((m, k // G)).astype(np.float32),
        )

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
        (lambda t, n: t)
        if swa
        else (lambda t, n: rq._each(t, lambda a: rq._pad_head(a, n, dh, DH)))
    )
    qkv = [head(w["q"], NQ)]
    if "k" in w:
        qkv += [head(rq._each(w[x], lambda a: np.tile(a, (NKV, 1))), NKV) for x in "kv"]
    qkv = rq._each(rq._cat(*qkv), lambda a: rq._pad_rows(a, fd.M))

    def slots(a, g):  # [head | 0] per slot; identity on a full layer
        out = np.zeros((K, NQ * DH // g), a.dtype)
        for h in range(NQ):
            out[:, h * DH // g : (h * DH + dh) // g] = a[
                :, h * dh // g : (h + 1) * dh // g
            ]
        return out

    o = rq._each(w["o"], lambda a: slots(a, 1), lambda a: slots(a, G))
    full = 2 * fd.MODEL["INTER_NARROW"]
    up = rq._each(w["up"], lambda a: rq._pad_rows(a, full))
    gate = rq._each(w["gate"], lambda a: rq._pad_rows(a, full))
    q = [None] * NPH
    q[0], q[fd.OPROJ_PHASE] = qkv, o
    q[GP] = rq._interleave512(up, gate, fd.GLU_CHUNK)
    q[DP] = rq._each(
        w["down"],
        lambda a: rq._pad_cols(a, full),
        lambda a: rq._pad_cols(a, full // G),
    )
    return np.concatenate(
        [
            fd.pack_q4k_cascade(*q[p], NCX, fd.NCY, iter_major=True, dual_chan=DUAL)
            for p in range(NPH)
        ]
    )


# one packed block of zero weights; every block of one is identical
_rows = NCX * fd.NCY * fd.ROW_BLOCK
_z = (
    np.zeros((_rows, fd.COL_BLOCK), np.uint8),
    np.zeros((_rows, fd.COL_BLOCK // G), np.float32),
    np.zeros((_rows, fd.COL_BLOCK // G), np.float32),
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
