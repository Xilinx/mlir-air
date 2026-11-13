# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Several runs of the fused prefill device's ops in one dispatch.

An op's runtime sequence (TxnBuilder) is a TXN stream: a 4-word header (op
count in word 2, bytes in word 3) and its ops, among them a DDR_PATCH per
buffer descriptor that points at a kernel BO argument plus a byte offset
(aie_runtime TxnEncoding.h). Copies of a sequence one after another, each with
its patches' offsets moved, run the op once per copy on other parts of the
same BOs: one dispatch where there were as many as copies. Each copy ends on
its own syncs, so it runs after the one before it has finished, as separate
dispatches would.
"""

import numpy as np

WRITE, BLOCKWRITE, MASKWRITE, MASKPOLL, TCT, DDR_PATCH = 0, 1, 3, 4, 0x80, 0x81


def ops(words):
    """(word position, opcode, words) of each op of a TXN stream."""
    w = np.asarray(words, np.uint32)
    pos, out = 4, []
    for _ in range(int(w[2])):
        opc = int(w[pos]) & 0xFF
        if opc == WRITE:
            n = w[pos + 5]
        elif opc in (MASKWRITE, MASKPOLL):
            n = w[pos + 6]
        elif opc == BLOCKWRITE:
            n = w[pos + 3]
        elif opc in (TCT, DDR_PATCH):
            n = w[pos + 1]
        else:
            raise ValueError(f"TXN opcode {opc:#x} at word {pos}")
        out.append((pos, opc, int(n) // 4))
        pos += int(n) // 4
    if pos != w.size or pos * 4 != int(w[3]):
        raise ValueError(f"TXN stream of {w.size} words ends at word {pos}")
    return out


class Copies:
    """An op's TXN stream, to be repeated with each copy's offsets added to
    its DDR patches of given arguments."""

    def __init__(self, words):
        w = np.asarray(words, np.uint32)
        self.head, self.body, self.nops = w[:4].copy(), w[4:].copy(), int(w[2])
        p = [pos - 4 for pos, opc, _ in ops(w) if opc == DDR_PATCH]
        self.pos = np.asarray(p, np.int64)
        b = self.body
        self.arg = b[self.pos + 8].astype(np.int64)
        self.plus = b[self.pos + 10].astype(np.uint64) | (
            b[self.pos + 11].astype(np.uint64) << np.uint64(32)
        )

    def words(self, n):
        return 4 + n * self.body.size


def concat(runs, out=None):
    """One TXN stream for runs [(Copies, offsets int64 [copies, args])]: each
    entry's copies in turn, copy i's patch of argument a moved by
    offsets[i, a] bytes. Writes into `out` (uint32, at least as long) when
    given; returns the stream."""
    runs = [(c, np.asarray(o, np.int64)) for c, o in runs if len(o)]
    n = sum(c.body.size * len(o) for c, o in runs)
    w = np.empty(4 + n, np.uint32) if out is None else out[: 4 + n]
    w[:4] = runs[0][0].head
    w[2] = sum(c.nops * len(o) for c, o in runs)
    w[3] = (4 + n) * 4
    at = 4
    for c, o in runs:
        k, m = len(o), c.body.size
        seg = w[at : at + k * m].reshape(k, m)
        seg[:] = c.body
        plus = c.plus[None, :] + o[:, c.arg].astype(np.uint64)
        seg[:, c.pos + 10] = (plus & np.uint64(0xFFFFFFFF)).astype(np.uint32)
        seg[:, c.pos + 11] = (plus >> np.uint64(32)).astype(np.uint32)
        at += k * m
    return w
