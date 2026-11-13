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


# a core tile's row in a TXN register address, and its lock registers
_ROW, _LOCKS = lambda a: (a >> 20) & 0x1F, 0x1F000


class _Form:
    """A TXN stream's ops (no header) and its DDR patches."""

    def __init__(self, w, keep):
        body = [w[p : p + n] for p, opc, n in ops(w) if keep(w, p, opc)]
        self.nops = len(body)
        self.body = np.concatenate(body).astype(np.uint32)
        pos, at = [], 0
        for b in body:
            if int(b[0]) & 0xFF == DDR_PATCH:
                pos.append(at)
            at += b.size
        self.pos = np.asarray(pos, np.int64)
        self.arg = self.body[self.pos + 8].astype(np.int64)
        self.plus = self.body[self.pos + 10].astype(np.uint64) | (
            self.body[self.pos + 11].astype(np.uint64) << np.uint64(32)
        )


class Copies:
    """An op's TXN stream, to be repeated with each copy's offsets added to
    its DDR patches of given arguments. A copy that follows another of the
    same op skips the writes to the cores' data memory, the RTPs, which hold
    what the copy before wrote; it still writes their locks, which start the
    cores on it. The firmware is slow at those writes: on the fused prefill's
    GEMMs they were a sixth of an op's time."""

    def __init__(self, words):
        w = np.asarray(words, np.uint32)
        self.head = w[:4].copy()

        def repeat(w, p, opc):
            if opc != WRITE:
                return True
            a = int(w[p + 2])
            return _ROW(a) < 2 or (a & 0xFFFFF) >= _LOCKS

        self.first = _Form(w, lambda w, p, opc: True)
        self.rest = _Form(w, repeat)


def size(runs):
    """Words of concat(runs)."""
    return 4 + sum(
        c.first.body.size + (len(o) - 1) * c.rest.body.size for c, o in runs if len(o)
    )


def _put(w, at, f, o):
    """Copies of form f at w[at:], one per row of offsets o; the end."""
    k, m = len(o), f.body.size
    seg = w[at : at + k * m].reshape(k, m)
    seg[:] = f.body
    plus = f.plus[None, :] + o[:, f.arg].astype(np.uint64)
    seg[:, f.pos + 10] = (plus & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    seg[:, f.pos + 11] = (plus >> np.uint64(32)).astype(np.uint32)
    return at + k * m


def concat(runs, out=None):
    """One TXN stream for runs [(Copies, offsets int64 [copies, args])]: each
    entry's copies in turn, copy i's patch of argument a moved by
    offsets[i, a] bytes. Writes into `out` (uint32, at least as long) when
    given; returns the stream."""
    runs = [(c, np.asarray(o, np.int64)) for c, o in runs if len(o)]
    n = size(runs)
    w = np.empty(n, np.uint32) if out is None else out[:n]
    w[:4] = runs[0][0].head
    w[2] = sum(c.first.nops + (len(o) - 1) * c.rest.nops for c, o in runs)
    w[3] = n * 4
    at = 4
    for c, o in runs:
        at = _put(w, at, c.first, o[:1])
        at = _put(w, at, c.rest, o[1:])
    return w
