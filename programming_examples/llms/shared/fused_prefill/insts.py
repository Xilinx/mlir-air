# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Edits to the insts (TXN control stream) of the fused prefill device.

strip_columns keeps only the ops on the given columns, so a dispatch drives one
herd group and the others never get a lock release.

LinearInsts covers an op whose insts depend on a few integer parameters, such
as attention's K-block count and offsets. The op is built at a base point and
once more per parameter at base + delta, and the insts at any point are then
base + sum(change * (value - base) / delta), which must come out whole.
"""

import struct

import numpy as np

_HDR = 16


def _ops(buf):
    """(opcode, column, offset, size) per op of a TXN v1.0 stream."""
    out, off = [], _HDR
    while off < len(buf):
        w0 = struct.unpack_from("<I", buf, off)[0]
        code = w0 & 0xFF
        if code == 0:  # write32: op, 0, addr, 0, value, size
            n, col = struct.unpack_from("<I", buf, off + 20)[0], buf[off + 11] >> 1
        elif code == 1:  # blockwrite: op, 0, addr, size
            n, col = struct.unpack_from("<I", buf, off + 12)[0], buf[off + 11] >> 1
        elif code == 3:  # maskwrite: op, 0, addr, 0, value, mask, size
            n, col = struct.unpack_from("<I", buf, off + 24)[0], buf[off + 11] >> 1
        elif code == 0x80:  # tct sync: op, size, col << 16 | row << 8 | dir, ...
            n, col = struct.unpack_from("<I", buf, off + 4)[0], buf[off + 10]
        elif code == 0x81:  # ddr patch: op, size, ..., BD register address at word 6
            n, col = struct.unpack_from("<I", buf, off + 4)[0], buf[off + 27] >> 1
        else:
            raise ValueError(f"unhandled TXN opcode {code:#x} at {off:#x}")
        if n <= 0 or off + n > len(buf):
            raise ValueError(f"bad TXN op size {n} at {off:#x}")
        out.append((code, col, off, n))
        off += n
    return out


def strip_columns(buf, cols):
    """The insts with only the ops on columns `cols`."""
    kept = [buf[off : off + n] for _, col, off, n in _ops(buf) if col in cols]
    body = b"".join(kept)
    hdr = bytearray(buf[:_HDR])
    struct.pack_into("<II", hdr, 8, len(kept), _HDR + len(body))
    return bytes(hdr) + body


class LinearInsts:
    def __init__(self, base, base_point, variants):
        """base: insts at base_point (dict name -> int); variants: name ->
        (insts with only that parameter moved, delta)."""
        self.base = np.frombuffer(base, np.uint32).astype(np.int64)
        self.point = dict(base_point)
        self.coef = {}
        for name, (buf, delta) in variants.items():
            w = np.frombuffer(buf, np.uint32).astype(np.int64)
            if w.shape != self.base.shape:
                raise ValueError(f"{name}: the insts change length with the parameter")
            d = w - self.base
            idx = np.nonzero(d)[0]
            self.coef[name] = (idx, d[idx], delta)

    def at(self, **point):
        w = self.base.copy()
        for name, v in point.items():
            idx, d, delta = self.coef[name]
            num = d * (v - self.point[name])
            if np.any(num % delta):
                raise ValueError(f"{name} = {v} is off the insts' lattice")
            w[idx] += num // delta
        if w.min() < 0 or w.max() >= 1 << 32:
            raise ValueError(f"insts word out of range at {point}")
        return w.astype(np.uint32)
