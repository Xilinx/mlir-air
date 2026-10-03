# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""DecodeInstsGen on synthetic templates: L-linear words follow the two-build
slope, and with decode_L<N>.rb.insts.bin the readback lengths follow ceil(L/16).

Weight-free and device-free. Each stream has an append offset ((L-1)*512), an
RTP-L word (L) and two readback lengths (blocks * 8192, K and V).
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decode_insts_gen import DecodeInstsGen  # noqa: E402

N, MAXL = 64, 64


def stream(L, blocks):
    w = np.arange(N, dtype=np.uint32) * 7 + 3
    w[10], w[20] = (L - 1) * 512, L
    w[30] = w[31] = blocks * 8192
    return w


def write(d, name, words):
    words.tofile(d / f"{name}.insts.bin")


def expect(L, follow):
    return stream(L, (L + 15) // 16 if follow else MAXL // 16)


with tempfile.TemporaryDirectory() as td:
    d = Path(td)
    for L in (MAXL, MAXL - 1):
        write(d, f"decode_L{L}", stream(L, MAXL // 16))
        (d / f"decode_L{L}.xclbin").touch()
    g = DecodeInstsGen(str(d))
    ok = not g.exact and all(
        np.array_equal(g.insts_for(MAXL, L), expect(L, False)) for L in (1, 17, 64)
    )
    print(f"without .rb: readback stays at ATTN_MAXL: {ok}")

    write(d, f"decode_L{MAXL}.rb", stream(MAXL, MAXL // 16 - 1))
    g = DecodeInstsGen(str(d))
    Ls = (1, 16, 17, 32, 33, 48, 49, 63, 64)
    ok = g.exact and all(
        np.array_equal(g.insts_for(MAXL, L), expect(L, True)) for L in Ls
    )
    print(f"with .rb: readback follows ceil(L/16): {ok}")
