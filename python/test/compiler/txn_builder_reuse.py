# ./python/test/compiler/txn_builder_reuse.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s %t | FileCheck %s

"""TxnBuilder assembles a stream once when it fits the last one's buffer.

A runtime sequence with a run-time trip count gives streams of different
lengths from one builder. The builder is called once when the stream fits the
buffer the previous call of the same function left, and twice (size, then
fill) only when it grows. Every call returns its own words.

The header is a stand-in that counts its calls, so no toolchain is needed
beyond the host compiler.
"""

import sys
from pathlib import Path

import numpy as np

from air.backend.txn_builder import TxnBuilder

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
header = out / "fake.txn.h"
header.write_text("""#include <cstdint>
#include <optional>
#include <vector>
inline int calls = 0;
inline std::optional<std::vector<uint32_t>> generate_txn_seq(int32_t n, int32_t v) {
  ++calls;
  if (n < 0) return std::nullopt;
  return std::vector<uint32_t>(n, (uint32_t)v);
}
extern "C" int fake_calls() { return calls; }
""")
t = TxnBuilder(str(header), workdir=str(out / "work"))
calls = t._lib.fake_calls


def check(n, v, want_calls):
    before = calls()
    w = t("seq", n, v)
    assert w.dtype == np.uint32 and w.size == n and (w == v).all(), w
    assert calls() - before == want_calls, (n, calls() - before)
    return w


# CHECK: first 2
first = check(8, 1, 2)
print("first", 2)
# CHECK: fits 1
shorter = check(3, 2, 1)
same = check(8, 3, 1)
print("fits", 1)
# CHECK: grows 2
check(20, 4, 2)
check(20, 5, 1)
print("grows", 2)
# CHECK: own words
assert (first == 1).all() and (shorter == 2).all() and (same == 3).all()
print("own words")
