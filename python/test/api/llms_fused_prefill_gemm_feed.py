# ./python/test/api/llms_fused_prefill_gemm_feed.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s %air_src_root | FileCheck %s

"""The fused prefill GEMM's weight feed (llms/shared/fused_prefill/device.py)
puts each B packet of a column on its own, so the packet's start is in the put's
offset and no stride grows with K. Checked on a down projection with a long K
(Qwen2.5-7B's, K=18944), where GNC packets span more than a shim BD stride can.
"""

import re
import sys
import types
from pathlib import Path

SRC = Path(sys.argv[1])
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(SRC / "programming_examples")
]
from air_examples.llms.shared.fused_prefill import device as D  # noqa: E402
from air_examples.llms.shared.fused_prefill.models import MODELS, spec  # noqa: E402

SHIM_MAX_STRIDE = 1 << 20  # 32-bit words
K, N = 18944, 3840
rounds = N // D.NR
L = D.k_pad(K) // D.TK * D.QB  # bytes of one column's packets for a round
cfg = spec(MODELS["llama31_8b_q4nx"]).CFG

PUT = re.compile(
    r"air\.channel\.put\s+@gBIn\[%c(\d+)\w*\]\s*\(%\w+\[([^\]]*)\]\s*\[([^\]]*)\]\s*\[([^\]]*)\]"
)
for pair in (0, 1):
    ir = str(
        D.build(cfg, "g", K, N, 0, 1, pair, name="g", own=True).build(target="npu2")
    )
    starts, max_stride = {}, 0
    for col, offs, sizes, strides in PUT.findall(ir):
        offs, sizes, strides = (
            [int(v) for v in t.split(",")] for t in (offs, sizes, strides)
        )
        starts.setdefault(int(col), []).append(
            sum(o * s for o, s in zip(offs, strides))
        )
        max_stride = max([max_stride] + [s for n, s in zip(sizes, strides) if n > 1])
    print(f"pair={pair} columns:", sorted(starts))
    print(
        f"pair={pair} packet starts:",
        all(
            starts[tx] == [(r * D.GNC + tx) * L for r in range(rounds)]
            for tx in range(D.GNC)
        ),
    )
    print(f"pair={pair} strides fit a shim BD:", max_stride // 4 <= SHIM_MAX_STRIDE)
# CHECK: pair=0 columns: [0, 1, 2, 3, 4, 5]
# CHECK: pair=0 packet starts: True
# CHECK: pair=0 strides fit a shim BD: True
# CHECK: pair=1 columns: [0, 1, 2, 3, 4, 5]
# CHECK: pair=1 packet starts: True
# CHECK: pair=1 strides fit a shim BD: True
