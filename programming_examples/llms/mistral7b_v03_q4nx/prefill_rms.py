# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# Adapted from the declared Granite Q4NX deployment example.
"""Row-parallel prefill wrapper for the existing Mistral decode RMS object."""

from pathlib import Path
from ml_dtypes import bfloat16


def build_module(m, n, np_dtype=bfloat16, vector_size=16, herd_x=8, *, object_path):
    from air import api as air
    from air.api import ops

    assert n == 4096 and m % herd_x == 0 and np_dtype is bfloat16
    x, weight, y = (
        air.tensor([m, n], air.bf16),
        air.tensor([n], air.bf16),
        air.tensor([m, n], air.bf16),
    )
    norm = air.extern(
        "rms_norm_aie", link_with=str(Path(object_path).resolve()), scalars=[air.i32]
    )
    with air.launch(name="weighted_rms_norm") as launch:

        @launch.body
        def _():
            with air.herd([range(herd_x)], shape=(herd_x,)) as herd:

                @herd.body
                def _(tx):
                    w = air.alloc([n], air.bf16, scope=herd.private())
                    inp = air.alloc([n], air.bf16, scope=herd.private())
                    out = air.alloc([n], air.bf16, scope=herd.private())
                    ops.load(w, weight[:])
                    for row in air.sequential(m // herd_x):
                        idx = tx * (m // herd_x) + row
                        ops.load(inp, x[idx : idx + 1, :])
                        norm(out, inp, w, 0)
                        ops.store(out, y[idx : idx + 1, :])

    return launch.build(target="npu2")
