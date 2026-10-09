# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""RoPE reading from a column-slice of a wide fused-QKV GEMM output buffer,
instead of its own whole tensor -- same trick as o_ffn_fused_gu.py's
SwiGLU-from-wide (row-iterate, column-slice one wide buffer), applied to RoPE.

shared/builders/rms_gemms_rope_multi.py's _build_rope_2d walks the WHOLE input
as one flat 1D run, which only works because a plain Q-only or K-only buffer
is truly row-major-contiguous across all (seq, head) pairs. A fused QKV buffer
[seq_len, emb_dim+2*kv_dim] is NOT flattenable that way -- row r's Q columns
end, then K's start, then V's, then row r+1's Q starts -- so this walks
(seq_row, head) explicitly and reads each head_dim-wide chunk from a known
column OFFSET within that row of the wide buffer, instead of a flat walk.

New isolated file: does not touch shared/builders/rms_gemms_rope_multi.py.
"""

from __future__ import annotations
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# programming_examples/ is published as the air_examples package rather than
# put on sys.path: every directory under it would otherwise become a
# top-level module name and shadow an installed package that shares it.
import types

sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(_HERE.parent.parent.parent)
]

from ml_dtypes import bfloat16
from air import api as air
from air.api import ops
from air.api.types import i32
from air_examples.llms.shared.builders.rms_gemms_rope_multi import _api_dtype


def build_rope_from_wide(
    seq_len,
    wide_cols,
    col_offset,
    n_heads_here,
    head_dim,
    np_dtype,
    herd_x=8,
    target="npu2",
):
    """RoPE over `n_heads_here` heads of `head_dim` each, read from columns
    [col_offset : col_offset + n_heads_here*head_dim] of a [seq_len, wide_cols]
    input buffer. Output is its own normal [seq_len, n_heads_here*head_dim]
    buffer (RoPE's output never needs to be wide).

    LUT layout matches _build_rope_2d exactly: (seq_len*n_heads_here, head_dim),
    row-major (seq, head) order -- np.repeat(base_lut[:seq_len], n_heads_here,
    axis=0), same array callers already build for the unfused path.
    """
    assert head_dim % 16 == 0
    assert seq_len % herd_x == 0
    rows_per_tile = seq_len // herd_x
    out_cols = n_heads_here * head_dim

    dtype = _api_dtype(np_dtype)
    IN = air.tensor([seq_len, wide_cols], dtype)
    LUT = air.tensor([seq_len * n_heads_here * head_dim], dtype)
    OUT = air.tensor([seq_len, out_cols], dtype)

    rope = air.extern("rope", link_with="rope.o", scalars=[i32])

    with air.launch(name="rope_from_wide") as launch:

        @launch.body
        def _():
            with air.segment(name="rope_wide_seg") as seg:

                @seg.body
                def _():
                    with air.herd(
                        [range(herd_x), range(1)],
                        name="rope_wide_herd",
                        shape=(herd_x, 1),
                    ) as h:

                        @h.body
                        def _(tx, ty):
                            l1_in = air.alloc([head_dim], dtype, scope=h.private())
                            l1_lut = air.alloc([head_dim], dtype, scope=h.private())
                            l1_out = air.alloc([head_dim], dtype, scope=h.private())

                            for local_row in air.sequential(0, rows_per_tile):
                                row = local_row + tx * rows_per_tile
                                for head in air.sequential(0, n_heads_here):
                                    c0 = col_offset + head * head_dim
                                    lut_row = row * n_heads_here + head
                                    lut_off = lut_row * head_dim

                                    ops.load(l1_in, IN[row, c0 : c0 + head_dim])
                                    ops.load(l1_lut, LUT[lut_off : lut_off + head_dim])
                                    rope(l1_in, l1_lut, l1_out, head_dim)
                                    ops.store(
                                        l1_out,
                                        OUT[
                                            row,
                                            head * head_dim : head * head_dim
                                            + head_dim,
                                        ],
                                    )

    return launch.build(target=target)
