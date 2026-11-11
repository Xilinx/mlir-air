# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# The Qwen3-Coder MoE block's shapes and weight layout, on the Qwen3.6 tile map
# without its shared expert:
#
#   columns 0, 1, 6, 7  the Q4_K GEMV array: two turns of four routed experts,
#                 the expert of rank i on array column i
#   column 2      the array's assemble memtile
#   (5, 3)        the router GEMV into float32 logits
#   (5, 4)        router: the top 8 of 128 and the experts' weight transfers
#   (5, 5)        GLU: the routed GLUs and the score-weighted sum of the
#                 experts' down rows
#
# The block runs inside a layer's dispatch: the array's packet egress needs
# more than one consumer. An expert's up and gate rows are interleaved in
# 128-row slices, [row group][col block][4 row blocks][block].
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "kernel_tests"))
# llms/ is imported as the air_examples package, not through sys.path
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(HERE.parents[1])
]

import common  # noqa: E402
import qwen3_coder_30b_a3b_q4k_weights as W  # noqa: E402

R, RB, CB, BB = 4, 32, 256, W.Q4K_BYTES
D, I, E, K = W.D, W.I_EXP, W.N_EXPERTS, W.TOP_K
NCB_X, NCB_H = D // CB, I // CB  # col blocks of x and of a GLU output
SLICES = I // (R * RB)  # an expert's 128-row slices of up (or gate)
RG_GU, RG_DN = 2 * SLICES, D // (R * RB)
TURNS, NCOL = K // 4, 4
NRB_R = E // RB  # router row blocks
COLUMNS, ASM, MOE_COL = [0, 1, 6, 7], 2, 5
LOGITS, ROUTER, GLU_AT = (MOE_COL, 3), (MOE_COL, 4), (MOE_COL, 5)


def interleave(up, gate, n):
    """n experts' up / gate row blocks [n * I / 32, NCB, BB] -> [n, RG_GU row
    groups (up slice s, then gate slice s), NCB, 4, BB]."""
    u = up.reshape(n, SLICES, R, NCB_X, BB).transpose(0, 1, 3, 2, 4)
    g = gate.reshape(n, SLICES, R, NCB_X, BB).transpose(0, 1, 3, 2, 4)
    return np.stack([u, g], 2).reshape(n, RG_GU, NCB_X, R, BB)


def down(dn, n):
    """n experts' down row blocks [n * D / 32, NCB_H, BB] -> [n, RG_DN, NCB_H,
    4, BB]."""
    return dn.reshape(n, RG_DN, R, NCB_H, BB).transpose(0, 1, 3, 2, 4)


def router_blocks(router):
    """router [E, D] -> bf16 [NRB_R, NCB_X, 256 x 32 column-major]."""
    wr = router.reshape(NRB_R, RB, NCB_X, CB).transpose(0, 2, 3, 1)
    return np.ascontiguousarray(wr.astype(bfloat16)).reshape(NRB_R, NCB_X, CB * RB)


def weights(w, layer):
    m = lambda n: np.asarray(  # noqa: E731
        w.raw(W.name(layer, f"mlp.{n}_exps_proj.weight"))
    )
    wgu = np.ascontiguousarray(interleave(m("up"), m("gate"), E))
    wdn = np.ascontiguousarray(down(m("down"), E))
    return wgu, wdn, router_blocks(w.router(layer))


def bench(art, inputs, n):
    """Qwen3.6's: wall time of n dispatches, the weights already on the device."""
    spec = importlib.util.spec_from_file_location(
        "q36_moe_block", common.Q36 / "moe_block.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.bench(art, inputs, n)
