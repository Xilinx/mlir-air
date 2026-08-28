# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# The DFlash drafter's whole non-decode half, as ONE 24-launch AIR function:
#
#     taps [CTX, 12800] --fc--> --hidden_norm--> target_hidden [CTX, 2560]
#                       --k_proj_L--> --k_norm_L--> --RoPE--> k_ctx_L
#                       --v_proj_L-->                         v_ctx_L
#
# This is everything the drafter needs that the decode engine cannot do, and it
# runs ONCE per block: dflash_draft_decomp.py measures that `target_hidden` is
# layer-invariant (exactly 0.0 across all five layers), so the context K/V are a
# function of the taps alone and can be written into the KV cache before the
# layer loop ever starts. After this pass the drafter is an ordinary
# bidirectional decode (section 3.2).
#
# WHY ONE FUNCTION AND NOT TWO ELFs. `target_hidden` is [32, 2560] = 160 KB and
# is consumed by ten launches. Dispatching fc and the context K/V separately
# means reading it back to the host and writing it out again, plus a second
# ELF load, for a value no host code looks at. Stitched, it stays a device
# buffer and the whole pre-pass is one dispatch.
#
# The two halves keep their own gates -- dflash_int4_fc_gate.py and
# dflash_ctxkv_int4_gate.py -- because a failure in the combined form is much
# cheaper to localise when each half has already been checked alone. This module
# adds only the wiring, and dflash_draft_prepass_gate.py checks exactly that:
# that fc's output is what the K/V launches actually read.

import dflash_ctxkv_int4_builder as CK
import dflash_int4_fc_builder as FC

CTX_PAD = FC.CTX_PAD
D = FC.D
N_LAYERS = CK.N_LAYERS
N_CHUNKS = FC.N_CHUNKS


def build_prepass_module(ctx_pad=CTX_PAD, n_chunks=N_CHUNKS, n_layers=N_LAYERS):
    """fc + hidden_norm + per-layer context K/V + k_norm + RoPE, in one func.

    Args are fc's (dflash_int4_fc_builder.fc_parts) followed by the context
    K/V's (dflash_ctxkv_int4_builder.ctxkv_parts) with fc's output substituted
    for `target_hidden`, so the K/V side declares one fewer argument than it
    does standalone. `prepass_arg_layout` reports the resulting indices rather
    than leaving callers to recompute them.
    """
    import dflash_int4 as I

    I.paths()
    from shared.infra.stitching import stitch_elf

    fc_args, fc_slices, th = FC.fc_parts(ctx_pad, n_chunks, with_norm=True)
    kv_args, kv_slices, prelude = CK.ctxkv_parts(
        ctx_pad,
        n_layers,
        with_knorm=True,
        with_rope=True,
        base=th + 1,
        th_arg=th,
    )
    return stitch_elf(
        "dflash_draft_prepass",
        fc_args + kv_args,
        fc_slices + kv_slices,
        prelude=prelude,
    )


def prepass_arg_layout(n_chunks=N_CHUNKS, n_layers=N_LAYERS):
    """Named arg indices for the combined func, so callers do not re-derive them.

    The offsets are a chain of three variable-length blocks; recomputing them at
    each call site is exactly the kind of thing that produces a module that
    compiles, runs, and returns another layer's answer.
    """
    p = n_chunks
    th = 4 * p  # fc's hidden_norm output = target_hidden
    off = th + 1
    nb = off + 4 * n_layers
    rb = nb + 2 * n_layers
    return {
        "taps": list(range(p)),  # A_i   [ctx_pad, FC_IN // p]
        "fc_w": list(range(p, 2 * p)),  # B_i   packed int4
        "fc_partial": list(range(2 * p, 3 * p)),  # P_i   [ctx_pad, 2560]
        "fc_fold": list(range(3 * p, 4 * p - 1)),  # S_j   [ctx_pad, 2560]
        "hn_w": 4 * p - 1,  # [2560]
        "target_hidden": th,  # [ctx_pad, 2560]
        "k_w": [off + 4 * L for L in range(n_layers)],
        "k_raw": [off + 4 * L + 1 for L in range(n_layers)],
        "v_w": [off + 4 * L + 2 for L in range(n_layers)],
        "v_ctx": [off + 4 * L + 3 for L in range(n_layers)],
        "k_norm_w": [nb + 2 * L for L in range(n_layers)],
        "k_nrm": [nb + 2 * L + 1 for L in range(n_layers)],
        "rope_lut": rb,
        "k_ctx": [rb + 1 + L for L in range(n_layers)],
        "n_args": rb + 1 + n_layers,
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    m = build_prepass_module()
    txt = str(m)
    n = txt.count("air.launch")
    lay = prepass_arg_layout()
    print(f"[prepass] {len(txt.splitlines())} lines, {n} air.launch ops, parsed OK")
    print(
        f"[prepass] {N_CHUNKS} fc GEMM + {N_CHUNKS - 1} add + norm + "
        f"{N_LAYERS} x (2 GEMM + k_norm + RoPE) = {n}, {lay['n_args']} args"
    )
    sys.exit(0 if n == 2 * N_CHUNKS + 4 * N_LAYERS else 1)
