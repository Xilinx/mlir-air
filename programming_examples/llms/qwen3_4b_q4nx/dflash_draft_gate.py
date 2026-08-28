#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Device gate for the whole DFlash DRAFT pass: pre-pass + bidirectional decode.

Runs one block end to end on NPU2 --

    taps  --(24-launch pre-pass)-->  per-layer context K/V
          --(seed the drafter's KV)-->
    [known token, mask x (B-1)]  --(5-layer bidirectional decode)-->  B logit rows

-- against `dflash_draft_oracle.py`'s dump of the SAME block from
z-lab/Qwen3-4B-DFlash-b16 itself. This is the first point where the drafter half
of DFlash exists on the array rather than as pieces.

WHAT THE NUMBERS CAN AND CANNOT BE. The device drafter is int4-AWQ for the
pre-pass and q4k-cascade for the 5 decode layers; the oracle is bf16. Logit
agreement is therefore loose by construction, and the thing that actually
matters is whether the DRAFT TOKENS match -- a draft token the target rejects
costs a slot, and that cost is what section 3.1 priced. So:

  token agreement    reported per slot, and the headline. Slot 0 is the token
                     the target already committed and is NOT a prediction; only
                     slots 1..B-1 are.
  correlation        >= --corr, per slot. Catches a logit vector that is
                     structurally wrong but happens to keep its maximum.
  KV seeding         the pre-pass output is compared against the oracle's own
                     cache rows before the decode runs, so a draft failure can
                     be attributed to the decode rather than its input.
  block K/V          read back out of the device KV cache after the dispatch.
                     LAYER 0's is the structural gate -- see the comment where
                     it is checked. Later layers accumulate and are reported.

A DISAGREEMENT HERE IS NOT AUTOMATICALLY A BUG. It is the quantization cost on
the drafter, and measuring it is the point -- section 3.1's 1.24x priced a bf16
drafter's acceptance distribution. `--seed-oracle` splits that cost between the
pre-pass and the decode layers.

    python3 dflash_draft_oracle.py --block 0 --block-size 8   # once, torch
    python3 dflash_draft_gate.py
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

TARGET_SOURCE = os.environ.get("Q4NX_MODEL_SOURCE", "FastFlowLM/Qwen3-4B-NPU2")


def _rel(a, b):
    import numpy as np

    return np.sqrt(((a - b) ** 2).mean()) / max(np.sqrt((b**2).mean()), 1e-9)


def _cos(a, b):
    import numpy as np

    a, b = a.reshape(-1), b.reshape(-1)
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9))


def run_prepass(o, verbose=False):
    """The 24-launch pre-pass on the oracle's taps. -> (target_hidden, k, v)."""
    import numpy as np
    from ml_dtypes import bfloat16

    import dflash_ctxkv_int4_builder as CK
    import dflash_draft_prepass as PP
    import dflash_int4 as I
    import dflash_int4_fc_builder as FC
    from qwen3_4b_draft_weights import DraftWeights

    I.paths()
    I.compile_int4_gemm_kernel()
    from shared.infra.external_kernels import compile_rope

    compile_rope()

    ctx_pos = o["ctx_positions"]
    ctx = len(ctx_pos)
    N, C, KVD, HD = PP.N_LAYERS, PP.CTX_PAD, CK.KV_DIM, CK.HEAD_DIM
    P, KC = PP.N_CHUNKS, FC.FC_IN // PP.N_CHUNKS
    rows = C * CK.N_KV_HEADS
    assert ctx <= C, f"oracle ctx {ctx} exceeds CTX_PAD {C}"

    dw = DraftWeights(target_source=TARGET_SOURCE)
    fc_pk = []
    for W in FC.split_fc_weight(np.asarray(dw.fc()), P):
        q, s, z = I.awq_quantize(W)
        fc_pk.append(np.ascontiguousarray(I.pack_for_device(q, s, z, C, KC, PP.D)))
    kpk, vpk = [], []
    for L in range(N):
        kw, vw = CK.layer_kv_weights(dw, L)
        for w, pk_ in ((kw, kpk), (vw, vpk)):
            q, s, z = I.awq_quantize(w)
            pk_.append(np.ascontiguousarray(I.pack_for_device(q, s, z, C, PP.D, KVD)))
    kn = [np.asarray(dw.bf16(f"layers.{L}.self_attn.k_norm.weight")) for L in range(N)]
    hn_w = np.asarray(dw.hidden_norm(), bfloat16)

    taps = np.zeros((C, FC.FC_IN), bfloat16)
    taps[:ctx] = np.asarray(o["taps"], bfloat16)
    As = FC.split_taps(taps, P)
    positions = np.zeros(C, np.int64)
    positions[:ctx] = ctx_pos

    lay = PP.prepass_arg_layout()
    ins = [None] * lay["n_args"]
    for i, a in enumerate(lay["taps"]):
        ins[a] = As[i]
    for i, a in enumerate(lay["fc_w"]):
        ins[a] = fc_pk[i]
    for a in lay["fc_partial"] + lay["fc_fold"]:
        ins[a] = np.zeros((C, PP.D), bfloat16)
    ins[lay["hn_w"]] = hn_w
    ins[lay["target_hidden"]] = np.zeros((C, PP.D), bfloat16)
    for L in range(N):
        ins[lay["k_w"][L]] = kpk[L]
        ins[lay["v_w"][L]] = vpk[L]
        ins[lay["k_raw"][L]] = np.zeros((C, KVD), bfloat16)
        ins[lay["v_ctx"][L]] = np.zeros((C, KVD), bfloat16)
        ins[lay["k_norm_w"][L]] = np.asarray(kn[L], bfloat16)
        ins[lay["k_nrm"][L]] = np.zeros((rows, HD), bfloat16)
        ins[lay["k_ctx"][L]] = np.zeros((rows, HD), bfloat16)
    ins[lay["rope_lut"]] = CK.rope_lut(positions)

    import filelock
    from air.backend.xrt import XRTBackend

    backend = XRTBackend(
        verbose=verbose,
        omit_while_true_loop=False,
        output_format="elf",
        instance_name="dflash_draft_prepass",
        runtime_loop_tiling_sizes=[2, 2],
        stack_size=16384,
    )
    compiled = backend.compile(PP.build_prepass_module())
    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        fn = backend.load(compiled)
        res = fn(*ins)
    # The pre-pass ELF must be UNLOADED before the drafter's is loaded: two
    # resident device programs in one process is not something this backend
    # promises, and the drafter needs the array to itself.
    backend.unload()

    th = np.asarray(res[lay["target_hidden"]]).reshape(C, PP.D).astype(np.float32)
    k = np.stack(
        [
            np.asarray(res[lay["k_ctx"][L]]).reshape(C, KVD).astype(np.float32)
            for L in range(N)
        ]
    )
    v = np.stack(
        [
            np.asarray(res[lay["v_ctx"][L]]).reshape(C, KVD).astype(np.float32)
            for L in range(N)
        ]
    )
    return th, k, v


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--oracle", default="dflash_draft_oracle.npz")
    ap.add_argument("--stack", default="6080")
    ap.add_argument(
        "--seed-oracle",
        action="store_true",
        help="seed the drafter's KV with the ORACLE's bf16 context rows instead "
        "of the device pre-pass's. Attribution, not a gate: the difference "
        "between the two runs is what the pre-pass's int4 quantization costs "
        "the draft, and what is left is the decode layers' own q4k.",
    )
    ap.add_argument("--corr", type=float, default=0.90)
    ap.add_argument("--cos-kv", type=float, default=0.99, help="the seeded rows")
    ap.add_argument(
        "--cos-blk0",
        type=float,
        default=0.995,
        help="layer 0's block K/V -- the structural check (see below)",
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    import numpy as np

    import dflash_draft_decoder as DD

    o = np.load(args.oracle)
    ctx_pos = o["ctx_positions"]
    ctx = len(ctx_pos)
    blk = [int(t) for t in o["block_ids"]]
    B = len(blk)
    start = int(o["start"])
    print(
        f"[draft gate] block {int(o['block'])}: ctx {ctx} at {ctx_pos.tolist()}, "
        f"{B} block tokens at {start}..{start + B - 1}, mask={int(o['mask_token_id'])}",
        flush=True,
    )
    assert start == int(ctx_pos[-1]) + 1, (start, ctx_pos[-1])

    # ---- pre-pass -----------------------------------------------------------
    th, k_dev, v_dev = run_prepass(o, args.verbose)
    bad = 0
    for L in range(k_dev.shape[0]):
        ck = _cos(k_dev[L][:ctx], np.asarray(o["k_ctx"][L], np.float32))
        cv = _cos(v_dev[L][:ctx], np.asarray(o["v_ctx"][L], np.float32))
        ok = min(ck, cv) >= args.cos_kv
        bad += not ok
        print(
            f"  seed layer {L}: k cos {ck:.6f}, v cos {cv:.6f}"
            + ("" if ok else "   <-- FAIL")
        )

    # ---- drafter decode -----------------------------------------------------
    dec = DD.build_draft_decoder(TARGET_SOURCE, max_L=ctx + B + 1, stack=args.stack)
    if ctx + B >= dec.ATTN_MAXL:
        print(f"[draft gate] ctx+B={ctx+B} >= ATTN_MAXL={dec.ATTN_MAXL}; abort")
        return 1
    if args.seed_oracle:
        print("  [seeding from the ORACLE's bf16 rows, not the pre-pass]")
        DD.seed_context_kv(
            dec,
            np.asarray(o["k_ctx"], np.float32),
            np.asarray(o["v_ctx"], np.float32),
            ctx,
        )
    else:
        DD.seed_context_kv(dec, k_dev, v_dev, ctx)
    got = np.asarray(DD.draft_block(dec, blk, ctx), np.float32)

    # The block's OWN K/V, as the engine appended them.
    #
    # LAYER 0 IS THE STRUCTURAL CHECK AND THE ONLY ONE GATED TIGHTLY. Its block
    # K/V is a function of the mask-token embedding, input_layernorm, the
    # batch-8 k/v projection, k_norm, the RoPE angles at positions ctx..ctx+B-1,
    # and the append slot -- and of nothing that has been through a quantized
    # layer yet. A wrong slot or a wrong block position shows up here and
    # nowhere else, because five layers of attention have not mixed it yet.
    #
    # LAYERS 1.. ACCUMULATE BY CONSTRUCTION and are reported, not gated. Each
    # one's K/V comes from a hidden state that has been through q4k layers, so
    # the cosine falls monotonically with depth. Measured on this block:
    #     pre-pass seed:  k 0.9996 0.9886 0.9834 0.9609 0.9738
    #                     v 0.9964 0.9774 0.9595 0.9288 0.9284
    #     oracle bf16 seed: k 0.9996 0.9947 0.9928 0.9842 0.9848
    #                     v 0.9964 0.9896 0.9794 0.9664 0.9607
    # -- so roughly half the depth-4 drift is the pre-pass's int4 and the rest
    # is the decode layers' own q4k. The DRAFT TOKENS are identical either way,
    # which is the part that matters.
    kv = DD.read_block_kv(dec, ctx, B)
    print()
    for L in range(dec.UNI_DEC):
        ck = _cos(kv[0][L], np.asarray(o["k_blk"][L], np.float32))
        cv = _cos(kv[1][L], np.asarray(o["v_blk"][L], np.float32))
        gated = L == 0
        ok = (min(ck, cv) >= args.cos_blk0) if gated else True
        bad += not ok
        print(
            f"  block K/V layer {L}: k cos {ck:.6f}, v cos {cv:.6f}"
            + ("   [structural]" if gated else "   (accumulates)")
            + ("" if ok else "   <-- FAIL")
        )

    # Row t of the device output predicts slot t+1, so it lines up with the
    # oracle's draft_logits row t. Slot 0's row predicts slot 1 and IS used;
    # the oracle's own head takes hidden[1:], i.e. rows 1..B-1 -- the same
    # thing indexed from the other end. Compare rows 1..B-1 against the
    # oracle's 0..B-2, which is what `hidden[0, 1:, :]` means.
    ref = np.asarray(o["draft_logits"], np.float32)
    guess_o = [int(x) for x in o["draft_guess"]]
    true_next = [int(x) for x in o["true_next"]]
    guess_d = [int(got[t + 1].argmax()) for t in range(B - 1)]

    print(f"\n[draft gate] {B - 1} predicted slots (slot 0 is the committed token)")
    agree = 0
    for t in range(B - 1):
        c = _cos(got[t + 1], ref[t])
        r = _rel(got[t + 1], ref[t])
        same = guess_d[t] == guess_o[t]
        agree += same
        ok = c >= args.corr
        bad += not ok
        print(
            f"  slot {t + 1} (pos {start + t + 1}): device {guess_d[t]:>6} "
            f"oracle {guess_o[t]:>6} {'=' if same else 'X'}  true {true_next[t]:>6}"
            f"   cos {c:.6f}  rel {r:.3e}" + ("" if ok else "   <-- FAIL")
        )

    def prefix_acc(g):
        n = 0
        for a, b in zip(g, true_next):
            if a != b:
                break
            n += 1
        return n + 1  # + the committed token, i.e. `produced`

    print(
        f"\n  draft tokens matching the bf16 drafter: {agree}/{B - 1}\n"
        f"  produced (accepted prefix + 1): device {prefix_acc(guess_d)}, "
        f"oracle {prefix_acc(guess_o)}, of {B}"
    )
    print("\n" + ("PASS" if not bad else f"FAIL ({bad})"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
