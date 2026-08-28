# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Fused-decode numerics gate for LFM2-1.2B -- all 16 layers, ONE dispatch.
#
# Runs the whole decode on the NPU with real Q4_0 weights (both layer types
# interleaved on LFM2's irregular 6-of-16 schedule) and scores the logits
# against a NumPy reference of the identical computation. `make verify` is the
# end-to-end correctness gate (top-k token-set inclusion, prefill + decode, vs
# HF bf16); this one isolates the decode, needs no prefill, and localises a
# failure to a layer -- so it is the first thing to run after touching the
# fused-decode builder or its kernels.
#
# ## How the attention score scale is taken out of the question
#
#   The device's flash attention normalises by its own softmax denominator, and
#   the score scale (1/sqrt(DH), wherever it is folded) is not recoverable from
#   the weight packer. Rather than guess it, this gate uploads ZERO q-norm
#   weights: QK-norm scales q by that weight, so q becomes 0, every score
#   becomes 0 whatever the scale, the softmax over the ATTN_L cache slots is
#   uniform, and -- because the slots this token did not write hold V = 0 -- the
#   attention output is exactly V_thistoken / ATTN_L.
#
#   Consequence: RoPE, QK-norm and the score path are NOT covered here (they are
#   covered by `make verify`). What IS covered, and is covered nowhere else:
#
#     - the two layer types INTERLEAVED on the real schedule, in one binary,
#       selected per wave by the arm;
#     - every ShortConv layer's gate, taps and carried state at full depth,
#       including the state carried across a real token sequence (--seq) and
#       read back from the device;
#     - QKV and o-proj packing on attention layers, the GLU interleave, SwiGLU,
#       both RMSNorms per layer, the final norm and the tied LM head;
#     - that a ShortConv layer does not corrupt an attention layer's residual
#       stream, or the reverse.
#
# ## Read the DEPTH caveat before reading a number
#
#   bf16 on device against an f32 NumPy reference compounds over residual
#   layers, so the cosine falls with depth for reasons that are not defects.
#   Measured on a KNOWN-GOOD build: 0.9962 at 1 layer, 0.8982 at 6, 0.7823 at
#   10 for a ShortConv-only stack. The full 16-layer hybrid scores 0.9138 --
#   BETTER than that partial stack at 6 layers, because a complete model is far
#   better conditioned than a truncated one.
#
#   So: judge exactness on a ONE-layer build (DECODE_UNI_DEC=1, expect >= 0.996),
#   and judge the whole model on cosine plus top-1 agreement. The carried-state
#   check is depth-independent by construction -- it bounds what ONE layer adds,
#   not the total -- so it means the same thing at either depth.
#
# Usage:
#   flock -x -w 3000 /tmp/mlir-air-npu.lock \
#     python3 lfm2_1_2b_q4nx_decode_gate.py --seq
#   flock -x -w 3000 /tmp/mlir-air-npu.lock \
#     python3 lfm2_1_2b_q4nx_decode_gate.py --seq --state-verbose
import argparse
import glob
import os
import re
import sys

import numpy as np
from ml_dtypes import bfloat16

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEC = os.path.abspath(os.path.join(_HERE, "..", "..", "fused_decode"))
for _p in (_HERE, _DEC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

EPS = 1e-5
N_LAYERS = 16
ATTN_IDXS = (2, 5, 8, 10, 12, 14)
K_DIM, CONV_DIM, TAPS = 2048, 2048, 3
N_HEADS, N_KV, DH = 32, 8, 64


def sig_sizes(mlir_path):
    with open(mlir_path) as f:
        for line in f:
            if "func.func @q4nx_decode" in line:
                return [int(n) for n in re.findall(r"memref<(\d+)xbf16>", line)]
    raise SystemExit(f"no @q4nx_decode signature in {mlir_path}")


def _template(d, lbuild):
    """Locate the built decode template: (xclbin, insts, air.mlir).

    `make compile-decode` writes decode_L<ATTN_MAXL>.{xclbin,insts.bin} plus the
    air.mlir it was emitted from. A raw builder run leaves decode.xclbin; accept
    that too so an ad-hoc build can be scored without renaming it.
    """
    if lbuild:
        cand = [os.path.join(d, f"decode_L{lbuild}")]
    else:
        cand = sorted(
            os.path.join(d, os.path.basename(x)[: -len(".xclbin")])
            for x in glob.glob(os.path.join(d, "decode_L*.xclbin"))
        )
    cand.append(os.path.join(d, "decode"))
    for c in cand:
        if os.path.exists(c + ".xclbin") and os.path.exists(c + ".insts.bin"):
            mlir = c + ".air.mlir"
            if not os.path.exists(mlir):
                mlir = os.path.join(d, "air.mlir")
            if not os.path.exists(mlir):
                raise SystemExit(
                    f"{c}.xclbin has no matching air.mlir -- the BO sizes are read "
                    f"from the compiled signature so they cannot drift from the build"
                )
            return c + ".xclbin", c + ".insts.bin", mlir
    raise SystemExit(
        f"no decode template in {d} -- run `make compile-decode` first "
        f"(looked for {', '.join(os.path.basename(c) + '.xclbin' for c in cand)})"
    )


def rms(x, w, eps=EPS):
    return (x / np.sqrt((x * x).mean() + eps)) * w


def reference_logits(tok, hf, q4, attn_l, state=None, n_layers=N_LAYERS):
    """NumPy forward through all 16 layers, both types, in execution order.

    `state` is float32 [N_LAYERS, 2, CONV_DIM] -- indexed by MODEL layer so it
    lines up with the device's arg4 slab, which gives every layer a slot
    whether or not it is a ShortConv layer. Updated in place, as the kernel
    does. None = zero state (first token of a sequence).
    """
    x = hf.bf16("model.embed_tokens.weight")[tok].astype(np.float32)
    for li in range(n_layers):
        p = f"model.layers.{li}"
        h = rms(x, hf.bf16(f"{p}.operator_norm.weight"))
        if li in ATTN_IDXS:
            # q-norm is uploaded as ZERO, so every score is 0, the softmax over
            # the attn_l cache slots is uniform, and the 15 slots this token did
            # not write hold V = 0. Attention output is V / attn_l exactly.
            v = h @ q4(f"{p}.self_attn.v_proj.weight")
            vh = v.reshape(N_KV, DH) / float(attn_l)
            ao = np.repeat(vh, N_HEADS // N_KV, axis=0).reshape(N_HEADS * DH)
            x = x + ao @ q4(f"{p}.self_attn.out_proj.weight")
        else:
            bcx = h @ q4(f"{p}.conv.in_proj.weight")
            B = bcx[:CONV_DIM]
            C = bcx[CONV_DIM : 2 * CONV_DIM]
            X = bcx[2 * CONV_DIM :]
            bx = B * X
            w = (
                hf.bf16(f"{p}.conv.conv.weight")
                .reshape(CONV_DIM, TAPS)
                .astype(np.float32)
            )
            s0, s1 = (
                (np.zeros(CONV_DIM, np.float32), np.zeros(CONV_DIM, np.float32))
                if state is None
                else (state[li, 0], state[li, 1])
            )
            # taps are oldest-first: y = w0*BX[t-2] + w1*BX[t-1] + w2*BX[t]
            conv = w[:, 0] * s0 + w[:, 1] * s1 + w[:, 2] * bx
            if state is not None:
                state[li, 0] = s1
                # DDR holds bf16; that is hardware, not a modelling choice.
                state[li, 1] = bx.astype(bfloat16).astype(np.float32)
            y = C * conv
            x = x + y @ q4(f"{p}.conv.out_proj.weight")

        h = rms(x, hf.bf16(f"{p}.ffn_norm.weight"))
        g = h @ q4(f"{p}.feed_forward.w1.weight")
        u = h @ q4(f"{p}.feed_forward.w3.weight")
        x = x + (g / (1.0 + np.exp(-g)) * u) @ q4(f"{p}.feed_forward.w2.weight")
    x = rms(x, hf.bf16("model.embedding_norm.weight"))
    return x @ q4("model.embed_tokens.weight")  # tied head


def main():
    ap = argparse.ArgumentParser(description="LFM2 HYBRID (16-layer) numerics gate")
    ap.add_argument(
        "--dir",
        default=".",
        help="directory holding the built decode template (default: this one, "
        "which is where `make compile-decode` puts decode_L<ATTN_MAXL>.*)",
    )
    ap.add_argument(
        "--lbuild",
        type=int,
        default=0,
        help="ATTN_MAXL of the template to score (0 = pick the only one present)",
    )
    ap.add_argument("--tokens", default="1000,42,7,15000,65000")
    ap.add_argument(
        "--cache", default=os.path.join(_HERE, "build_peano", "lfm2_all_full.npz")
    )
    ap.add_argument("--cos-min", type=float, default=0.90)
    ap.add_argument("--timeout-ms", type=int, default=60000)
    ap.add_argument(
        "--seq",
        action="store_true",
        help="treat --tokens as ONE sequence: carry the ShortConv state on both "
        "sides and check the device's state read-back",
    )
    ap.add_argument(
        "--state-rel",
        type=float,
        default=0.05,
        help="max relative-L2 drift ONE MODEL LAYER may add to the carried "
        "ShortConv state. Bounds the increment, not the total, so the same "
        "value is meaningful at any depth (see the use site).",
    )
    ap.add_argument(
        "--state-verbose",
        action="store_true",
        help="break the state read-back down per layer, in depth order",
    )
    ap.add_argument("--dump", default="")
    ap.add_argument(
        "--layers",
        type=int,
        default=0,
        help="cross-check only: assert the build streams exactly N layers. The "
        "count is DERIVED from the compiled weight-BO size, so this never needs "
        "to be passed. Short builds (DECODE_UNI_DEC=N) are the bisect -- the "
        "weight, rms and state slabs are all per-layer prefixes, so --layers 1 "
        "is ShortConv only and --layers 3 adds the first attention layer "
        "(LFM2 is 0,1=conv, 2=attn).",
    )
    a = ap.parse_args()

    from q4_0_codec import HFModel, requant_q4_0  # noqa: E402
    from lfm2_requant import HF_REPO  # noqa: E402
    from lfm2_1_2b_q4nx_weights import dequant_q4_0  # noqa: E402

    if not os.path.exists(a.cache):
        raise SystemExit(
            f"missing {a.cache} -- pack it with build_requant_cache(layer_kind='all')"
        )
    z = np.load(a.cache)
    hf = HFModel(HF_REPO)

    _memo = {}

    def q4(name):
        if name not in _memo:
            q, sc = requant_q4_0(hf.bf16(name))
            _memo[name] = np.ascontiguousarray(dequant_q4_0(q, sc).T)
        return _memo[name]

    d = a.dir if os.path.isabs(a.dir) else os.path.join(_HERE, a.dir)
    xclbin_p, insts_p, mlir_p = _template(d, a.lbuild)
    n_x, n_w, n_r, n_y, n_kv = sig_sizes(mlir_p)
    insts = np.fromfile(insts_p, dtype=np.uint32)

    # DERIVE the layer count from the compiled weight-BO size rather than taking
    # it on trust: the reference has to walk exactly the layers the build runs,
    # and a mismatch is otherwise a wall of confusing cosines (or, if the sizes
    # happen to line up, a silently wrong answer). --layers overrides only to
    # cross-check, and is asserted against the derived value.
    _w_layer = int(z["W"].shape[1])
    _w_vocab = int(z["WV"].size)
    NL, _rem = divmod(n_w - _w_vocab, _w_layer)
    if _rem or not 0 < NL <= N_LAYERS:
        raise SystemExit(
            f"cannot derive the layer count: weight BO is {n_w} bf16, which is "
            f"not {_w_vocab} (lm head) + N x {_w_layer} (per layer). Is the "
            f"weight cache from a different model or a different builder?"
        )
    if a.layers and a.layers != NL:
        raise SystemExit(
            f"--layers {a.layers} does not match this build, which streams {NL} "
            f"layers. Rebuild with DECODE_UNI_DEC={a.layers} or drop --layers."
        )

    # rms slab: [16 x (operator_norm | ffn_norm)] [16 x rope_w] [final norm].
    # rope_w is per layer and already in device layout in the cache: taps
    # tap-major for a ShortConv layer, [cos|sin | q_norm | k_norm] for an
    # attention one. Derive its width from the signature rather than assuming.
    ROPE_W_LEN = (n_r - NL * 2 * K_DIM - K_DIM) // NL
    assert ROPE_W_LEN * NL + NL * 2 * K_DIM + K_DIM == n_r, "rms layout"
    # arg4 = [16 KV-cache slabs][16 ShortConv state slabs]; ATTN_L falls out.
    CONV_ST_LAYER = 2 * CONV_DIM
    KV_LAYER = (n_kv - NL * CONV_ST_LAYER) // NL
    ATTN_L = KV_LAYER // (2 * N_KV * DH)
    CONV_ST_BASE = NL * KV_LAYER
    assert KV_LAYER * NL + CONV_ST_LAYER * NL == n_kv, "arg4 layout"
    print(
        f"{a.dir}: layers={NL}  ATTN_L={ATTN_L}  ROPE_W_LEN={ROPE_W_LEN}  "
        f"W={n_w}  attention output modelled as V/{ATTN_L}"
    )

    W = np.concatenate([z["W"][:NL].reshape(-1), z["WV"].reshape(-1)])
    assert W.size == n_w, (W.size, n_w)

    r = np.zeros(n_r, np.uint16)
    for i in range(NL):
        r[i * 2 * K_DIM : i * 2 * K_DIM + K_DIM] = z["RMS_in"][i].view(np.uint16)
        r[i * 2 * K_DIM + K_DIM : (i + 1) * 2 * K_DIM] = z["RMS_post"][i].view(
            np.uint16
        )
    rope_base = NL * 2 * K_DIM
    rw = z["ROPE_W"].view(bfloat16).astype(np.float32).reshape(N_LAYERS, ROPE_W_LEN)
    for i in range(NL):
        slab = rw[i].copy()
        if i in ATTN_IDXS:
            # ZERO the q-norm weight -- this is what makes the scale drop out.
            # cos/sin ([0:DH]) stay zero: with q == 0 they cannot matter, and
            # this gate deliberately does not claim to cover RoPE.
            slab[DH : 2 * DH] = 0.0
        b = rope_base + i * ROPE_W_LEN
        r[b : b + ROPE_W_LEN] = slab.astype(bfloat16).view(np.uint16)
    r[rope_base + NL * ROPE_W_LEN :] = z["NORM"].view(np.uint16)

    import pyxrt as xrt

    dev = xrt.device(0)
    xb = xrt.xclbin(xclbin_p)
    dev.register_xclbin(xb)
    ctx = xrt.hw_context(dev, xb.get_uuid())
    krn = xrt.kernel(ctx, [k.get_name() for k in xb.get_kernels()][0])

    HO = xrt.bo.host_only
    bo_i = xrt.bo(dev, insts.nbytes, xrt.bo.cacheable, krn.group_id(1))
    bo_x = xrt.bo(dev, n_x * 2, HO, krn.group_id(3))
    bo_w = xrt.bo(dev, n_w * 2, HO, krn.group_id(4))
    bo_r = xrt.bo(dev, n_r * 2, HO, krn.group_id(5))
    bo_y = xrt.bo(dev, n_y * 2, HO, krn.group_id(6))
    bo_kv = xrt.bo(dev, n_kv * 2, HO, krn.group_id(7))

    DECODE_Y = n_y - 65536  # logits live at the TAIL of y
    toks = [int(t) for t in a.tokens.split(",")]
    kv_host = np.zeros(n_kv, np.uint16)
    ref_state = np.zeros((NL, 2, CONV_DIM), np.float32) if a.seq else None
    n_ok, worst, dumps = 0, 1.0, {}
    worst_rel, n_st, n_st_bad = 0.0, 0, 0

    for tok in toks:
        x0 = hf.bf16("model.embed_tokens.weight")[tok].astype(bfloat16).view(np.uint16)
        bo_i.write(insts.tobytes(), 0)
        bo_i.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        for bo, arr in (
            (bo_x, x0),
            (bo_w, W.view(np.uint16)),
            (bo_r, r),
            (bo_kv, kv_host),
        ):
            bo.write(np.ascontiguousarray(arr).tobytes(), 0)
            bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        # Poison y: an all-zero read must not be mistaken for a result.
        bo_y.write(np.full(n_y, 0xAB, np.uint16).tobytes(), 0)
        bo_y.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        run = xrt.run(krn)
        for i, v in enumerate((3, bo_i, len(insts), bo_x, bo_w, bo_r, bo_y, bo_kv)):
            run.set_arg(i, v)
        run.start()
        st = run.wait(a.timeout_ms)
        if "COMPLETED" not in str(st).upper():
            print(f"tok {tok}: {st}")
            return 1
        bo_y.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        # XRT hands back a BO VIEW -- copy before the next iteration overwrites.
        y = np.frombuffer(bo_y.read(n_y * 2, 0), np.uint16).copy()
        dev_logits = y[DECODE_Y:].view(bfloat16).astype(np.float32)
        dumps[str(tok)] = dev_logits
        # Say WHY a comparison is meaningless before reporting a cosine for it.
        # 0xAB is the poison pattern written above, so a poisoned word is one
        # the kernel never wrote; NaN is something the kernel did write.
        n_poison = int((y[DECODE_Y:] == 0xAB).sum())
        n_nan = int(np.isnan(dev_logits).sum())
        if n_poison or n_nan:
            print(
                f"tok {tok:6d}: UNUSABLE -- {n_poison} of {dev_logits.size} logit "
                f"words never written (poison), {n_nan} NaN"
            )

        ref = reference_logits(tok, hf, q4, ATTN_L, ref_state, NL)
        cos = float(
            dev_logits
            @ ref
            / (np.linalg.norm(dev_logits) * np.linalg.norm(ref) + 1e-30)
        )
        # min() against NaN returns the FIRST argument in Python, so a NaN
        # cosine would silently leave `worst` at 1.0 and the summary would
        # report a perfect score for a run that produced nothing.
        if np.isnan(cos):
            worst = float("nan")
        elif not np.isnan(worst):
            worst = min(worst, cos)
        n_ok += bool(cos >= a.cos_min)
        print(
            f"tok {tok:6d}: cos={cos:.6f}  dev_top1={int(dev_logits.argmax()):6d}"
            f"  ref_top1={int(ref.argmax()):6d}"
            f"  top1={'y' if dev_logits.argmax() == ref.argmax() else 'n'}"
        )

        if a.seq:
            bo_kv.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            kv_host = np.frombuffer(bo_kv.read(n_kv * 2, 0), np.uint16).copy()
            st_all = (kv_host[CONV_ST_BASE:].view(bfloat16).astype(np.float32)).reshape(
                NL, 2, CONV_DIM
            )
            hi = 0.0
            per_layer = {}
            for li in range(NL):
                if li in ATTN_IDXS:
                    continue  # slot exists but nothing reads it
                for sl in range(2):
                    dv, rv = st_all[li, sl], ref_state[li, sl]
                    nd, nr = np.linalg.norm(dv), np.linalg.norm(rv)
                    if nd < 1e-20 and nr < 1e-20:
                        continue
                    # RELATIVE L2, not cosine: a uniformly k*correct state
                    # scores ~1.0 on cosine while every tap that consumes it is
                    # wrong by k.
                    rel = float(np.linalg.norm(dv - rv) / (nr + 1e-30))
                    hi = max(hi, rel)
                    per_layer[li] = max(per_layer.get(li, 0.0), rel)
                    n_st += 1
            # Bound the PER-MODEL-LAYER increment, not the total.
            #
            # bf16-on-device against an f32 reference accumulates down the
            # residual stream, so the total drift necessarily grows with depth
            # and any fixed ceiling on it is really a statement about how deep
            # the build is. What does not grow is how much ONE layer adds -- and
            # a DEFECTIVE layer adds a lot. So this gates the increment, and the
            # same number is meaningful at any depth.
            #
            # Divided by the number of MODEL layers spanned, because LFM2's
            # schedule is irregular: consecutive ShortConv layers are sometimes
            # adjacent and sometimes have an attention layer between them, so a
            # raw difference between neighbours in this list would compare one
            # layer of accumulation against two.
            #
            # The margin is wide, which is what makes the bound safe to set.
            # Measured on a known-good 16-layer build: totals run 0.015 -> 0.32
            # and the largest per-model-layer increment is 0.036. A layer that
            # is actually broken is nowhere near that -- when rope was
            # corrupting the ShortConv input, layer 0 alone sat at 0.995. So
            # 0.05 leaves half again over observed noise and stays ~20x below a
            # real defect.
            _prev_r, _prev_li = 0.0, -1
            for _li, _r in sorted(per_layer.items()):
                _span = _li - _prev_li
                _inc = (_r - _prev_r) / _span
                if _inc > a.state_rel:
                    n_st_bad += 1
                    print(
                        f"          layer {_li}: state drift {_prev_r:.4f} -> "
                        f"{_r:.4f} over {_span} model layer(s) = {_inc:.4f}/layer "
                        f"(> {a.state_rel})"
                    )
                if _r > _prev_r:
                    _prev_r, _prev_li = _r, _li
            worst_rel = max(worst_rel, hi)
            print(f"          ShortConv state read-back: max rel_l2 {hi:.4f}")
            if a.state_verbose:
                # Per layer, in DEPTH order. Drift that grows smoothly is the
                # compounding above; a single layer standing out is a bug in
                # that layer.
                print(
                    "            per layer: "
                    + "  ".join(f"L{li}:{r:.3f}" for li, r in sorted(per_layer.items()))
                )

    if a.dump:
        np.savez(a.dump, **dumps)
        print(f"wrote {a.dump}")
    print(f"\nmin cosine {worst:.6f} over {len(toks)} tokens (gate >= {a.cos_min})")
    ok = n_ok == len(toks)
    if a.seq:
        print(
            f"state slots checked {n_st}, max rel_l2 {worst_rel:.4f}; "
            f"{n_st_bad} layer(s) adding more than {a.state_rel} of drift per "
            f"model layer (gate: 0)"
        )
        ok = ok and n_st > 0 and n_st_bad == 0
    print("HYBRID NUMERICS PASS" if ok else "HYBRID NUMERICS FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
