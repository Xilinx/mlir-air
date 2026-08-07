#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# End-to-end Qwen2.5-3B on NPU2 with NO REFERENCE MODEL anywhere in the loop --
# every token comes from the NPU prefill and the NPU fused decode, never from a
# host copy of the model. What DOES still run on the host each token is the
# final RMSNorm + LM-head projection (`lg = ... @ emb.T` below) and the argmax:
# the 36 transformer layers are on device, the last projection is not. Moving it
# on-device is a performance item, not a correctness one -- the decode's own
# hidden state is what feeds it.
#
#   llms/qwen25_3b_q4  --dump-kv  ->  this script  ->  fused_decode_qwen
#   (NPU prefill, 36 layers)          (KV hand-off)    (NPU decode, 36 layers,
#                                                        one dispatch/token)
#
# Both halves quantize with the SAME Q4_0 codec (qwen25_3b_requant), so the
# prefill's roped-K / biased-V are exactly what the decode's own rope core would
# have appended for those positions.
#
# LAYOUT HAND-OFF. The prefill emits per layer a seq-major [ctx, 256] roped-K and
# a [ctx, 256] biased-V (256 = 2 kv heads x head_dim 128, head-major). The fused
# decode wants, per layer, one region-major slab: K rows at offset 0 and V rows
# at offset NGRP*REGION_STRIDE, each [ATTN_L, REGION_W] with REGION_W == 256.
# Same element order, so the hand-off is a slice, not a shuffle -- the check
# below asserts that rather than assuming it.
#
# SLIDING WINDOW. The device always appends this token's K/V at slot ATTN_L-1
# and attends over all ATTN_L slots, so the host keeps a window of exactly the
# last ATTN_L positions and shifts it left after every token. Seeding it with
# the LAST ATTN_L rows of the prefill (absolute positions ctx-ATTN_L .. ctx-1)
# keeps RoPE consistent -- RoPE is absolute, so a cached entry carries the
# roping of the position it was computed at.
#
#   cd programming_examples/llms/qwen25_3b_q4
#   python3 qwen25_3b_q4_prefill.py --dump-kv /tmp/kv.npz --prompt "..."
#   cd ../../fused_decode
#   QWEN_NLAYERS=36 python3 fused_decode_qwen.py      # 36-layer xclbin
#   python3 qwen_prefill_to_decode.py --kv /tmp/kv.npz --n-gen 20
import argparse
import os
import sys
import time

import numpy as np
import pyxrt as xrt
from ml_dtypes import bfloat16

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fused_decode_qwen as m  # noqa: E402
import qwen25_3b_requant as qr  # noqa: E402

TO = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
FR = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE


def pack_layer(hf, ap, k):
    """One layer's Q4_0 cascade-packed weight stream (4 phases)."""
    R = {nm: hf.bf16(f"model.layers.{k}.{t}.weight") for nm, t in qr._PROJ.items()}
    ph = [None] * m.NPH
    ph[0] = qr.requant_q4_0(np.concatenate([R["q"], R["k"], R["v"]], 0))
    ph[m.OPROJ_PHASE] = qr.requant_q4_0(R["o"][:, ap])
    qu, su = qr.requant_q4_0(qr._pad_rows(R["up"], m.INTERMEDIATE))
    qg, sg = qr.requant_q4_0(qr._pad_rows(R["gate"], m.INTERMEDIATE))
    ph[m.GATEUP_PHASE] = (
        qr._interleave_chunks(qu, qg, m.PAYLOAD),
        qr._interleave_chunks(su, sg, m.PAYLOAD),
    )
    ph[m.DOWN_PHASE] = qr.requant_q4_0(qr._pad_cols(R["down"], m.INTERMEDIATE))
    return np.concatenate(
        [qr.pack_q4k_cascade_fast(*ph[p], m.NCX, m.NCY) for p in range(m.NPH)]
    )


def main():
    ap_ = argparse.ArgumentParser(description="Qwen2.5-3B NPU prefill -> NPU decode")
    ap_.add_argument("--kv", default="/tmp/qwen_prefill_kv.npz", help="--dump-kv file")
    ap_.add_argument(
        "--n-gen", type=int, default=int(os.environ.get("QWEN_NGEN", "20"))
    )
    ap_.add_argument("--xclbin", default="decode_qwen.xclbin")
    ap_.add_argument("--insts", default="decode_qwen.insts.bin")
    args = ap_.parse_args()

    NL, L = m.NLAYERS, m.ATTN_L
    z = np.load(args.kv)
    ids = [int(t) for t in z["ids"]]
    ctx = int(z["ctx"])
    kp = z["k"].view(bfloat16).astype(np.float32)  # [n_layers, ctx, DK]
    vp = z["v"].view(bfloat16).astype(np.float32)
    assert kp.shape == (NL, ctx, m.DK), f"prefill KV {kp.shape} != {(NL, ctx, m.DK)}"
    assert m.REGION_W == m.DK, (m.REGION_W, m.DK)
    assert ctx >= L, f"prompt gave {ctx} tokens, decode window ATTN_L={L}"
    print(
        f"[handoff] prefill KV: {NL} layers x {ctx} tokens; "
        f"seeding the last ATTN_L={L} positions ({ctx - L}..{ctx - 1})",
        flush=True,
    )

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(qr.HF_REPO)
    hf = qr.HFModel()
    aperm = qr.attn_out_perm(m)
    emb = hf.bf16("model.embed_tokens.weight")
    norm = hf.bf16("model.norm.weight")

    # Host KV window, seeded from the prefill (no reference model involved).
    Kc = np.ascontiguousarray(kp[:, ctx - L :, :])
    Vc = np.ascontiguousarray(vp[:, ctx - L :, :])

    X = max(m.X_CHUNKS * 2 * m.COL_BLOCK, NL * m.XLAYER)
    WSZ = NL * m.W_LAYER
    KVL = m.N_ATTN_CU * 2 * m.ATTN_ROUNDS * m.KVBLK
    KVN = NL * KVL
    YN = max(m.DEST_TOTAL * m.PAYLOAD, m.K + m.DQ)

    print(f"[handoff] packing Q4_0 weights for {NL} layers...", flush=True)
    xd = np.zeros(X, bfloat16)
    wd = np.empty(WSZ, np.int16)
    for k in range(NL):
        b = k * m.XLAYER
        wd[k * m.W_LAYER : (k + 1) * m.W_LAYER] = pack_layer(hf, aperm, k)
        xd[b + m.XO_ROPE + m.DH : b + m.XO_ROPE + m.ROPE_W] = np.concatenate(
            [hf.bf16(f"model.layers.{k}.{t}.bias") for t in qr._BIAS]
        ).astype(bfloat16)
        xd[b + m.XO_RMSW : b + m.XO_RMSW + m.K] = hf.bf16(
            f"model.layers.{k}.input_layernorm.weight"
        ).astype(bfloat16)
        xd[b + m.XO_RMSW2 : b + m.XO_RMSW2 + m.K] = hf.bf16(
            f"model.layers.{k}.post_attention_layernorm.weight"
        ).astype(bfloat16)

    dev = xrt.device(0)
    xb = xrt.xclbin(args.xclbin)
    dev.register_xclbin(xb)
    hwc = xrt.hw_context(dev, xb.get_uuid())
    kern = xrt.kernel(
        hwc, [q for q in xb.get_kernels() if "MLIR_AIE" in q.get_name()][0].get_name()
    )
    g_, HO = kern.group_id, xrt.bo.host_only
    x_bo = xrt.bo(dev, X * 2, HO, g_(3))
    w_bo = xrt.bo(dev, WSZ * 2, HO, g_(4))
    kv_bo = xrt.bo(dev, KVN * 2, HO, g_(5))
    y_bo = xrt.bo(dev, YN * 2, HO, g_(6))
    insts = np.fromfile(args.insts, dtype=np.uint32)
    ib = xrt.bo(dev, insts.nbytes, xrt.bo.cacheable, g_(1))
    ib.write(insts, 0)
    ib.sync(TO)
    w_bo.write(wd.view(np.uint16), 0)
    w_bo.sync(TO)  # weights are token-invariant

    out_ids = []
    cur = ids[-1]
    pos = ctx - 1  # re-run the last prompt token to produce the first logit
    dev_ms = 0.0
    t_all = time.time()
    for step in range(args.n_gen):
        lut = qr.rope_lut(pos, m.DH).astype(bfloat16)
        for k in range(NL):
            xd[k * m.XLAYER + m.XO_ROPE : k * m.XLAYER + m.XO_ROPE + m.DH] = lut
        xd[m.XO_RMSIN : m.XO_RMSIN + m.K] = np.asarray(emb[cur], np.float32).astype(
            bfloat16
        )
        kvd = np.zeros(KVN, bfloat16)
        for k in range(NL):
            kvd[k * KVL :][: L * m.REGION_W] = Kc[k].astype(bfloat16).ravel()
            kvd[k * KVL + m.NGRP * m.REGION_STRIDE :][: L * m.REGION_W] = (
                Vc[k].astype(bfloat16).ravel()
            )
        x_bo.write(xd.view(np.uint16), 0)
        x_bo.sync(TO)
        kv_bo.write(kvd.view(np.uint16), 0)
        kv_bo.sync(TO)

        t0 = time.time()
        st = kern(3, ib, insts.size, x_bo, w_bo, kv_bo, y_bo).wait(20000)
        dev_ms += (time.time() - t0) * 1e3
        if "COMPLETED" not in str(st):
            print("DISPATCH FAILED:", st)
            return 1
        x_bo.sync(FR)
        kv_bo.sync(FR)

        h = np.frombuffer(x_bo.read(X * 2, 0), dtype=bfloat16).astype(np.float32)[
            m.XO_RMSIN : m.XO_RMSIN + m.K
        ]
        lg = (h / np.sqrt((h * h).mean() + 1e-6) * norm) @ emb.T
        nxt = int(np.argmax(lg))
        out_ids.append(nxt)

        # Take this token's K/V (device wrote slot L-1) and slide the window.
        kvo = np.frombuffer(kv_bo.read(KVN * 2, 0), dtype=bfloat16).astype(np.float32)
        for k in range(NL):
            newk = kvo[k * KVL :][: L * m.REGION_W].reshape(L, m.REGION_W)[L - 1]
            newv = kvo[k * KVL + m.NGRP * m.REGION_STRIDE :][: L * m.REGION_W].reshape(
                L, m.REGION_W
            )[L - 1]
            Kc[k] = np.roll(Kc[k], -1, 0)
            Kc[k][L - 2], Kc[k][L - 1] = newk, 0
            Vc[k] = np.roll(Vc[k], -1, 0)
            Vc[k][L - 2], Vc[k][L - 1] = newv, 0
        cur = nxt
        pos += 1
        print(f"  [{step:2d}] {nxt:6d} {tok.decode([nxt])!r}", flush=True)

    wall = time.time() - t_all
    n = max(len(out_ids), 1)
    print("\nprompt    :", repr(tok.decode(ids)))
    print("generated :", repr(tok.decode(out_ids)))
    print(
        f"\n{len(out_ids)} tokens | device {dev_ms/n:.1f} ms/tok "
        f"({1000/(dev_ms/n):.2f} tok/s device-only) | end-to-end {wall/n:.2f} s/tok"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
