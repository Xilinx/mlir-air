#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# End-to-end Qwen2.5-3B on NPU2 with NO REFERENCE MODEL anywhere in the loop --
# every token comes from the NPU prefill and the NPU fused decode, never from a
# host copy of the model. What DOES still run on the host each token is the
# final RMSNorm + LM-head projection (`_logits` below) and the argmax: the 36
# transformer layers are on device, the last projection is not. Moving it
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
# The generation loop lives in `QwenFusedDecoder`, not in main(), so the shared
# verify subsystem can drive it one token at a time: `seed_kv()` + `dispatch()`
# are the same two entry points llama32_1b_q4nx's FusedDecoder exposes, which is
# what llms/qwen25_3b_q4/verify_adapter.py binds to. main() is a thin CLI over it.
#
#   cd programming_examples/llms/qwen25_3b_q4
#   make compile-decode          # kernels + decode_qwen.xclbin
#   make gen                     # prefill -> hand-off -> N_GEN tokens
#
# or by hand:
#
#   python3 qwen25_3b_q4_prefill.py --dump-kv /tmp/kv.npz --prompt "<32+ tokens>"
#   cd ../../fused_decode
#   QWEN_NLAYERS=36 python3 fused_decode_qwen.py      # 36-layer xclbin
#   QWEN_NLAYERS=36 python3 qwen_prefill_to_decode.py --kv /tmp/kv.npz --n-gen 16
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
    # dual_chan must match the flag the decode xclbin was BUILT with: it reorders
    # the cascade so each of a column's two shim channels reads a contiguous run.
    return np.concatenate(
        [
            qr.pack_q4k_cascade_fast(
                *ph[p], m.NCX, m.NCY, dual_chan=bool(getattr(m, "W_DUAL_CHAN", 0))
            )
            for p in range(m.NPH)
        ]
    )


def _check_w_dual_chan_stamp(xclbin):
    """Refuse a weight layout the xclbin was not built for.

    The xclbin's layout is set by W_DUAL_CHAN at BUILD time; pack_layer uses the
    CURRENT module value. A mismatch is silent garbage, so refuse. The stamp sits
    next to the xclbin it describes. An UNREADABLE stamp is just as unsafe as a
    mismatched one (an xclbin built before the stamp existed has an unknown
    layout), so that is refused too rather than assumed compatible."""
    want = int(getattr(m, "W_DUAL_CHAN", 0))
    stamp = os.path.splitext(xclbin)[0] + ".flags"
    try:
        with open(stamp) as f:
            got = int(f.read().strip().split("=")[1])
    except (OSError, IndexError, ValueError):
        got = None
    if got is None:
        raise SystemExit(
            f"cannot read the W_DUAL_CHAN build stamp {stamp}, so the weight "
            f"layout of {xclbin} is unknown and packing for "
            f"W_DUAL_CHAN={want} may silently produce garbage. Rebuild with "
            f"QWEN_NLAYERS=.. W_DUAL_CHAN={want} python3 fused_decode_qwen.py"
        )
    if got != want:
        raise SystemExit(
            f"{xclbin} was built with W_DUAL_CHAN={got} but this run "
            f"packs weights for W_DUAL_CHAN={want}. Rebuild the xclbin with the "
            f"same setting (QWEN_NLAYERS=.. W_DUAL_CHAN={want} python3 "
            f"fused_decode_qwen.py) or re-run with W_DUAL_CHAN={got}."
        )


class QwenFusedDecoder:
    """One-dispatch-per-token Qwen2.5-3B decode over the fused NPU superkernel.

    Owns the resident BOs, the token-invariant Q4_0 weight stream and the host
    sliding KV window. Two entry points, matching llama32_1b_q4nx's FusedDecoder
    so the shared verify Runner contract can bind to either:

        seed_kv(K, V, ctx)      seed the window from a prefill's KV cache
        dispatch(token, pos)    one token -> logits [VOCAB]

    Construction packs 36 layers of Q4_0 weights on the host (~1 min) and writes
    them to the device once; they never change again.
    """

    def __init__(self, xclbin="decode_qwen.xclbin", insts="decode_qwen.insts.bin"):
        self.NL = m.NLAYERS
        # ATTN_MAXL is the decode's fixed context window: the device attends over
        # exactly this many slots and appends at slot ATTN_MAXL-1.
        self.ATTN_MAXL = m.ATTN_L
        assert m.REGION_W == m.DK, (m.REGION_W, m.DK)
        _check_w_dual_chan_stamp(xclbin)

        hf = qr.HFModel()
        aperm = qr.attn_out_perm(m)
        # Kept for the host-side final RMSNorm + LM head (tied embeddings).
        self.emb = hf.bf16("model.embed_tokens.weight")
        self.norm = hf.bf16("model.norm.weight")

        L, NL = self.ATTN_MAXL, self.NL
        self.X = max(m.X_CHUNKS * 2 * m.COL_BLOCK, NL * m.XLAYER)
        self.WSZ = NL * m.W_LAYER
        self.KVL = m.N_ATTN_CU * 2 * m.ATTN_ROUNDS * m.KVBLK
        self.KVN = NL * self.KVL
        self.YN = max(m.DEST_TOTAL * m.PAYLOAD, m.K + m.DQ)

        print(f"[handoff] packing Q4_0 weights for {NL} layers...", flush=True)
        self.xd = np.zeros(self.X, bfloat16)
        wd = np.empty(self.WSZ, np.int16)
        for k in range(NL):
            b = k * m.XLAYER
            wd[k * m.W_LAYER : (k + 1) * m.W_LAYER] = pack_layer(hf, aperm, k)
            self.xd[b + m.XO_ROPE + m.DH : b + m.XO_ROPE + m.ROPE_W] = np.concatenate(
                [hf.bf16(f"model.layers.{k}.{t}.bias") for t in qr._BIAS]
            ).astype(bfloat16)
            self.xd[b + m.XO_RMSW : b + m.XO_RMSW + m.K] = hf.bf16(
                f"model.layers.{k}.input_layernorm.weight"
            ).astype(bfloat16)
            self.xd[b + m.XO_RMSW2 : b + m.XO_RMSW2 + m.K] = hf.bf16(
                f"model.layers.{k}.post_attention_layernorm.weight"
            ).astype(bfloat16)

        dev = xrt.device(0)
        xb = xrt.xclbin(xclbin)
        dev.register_xclbin(xb)
        hwc = xrt.hw_context(dev, xb.get_uuid())
        self.kern = xrt.kernel(
            hwc,
            [q for q in xb.get_kernels() if "MLIR_AIE" in q.get_name()][0].get_name(),
        )
        g_, HO = self.kern.group_id, xrt.bo.host_only
        self.x_bo = xrt.bo(dev, self.X * 2, HO, g_(3))
        self.w_bo = xrt.bo(dev, self.WSZ * 2, HO, g_(4))
        self.kv_bo = xrt.bo(dev, self.KVN * 2, HO, g_(5))
        self.y_bo = xrt.bo(dev, self.YN * 2, HO, g_(6))
        self._insts = np.fromfile(insts, dtype=np.uint32)
        self.ib = xrt.bo(dev, self._insts.nbytes, xrt.bo.cacheable, g_(1))
        self.ib.write(self._insts, 0)
        self.ib.sync(TO)
        self.w_bo.write(wd.view(np.uint16), 0)
        self.w_bo.sync(TO)  # weights are token-invariant

        # Host sliding KV window, [NL, ATTN_MAXL, DK]. seed_kv() fills it.
        self.Kc = np.zeros((NL, L, m.DK), np.float32)
        self.Vc = np.zeros((NL, L, m.DK), np.float32)
        self.dev_ms = 0.0  # accumulated device dispatch time

    def seed_kv(self, K, V, ctx):
        """Seed the window with the LAST ATTN_MAXL positions of a prefill's KV.

        K/V are [n_layers, ctx, DK]. RoPE is absolute, so a cached entry carries
        the roping of the position it was computed at -- taking the tail keeps
        the window's positions contiguous with the next token's."""
        L = self.ATTN_MAXL
        if ctx < L:
            raise SystemExit(
                f"prompt gave {ctx} tokens, decode window ATTN_L={L}. The decode "
                f"seeds its window with the last {L} prefill positions, so the "
                f"prompt must be at least that long (or rebuild with a smaller "
                f"ATTN_L)."
            )
        assert K.shape == (self.NL, ctx, m.DK), (K.shape, (self.NL, ctx, m.DK))
        self.Kc[:] = K[:, ctx - L :, :]
        self.Vc[:] = V[:, ctx - L :, :]

    def _logits(self, h):
        """Host final RMSNorm + tied LM head over the decode's hidden state."""
        return (h / np.sqrt((h * h).mean() + 1e-6) * self.norm) @ self.emb.T

    def dispatch(self, token, pos):
        """One fused decode dispatch: token at absolute position `pos` -> logits.

        Also slides the KV window by one, consuming the K/V the device just
        appended at slot ATTN_MAXL-1, so consecutive calls chain."""
        L, NL, KVL = self.ATTN_MAXL, self.NL, self.KVL
        lut = qr.rope_lut(pos, m.DH).astype(bfloat16)
        for k in range(NL):
            self.xd[k * m.XLAYER + m.XO_ROPE : k * m.XLAYER + m.XO_ROPE + m.DH] = lut
        self.xd[m.XO_RMSIN : m.XO_RMSIN + m.K] = np.asarray(
            self.emb[token], np.float32
        ).astype(bfloat16)
        kvd = np.zeros(self.KVN, bfloat16)
        for k in range(NL):
            kvd[k * KVL :][: L * m.REGION_W] = self.Kc[k].astype(bfloat16).ravel()
            kvd[k * KVL + m.NGRP * m.REGION_STRIDE :][: L * m.REGION_W] = (
                self.Vc[k].astype(bfloat16).ravel()
            )
        self.x_bo.write(self.xd.view(np.uint16), 0)
        self.x_bo.sync(TO)
        self.kv_bo.write(kvd.view(np.uint16), 0)
        self.kv_bo.sync(TO)

        t0 = time.time()
        st = self.kern(
            3, self.ib, self._insts.size, self.x_bo, self.w_bo, self.kv_bo, self.y_bo
        ).wait(20000)
        self.dev_ms += (time.time() - t0) * 1e3
        if "COMPLETED" not in str(st):
            raise RuntimeError(f"decode dispatch pos{pos} state={st}")
        self.x_bo.sync(FR)
        self.kv_bo.sync(FR)

        h = np.frombuffer(self.x_bo.read(self.X * 2, 0), dtype=bfloat16).astype(
            np.float32
        )[m.XO_RMSIN : m.XO_RMSIN + m.K]
        logits = self._logits(h)

        # Take this token's K/V (device wrote slot L-1) and slide the window.
        kvo = np.frombuffer(self.kv_bo.read(self.KVN * 2, 0), dtype=bfloat16).astype(
            np.float32
        )
        for k in range(NL):
            newk = kvo[k * KVL :][: L * m.REGION_W].reshape(L, m.REGION_W)[L - 1]
            newv = kvo[k * KVL + m.NGRP * m.REGION_STRIDE :][: L * m.REGION_W].reshape(
                L, m.REGION_W
            )[L - 1]
            self.Kc[k] = np.roll(self.Kc[k], -1, 0)
            self.Kc[k][L - 2], self.Kc[k][L - 1] = newk, 0
            self.Vc[k] = np.roll(self.Vc[k], -1, 0)
            self.Vc[k][L - 2], self.Vc[k][L - 1] = newv, 0
        return logits


def main():
    ap_ = argparse.ArgumentParser(description="Qwen2.5-3B NPU prefill -> NPU decode")
    ap_.add_argument("--kv", default="/tmp/qwen_prefill_kv.npz", help="--dump-kv file")
    ap_.add_argument(
        "--n-gen", type=int, default=int(os.environ.get("QWEN_NGEN", "20"))
    )
    ap_.add_argument("--xclbin", default="decode_qwen.xclbin")
    ap_.add_argument("--insts", default="decode_qwen.insts.bin")
    args = ap_.parse_args()

    NL = m.NLAYERS
    z = np.load(args.kv)
    ids = [int(t) for t in z["ids"]]
    ctx = int(z["ctx"])
    kp = z["k"].view(bfloat16).astype(np.float32)  # [n_layers, ctx, DK]
    vp = z["v"].view(bfloat16).astype(np.float32)
    assert kp.shape == (NL, ctx, m.DK), f"prefill KV {kp.shape} != {(NL, ctx, m.DK)}"

    dec = QwenFusedDecoder(xclbin=args.xclbin, insts=args.insts)
    print(
        f"[handoff] prefill KV: {NL} layers x {ctx} tokens; seeding the last "
        f"ATTN_L={dec.ATTN_MAXL} positions ({ctx - dec.ATTN_MAXL}..{ctx - 1})",
        flush=True,
    )
    dec.seed_kv(kp, vp, ctx)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(qr.HF_REPO)

    out_ids = []
    cur = ids[-1]
    pos = ctx - 1  # re-run the last prompt token to produce the first logit
    t_all = time.time()
    for step in range(args.n_gen):
        nxt = int(np.argmax(dec.dispatch(cur, pos)))
        out_ids.append(nxt)
        cur = nxt
        pos += 1
        print(f"  [{step:2d}] {nxt:6d} {tok.decode([nxt])!r}", flush=True)

    wall = time.time() - t_all
    n = max(len(out_ids), 1)
    print("\nprompt    :", repr(tok.decode(ids)))
    print("generated :", repr(tok.decode(out_ids)))
    print(
        f"\n{len(out_ids)} tokens | device {dec.dev_ms/n:.1f} ms/tok "
        f"({1000/(dec.dev_ms/n):.2f} tok/s device-only) | end-to-end {wall/n:.2f} s/tok"
    )
    # Machine-readable lines for llms/bench/extract_perf.py (TOKS_LABEL_RE and
    # CTX_RE). The human line above does not parse: its TOKS_RE alternative needs
    # "tok/s" immediately before the closing paren, and ours is followed by
    # "device-only".
    print(f"Tokens/second: {1000/(dec.dev_ms/n):.2f}")
    print(f"prompt_len: {ctx}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
