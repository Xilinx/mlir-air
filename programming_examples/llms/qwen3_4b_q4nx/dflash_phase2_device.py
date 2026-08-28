#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Phase 2 of the DFlash drafter bring-up (docs/DFlashFeasibility.md, "## 8"):
device half. Records everything the offline (torch, CPU) drafter replay needs,
in ONE real NPU run, so no further device calls are needed after this:

  - the numpy-prefill oracle's own per-layer tapped hidden states over the
    PROMPT (block 0's `target_hidden` context, per the real spec_generate:
    the first draft call's context is the prefill's hidden_states, NOT
    extended by the freshly-sampled first token yet)
  - a plain greedy decode of N_TOKENS, with DECODE_HIDDEN_TAPS=1, recording
    both the argmax token AND the tapped hidden states at every position

This is legitimate because the target's greedy decode is a deterministic
function of the token prefix alone: DFlash's own lossless guarantee is that
the accepted output IS the target's greedy stream, so recording that stream
once up front and replaying the drafter against sliding windows of it
offline is equivalent to interleaved draft/verify -- see
hidden_taps_verify.py's module docstring for why device work and torch work
are kept in separate processes (a segfault when both share a process).
"""

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

os.environ.setdefault("Q4NX_QWEN3_4B_DECODE_DIR", str(_HERE / "hidden_taps_test"))
os.environ["DECODE_HIDDEN_TAPS"] = "1"
os.environ.setdefault("W_DUAL_CHAN", "1")

import numpy as np
from qwen3_4b_q4nx_inference import FusedDecoder, MODEL_DEFAULT, PARIS_PROMPT, EOS_IDS
import qwen3_4b_q4nx_weights as gw

# HF hidden_states index for target_layer_ids=[1,9,17,25,33] (utils.py's
# extract_context_feature, offset=1): the AIR slot / HF hidden_states index
# convention is identical (slot k = output after k layers).
TARGET_LAYER_IDS = [1, 9, 17, 25, 33]
TAP_SLOTS = [lid + 1 for lid in TARGET_LAYER_IDS]  # [2,10,18,26,34]


class HiddenTapsFusedDecoder(FusedDecoder):
    """Same enlarged-X-BO subclass as hidden_taps_device.py -- duplicated
    rather than imported so this script has no import-order dependency on
    that one; both are new sibling scripts, neither modifies the verified
    driver."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.X_SLOTS = self.UNI_DEC + 1
        HO = self.xrt.bo.host_only
        g = self.kern.group_id
        self.x_bo = self.xrt.bo(self.dev, self.X_SLOTS * self.K * 2, HO, g(3))
        self.last_taps = None

    def dispatch(self, tok, p):
        lg = super().dispatch(tok, p)
        xrt = self.xrt
        FROM = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        self.x_bo.sync(FROM, self.X_SLOTS * self.K * 2, 0)
        flat = np.frombuffer(
            self.x_bo.map(), dtype=self.bf16, count=self.X_SLOTS * self.K, offset=0
        ).astype(np.float32)
        self.last_taps = flat.reshape(self.X_SLOTS, self.K)
        return lg


def forward_prompt_with_taps(model, prompt_ids, tap_slots):
    """qwen3_8b_q4nx_weights.forward_prompt, extended to also return the
    tapped hidden states (output-after-k-layers, k in tap_slots) at every
    prompt position. Duplicated (not monkeypatched) so the shared reference
    function stays untouched for every other example that relies on it."""
    from ml_dtypes import bfloat16
    from qwen3_8b_q4nx_weights import (
        _bf,
        _rmsnorm,
        _apply_rope_halfsplit,
        _silu,
        NUM_LAYERS,
        DK,
        N_Q_HEADS,
        N_KV_HEADS,
        Q_PER_KV,
        DH,
        DQ,
        ROPE_THETA,
        ATTN_SCALE,
        _DH_2,
    )

    embed, final_norm, lm_head = model.embed_norm_lmhead()
    ids = np.asarray(prompt_ids, dtype=np.int64)
    P = ids.shape[0]
    x = _bf(embed[ids].astype(np.float32))

    Kc = np.zeros((NUM_LAYERS, P, DK), np.float32)
    Vc = np.zeros((NUM_LAYERS, P, DK), np.float32)
    pos = np.arange(P)
    taps = {s: None for s in tap_slots}
    if 0 in taps:
        taps[0] = x.copy()

    for L in range(NUM_LAYERS):
        w = model.layer_weights(L)
        n_in, n_pa = model.layer_rms(L)
        qn, kn = model.layer_qk_norm(L)

        residual = x
        h = _rmsnorm(x, n_in)
        q = _bf(h @ w["q"]).reshape(P, N_Q_HEADS, DH).astype(np.float32)
        k = _bf(h @ w["k"]).reshape(P, N_KV_HEADS, DH).astype(np.float32)
        v = _bf(h @ w["v"]).reshape(P, N_KV_HEADS, DH).astype(np.float32)

        inv = 1.0 / (ROPE_THETA ** (np.arange(_DH_2) / _DH_2))
        ang = pos[:, None] * inv[None, :]
        cos = np.cos(ang).astype(bfloat16).astype(np.float32)[:, None, :]
        sin = np.sin(ang).astype(bfloat16).astype(np.float32)[:, None, :]
        q = _apply_rope_halfsplit(_rmsnorm(q, qn), cos, sin)
        k = _apply_rope_halfsplit(_rmsnorm(k, kn), cos, sin)
        Kc[L] = _bf(k).reshape(P, DK)
        Vc[L] = _bf(v).reshape(P, DK)

        o = np.zeros((P, N_Q_HEADS, DH), np.float32)
        for hq in range(N_Q_HEADS):
            hk = hq // Q_PER_KV
            scores = (q[:, hq, :] @ k[:, hk, :].T) * ATTN_SCALE
            mask = pos[None, :] > pos[:, None]
            scores = np.where(mask, -1e30, scores)
            scores -= scores.max(axis=-1, keepdims=True)
            e = np.exp(scores)
            attn = e / e.sum(axis=-1, keepdims=True)
            o[:, hq, :] = attn @ v[:, hk, :]
        attn_out = _bf(o.reshape(P, DQ) @ w["o"])
        x = _bf(residual + attn_out)

        residual = x
        h2 = _rmsnorm(x, n_pa)
        act = _bf(_silu(_bf(h2 @ w["gate"])) * _bf(h2 @ w["up"]))
        x = _bf(residual + _bf(act @ w["down"]))

        if (L + 1) in taps:
            taps[L + 1] = x.copy()

    xf = _rmsnorm(x, final_norm)
    logits = (xf.astype(np.float32) @ lm_head.astype(np.float32).T).astype(np.float32)
    return Kc, Vc, logits, taps


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "dflash_phase2_device.npz"
    n_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 96

    model = MODEL_DEFAULT
    prompt_text = sys.argv[3] if len(sys.argv) > 3 else None
    if prompt_text:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
        prompt = tok.encode(prompt_text, add_special_tokens=False)
        print(f"[dflash_phase2_device] prompt: {prompt_text!r} -> {prompt}", flush=True)
    else:
        prompt = list(PARIS_PROMPT)
    P = len(prompt)

    print(
        "[dflash_phase2_device] numpy-prefill oracle (KV seed + prompt taps)...",
        flush=True,
    )
    qm = gw.Q4nxModel(model)
    Kc, Vc, logits, prompt_taps = forward_prompt_with_taps(qm, prompt, TAP_SLOTS)
    first = int(logits[-1].argmax())
    print(f"[dflash_phase2_device] first token = {first}", flush=True)
    # [P, 5, K] in TAP_SLOTS order -- extract_context_feature's own concat order.
    prompt_taps_arr = np.stack([prompt_taps[s] for s in TAP_SLOTS], axis=1)

    print(
        f"[dflash_phase2_device] opening HIDDEN_TAPS decode template (N_TOKENS={n_tokens})...",
        flush=True,
    )
    dec = HiddenTapsFusedDecoder(model=model, max_L=P + n_tokens)
    dec.seed_kv(Kc, Vc, P)

    tokens = list(prompt) + [first]
    gen_ids = [first]
    gen_taps = []  # list of [5, K], TAP_SLOTS order, one per generated position
    n_eff = min(n_tokens, dec.ATTN_MAXL - P)
    for p in range(P, P + n_eff):
        lg = dec.dispatch(tokens[p], p)
        taps_p = dec.last_taps[TAP_SLOTS]  # [5, K]
        gen_taps.append(taps_p)
        pred = int(lg.argmax())
        if pred in EOS_IDS:
            print(f"[dflash_phase2_device] pos{p} -> EOS ({pred}), stop", flush=True)
            break
        gen_ids.append(pred)
        if p + 1 >= len(tokens):
            tokens.append(pred)
    gen_taps_arr = np.stack(gen_taps, axis=0)  # [n_generated, 5, K]

    np.savez(
        out_path,
        prompt=np.array(prompt),
        first=first,
        prompt_taps=prompt_taps_arr,  # [P, 5, K]
        gen_ids=np.array(gen_ids),  # [1 + n_generated_after_first]
        gen_taps=gen_taps_arr,  # [n_dispatches, 5, K], gen_taps[i] is the tap at position P+i
        tap_slots=np.array(TAP_SLOTS),
    )
    print(
        f"[dflash_phase2_device] saved: prompt_taps{prompt_taps_arr.shape}, "
        f"gen_ids({len(gen_ids)}), gen_taps{gen_taps_arr.shape} -> {out_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
