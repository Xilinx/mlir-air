# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Capture SUB-LAYER ground truth from the full-precision Gemma4-E2B checkpoint.
#
# FLM's shipped reference (gemma4_e2b_ref.safetensors) only has one tensor per
# decoder layer, so a wrong layer shows up as a single number and every sub-block
# -- attention, FFN, the per-layer-embedding injection -- is a suspect. That cost
# several rounds of hypothesis-and-refute on layer 8. This dumps the boundaries
# inside the layer so the broken block can be named directly.
#
# Same checkpoint and dtype FLM used (references/gen_gemma4_ref.py pins
# MODEL_DIR and loads float32), and the same token ids, so the captured
# layer output is directly comparable to their layer_{L}.
import argparse
import json
import os
import struct
from pathlib import Path

import numpy as np
import torch

# Both inputs are LOCAL development artifacts, not published: the full-precision
# checkpoint FLM's own gen_gemma4_ref.py pins as MODEL_DIR, and FLM's golden
# activations from an FLM source tree. This script only exists for bisecting
# INSIDE a layer, so it is never part of a gate; override via the environment.
MODEL_DIR = os.environ.get("GEMMA4_FP_CHECKPOINT", "")
REF = os.environ.get("FLM_REFERENCE", "")


def ref_input_ids(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        e = hdr["input_ids"]
        s, t = e["data_offsets"]
        f.seek(base + s)
        return np.frombuffer(f.read(t - s), np.int32).tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", default="7,8,9")
    ap.add_argument("--out", default="sublayer_ref.npz")
    a = ap.parse_args()

    if not MODEL_DIR or not REF:
        raise SystemExit(
            "Set GEMMA4_FP_CHECKPOINT (the full-precision checkpoint FLM's "
            "gen_gemma4_ref.py pins as MODEL_DIR) and FLM_REFERENCE (their "
            "golden activations). Both are local FLM development artifacts, "
            "not published, which is why this script is a bisecting tool "
            "rather than a gate."
        )
    want = [int(x) for x in a.layers.split(",")]

    from transformers import AutoModelForCausalLM

    ids = ref_input_ids(REF)
    print("input_ids:", ids, flush=True)
    print("loading model (float32, cpu)...", flush=True)
    # No device_map: it pulls in `accelerate`, and CPU is the default anyway.
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
    model.eval()
    tm = model.model
    tm = getattr(tm, "language_model", tm)
    print(f"layers={len(tm.layers)} hidden={tm.config.hidden_size}", flush=True)

    cap = {}

    def save(tag):
        def hook(mod, args, out):
            t = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(t):
                cap[tag] = t.detach()[0].float().numpy().copy()

        return hook

    def save_in(tag):
        def hook(mod, args, kwargs):
            t = args[0] if args else kwargs.get("hidden_states")
            if torch.is_tensor(t):
                cap[tag] = t.detach()[0].float().numpy().copy()

        return hook

    hs = []
    for L in want:
        lay = tm.layers[L]
        hs.append(lay.register_forward_pre_hook(save_in(f"L{L}.in"), with_kwargs=True))
        hs.append(lay.register_forward_hook(save(f"L{L}.out")))
        # Every named sub-block boundary that exists on this module.
        for name in (
            "input_layernorm",
            "self_attn",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "mlp",
            "post_feedforward_layernorm",
            "per_layer_input_gate",
            "per_layer_projection",
            "post_per_layer_input_norm",
        ):
            sub = getattr(lay, name, None)
            if sub is not None:
                hs.append(sub.register_forward_hook(save(f"L{L}.{name}")))

    with torch.no_grad():
        model(input_ids=torch.tensor([ids], dtype=torch.long), use_cache=False)
    for h in hs:
        h.remove()

    np.savez(a.out, input_ids=np.array(ids, np.int32), **cap)
    print(f"wrote {a.out} with {len(cap)} tensors:", flush=True)
    for k in sorted(cap):
        v = cap[k]
        print(f"  {k:38} {str(v.shape):14} rms={np.sqrt((v**2).mean()):.4f}")


if __name__ == "__main__":
    main()
