#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Isolates whether Q4NX quantization of the target explains the low
acceptance rate measured in dflash_phase2_device.py/dflash_phase2_replay.py
(2.22/16 mean over 10 prompts, docs/DFlashFeasibility.md's "## 8"), against
the DFlash paper's own claimed "over 6x lossless acceleration".

Records the SAME data dflash_phase2_device.py does (prompt taps, greedy
continuation, per-position tapped hidden states at target_layer_ids), but
from the clean HF bf16 `Qwen/Qwen3-4B` model instead of the real NPU2 Q4NX
device -- pure CPU, no XRT, no torch/XRT-in-one-process concern. Output is
the same npz shape, so dflash_phase2_replay.py runs against it unmodified
(pass --embed-source=bf16 so the drafter's mask-token embedding and final
lm-head projection also come from the clean bf16 model, not the Q4NX one --
otherwise a clean-hidden-states-in, quantized-embedding-out mismatch would
confound the comparison).
"""

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

TARGET_LAYER_IDS = [1, 9, 17, 25, 33]
TAP_SLOTS = [lid + 1 for lid in TARGET_LAYER_IDS]
EOS_IDS = (151643, 151645)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "dflash_phase2_bf16.npz"
    n_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 96
    prompt_text = sys.argv[3] if len(sys.argv) > 3 else "The capital of France is"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("[bf16_reference] loading Qwen/Qwen3-4B (bf16, CPU)...", flush=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16)
    model.eval()

    prompt = tok.encode(prompt_text, add_special_tokens=False)
    P = len(prompt)
    print(f"[bf16_reference] prompt: {prompt_text!r} -> {prompt}", flush=True)

    input_ids = torch.tensor(prompt, dtype=torch.long).unsqueeze(0)
    past = None
    prompt_taps = []
    with torch.no_grad():
        out = model(
            input_ids, output_hidden_states=True, use_cache=True, return_dict=True
        )
    past = out.past_key_values
    hs = out.hidden_states  # tuple len 37: [0]=embed, [k]=raw output of layer k-1
    for p in range(P):
        prompt_taps.append(
            np.stack([hs[s][0, p, :].float().numpy() for s in TAP_SLOTS], axis=0)
        )
    prompt_taps_arr = np.stack(prompt_taps, axis=0)  # [P, 5, K]

    first = int(out.logits[0, -1].argmax())
    print(f"[bf16_reference] first token = {first}", flush=True)

    tokens = list(prompt) + [first]
    gen_ids = [first]
    gen_taps = []
    tok_id = first
    with torch.no_grad():
        for i in range(n_tokens):
            step_out = model(
                torch.tensor([[tok_id]], dtype=torch.long),
                past_key_values=past,
                output_hidden_states=True,
                use_cache=True,
                return_dict=True,
            )
            past = step_out.past_key_values
            hs = step_out.hidden_states
            gen_taps.append(
                np.stack([hs[s][0, -1, :].float().numpy() for s in TAP_SLOTS], axis=0)
            )
            pred = int(step_out.logits[0, -1].argmax())
            if pred in EOS_IDS:
                print(f"[bf16_reference] pos{P+i} -> EOS ({pred}), stop", flush=True)
                break
            gen_ids.append(pred)
            tok_id = pred
    gen_taps_arr = np.stack(gen_taps, axis=0)

    np.savez(
        out_path,
        prompt=np.array(prompt),
        first=first,
        prompt_taps=prompt_taps_arr,
        gen_ids=np.array(gen_ids),
        gen_taps=gen_taps_arr,
        tap_slots=np.array(TAP_SLOTS),
    )
    print(
        f"[bf16_reference] saved: prompt_taps{prompt_taps_arr.shape}, "
        f"gen_ids({len(gen_ids)}), gen_taps{gen_taps_arr.shape} -> {out_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
