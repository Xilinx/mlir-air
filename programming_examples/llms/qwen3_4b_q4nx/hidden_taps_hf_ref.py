#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""HF half of hidden_taps_verify.py: compute the real HF bf16 reference's
hidden states at the requested layer slots. Kept in its own process (invoked
via subprocess by hidden_taps_verify.py) -- see that file's module docstring
for why torch/transformers cannot share a process with an open XRT session.
"""

import sys
import json
import numpy as np


def main():
    device_npz = sys.argv[1]
    out_path = sys.argv[2]
    slots = [int(s) for s in sys.argv[3].split(",")]

    d = np.load(device_npz)
    prompt = list(d["prompt"])
    first = int(d["first"])
    sequence = prompt + [first]

    import torch
    from transformers import AutoModelForCausalLM

    print(
        "[hidden_taps_hf_ref] loading HF reference: Qwen/Qwen3-4B (bf16, CPU)...",
        flush=True,
    )
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16)
    model.eval()
    input_ids = torch.tensor(sequence, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        out = model(
            input_ids, output_hidden_states=True, use_cache=False, return_dict=True
        )
    hs = (
        out.hidden_states
    )  # tuple len n_layers+1: [0]=embed, [k]=raw output of layer k-1
    taps = {str(s): hs[s][0, -1, :].float().numpy() for s in slots}
    np.savez(out_path, **taps)
    print(f"[hidden_taps_hf_ref] saved {len(taps)} slots -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
