#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Device half of hidden_taps_verify.py: run one HIDDEN_TAPS decode dispatch on
real NPU2 hardware and save the tap readback to disk. Split into its own
process (invoked by hidden_taps_verify.py via subprocess) because loading
torch/transformers in the same process as an open XRT device session segfaults
-- see hidden_taps_verify.py's module docstring.
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
from qwen3_4b_q4nx_inference import FusedDecoder, MODEL_DEFAULT, PARIS_PROMPT
import qwen3_4b_q4nx_weights as gw


class HiddenTapsFusedDecoder(FusedDecoder):
    """FusedDecoder, but with the X BO sized for every HIDDEN_TAPS slot and a
    readback of all of them after dispatch -- a new sibling class, not a
    modification of the verified driver. The base class's x_bo (sized for
    exactly one slot) is reallocated here, at the same XRT group id (3),
    immediately after the base __init__ runs."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.X_SLOTS = self.UNI_DEC + 1
        HO = self.xrt.bo.host_only
        g = self.kern.group_id
        self.x_bo = self.xrt.bo(self.dev, self.X_SLOTS * self.K * 2, HO, g(3))
        self.last_taps = None  # set by dispatch(): [X_SLOTS, K] float32

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


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "hidden_taps_device.npz"
    model = MODEL_DEFAULT
    prompt = list(PARIS_PROMPT)
    P = len(prompt)

    print("[hidden_taps_device] numpy-prefill oracle (KV seed)...", flush=True)
    qm = gw.Q4nxModel(model)
    Kc, Vc, logits = gw.forward_prompt(qm, prompt)
    first = int(logits[-1].argmax())
    print(f"[hidden_taps_device] first token = {first}", flush=True)

    print("[hidden_taps_device] opening HIDDEN_TAPS decode template...", flush=True)
    dec = HiddenTapsFusedDecoder(model=model, max_L=P + 1)
    dec.seed_kv(Kc, Vc, P)
    dec.dispatch(first, P)
    device_taps = dec.last_taps  # [X_SLOTS, K], slot k = output after k layers

    np.savez(out_path, taps=device_taps, prompt=np.array(prompt), first=first, P=P)
    print(f"[hidden_taps_device] saved {device_taps.shape} taps -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
