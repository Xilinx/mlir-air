#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Phase 1 of the DFlash drafter bring-up (docs/DFlashFeasibility.md, "## 8"):
make DECODE_HIDDEN_TAPS real.

DECODE_HIDDEN_TAPS has existed in fused_decode.py since before this session
(commit 472aa0f2) but has only ever been exercised as an IR-diff/build-smoke
gate -- nothing has ever read the tap slots back off a real device and
compared them to a reference. This does that, for the first time.

It builds against a SEPARATE decode template (decode_L15/L16.xclbin under
hidden_taps_test/, DECODE_HIDDEN_TAPS=1) rather than touching the verified
production driver (qwen3_4b_q4nx_inference.py) or its decode_L2047/L2048
templates. HIDDEN_TAPS only changes IR-level addressing in the X DDR buffer
(X_SLOTS = UNI_DEC+1 instead of 1, layer iv reads slot iv / writes slot
iv+1); no kernel .cc changed, so it's a same-kernels, different-xclbin build
(see hidden_taps_test/'s build log).

Method: seed the decode's KV cache from the numpy-prefill oracle (fast, no
NPU needed for prefill), dispatch ONE token at position P (the prefill's own
first generated token, matching PARIS_FIRST), then read back every layer
boundary in the enlarged X BO and compare against a real HF bf16 reference
forward pass's own `output_hidden_states` at the same causal position.

TWO PROCESSES, ON PURPOSE. Loading torch/transformers in the same process as
an open XRT device session segfaults (exit 139) -- confirmed empirically: the
device dispatch alone is fine, and the HF `from_pretrained` call alone is
fine, but doing both in one process crashes right after HF's weight-loading
step. Root cause not isolated (native-library conflict between torch's CPU
runtime and XRT's, most likely) and not worth chasing further when the fix is
free: `hidden_taps_device.py` and `hidden_taps_hf_ref.py` each run as a
`subprocess`, communicating through a small `.npz` file, so neither ever
shares a process with the other's heavy native dependencies.

The 5 layers DFlash actually needs are target_layer_ids=[1,9,17,25,33]
(z-lab/Qwen3-4B-DFlash-b16's config), which land at HF hidden_states indices
[2,10,18,26,34] (utils.py's extract_context_feature: offset=1). AIR's own
slot k is the output after k layers have run (slot 0 = raw embedding), the
same convention HF's hidden_states list uses for indices 1..n_layers-1 (HF
v5.3: index n_layers is the POST-FINAL-NORM last-layer output, not the raw one
-- so slot 36 is intentionally not compared here).
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent

TARGET_LAYER_IDS = [1, 9, 17, 25, 33]
SLOTS_TO_CHECK = sorted({0} | {lid + 1 for lid in TARGET_LAYER_IDS})


def compare(name, device_vec, ref_vec):
    d = device_vec.astype(np.float64)
    r = ref_vec.astype(np.float64)
    cos = float(np.dot(d, r) / (np.linalg.norm(d) * np.linalg.norm(r) + 1e-12))
    rel_l2 = float(np.linalg.norm(d - r) / (np.linalg.norm(r) + 1e-12))
    max_abs = float(np.max(np.abs(d - r)))
    print(f"  {name:<10} cos={cos:.6f}  rel_l2={rel_l2:.4f}  max_abs={max_abs:.3f}")
    return cos, rel_l2


def main():
    with tempfile.TemporaryDirectory() as td:
        device_npz = str(Path(td) / "device_taps.npz")
        hf_npz = str(Path(td) / "hf_taps.npz")

        print(
            "[hidden_taps_verify] === subprocess 1/2: device dispatch ===", flush=True
        )
        r1 = subprocess.run(
            [sys.executable, str(_HERE / "hidden_taps_device.py"), device_npz],
            cwd=str(_HERE),
        )
        if r1.returncode != 0:
            if Path(device_npz).exists():
                # The known post-decode teardown segfault (XRT device/BO teardown
                # ordering, exit 139 / 0xC0000005 on Windows) -- unrelated to
                # correctness, already flagged in docs/DFlashFeasibility.md as an
                # open item. The tap data is written before device teardown runs,
                # so a crash here does not mean the dispatch itself failed.
                print(
                    f"[hidden_taps_verify] subprocess 1 exited {r1.returncode} "
                    "(known post-decode teardown segfault) but device_taps.npz "
                    "was written -- continuing",
                    flush=True,
                )
            else:
                r1.check_returncode()

        print(
            "[hidden_taps_verify] === subprocess 2/2: HF bf16 reference ===", flush=True
        )
        r2 = subprocess.run(
            [
                sys.executable,
                str(_HERE / "hidden_taps_hf_ref.py"),
                device_npz,
                hf_npz,
                ",".join(str(s) for s in SLOTS_TO_CHECK),
            ],
            cwd=str(_HERE),
        )
        if r2.returncode != 0:
            r2.check_returncode()

        with np.load(device_npz) as d:
            device_taps = d["taps"]  # [X_SLOTS, K]
            P = int(d["P"])
        with np.load(hf_npz) as h:
            h = {k: v for k, v in h.items()}

        print(
            f"\n[hidden_taps_verify] comparing {len(SLOTS_TO_CHECK)} slots "
            f"(device HIDDEN_TAPS readback vs HF bf16 reference), position P={P}:"
        )
        all_cos = []
        for s in SLOTS_TO_CHECK:
            cos, rel_l2 = compare(f"slot{s}", device_taps[s], h[str(s)])
            all_cos.append(cos)
        print(f"\n[hidden_taps_verify] mean cosine similarity: {np.mean(all_cos):.6f}")
        ok = all(c > 0.98 for c in all_cos)
        print(
            "[hidden_taps_verify]",
            "PASS" if ok else "FAIL",
            "(threshold: cos > 0.98 per slot, matching Q4NX-vs-bf16 numeric drift "
            "already measured elsewhere in this repo, not bit-exactness)",
        )
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
