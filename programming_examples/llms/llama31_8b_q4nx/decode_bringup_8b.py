# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Standalone NPU2 bring-up for the FLM-faithful Llama-3.1-8B fused decode
# superkernel (fused_decode retuned to 8B: MODEL_DIM=4096, DH=128, 32 layers,
# INTERMEDIATE=14336, untied LM head).
#
# Feeds "The capital city of France is called" token-by-token THROUGH THE ON-DEVICE DECODE
# (no prefill, no CPU attention): the decode kernel appends each token's roped-K/
# raw-V into the on-device KV cache and runs attention on the AIE array for every
# step. The last step's argmax must be 12366 (" Paris"). This validates that the
# 8B on-NPU decode (proj -> rope -> flash-attn -> o -> FFN x32 -> lm_head) is
# numerically correct on hardware -- the faithfulness prerequisite before wiring
# it into the generate loop.
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_LLMS = _HERE.parent
_PROG = _LLMS.parent
for _p in (str(_PROG), str(_LLMS), str(_PROG / "fused_decode"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from q4nx_decode_8b import FusedDecode8B, generate  # noqa: E402
from llama31_8b_q4nx_weights import Q4NX_REPO  # noqa: E402

MODEL_SOURCE = os.environ.get("Q4NX_MODEL_SOURCE", Q4NX_REPO)
# Bare "The capital of France is" is a near-tie for this model (" a" edges out
# " Paris" by 0.06 logits, in the HF bf16 reference too), so gate on the
# "...is called" phrasing, where " Paris" leads by 3.4.
PROMPT = [
    128000,
    791,
    6864,
    3363,
    315,
    9822,
    374,
    2663,
]  # "The capital city of France is called"
EXPECT = 12366  # " Paris"


def main():
    dec = FusedDecode8B(MODEL_SOURCE, templates=_HERE)
    print(
        f"[bringup-8b] layers={dec.N_LAYERS} K={dec.K} DH={dec.DH} "
        f"vocab={dec.VOCAB_SIZE} ATTN_MAXL={dec.ATTN_MAXL}",
        flush=True,
    )
    logits = None
    for p, t in enumerate(PROMPT):
        logits = dec.dispatch(t, p)
    top = logits.argsort()[-5:][::-1]
    print(f"[bringup-8b] top-5 ids {list(map(int, top))}")
    got = int(logits.argmax())
    ok = got == EXPECT
    print(f"[bringup-8b] argmax {got} (expect {EXPECT}) -> {'PASS' if ok else 'FAIL'}")
    if ok:
        gen = generate(dec, PROMPT, 12, greedy=True)
        print(f"[bringup-8b] continuation ids {gen}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
