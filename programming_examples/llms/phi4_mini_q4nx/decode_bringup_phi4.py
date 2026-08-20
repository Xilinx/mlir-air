# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Standalone NPU2 bring-up for the FLM-faithful Phi-4-mini fused decode
# superkernel (fused_decode retuned to Phi-4: MODEL_DIM=3072, DH=128, 32 layers,
# INTERMEDIATE=8192, VOCAB=200064, and PARTIAL rotary -- RoPE over 96 of the 128
# head dims, the rest passed through).
#
# Feeds "The capital of France is" token-by-token THROUGH THE ON-DEVICE DECODE
# (no prefill, no CPU attention): the decode kernel appends each token's roped-K/
# raw-V into the on-device KV cache and runs attention on the AIE array for every
# step. The last step's argmax must be 12650 (" Paris"). Phi-4's tokenizer is
# o200k-based, so both the prompt ids and the Paris id differ from the Llama
# examples, and no BOS is prepended.
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_LLMS = _HERE.parent
_PROG = _LLMS.parent
for _p in (str(_PROG), str(_LLMS), str(_PROG / "fused_decode"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from q4nx_decode_phi4 import FusedDecodePhi4, generate  # noqa: E402
from phi4_mini_q4nx_weights import Q4NX_REPO  # noqa: E402

MODEL_SOURCE = os.environ.get("Q4NX_MODEL_SOURCE", Q4NX_REPO)
PROMPT = [976, 9029, 328, 10128, 382]  # "The capital of France is"
EXPECT = 12650  # " Paris"


def main():
    dec = FusedDecodePhi4(MODEL_SOURCE, templates=_HERE)
    print(
        f"[bringup-phi4] layers={dec.N_LAYERS} K={dec.K} DH={dec.DH} "
        f"rope_w={dec.ROPE_W} vocab={dec.VOCAB_SIZE} ATTN_MAXL={dec.ATTN_MAXL}",
        flush=True,
    )
    logits = None
    for p, t in enumerate(PROMPT):
        logits = dec.dispatch(t, p)
    top = logits.argsort()[-5:][::-1]
    print(f"[bringup-phi4] top-5 ids {list(map(int, top))}")
    got = int(logits.argmax())
    ok = got == EXPECT
    print(
        f"[bringup-phi4] argmax {got} (expect {EXPECT}) -> {'PASS' if ok else 'FAIL'}"
    )
    if ok:
        gen = generate(dec, PROMPT, 12, greedy=True)
        print(f"[bringup-phi4] continuation ids {gen}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
