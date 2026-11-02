# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Qwen3.5-0.8B (text) with FastFlowLM's Q4NX weights on NPU2: prefill and
greedy autoregressive decode.

Weights: model.q4nx of FastFlowLM/Qwen3.5-0.8B-NPU2 (q35_weights).

Prefill: the shared fused_prefill device (shared/fused_prefill/qwen35.py, device
spec q35_spec): every projection GEMM, the softmax attention and the lm_head on
the NPU in 128-row chunks; the element-wise work and the Gated DeltaNet scan on
the host, overlapping the device's work.

Decode: one dispatch per token on programming_examples/fused_decode_gdn
(q35_fdg): all 24 layers and the full-vocabulary lm_head. After prefill the
host hands the prompt's KV rows, conv history and DeltaNet states to it once.

  python3 q35_npu.py --build BUILD --capacity C --prompt "..." --tokens 32
"""

import argparse
import sys
import time
import types
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(HERE.parents[1])
]

from air_examples.llms.qwen35_0_8b_q4nx import q35_spec as S  # noqa: E402
from air_examples.llms.qwen35_0_8b_q4nx import q35_weights as W  # noqa: E402

# The BF16 checkpoint the bundle was quantized from: tokenizer, configuration
# and the verification reference.
HF_REPO = "Qwen/Qwen3.5-0.8B"
HF_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
CAPACITIES = (1024, 2048, 4096, 8192)


def text_config():
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(HF_REPO, revision=HF_REVISION).text_config


def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(HF_REPO, revision=HF_REVISION)


def rotary_table(cfg, n):
    """HF's rotary table of the first n positions, bf16 [n, 64] (cos, sin)."""
    import torch
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5TextRotaryEmbedding,
    )

    with torch.no_grad():
        rot = Qwen3_5TextRotaryEmbedding(config=cfg)
        pos = torch.arange(n).view(1, 1, -1).expand(3, 1, -1)
        cos, sin = rot(torch.zeros(1, n, W.D, dtype=torch.bfloat16), pos)
    return cos[0].float().numpy(), sin[0].float().numpy()


class Qwen35Npu:
    """Prefill on the fused_prefill device, decode on fused_decode_gdn of
    capacity C (the KV cache; prompt + generated tokens <= C)."""

    def __init__(self, build_dir, capacity=1024, bundle=None):
        from air_examples.llms.shared.fused_prefill.qwen35 import Qwen35Prefill
        from air_examples.llms.qwen35_0_8b_q4nx.q35_fdg import FdgDecode

        t0 = time.perf_counter()
        build_dir = Path(build_dir)
        self.cfg = cfg = text_config()
        self.kinds = cfg.layer_types[: cfg.num_hidden_layers]
        self.capacity = capacity
        self.Wt = Wt = W.Q4nx(bundle)
        self.embed = Wt.embed()  # tied: lm_head's Q8 values as bf16
        cos, sin = rotary_table(cfg, max(capacity, S.max_len) + 32)
        self.fp = Qwen35Prefill(
            build_dir / "device", Wt, self.kinds, cos, sin, self.embed
        )
        self.dec = FdgDecode(
            build_dir, capacity, Wt, cos.astype(bfloat16), sin.astype(bfloat16)
        )
        self.times = {}
        self.load_seconds = time.perf_counter() - t0

    def prefill(self, ids):
        """Run the prompt; returns the last position's logits (float32
        [vocab]). The decode state is handed to the decode at its first step."""
        n = len(ids)
        if n + 1 > self.capacity:
            raise ValueError(
                f"prompt of {n} tokens leaves no room in capacity {self.capacity}"
            )
        self.fp.times = {}
        z = self.fp.prefill(ids)
        for k, v in self.fp.times.items():
            self.times["pf_" + k] = v
        self.handoff_pending = True
        self.pos = n
        return z

    def decode(self, token):
        """One step: the token at position self.pos -> next logits."""
        if self.pos >= self.capacity:
            raise ValueError(
                f"position {self.pos} is past the decode capacity {self.capacity}"
            )
        if self.handoff_pending:
            t0 = time.perf_counter()
            self.dec.handoff(*self.fp.decode_state())
            self.handoff_pending = False
            self.times["handoff"] = (
                self.times.get("handoff", 0) + time.perf_counter() - t0
            )
        z = self.dec.step(self.embed[token], self.pos)
        self.times.update(self.dec.times)
        self.pos += 1
        return z

    def generate(self, ids, n_out):
        """Greedy: returns (tokens, logits list, ttft seconds, decode seconds)."""
        self.times = {}
        self.dec.times = {}
        t0 = time.perf_counter()
        z = self.prefill(ids)
        tok = int(np.argmax(z))
        ttft = time.perf_counter() - t0
        toks, logits = [tok], [z]
        t1 = time.perf_counter()
        for _ in range(n_out - 1):
            z = self.decode(tok)
            tok = int(np.argmax(z))
            toks.append(tok)
            logits.append(z)
        return toks, logits, ttft, time.perf_counter() - t1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--capacity", type=int, default=1024, choices=CAPACITIES)
    ap.add_argument(
        "--bundle", default=None, help="model.q4nx (default: the pinned Hub revision)"
    )
    ap.add_argument(
        "--warmup", type=int, default=0, help="untimed requests before the timed one"
    )
    a = ap.parse_args()
    tok = tokenizer()
    ids = tok(a.prompt)["input_ids"]
    m = Qwen35Npu(a.build, a.capacity, a.bundle)
    for _ in range(a.warmup):
        m.generate(ids, a.tokens)
    toks, _, ttft, dt = m.generate(ids, a.tokens)
    print(repr(tok.decode(toks)))
    # the lines bench/extract_perf.py reads
    print(f"Inference: prompt_len={len(ids)}, n_tokens={len(toks)}")
    print(f"Time to first token (TTFT): {ttft:.4f} s")
    print(f"Tokens/second: {(len(toks) - 1) / dt:.2f}")


if __name__ == "__main__":
    main()
