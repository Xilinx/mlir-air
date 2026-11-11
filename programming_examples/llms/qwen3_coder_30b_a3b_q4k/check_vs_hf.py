# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The reference's layers against transformers' Qwen3-MoE decoder layers fed
the same dequantized weights, on a prompt and then decode steps that carry the
KV cache over."""

import argparse
import json
import sys

import numpy as np
import torch

import qwen3_coder_30b_a3b_q4k_weights as W
from reference import Reference, t


def hf_config():
    from huggingface_hub import hf_hub_download
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

    path = hf_hub_download(W.REPO, "config.json", revision=W.REVISION)
    c = Qwen3MoeConfig(**json.load(open(path)))
    c._attn_implementation = "eager"
    return c


def hf_layer(w, cfg, l):
    """transformers' decoder layer l with the bundle's weights."""
    from transformers.models.qwen3_moe import modeling_qwen3_moe as M

    layer = M.Qwen3MoeDecoderLayer(cfg, l).float().eval()
    sd = {}
    sd["input_layernorm.weight"], sd["post_attention_layernorm.weight"] = w.norms(l)
    q, k, v, o, qn, kn = w.attn(l)
    sd["self_attn.q_proj.weight"], sd["self_attn.k_proj.weight"] = q, k
    sd["self_attn.v_proj.weight"], sd["self_attn.o_proj.weight"] = v, o
    sd["self_attn.q_norm.weight"], sd["self_attn.k_norm.weight"] = qn, kn
    sd["mlp.gate.weight"] = w.router(l)
    ex = [w.expert(l, e) for e in range(W.N_EXPERTS)]
    sd["mlp.experts.gate_up_proj"] = np.stack([np.concatenate(e[:2]) for e in ex])
    sd["mlp.experts.down_proj"] = np.stack([e[2] for e in ex])
    w.expert.cache_clear()
    missing = layer.load_state_dict({k: t(v) for k, v in sd.items()}, strict=False)
    assert not missing.missing_keys and not missing.unexpected_keys, missing
    return layer


def cos(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(a @ b / (a.norm() * b.norm()))


@torch.no_grad()
def check(w, cfg, l, x, steps):
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3_moe import modeling_qwen3_moe as M

    layer = hf_layer(w, cfg, l)
    rot = M.Qwen3MoeRotaryEmbedding(cfg)
    cache = DynamicCache(config=cfg)
    ref = Reference(w, layers=[l])
    worst = 1.0
    pos = 0
    for chunk in [x[:-steps]] + [x[i : i + 1] for i in range(len(x) - steps, len(x))]:
        n = chunk.shape[0]
        ids = torch.arange(pos, pos + n)[None]
        mask = None
        if n > 1:
            mask = torch.full((n, pos + n), float("-inf")).triu(pos + 1)[None, None]
        h = layer(
            chunk[None],
            position_embeddings=rot(chunk[None], ids),
            attention_mask=mask,
            position_ids=ids,
            past_key_values=cache,
        )
        h = h[0] if isinstance(h, tuple) else h
        got = ref.layer(l, chunk)
        ref.pos = pos = pos + n
        c = cos(got, h[0])
        # a NaN would never lower the minimum
        worst = min(worst, c) if np.isfinite(c) else -1.0
        print(f"layer {l} tokens {pos - n}..{pos - 1}: cos {c:.7f}")
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,47")
    ap.add_argument("--tokens", type=int, default=12)
    ap.add_argument(
        "--steps", type=int, default=3, help="decode steps after the prompt"
    )
    a = ap.parse_args()
    w = W.Qwen3CoderQ4k()
    cfg = hf_config()
    torch.manual_seed(0)
    # token embeddings as the layer input
    x = t(w.embed(list(range(1000, 1000 + a.tokens))))
    worst = min(check(w, cfg, int(l), x, a.steps) for l in a.layers.split(","))
    ok = worst >= 0.9999
    print("PASS!" if ok else f"failed: worst cos {worst:.7f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
