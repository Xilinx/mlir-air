# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The reference's layers against transformers' Qwen3.6 MoE layers
(qwen3_5_moe) fed the same dequantized weights: a Gated DeltaNet layer and a
full-attention layer, on a prompt and then a decode step that carries the
caches over."""

import argparse
import json
import sys

import numpy as np
import torch

import qwen36_35b_a3b_q4nx_weights as W
from reference import Reference, t


def hf_config(w):
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeTextConfig,
    )

    cfg = json.load(open(w.path.parent / "config.json"))
    c = Qwen3_5MoeTextConfig(**cfg.get("text_config", cfg))
    c._attn_implementation = "eager"
    return c


def hf_layer(w, cfg, l):
    """transformers' decoder layer l with the bundle's weights."""
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M

    layer = M.Qwen3_5MoeDecoderLayer(cfg, l).float().eval()
    sd = {}
    n1, n2 = w.norms(l)
    sd["input_layernorm.weight"] = n1 - 1  # the model's RMSNorm takes 1 + w
    sd["post_attention_layernorm.weight"] = n2 - 1
    if W.is_attention(l):
        q, g, k, v, o, qn, kn = w.attn(l)
        # transformers' q_proj rows run [q, gate] per head
        qg = np.stack([q.reshape(W.NQ, W.DH, -1), g.reshape(W.NQ, W.DH, -1)], 1)
        sd["self_attn.q_proj.weight"] = qg.reshape(2 * W.NQ * W.DH, -1)
        sd["self_attn.k_proj.weight"], sd["self_attn.v_proj.weight"] = k, v
        sd["self_attn.o_proj.weight"] = o
        sd["self_attn.q_norm.weight"], sd["self_attn.k_norm.weight"] = qn - 1, kn - 1
    else:
        qkv, z, out, al, be, conv, nrm, a, dt = w.gdn(l)
        p = "linear_attn."
        sd[p + "in_proj_qkv.weight"], sd[p + "in_proj_z.weight"] = qkv, z
        sd[p + "in_proj_a.weight"], sd[p + "in_proj_b.weight"] = al, be
        sd[p + "out_proj.weight"] = out
        sd[p + "conv1d.weight"] = conv[:, None, :]
        sd[p + "norm.weight"] = nrm
        sd[p + "A_log"] = np.log(-a)
        sd[p + "dt_bias"] = dt
    router, sgate = w.router(l)
    sd["mlp.gate.weight"] = router
    sd["mlp.shared_expert_gate.weight"] = sgate[None]
    sg, su, sdn = w.shared(l)
    sd["mlp.shared_expert.gate_proj.weight"] = sg
    sd["mlp.shared_expert.up_proj.weight"] = su
    sd["mlp.shared_expert.down_proj.weight"] = sdn
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
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M
    from transformers.cache_utils import DynamicCache

    layer = hf_layer(w, cfg, l)
    rot = M.Qwen3_5MoeTextRotaryEmbedding(cfg)
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
        worst = min(worst, c)
        kind = "attention" if W.is_attention(l) else "GDN"
        print(f"layer {l} ({kind}) tokens {pos - n}..{pos - 1}: cos {c:.7f}")
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,3")
    ap.add_argument("--tokens", type=int, default=12)
    ap.add_argument(
        "--steps", type=int, default=3, help="decode steps after the prompt"
    )
    a = ap.parse_args()
    w = W.Qwen36Q4nx()
    cfg = hf_config(w)
    torch.manual_seed(0)
    # token embeddings as the layer input
    ids = list(range(1000, 1000 + a.tokens))
    x = t(w.embed(ids))
    worst = min(check(w, cfg, int(l), x, a.steps) for l in a.layers.split(","))
    ok = worst >= 0.9999
    print("PASS!" if ok else f"failed: worst cos {worst:.7f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
