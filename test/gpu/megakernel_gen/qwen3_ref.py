#!/usr/bin/env python3
"""Qwen3 forward in numpy, for checking what the generated chain computes.

This reads the Hugging Face checkpoint directly rather than the blob weights.py
writes, so it checks the conversion as well as the chain: if the two agree on
the predicted tokens, the transposes, the fused layouts and the architecture in
gen.py are all right.

    ./qwen3_ref.py /shared/erweiw/qwen3-0.6b 3838,1128,525,498
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from weights import read_safetensors  # noqa: E402


def rmsnorm(x, w, eps):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps) * w


def rope(v, pos, cos, sin):
    # v is [tokens, heads, head_dim]; the tables are [tokens, head_dim] and
    # apply to every head of a token.
    h2 = v.shape[-1] // 2
    rot = np.concatenate([-v[..., h2:], v[..., :h2]], -1)
    c, s = cos[pos][:, None, :], sin[pos][:, None, :]
    return v * c + rot * s


def forward(src, tokens, layers=None):
    cfg = json.loads((Path(src) / "config.json").read_text())
    t = read_safetensors(Path(src) / "model.safetensors")
    L = layers or cfg["num_hidden_layers"]
    dim, inter = cfg["hidden_size"], cfg["intermediate_size"]
    nq, nkv, hd = (cfg["num_attention_heads"], cfg["num_key_value_heads"],
                   cfg["head_dim"])
    eps, theta = cfg["rms_norm_eps"], cfg["rope_theta"]
    g = nq // nkv
    T = len(tokens)

    h2 = hd // 2
    inv = 1.0 / (theta ** (2 * np.arange(h2) / hd))
    ang = np.arange(T)[:, None] * inv[None, :]
    cos = np.concatenate([np.cos(ang)] * 2, 1).astype(np.float32)
    sin = np.concatenate([np.sin(ang)] * 2, 1).astype(np.float32)

    emb = t["model.embed_tokens.weight"]
    x = emb[np.asarray(tokens)].astype(np.float32)
    causal = np.arange(T)[None, :] > np.arange(T)[:, None]

    for l in range(L):
        p = f"model.layers.{l}."
        r = rmsnorm(x, t[p + "input_layernorm.weight"], eps)
        q = (r @ t[p + "self_attn.q_proj.weight"].T).reshape(T, nq, hd)
        k = (r @ t[p + "self_attn.k_proj.weight"].T).reshape(T, nkv, hd)
        v = (r @ t[p + "self_attn.v_proj.weight"].T).reshape(T, nkv, hd)
        q = rmsnorm(q, t[p + "self_attn.q_norm.weight"], eps)
        k = rmsnorm(k, t[p + "self_attn.k_norm.weight"], eps)
        pos = np.arange(T)
        q = rope(q, pos, cos, sin)
        k = rope(k, pos, cos, sin)
        o = np.empty((T, nq, hd), dtype=np.float32)
        for hi in range(nq):
            s = (q[:, hi] @ k[:, hi // g].T) / np.sqrt(hd)
            s = np.where(causal, -np.inf, s)
            s = np.exp(s - s.max(-1, keepdims=True))
            s /= s.sum(-1, keepdims=True)
            o[:, hi] = s @ v[:, hi // g]
        x = x + o.reshape(T, nq * hd) @ t[p + "self_attn.o_proj.weight"].T
        xa = rmsnorm(x, t[p + "post_attention_layernorm.weight"], eps)
        gt = xa @ t[p + "mlp.gate_proj.weight"].T
        up = xa @ t[p + "mlp.up_proj.weight"].T
        # exp(-gt) overflows for very negative gt; the identity below is the
        # same function written so neither branch overflows.
        act = np.where(gt >= 0, gt / (1.0 + np.exp(-np.abs(gt))),
                       gt * np.exp(-np.abs(gt)) / (1.0 + np.exp(-np.abs(gt)))) * up
        x = x + act @ t[p + "mlp.down_proj.weight"].T

    xf = rmsnorm(x, t["model.norm.weight"], eps)
    lm = emb if cfg.get("tie_word_embeddings") else t["lm_head.weight"]
    return xf @ lm.T


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    tokens = [int(v) for v in argv[2].split(",")]
    layers = int(argv[3]) if len(argv) > 3 else None
    lg = forward(argv[1], tokens, layers)
    print("prompt:", tokens)
    print("argmax per position:", lg.argmax(-1).tolist())
    top = np.argsort(lg[-1])[::-1][:5]
    print("last position top-5:",
          [(int(i), round(float(lg[-1, i]), 3)) for i in top])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
