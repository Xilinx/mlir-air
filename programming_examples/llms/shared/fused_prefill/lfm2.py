# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""LFM2 on the fused prefill device. Its attention layers run as in dense.py;
a ShortConv layer's projections run on the GEMM engine and its gating and
causal depthwise convolution on the host, carrying the last conv_L_cache - 1
rows of the gated signal across chunks.

The weights are AIR's Q4_0 requantization of the HF checkpoint, the values
lfm2_1_2b_q4nx's decode runs on, given to the affine int4 GEMM as q + 8 with
min = -8 scale.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ml_dtypes import bfloat16

# q4_0_codec imports its fused_decode siblings top-level
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "fused_decode"))
from q4_0_codec import HFModel, requant_q4_0  # noqa: E402

from . import device as D  # noqa: E402
from . import hostops as H  # noqa: E402
from . import packing as P  # noqa: E402
from .engine import TO  # noqa: E402
from .dense import DensePrefill  # noqa: E402


def resolve(desc, model=None):
    """The HF checkpoint's model.safetensors (its directory holds the config
    and tokenizer)."""
    m = Path(model) if model else None
    if m and m.is_dir():
        return m / "model.safetensors"
    if m and m.is_file():
        return m
    from huggingface_hub import snapshot_download

    d = snapshot_download(desc.repo, allow_patterns=["*.safetensors", "*.json"])
    return Path(d) / "model.safetensors"


def config(desc, path):
    c = json.loads((Path(path).parent / "config.json").read_text())
    got = (c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"])
    if got != (desc.d, desc.heads, desc.kv_heads) or c["vocab_size"] != desc.vocab:
        raise ValueError(f"{path}: config does not match the build's {desc}")
    inv = 1.0 / c["rope_theta"] ** (np.arange(0, desc.dh, 2) / desc.dh)
    return c["norm_eps"], inv, set(c["full_attn_idxs"]), c["conv_L_cache"] - 1


def q4(hf, name):
    """An [out, in] projection as raw (q, scale, min) for pack_q4."""
    q, sc = requant_q4_0(hf.bf16(name))
    sc = sc.astype(bfloat16).astype(np.float32)
    return (q.view(np.int8) + 8).astype(np.uint8), sc, -8 * sc


class Lfm2Prefill(DensePrefill):
    def load_weights(self, model=None):
        e = self.desc
        path = resolve(e, model)
        self.eps, inv, self.attn_layers, self.halo = config(e, path)
        self.inv_freq, self.rot, self.rope_scale = {"a": inv}, e.dh, 1.0
        hf = HFModel(str(path.parent))
        dq, dk = e.heads * e.dh, e.kv_heads * e.dh
        self.norms, self.taps = [], {}
        for L in range(e.layers):
            p = f"model.layers.{L}."
            nm = {
                "input": hf.bf16(p + "operator_norm.weight"),
                "post_attn": hf.bf16(p + "ffn_norm.weight"),
            }
            if L in self.attn_layers:
                raw = [q4(hf, p + f"self_attn.{w}_proj.weight") for w in "qkv"]
                qkv = [np.concatenate(a) for a in zip(*raw)]
                self.gemm_site(f"{L}.qkv", e.d, dq + 2 * dk, P.pack_q4(*qkv), 1)
                o = q4(hf, p + "self_attn.out_proj.weight")
                self.gemm_site(f"{L}.o", dq, e.d, P.pack_q4(*o), 1)
                nm["q_norm"] = hf.bf16(p + "self_attn.q_layernorm.weight")
                nm["k_norm"] = hf.bf16(p + "self_attn.k_layernorm.weight")
            else:
                w = q4(hf, p + "conv.in_proj.weight")
                self.gemm_site(f"{L}.in", e.d, 3 * e.d, P.pack_q4(*w), 1)
                w = q4(hf, p + "conv.out_proj.weight")
                self.gemm_site(f"{L}.out", e.d, e.d, P.pack_q4(*w), 1)
                self.taps[L] = hf.bf16(p + "conv.conv.weight").reshape(e.d, -1).T
            for name, w, rows, k in (
                ("gate", "w1", e.inter, e.d),
                ("up", "w3", e.inter, e.d),
                ("down", "w2", e.d, e.inter),
            ):
                raw = q4(hf, p + f"feed_forward.{w}.weight")
                self.gemm_site(f"{L}.{name}", k, rows, P.pack_q4(*raw), 1)
            self.norms.append(nm)
        self.final_norm = hf.bf16("model.embedding_norm.weight")
        self.embed = hf.bf16("model.embed_tokens.weight")
        self.lm = D.n_split(e.vocab)
        for i, (n0, n1) in enumerate(self.lm):
            self.gemm_site(f"lm.{i}", e.d, n1 - n0, P.pack_bf16(self.embed[n0:n1]), 0)
        self.attn_setup()
        self.kvb = {L: self.kv_bo("a") for L in self.attn_layers}
        # the shape lfm2_1_2b_q4nx's driver reads to lay out its decode seed
        self.config = SimpleNamespace(
            n_layers=e.layers,
            kv_dim=dk,
            conv_dim=e.d,
            conv_L_cache=self.halo + 1,
            is_attn_layer=lambda L: L in self.attn_layers,
        )

    def _embed(self, ids):
        self.state = {
            L: np.zeros((self.halo, self.desc.d), np.float32) for L in self.taps
        }
        return self.embed[np.asarray(ids)]

    def _mix_all(self, L, x, chunks):
        if L in self.attn_layers:
            return super()._mix_all(L, x, chunks)
        nm, d, TM, TK = self.norms[L], self.desc.d, D.TM, D.TK
        pend = []
        for c, (r0, t) in enumerate(chunks):
            a = self.a_chunk(d, c)
            H.rms_tile(x[r0 : r0 + t], nm["input"], self.eps, a[1], TM, TK)
            a[0].sync(TO)
            pend.append(self.gemm_start(f"{L}.in", c))
        taps, out = self.taps[L], []
        for c, (r0, t) in enumerate(chunks):
            bcx = H.bf16_to_f32(self.gemm_wait(pend[c], t), t, 3 * d)
            b, cx, v = bcx[:, :d], bcx[:, d : 2 * d], bcx[:, 2 * d :]
            g = np.concatenate([self.state[L], b * v])
            y = sum(taps[j] * g[j : len(g) - self.halo + j] for j in range(len(taps)))
            self.state[L] = g[-self.halo :]
            a = self.a_chunk(d, c)
            H.tile_a(cx * y, a[1], TM, TK)
            a[0].sync(TO)
            out.append(self.gemm_start(f"{L}.out", c))
        return out

    def kv_view(self, L):
        if L not in self.attn_layers:
            return None, None
        return super().kv_view(L)

    def get_conv_state(self, L):
        """[conv_L_cache - 1, conv_dim] carried state of ShortConv layer L."""
        return self.state[L].astype(bfloat16)


def reference(desc, model, ids):
    """Last-token logits of the prompt in fp32 numpy, from the same Q4_0
    weights."""
    path = resolve(desc, model)
    eps, inv, attn_layers, halo = config(desc, path)
    hf = HFModel(str(path.parent))
    e, n, G = desc, len(ids), desc.heads // desc.kv_heads

    def w4(name):
        q, s, m = q4(hf, name)
        return q * np.repeat(s, 32, 1) + np.repeat(m, 32, 1)

    def rms(x, w):
        return x / np.sqrt(np.mean(x * x, -1, keepdims=True) + eps) * w

    def rope(x):
        ang = np.arange(n)[:, None, None] * inv[None, None, :]
        c, s, h = np.cos(ang), np.sin(ang), e.dh // 2
        a, b = x[..., :h].copy(), x[..., h:].copy()
        return np.concatenate([a * c - b * s, b * c + a * s], -1)

    x = hf.bf16("model.embed_tokens.weight")[np.asarray(ids)]
    causal = np.arange(n)[None] > np.arange(n)[:, None]
    for L in range(e.layers):
        p = f"model.layers.{L}."
        h = rms(x, hf.bf16(p + "operator_norm.weight"))
        if L in attn_layers:
            q = (h @ w4(p + "self_attn.q_proj.weight").T).reshape(n, e.heads, e.dh)
            k = (h @ w4(p + "self_attn.k_proj.weight").T).reshape(n, e.kv_heads, e.dh)
            v = (h @ w4(p + "self_attn.v_proj.weight").T).reshape(n, e.kv_heads, e.dh)
            q = rope(rms(q, hf.bf16(p + "self_attn.q_layernorm.weight")))
            k = rope(rms(k, hf.bf16(p + "self_attn.k_layernorm.weight")))
            o = np.empty((n, e.heads, e.dh), np.float32)
            for hq in range(e.heads):
                s = q[:, hq] @ k[:, hq // G].T / np.sqrt(e.dh)
                s = np.exp(np.where(causal, -np.inf, s) - s.max(-1, keepdims=True))
                o[:, hq] = s / s.sum(-1, keepdims=True) @ v[:, hq // G]
            x = x + o.reshape(n, -1) @ w4(p + "self_attn.out_proj.weight").T
        else:
            bcx = h @ w4(p + "conv.in_proj.weight").T
            b, c, v = bcx[:, : e.d], bcx[:, e.d : 2 * e.d], bcx[:, 2 * e.d :]
            g = np.concatenate([np.zeros((halo, e.d)), b * v])
            taps = hf.bf16(p + "conv.conv.weight").reshape(e.d, -1).T
            y = sum(taps[j] * g[j : j + n] for j in range(len(taps)))
            x = x + (c * y) @ w4(p + "conv.out_proj.weight").T
        h = rms(x, hf.bf16(p + "ffn_norm.weight"))
        gt = h @ w4(p + "feed_forward.w1.weight").T
        up = h @ w4(p + "feed_forward.w3.weight").T
        x = x + (gt / (1 + np.exp(-gt)) * up) @ w4(p + "feed_forward.w2.weight").T
    last = rms(x[-1:], hf.bf16("model.embedding_norm.weight"))
    return (last @ hf.bf16("model.embed_tokens.weight").T)[0]
