# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The Nemotron-3.5-Bolt-E4B DFlash drafter's weights, from NVIDIA's BF16
GGUF, and a NumPy forward of it.

The draft has no embedding or LM head of its own: the caller embeds the block
with the target's table and runs the target's LM head on what propose()
returns. Context features (the target's hidden states after the tap layers,
concatenated) go through fc and a norm once, as they are accepted; every layer
projects its context K/V from them.
"""

import os

import numpy as np


def _gguf_path():
    from model_paths import DRAFT_FILE, default_draft_snapshot

    return default_draft_snapshot() / DRAFT_FILE


def _f32(t):
    shape = tuple(int(s) for s in reversed(t.shape))
    raw = np.asarray(t.data)
    if t.tensor_type.name == "F32":
        return raw.reshape(shape).astype(np.float32)
    assert t.tensor_type.name == "BF16", t.tensor_type.name
    u = np.frombuffer(raw.tobytes(), np.uint16).reshape(shape)
    return (u.astype(np.uint32) << 16).view(np.float32)


def rms(x, w, eps):
    x = np.asarray(x, np.float32)
    y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return y if w is None else y * w


def rope(x, pos, theta):
    """Half-split rope on [t, heads, dh] at positions pos."""
    half = x.shape[-1] // 2
    inv = 1.0 / theta ** (np.arange(half, dtype=np.float64) / half)
    a = np.asarray(pos, np.float64)[:, None] * inv
    c = np.cos(a).astype(np.float32)[:, None]
    s = np.sin(a).astype(np.float32)[:, None]
    lo, hi = x[..., :half], x[..., half:]
    return np.concatenate([lo * c - hi * s, lo * s + hi * c], axis=-1)


class Draft:
    def __init__(self, path=None):
        import gguf

        r = gguf.GGUFReader(str(path or _gguf_path()))
        f = lambda k: r.fields[k].contents()  # noqa: E731
        self.n_layers = f("dflash.block_count")
        self.heads = f("dflash.attention.head_count")
        self.kv_heads = f("dflash.attention.head_count_kv")
        self.dh = f("dflash.attention.key_length")
        self.eps = float(f("dflash.attention.layer_norm_rms_epsilon"))
        self.theta = float(f("dflash.rope.freq_base"))
        self.block = f("dflash.block_size")
        self.window = f("dflash.attention.sliding_window")
        # hidden_states indices, so layer ids + 1
        self.tap_layers = [i - 1 for i in f("dflash.target_layers")]
        assert not f("dflash.attention.causal")
        w = {t.name: _f32(t) for t in r.tensors}
        self.fc, self.hidden_norm = w["fc.weight"], w["enc.output_norm.weight"]
        self.norm = w["output_norm.weight"]
        self.layers = [
            {
                k: w[f"blk.{i}.{k}.weight"]
                for k in (
                    "attn_norm",
                    "attn_q",
                    "attn_k",
                    "attn_v",
                    "attn_q_norm",
                    "attn_k_norm",
                    "attn_output",
                    "ffn_norm",
                    "ffn_gate",
                    "ffn_up",
                    "ffn_down",
                )
            }
            for i in range(self.n_layers)
        ]
        self.reset()

    def reset(self):
        self.ck = [[] for _ in self.layers]
        self.cv = [[] for _ in self.layers]
        self.n_ctx = 0

    def _kv(self, lw, x, pos):
        k = (x @ lw["attn_k"].T).reshape(len(x), self.kv_heads, self.dh)
        k = rope(rms(k, lw["attn_k_norm"], self.eps), pos, self.theta)
        v = (x @ lw["attn_v"].T).reshape(len(x), self.kv_heads, self.dh)
        return k, v

    def add_context(self, taps):
        """taps [m, n_taps * D]: the target's features of the next m positions."""
        pos = np.arange(self.n_ctx, self.n_ctx + len(taps))
        feat = rms(np.asarray(taps, np.float32) @ self.fc.T, self.hidden_norm, self.eps)
        for i, lw in enumerate(self.layers):
            k, v = self._kv(lw, feat, pos)
            self.ck[i].append(k)
            self.cv[i].append(v)
        self.n_ctx += len(taps)

    def propose(self, emb):
        """emb [block, D]: the block's embeddings (anchor token, then mask
        tokens) at the positions after the context. Returns the final-norm
        hidden states [block, D]."""
        h = np.asarray(emb, np.float32)
        t = len(h)
        pos = np.arange(self.n_ctx, self.n_ctx + t)
        g = self.heads // self.kv_heads
        for i, lw in enumerate(self.layers):
            x = rms(h, lw["attn_norm"], self.eps)
            q = (x @ lw["attn_q"].T).reshape(t, self.heads, self.dh)
            q = rope(rms(q, lw["attn_q_norm"], self.eps), pos, self.theta)
            k, v = self._kv(lw, x, pos)
            k = np.concatenate(self.ck[i] + [k])[-self.window :]
            v = np.concatenate(self.cv[i] + [v])[-self.window :]
            # [kv_heads, g, t, dh] x [kv_heads, n, dh]
            qh = q.reshape(t, self.kv_heads, g, self.dh).transpose(1, 2, 0, 3)
            s = np.einsum("hgtd,nhd->hgtn", qh, k) / np.sqrt(self.dh)
            s = np.exp(s - s.max(-1, keepdims=True))
            s /= s.sum(-1, keepdims=True)
            o = np.einsum("hgtn,nhd->thgd", s, v).reshape(t, -1)
            h = h + o @ lw["attn_output"].T
            x = rms(h, lw["ffn_norm"], self.eps)
            gt, up = x @ lw["ffn_gate"].T, x @ lw["ffn_up"].T
            h = h + (gt / (1 + np.exp(-gt)) * up) @ lw["ffn_down"].T
        return rms(h, self.norm, self.eps)
