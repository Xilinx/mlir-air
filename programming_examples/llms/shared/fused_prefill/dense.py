# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Dense decoders (llama / qwen / phi / gemma3) on the fused prefill device.

A model is a Desc (models.py): its bundle and the shape the device is built
for. spec() gives its device config, GEMMs and calibration points;
DensePrefill runs it, taking norm eps and rope from the bundle's config.json.
GQA attention takes columns 0 and 7, half the KV heads each, in one dispatch.
A sliding-window model has two attention ops, full ("f") and sliding ("s"),
which differ only in the window.
"""

import json
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16

from . import device as D
from . import hostops as H
from . import packing as P
from .engine import TO, Engine
from .models import layer_op, spec


def resolve(desc, model=None):
    """model.q4nx of the desc's bundle (or of `model`, a file or directory)."""
    m = Path(model) if model else None
    if m and m.is_file():
        return m
    if m and (m / "model.q4nx").is_file():
        return m / "model.q4nx"
    from huggingface_hub import hf_hub_download

    for f in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        hf_hub_download(model or desc.repo, f)
    return Path(hf_hub_download(model or desc.repo, "model.q4nx"))


def config(desc, path):
    """(eps, {op: inv_freq}, rot, rope_scale) from the bundle's config.json,
    after checking its shape against desc."""
    c = json.loads((Path(path).parent / "config.json").read_text())
    c = c.get("text_config", c)
    got = (
        c["hidden_size"],
        c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"],
        c["num_attention_heads"],
        c["num_key_value_heads"],
        c["intermediate_size"],
        c["num_hidden_layers"],
        c["vocab_size"],
    )
    want = desc[1:8]
    if got != want:
        raise ValueError(f"{path}: config {got} does not match the build's {desc}")
    rot = int(desc.dh * c.get("partial_rotary_factor", 1.0))
    theta = c["rope_theta"]
    inv = 1.0 / theta ** (np.arange(0, rot, 2, dtype=np.float64) / rot)
    rs, scale = c.get("rope_scaling") or {}, 1.0
    kind = rs.get("rope_type", rs.get("type"))
    if kind == "llama3":
        old, lo, hi = (
            rs["original_max_position_embeddings"],
            rs["low_freq_factor"],
            rs["high_freq_factor"],
        )
        wl = 2 * np.pi / inv
        smooth = (old / wl - lo) / (hi - lo)
        med = (1 - smooth) * inv / rs["factor"] + smooth * inv
        inv = np.where(wl > old / lo, inv / rs["factor"], inv)
        inv = np.where((wl <= old / lo) & (wl >= old / hi), med, inv)
    elif kind == "longrope":
        # short factors: prompts stay within the original context; HF's
        # cos/sin scaling applies at every length
        orig = c["original_max_position_embeddings"]
        inv = inv / np.asarray(rs["short_factor"], np.float64)
        mp = c["max_position_embeddings"]
        if mp > orig:
            scale = float(np.sqrt(1 + np.log(mp / orig) / np.log(orig)))
    elif kind == "linear":
        inv = inv / rs["factor"]
    elif kind is not None:
        raise ValueError(f"{path}: rope_scaling {kind} is not supported")
    if not desc.window:
        return c["rms_norm_eps"], {"a": inv}, rot, scale
    # the sliding layers rotate with their own base and no scaling
    local = c["rope_local_base_freq"]
    loc = 1.0 / local ** (np.arange(0, rot, 2, dtype=np.float64) / rot)
    return c["rms_norm_eps"], {"f": inv, "s": loc}, rot, scale


def _bf(a):
    return np.asarray(a, np.float32).astype(bfloat16).astype(np.float32)


class DensePrefill(Engine):
    """prefill(ids) returns the last token's logits; kv_stack() the
    [layers, P, kv_heads * dh] roped K and raw V the fused decoder seeds from."""

    def __init__(self, build_dir, desc):
        self.desc = desc
        super().__init__(build_dir, spec(desc).CFG, desc.max_len)
        self.current_context_length = 0
        self.rope_cache = {}
        self.kv = []

    def load_weights(self, model=None):
        e = self.desc
        path = resolve(e, model)
        self.eps, self.inv_freq, self.rot, self.rope_scale = config(e, path)
        norms = {"input": "input_layernorm", "post_attn": "post_attention_layernorm"}
        if e.gemma:
            norms.update(
                pre_ffn="pre_feedforward_layernorm",
                post_ffn="post_feedforward_layernorm",
            )
        b = self.bundle = P.Bundle(path)
        dq, dk = e.heads * e.dh, e.kv_heads * e.dh
        self.norms, self.bias = [], []
        for L in range(e.layers):
            p = f"model.layers.{L}."
            raw = [
                b.q4nx(p + f"self_attn.{w}_proj.weight", r, e.d)
                for w, r in (("q", dq), ("k", dk), ("v", dk))
            ]
            qkv = [np.concatenate(a) for a in zip(*raw)]
            self.gemm_site(f"{L}.qkv", e.d, dq + 2 * dk, P.pack_q4(*qkv), 1)
            for name, t, rows, k in (
                ("o", "self_attn.o_proj", e.d, dq),
                ("gate", "mlp.gate_proj", e.inter, e.d),
                ("up", "mlp.up_proj", e.inter, e.d),
                ("down", "mlp.down_proj", e.d, e.inter),
            ):
                raw = b.q4nx(p + t + ".weight", rows, k)
                self.gemm_site(f"{L}.{name}", k, rows, P.pack_q4(*raw), 1)
            nm = {k: b.bf16(p + n + ".weight") for k, n in norms.items()}
            if e.qk_norm:
                nm["q_norm"] = b.bf16(p + "self_attn.q_norm.weight")
                nm["k_norm"] = b.bf16(p + "self_attn.k_norm.weight")
            self.norms.append(nm)
            if e.qkv_bias:
                self.bias.append(
                    np.concatenate(
                        [b.bf16(p + f"self_attn.{w}_proj.bias") for w in "qkv"]
                    )
                )
        self.final_norm = b.bf16("model.norm.weight")
        self.lm = D.n_split(e.vocab)
        if e.tied:
            emb = b.bf16("model.embed_tokens.weight")
        else:
            raw = b.q4nx("lm_head.weight", e.vocab, e.d)
        for i, (n0, n1) in enumerate(self.lm):
            if e.tied:
                self.gemm_site(f"lm.{i}", e.d, n1 - n0, P.pack_bf16(emb[n0:n1]), 0)
            else:
                self.gemm_site(
                    f"lm.{i}", e.d, n1 - n0, P.pack_q4(*(a[n0:n1] for a in raw)), 1
                )
        self.attn_setup()
        self.kvb = [self.kv_bo(layer_op(e, L)) for L in range(e.layers)]
        self.warm("0.up")

    def _rope(self, op, r0, t):
        if (op, r0, t) not in self.rope_cache:
            ang = np.arange(r0, r0 + t)[:, None] * self.inv_freq[op][None, :]
            self.rope_cache[(op, r0, t)] = tuple(
                np.ascontiguousarray(_bf(f(ang) * self.rope_scale))
                for f in (np.cos, np.sin)
            )
        return self.rope_cache[(op, r0, t)]

    def prefill(self, ids):
        self.resume()
        ids = [int(t) for t in np.asarray(ids).reshape(-1)]
        n = len(ids)
        if n > self.max_len:
            raise ValueError(f"prompt of {n} tokens exceeds max_len {self.max_len}")
        emb = self._embed(ids)
        self.kv = [([], []) for _ in range(self.desc.layers)]
        x = np.array(emb, np.float32, order="C")
        chunks = [(r0, min(D.M, n - r0)) for r0 in range(0, n, D.M)]
        for L in range(self.desc.layers):
            self._ffn_all(L, x, chunks, self._mix_all(L, x, chunks))
        self.current_context_length = n
        last = H.rms(x[-1:], self.final_norm, self.eps)
        return np.concatenate(
            [self.mm(f"lm.{i}", last)[0] for i in range(len(self.lm))]
        )

    def _mix_all(self, L, x, chunks):
        """Attention of layer L over the prompt x [n, d], up to the queued
        output projections, one per chunk. Each stage queues one dispatch per
        chunk, so the host's work on a chunk overlaps the device's on the
        chunks after it."""
        e, nm, eps = self.desc, self.norms[L], self.eps
        dq, dk = e.heads * e.dh, e.kv_heads * e.dh
        op, TM, TK = layer_op(e, L), D.TM, D.TK
        kvb = self.kvb[L]
        rows = [x[r0 : r0 + t] for r0, t in chunks]
        pend = []
        for c, (r0, t) in enumerate(chunks):
            a = self.a_chunk(e.d, c)
            H.rms_tile(rows[c], nm["input"], eps, a[1], TM, TK)
            a[0].sync(TO)
            pend.append(self.gemm_start(f"{L}.qkv", c))
        att = []
        for c, (r0, t) in enumerate(chunks):
            qkv = self.gemm_wait(pend[c], t)
            self._qkv_post(L, kvb, c, r0, t, qkv, op)
            att.append(self.attn_start(kvb, c, r0, t))
        pend = []
        for c, (r0, t) in enumerate(chunks):
            o = self.attn_wait(att[c])
            a = self.a_chunk(dq, c)
            h0 = 0
            for om in o:
                H.o_tile(om, t, h0, dq, a[1], TM, TK)
                h0 += om.shape[0]
            a[0].sync(TO)
            pend.append(self.gemm_start(f"{L}.o", c))
        return pend

    def _ffn_all(self, L, x, chunks, pend):
        """The rest of layer L: the residual adds and the FFN, in place on x;
        pend are the queued output projections of _mix_all."""
        e, nm, eps, TM, TK = self.desc, self.norms[L], self.eps, D.TM, D.TK
        rows = [x[r0 : r0 + t] for r0, t in chunks]
        post = nm.get("post_attn") if e.gemma else None
        pre = nm["pre_ffn"] if e.gemma else nm["post_attn"]
        gu = []
        for c, (r0, t) in enumerate(chunks):
            o = self.gemm_wait(pend[c], t)
            a = self.a_chunk(e.d, c)
            H.add_rms_tile(rows[c], o, t, post, eps, pre, a[1], TM, TK)
            a[0].sync(TO)
            gu.append(
                (self.gemm_start(f"{L}.gate", c, 1), self.gemm_start(f"{L}.up", c))
            )
        pend = []
        for c, (r0, t) in enumerate(chunks):
            g = self.gemm_wait(gu[c][0], t)
            u = self.gemm_wait(gu[c][1], t)
            a = self.a_chunk(e.inter, c)
            H.glu_tile(g, u, t, e.inter, a[1], TM, TK)
            a[0].sync(TO)
            pend.append(self.gemm_start(f"{L}.down", c))
        post = nm.get("post_ffn") if e.gemma else None
        for c, (r0, t) in enumerate(chunks):
            f = self.gemm_wait(pend[c], t)
            H.add_rms_tile(rows[c], f, t, post, eps, None, None, TM, TK)

    def _qkv_post(self, L, kvb, c, r0, t, qkv, op):
        """q into chunk c's attention buffers, K/V into the layer's records
        and the kept cache."""
        e, nm, eps = self.desc, self.norms[L], self.eps
        dq, dk, dh = e.heads * e.dh, e.kv_heads * e.dh, e.dh
        cs, sn = self._rope(op, r0, t)
        rope = (cs, sn, self.rot)
        bias = self.bias[L] if e.qkv_bias else None
        ch = self.attn_chunk(kvb["op"], c)
        h0 = 0
        for p in ch["parts"]:
            qm = p["qm"]
            hg = qm.shape[0]
            H.head_post(
                qkv,
                t,
                h0 * dh,
                hg,
                dh,
                bias,
                nm.get("q_norm"),
                eps,
                rope,
                1.0,
                qm,
                D.M * dh,
                dh,
            )
            if t < D.M:
                qm[:, t:] = 0
            h0 += hg
        kb = np.empty((t, dk), bfloat16)
        vb = np.empty((t, dk), bfloat16)
        H.head_post(
            qkv,
            t,
            dq,
            e.kv_heads,
            dh,
            bias,
            nm.get("k_norm"),
            eps,
            rope,
            1.0,
            kb,
            dh,
            dk,
        )
        H.head_post(
            qkv, t, dq + dk, e.kv_heads, dh, bias, None, eps, None, 1.0, vb, dh, dk
        )
        self.kv[L][0].append(kb)
        self.kv[L][1].append(vb)
        h0 = 0
        for p in kvb["parts"]:
            g = self.cfg.attn[p["a"]]
            b0, rec = r0 // g.lkp, D.kv_rec(g)
            nblk = p["m"].shape[1]
            nb = -(-t // g.lkp)
            for h in range(g.kv_heads):
                H.kv_rec(kb, vb, (h0 + h) * dh, t, dh, g.lkp, g.dvt, p["m"][h, b0])
                p["bo"].sync(TO, nb * rec * 2, (h * nblk + b0) * rec * 2)
            h0 += g.kv_heads

    def _embed(self, ids):
        return self.bundle.rows("model.embed_tokens.weight", ids)

    def get_current_context_length(self):
        return self.current_context_length

    def get_k_cache(self, L, idx=None):
        k = self.kv_view(L)[0]
        return k if idx is None else k[idx]

    def get_v_cache(self, L, idx=None):
        v = self.kv_view(L)[1]
        return v if idx is None else v[idx]

    def kv_view(self, L):
        return tuple(np.concatenate(a).astype(np.float32) for a in self.kv[L])

    def kv_stack(self):
        kv = [self.kv_view(L) for L in range(len(self.kv))]
        return tuple(np.stack([a[i] for a in kv]) for i in range(2))

    def clear_context(self):
        self.current_context_length = 0
        self.kv = []


def load(name, build_dir, model=None):
    """The fused prefill of model `name` (models.py) from its build, with
    weights loaded from `model` (default: its repo)."""
    from .models import MODELS

    desc = MODELS[name]
    if desc.family == "lfm2":
        from .lfm2 import Lfm2Prefill as cls
    else:
        cls = DensePrefill
    pf = cls(build_dir, desc)
    pf.load_weights(model)
    return pf


def reference(desc, model, ids):
    """Last-token logits of the prompt in fp32 numpy, from the same Q4NX
    weights: the gate the fused prefill is checked against."""
    path = resolve(desc, model)
    eps, inv, rot, scale = config(desc, path)
    b = P.Bundle(path)
    e, n = desc, len(ids)
    dq, dk, G = e.heads * e.dh, e.kv_heads * e.dh, e.heads // e.kv_heads

    def w4(name, rows, k):
        q, s, m = b.q4nx(name, rows, k)
        return (q * np.repeat(s, 32, 1) + np.repeat(m, 32, 1)).astype(np.float32)

    def rms(x, w):
        return x / np.sqrt(np.mean(x * x, -1, keepdims=True) + eps) * w

    def rope(x, op):
        ang = np.arange(n)[:, None, None] * inv[op][None, None, :]
        c, s, h = np.cos(ang) * scale, np.sin(ang) * scale, rot // 2
        a, bb = x[..., :h].copy(), x[..., h:rot].copy()
        x[..., :h], x[..., h:rot] = a * c - bb * s, bb * c + a * s
        return x

    def act(g):
        if e.gemma:
            return 0.5 * g * (1 + np.tanh(np.sqrt(2 / np.pi) * (g + 0.044715 * g**3)))
        return g / (1 + np.exp(-g))

    x = b.rows("model.embed_tokens.weight", ids)
    pos = np.arange(n)
    for L in range(e.layers):
        p = f"model.layers.{L}."
        op = layer_op(e, L)
        masked = pos[None] > pos[:, None]
        if op == "s":
            masked |= pos[None] < pos[:, None] - e.window
        h = rms(x, b.bf16(p + "input_layernorm.weight"))
        qkv = np.concatenate(
            [
                h @ w4(p + f"self_attn.{w}_proj.weight", r, e.d).T
                for w, r in (("q", dq), ("k", dk), ("v", dk))
            ],
            1,
        )
        if e.qkv_bias:
            qkv += np.concatenate(
                [b.bf16(p + f"self_attn.{w}_proj.bias") for w in "qkv"]
            )
        q = qkv[:, :dq].reshape(n, e.heads, e.dh)
        k = qkv[:, dq : dq + dk].reshape(n, e.kv_heads, e.dh)
        v = qkv[:, dq + dk :].reshape(n, e.kv_heads, e.dh)
        if e.qk_norm:
            q = rms(q, b.bf16(p + "self_attn.q_norm.weight"))
            k = rms(k, b.bf16(p + "self_attn.k_norm.weight"))
        q, k = rope(q, op), rope(k, op)
        o = np.empty((n, e.heads, e.dh), np.float32)
        for hq in range(e.heads):
            s = q[:, hq] @ k[:, hq // G].T / np.sqrt(e.dh)
            s = np.where(masked, -np.inf, s)
            s = np.exp(s - s.max(-1, keepdims=True))
            o[:, hq] = s / s.sum(-1, keepdims=True) @ v[:, hq // G]
        a = o.reshape(n, dq) @ w4(p + "self_attn.o_proj.weight", e.d, dq).T
        if e.gemma:
            a = rms(a, b.bf16(p + "post_attention_layernorm.weight"))
            h = rms(x + a, b.bf16(p + "pre_feedforward_layernorm.weight"))
        else:
            h = rms(x + a, b.bf16(p + "post_attention_layernorm.weight"))
        x = x + a
        g = h @ w4(p + "mlp.gate_proj.weight", e.inter, e.d).T
        u = h @ w4(p + "mlp.up_proj.weight", e.inter, e.d).T
        f = (act(g) * u) @ w4(p + "mlp.down_proj.weight", e.d, e.inter).T
        if e.gemma:
            f = rms(f, b.bf16(p + "post_feedforward_layernorm.weight"))
        x = x + f
    last = rms(x[-1:], b.bf16("model.norm.weight"))
    if e.tied:
        head = b.bf16("model.embed_tokens.weight")
    else:
        head = w4("lm_head.weight", e.vocab, e.d)
    return (last @ head.T)[0]
