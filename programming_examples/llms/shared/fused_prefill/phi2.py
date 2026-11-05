# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Phi-2 (affine-Q4 weights, affine.py) on the fused prefill device.

One biased LayerNorm feeds both the attention and the MLP, and both add to the
residual:

  h = layernorm(x);  x += dense(attention(h)) + fc2(gelu(fc1(h)))

with biases on q, k, v, dense, fc1, fc2 and the LM head, gelu in its tanh form
and rope on the first dims of each head (partial_rotary_factor).

The attention runs at head dim Desc.attn_dh. A head's dh values sit in its
attn_dh slot at column (h * dh) % 32, the rest zero, as in the fused decode's
KV. That keeps every column of dense's padded input at its place within a
32-column group, and each padded group draws on one group of the original, so
the canonical Q4 dense loads unchanged.
"""

import numpy as np
from ml_dtypes import bfloat16

from . import device as D
from . import hostops as H
from .affine import AffinePrefill
from .engine import TO
from .models import layer_op


def head_cols(heads, dh, adh):
    """The column in [heads * adh] of each of the heads * dh head values."""
    h, d = np.indices((heads, dh))
    return (h * adh + (h * dh) % 32 + d).ravel()


def spread_cols(t, cols, cmap):
    """Q4 triple t [rows, K] with column j at cmap[j] of [rows, cols], and the
    other columns zero codes."""
    q, s, m = t
    dst, src = cmap // 32, np.arange(q.shape[1]) // 32
    pairs = np.unique(np.stack([dst, src]), axis=1)
    if len(np.unique(pairs[0])) != pairs.shape[1]:
        raise ValueError("a padded group would draw on two groups")
    out = [np.zeros((q.shape[0], cols), q.dtype)]
    out[0][:, cmap] = q
    for a in (s, m):
        z = np.zeros((q.shape[0], cols // 32), a.dtype)
        z[:, pairs[0]] = a[:, pairs[1]]
        out.append(z)
    return tuple(out)


def place(a, h0, dh):
    """a [heads, rows, adh] holds heads h0.. at column 0 of their slots;
    move each to column ((h0 + h) * dh) % 32, as head_cols."""
    off = ((h0 + np.arange(a.shape[0])) * dh) % 32
    for o in np.unique(off[off > 0]):
        i = np.flatnonzero(off == o)
        a[i, :, o : o + dh] = a[i, :, :dh]
        a[i, :, :o] = 0


def _f32(a):
    return np.ascontiguousarray(a, np.float32)


def layernorm(x, w, b, eps):
    x = np.asarray(x, np.float32)
    m = x.mean(-1, keepdims=True)
    v = np.square(x - m).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


class Phi2Prefill(AffinePrefill):
    FINAL_NORM = "model.final_layernorm.weight"

    def read_config(self, path):
        c = super().read_config(path)
        e = self.desc
        # the attention scales by 1 / sqrt(attn_dh); the model's is 1 / sqrt(dh)
        self.q_scale = float(np.sqrt(e.attn_dh / e.dh))
        return c

    def load_bundle(self, b):
        e = self.desc
        self.bundle = b
        dq, k = e.heads * e.dh, e.heads * e.attn_dh
        cols = head_cols(e.heads, e.dh, e.attn_dh)
        self.layers = []
        for L in range(e.layers):
            p = f"model.layers.{L}."
            raw = [b.q4nx(p + f"self_attn.{n}_proj.weight", dq, e.d) for n in "qkv"]
            qkv = [np.concatenate(a) for a in zip(*raw)]
            self.gemm_site(f"{L}.qkv", e.d, 3 * dq, self.pack_q4(*qkv), 1)
            o = spread_cols(b.q4nx(p + "self_attn.dense.weight", e.d, dq), k, cols)
            self.gemm_site(f"{L}.o", k, e.d, self.pack_q4(*o), 1)
            fc1 = b.q4nx(p + "mlp.fc1.weight", e.inter, e.d)
            self.gemm_site(f"{L}.fc1", e.d, e.inter, self.pack_q4(*fc1), 1)
            fc2 = b.q4nx(p + "mlp.fc2.weight", e.d, e.inter)
            self.gemm_site(f"{L}.fc2", e.inter, e.d, self.pack_q4(*fc2), 1)

            def bias(n):
                return _f32(b.bf16(p + n + ".bias"))

            self.layers.append(
                dict(
                    ln=(
                        _f32(b.bf16(p + "input_layernorm.weight")),
                        bias("input_layernorm"),
                    ),
                    qkv=np.concatenate([bias(f"self_attn.{n}_proj") for n in "qkv"]),
                    fc1=bias("mlp.fc1"),
                    # dense's and fc2's: their outputs add to the residual together
                    out=bias("self_attn.dense") + bias("mlp.fc2"),
                )
            )
        self.final_ln = (
            _f32(b.bf16(self.FINAL_NORM)),
            _f32(b.bf16("model.final_layernorm.bias")),
        )
        self.lm = D.n_split(e.vocab)
        raw = b.q4nx("lm_head.weight", e.vocab, e.d)
        for i, (n0, n1) in enumerate(self.lm):
            self.gemm_site(
                f"lm.{i}", e.d, n1 - n0, self.pack_q4(*(a[n0:n1] for a in raw)), 1
            )
        self.lm_bias = _f32(b.bf16("lm_head.bias"))
        self.attn_setup()
        self.kvb = [self.kv_bo(layer_op(e, L)) for L in range(e.layers)]
        self.warm("0.fc1")

    def _logits(self, x):
        h = layernorm(x, *self.final_ln, self.eps)
        z = np.concatenate([self.mm(f"lm.{i}", h) for i in range(len(self.lm))], 1)
        return z + self.lm_bias

    def _mix_all(self, L, x, chunks):
        """Layer L up to its queued dense and fc2 GEMMs, one pair per chunk:
        the LayerNorm feeds qkv and fc1 from one A; the host's GELU runs while
        the device works on the attention."""
        e, ly, TM, TK = self.desc, self.layers[L], D.TM, D.TK
        kvb, op = self.kvb[L], layer_op(e, L)
        pend = []
        for c, (r0, t) in enumerate(chunks):
            a = self.a_chunk(e.d, c)
            H.ln_tile(x[r0 : r0 + t], *ly["ln"], self.eps, a[1], TM, TK)
            a[0].sync(TO)
            pend.append(
                (self.gemm_start(f"{L}.qkv", c), self.gemm_start(f"{L}.fc1", c))
            )
        att = []
        for c, (r0, t) in enumerate(chunks):
            qkv = self.gemm_wait(pend[c][0], t)
            self._qkv_post(L, kvb, c, r0, t, qkv, op)
            att.append(self.attn_start(kvb, c, r0, t))
        mlp = []
        for c, (r0, t) in enumerate(chunks):
            f = self.gemm_wait(pend[c][1], t)
            a = self.a_chunk(e.inter, c)
            H.bias_gelu_tile(f, ly["fc1"], t, e.inter, a[1], TM, TK)
            a[0].sync(TO)
            mlp.append(self.gemm_start(f"{L}.fc2", c))
        k, out = e.heads * e.attn_dh, []
        for c, (r0, t) in enumerate(chunks):
            a = self.a_chunk(k, c)
            h0 = 0
            for p in self.attn_wait(att[c]):
                self.o_read(p, t, h0, k, a[1])
                h0 += p["hg"]
            a[0].sync(TO)
            out.append(self.gemm_start(f"{L}.o", c))
        return list(zip(out, mlp))

    def _ffn_all(self, L, x, chunks, pend):
        """The residual add of layer L's dense and fc2 outputs, in place on x."""
        ly = self.layers[L]
        for c, (r0, t) in enumerate(chunks):
            rows = x[r0 : r0 + t]
            for p in pend[c]:
                o = self.gemm_wait(p, t)
                H.add_rms_tile(rows, o, t, None, self.eps, None, None, D.TM, D.TK)
            rows += ly["out"]

    def _qkv_post(self, L, kvb, c, r0, t, qkv, op):
        e, bias = self.desc, self.layers[L]["qkv"]
        dh, adh, dq = e.dh, e.attn_dh, e.heads * e.dh
        cs, sn = self._rope(op, r0, t)
        rope = (cs, sn, self.rot)
        nk = e.kv_heads * adh
        kb = np.zeros((t, nk), bfloat16)
        vb = np.zeros((t, nk), bfloat16)
        H.head_post(
            qkv, t, dq, e.kv_heads, dh, bias, None, self.eps, rope, 1.0, kb, adh, nk
        )
        H.head_post(
            qkv, t, 2 * dq, e.kv_heads, dh, bias, None, self.eps, None, 1.0, vb, adh, nk
        )
        for a in (kb, vb):
            place(a.reshape(t, e.kv_heads, adh).transpose(1, 0, 2), 0, dh)
        # In Phi-2's last layers every key shares a component large enough
        # that raw scores reach 1e6 while a row's scores differ by thousands:
        # past what bf16 scores and bfp16 operands resolve. The attention reads
        # each key less the prompt's first key, which takes q.k0 off every
        # score of a row and leaves the softmax as it was. The decode gets the
        # keys themselves.
        self.kv[L][0].append(kb)
        self.kv[L][1].append(vb)
        kd = np.empty_like(kb)
        H.sub_row(kb, self.kv[L][0][0][0].astype(np.float32), kd)
        ch = self.attn_chunk(kvb["op"], c)
        h0 = 0
        for p in ch["parts"]:
            hg = p["hg"]
            qm = self.q_stage(p)
            H.head_post(
                qkv,
                t,
                h0 * dh,
                hg,
                dh,
                bias,
                None,
                self.eps,
                rope,
                self.q_scale,
                qm,
                D.M * adh,
                adh,
            )
            place(qm[:, :t], h0, dh)
            if t < D.M:
                qm[:, t:] = 0
            self.q_write(p, qm)
            h0 += hg
        h0 = 0
        for p in kvb["parts"]:
            g = self.cfg.attn[p["a"]]
            b0, rec = r0 // g.lkp, self.kv_rec(g)
            nblk = p["m"].shape[1]
            nb = -(-t // g.lkp)
            for h in range(g.kv_heads):
                self.kv_write(kd, vb, (h0 + h) * adh, t, g, p["m"][h, b0])
                p["bo"].sync(TO, nb * rec * 2, (h * nblk + b0) * rec * 2)
            h0 += g.kv_heads
