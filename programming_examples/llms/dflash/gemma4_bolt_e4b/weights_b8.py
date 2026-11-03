# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Host-side buffers of the Gemma4-E4B eight-token decoder.

The device computes every layer at the full-attention width (b8_config): a
sliding layer's q and k heads spread as [lo | 0 | hi | 0], so the rope
kernel's (i, i + 256) pairing rotates the head's own (i, i + 128) pairs, its
v heads at the front of their 512-wide slot, and its q/k norm weights
carrying the sqrt(256 / 512) that cancels RMSNorm over the padded head (the
layout of gemma4_e2b_q4nx_requant.py). The weights it streams are each
class's real ones: layer_matrices(compact=True).

The per-layer embedding's gate and projection ship as bf16. One group-32
affine Q4 term loses too much of them for the residual stream to survive
the layers, so each is packed as two terms side by side along K, [Q1 | Q2]
with Q2 the Q4 of what Q1 leaves; the device reads its input twice, [x | x],
and the product is x (Q1 + Q2). Gate rows and projection columns are padded
to 512.

The device has no output scale. It keeps layer L's residual divided by
c_L, the product of the earlier layers' scales (residual_scales): the post
norms' weights take 1/c_L, the per-layer embedding gate, which reads the
residual itself, takes c_L, and the norms do not see it.
"""

import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from b8_config import DH, KV_HEADS, Q_HEADS, PLE_GATE_N, PLE_PAD, ROPE_W_LEN

LLMS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LLMS / "gemma4_e4b_q4nx"))
import gemma4_e4b_variant  # noqa: E402,F401  (GEMMA4_VARIANT=e4b)
import gemma4_e2b_q4nx_weights as gw  # noqa: E402
from gemma4_e2b_q4nx_requant import (  # noqa: E402
    _cat,
    _each,
    _head_perm,
    _interleave512,
    _pad_cols,
    _pad_rows,
)
from air_examples.llms.shared.infra import host_weights  # noqa: E402


def _spread(a, n_heads, dh, axis=0, front=False, slot=DH):
    """Heads of width dh into `slot`-wide slots along `axis`: interleaved for
    q/k, at the front for v (front=True)."""
    if dh == slot:
        return a
    src = np.moveaxis(a, axis, 0)
    out = np.zeros((n_heads * slot,) + src.shape[1:], src.dtype)
    p = np.arange(dh) if front else _head_perm(dh, slot)
    for h in range(n_heads):
        out[h * slot + p] = src[h * dh : (h + 1) * dh]
    return np.moveaxis(out, 0, axis)


def _zero_q4(m, k, group):
    return (
        np.zeros((m, k), np.uint8),
        np.zeros((m, k // group), np.float32),
        np.zeros((m, k // group), np.float32),
    )


def two_term_q4(w, group):
    """[Q1 | Q2] along K: Q1 = Q4(w), Q2 = Q4(w - Q1)."""
    t1 = host_weights.requant_q4k(w, group)
    t2 = host_weights.requant_q4k(w - host_weights.dequant_q4(*t1, group), group)
    return tuple(np.concatenate([a, b], 1) for a, b in zip(t1, t2))


def residual_scales(qm, layers):
    """c_0..c_layers: c_L is the product of the scales of layers below L."""
    c = [1.0]
    for L in range(layers):
        c.append(c[-1] * qm.layer_norms(L)["out_scale"])
    return c


def _half_major(a, slot):
    """Columns of `slot`-wide head slots reordered [half][head][slot / 2]:
    the attention gather's order of the o projection's input."""
    r = a.shape[0]
    return a.reshape(r, Q_HEADS, 2, slot // 2).transpose(0, 2, 1, 3).reshape(r, -1)


def layer_matrices(f, qm, L, c=1.0, compact=False):
    """(q, scale, min) of every projection phase of layer L, [out, in];
    c is the layer's residual scale. By default every layer at the
    full-attention width (the layout the device computes in); compact=True
    gives what the device streams: a sliding layer's q/k/v rows unspread and
    its o projection's real columns, no k/v rows for a KV-shared layer, and
    a full layer's o columns half-major."""
    G = f.GROUP
    w = qm.layer_q4nx(L)
    dh = gw.head_dim(L)
    if compact:
        qkv = [w["q"]] + ([w["k"], w["v"]] if "k" in w else [])
        o = w["o"]
        if dh == DH:
            o = _each(
                o, lambda a: _half_major(a, DH), lambda a: _half_major(a, DH // G)
            )
        return [_cat(*qkv), o] + _ffn_ple_matrices(f, qm, L, c, w)
    q = _each(w["q"], lambda a: _spread(a, Q_HEADS, dh))
    if "k" in w:
        k = _each(w["k"], lambda a: _spread(a, KV_HEADS, dh))
        v = _each(w["v"], lambda a: _spread(a, KV_HEADS, dh, front=True))
    else:
        k = v = _zero_q4(KV_HEADS * DH, f.K, G)
    o = _each(
        w["o"],
        lambda a: _spread(a, Q_HEADS, dh, axis=1, front=True),
        lambda a: _spread(a, Q_HEADS, dh // G, axis=1, front=True, slot=DH // G),
    )
    return [_cat(q, k, v), o] + _ffn_ple_matrices(f, qm, L, c, w)


def _ffn_ple_matrices(f, qm, L, c, w):
    G = f.GROUP
    pw = qm.layer_ple(L)
    gate = _pad_rows(np.asarray(pw["inp_gate"], np.float32) * c, PLE_GATE_N)
    proj = _pad_cols(np.asarray(pw["per_layer_projection"], np.float32), PLE_PAD)
    return [
        _interleave512(w["up"], w["gate"], f.GLU_SLICE // 2),
        w["down"],
        two_term_q4(gate, G),
        two_term_q4(proj, G),
    ]


def pack_layer(f, qm, L, c=1.0):
    """Layer L's slab: its class's weights (b8_config.class_geometry)."""
    mats = layer_matrices(f, qm, L, c, compact=True)
    packed = np.concatenate(
        [
            f.pack_q4k_cascade(*m, f.NCX, f.NCY, iter_major=True, dual_chan=True)
            for m in mats
        ]
    )
    slab = f.W_OFFS[L + 1] - f.W_OFFS[L]
    assert packed.size == slab, (packed.size, slab)
    return packed


def pack_lm_head(f, qm):
    VP, VPF, G = f.VOCAB_SIZE_PADDED, f.UNI_LM * f.VOCAB_SIZE_PADDED, f.GROUP
    lq = np.zeros((VPF, f.K), np.uint8)
    ls = np.zeros((VPF, f.K // G), np.float32)
    lmn = np.zeros((VPF, f.K // G), np.float32)
    for r0 in range(0, gw.VOCAB, 16384):
        r1 = min(r0 + 16384, gw.VOCAB)
        lq[r0:r1], ls[r0:r1], lmn[r0:r1] = qm.lm_head_q4nx(r0, r1)
    return [
        f.pack_q4k_cascade(
            lq[c * VP : (c + 1) * VP],
            ls[c * VP : (c + 1) * VP],
            lmn[c * VP : (c + 1) * VP],
            f.NCX,
            f.NCY,
            iter_major=True,
            dual_chan=True,
        )
        for c in range(f.UNI_LM)
    ]


def norm_slab(f, qm, L, c=1.0):
    """[input | pre_ffn | post_attn | post_ffn | post_ple | unused x16];
    the post norms divided by the layer's residual scale c."""
    nm = qm.layer_norms(L)
    s = np.zeros(f.RMS_LAYER, np.float32)
    for i, key in enumerate(("input", "pre_ffn", "post_attn", "post_ffn", "post_ple")):
        s[i * f.K : (i + 1) * f.K] = np.asarray(nm[key], np.float32) / (
            c if i >= 2 else 1.0
        )
    return s.astype(bfloat16)


def qk_norms(qm, L):
    """q_norm and k_norm laid out like the spread heads, with the padding fix."""
    nm = qm.layer_norms(L)
    dh = gw.head_dim(L)
    fix = 1.0 if dh == DH else float(np.sqrt(dh / DH))
    p = _head_perm(dh, DH)
    out = []
    for key in ("q_norm", "k_norm"):
        a = np.zeros(DH, np.float32)
        a[p] = np.asarray(nm[key], np.float32) * fix
        out.append(a)
    return out


def rope_rows(L, positions, norms, rope_freqs):
    """The rope table rows of layer L at `positions`:
    [cos(256) | sin(256) | q_norm(512) | k_norm(512) | v width, 0 x 15]."""
    dh = gw.head_dim(L)
    rows = np.zeros((len(positions), ROPE_W_LEN), np.float32)
    half = DH // 2
    for i, p in enumerate(positions):
        cos, sin, _ = gw.rope_lut(p, L, rope_freqs=rope_freqs)
        c = np.ones(half, np.float32)
        s = np.zeros(half, np.float32)
        c[: dh // 2], s[: dh // 2] = cos, sin
        rows[i, :half], rows[i, half:DH] = c, s
    rows[:, DH : 2 * DH] = norms[0]
    rows[:, 2 * DH : 3 * DH] = norms[1]
    rows[:, 3 * DH] = dh
    return rows.astype(bfloat16)


class RopeRows:
    """rope_rows of every layer at once: the norms and widths are fixed per
    layer, and cos/sin depend only on whether the layer slides."""

    def __init__(self, qm, layers):
        self.freqs = qm.rope_freqs()
        self.static = np.stack(
            [rope_rows(L, [0], qk_norms(qm, L), self.freqs)[0] for L in range(layers)]
        ).astype(np.float32)
        self.sliding = np.array([gw.is_sliding(L) for L in range(layers)])

    def __call__(self, positions):
        """[layers, len(positions), ROPE_W_LEN] bf16."""
        positions = list(positions)
        rows = np.repeat(self.static[:, None], len(positions), axis=1)
        for slides in (True, False):
            sel = self.sliding == slides
            if not sel.any():
                continue
            L = int(np.argmax(sel))
            for i, p in enumerate(positions):
                cos, sin, dh = gw.rope_lut(p, L, rope_freqs=self.freqs)
                rows[sel, i, : dh // 2] = cos
                rows[sel, i, DH // 2 : DH // 2 + dh // 2] = sin
        return rows.astype(bfloat16)


class TokenRows:
    """Rows of an int8 group-32 table (Q4nxModel.embed_rows) gathered through
    a memory map of the model file instead of one read per token."""

    def __init__(self, qm, name):
        e, es = qm._hdr[name], qm._hdr[name + ".scale"]
        V, W = e["shape"]
        mm = np.memmap(qm.path, np.uint8, "r")
        q0, s0 = (qm._data_off + h["data_offsets"][0] for h in (e, es))
        self.q = mm[q0 : q0 + V * W].view(np.int8).reshape(V, W)
        self.s = mm[s0 : s0 + V * W // 8].view(np.float32).reshape(V, W // 32)

    def __call__(self, ids):
        ids = np.asarray(ids, np.int64).reshape(-1)
        q = self.q[ids].astype(np.float32).reshape(len(ids), -1, 32)
        return (q * self.s[ids][:, :, None]).reshape(len(ids), -1)


class PerLayerInputs:
    """The per-layer inputs of any tokens: gemma4_e2b_q4nx_weights's
    per_layer_inputs with every layer's model projection read once, as one
    [layers * 256, D] matrix, multiplied in bf16."""

    def __init__(self, qm, layers):
        import torch

        self.qm, self.layers = qm, layers
        mp = [
            np.asarray(qm.layer_ple(L)["model_proj"], np.float32) for L in range(layers)
        ]
        self.mp = torch.from_numpy(np.concatenate(mp)).to(torch.bfloat16)
        self.norm = np.asarray(qm.globals()["ple_proj_norm"], np.float32)
        self.embed = TokenRows(qm, "model.embed_tokens.weight")
        self.table = TokenRows(qm, "model.per_layer_token_embd.weight")

    def __call__(self, ids, emb=None):
        """[layers, tokens, PLE_PAD] bf16, the real 256 then zeros. emb: the
        tokens' scaled embedding rows, if the caller has them."""
        import torch

        n, T = self.layers, len(ids)
        if emb is None:
            emb = self.embed(ids) * gw.EMBED_SCALE
        x = torch.from_numpy(np.ascontiguousarray(emb, np.float32)).to(torch.bfloat16)
        proj = (x @ self.mp.T).float().numpy()
        proj = proj.reshape(T, n, gw.PLI_D) * gw.PLE_MODEL_PROJ_SCALE
        tbl = self.table(ids)
        tbl = tbl.reshape(T, gw.NUM_LAYERS, gw.PLI_D)[:, :n] * gw.PLE_EMBED_SCALE
        pli = (gw._rmsnorm(proj, self.norm) + tbl) * gw.PLE_INPUT_SCALE
        out = np.zeros((n, T, PLE_PAD), np.float32)
        out[:, :, : gw.PLI_D] = pli.transpose(1, 0, 2)
        return out.astype(bfloat16)


def per_layer_inputs(qm, ids, layers):
    """[layers, tokens, PLE_PAD] bf16 (uncached; see PerLayerInputs)."""
    return PerLayerInputs(qm, layers)(ids)
