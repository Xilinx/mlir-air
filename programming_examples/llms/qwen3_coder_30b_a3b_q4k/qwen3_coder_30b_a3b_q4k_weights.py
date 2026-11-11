# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Qwen3-Coder-30B-A3B-Instruct (Qwen/Qwen3-Coder-30B-A3B-Instruct), quantized
from the HF bf16 checkpoint into the block formats of FastFlowLM's Qwen3.5 /
Qwen3.6 bundles (q35_weights), in the same container:

  Q4K   every projection, the experts included: 32 rows x 256 columns per
        4736-byte block, w = d * sc * q + dmin * m (per row and 256 columns:
        bf16 d >= 0, dmin <= 0; per row and 32 columns: uint8 sc, m)
  Q8    lm_head: 32 x 256 per 8704-byte block, w = scale * q (bf16 scale per
        row and 32 columns)
  bf16  embedding, norms, q_norm / k_norm, router

`prepare()` writes the bundle once (QWEN3C_BUNDLE, default under HF_HOME).
Per layer:

  attention  q_proj [4096, D], k / v [512, D], o [D, 4096], q_norm / k_norm [128]
  MoE        gate / up experts [128 x 768, D], down experts [128 x D, 768],
             router [128, D]
"""

import functools
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

# llms/ is imported as the air_examples package, not through sys.path.
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(Path(__file__).resolve().parents[2])
]

from air_examples.llms.qwen35_0_8b_q4nx import q35_weights  # noqa: E402

REPO = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
REVISION = "b2cff646eb4bb1d68355c01b18ae02e7cf42d120"

N_LAYERS, D, VOCAB, EPS = 48, 2048, 151936, 1e-6
NQ, NKV, DH, THETA = 32, 4, 128, 1e7
N_EXPERTS, TOP_K, I_EXP = 128, 8, 768

RB, CB, G = q35_weights.RB, q35_weights.CB, q35_weights.G
Q4K_BYTES, Q8_BYTES = q35_weights.Q4K_BYTES, q35_weights.Q8_BYTES


def name(layer, part):
    return f"model.layer.{layer}.{part}"


# ---- quantizers (torch, float32 in; numpy uint8 blocks out)


def _bf16(t):
    import torch

    return t.to(torch.bfloat16).to(torch.float32)


def _group_affine(x, steps):
    """Per row of x [n, 32]: (scale >= 0, lo <= 0) of w ~ scale * q + lo,
    q in [0, 15]. The candidate grid and the weighted least-squares fit are
    llama.cpp's make_qkx2_quants (weights av_x + |x|)."""
    import torch

    lo = x.amin(1).clamp(max=0)
    hi = x.amax(1)
    span = (hi - lo).clamp(min=1e-30)
    w = x.pow(2).mean(1, keepdim=True).sqrt() + x.abs()
    best_s, best_lo = span / 15, lo
    q = ((x - lo[:, None]) / best_s[:, None]).round().clamp(0, 15)
    best_err = (w * (best_s[:, None] * q + lo[:, None] - x).pow(2)).sum(1)
    sw, swx = w.sum(1), (w * x).sum(1)
    for i in range(steps + 1):
        iscale = (15 - 1 + 0.1 * i) / span
        q = ((x - lo[:, None]) * iscale[:, None]).round().clamp(0, 15)
        swl, swl2 = (w * q).sum(1), (w * q * q).sum(1)
        swlx = (w * q * x).sum(1)
        det = sw * swl2 - swl * swl
        ok = det > 0
        s = torch.where(ok, (sw * swlx - swx * swl) / det.clamp(min=1e-30), best_s)
        m = torch.where(ok, (swl2 * swx - swl * swlx) / det.clamp(min=1e-30), lo)
        pos = m > 0  # the min stays <= 0: refit the scale alone
        s = torch.where(pos, swlx / swl2.clamp(min=1e-30), s)
        m = torch.where(pos, torch.zeros_like(m), m)
        s = s.clamp(min=0)
        err = (w * (s[:, None] * q + m[:, None] - x).pow(2)).sum(1)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_s = torch.where(better, s, best_s)
        best_lo = torch.where(better, m, best_lo)
    return best_s, best_lo


def quantize_q4k(w, steps=20):
    """float32 [rows, K] (rows % 32 == 0, K % 256 == 0) -> uint8 blocks
    [rows / 32, K / 256, 4736]."""
    import torch

    w = torch.as_tensor(w, dtype=torch.float32)
    rows, K = w.shape
    nbi, nbj, ng = rows // RB, K // CB, CB // G
    s, lo = _group_affine(w.reshape(-1, G), steps)
    s, lo = s.reshape(rows, nbj, ng), lo.reshape(rows, nbj, ng)
    d = _bf16(s.amax(2) / 255)
    dm = _bf16(lo.amin(2) / 255)
    sc = torch.where(
        d[..., None] > 0, (s / d[..., None].clamp(min=1e-30)).round(), 0
    ).clamp(0, 255)
    m = torch.where(
        dm[..., None] < 0, (lo / dm[..., None].clamp(max=-1e-30)).round(), 0
    ).clamp(0, 255)
    s2 = (d[..., None] * sc).repeat_interleave(G, 2).reshape(rows, K)
    lo2 = (dm[..., None] * m).repeat_interleave(G, 2).reshape(rows, K)
    q = torch.where(s2 > 0, ((w - lo2) / s2.clamp(min=1e-30)).round(), 0).clamp(0, 15)

    q = q.to(torch.uint8).reshape(nbi, RB, nbj, CB).permute(0, 2, 3, 1)  # [i,j,col,row]
    nib = (q[..., 0::2] | (q[..., 1::2] << 4)).reshape(nbi, nbj, CB * RB // 2)

    def grp(a):  # [rows, nbj, 8] -> [i, j, 8 groups, 32 rows]
        return a.to(torch.uint8).reshape(nbi, RB, nbj, ng).permute(0, 2, 3, 1)

    def row(a):  # [rows, nbj] bf16 -> [i, j, 32 rows] as bytes
        a = a.to(torch.bfloat16).reshape(nbi, RB, nbj).permute(0, 2, 1).contiguous()
        return a.view(torch.uint8).reshape(nbi, nbj, 2 * RB)

    out = torch.cat(
        [
            grp(sc).reshape(nbi, nbj, -1),
            grp(m).reshape(nbi, nbj, -1),
            nib,
            row(d),
            row(dm),
        ],
        2,
    )
    assert out.shape[2] == Q4K_BYTES
    return out.numpy()


def quantize_q8(w):
    """float32 [rows, K] -> uint8 blocks [rows / 32, K / 256, 8704]."""
    import torch

    w = torch.as_tensor(w, dtype=torch.float32)
    rows, K = w.shape
    nbi, nbj, ng = rows // RB, K // CB, CB // G
    sc = _bf16(w.reshape(rows, K // G, G).abs().amax(2) / 127)
    q = torch.where(
        sc.repeat_interleave(G, 1) > 0,
        (w / sc.repeat_interleave(G, 1).clamp(min=1e-30)).round(),
        0,
    ).clamp(-128, 127)
    scb = (
        sc.to(torch.bfloat16)
        .reshape(nbi, RB, nbj, ng)
        .permute(0, 2, 3, 1)
        .contiguous()
        .view(torch.uint8)
        .reshape(nbi, nbj, -1)
    )
    qb = (
        q.to(torch.int8)
        .reshape(nbi, RB, nbj, CB)
        .permute(0, 2, 3, 1)
        .contiguous()
        .view(torch.uint8)
        .reshape(nbi, nbj, -1)
    )
    out = torch.cat([scb, qb], 2)
    assert out.shape[2] == Q8_BYTES
    return out.numpy()


# ---- the bundle


def bundle_path(source=None):
    """`source`, else $QWEN3C_BUNDLE, else qwen3_coder_30b_a3b.q4k under
    HF_HOME."""
    if source:
        return Path(source)
    if os.environ.get("QWEN3C_BUNDLE"):
        return Path(os.environ["QWEN3C_BUNDLE"])
    hf = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    return Path(hf) / "qwen3_coder_30b_a3b.q4k"


def _plan():
    """[(bundle name, kind, shape, HF source)] in file order."""
    blk = lambda r, k, nb: [r // RB, k // CB, nb]  # noqa: E731
    out = [
        ("model.embed_tokens.weight", "BF16", [VOCAB, D], "model.embed_tokens.weight"),
        ("model.norm.weight", "BF16", [D], "model.norm.weight"),
        ("lm_head.weight", "Q8", blk(VOCAB, D, Q8_BYTES), "lm_head.weight"),
    ]
    for l in range(N_LAYERS):
        h = f"model.layers.{l}."
        for p, r, k in (
            ("q_proj", NQ * DH, D),
            ("k_proj", NKV * DH, D),
            ("v_proj", NKV * DH, D),
            ("o_proj", D, NQ * DH),
        ):
            out.append(
                (
                    name(l, f"self_attn.{p}.weight"),
                    "Q4K",
                    blk(r, k, Q4K_BYTES),
                    h + f"self_attn.{p}.weight",
                )
            )
        for p in ("q_norm", "k_norm"):
            out.append(
                (
                    name(l, f"self_attn.{p}.weight"),
                    "BF16",
                    [DH],
                    h + f"self_attn.{p}.weight",
                )
            )
        for p in ("input_layernorm", "post_attention_layernorm"):
            out.append((name(l, f"{p}.weight"), "BF16", [D], h + f"{p}.weight"))
        out.append(
            (
                name(l, "mlp.moe_router.weight"),
                "BF16",
                [N_EXPERTS, D],
                h + "mlp.gate.weight",
            )
        )
        for p, r, k in (
            ("gate", N_EXPERTS * I_EXP, D),
            ("up", N_EXPERTS * I_EXP, D),
            ("down", N_EXPERTS * D, I_EXP),
        ):
            out.append(
                (
                    name(l, f"mlp.{p}_exps_proj.weight"),
                    "Q4K",
                    blk(r, k, Q4K_BYTES),
                    (h + "mlp.experts.{e}." + f"{p}_proj.weight"),
                )
            )
    return out


def prepare(hf_dir=None, out=None, steps=20, log=print):
    """Quantize the HF checkpoint into the bundle (once)."""
    import torch
    from safetensors import safe_open

    if hf_dir is None:
        from huggingface_hub import snapshot_download

        hf_dir = snapshot_download(REPO, revision=REVISION)
    hf_dir = Path(hf_dir)
    index = json.loads((hf_dir / "model.safetensors.index.json").read_text())
    where = index["weight_map"]
    files = {}

    def hf(t):
        f = where[t]
        if f not in files:
            files[f] = safe_open(str(hf_dir / f), "pt")
        return files[f].get_tensor(t).to(torch.float32)

    plan = _plan()
    nbytes = lambda shape: int(np.prod(shape)) * (
        1 if len(shape) == 3 else 2
    )  # noqa: E731
    hdr, o = {}, 0
    for n, kind, shape, _ in plan:
        dt = "I8" if kind in ("Q4K", "Q8") else "BF16"
        hdr[n] = {"dtype": dt, "shape": shape, "data_offsets": [o, o + nbytes(shape)]}
        o += nbytes(shape)
    hb = json.dumps(hdr).encode()
    hb += b" " * (-len(hb) % 8)
    out = bundle_path(out)
    tmp = out.with_suffix(".partial")
    with open(tmp, "wb") as f:
        f.write(np.uint64(len(hb)).tobytes())
        f.write(hb)
        for i, (n, kind, shape, src) in enumerate(plan):
            if kind == "BF16":
                b = hf(src).to(torch.bfloat16).contiguous().view(torch.uint8).numpy()
            elif kind == "Q8":
                b = quantize_q8(hf(src))
            elif "{e}" in src:  # experts stacked: one expert at a time (memory)
                b = np.concatenate(
                    [quantize_q4k(hf(src.format(e=e)), steps) for e in range(N_EXPERTS)]
                )
            else:
                b = quantize_q4k(hf(src), steps)
            assert b.nbytes == nbytes(shape), (n, b.nbytes, nbytes(shape))
            f.write(np.ascontiguousarray(b).tobytes())
            if log and (i % 12 == 0 or i == len(plan) - 1):
                log(f"[prepare] {i + 1}/{len(plan)} {n}")
    os.replace(tmp, out)
    return out


class Qwen3CoderQ4k(q35_weights.Q4nx):
    """The bundle by layer, as float32 numpy (dequantized) or raw blocks."""

    def __init__(self, path=None):
        self.path = bundle_path(path)
        if not self.path.is_file():
            raise FileNotFoundError(
                f"{self.path}: no bundle; run `python {Path(__file__).name} --prepare`"
            )
        with open(self.path, "rb") as f:
            n = int(np.frombuffer(f.read(8), "<u8")[0])
            self.hdr = json.loads(f.read(n))
        self.off = 8 + n
        self.mm = np.memmap(self.path, np.uint8, "r")

    def tokenizer(self):
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        return Tokenizer.from_file(
            hf_hub_download(REPO, "tokenizer.json", revision=REVISION)
        )

    def mat(self, layer, part, rows, K):
        return self.dequant(name(layer, part), rows, K)

    def vec(self, layer, part):
        return self.bf16(name(layer, part))

    def attn(self, layer):
        """(q [4096, D], k, v [512, D], o [D, 4096], q_norm, k_norm)."""
        sa = "self_attn."
        return (
            self.mat(layer, sa + "q_proj.weight", NQ * DH, D),
            self.mat(layer, sa + "k_proj.weight", NKV * DH, D),
            self.mat(layer, sa + "v_proj.weight", NKV * DH, D),
            self.mat(layer, sa + "o_proj.weight", D, NQ * DH),
            self.vec(layer, sa + "q_norm.weight"),
            self.vec(layer, sa + "k_norm.weight"),
        )

    def norms(self, layer):
        return (
            self.vec(layer, "input_layernorm.weight"),
            self.vec(layer, "post_attention_layernorm.weight"),
        )

    def router(self, layer):
        """[N_EXPERTS, D]."""
        return self.vec(layer, "mlp.moe_router.weight")

    # dequantized experts are large: room for a token's and some reuse
    @functools.lru_cache(maxsize=512)
    def expert(self, layer, e):
        """(gate, up [I_EXP, D], down [D, I_EXP]) of expert e."""
        rows = lambda part, n, K: self.dequant_rows(  # noqa: E731
            name(layer, f"mlp.{part}_exps_proj.weight"), e * n, (e + 1) * n, K
        )
        return rows("gate", I_EXP, D), rows("up", I_EXP, D), rows("down", D, I_EXP)

    def embed(self, ids):
        t = np.asarray(self.raw("model.embed_tokens.weight")).view(bfloat16)
        return t.reshape(VOCAB, D)[np.asarray(ids)].astype(np.float32)

    def final_norm(self):
        return self.bf16("model.norm.weight")

    @functools.cached_property
    def lm_head(self):
        """float32 [VOCAB, D] (the Q8 values, exact)."""
        return self.dequant("lm_head.weight", VOCAB, D)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true", help="write the bundle")
    ap.add_argument("--hf-dir", help="HF checkpoint dir (default: Hub at REVISION)")
    ap.add_argument("--out", help="bundle path (default: $QWEN3C_BUNDLE)")
    ap.add_argument("--steps", type=int, default=20)
    a = ap.parse_args()
    if a.prepare:
        print(prepare(a.hf_dir, a.out, a.steps))
