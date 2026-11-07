# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Q4NX weight loader + CPU reference for GPT-OSS-20B, reading FastFlowLM's own
# bundle (FastFlowLM/GPT-OSS-20B-NPU2 model.q4nx) in the layout its kernels use:
#   FLM_Q4NX_Converter/q4nx/models/gpt_oss.py      (bundle layout)
#   FLM_Xclbin/GPTOSS/gpt_npu_bin/kernel/mvm_MXFP4.hpp, MXFP4.h  (MXFP4 decode)
#
# Two block codecs share one 32x128 / 2560-byte block:
#   Q4_1 (attention projections, LM head): bf16 scale[4][32], bf16 min[4][32],
#     then nibbles [2 row stripes][128 cols][8 bytes], byte k of a stripe holding
#     rows 2k (low) and 2k+1 (high); w = scale*q + min.
#   MXFP4 (experts): e8m0 scale[4][32], 384 bytes pad (column block 0 carries
#     the 32 row biases in bytes 128..191), then nibbles [2 stripes][64 byte
#     cols][16 bytes]; bytes 0..7 hold element column 2c, bytes 8..15 column
#     2c+1, each byte packing rows 2k (low) and 2k+1 (high); w = e2m1 * 2^(e-127).
import math
import os
from pathlib import Path

import torch
from safetensors import safe_open

D = 2880
D_PAD = 2944  # D rounded up to the 128-column block
N_LAYERS = 24
N_Q_HEADS = 64
N_KV_HEADS = 8
DH = 64
INTER = 2880
N_EXPERTS = 32
TOP_K = 4
VOCAB = 201088
RMS_EPS = 1e-5
SLIDING_WINDOW = 128
SWIGLU_LIMIT = 7.0
SWIGLU_ALPHA = 1.702
ROPE_THETA = 150000.0
YARN_FACTOR = 32.0
YARN_ORIG_MAX_POS = 4096
YARN_BETA_FAST = 32.0
YARN_BETA_SLOW = 1.0
LAYER_TYPES = ["sliding" if i % 2 == 0 else "full" for i in range(N_LAYERS)]

ROW_BLOCK = 32
COL_BLOCK = 128
BLOCK_BYTES = 2560
N_GROUPS = COL_BLOCK // 32
ROWS_PER_GROUP_OF_BLOCKS = 4  # FLM packs 4 row blocks per column (one per core)
EXPERT_ROW_GROUPS = 69  # 46 gate/up (interleaved) + 23 down
EXPERT_BYTES = EXPERT_ROW_GROUPS * (D_PAD // COL_BLOCK) * 4 * BLOCK_BYTES

# FastFlowLM's GPT-OSS bundle on the Hub; GPT_OSS_FLM_DIR names a local copy.
FLM_REPO = "FastFlowLM/GPT-OSS-20B-NPU2"
FLM_REVISION = "v0.9.20-merge-lm-head"


def flm_model_dir():
    """The directory holding model.q4nx and tokenizer.json."""
    local = os.environ.get("GPT_OSS_FLM_DIR")
    if local:
        return Path(local)
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            FLM_REPO,
            revision=FLM_REVISION,
            allow_patterns=["model.q4nx", "tokenizer.json", "config.json"],
        )
    )


_E2M1 = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=torch.float32,
)


def _e8m0(e):
    """e8m0 exponent bytes -> float32 2^(e-127); e=0 is 2^-127."""
    e = e.to(torch.int32)
    return torch.ldexp(torch.ones_like(e, dtype=torch.float32), e - 127)


def _bf16(t):
    return t.contiguous().view(torch.bfloat16).float()


def _ungroup_rows(blocks, groups):
    """[nrg, ncb, groups, ...] grouped row blocks -> [nrg*groups, ncb, ...]."""
    return blocks.transpose(1, 2).reshape(-1, *blocks.shape[1:2], *blocks.shape[3:])


def dequant_q4_1(blocks):
    """Q4_1 blocks [nrb, ncb, 2560] uint8 -> float32 [nrb*32, ncb*128]."""
    nrb, ncb, _ = blocks.shape
    sc = _bf16(blocks[..., 0:256]).reshape(nrb, ncb, N_GROUPS, ROW_BLOCK)
    mn = _bf16(blocks[..., 256:512]).reshape(nrb, ncb, N_GROUPS, ROW_BLOCK)
    qb = blocks[..., 512:].reshape(nrb, ncb, 2, COL_BLOCK, 8)
    q = torch.stack((qb & 0xF, qb >> 4), dim=-1)  # [.., stripe, col, k, lo/hi]
    q = q.reshape(nrb, ncb, 2, COL_BLOCK, 16).permute(0, 1, 2, 4, 3)
    q = q.reshape(nrb, ncb, ROW_BLOCK, N_GROUPS, 32).float()
    w = q * sc.permute(0, 1, 3, 2)[..., None] + mn.permute(0, 1, 3, 2)[..., None]
    return (
        w.reshape(nrb, ncb, ROW_BLOCK, COL_BLOCK)
        .permute(0, 2, 1, 3)
        .reshape(nrb * ROW_BLOCK, ncb * COL_BLOCK)
    )


def dequant_mxfp4(blocks):
    """MXFP4 blocks [nrb, ncb, 2560] uint8 -> (float32 W [nrb*32, ncb*128],
    float32 bias [nrb*32] from column block 0)."""
    nrb, ncb, _ = blocks.shape
    sc = _e8m0(blocks[..., 0:128]).reshape(nrb, ncb, N_GROUPS, ROW_BLOCK)
    bias = _bf16(blocks[:, 0, 128:192]).reshape(nrb * ROW_BLOCK)
    qb = blocks[..., 512:].reshape(
        nrb, ncb, 2, 64, 2, 8
    )  # stripe, bytecol, even/odd col, k
    q = torch.stack((qb & 0xF, qb >> 4), dim=-1)  # .., k, row lo/hi
    q = q.reshape(nrb, ncb, 2, 64, 2, 16)  # stripe, bytecol, col parity, row-in-stripe
    q = q.permute(0, 1, 2, 5, 3, 4).reshape(nrb, ncb, ROW_BLOCK, COL_BLOCK)
    v = _E2M1[q.long()].reshape(nrb, ncb, ROW_BLOCK, N_GROUPS, 32)
    w = v * sc.permute(0, 1, 3, 2)[..., None]
    w = w.reshape(nrb, ncb, ROW_BLOCK, COL_BLOCK).permute(0, 2, 1, 3)
    return w.reshape(nrb * ROW_BLOCK, ncb * COL_BLOCK), bias


class GptOssQ4nx:
    """Lazy reader of FLM's model.q4nx; every weight comes back as float32 in
    logical [out, in] order with FLM's padding stripped."""

    def __init__(self, model_dir=None):
        self.path = Path(model_dir or flm_model_dir()) / "model.q4nx"
        self._f = safe_open(str(self.path), framework="pt")
        self._expert_cache = {}

    def raw(self, name):
        return self._f.get_tensor(name)

    def vec(self, name):
        return self.raw(name).float()

    def _q4_1(self, name, rows, cols, groups=ROWS_PER_GROUP_OF_BLOCKS):
        t = self.raw(name).view(torch.uint8)  # [nrg, ncb, groups, 2560]
        w = dequant_q4_1(_ungroup_rows(t, groups))
        return w[:rows, :cols]

    def embed(self, ids):
        if not hasattr(self, "_embed"):
            self._embed = self.raw("model.embed_tokens.weight")
        return self._embed[ids].float()

    def lm_head(self):
        return self._q4_1("lm_head.weight", VOCAB, D, groups=2)

    def attn(self, layer):
        p = f"model.layers.{layer}.self_attn."
        return dict(
            wq=self._q4_1(p + "q_proj.weight", N_Q_HEADS * DH, D),
            wk=self._q4_1(p + "k_proj.weight", N_KV_HEADS * DH, D),
            wv=self._q4_1(p + "v_proj.weight", N_KV_HEADS * DH, D),
            wo=self._q4_1(p + "o_proj.weight", D, N_Q_HEADS * DH),
            bq=self.vec(p + "q_proj.bias"),
            bk=self.vec(p + "k_proj.bias"),
            bv=self.vec(p + "v_proj.bias"),
            bo=self.vec(p + "o_proj.bias"),
            sinks=self.vec(p + "sinks.weight"),
        )

    def norms(self, layer):
        p = f"model.layers.{layer}."
        return (
            self.vec(p + "input_layernorm.weight"),
            self.vec(p + "post_attention_layernorm.weight"),
        )

    def router(self, layer):
        p = f"model.layers.{layer}.mlp.router."
        return self.vec(p + "weight_prefill")[:, :D], self.vec(p + "bias")

    def expert(self, layer, e):
        """(gate [INTER, D], up [INTER, D], down [D, INTER], gate_b, up_b,
        down_b), biases read from the blocks the kernel reads them from."""
        key = (layer, e)
        if key not in self._expert_cache:
            # one layer's experts: a prompt-at-once forward reads no more
            if len(self._expert_cache) >= N_EXPERTS:
                self._expert_cache.clear()
            name = f"model.layers.{layer}.ffn_gate_up_down_exps.weight"
            t = self._f.get_slice(name)[e].view(torch.uint8)  # [69, 23, 4, 2560]
            gu = t[:46].reshape(23, 2, *t.shape[1:])  # [rg, gate/up, cb, 4, B]
            gate, gate_b = dequant_mxfp4(_ungroup_rows(gu[:, 0], 4))
            up, up_b = dequant_mxfp4(_ungroup_rows(gu[:, 1], 4))
            down, down_b = dequant_mxfp4(_ungroup_rows(t[46:], 4))
            self._expert_cache[key] = (
                gate[:INTER, :D],
                up[:INTER, :D],
                down[:D, :INTER],
                gate_b[:INTER],
                up_b[:INTER],
                down_b[:D],
            )
        return self._expert_cache[key]

    def final_norm(self):
        return self.vec("model.norm.weight")


# ---------------------------------------------------------------- reference


def yarn_inv_freq():
    """YaRN inverse frequencies and cos/sin scale (HF rope_type "yarn",
    truncate=False)."""
    pos_freqs = ROPE_THETA ** (torch.arange(0, DH, 2, dtype=torch.float64) / DH)
    extrap = 1.0 / pos_freqs
    interp = 1.0 / (YARN_FACTOR * pos_freqs)

    def corr(rot):
        return (DH * math.log(YARN_ORIG_MAX_POS / (rot * 2 * math.pi))) / (
            2 * math.log(ROPE_THETA)
        )

    low = max(corr(YARN_BETA_FAST), 0)
    high = min(corr(YARN_BETA_SLOW), DH - 1)
    if low == high:
        high += 0.001
    ramp = ((torch.arange(DH // 2, dtype=torch.float64) - low) / (high - low)).clamp(
        0, 1
    )
    extrap_factor = 1 - ramp
    inv_freq = interp * (1 - extrap_factor) + extrap * extrap_factor
    scale = 0.1 * math.log(YARN_FACTOR) + 1.0
    return inv_freq.float(), scale


def rope_cos_sin(positions):
    inv_freq, scale = yarn_inv_freq()
    ang = torch.as_tensor(positions, dtype=torch.float32)[:, None] * inv_freq[None]
    return torch.cos(ang) * scale, torch.sin(ang) * scale


def apply_rope(x, cos, sin):
    """Half-split rotation; x [..., T, DH], cos/sin [T, DH/2]."""
    a, b = x[..., : DH // 2], x[..., DH // 2 :]
    return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1)


def rmsnorm(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + RMS_EPS) * w


def swiglu(gate, up):
    gate = gate.clamp(max=SWIGLU_LIMIT)
    up = up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
    return (up + 1) * gate * torch.sigmoid(gate * SWIGLU_ALPHA)


def route(x, wr, br):
    """Router: logits, top-4 indices, softmax over the top-4 logits."""
    logits = x @ wr.T + br
    top_v, top_i = torch.topk(logits, TOP_K, dim=-1)
    return logits, top_i, torch.softmax(top_v, dim=-1)


def moe(w, layer, x):
    wr, br = w.router(layer)
    _, top_i, top_w = route(x, wr, br)
    out = torch.zeros_like(x)
    for t in range(x.shape[0]):
        for k in range(TOP_K):
            gate, up, down, gb, ub, db = w.expert(layer, int(top_i[t, k]))
            h = swiglu(x[t] @ gate.T + gb, x[t] @ up.T + ub)
            out[t] += top_w[t, k] * (h @ down.T + db)
    return out, top_i


def attention(a, layer, x, kv, positions):
    """x [T, D] at `positions`; kv = (k [KV, L, DH], v) cache, appended in place."""
    T = x.shape[0]
    q = (x @ a["wq"].T + a["bq"]).reshape(T, N_Q_HEADS, DH).transpose(0, 1)
    k = (x @ a["wk"].T + a["bk"]).reshape(T, N_KV_HEADS, DH).transpose(0, 1)
    v = (x @ a["wv"].T + a["bv"]).reshape(T, N_KV_HEADS, DH).transpose(0, 1)
    cos, sin = rope_cos_sin(positions)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    kc = torch.cat((kv[0], k), dim=1)
    vc = torch.cat((kv[1], v), dim=1)
    kv[0], kv[1] = kc, vc
    rep = N_Q_HEADS // N_KV_HEADS
    kr, vr = kc.repeat_interleave(rep, 0), vc.repeat_interleave(rep, 0)
    s = (q @ kr.transpose(1, 2)) / math.sqrt(DH)  # [H, T, L]
    qpos = torch.as_tensor(positions)[:, None]
    kpos = torch.arange(kc.shape[1])[None]
    keep = kpos <= qpos
    if LAYER_TYPES[layer] == "sliding":
        keep &= kpos > qpos - SLIDING_WINDOW
    s = s.masked_fill(~keep, float("-inf"))
    sink = a["sinks"].reshape(N_Q_HEADS, 1, 1).expand(-1, T, 1)
    p = torch.softmax(torch.cat((s, sink), dim=-1), dim=-1)[..., :-1]
    o = (p @ vr).transpose(0, 1).reshape(T, N_Q_HEADS * DH)
    return o @ a["wo"].T + a["bo"]


class Reference:
    """Token-by-token or prompt-at-once CPU forward with a KV cache."""

    def __init__(self, w):
        self.w = w
        self.kv = [
            [torch.zeros(N_KV_HEADS, 0, DH), torch.zeros(N_KV_HEADS, 0, DH)]
            for _ in range(N_LAYERS)
        ]
        self.pos = 0
        self._head = None
        self._final = w.final_norm()

    @torch.no_grad()
    def forward(self, ids, collect=None):
        """Returns logits of the last position; `collect` gets per-layer hidden
        states and expert choices when given a list."""
        positions = list(range(self.pos, self.pos + len(ids)))
        x = self.w.embed(torch.as_tensor(ids))
        for layer in range(N_LAYERS):
            n1, n2 = self.w.norms(layer)
            x = x + attention(
                self.w.attn(layer), layer, rmsnorm(x, n1), self.kv[layer], positions
            )
            m, top_i = moe(self.w, layer, rmsnorm(x, n2))
            x = x + m
            if collect is not None:
                collect.append((x.clone(), top_i.clone()))
        self.pos += len(ids)
        if self._head is None:
            self._head = self.w.lm_head()
        return rmsnorm(x[-1], self._final) @ self._head.T
