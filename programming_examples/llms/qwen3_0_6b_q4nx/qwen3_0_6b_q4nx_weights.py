# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Q4NX (I8-packed) weight loader for the Qwen3-0.6B Q4NX example.
#
# FastFlowLM ships Qwen3 as a `model.q4nx` safetensors bundle in a GGUF-Q4_1-
# derived codec: per-projection weights are stored as dtype "I8" with a PACKED
# header shape (n_chunks, 5120), NOT the real (out, in). Each 32x256 chunk packs
# 512 B scale (bf16) + 512 B min (bf16) + 4096 B of 4-bit quants, group_size=32,
# parallel=16. Dequant is  w = scale * q + min  (additive min), with the row
# de-interleave  row_in_block = g*16 + r*2 + b  (g=2, r=8, b=2 nibbles/byte).
# Validated: dequant vs HF Qwen/Qwen3-0.6B q_proj cosine = 0.997.
#
# This differs from the Llama-3.2 `Q4NX` codec (real shapes, w = scale*(q-zero)),
# so it needs its own reader; the dequantized bf16 matrices then drop straight
# into the bf16 qwen3_0_6b LlamaWeights container and the sibling driver runs
# unchanged.
import json
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_HERE = Path(__file__).resolve().parent
_LLMS = _HERE.parent
_QWEN3 = _LLMS / "qwen3_0_6b"
for _p in (str(_LLMS), str(_QWEN3), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qwen3_0_6b_weights import (  # noqa: E402
    LlamaConfig,
    LayerWeights,
    LlamaWeights,
    generate_rope_lut,
)

# Q4NX/I8 block geometry (fixed by the FastFlowLM packer).
_ROW_BLOCK = 32
_COL_BLOCK = 256
_GROUP = 32
_PARALLEL = 16
_CHUNK_BYTES = 5120  # 512 (scale bf16) + 512 (min bf16) + 4096 (4-bit quants)


class _Q4nxI8Model:
    """mmap + parse a GGUF-Q4_1-derived `model.q4nx`; dequant I8-packed tensors.

    Weights carry a PACKED header shape; the caller supplies the real (out, in).
    """

    def __init__(self, model):
        path = _resolve(model)
        self._mm = np.memmap(path, dtype=np.uint8, mode="r")
        hlen = int(np.frombuffer(self._mm[:8].tobytes(), dtype="<u8")[0])
        self._hdr = json.loads(self._mm[8 : 8 + hlen].tobytes())
        self._base = 8 + hlen

    def _raw(self, name):
        o0, o1 = self._hdr[name]["data_offsets"]
        return np.frombuffer(
            self._mm[self._base + o0 : self._base + o1].tobytes(), dtype=np.uint8
        )

    def bf16(self, name):
        o0, o1 = self._hdr[name]["data_offsets"]
        v = np.frombuffer(
            self._mm[self._base + o0 : self._base + o1].tobytes(), dtype=bfloat16
        )
        return v.astype(np.float32).reshape(self._hdr[name]["shape"])

    def dequant(self, name, out, in_):
        """Dequantize an I8-packed tensor to float32 [out, in_] (w = scale*q + min)."""
        R, C, G, P = _ROW_BLOCK, _COL_BLOCK, _GROUP, _PARALLEL
        p, q = out // R, in_ // C
        merged = (
            self._raw(name).reshape(p * q, _CHUNK_BYTES).reshape(p, q, _CHUNK_BYTES)
        )
        # scale/min: bf16, stored (c, r) per chunk -> real [out, in_/G] colgroups.
        d = (
            merged[:, :, 0:512]
            .copy()
            .view(bfloat16)
            .astype(np.float32)
            .reshape(p, q, C // G, R)
        )
        m = (
            merged[:, :, 512:1024]
            .copy()
            .view(bfloat16)
            .astype(np.float32)
            .reshape(p, q, C // G, R)
        )
        d = d.transpose(0, 3, 1, 2).reshape(out, in_ // G)  # (p,r)->out, (q,c)->colg
        m = m.transpose(0, 3, 1, 2).reshape(out, in_ // G)
        # quants: 4096 bytes/chunk = (g=2, c=256, r=8), 2 nibbles/byte (b).
        qb = merged[:, :, 1024:_CHUNK_BYTES].reshape(p, q, 2, C, P // 2)
        lo = (qb & 0x0F).astype(np.float32)
        hi = ((qb >> 4) & 0x0F).astype(np.float32)
        qr = np.empty((p, q, R, C), np.float32)
        for g in range(2):
            for r in range(P // 2):
                qr[:, :, g * 16 + r * 2 + 0, :] = lo[:, :, g, :, r]
                qr[:, :, g * 16 + r * 2 + 1, :] = hi[:, :, g, :, r]
        qr = qr.transpose(0, 2, 1, 3).reshape(out, in_)
        return np.repeat(d, G, axis=1) * qr + np.repeat(m, G, axis=1)


def _resolve(model):
    import os

    if os.path.isfile(model):
        return model
    if os.path.isdir(model):
        p = os.path.join(model, "model.q4nx")
        if os.path.isfile(p):
            return p
    from huggingface_hub import hf_hub_download

    return hf_hub_download(model, "model.q4nx")


def load_q4nx_weights(model_source, config=None):
    """Load Qwen3-0.6B weights from a FastFlowLM `model.q4nx` bundle into a
    qwen3_0_6b LlamaWeights (I8 Q4NX projections dequant'd to bf16 on the host;
    norms/QK-norm/embed read as bf16; tied lm_head = embed_tokens)."""
    if config is None:
        config = LlamaConfig()
    qm = _Q4nxI8Model(model_source)

    E = config.emb_dim
    QD = config.n_heads * config.head_dim
    KVD = config.n_kv_heads * config.head_dim
    H = config.hidden_dim
    # real (out, in) per projection.
    dims = {
        "q_proj": (QD, E),
        "k_proj": (KVD, E),
        "v_proj": (KVD, E),
        "o_proj": (E, QD),
        "gate_proj": (H, E),
        "up_proj": (H, E),
        "down_proj": (E, H),
    }

    def W(li, proj):
        out, in_ = dims[proj]
        w = qm.dequant(
            f"model.layers.{li}.{_MLP.get(proj, 'self_attn.'+proj)}.weight", out, in_
        )
        return np.ascontiguousarray(w.T, dtype=bfloat16)  # [in, out] = (in, out)

    layers = []
    for li in range(config.n_layers):
        b = f"model.layers.{li}"
        layers.append(
            LayerWeights(
                attn_norm=np.asarray(qm.bf16(f"{b}.input_layernorm.weight"), bfloat16),
                wq=W(li, "q_proj"),
                wk=W(li, "k_proj"),
                wv=W(li, "v_proj"),
                wo=W(li, "o_proj"),
                ffn_norm=np.asarray(
                    qm.bf16(f"{b}.post_attention_layernorm.weight"), bfloat16
                ),
                w_gate=W(li, "gate_proj"),
                w_up=W(li, "up_proj"),
                w_down=W(li, "down_proj"),
                q_norm=np.asarray(qm.bf16(f"{b}.self_attn.q_norm.weight"), bfloat16),
                k_norm=np.asarray(qm.bf16(f"{b}.self_attn.k_norm.weight"), bfloat16),
            )
        )

    embed = np.asarray(qm.bf16("model.embed_tokens.weight"), bfloat16)
    norm = np.asarray(qm.bf16("model.norm.weight"), bfloat16)
    # Qwen3-0.6B ties embeddings (the bundle's separate I8 lm_head is ignored).
    return LlamaWeights(
        embed_table=embed, layers=layers, final_norm=norm, lm_head=embed
    )


# gate/up/down live under mlp.*; the rest under self_attn.* (handled in W()).
_MLP = {
    "gate_proj": "mlp.gate_proj",
    "up_proj": "mlp.up_proj",
    "down_proj": "mlp.down_proj",
}
