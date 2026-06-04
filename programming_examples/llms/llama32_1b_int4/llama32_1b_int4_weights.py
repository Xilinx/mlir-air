# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""End-to-end AWQ weight loader for the int4 Llama-3.2-1B example.

Loads an AutoAWQ checkpoint once and populates BOTH:

* bf16 dequant fields on each LayerWeights (wq / wk / wv / wo / w_gate /
  w_up / w_down) — consumed by the NPU **bf16 prefill** path (the int4
  prefill driver with `--prefill-dtype=bf16`).
* per-layer decode-side packed BO attributes (`_wq_packed`, `_wk_packed`,
  `_wv_packed`, `_wo_packed`, `_wgateup_packed`, `_wdown_packed`) —
  consumed by the NPU **int4 decode** ELFs (`rms_qkv_int4_rope` and
  `o_gemv_ffn_int4`).

This is the loader the e2e inference driver uses. Sibling `awq_pack.py`
loads the prefill **GEMM** packed BO layout (used by the int4 prefill
stitchers) — it's a different mlir-air kernel layout and lives in its
own loader so each compute path's loader stays focused.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from ml_dtypes import bfloat16

_THIS_DIR = Path(__file__).resolve().parent
_LLMS_DIR = _THIS_DIR.parent
_LLAMA_BF16_DIR = _LLMS_DIR / "llama32_1b"
_PROG_EXAMPLES = _LLMS_DIR.parent  # programming_examples/ (for cross-area imports)
for p in (str(_LLMS_DIR), str(_LLAMA_BF16_DIR), str(_THIS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from llama32_1b_weights import (  # noqa: E402
    LlamaConfig,
    LayerWeights,
    LlamaWeights,
    _resolve_safetensor_files,
    _load_tensor,
)
from awq_repacker import (  # noqa: E402
    dequant_to_bf16,
    repack_hf_awq_linear,
    repack_for_gemv,
)

sys.path.insert(
    0,
    str(_PROG_EXAMPLES / "matrix_vector_multiplication" / "int4_awq"),
)
from matvec_int4_packed import pack_inputs as _pack_inputs  # noqa: E402

# HF AutoAWQ Linear suffix -> dataclass field name.
_HF_AWQ_LINEARS = {
    "self_attn.q_proj": "wq",
    "self_attn.k_proj": "wk",
    "self_attn.v_proj": "wv",
    "self_attn.o_proj": "wo",
    "mlp.gate_proj": "w_gate",
    "mlp.up_proj": "w_up",
    "mlp.down_proj": "w_down",
}

# Field name -> per-layer packed-BO attribute used by the decode ELFs.
# w_gate / w_up are not here — they get interleaved at the nibble level
# into a single `_wgateup_packed` BO that the int4 FFN ELF consumes.
_AWQ_PACKED_ATTR = {
    "wq": "_wq_packed",
    "wk": "_wk_packed",
    "wv": "_wv_packed",
    "wo": "_wo_packed",
    "w_down": "_wdown_packed",
}


def load_weights_awq(
    model_name_or_path: str,
    config: Optional[LlamaConfig] = None,
    group_size: int = 128,
    m_tile: int = 8,
    k_chunk: int = 2048,
    n_cores: int = 8,
) -> LlamaWeights:
    """Load a HuggingFace AutoAWQ Llama checkpoint into a dual-layout
    LlamaWeights (bf16 dequant + decode-side packed BOs).

    Args:
        model_name_or_path: local dir or HF model id of an AutoAWQ checkpoint.
        config: model hyperparameters (defaults to Llama-3.2-1B).
        group_size: AWQ group size (typical 128). Must match the checkpoint.
        m_tile, k_chunk, n_cores: GEMV packed-BO tiling parameters; defaults
            match the int4 decode ELF builders.

    Returns:
        LlamaWeights with bf16 dequant fields AND per-layer decode-side
        packed-BO attributes attached.
    """
    from safetensors import safe_open

    if config is None:
        config = LlamaConfig()

    safetensor_files = _resolve_safetensor_files(model_name_or_path)

    key_to_file = {}
    for filepath in safetensor_files:
        with safe_open(filepath, framework="numpy") as f:
            for key in f.keys():
                key_to_file[key] = filepath

    embed_key = "model.embed_tokens.weight"
    if embed_key not in key_to_file:
        raise KeyError(f"Missing weight: {embed_key}")
    with safe_open(key_to_file[embed_key], framework="numpy") as f:
        embed_table = _load_tensor(f, embed_key, bfloat16)

    layers: List[LayerWeights] = []
    for layer_idx in range(config.n_layers):
        attn_norm_key = f"model.layers.{layer_idx}.input_layernorm.weight"
        ffn_norm_key = f"model.layers.{layer_idx}.post_attention_layernorm.weight"
        with safe_open(key_to_file[attn_norm_key], framework="numpy") as f:
            attn_norm = _load_tensor(f, attn_norm_key, bfloat16)
        with safe_open(key_to_file[ffn_norm_key], framework="numpy") as f:
            ffn_norm = _load_tensor(f, ffn_norm_key, bfloat16)

        linear_bf16 = {}
        linear_packed = {}
        gate_quants = up_quants = None
        for hf_prefix, field_name in _HF_AWQ_LINEARS.items():
            base = f"model.layers.{layer_idx}.{hf_prefix}"
            qw_key, qz_key, s_key = (
                f"{base}.qweight",
                f"{base}.qzeros",
                f"{base}.scales",
            )
            for k in (qw_key, qz_key, s_key):
                if k not in key_to_file:
                    raise KeyError(
                        f"Missing AWQ tensor: {k} (is this an AutoAWQ checkpoint?)"
                    )
            with safe_open(key_to_file[qw_key], framework="numpy") as f:
                qw = f.get_tensor(qw_key)
            with safe_open(key_to_file[qz_key], framework="numpy") as f:
                qz = f.get_tensor(qz_key)
            with safe_open(key_to_file[s_key], framework="numpy") as f:
                sc = f.get_tensor(s_key)
            if qw.dtype != np.int32:
                qw = qw.astype(np.int32)
            if qz.dtype != np.int32:
                qz = qz.astype(np.int32)
            # (a) bf16 dequant for the bf16 prefill stitchers; shape [in, out].
            linear_bf16[field_name] = dequant_to_bf16(qw, qz, sc, group_size)
            # (b) decode-side packed BO; gate/up deferred for nibble-level interleave.
            if field_name == "w_gate":
                gate_quants = repack_hf_awq_linear(qw, qz, sc, group_size)
            elif field_name == "w_up":
                up_quants = repack_hf_awq_linear(qw, qz, sc, group_size)
            else:
                linear_packed[field_name] = repack_for_gemv(
                    qw,
                    qz,
                    sc,
                    group_size,
                    M_TILE=m_tile,
                    K_CHUNK=k_chunk,
                    N_CORES=n_cores,
                )

        # Interleave gate/up at the (A_q, A_s, A_z) level: row 2i = gate[i],
        # row 2i+1 = up[i]. Pack into one BO for arg7 of o_gemv_ffn_int4.
        if gate_quants is None or up_quants is None:
            raise RuntimeError(
                "Could not find both mlp.gate_proj and mlp.up_proj AWQ tensors"
            )
        g_q, g_s, g_z = gate_quants
        u_q, u_s, u_z = up_quants
        h_out, k_half = g_q.shape
        if u_q.shape != (h_out, k_half):
            raise RuntimeError("gate_proj and up_proj have different shapes")
        gu_q = np.empty((2 * h_out, k_half), dtype=np.uint8)
        gu_q[0::2] = g_q
        gu_q[1::2] = u_q
        n_groups = g_s.shape[0]
        gu_s = np.empty((n_groups, 2 * h_out), dtype=g_s.dtype)
        gu_s[:, 0::2] = g_s
        gu_s[:, 1::2] = u_s
        gu_z = np.empty((n_groups, 2 * h_out), dtype=np.uint8)
        gu_z[:, 0::2] = g_z
        gu_z[:, 1::2] = u_z
        M_gateup = 2 * h_out
        K_full = k_half * 2
        gateup_packed = _pack_inputs(
            gu_q,
            gu_s,
            gu_z,
            M_gateup,
            K_full,
            group_size,
            m_tile,
            k_chunk,
            n_cores,
            M_gateup,
        )

        layer = LayerWeights(
            attn_norm=attn_norm,
            ffn_norm=ffn_norm,
            **linear_bf16,
        )
        for field_name, packed in linear_packed.items():
            setattr(layer, _AWQ_PACKED_ATTR[field_name], packed)
        layer._wgateup_packed = gateup_packed
        layers.append(layer)

        if (layer_idx + 1) % 4 == 0 or layer_idx == 0:
            print(f"  AWQ layer {layer_idx + 1}/{config.n_layers} loaded")

    norm_key = "model.norm.weight"
    with safe_open(key_to_file[norm_key], framework="numpy") as f:
        final_norm = _load_tensor(f, norm_key, bfloat16)
    lm_head_key = "lm_head.weight"
    if lm_head_key in key_to_file:
        with safe_open(key_to_file[lm_head_key], framework="numpy") as f:
            lm_head = _load_tensor(f, lm_head_key, bfloat16)
    else:
        print("  Tied embeddings: reusing embed_table as lm_head.")
        lm_head = embed_table

    return LlamaWeights(
        embed_table=embed_table,
        layers=layers,
        final_norm=final_norm,
        lm_head=lm_head,
    )
