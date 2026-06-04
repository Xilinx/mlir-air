# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AWQ uint4 -> bf16 dequant -> bfp16ebs8 packed BO loader."""

import os
import sys
from typing import Optional

import numpy as np
from ml_dtypes import bfloat16

_HERE = os.path.dirname(os.path.abspath(__file__))
_BFP_GEMM_DIR = os.path.normpath(
    os.path.join(_HERE, "..", "..", "matrix_multiplication", "bf16_x_bfp16")
)
if _BFP_GEMM_DIR not in sys.path:
    sys.path.insert(0, _BFP_GEMM_DIR)

from awq_pack import (  # noqa: E402
    _HF_AWQ_LAYER_MAP,
    _resolve_safetensor_files,
    awq_dequant_layer,
)
from llama32_1b_weights import LayerWeights, LlamaConfig, LlamaWeights  # noqa: E402
from matmul_bf16_x_bfp16 import (  # noqa: E402
    bfp_tile_bytes,
    pack_b_bfp16ebs8,
)


def awq_pack_for_npu_bfp16(
    qweight_i32, qzeros_i32, scales_bf16, gs=128, n_tile=64, k_chunk=128, M_seq=2048
):
    """AWQ qweight/qzeros/scales -> bfp16ebs8 packed BO uint8."""
    del M_seq  # unused; kept for signature parity with the int4 packer
    W_dense_bf16 = awq_dequant_layer(qweight_i32, qzeros_i32, scales_bf16, gs=gs)
    return pack_b_bfp16ebs8(W_dense_bf16, n_tile, k_chunk)


def load_awq_weights_bfp(
    model_path: str,
    config: Optional[LlamaConfig] = None,
    n_tile: int = 64,
    k_chunk: int = 128,
    seq_len: int = 2048,
):
    """AWQ HF checkpoint -> (LlamaWeights bf16, list[dict] of bfp16 BOs).

    Drop-in replacement for awq_pack.load_awq_weights when the prefill
    driver wants bfp16 weight BOs. Same LlamaWeights output (bf16
    dequantized projections) for the CPU/HF reference path; the per-layer
    packed dict carries bfp16ebs8 BOs instead of int4 Q+S+Z BOs.
    """
    from safetensors import safe_open
    import torch

    if config is None:
        config = LlamaConfig()

    files = _resolve_safetensor_files(model_path)
    key_to_file = {}
    for fp in files:
        with safe_open(fp, framework="pt") as f:
            for k in f.keys():
                key_to_file[k] = fp

    def _get(k, as_int32=False):
        with safe_open(key_to_file[k], framework="pt") as f:
            t = f.get_tensor(k)
        if as_int32:
            return t.numpy().astype(np.int32)
        if t.dtype == torch.bfloat16:
            return t.view(torch.int16).numpy().view(bfloat16)
        return t.numpy()

    embed = _get("model.embed_tokens.weight")
    assert embed.shape == (config.vocab_size, config.emb_dim), embed.shape
    final_norm = _get("model.norm.weight")
    assert final_norm.shape == (config.emb_dim,)

    layers_bf16 = []
    layers_packed = []
    for li in range(config.n_layers):
        base = f"model.layers.{li}"
        layer_kw = {
            "attn_norm": _get(f"{base}.input_layernorm.weight"),
            "ffn_norm": _get(f"{base}.post_attention_layernorm.weight"),
        }
        packed_kw = {}
        for hf_suffix, field in _HF_AWQ_LAYER_MAP.items():
            qw = _get(f"{base}.{hf_suffix}.qweight", as_int32=True)
            qz = _get(f"{base}.{hf_suffix}.qzeros", as_int32=True)
            sc = _get(f"{base}.{hf_suffix}.scales")
            layer_kw[field] = awq_dequant_layer(qw, qz, sc, gs=128)
            packed_kw[field] = awq_pack_for_npu_bfp16(
                qw, qz, sc, gs=128, n_tile=n_tile, k_chunk=k_chunk, M_seq=seq_len
            )
        layers_bf16.append(LayerWeights(**layer_kw))
        layers_packed.append(packed_kw)

    weights = LlamaWeights(
        embed_table=embed,
        layers=layers_bf16,
        final_norm=final_norm,
        lm_head=embed,  # tied, matches the int4 path
    )
    return weights, layers_packed
