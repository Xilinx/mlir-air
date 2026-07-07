# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Qwen2.5-3B Weight Loader

Mirrors llama32_1b_weights.py with Qwen2.5 deltas:
  - QKV bias (attention_bias / Qwen2 family): q_proj.bias (q_dim,),
    k_proj.bias (kv_dim,), v_proj.bias (kv_dim,). The bias is FUSED into the
    NPU rms_qkv_bias_rope ELF on-device (passed as static ELF args), not
    added on the host.
  - No QK-norm (that is Qwen3).
  - Dims (3B): emb=2048, hidden=11008, kv_dim=256 (2*128), head_dim=128.
  - Tied embeddings.

Weight convention: HF (out,in); our GEMM y=x@W → transpose to (in,out).
"""

import os
import glob as glob_module
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from ml_dtypes import bfloat16


@dataclass
class LlamaConfig:
    """Qwen2.5-3B hyperparameters (named LlamaConfig for downstream reuse)."""

    n_layers: int = 36
    emb_dim: int = 2048
    n_heads: int = 16
    head_dim: int = 128
    n_kv_heads: int = 2  # GQA: 8 Q heads per KV head (16 heads / 2 kv)
    hidden_dim: int = 11008
    vocab_size: int = 151936
    rope_base: float = 1000000.0
    qkv_bias: bool = True
    qk_norm: bool = False
    tie_word_embeddings: bool = True
    dtype: np.dtype = bfloat16


@dataclass
class LayerWeights:
    attn_norm: np.ndarray  # (emb_dim,)
    wq: np.ndarray  # (emb_dim, n_heads*head_dim)
    wk: np.ndarray  # (emb_dim, n_kv_heads*head_dim)
    wv: np.ndarray  # (emb_dim, n_kv_heads*head_dim)
    wo: np.ndarray  # (n_heads*head_dim, emb_dim)
    ffn_norm: np.ndarray  # (emb_dim,)
    w_gate: np.ndarray  # (emb_dim, hidden_dim)
    w_up: np.ndarray  # (emb_dim, hidden_dim)
    w_down: np.ndarray  # (hidden_dim, emb_dim)
    bq: np.ndarray  # (n_heads*head_dim,)   QKV bias
    bk: np.ndarray  # (n_kv_heads*head_dim,)
    bv: np.ndarray  # (n_kv_heads*head_dim,)


@dataclass
class LlamaWeights:
    embed_table: np.ndarray
    layers: List[LayerWeights] = field(default_factory=list)
    final_norm: np.ndarray = None
    lm_head: np.ndarray = None


# (field, needs_transpose). Biases are 1-D (no transpose).
_HF_LAYER_MAP = {
    "input_layernorm.weight": ("attn_norm", False),
    "self_attn.q_proj.weight": ("wq", True),
    "self_attn.k_proj.weight": ("wk", True),
    "self_attn.v_proj.weight": ("wv", True),
    "self_attn.o_proj.weight": ("wo", True),
    "self_attn.q_proj.bias": ("bq", False),
    "self_attn.k_proj.bias": ("bk", False),
    "self_attn.v_proj.bias": ("bv", False),
    "post_attention_layernorm.weight": ("ffn_norm", False),
    "mlp.gate_proj.weight": ("w_gate", True),
    "mlp.up_proj.weight": ("w_up", True),
    "mlp.down_proj.weight": ("w_down", True),
}


def _resolve_safetensor_files(model_path: str) -> List[str]:
    if os.path.isdir(model_path):
        files = sorted(glob_module.glob(os.path.join(model_path, "*.safetensors")))
        if not files:
            raise FileNotFoundError(f"No .safetensors files found in {model_path}")
        return files

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    pattern_glob = "*.safetensors"
    try:
        local_dir = snapshot_download(
            model_path,
            allow_patterns=["*.safetensors", "*.json"],
            local_files_only=True,
        )
        if not glob_module.glob(os.path.join(local_dir, pattern_glob)):
            raise LocalEntryNotFoundError(
                f"local cache for {model_path} has no .safetensors"
            )
    except LocalEntryNotFoundError:
        local_dir = snapshot_download(
            model_path, allow_patterns=["*.safetensors", "*.json"]
        )
    files = sorted(glob_module.glob(os.path.join(local_dir, pattern_glob)))
    if not files:
        raise FileNotFoundError(
            f"No .safetensors files found after downloading {model_path}"
        )
    return files


def _load_tensor(file_handle, key: str, dtype) -> np.ndarray:
    tensor = file_handle.get_tensor(key)
    if hasattr(tensor, "numpy"):
        tensor = tensor.numpy()
    return tensor.astype(dtype)


def load_weights(
    model_name_or_path: str, dtype=bfloat16, config: Optional[LlamaConfig] = None
) -> LlamaWeights:
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
        embed_table = _load_tensor(f, embed_key, dtype)
    assert embed_table.shape == (config.vocab_size, config.emb_dim), embed_table.shape

    qd = config.n_heads * config.head_dim
    kvd = config.n_kv_heads * config.head_dim

    layers = []
    for layer_idx in range(config.n_layers):
        layer_tensors = {}
        for hf_suffix, (field_name, needs_transpose) in _HF_LAYER_MAP.items():
            hf_key = f"model.layers.{layer_idx}.{hf_suffix}"
            if hf_key not in key_to_file:
                raise KeyError(f"Missing weight for layer {layer_idx}: {hf_key}")
            with safe_open(key_to_file[hf_key], framework="numpy") as f:
                tensor = _load_tensor(f, hf_key, dtype)
            if needs_transpose:
                tensor = np.ascontiguousarray(tensor.T)
            layer_tensors[field_name] = tensor

        layer = LayerWeights(**layer_tensors)
        assert layer.wq.shape == (
            config.emb_dim,
            qd,
        ), f"L{layer_idx} wq {layer.wq.shape}"
        assert layer.wk.shape == (
            config.emb_dim,
            kvd,
        ), f"L{layer_idx} wk {layer.wk.shape}"
        assert layer.wo.shape == (
            qd,
            config.emb_dim,
        ), f"L{layer_idx} wo {layer.wo.shape}"
        assert layer.bq.shape == (qd,), f"L{layer_idx} bq {layer.bq.shape}"
        assert layer.bk.shape == (kvd,), f"L{layer_idx} bk {layer.bk.shape}"
        assert layer.bv.shape == (kvd,), f"L{layer_idx} bv {layer.bv.shape}"
        layers.append(layer)

    norm_key = "model.norm.weight"
    with safe_open(key_to_file[norm_key], framework="numpy") as f:
        final_norm = _load_tensor(f, norm_key, dtype)
    assert final_norm.shape == (config.emb_dim,)

    lm_head_key = "lm_head.weight"
    if lm_head_key in key_to_file:
        with safe_open(key_to_file[lm_head_key], framework="numpy") as f:
            lm_head = _load_tensor(f, lm_head_key, dtype)
        assert lm_head.shape == (config.vocab_size, config.emb_dim)
    else:
        print("  Tied embeddings: reusing embed_table as lm_head.")
        lm_head = embed_table

    return LlamaWeights(
        embed_table=embed_table, layers=layers, final_norm=final_norm, lm_head=lm_head
    )


def generate_rope_lut(
    config: Optional[LlamaConfig] = None, seq_len: int = 2048, dtype=bfloat16
) -> np.ndarray:
    if config is None:
        config = LlamaConfig()
    head_dim = config.head_dim
    half = head_dim // 2
    theta = config.rope_base
    dim_indices = np.arange(0, head_dim, 2, dtype=np.float64)
    inv_freq = 1.0 / (theta ** (dim_indices / head_dim))
    positions = np.arange(seq_len, dtype=np.float64)
    angles = np.outer(positions, inv_freq)
    lut = np.empty((seq_len, head_dim), dtype=np.float64)
    lut[:, :half] = np.cos(angles)
    lut[:, half:] = np.sin(angles)
    return lut.astype(dtype)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    args = parser.parse_args()
    config = LlamaConfig()
    w = load_weights(args.model_path, config=config)
    L0 = w.layers[0]
    print(
        f"embed {w.embed_table.shape} final_norm {w.final_norm.shape} lm_head {w.lm_head.shape}"
    )
    print(
        f"L0 wq {L0.wq.shape} wk {L0.wk.shape} wo {L0.wo.shape} bq {L0.bq.shape} bk {L0.bk.shape}"
    )
    print(f"{len(w.layers)} layers loaded.")
