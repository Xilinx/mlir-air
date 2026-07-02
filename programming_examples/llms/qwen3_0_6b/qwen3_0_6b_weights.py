# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Qwen3-0.6B Weight Loader

Loads Qwen3-0.6B weights from HuggingFace safetensors into numpy arrays for
MLIR-AIR kernel invocations. Mirrors llama32_1b_weights.py with two Qwen3
deltas:
  - QK-norm: per-head RMSNorm weights q_norm/k_norm of shape (head_dim,),
    applied to Q and K (per head) after projection, before RoPE.
  - Decoupled head_dim: n_heads*head_dim (2048) != hidden_size (1024); the
    q_proj is (emb_dim, n_heads*head_dim), k/v_proj (emb_dim, n_kv*head_dim).
  - No qkv bias (attention_bias=false). (Qwen2.5 family DOES have bias.)
  - Tied embeddings (lm_head = embed_tokens).

Weight convention: HF stores linear weights as (out, in); our GEMM is
y = x @ W, so projections are transposed to (in, out) on load.
"""

import os
import glob as glob_module
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from ml_dtypes import bfloat16


@dataclass
class LlamaConfig:
    """Qwen3-0.6B hyperparameters (named LlamaConfig for downstream reuse)."""

    n_layers: int = 28
    emb_dim: int = 1024
    n_heads: int = 16
    head_dim: int = 128
    n_kv_heads: int = 8  # GQA: 2 Q heads per KV head
    hidden_dim: int = 3072
    vocab_size: int = 151936
    rope_base: float = 1000000.0
    qk_norm: bool = True
    tie_word_embeddings: bool = True
    dtype: np.dtype = bfloat16


@dataclass
class LayerWeights:
    """Weight matrices for a single Qwen3 transformer layer.

    All projection shapes follow W such that y = x @ W, i.e. (in, out).
    q_norm/k_norm are per-head RMSNorm weights of shape (head_dim,).
    """

    attn_norm: np.ndarray  # (emb_dim,)
    wq: np.ndarray  # (emb_dim, n_heads*head_dim)
    wk: np.ndarray  # (emb_dim, n_kv_heads*head_dim)
    wv: np.ndarray  # (emb_dim, n_kv_heads*head_dim)
    wo: np.ndarray  # (n_heads*head_dim, emb_dim)
    ffn_norm: np.ndarray  # (emb_dim,)
    w_gate: np.ndarray  # (emb_dim, hidden_dim)
    w_up: np.ndarray  # (emb_dim, hidden_dim)
    w_down: np.ndarray  # (hidden_dim, emb_dim)
    q_norm: np.ndarray  # (head_dim,)  Qwen3 QK-norm
    k_norm: np.ndarray  # (head_dim,)  Qwen3 QK-norm


@dataclass
class LlamaWeights:
    embed_table: np.ndarray  # (vocab_size, emb_dim)
    layers: List[LayerWeights] = field(default_factory=list)
    final_norm: np.ndarray = None  # (emb_dim,)
    lm_head: np.ndarray = None  # (vocab_size, emb_dim)


# HF suffix -> (field, needs_transpose). norms are 1-D (no transpose).
_HF_LAYER_MAP = {
    "input_layernorm.weight": ("attn_norm", False),
    "self_attn.q_proj.weight": ("wq", True),
    "self_attn.k_proj.weight": ("wk", True),
    "self_attn.v_proj.weight": ("wv", True),
    "self_attn.o_proj.weight": ("wo", True),
    "self_attn.q_norm.weight": ("q_norm", False),
    "self_attn.k_norm.weight": ("k_norm", False),
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
    model_name_or_path: str,
    dtype=bfloat16,
    config: Optional[LlamaConfig] = None,
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
    assert embed_table.shape == (config.vocab_size, config.emb_dim), (
        f"embed_table shape mismatch: expected "
        f"({config.vocab_size}, {config.emb_dim}), got {embed_table.shape}"
    )

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

        assert layer.attn_norm.shape == (config.emb_dim,)
        assert layer.wq.shape == (config.emb_dim, qd), f"L{layer_idx} wq {layer.wq.shape}"
        assert layer.wk.shape == (config.emb_dim, kvd), f"L{layer_idx} wk {layer.wk.shape}"
        assert layer.wv.shape == (config.emb_dim, kvd), f"L{layer_idx} wv {layer.wv.shape}"
        assert layer.wo.shape == (qd, config.emb_dim), f"L{layer_idx} wo {layer.wo.shape}"
        assert layer.ffn_norm.shape == (config.emb_dim,)
        assert layer.w_gate.shape == (config.emb_dim, config.hidden_dim)
        assert layer.w_up.shape == (config.emb_dim, config.hidden_dim)
        assert layer.w_down.shape == (config.hidden_dim, config.emb_dim)
        assert layer.q_norm.shape == (config.head_dim,), f"L{layer_idx} q_norm {layer.q_norm.shape}"
        assert layer.k_norm.shape == (config.head_dim,), f"L{layer_idx} k_norm {layer.k_norm.shape}"

        layers.append(layer)

    norm_key = "model.norm.weight"
    if norm_key not in key_to_file:
        raise KeyError(f"Missing weight: {norm_key}")
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
        embed_table=embed_table,
        layers=layers,
        final_norm=final_norm,
        lm_head=lm_head,
    )


def generate_rope_lut(
    config: Optional[LlamaConfig] = None,
    seq_len: int = 2048,
    dtype=bfloat16,
) -> np.ndarray:
    """Concatenated-layout RoPE LUT [cos..., sin...] of shape (seq_len, head_dim)."""
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

    parser = argparse.ArgumentParser(description="Load Qwen3-0.6B weights, print shapes")
    parser.add_argument("model_path", type=str)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    args = parser.parse_args()

    dtype = bfloat16 if args.dtype == "bfloat16" else np.float32
    config = LlamaConfig()
    print(f"Loading weights from: {args.model_path}")
    weights = load_weights(args.model_path, dtype=dtype, config=config)
    print(f"  embed_table : {weights.embed_table.shape}")
    print(f"  final_norm  : {weights.final_norm.shape}")
    print(f"  lm_head     : {weights.lm_head.shape} (tied={weights.lm_head is weights.embed_table})")
    L0 = weights.layers[0]
    print(f"  L0 wq {L0.wq.shape} wk {L0.wk.shape} wv {L0.wv.shape} wo {L0.wo.shape}")
    print(f"  L0 q_norm {L0.q_norm.shape} k_norm {L0.k_norm.shape}")
    print(f"  {len(weights.layers)} layers loaded successfully.")
