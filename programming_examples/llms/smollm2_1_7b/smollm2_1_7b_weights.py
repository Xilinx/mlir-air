# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SmolLM2-1.7B Weight Loader

Loads SmolLM2-1.7B weights from HuggingFace safetensors format and provides
them as numpy arrays suitable for MLIR-AIR kernel invocations.

Weight convention:
    HuggingFace stores linear weights as (out_features, in_features).
    Our GEMM convention is y = x @ W, so we need W as (in_features, out_features).
    All projection weights are transposed during loading.

Usage:
    from smollm2_1_7b_weights import load_weights, LlamaConfig

    config = LlamaConfig()
    weights = load_weights("HuggingFaceTB/SmolLM2-1.7B")
    print(weights.layers[0].wq.shape)  # (2048, 2048)
"""

import os
import glob as glob_module
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from ml_dtypes import bfloat16

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LlamaConfig:
    """SmolLM2-1.7B model hyperparameters."""

    n_layers: int = 24
    emb_dim: int = 2048
    n_heads: int = 32
    head_dim: int = 64
    n_kv_heads: int = 32  # MHA: 1 Q head per KV head (no GQA)
    hidden_dim: int = 8192
    vocab_size: int = 49152
    rope_base: float = 130000.0
    dtype: np.dtype = bfloat16


# ---------------------------------------------------------------------------
# Per-layer weight container
# ---------------------------------------------------------------------------


@dataclass
class LayerWeights:
    """Weight matrices for a single transformer layer.

    All shapes follow the convention W such that y = x @ W, i.e.
    (in_features, out_features).

    Attributes:
        attn_norm:  (emb_dim,)              RMSNorm weight for attention
        wq:         (emb_dim, emb_dim)      Q projection
        wk:         (emb_dim, n_kv_heads*head_dim)  K projection
        wv:         (emb_dim, n_kv_heads*head_dim)  V projection
        wo:         (emb_dim, emb_dim)      O projection
        ffn_norm:   (emb_dim,)              RMSNorm weight for FFN
        w_gate:     (emb_dim, hidden_dim)   Gate projection (SwiGLU)
        w_up:       (emb_dim, hidden_dim)   Up projection (SwiGLU)
        w_down:     (hidden_dim, emb_dim)   Down projection
    """

    # Shapes for SmolLM2-1.7B (pure MHA: kv_dim = n_kv_heads*head_dim =
    # 32*64 = 2048 = emb_dim, so wk/wv are square like wq).
    attn_norm: np.ndarray  # (2048,)
    wq: np.ndarray  # (2048, 2048)
    wk: np.ndarray  # (2048, 2048)
    wv: np.ndarray  # (2048, 2048)
    wo: np.ndarray  # (2048, 2048)
    ffn_norm: np.ndarray  # (2048,)
    w_gate: np.ndarray  # (2048, 8192)
    w_up: np.ndarray  # (2048, 8192)
    w_down: np.ndarray  # (8192, 2048)


# ---------------------------------------------------------------------------
# Full model weight container
# ---------------------------------------------------------------------------


@dataclass
class LlamaWeights:
    """All weights for a SmolLM2-1.7B model.

    Attributes:
        embed_table:  (vocab_size, emb_dim)  Token embeddings
        layers:       list of n_layers (24) LayerWeights
        final_norm:   (emb_dim,)             Final RMSNorm weight
        lm_head:      (vocab_size, emb_dim)  Output projection (may be tied)
    """

    embed_table: np.ndarray  # (vocab_size, emb_dim) = (49152, 2048)
    layers: List[LayerWeights] = field(default_factory=list)
    final_norm: np.ndarray = None  # (emb_dim,) = (2048,)
    lm_head: np.ndarray = None  # (vocab_size, emb_dim) = (49152, 2048)


# ---------------------------------------------------------------------------
# HuggingFace name mapping
# ---------------------------------------------------------------------------

# Map from HuggingFace parameter names to our field names.
# Weights marked with transpose=True are stored as (out, in) in HF and need
# to be transposed to (in, out) for our y = x @ W convention.

_HF_LAYER_MAP = {
    "input_layernorm.weight": ("attn_norm", False),
    "self_attn.q_proj.weight": ("wq", True),
    "self_attn.k_proj.weight": ("wk", True),
    "self_attn.v_proj.weight": ("wv", True),
    "self_attn.o_proj.weight": ("wo", True),
    "post_attention_layernorm.weight": ("ffn_norm", False),
    "mlp.gate_proj.weight": ("w_gate", True),
    "mlp.up_proj.weight": ("w_up", True),
    "mlp.down_proj.weight": ("w_down", True),
}


# ---------------------------------------------------------------------------
# Safetensors loading helpers
# ---------------------------------------------------------------------------


def _resolve_safetensor_files(model_path: str) -> List[str]:
    """Find all safetensor shard files for a model.

    Args:
        model_path: Either a local directory path or a HuggingFace model ID
                    (e.g. "HuggingFaceTB/SmolLM2-1.7B").

    Returns:
        List of absolute paths to .safetensors files.
    """
    if os.path.isdir(model_path):
        # Local directory -- find all safetensors files
        pattern = os.path.join(model_path, "*.safetensors")
        files = sorted(glob_module.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No .safetensors files found in {model_path}")
        return files

    # HuggingFace model ID -- resolve via huggingface_hub. Try the offline
    # path first so a cache hit doesn't print HF's "Fetching N files /
    # Download complete: 0.00B" progress UI; fall back to a network
    # download only if the cache is missing or incomplete.
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    pattern_glob = "*.safetensors"
    try:
        local_dir = snapshot_download(
            model_path,
            allow_patterns=["*.safetensors", "*.json"],
            local_files_only=True,
        )
        # local_files_only=True returns whatever subset of allow_patterns is
        # already cached; it does NOT raise when some files match. A persistent
        # CI runner that previously did `AutoConfig.from_pretrained` will have
        # config.json cached but no safetensors. Force the network branch
        # when no .safetensors are actually present.
        if not glob_module.glob(os.path.join(local_dir, pattern_glob)):
            raise LocalEntryNotFoundError(
                f"local cache for {model_path} has no .safetensors"
            )
    except LocalEntryNotFoundError:
        local_dir = snapshot_download(
            model_path,
            allow_patterns=["*.safetensors", "*.json"],
        )
    files = sorted(glob_module.glob(os.path.join(local_dir, pattern_glob)))
    if not files:
        raise FileNotFoundError(
            f"No .safetensors files found after downloading {model_path}"
        )
    return files


def _load_tensor(file_handle, key: str, dtype) -> np.ndarray:
    """Load a single tensor from an open safetensors file handle.

    The safetensors library returns tensors as numpy arrays. We cast to the
    requested dtype after loading.
    """
    tensor = file_handle.get_tensor(key)
    # safetensors returns numpy arrays; ensure correct dtype
    if hasattr(tensor, "numpy"):
        tensor = tensor.numpy()
    return tensor.astype(dtype)


# ---------------------------------------------------------------------------
# Main loading function
# ---------------------------------------------------------------------------


def _check_shape(name: str, arr: np.ndarray, expected: tuple) -> None:
    """Raise ValueError on shape mismatch. Used instead of `assert` so the
    checks survive `python -O` (which strips asserts) — this is a user-facing
    weight loader and a silent shape mismatch turns into a hard-to-debug
    downstream failure."""
    if arr.shape != expected:
        raise ValueError(f"{name} shape mismatch: expected {expected}, got {arr.shape}")


def load_weights(
    model_name_or_path: str,
    dtype=bfloat16,
    config: Optional[LlamaConfig] = None,
) -> LlamaWeights:
    """Load SmolLM2-1.7B weights from safetensors into numpy arrays.

    Supports both local directories and HuggingFace model IDs.

    Args:
        model_name_or_path: Path to a local directory containing .safetensors
            files, or a HuggingFace model ID like "HuggingFaceTB/SmolLM2-1.7B".
        dtype: Target numpy dtype for all weight arrays. Default is bfloat16.
        config: Optional LlamaConfig. If None, uses default SmolLM2-1.7B config.

    Returns:
        A LlamaWeights instance with all weights loaded and correctly shaped.
    """
    from safetensors import safe_open

    if config is None:
        config = LlamaConfig()

    safetensor_files = _resolve_safetensor_files(model_name_or_path)

    # Collect all key -> file mappings across shards
    key_to_file = {}
    for filepath in safetensor_files:
        with safe_open(filepath, framework="numpy") as f:
            for key in f.keys():
                key_to_file[key] = filepath

    # --- Load embedding table ---
    embed_key = "model.embed_tokens.weight"
    if embed_key not in key_to_file:
        raise KeyError(f"Missing weight: {embed_key}")
    with safe_open(key_to_file[embed_key], framework="numpy") as f:
        embed_table = _load_tensor(f, embed_key, dtype)
    _check_shape("embed_table", embed_table, (config.vocab_size, config.emb_dim))

    # --- Load per-layer weights ---
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

        # Sanity check shapes
        _lp = f"Layer {layer_idx}"
        _check_shape(f"{_lp} attn_norm", layer.attn_norm, (config.emb_dim,))
        _check_shape(
            f"{_lp} wq", layer.wq, (config.emb_dim, config.n_heads * config.head_dim)
        )
        _check_shape(
            f"{_lp} wk", layer.wk, (config.emb_dim, config.n_kv_heads * config.head_dim)
        )
        _check_shape(
            f"{_lp} wv", layer.wv, (config.emb_dim, config.n_kv_heads * config.head_dim)
        )
        _check_shape(f"{_lp} wo", layer.wo, (config.emb_dim, config.emb_dim))
        _check_shape(f"{_lp} ffn_norm", layer.ffn_norm, (config.emb_dim,))
        _check_shape(f"{_lp} w_gate", layer.w_gate, (config.emb_dim, config.hidden_dim))
        _check_shape(f"{_lp} w_up", layer.w_up, (config.emb_dim, config.hidden_dim))
        _check_shape(f"{_lp} w_down", layer.w_down, (config.hidden_dim, config.emb_dim))

        layers.append(layer)

    # --- Load final RMSNorm ---
    norm_key = "model.norm.weight"
    if norm_key not in key_to_file:
        raise KeyError(f"Missing weight: {norm_key}")
    with safe_open(key_to_file[norm_key], framework="numpy") as f:
        final_norm = _load_tensor(f, norm_key, dtype)
    _check_shape("final_norm", final_norm, (config.emb_dim,))

    # --- Load lm_head (or tie to embeddings) ---
    lm_head_key = "lm_head.weight"
    if lm_head_key in key_to_file:
        with safe_open(key_to_file[lm_head_key], framework="numpy") as f:
            lm_head = _load_tensor(f, lm_head_key, dtype)
        # lm_head is stored as (vocab_size, emb_dim) in HF -- no transpose
        # because the output logits = hidden @ lm_head.T is handled at
        # inference time, and we store it in the same layout as embed_table.
        _check_shape("lm_head", lm_head, (config.vocab_size, config.emb_dim))
    else:
        # SmolLM2 (like the small Llamas) ties lm_head to embed_tokens — the
        # checkpoint omits lm_head.weight by design and the runtime is
        # expected to compute logits = h @ embed_table.T.
        print("  Tied embeddings: reusing embed_table as lm_head.")
        lm_head = embed_table

    return LlamaWeights(
        embed_table=embed_table,
        layers=layers,
        final_norm=final_norm,
        lm_head=lm_head,
    )


# ---------------------------------------------------------------------------
# Synthetic-weights builder (CI smoke / verify without HuggingFace download)
# ---------------------------------------------------------------------------


def synthetic_weights(
    config: Optional[LlamaConfig] = None,
    seed: int = 42,
    scale: float = 0.02,
) -> "LlamaWeights":
    """Build a LlamaWeights object filled with deterministic random values.

    The same RNG seed produces identical weights for the NPU and CPU reference
    paths, so `--verify` mode can compare them numerically without ever
    touching HuggingFace. Output magnitudes match HF Llama's init scale
    (~0.02) so activations stay within BF16 dynamic range.
    """
    if config is None:
        config = LlamaConfig()

    rng = np.random.default_rng(seed)

    def _rand(shape):
        return (rng.standard_normal(shape).astype(np.float32) * scale).astype(bfloat16)

    def _ones(shape):
        return np.ones(shape, dtype=bfloat16)

    embed = _rand((config.vocab_size, config.emb_dim))
    layers = [
        LayerWeights(
            attn_norm=_ones((config.emb_dim,)),
            wq=_rand((config.emb_dim, config.emb_dim)),
            wk=_rand((config.emb_dim, config.n_kv_heads * config.head_dim)),
            wv=_rand((config.emb_dim, config.n_kv_heads * config.head_dim)),
            wo=_rand((config.emb_dim, config.emb_dim)),
            ffn_norm=_ones((config.emb_dim,)),
            w_gate=_rand((config.emb_dim, config.hidden_dim)),
            w_up=_rand((config.emb_dim, config.hidden_dim)),
            w_down=_rand((config.hidden_dim, config.emb_dim)),
        )
        for _ in range(config.n_layers)
    ]
    return LlamaWeights(
        embed_table=embed,
        layers=layers,
        final_norm=_ones((config.emb_dim,)),
        lm_head=embed,  # tied
    )


# ---------------------------------------------------------------------------
# RoPE look-up table generation
# ---------------------------------------------------------------------------


def generate_rope_lut(
    config: Optional[LlamaConfig] = None,
    seq_len: int = 2048,
    dtype=bfloat16,
) -> np.ndarray:
    """Generate a pre-computed RoPE (Rotary Position Embedding) look-up table.

    The LUT uses concatenated layout: [cos_0, ..., cos_{half-1}, sin_0, ..., sin_{half-1}]
    matching the half-split RoPE kernel (rope_halfsplit.cc) and HuggingFace Llama convention.

    For position *pos* and dimension index *i* (0-indexed, i < head_dim/2):
        freq_i           = 1.0 / (theta ^ (2*i / head_dim))
        angle            = pos * freq_i
        LUT[pos, i]             = cos(angle)
        LUT[pos, i + head_dim/2] = sin(angle)

    Args:
        config: Model config (uses rope_base and head_dim). Defaults to
                SmolLM2-1.7B config.
        seq_len: Maximum sequence length for the LUT.
        dtype: Output dtype. Default is bfloat16.

    Returns:
        np.ndarray of shape (seq_len, head_dim) with concatenated [cos..., sin...].
    """
    if config is None:
        config = LlamaConfig()

    head_dim = config.head_dim
    half = head_dim // 2
    theta = config.rope_base

    # Compute inverse frequencies: shape (head_dim/2,)
    dim_indices = np.arange(0, head_dim, 2, dtype=np.float64)
    inv_freq = 1.0 / (theta ** (dim_indices / head_dim))

    # Compute angles: shape (seq_len, head_dim/2)
    positions = np.arange(seq_len, dtype=np.float64)
    angles = np.outer(positions, inv_freq)  # (seq_len, head_dim/2)

    # Compute cos and sin
    cos_vals = np.cos(angles)  # (seq_len, head_dim/2)
    sin_vals = np.sin(angles)  # (seq_len, head_dim/2)

    # Concatenate: [cos_0, ..., cos_{half-1}, sin_0, ..., sin_{half-1}]
    lut = np.empty((seq_len, head_dim), dtype=np.float64)
    lut[:, :half] = cos_vals
    lut[:, half:] = sin_vals

    return lut.astype(dtype)


# ---------------------------------------------------------------------------
# Main -- test loading and print shapes
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load SmolLM2-1.7B weights and print shapes",
    )
    parser.add_argument(
        "model_path",
        type=str,
        help=(
            "Path to local model directory or HuggingFace model ID "
            "(e.g. HuggingFaceTB/SmolLM2-1.7B)"
        ),
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bfloat16", "float32"],
        default="bfloat16",
        help="Data type for loaded weights",
    )
    parser.add_argument(
        "--rope-seq-len",
        type=int,
        default=2048,
        help="Sequence length for RoPE LUT generation",
    )
    args = parser.parse_args()

    dtype = bfloat16 if args.dtype == "bfloat16" else np.float32
    config = LlamaConfig()

    print(f"Loading weights from: {args.model_path}")
    print(f"Target dtype: {args.dtype}")
    print(f"Config: {config}")
    print()

    weights = load_weights(args.model_path, dtype=dtype, config=config)

    print("=== Global weights ===")
    print(
        f"  embed_table : {weights.embed_table.shape}  dtype={weights.embed_table.dtype}"
    )
    print(
        f"  final_norm  : {weights.final_norm.shape}  dtype={weights.final_norm.dtype}"
    )
    print(f"  lm_head     : {weights.lm_head.shape}  dtype={weights.lm_head.dtype}")
    tied = weights.lm_head is weights.embed_table
    print(f"  lm_head tied to embed_table: {tied}")
    print()

    print(f"=== Per-layer weights ({config.n_layers} layers) ===")
    for i, layer in enumerate(weights.layers):
        print(f"  Layer {i:2d}:")
        print(f"    attn_norm : {layer.attn_norm.shape}")
        print(f"    wq        : {layer.wq.shape}")
        print(f"    wk        : {layer.wk.shape}")
        print(f"    wv        : {layer.wv.shape}")
        print(f"    wo        : {layer.wo.shape}")
        print(f"    ffn_norm  : {layer.ffn_norm.shape}")
        print(f"    w_gate    : {layer.w_gate.shape}")
        print(f"    w_up      : {layer.w_up.shape}")
        print(f"    w_down    : {layer.w_down.shape}")
        # Only print first layer in detail; just confirm rest exist
        if i == 0:
            continue
        if i == 1:
            print("    ... (remaining layers have identical shapes)")
            break
    print()

    print("=== RoPE LUT ===")
    rope_lut = generate_rope_lut(config, seq_len=args.rope_seq_len, dtype=dtype)
    print(f"  rope_lut    : {rope_lut.shape}  dtype={rope_lut.dtype}")
    # Show a few values at position 0 (should be [1, 0, 1, 0, ...] since cos(0)=1, sin(0)=0)
    print(f"  rope_lut[0, :8] = {rope_lut[0, :8].astype(np.float32)}")
    print(f"  rope_lut[1, :8] = {rope_lut[1, :8].astype(np.float32)}")
    print()

    print("All weights loaded successfully.")
