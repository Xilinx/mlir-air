# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""LLAMA-3.2-1B Weight Loader

Loads LLAMA-3.2-1B weights from HuggingFace safetensors format and provides
them as numpy arrays suitable for MLIR-AIR kernel invocations.

Weight convention:
    HuggingFace stores linear weights as (out_features, in_features).
    Our GEMM convention is y = x @ W, so we need W as (in_features, out_features).
    All projection weights are transposed during loading.

Usage:
    from llama32_1b_weights import load_weights, LlamaConfig

    config = LlamaConfig()
    weights = load_weights("meta-llama/Llama-3.2-1B")
    print(weights.layers[0].wq.shape)  # (2048, 2048)
"""

import os
import sys
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
    """LLAMA-3.2-1B model hyperparameters."""

    n_layers: int = 16
    emb_dim: int = 2048
    n_heads: int = 32
    head_dim: int = 64
    n_kv_heads: int = 8  # GQA: 4 Q heads per KV head
    hidden_dim: int = 8192
    vocab_size: int = 128256
    rope_base: float = 500000.0
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

    attn_norm: np.ndarray  # (2048,)
    wq: np.ndarray  # (2048, 2048)
    wk: np.ndarray  # (2048, 512)
    wv: np.ndarray  # (2048, 512)
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
    """All weights for a LLAMA-3.2-1B model.

    Attributes:
        embed_table:  (vocab_size, emb_dim)  Token embeddings
        layers:       list of 16 LayerWeights
        final_norm:   (emb_dim,)             Final RMSNorm weight
        lm_head:      (vocab_size, emb_dim)  Output projection (may be tied)
    """

    embed_table: np.ndarray  # (128256, 2048)
    layers: List[LayerWeights] = field(default_factory=list)
    final_norm: np.ndarray = None  # (2048,)
    lm_head: np.ndarray = None  # (128256, 2048)


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

# AutoAWQ stores each Linear as three tensors. Field name = the dataclass
# field that owns the bf16 dequant (used by the CPU prefill placeholder).
_HF_AWQ_LINEARS = {
    "self_attn.q_proj": "wq",
    "self_attn.k_proj": "wk",
    "self_attn.v_proj": "wv",
    "self_attn.o_proj": "wo",
    "mlp.gate_proj": "w_gate",
    "mlp.up_proj": "w_up",
    "mlp.down_proj": "w_down",
}


# ---------------------------------------------------------------------------
# Safetensors loading helpers
# ---------------------------------------------------------------------------


def _resolve_safetensor_files(model_path: str) -> List[str]:
    """Find all safetensor shard files for a model.

    Args:
        model_path: Either a local directory path or a HuggingFace model ID
                    (e.g. "meta-llama/Llama-3.2-1B").

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

    try:
        local_dir = snapshot_download(
            model_path,
            allow_patterns=["*.safetensors", "*.json"],
            local_files_only=True,
        )
    except LocalEntryNotFoundError:
        local_dir = snapshot_download(
            model_path,
            allow_patterns=["*.safetensors", "*.json"],
        )
    pattern = os.path.join(local_dir, "*.safetensors")
    files = sorted(glob_module.glob(pattern))
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


def load_weights(
    model_name_or_path: str,
    dtype=bfloat16,
    config: Optional[LlamaConfig] = None,
) -> LlamaWeights:
    """Load LLAMA-3.2-1B weights from safetensors into numpy arrays.

    Supports both local directories and HuggingFace model IDs.

    Args:
        model_name_or_path: Path to a local directory containing .safetensors
            files, or a HuggingFace model ID like "meta-llama/Llama-3.2-1B".
        dtype: Target numpy dtype for all weight arrays. Default is bfloat16.
        config: Optional LlamaConfig. If None, uses default LLAMA-3.2-1B config.

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
    assert embed_table.shape == (config.vocab_size, config.emb_dim), (
        f"embed_table shape mismatch: expected "
        f"({config.vocab_size}, {config.emb_dim}), got {embed_table.shape}"
    )

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
        assert layer.attn_norm.shape == (
            config.emb_dim,
        ), f"Layer {layer_idx} attn_norm: {layer.attn_norm.shape}"
        assert layer.wq.shape == (
            config.emb_dim,
            config.n_heads * config.head_dim,
        ), f"Layer {layer_idx} wq: {layer.wq.shape}"
        assert layer.wk.shape == (
            config.emb_dim,
            config.n_kv_heads * config.head_dim,
        ), f"Layer {layer_idx} wk: {layer.wk.shape}"
        assert layer.wv.shape == (
            config.emb_dim,
            config.n_kv_heads * config.head_dim,
        ), f"Layer {layer_idx} wv: {layer.wv.shape}"
        assert layer.wo.shape == (
            config.emb_dim,
            config.emb_dim,
        ), f"Layer {layer_idx} wo: {layer.wo.shape}"
        assert layer.ffn_norm.shape == (
            config.emb_dim,
        ), f"Layer {layer_idx} ffn_norm: {layer.ffn_norm.shape}"
        assert layer.w_gate.shape == (
            config.emb_dim,
            config.hidden_dim,
        ), f"Layer {layer_idx} w_gate: {layer.w_gate.shape}"
        assert layer.w_up.shape == (
            config.emb_dim,
            config.hidden_dim,
        ), f"Layer {layer_idx} w_up: {layer.w_up.shape}"
        assert layer.w_down.shape == (
            config.hidden_dim,
            config.emb_dim,
        ), f"Layer {layer_idx} w_down: {layer.w_down.shape}"

        layers.append(layer)

    # --- Load final RMSNorm ---
    norm_key = "model.norm.weight"
    if norm_key not in key_to_file:
        raise KeyError(f"Missing weight: {norm_key}")
    with safe_open(key_to_file[norm_key], framework="numpy") as f:
        final_norm = _load_tensor(f, norm_key, dtype)
    assert final_norm.shape == (
        config.emb_dim,
    ), f"final_norm shape mismatch: {final_norm.shape}"

    # --- Load lm_head (or tie to embeddings) ---
    lm_head_key = "lm_head.weight"
    if lm_head_key in key_to_file:
        with safe_open(key_to_file[lm_head_key], framework="numpy") as f:
            lm_head = _load_tensor(f, lm_head_key, dtype)
        # lm_head is stored as (vocab_size, emb_dim) in HF -- no transpose
        # because the output logits = hidden @ lm_head.T is handled at
        # inference time, and we store it in the same layout as embed_table.
        assert lm_head.shape == (
            config.vocab_size,
            config.emb_dim,
        ), f"lm_head shape mismatch: {lm_head.shape}"
    else:
        # LLAMA-3.2-1B (and other small Llamas) tie lm_head to embed_tokens
        # — the checkpoint omits lm_head.weight by design and the runtime
        # is expected to compute logits = h @ embed_table.T.
        print("  Tied embeddings: reusing embed_table as lm_head.")
        lm_head = embed_table

    return LlamaWeights(
        embed_table=embed_table,
        layers=layers,
        final_norm=final_norm,
        lm_head=lm_head,
    )


# ---------------------------------------------------------------------------
# HuggingFace AutoAWQ loader
# ---------------------------------------------------------------------------

# Mapping LayerWeights field name -> per-layer packed-BO attribute name.
# Packed BOs are dynamically attached to each LayerWeights instance (same
# pattern inference.py uses for ._wq_t etc.), keyed by the dataclass field
# to avoid extending the dataclass schema. The decode-side runtime reads
# `layer._wq_packed`, `layer._wo_packed`, etc. The fused FFN ELF
# (o_gemv_ffn_int4) wants gate+up as ONE row-interleaved packed BO and
# is exposed separately as `_wgateup_packed`.
_AWQ_PACKED_ATTR = {
    "wq": "_wq_packed",
    "wk": "_wk_packed",
    "wv": "_wv_packed",
    "wo": "_wo_packed",
    "w_down": "_wdown_packed",
}
# w_gate and w_up are NOT in _AWQ_PACKED_ATTR — they go through a row-level
# interleave first (gate[i] -> row 2i, up[i] -> row 2i+1) before packing,
# so the int4 FFN ELF can consume them in one BO.


def load_weights_awq(
    model_name_or_path: str,
    config: Optional[LlamaConfig] = None,
    group_size: int = 128,
    m_tile: int = 8,
    k_chunk: int = 2048,
    n_cores: int = 8,
) -> LlamaWeights:
    """Load a HuggingFace AutoAWQ Llama checkpoint.

    For each Linear, stashes BOTH:
      - the bf16 dequant on the existing LayerWeights field (wq/wk/.../w_down),
        for the CPU prefill placeholder via reference.transformer_block;
      - the per-tile packed uint8 BO on `_<field>_packed`, for the NPU int4
        decode ELFs (rms_qkv_int4_rope and o_gemv_ffn_int4).

    Args:
        model_name_or_path: local dir or HF model id of an AutoAWQ checkpoint.
        config: model hyperparameters (defaults to Llama-3.2-1B).
        group_size: AWQ group size (typical 128). Must match the checkpoint.
        m_tile, k_chunk, n_cores: GEMV packed-BO tiling parameters; defaults
            match the int4 decode ELF builders.

    Returns:
        LlamaWeights with packed-BO attributes attached.
    """
    from safetensors import safe_open
    from awq_repacker import dequant_to_bf16, repack_for_gemv, repack_hf_awq_linear

    sys.path.insert(
        0,
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "matrix_vector_multiplication",
            "int4_awq",
        ),
    )
    from matvec_int4_packed import pack_inputs as _pack_inputs

    if config is None:
        config = LlamaConfig()

    safetensor_files = _resolve_safetensor_files(model_name_or_path)

    key_to_file = {}
    for filepath in safetensor_files:
        with safe_open(filepath, framework="numpy") as f:
            for key in f.keys():
                key_to_file[key] = filepath

    # --- Embedding table ---
    embed_key = "model.embed_tokens.weight"
    if embed_key not in key_to_file:
        raise KeyError(f"Missing weight: {embed_key}")
    with safe_open(key_to_file[embed_key], framework="numpy") as f:
        embed_table = _load_tensor(f, embed_key, bfloat16)

    # --- Per-layer ---
    layers: List[LayerWeights] = []
    for layer_idx in range(config.n_layers):
        # Non-quantized: layernorms.
        attn_norm_key = f"model.layers.{layer_idx}.input_layernorm.weight"
        ffn_norm_key = f"model.layers.{layer_idx}.post_attention_layernorm.weight"
        with safe_open(key_to_file[attn_norm_key], framework="numpy") as f:
            attn_norm = _load_tensor(f, attn_norm_key, bfloat16)
        with safe_open(key_to_file[ffn_norm_key], framework="numpy") as f:
            ffn_norm = _load_tensor(f, ffn_norm_key, bfloat16)

        # Each quantized Linear: load qweight/qzeros/scales, then both
        # (a) dequant to bf16 -> existing LayerWeights field, for CPU prefill,
        # (b) repack-and-pack -> packed BO, for NPU int4 decode.
        # gate/up are special: their NPU packed BO must be ONE interleaved
        # matrix (gate row 0, up row 0, gate row 1, ...) so the int4 ELF2
        # (matvec_int4_swiglu_rms) can consume them in one input slot.
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
            # (a) bf16 dequant for CPU prefill: shape [in, out] (matches
            # transformer_block's wq[in, out] convention — no transpose).
            linear_bf16[field_name] = dequant_to_bf16(qw, qz, sc, group_size)
            # (b) packed BO for NPU decode; gate/up are deferred until both
            # are loaded so we can interleave them at the nibble level.
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
        # row 2i+1 = up[i]. Then pack into one BO sized [2*hidden, emb] for
        # arg7 of o_gemv_ffn_int4.
        if gate_quants is None or up_quants is None:
            raise RuntimeError(
                "Could not find both mlp.gate_proj and mlp.up_proj AWQ tensors"
            )
        g_q, g_s, g_z = gate_quants
        u_q, u_s, u_z = up_quants
        h_out, k_half = g_q.shape  # h_out = hidden_dim, k_half = K/2
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

    # --- Final norm and lm_head (both bf16 in this checkpoint) ---
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
                LLAMA-3.2-1B config.
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
        description="Load LLAMA-3.2-1B weights and print shapes",
    )
    parser.add_argument(
        "model_path",
        type=str,
        help=(
            "Path to local model directory or HuggingFace model ID "
            "(e.g. meta-llama/Llama-3.2-1B)"
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
