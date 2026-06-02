# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AWQ loader + packers for the int4-AWQ prefill stitchers.

Two consumers of an AWQ "gemm v1" checkpoint:

* `awq_pack_for_npu`: bit-shuffle qweight/qzeros/scales into the packed BO
  layout that `matmul_int4_packed.build_module` consumes — no
  re-quantization. The kernel's in-tile dequant uses the same `(q-z)*s`
  formula AWQ does, so this round-trips with **zero numerical loss**
  vs. running AutoAWQ's CUDA dequant on the same checkpoint.

* `awq_dequant_layer`: same arithmetic, but produce a dense
  `(in_features, out_features)` bf16 weight matrix for the CPU / HF
  reference path.

`load_awq_weights` wires both into the existing `LlamaWeights` dataclass
so the prefill driver and the bf16 reference both see weights in the
same shape, and the only thing that differs is what compute path they
flow through.

`fake_quantize_awq_int4` + `pack_weight_for_int4_gemm` are kept for the
GEMM standalone example (which has no AWQ checkpoint of its own).
"""

import glob
import os
import sys
from typing import Optional

import numpy as np
from ml_dtypes import bfloat16


_INT4_GEMM_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "matrix_multiplication",
    "int4_awq",
)
if _INT4_GEMM_DIR not in sys.path:
    sys.path.insert(0, _INT4_GEMM_DIR)
from matmul_int4_packed import pack_inputs as pack_inputs_gemm  # noqa: E402


_LLAMA_BF16_DIR = os.path.join(os.path.dirname(__file__), "..", "llama32_1b")
if _LLAMA_BF16_DIR not in sys.path:
    sys.path.insert(0, _LLAMA_BF16_DIR)
from llama32_1b_weights import LlamaConfig, LayerWeights, LlamaWeights  # noqa: E402


# ---------------------------------------------------------------------------
# AWQ gemm-v1 packing constants
# ---------------------------------------------------------------------------
#
# AutoAWQ "gemm" version with "pack_method": "reorder" stores 8 uint4
# values per int32 along the OUT_FEATURES axis, and within each 8-tuple
# the stored order is permuted as [0, 2, 4, 6, 1, 3, 5, 7]. To recover
# the natural-order weight we shift-and-mask the 8 nibbles, then
# inverse-permute.

_AWQ_REORDER_PERM = np.array([0, 2, 4, 6, 1, 3, 5, 7], dtype=np.int64)
_AWQ_INV_PERM = np.argsort(_AWQ_REORDER_PERM)  # [0, 4, 1, 5, 2, 6, 3, 7]
_AWQ_SHIFTS = np.array([0, 4, 8, 12, 16, 20, 24, 28], dtype=np.uint32)


def _awq_unpack_int32_to_nibbles(packed_i32, n_outer):
    """Unpack (n_outer, n_packed) int32 -> (n_outer, n_packed*8) uint8.

    Each int32 contains 8 packed 4-bit values along its bit axis; the
    permutation within the group of 8 is undone here so the returned
    columns are in natural [0, 1, 2, ..., N-1] order.
    """
    u32 = packed_i32.view(np.uint32)
    u = ((u32[..., None] >> _AWQ_SHIFTS) & 0xF).astype(np.uint8)  # (..., n_pack, 8)
    u = u[..., _AWQ_INV_PERM]
    return u.reshape(n_outer, -1)


# ---------------------------------------------------------------------------
# Path 1: dequant to bf16 — used by the HF / CPU reference
# ---------------------------------------------------------------------------


def awq_dequant_layer(qweight_i32, qzeros_i32, scales_bf16, gs=128):
    """AWQ qweight/qzeros/scales -> dense `(K, N)` bf16 weight.

    Dequant: W[k, n] = (q[k, n] - z[k//gs, n]) * s[k//gs, n].
    Output layout matches our `y = x @ W` convention (in_features, out_features).
    """
    K, n_packed = qweight_i32.shape
    N = n_packed * 8
    n_groups = K // gs
    assert K % gs == 0, f"K={K} not divisible by gs={gs}"
    assert qzeros_i32.shape == (n_groups, n_packed)
    assert scales_bf16.shape == (n_groups, N)

    W_KN = _awq_unpack_int32_to_nibbles(qweight_i32, K)
    Z_GN = _awq_unpack_int32_to_nibbles(qzeros_i32, n_groups)
    S = np.asarray(scales_bf16).astype(np.float32)

    Z_full = np.repeat(Z_GN.astype(np.float32), gs, axis=0)
    S_full = np.repeat(S, gs, axis=0)
    W_dq = (W_KN.astype(np.float32) - Z_full) * S_full
    return W_dq.astype(bfloat16)  # (K, N)


# ---------------------------------------------------------------------------
# Path 2: bit-shuffle AWQ -> our kernel's packed BO layout (no re-quant)
# ---------------------------------------------------------------------------


def awq_pack_for_npu(qweight_i32, qzeros_i32, scales_bf16,
                     gs=128, n_tile=16, k_chunk=128, M_seq=2048):
    """AWQ qweight/qzeros/scales -> `(N/n_tile, K/k_chunk, tile_bytes)` u8.

    Pure layout conversion: the nibble values are preserved exactly, just
    re-shuffled into the K-packed byte order the int4 GEMM kernel
    expects (low4 = column 2k, high4 = column 2k+1, output-major rows).
    M_seq is unused by `pack_inputs` for layout; it is only forwarded for
    its signature compatibility.
    """
    K, n_packed = qweight_i32.shape
    N = n_packed * 8
    n_groups = K // gs

    W_KN = _awq_unpack_int32_to_nibbles(qweight_i32, K)
    Z_GN = _awq_unpack_int32_to_nibbles(qzeros_i32, n_groups)

    # Transpose K↔N -> output-major.
    W_NK = np.ascontiguousarray(W_KN.T)

    # K-pack: byte = low4(col 2k) | (high4(col 2k+1) << 4).
    W_q = (W_NK[:, 0::2] | (W_NK[:, 1::2] << 4)).astype(np.uint8)  # (N, K/2)

    return pack_inputs_gemm(
        W_q, np.asarray(scales_bf16).astype(bfloat16), Z_GN,
        M=M_seq, K=K, N=N, GS=gs, N_TILE=n_tile, K_CHUNK=k_chunk,
    )


# ---------------------------------------------------------------------------
# Whole-model loader: AWQ HF checkpoint -> (bf16 LlamaWeights, packed BOs)
# ---------------------------------------------------------------------------


# HF projection layer suffix -> (our field name, transpose-needed-for-bf16-view)
# Every quantized matrix in this checkpoint stores qweight as
# (in_features, out_features/8), which is already the (K, N) order our
# code wants, so no transpose is needed after `awq_dequant_layer`.
_HF_AWQ_LAYER_MAP = {
    "self_attn.q_proj": "wq",
    "self_attn.k_proj": "wk",
    "self_attn.v_proj": "wv",
    "self_attn.o_proj": "wo",
    "mlp.gate_proj": "w_gate",
    "mlp.up_proj": "w_up",
    "mlp.down_proj": "w_down",
}


def _resolve_safetensor_files(model_path):
    if os.path.isdir(model_path):
        files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
        if not files:
            raise FileNotFoundError(f"No .safetensors in {model_path}")
        return files
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    try:
        local = snapshot_download(model_path,
            allow_patterns=["*.safetensors", "*.json"], local_files_only=True)
    except LocalEntryNotFoundError:
        local = snapshot_download(model_path,
            allow_patterns=["*.safetensors", "*.json"])
    return sorted(glob.glob(os.path.join(local, "*.safetensors")))


def load_awq_weights(
    model_path: str,
    config: Optional[LlamaConfig] = None,
    gs: int = 128,
    n_tile: int = 16,
    k_chunk: int = 128,
    seq_len: int = 2048,
):
    """Load an AWQ HF checkpoint into both a bf16 `LlamaWeights` (for the
    reference path) and a parallel list of per-layer packed BOs (for NPU).

    Returns: (LlamaWeights, packed_layers)
        LlamaWeights: drop-in bf16 view (every projection dequantized via
            `(q - z) * s`).
        packed_layers: list of len `n_layers`, each a dict with keys
            wq, wk, wv, wo, w_gate, w_up, w_down -> packed uint8 3D BOs.
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
            # bit-reinterpret (not cast) the bf16 buffer through int16 so
            # numpy can take it; ml_dtypes then views the bytes as bf16.
            return t.view(torch.int16).numpy().view(bfloat16)
        return t.numpy()

    # Sanity guard: torch.bfloat16 + numpy interop is fragile — the
    # `view(int16) -> numpy.view(bfloat16)` trick assumes the underlying
    # tensor is contiguous; non-contiguous bf16 tensors silently produce
    # garbage. safetensors always gives contiguous tensors so this is fine.

    # --- Embedding + final norm
    embed = _get("model.embed_tokens.weight")
    assert embed.shape == (config.vocab_size, config.emb_dim), embed.shape
    final_norm = _get("model.norm.weight")
    assert final_norm.shape == (config.emb_dim,)

    # --- Per-layer
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
            layer_kw[field] = awq_dequant_layer(qw, qz, sc, gs=gs)
            packed_kw[field] = awq_pack_for_npu(
                qw, qz, sc, gs=gs, n_tile=n_tile, k_chunk=k_chunk, M_seq=seq_len,
            )
        layers_bf16.append(LayerWeights(**layer_kw))
        layers_packed.append(packed_kw)

    # lm_head: this checkpoint has the lm_head quantized too, but with
    # `tie_word_embeddings: true` HF uses embed_tokens at inference time.
    # Match HF: tie our lm_head to embed_table.
    weights = LlamaWeights(
        embed_table=embed,
        layers=layers_bf16,
        final_norm=final_norm,
        lm_head=embed,
    )
    return weights, layers_packed


# ---------------------------------------------------------------------------
# Legacy fake-quant path (kept for the standalone GEMM example)
# ---------------------------------------------------------------------------


def fake_quantize_awq_int4(W_bf16, gs=128):
    """Asymmetric uint4 per-group quantization of a [M, K] bf16 weight.

    Returns (W_q [M, K/2] uint8 packed, W_s [n_groups, M] bf16,
    W_z [n_groups, M] uint8) matching the layout pack_inputs expects.
    """
    M, K = W_bf16.shape
    assert K % gs == 0, f"K={K} not divisible by gs={gs}"
    n_groups = K // gs
    W_f32 = W_bf16.astype(np.float32)

    W_grp = W_f32.reshape(M, n_groups, gs)
    w_min = W_grp.min(axis=2)
    w_max = W_grp.max(axis=2)
    scale = (w_max - w_min) / 15.0
    scale = np.where(scale == 0, 1e-7, scale)
    zero = np.round(-w_min / scale).clip(0, 15).astype(np.uint8)

    q = np.round(W_f32 / np.repeat(scale, gs, axis=1)
                 + np.repeat(zero.astype(np.float32), gs, axis=1)
                 ).clip(0, 15).astype(np.uint8)
    W_q = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
    W_s = scale.T.astype(bfloat16)
    W_z = zero.T.astype(np.uint8)
    return W_q, W_s, W_z


def pack_weight_for_int4_gemm(W_bf16, M_seq, gs=128, n_tile=16, k_chunk=128):
    """Quantize and pack a prefill-orientation [K, N] bf16 weight for the
    int4-AWQ GEMM kernel (matmul_int4_packed).

    Prefill stores weights as [K, N] (K outer; matches A @ W convention with
    A: [seq, K]). The int4 packer expects the weight in [N, K] output-major,
    so transpose first.

    Returns packed BO of shape [N // n_tile, K // k_chunk, tile_bytes] uint8.
    """
    W_NK = np.ascontiguousarray(W_bf16.T)
    N, K = W_NK.shape
    W_q, W_s, W_z = fake_quantize_awq_int4(W_NK, gs=gs)
    return pack_inputs_gemm(W_q, W_s, W_z, M_seq, K, N, gs, n_tile, k_chunk)
