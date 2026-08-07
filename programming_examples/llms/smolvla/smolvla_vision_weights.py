# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SmolVLA Vision-Encoder (SigLIP ViT) Weight Loader

Loads the SmolVLA (`lerobot/smolvla_base`) vision-tower weights from
HuggingFace safetensors and provides them as numpy arrays for MLIR-AIR kernel
invocations. Loads the SigLIP ViT weights
(12-layer bidirectional encoder) + the connector, not the language backbone.

The vision tower lives nested under the VLM submodule:

    model.vlm_with_expert.vlm.model.vision_model.{embeddings,encoder.layers.N,post_layernorm}
    model.vlm_with_expert.vlm.model.connector.modality_projection.proj.weight

Differences from the backbone loader (correctness-critical):
  - Every Linear has a BIAS (q/k/v/out proj + fc1/fc2). Loaded alongside each weight.
  - Norm is LayerNorm (affine gamma + beta), not RMSNorm. Both loaded.
  - No GQA (q=k=v all 768), no RoPE. Position info is an additive embedding.
  - patch_embedding is a Conv2d (768,3,16,16); since stride==kernel==16 (non-
    overlapping), it is reshaped to a (768,768) matrix for im2col+linear on host.
  - connector proj is (960, 12288), NO bias.

Weight convention: HF stores Linear as (out, in); our GEMM is y = x @ W, so we
transpose to (in, out). Biases are (out,) and need no transpose.
"""

import os
import glob as glob_module
from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
from ml_dtypes import bfloat16

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SigLIPVisionConfig:
    """SmolVLA vision-encoder (SigLIP ViT) hyperparameters.

    Standard bidirectional ViT: pre-norm LayerNorm, MHA (no GQA), GELU-tanh MLP.
    """

    n_layers: int = 12
    emb_dim: int = 768  # hidden_size
    n_heads: int = 12  # MHA, no GQA
    head_dim: int = 64  # 768 / 12
    hidden_dim: int = 3072  # intermediate_size
    patch_size: int = 16
    img_size: int = 512
    num_patches: int = 1024  # (512/16)^2
    layer_norm_eps: float = 1e-6  # SigLIP uses 1e-6 (backbone RMSNorm was 1e-5)
    attn_scale: float = 1.0 / 8.0  # head_dim^-0.5 = 64^-0.5 = 1/8
    connector_out: int = 960  # backbone emb_dim (connector projects into it)
    connector_in: int = 12288  # 768 * 16 (pixel-shuffle factor-4 space-to-depth)
    dtype: Any = bfloat16


# ---------------------------------------------------------------------------
# Per-layer weight container
# ---------------------------------------------------------------------------


@dataclass
class VisionLayerWeights:
    """Weights for a single SigLIP encoder layer.

    Linear weights are (in, out) for y = x @ W; biases are (out,).
    """

    ln1_w: np.ndarray  # (768,) LayerNorm1 gamma
    ln1_b: np.ndarray  # (768,) LayerNorm1 beta
    wq: np.ndarray  # (768, 768)
    bq: np.ndarray  # (768,)
    wk: np.ndarray  # (768, 768)
    bk: np.ndarray  # (768,)
    wv: np.ndarray  # (768, 768)
    bv: np.ndarray  # (768,)
    wo: np.ndarray  # (768, 768) out_proj
    bo: np.ndarray  # (768,)
    ln2_w: np.ndarray  # (768,) LayerNorm2 gamma
    ln2_b: np.ndarray  # (768,) LayerNorm2 beta
    w_fc1: np.ndarray  # (768, 3072)
    b_fc1: np.ndarray  # (3072,)
    w_fc2: np.ndarray  # (3072, 768)
    b_fc2: np.ndarray  # (768,)


# ---------------------------------------------------------------------------
# Full vision-tower weight container
# ---------------------------------------------------------------------------


@dataclass
class VisionWeights:
    """All weights for the SmolVLA vision tower + connector.

    Attributes:
        patch_w:    (768, 768)  conv reshaped to (out, C*kh*kw) then transposed to
                                (C*kh*kw, out) = (768, 768) for im2col @ W.
        patch_b:    (768,)      patch_embedding bias
        pos_embed:  (1024, 768) position_embedding.weight (added directly)
        layers:     list of n_layers (12) VisionLayerWeights
        post_ln_w:  (768,)      post_layernorm gamma
        post_ln_b:  (768,)      post_layernorm beta
        connector_w:(12288, 960) connector proj, transposed for y = x @ W (no bias)
    """

    patch_w: np.ndarray
    patch_b: np.ndarray
    pos_embed: np.ndarray
    layers: List[VisionLayerWeights] = field(default_factory=list)
    post_ln_w: np.ndarray = None
    post_ln_b: np.ndarray = None
    connector_w: np.ndarray = None


# ---------------------------------------------------------------------------
# HuggingFace name mapping
# ---------------------------------------------------------------------------

# Per-layer HF suffix -> (field, needs_transpose). Weights (Linear) transpose
# (out,in)->(in,out); biases and norm gammas/betas do not.
_HF_LAYER_MAP = {
    "layer_norm1.weight": ("ln1_w", False),
    "layer_norm1.bias": ("ln1_b", False),
    "self_attn.q_proj.weight": ("wq", True),
    "self_attn.q_proj.bias": ("bq", False),
    "self_attn.k_proj.weight": ("wk", True),
    "self_attn.k_proj.bias": ("bk", False),
    "self_attn.v_proj.weight": ("wv", True),
    "self_attn.v_proj.bias": ("bv", False),
    "self_attn.out_proj.weight": ("wo", True),
    "self_attn.out_proj.bias": ("bo", False),
    "layer_norm2.weight": ("ln2_w", False),
    "layer_norm2.bias": ("ln2_b", False),
    "mlp.fc1.weight": ("w_fc1", True),
    "mlp.fc1.bias": ("b_fc1", False),
    "mlp.fc2.weight": ("w_fc2", True),
    "mlp.fc2.bias": ("b_fc2", False),
}

_VISION_PREFIX = "model.vlm_with_expert.vlm.model.vision_model."
_CONNECTOR_KEY = (
    "model.vlm_with_expert.vlm.model.connector.modality_projection.proj.weight"
)


# ---------------------------------------------------------------------------
# Safetensors loading helpers (mirror the backbone loader)
# ---------------------------------------------------------------------------


def _resolve_safetensor_files(model_path: str) -> List[str]:
    if os.path.isdir(model_path):
        files = sorted(glob_module.glob(os.path.join(model_path, "*.safetensors")))
        if not files:
            raise FileNotFoundError(f"No .safetensors files found in {model_path}")
        return files

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        local_dir = snapshot_download(
            model_path,
            allow_patterns=["*.safetensors", "*.json"],
            local_files_only=True,
        )
        if not glob_module.glob(os.path.join(local_dir, "*.safetensors")):
            raise LocalEntryNotFoundError(
                f"local cache for {model_path} has no .safetensors"
            )
    except LocalEntryNotFoundError:
        local_dir = snapshot_download(
            model_path, allow_patterns=["*.safetensors", "*.json"]
        )
    files = sorted(glob_module.glob(os.path.join(local_dir, "*.safetensors")))
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


def _check_shape(name: str, arr: np.ndarray, expected: tuple) -> None:
    if arr.shape != expected:
        raise ValueError(f"{name} shape mismatch: expected {expected}, got {arr.shape}")


# ---------------------------------------------------------------------------
# Main loading function
# ---------------------------------------------------------------------------


def load_vision_weights(
    model_name_or_path: str,
    dtype=bfloat16,
    config: Optional[SigLIPVisionConfig] = None,
) -> VisionWeights:
    """Load the SmolVLA vision-tower weights from safetensors.

    Loads ONLY vision_model (patch embed, 12 encoder layers, post_layernorm) +
    the connector projection. The backbone, action-expert are ignored.
    """
    from safetensors import safe_open

    if config is None:
        config = SigLIPVisionConfig()

    files = _resolve_safetensor_files(model_name_or_path)
    key_to_file = {}
    for filepath in files:
        with safe_open(filepath, framework="numpy") as f:
            for key in f.keys():
                key_to_file[key] = filepath

    def _get(key, dtype_=dtype):
        if key not in key_to_file:
            raise KeyError(f"Missing weight: {key}")
        with safe_open(key_to_file[key], framework="numpy") as f:
            return _load_tensor(f, key, dtype_)

    # --- patch embedding: Conv2d (768,3,16,16) -> (in=3*16*16, out=768) ---
    # HF conv weight is (out, C, kh, kw). Flatten to (out, C*kh*kw) with C-major
    # order (channel outermost, then kh, then kw), then transpose to (in, out)
    # for the im2col @ W convention. The host im2col MUST extract patches in the
    # matching (c, kh, kw) order.
    conv_w = _get(_VISION_PREFIX + "embeddings.patch_embedding.weight")  # (768,3,16,16)
    _check_shape(
        "patch conv", conv_w, (config.emb_dim, 3, config.patch_size, config.patch_size)
    )
    patch_w = np.ascontiguousarray(
        conv_w.reshape(config.emb_dim, -1).T
    )  # (3*16*16, 768) = (768, 768)
    patch_b = _get(_VISION_PREFIX + "embeddings.patch_embedding.bias")  # (768,)
    _check_shape("patch_b", patch_b, (config.emb_dim,))

    pos_embed = _get(
        _VISION_PREFIX + "embeddings.position_embedding.weight"
    )  # (1024,768)
    _check_shape("pos_embed", pos_embed, (config.num_patches, config.emb_dim))

    # --- per-layer weights ---
    layers = []
    for i in range(config.n_layers):
        t = {}
        for suffix, (field_name, needs_t) in _HF_LAYER_MAP.items():
            key = f"{_VISION_PREFIX}encoder.layers.{i}.{suffix}"
            arr = _get(key)
            if needs_t:
                arr = np.ascontiguousarray(arr.T)
            t[field_name] = arr
        layer = VisionLayerWeights(**t)
        lp = f"Vision layer {i}"
        _check_shape(f"{lp} wq", layer.wq, (config.emb_dim, config.emb_dim))
        _check_shape(f"{lp} wk", layer.wk, (config.emb_dim, config.emb_dim))
        _check_shape(f"{lp} wv", layer.wv, (config.emb_dim, config.emb_dim))
        _check_shape(f"{lp} wo", layer.wo, (config.emb_dim, config.emb_dim))
        _check_shape(f"{lp} w_fc1", layer.w_fc1, (config.emb_dim, config.hidden_dim))
        _check_shape(f"{lp} w_fc2", layer.w_fc2, (config.hidden_dim, config.emb_dim))
        _check_shape(f"{lp} b_fc1", layer.b_fc1, (config.hidden_dim,))
        _check_shape(f"{lp} bq", layer.bq, (config.emb_dim,))
        layers.append(layer)

    post_ln_w = _get(_VISION_PREFIX + "post_layernorm.weight")
    post_ln_b = _get(_VISION_PREFIX + "post_layernorm.bias")
    _check_shape("post_ln_w", post_ln_w, (config.emb_dim,))

    # --- connector proj: (960, 12288) -> transpose to (12288, 960), no bias ---
    conn = _get(_CONNECTOR_KEY)  # (960, 12288)
    _check_shape("connector", conn, (config.connector_out, config.connector_in))
    connector_w = np.ascontiguousarray(conn.T)  # (12288, 960)

    return VisionWeights(
        patch_w=patch_w,
        patch_b=patch_b,
        pos_embed=pos_embed,
        layers=layers,
        post_ln_w=post_ln_w,
        post_ln_b=post_ln_b,
        connector_w=connector_w,
    )


# ---------------------------------------------------------------------------
# Main -- test loading and print shapes
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_PATH = "lerobot/smolvla_base"


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Load SmolVLA vision weights, print shapes"
    )
    parser.add_argument("model_path", nargs="?", default=_DEFAULT_MODEL_PATH)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    args = parser.parse_args()

    dtype = bfloat16 if args.dtype == "bfloat16" else np.float32
    config = SigLIPVisionConfig()
    print(f"Loading vision weights from: {args.model_path}  (dtype={args.dtype})")
    print(f"Config: {config}\n")

    w = load_vision_weights(args.model_path, dtype=dtype, config=config)
    print("=== Global ===")
    print(f"  patch_w    : {w.patch_w.shape}")
    print(f"  patch_b    : {w.patch_b.shape}")
    print(f"  pos_embed  : {w.pos_embed.shape}")
    print(f"  post_ln_w  : {w.post_ln_w.shape}")
    print(f"  connector_w: {w.connector_w.shape}")
    print(f"\n=== Per-layer ({config.n_layers}) ===")
    l0 = w.layers[0]
    print(f"  wq {l0.wq.shape} bq {l0.bq.shape} | wo {l0.wo.shape}")
    print(f"  w_fc1 {l0.w_fc1.shape} b_fc1 {l0.b_fc1.shape} | w_fc2 {l0.w_fc2.shape}")
    print(f"  ln1_w {l0.ln1_w.shape} ln1_b {l0.ln1_b.shape}")

    errors = []
    if len(w.layers) != config.n_layers:
        errors.append(f"expected {config.n_layers} layers, got {len(w.layers)}")
    if l0.wq.shape != (config.emb_dim, config.emb_dim):
        errors.append(f"wq {l0.wq.shape}")
    if w.connector_w.shape != (config.connector_in, config.connector_out):
        errors.append(f"connector_w {w.connector_w.shape}")
    if errors:
        print("\nSMOKE TEST FAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("\nSmoke test OK: all vision weights loaded.")
