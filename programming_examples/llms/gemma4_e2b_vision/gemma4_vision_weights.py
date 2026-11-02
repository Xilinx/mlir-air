# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Weight loader, image preprocessing and shared math for the Gemma4 E2B and E4B
# vision tower, which the two models share; only the projection into the text
# model's hidden size differs.
#
# The weights come from the bf16 HF checkpoint, which also carries the
# ClippableLinear bounds, so the tower and its HF oracle read the same file.
# Preprocessing uses Gemma4ImageProcessorPil, which needs no torchvision.
import os
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

# ---------------------------------------------------------------- dims
# vision_config of Gemma4-E2B-IT and -E4B-IT; load_vision_config() asserts
# each one.
D = 768  # hidden_size, == 3 * patch_size**2
INTER = 3072  # intermediate_size
NUM_LAYERS = 16
N_HEADS = 12
DH = 64  # head_dim (MHA: num_key_value_heads == num_attention_heads)
PATCH = 16
POOL_K = 3  # pooling_kernel_size
POS_TABLE = 10240  # position_embedding_size
ROPE_THETA = 100.0
RMS_EPS = 1e-6
DEFAULT_OUTPUT_LEN = 280  # soft tokens per image

# Gemma4VisionAttention sets scaling = 1.0 rather than 1/sqrt(head_dim).
ATTN_SCALE = 1.0
# Gemma4VisionPooler multiplies by sqrt(hidden_size) after averaging.
POOL_SCALE = float(np.sqrt(D))

# GEMMA4_VARIANT selects the model, as for ../gemma4_e2b_q4nx.
VARIANT = os.environ.get("GEMMA4_VARIANT", "e2b")
_SOURCES = {"e2b": "unsloth/gemma-4-E2B-it", "e4b": "unsloth/gemma-4-E4B-it"}
if VARIANT not in _SOURCES:
    raise SystemExit(f"GEMMA4_VARIANT={VARIANT!r}: want one of {sorted(_SOURCES)}")
MODEL_DEFAULT = os.environ.get("G4_VISION_MODEL_SOURCE", _SOURCES[VARIANT])

_PREFIX = "model.vision_tower"
_EMBED_PROJ = "model.embed_vision.embedding_projection.weight"

# The seven ClippableLinear instances of a layer, by their checkpoint path.
_CLIPPED = {
    "q": f"{_PREFIX}.encoder.layers.%d.self_attn.q_proj",
    "k": f"{_PREFIX}.encoder.layers.%d.self_attn.k_proj",
    "v": f"{_PREFIX}.encoder.layers.%d.self_attn.v_proj",
    "o": f"{_PREFIX}.encoder.layers.%d.self_attn.o_proj",
    "gate": f"{_PREFIX}.encoder.layers.%d.mlp.gate_proj",
    "up": f"{_PREFIX}.encoder.layers.%d.mlp.up_proj",
    "down": f"{_PREFIX}.encoder.layers.%d.mlp.down_proj",
}

_NORMS = {
    "input_ln": f"{_PREFIX}.encoder.layers.%d.input_layernorm.weight",
    "post_attn_ln": f"{_PREFIX}.encoder.layers.%d.post_attention_layernorm.weight",
    "pre_ffn_ln": f"{_PREFIX}.encoder.layers.%d.pre_feedforward_layernorm.weight",
    "post_ffn_ln": f"{_PREFIX}.encoder.layers.%d.post_feedforward_layernorm.weight",
    "q_norm": f"{_PREFIX}.encoder.layers.%d.self_attn.q_norm.weight",
    "k_norm": f"{_PREFIX}.encoder.layers.%d.self_attn.k_norm.weight",
}


# ---------------------------------------------------------------- math helpers
def rmsnorm(x, w=None, eps=RMS_EPS):
    """Gemma4RMSNorm: normalize in f32, then scale by w directly.

    Not the Gemma2/3 (1 + w) convention: Gemma4RMSNorm.forward is
    `normed * self.weight`. `w=None` is Gemma4RMSNorm(with_scale=False), used
    by the vision v_norm and by the multimodal embedder's pre-projection norm.
    """
    xf = np.asarray(x, np.float32)
    ms = np.einsum("...i,...i->...", xf, xf) / xf.shape[-1] + eps
    out = xf * ms[..., None] ** -0.5
    if w is not None:
        out *= w
    return out


def gelu_tanh(x):
    """ACT2FN["gelu_pytorch_tanh"]."""
    xf = np.asarray(x, np.float32)
    return 0.5 * xf * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (xf + 0.044715 * xf**3)))


def rope_tables(position_ids, head_dim=DH, theta=ROPE_THETA):
    """cos/sin for Gemma4's 2-D vision RoPE, shaped [L, head_dim].

    Gemma4VisionRotaryEmbedding gives each of the two spatial dims half of
    head_dim and the SAME frequency range, computed over head_dim // 2. Each
    dim contributes cat(freqs, freqs), and the two are concatenated.
    """
    spatial = head_dim // 2
    inv = 1.0 / (theta ** (np.arange(0, spatial, 2, dtype=np.float32) / spatial))
    pos = np.asarray(position_ids, np.float32)  # [L, 2]
    cos, sin = [], []
    for dim in range(2):
        freqs = pos[:, dim][:, None] * inv[None, :]  # [L, spatial/2]
        emb = np.concatenate([freqs, freqs], axis=-1)  # [L, spatial]
        cos.append(np.cos(emb))
        sin.append(np.sin(emb))
    return (
        np.concatenate(cos, axis=-1).astype(np.float32),
        np.concatenate(sin, axis=-1).astype(np.float32),
    )


def apply_rope_2d(x, cos, sin):
    """apply_multidimensional_rope for ndim=2 over x of shape [L, H, DH].

    head_dim splits into two equal parts; part 0 is rotated by the x position
    and part 1 by the y position, each with the usual half-split rotate_half.
    """
    per = x.shape[-1] // 2
    out = []
    for p in range(2):
        sl = slice(p * per, (p + 1) * per)
        xp = np.asarray(x[..., sl], np.float32)
        c = cos[:, None, sl]
        s = sin[:, None, sl]
        h = per // 2
        rot = np.concatenate([-xp[..., h:], xp[..., :h]], axis=-1)
        out.append(xp * c + rot * s)
    return np.concatenate(out, axis=-1)


# ---------------------------------------------------------------- checkpoint
def _resolve(source):
    """A local dir, a local .safetensors, or a hub repo id -> safetensors path."""
    source = source or MODEL_DEFAULT
    p = Path(source)
    if p.is_file():
        return p
    if p.is_dir():
        hits = sorted(p.glob("*.safetensors"))
        if not hits:
            raise FileNotFoundError(f"no .safetensors under {p}")
        return hits[0]
    from huggingface_hub import snapshot_download

    d = Path(
        snapshot_download(source, allow_patterns=["*.safetensors", "*.json", "*.jinja"])
    )
    hits = sorted(d.glob("*.safetensors"))
    if not hits:
        raise FileNotFoundError(f"no .safetensors in {d}")
    return hits[0]


def checkpoint_identity(source=MODEL_DEFAULT):
    """A string that changes when the checkpoint's weights or config do: the
    resolved file (a hub snapshot path carries the revision), its size and
    mtime, and a hash of config.json. For keying cached reference outputs."""
    import hashlib

    path = _resolve(source)
    st = path.stat()
    cfg = hashlib.sha256((path.parent / "config.json").read_bytes()).hexdigest()
    return f"{path.resolve()}:{st.st_size}:{st.st_mtime_ns}:{cfg}"


def load_vision_config(source=MODEL_DEFAULT):
    """vision_config from the checkpoint, with the constants above asserted."""
    import json

    path = _resolve(source)
    cfg = json.loads((path.parent / "config.json").read_text())["vision_config"]
    for name, ours in (
        ("hidden_size", D),
        ("intermediate_size", INTER),
        ("num_hidden_layers", NUM_LAYERS),
        ("num_attention_heads", N_HEADS),
        ("head_dim", DH),
        ("patch_size", PATCH),
        ("pooling_kernel_size", POOL_K),
        ("position_embedding_size", POS_TABLE),
    ):
        assert (
            cfg[name] == ours
        ), f"vision_config.{name}={cfg[name]!r}, we assume {ours}"
    assert cfg["rope_parameters"]["rope_theta"] == ROPE_THETA
    assert cfg["hidden_activation"] == "gelu_pytorch_tanh"
    assert cfg["use_clipped_linears"] is True
    assert cfg["num_key_value_heads"] == N_HEADS, "vision attention is MHA"
    assert cfg["standardize"] is False, "std_bias/std_scale path not implemented"
    return cfg


class VisionWeights:
    """Every vision tensor, as numpy.

    `wT(layer, name)` gives a projection as the [in, out] bf16 a GEMM wants.
    Clamp bounds come back as (lo, hi) float pairs, +/-inf where the
    checkpoint leaves them unset. Derived tensors are memoised, since an
    encode asks for each of them once per image.
    """

    def __init__(self, source=MODEL_DEFAULT):
        from safetensors import safe_open

        import json

        self.path = _resolve(source)
        self.config = load_vision_config(self.path)
        # The text model's hidden size, which the soft tokens are projected to.
        full = json.loads((self.path.parent / "config.json").read_text())
        self.out_dim = full["text_config"]["hidden_size"]
        self._f = safe_open(str(self.path), "np")
        self._cache = {}
        self._derived = {}

    def _get(self, name):
        if name not in self._cache:
            self._cache[name] = self._f.get_tensor(name)
        return self._cache[name]

    def _memo(self, key, make):
        if key not in self._derived:
            self._derived[key] = make()
        return self._derived[key]

    def norm(self, layer, which):
        return np.asarray(self._get(_NORMS[which] % layer), np.float32)

    def w(self, layer, which):
        """Projection weight as stored, [out, in] bf16."""
        return self._get(_CLIPPED[which] % layer + ".linear.weight")

    def wT(self, layer, which):
        """Projection weight as [in, out] bf16."""
        return self._memo(
            ("wT", layer, which),
            lambda: np.ascontiguousarray(self.w(layer, which).T).astype(bfloat16),
        )

    def clamp(self, layer, which):
        """((in_lo, in_hi), (out_lo, out_hi)) for one ClippableLinear."""
        base = _CLIPPED[which] % layer
        keys = set(self._f.keys())

        def _s(suffix, default):
            name = f"{base}.{suffix}"
            if name not in keys:
                return default
            return float(np.asarray(self._get(name), np.float32))

        return (
            (_s("input_min", -np.inf), _s("input_max", np.inf)),
            (_s("output_min", -np.inf), _s("output_max", np.inf)),
        )

    def patch_proj(self):
        """Patch embedder projection as stored, [D, 3*PATCH*PATCH] bf16."""
        return self._get(f"{_PREFIX}.patch_embedder.input_proj.weight")

    def patch_proj_T(self):
        return self._memo(
            "patchT",
            lambda: np.ascontiguousarray(self.patch_proj().T).astype(bfloat16),
        )

    def position_table(self):
        """[2, POS_TABLE, D] float32, indexed directly."""
        return self._memo(
            "postab",
            lambda: np.asarray(
                self._get(f"{_PREFIX}.patch_embedder.position_embedding_table"),
                np.float32,
            ),
        )

    def embed_proj(self):
        """Multimodal embedder projection as stored, [out_dim, D] bf16."""
        return self._get(_EMBED_PROJ)

    def embed_proj_T(self):
        return self._memo(
            "embprojT",
            lambda: np.ascontiguousarray(self.embed_proj().T).astype(bfloat16),
        )


# ---------------------------------------------------------------- preprocessing
def preprocess(images, **kwargs):
    """PIL image(s) -> (pixel_values [L,768] f32, position_ids [L,2] int, n_soft).

    Padding patches carry position (-1, -1); their rows are zeroed after the
    patch embedding, as Gemma4VisionPatchEmbedder does.
    """
    from transformers.models.gemma4.image_processing_pil_gemma4 import (
        Gemma4ImageProcessorPil,
    )

    if not isinstance(images, (list, tuple)):
        images = [images]
    out = Gemma4ImageProcessorPil()(images=list(images), return_tensors="np", **kwargs)
    return (
        np.asarray(out["pixel_values"][0], np.float32),
        np.asarray(out["image_position_ids"][0], np.int64),
        int(np.asarray(out["num_soft_tokens_per_image"]).reshape(-1)[0]),
    )


def pool_indices(position_ids):
    """Gemma4VisionPooler._avg_pool_by_positions, as a scatter index.

    Returns (kernel_idx [L], n_slots, k_squared, keep [n_slots] bool).

    The slot count is L // k**2 over the PADDED patch count, as
    Gemma4VisionModel.forward derives it from pixel_values. Padding patches
    clamp to position (0,0) and land in slot 0, which is harmless because
    their hidden states are zeroed first and the divisor is the constant k**2,
    not the occupancy. Slots no patch reaches are dropped by `keep`, which is
    HF's pooler_mask.
    """
    L = position_ids.shape[0]
    n_slots = L // (POOL_K * POOL_K)
    if n_slots * POOL_K * POOL_K != L:
        raise ValueError(f"{L} patches is not a multiple of {POOL_K}**2")
    clamped = np.maximum(position_ids, 0)
    max_x = clamped[:, 0].max() + 1
    kern = clamped // POOL_K
    idx = (kern[:, 0] + (max_x // POOL_K) * kern[:, 1]).astype(np.int64)
    keep = np.zeros(n_slots, bool)
    keep[idx] = True
    return idx, n_slots, POOL_K * POOL_K, keep


def pool(hidden, position_ids):
    """Average-pool [L, D] to [n_soft, D], scale by sqrt(D), strip empty slots."""
    idx, n_slots, k_sq, keep = pool_indices(position_ids)
    out = np.zeros((n_slots, hidden.shape[-1]), np.float32)
    np.add.at(out, idx, np.asarray(hidden, np.float32))
    return (out / k_sq * POOL_SCALE)[keep]
