# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SmolVLA Vision-Encoder (SigLIP ViT) host-side math.

The parts of the vision path that do NOT run on the NPU:

  - `im2col_patch_embed` — a one-time reshape before the layer loop, not
    hot-loop work.
  - `pixel_shuffle` — the connector's space-to-depth reshape, pure data
    movement with zero arithmetic, verified bit-exact against HF.
  - `mha_bidirectional` — the documented CPU fallback for the attention
    backend (the shipping path uses the registry FlashAttention ELF).

The CPU reference this example verifies against is the unmodified lerobot
model itself (see `smolvla_cpu_baseline.py`), not a reimplementation here.

Math mirrors HF `transformers/models/smolvlm/modeling_smolvlm.py`:
  - bidirectional MHA, 12 heads, head_dim 64, scale 1/8, softmax in fp32
  - patch embed = im2col + linear (stride==kernel==16, non-overlapping)
  - connector = pixel-shuffle (space-to-depth factor 4) + linear (no bias)
"""

import numpy as np


def _softmax(x, axis=-1):
    x = x.astype(np.float32)
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def mha_bidirectional(q, k, v, n_heads, head_dim, scale):
    """Standard bidirectional multi-head attention, no mask, no GQA.

    q, k, v: (seq, n_heads*head_dim). Returns (seq, n_heads*head_dim).
    Softmax in fp32. scale = head_dim^-0.5 applied to the QK^T scores.
    """
    seq = q.shape[0]
    q = q.astype(np.float32).reshape(seq, n_heads, head_dim)
    k = k.astype(np.float32).reshape(seq, n_heads, head_dim)
    v = v.astype(np.float32).reshape(seq, n_heads, head_dim)
    out = np.empty((seq, n_heads, head_dim), dtype=np.float32)
    for h in range(n_heads):
        scores = (q[:, h, :] @ k[:, h, :].T) * scale  # (seq, seq)
        probs = _softmax(scores, axis=-1)
        out[:, h, :] = probs @ v[:, h, :]
    return out.reshape(seq, n_heads * head_dim)


def im2col_patch_embed(pixel_values, patch_w, patch_b, pos_embed, patch_size=16):
    """Patch embedding as im2col + linear + position embedding.

    pixel_values: (3, H, W) with H=W=512. patch_w: (C*ph*pw, out)=(768,768).
    patch_b: (out,). pos_embed: (num_patches, out)=(1024,768).

    Non-overlapping patches (stride==kernel==patch_size). Patch pixels are
    extracted in (c, kh, kw) order to match the HF conv weight reshape
    (out, C, kh, kw) -> (out, C*kh*kw). Token index = ph*grid + pw (row-major).
    For a full 512x512 image the position_ids collapse to arange(1024), so the
    full pos_embed matrix is added directly.
    """
    C, H, W = pixel_values.shape
    grid = H // patch_size  # 32
    num_patches = grid * grid  # 1024
    x = pixel_values.astype(np.float32)
    cols = np.empty((num_patches, C * patch_size * patch_size), dtype=np.float32)
    for ph in range(grid):
        for pw in range(grid):
            patch = x[
                :,
                ph * patch_size : (ph + 1) * patch_size,
                pw * patch_size : (pw + 1) * patch_size,
            ]  # (C, ph, pw)
            cols[ph * grid + pw] = patch.reshape(-1)  # (c, kh, kw) order
    out = cols @ patch_w.astype(np.float32) + patch_b.astype(np.float32)
    return out + pos_embed.astype(np.float32)  # (1024, 768)


def pixel_shuffle(x, scale_factor=4):
    """Connector pixel-shuffle (space-to-depth), verified bit-exact vs HF.

    x: (num_patches, emb) = (1024, 768). Returns (64, 12288).
    1024 = 32x32 spatial; factor 4 -> 8x8 = 64 tokens, 768*16 = 12288 channels.
    """
    x = x.astype(np.float32)
    n, emb = x.shape
    grid = int(round(np.sqrt(n)))  # 32
    s = scale_factor
    h2 = grid // s  # 8
    x = x.reshape(grid, grid, emb)  # (h, w, c)
    x = x.reshape(h2, s, h2, s, emb)  # (h2, dh, w2, dw, c)
    x = x.transpose(0, 2, 1, 3, 4)  # (h2, w2, dh, dw, c)
    return x.reshape(h2 * h2, s * s * emb)  # (64, 12288)
