"""KV cache state management for the per-token timed loop.

Two functions:
- build_initial_kv_cache(config, prompt_len, seed):
    Deterministic synthetic pre-fill of `prompt_len` positions for ALL layers.
    Returns dict {k_cache, v_cache, current_pos}. The cache shape is
    (n_layers, n_kv_heads, max_seq, head_dim) bf16.

- reset_position(cache, pos):
    Zero out the K/V cache slots at position `pos` for ALL layers.
    Used between trials to ensure each trial starts from the SAME state
    (the pre-filled prompt without the previously-generated token's k/v).
"""

import numpy as np
from ml_dtypes import bfloat16


def build_initial_kv_cache(config, prompt_len, seed):
    """Deterministic synthetic pre-fill of `prompt_len` cache positions.

    config keys required: n_layers, n_kv_heads, head_dim, max_seq

    Returns dict with:
        k_cache: (n_layers, n_kv_heads, max_seq, head_dim) bf16
        v_cache: same shape
        current_pos: int = prompt_len  (next slot to write)
    """
    rng = np.random.default_rng(seed)
    shape = (
        config["n_layers"],
        config["n_kv_heads"],
        config["max_seq"],
        config["head_dim"],
    )
    k = np.zeros(shape, dtype=bfloat16)
    v = np.zeros(shape, dtype=bfloat16)
    pre_shape = (
        config["n_layers"],
        config["n_kv_heads"],
        prompt_len,
        config["head_dim"],
    )
    k[:, :, :prompt_len, :] = (rng.standard_normal(pre_shape) * 0.5).astype(bfloat16)
    v[:, :, :prompt_len, :] = (rng.standard_normal(pre_shape) * 0.5).astype(bfloat16)
    return {"k_cache": k, "v_cache": v, "current_pos": prompt_len}


def reset_position(cache, pos):
    """Zero out the K/V cache slots at `pos` for ALL layers.

    Called between timing trials so each trial sees the same initial state
    (the pre-filled prompt's positions [0:prompt_len] but no new-token entry
    at `pos = prompt_len`).
    """
    cache["k_cache"][:, :, pos, :] = 0
    cache["v_cache"][:, :, pos, :] = 0
