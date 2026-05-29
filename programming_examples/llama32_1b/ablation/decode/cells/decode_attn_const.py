"""Decode CPU attention invariant runner.

`decode_attention_cpu` runs on the CPU and is structurally identical across
all 4 cells (it's not subject to NPU dispatch optimizations). This module
wraps the production function from llama32_1b_decode.py so every cell calls
exactly the same Python.

Returns (attn_out, elapsed_seconds). The elapsed_seconds is reported separately
in the per-token results table as the "CPU attention floor" — analogous to how
Plan 1 reports FA's invariant per-layer cost.

Note: production `decode_attention_cpu` reads `k_cache[:, :current_pos+1, :]`
internally, so the caller MUST have written the new k/v at slot `current_pos`
before calling this function. The KV-cache write happens in the per-token loop
(cells/per_token_loop.py) right after rms_gemv_rope returns, before this call.
"""

import time

from llama32_1b_decode import decode_attention_cpu


def run_decode_attention(
    q_roped, k_cache_layer, v_cache_layer, current_pos, n_heads, n_kv_heads, head_dim
):
    """Invoke the production decode_attention_cpu and time it.

    Args:
        q_roped: (emb_dim,) bf16 — current token's RoPE'd query
        k_cache_layer: (n_kv_heads, max_seq, head_dim) bf16 — this layer's K cache
                       (must already have new k written at slot current_pos)
        v_cache_layer: same shape — this layer's V cache (with new v at current_pos)
        current_pos: int — the current token's slot index
        n_heads, n_kv_heads, head_dim: ints — model config

    Returns:
        attn_out: (emb_dim,) bf16
        elapsed: float — wall time of the CPU attention call (seconds)
    """
    t0 = time.perf_counter()
    attn_out = decode_attention_cpu(
        q_roped,
        k_cache_layer,
        v_cache_layer,
        current_pos,
        n_heads,
        n_kv_heads,
        head_dim,
    )
    elapsed = time.perf_counter() - t0
    return attn_out, elapsed
