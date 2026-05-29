"""Per-token decode loop wrapper — the end-to-end timed unit for Plan 2.

Generates ONE decode token at a fixed `current_pos` from a pre-filled KV cache.
The cell-specific dispatch is injected via `run_rms_gemv_rope` and
`run_o_gemv_ffn` function arguments so the same wrapper works for all 4 cells.

For each of the 16 layers:
  1. NPU rms_gemv_rope (cell-specific)  → q_roped, k_roped, v
  2. Write k_roped, v into KV cache at current_pos
  3. CPU decode_attention_cpu (invariant) → attn_out
  4. NPU o_gemv_ffn (cell-specific)      → next-layer activation

After 16 layers:
  5. CPU final RMSNorm on the running hidden state (single row)
  6. NPU lm_head_gemv (invariant)       → logits → argmax → next_token

The `layer_inputs_per_layer` list contains per-layer weight bundles
(rms_gemv_rope's: norm_w, wq, wk, wv, lut_q, lut_k; o_gemv_ffn's: wo,
ffn_norm_w, w_gate, w_up, w_down). The cell-specific runners are
responsible for assembling these into the kernel-group's expected
argument order.

Returns a dict with per-stage wall times for downstream attribution.
"""

import time

import numpy as np
from ml_dtypes import bfloat16

from cells.decode_attn_const import run_decode_attention
from cells.lm_head_const import run_lm_head


def _final_rms_norm_cpu(x_bf16, weight_bf16, eps=1e-5):
    """Single-row RMSNorm on the final hidden state (mirrors production).

    x: (emb_dim,) bf16; weight: (emb_dim,) bf16. Returns (emb_dim,) bf16.
    """
    x_f32 = x_bf16.astype(np.float32)
    w_f32 = weight_bf16.astype(np.float32)
    rms = np.sqrt(np.mean(x_f32 * x_f32) + eps)
    return ((x_f32 / rms) * w_f32).astype(bfloat16)


def run_one_decode_token(
    cache,
    config,
    kv_cache,
    layer_inputs_per_layer,
    final_norm_w,
    lm_weight_parts,
    initial_x_decode,
    current_pos,
    run_rms_gemv_rope,
    run_o_gemv_ffn,
):
    """Generate ONE decode token end-to-end. THIS IS THE TIMED UNIT.

    Args:
        cache: shared KernelCache with all ELFs compiled + preloaded
        config: dict with emb_dim, n_heads, n_kv_heads, head_dim, n_layers, vocab_size
        kv_cache: dict from build_initial_kv_cache (mutated in-place)
        layer_inputs_per_layer: list of N dicts, one per layer, with weight tensors
        final_norm_w: (emb_dim,) bf16 — final RMSNorm weight
        lm_weight_parts: list of 8 (16384, emb_dim) arrays — LM head partitions
        initial_x_decode: (emb_dim,) bf16 — the token's embedding
        current_pos: int — the slot in KV cache to write the new k/v
        run_rms_gemv_rope: callable(cache, layer_inputs, layer_idx) -> dict with
                          q_roped, k_roped, v, _wall_s
        run_o_gemv_ffn:    callable(cache, layer_inputs, layer_idx) -> dict with
                          output, _wall_s

    Returns dict with:
        next_token: int
        per_layer_npu_wall_s: list of N floats (rms_gemv_rope + o_gemv_ffn per layer)
        per_layer_rms_gemv_rope_wall_s: list of N floats
        per_layer_o_gemv_ffn_wall_s: list of N floats
        cpu_attn_wall_s: float (sum across N layers)
        lm_head_wall_s: float
        total_wall_s: float (everything inside the timer)
    """
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    n_kv_heads = config["n_kv_heads"]
    head_dim = config["head_dim"]
    vocab_size = config["vocab_size"]

    per_layer_rg = []
    per_layer_of = []
    cpu_attn_total = 0.0
    x = initial_x_decode

    t_total_start = time.perf_counter()
    for L in range(n_layers):
        layer_in = dict(layer_inputs_per_layer[L])
        layer_in["x_in"] = x
        layer_in["current_pos"] = current_pos

        # 1. rms_gemv_rope (NPU, cell-specific)
        rg_out = run_rms_gemv_rope(cache, layer_in, layer_idx=L)
        per_layer_rg.append(rg_out["_wall_s"])

        q_roped = rg_out["q_roped"].astype(bfloat16)
        k_roped = rg_out["k_roped"].astype(bfloat16)
        v = rg_out["v"].astype(bfloat16)

        # 2. KV cache write (CPU)
        kv_cache["k_cache"][L, :, current_pos, :] = k_roped.reshape(
            n_kv_heads, head_dim
        )
        kv_cache["v_cache"][L, :, current_pos, :] = v.reshape(n_kv_heads, head_dim)

        # 3. CPU decode attention (invariant)
        attn_out, attn_t = run_decode_attention(
            q_roped.flatten(),
            kv_cache["k_cache"][L],
            kv_cache["v_cache"][L],
            current_pos,
            n_heads,
            n_kv_heads,
            head_dim,
        )
        cpu_attn_total += attn_t

        # 4. o_gemv_ffn (NPU, cell-specific)
        of_in = dict(layer_in)
        of_in["attn_out"] = attn_out.astype(bfloat16)
        of_in["x_residual"] = x  # the activation entering THIS layer
        of_out = run_o_gemv_ffn(cache, of_in, layer_idx=L)
        per_layer_of.append(of_out["_wall_s"])

        x = of_out["output"].astype(bfloat16).flatten()

    # 5. Final RMSNorm (CPU, single row)
    x_normed = _final_rms_norm_cpu(x, final_norm_w)

    # 6. LM head (NPU, invariant) + argmax
    next_token, lm_t = run_lm_head(cache, x_normed, vocab_size, config)

    total_wall = time.perf_counter() - t_total_start
    return {
        "next_token": next_token,
        "per_layer_npu_wall_s": [a + b for a, b in zip(per_layer_rg, per_layer_of)],
        "per_layer_rms_gemv_rope_wall_s": per_layer_rg,
        "per_layer_o_gemv_ffn_wall_s": per_layer_of,
        "cpu_attn_wall_s": cpu_attn_total,
        "lm_head_wall_s": lm_t,
        "total_wall_s": total_wall,
    }
