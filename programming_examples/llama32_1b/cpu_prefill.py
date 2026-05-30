# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU prefill placeholder for the int4-AWQ pipeline.

Wraps `llama32_1b_reference.transformer_block` into a drop-in replacement
for `llama32_1b_inference.run_npu_prefill` so the int4-AWQ end-to-end path
can bootstrap a KV cache without needing int4 prefill ELFs yet.

Per-layer K (post-RoPE) and V are harvested from each `transformer_block`
call's intermediates dict and written into the same KV cache layout the
NPU decode loop reads from. The final norm + LM head runs on the last
prompt-position hidden state to produce the first generated token, matching
`run_npu_prefill`'s return contract.

Runtime: numpy bf16 dequant + matmul; ~2 s for a 16-token prompt at 16
layers, scales linearly. For validation and short interactive prompts only;
production int4 prefill will land later as a separate project and replace
the import in `inference.py`.
"""

import time

import numpy as np
from ml_dtypes import bfloat16


def run_cpu_prefill(
    token_ids,
    weights,
    config,
    rope_lut_bf16,
    max_seq,
    tokenizer=None,
    quiet=False,
):
    """CPU prefill that mirrors `run_npu_prefill`'s return signature.

    Args:
        token_ids: list[int] of prompt token IDs.
        weights: LlamaWeights with bf16 dequant fields populated (set up by
            load_weights_awq via dequant_to_bf16). Packed BO attributes are
            ignored here.
        config: LlamaConfig.
        rope_lut_bf16: (max_seq, head_dim) RoPE LUT in bf16; converted to
            f32 internally for the reference math.
        max_seq: KV cache stride along the sequence dim.
        tokenizer: optional, used only for logging.
        quiet: suppress timing prints.

    Returns:
        prefill_token: int   -- first predicted token ID (greedy argmax)
        k_cache: ndarray (n_layers, n_kv_heads, max_seq, head_dim) bfloat16
        v_cache: ndarray (n_layers, n_kv_heads, max_seq, head_dim) bfloat16
        prompt_len: int      -- len(token_ids)
    """
    from llama32_1b_reference import rms_norm as _rms_norm
    from llama32_1b_reference import transformer_block as _transformer_block

    seq_len = len(token_ids)
    n_layers = config.n_layers
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim

    if not quiet:
        print(f"Running CPU prefill ({n_layers} layers, seq_len={seq_len})...")
    t0 = time.time()

    rope_lut_f32 = np.asarray(rope_lut_bf16, dtype=np.float32)

    # Token embedding -> initial hidden states.
    embed = np.asarray(weights.embed_table, dtype=np.float32)
    x = embed[np.asarray(token_ids)]  # (seq_len, emb_dim)

    k_cache = np.zeros((n_layers, n_kv_heads, max_seq, head_dim), dtype=bfloat16)
    v_cache = np.zeros((n_layers, n_kv_heads, max_seq, head_dim), dtype=bfloat16)

    for layer_idx in range(n_layers):
        lw = weights.layers[layer_idx]
        x, inters = _transformer_block(x, lw, rope_lut_f32, config)
        # k_roped, v: (seq_len, n_kv_heads * head_dim) -> (n_kv_heads, seq_len, head_dim)
        k_roped = inters["k_roped"].reshape(seq_len, n_kv_heads, head_dim)
        v = inters["v"].reshape(seq_len, n_kv_heads, head_dim)
        k_cache[layer_idx, :, :seq_len, :] = k_roped.transpose(1, 0, 2).astype(bfloat16)
        v_cache[layer_idx, :, :seq_len, :] = v.transpose(1, 0, 2).astype(bfloat16)

    # Final norm + LM head on the LAST prompt position only.
    final_norm = np.asarray(weights.final_norm, dtype=np.float32)
    h_last = _rms_norm(x[-1:], final_norm)  # (1, emb_dim) f32
    lm_head = np.asarray(weights.lm_head, dtype=np.float32)
    logits_row = (h_last @ lm_head.T).reshape(-1)  # (vocab_size,)
    prefill_token = int(np.argmax(logits_row))

    t_prefill = time.time() - t0
    if not quiet:
        msg = f"CPU prefill done in {t_prefill:.2f}s. First token: {prefill_token}"
        if tokenizer is not None:
            try:
                msg += f" ({tokenizer.decode([prefill_token])!r})"
            except Exception:
                pass
        print(msg)

    return prefill_token, k_cache, v_cache, seq_len
