"""LM head invariant runner — production-merged 8-partition GEMV in every cell.

The LM head (`lm_head_gemv.elf`) is structurally one merged ELF in production
and is held INVARIANT across the 4 cells of Plan 2 (rationale: see spec §4 —
mirrors Plan 1's treatment of FlashAttention). Reporting it as a separate
"fixed cost per token" line keeps the cells comparable on the parts that
DO change.

Three functions:
- compile_lm_head(cache, config): compiles the production lm_head_gemv ELF.
- preload_lm_head(cache, lm_weight_parts): one-time pre-upload of the 8
  partition weights into BOs (skipped on subsequent calls via static_input_indices).
- run_lm_head(cache, x_normed, vocab_size): invoke + concatenate partition
  outputs + argmax → returns (next_token_id, elapsed_seconds).
"""

import time

import numpy as np
from ml_dtypes import bfloat16

from kernel_builder.backend_presets import LM_GEMV_BACKEND

_LM_N_PART = 16384
_LM_N_PARTITIONS = 8


def compile_lm_head(cache, config):
    """Compile the production lm_head_gemv ELF (one-time)."""
    if "lm_head_gemv" in cache.artifacts:
        return
    from multi_launch_builder.lm_head_gemv_multi import build_lm_head_gemv_module

    mod = build_lm_head_gemv_module(
        emb_dim=config["emb_dim"],
        n_partitions=_LM_N_PARTITIONS,
        n_part=_LM_N_PART,
    )
    cache.compile_and_cache(
        "lm_head_gemv",
        mod,
        {**LM_GEMV_BACKEND, "verbose": getattr(cache, "verbose", False)},
    )
    cache._save_manifest()


def preload_lm_head(cache, lm_weight_parts, config):
    """One-time pre-upload of LM head partition weights.

    `lm_weight_parts`: list of 8 numpy arrays, each shape (_LM_N_PART, emb_dim).
    The first call materializes BOs and writes weights; subsequent run_lm_head
    calls skip weight upload via static_input_indices.
    """
    emb_dim = config["emb_dim"]
    inputs = [np.zeros(emb_dim, dtype=bfloat16)]
    for p in range(_LM_N_PARTITIONS):
        inputs.append(lm_weight_parts[p])
        inputs.append(np.zeros(_LM_N_PART, dtype=bfloat16))
    cache.load_and_run(
        "lm_head_gemv",
        LM_GEMV_BACKEND,
        *inputs,
        output_indices=[2 + 2 * p for p in range(_LM_N_PARTITIONS)],
        static_input_indices={1 + 2 * p for p in range(_LM_N_PARTITIONS)},
        intermediate_indices={2 + 2 * p for p in range(_LM_N_PARTITIONS)},
    )


def run_lm_head(cache, x_normed, vocab_size, config):
    """Run LM head; return (next_token_id, elapsed_seconds).

    `x_normed`: (emb_dim,) bf16 — the final RMSNorm output for the current token.
    `vocab_size`: int — usually 128256 for Llama-3.2-1B.

    Mirrors the production code in llama32_1b_inference.py:434-446.
    """
    emb_dim = config["emb_dim"]
    inputs = [x_normed.astype(bfloat16).flatten()]
    for p in range(_LM_N_PARTITIONS):
        # Placeholder weight — actual weight in BOs from preload + static_input_indices.
        inputs.append(np.zeros((_LM_N_PART, emb_dim), dtype=bfloat16))
        inputs.append(np.zeros(_LM_N_PART, dtype=bfloat16))

    t0 = time.perf_counter()
    results = cache.load_and_run(
        "lm_head_gemv",
        LM_GEMV_BACKEND,
        *inputs,
        output_indices=[2 + 2 * p for p in range(_LM_N_PARTITIONS)],
        static_input_indices={1 + 2 * p for p in range(_LM_N_PARTITIONS)},
        intermediate_indices={2 + 2 * p for p in range(_LM_N_PARTITIONS)},
    )
    elapsed = time.perf_counter() - t0

    logits = np.concatenate(
        [results[2 + 2 * p] for p in range(_LM_N_PARTITIONS)], axis=0
    )[:vocab_size]
    next_token = int(np.argmax(logits))
    return next_token, elapsed
