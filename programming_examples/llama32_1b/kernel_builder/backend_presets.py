# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Backend kwarg presets for the kernels used by the LLAMA-3 example.

These dicts are passed to `cache.load_and_run(...)` (and equivalent helpers)
as the per-kernel `backend_kwargs`. Centralized here so callers don't
re-build identical dicts on every invocation, and so prefill / decode /
inference share the same canonical values.
"""

# ---------------------------------------------------------------------------
# Generic / shared
# ---------------------------------------------------------------------------

SIMPLE_BACKEND = {"omit_while_true_loop": False}

# ---------------------------------------------------------------------------
# Prefill (multi-launch ELFs)
# ---------------------------------------------------------------------------

RMS_GEMMS_ROPE_BACKEND = {
    "omit_while_true_loop": False,
    "output_format": "elf",
    "instance_name": "rms_gemms_rope",
}

O_FFN_BACKEND = {
    "omit_while_true_loop": False,
    "output_format": "elf",
    "instance_name": "o_ffn",
}

# ---------------------------------------------------------------------------
# Decode (GEMV multi-launch ELFs)
# ---------------------------------------------------------------------------
#
# K=2048 GEMV uses ping-pong (L1 fits both buffers).
GEMV_K2048_BACKEND = {
    "omit_while_true_loop": False,
    "omit_pingpong": "",
    "use_lock_race_condition_fix": False,
    # tile=1 from the default-tile-1 cost model makes the multi-launch
    # llama32_1b compile time out at 600s on CI. Confirmed across two CI
    # runs (commits 0d03a530 and b3b6b5af).
    "runtime_loop_tiling_sizes": [16, 16],
}

RGR_BACKEND = {
    "output_format": "elf",
    "instance_name": "rms_gemv_rope",
    **GEMV_K2048_BACKEND,
}

# OGF includes the K=8192 down-projection GEMV; ping-pong off because
# L1 is too tight to hold both buffers for that K.
OGF_BACKEND = {
    "output_format": "elf",
    "instance_name": "o_gemv_ffn",
    "omit_pingpong": "all",
    **{k: v for k, v in GEMV_K2048_BACKEND.items() if k != "omit_pingpong"},
}

LM_GEMV_BACKEND = {
    "output_format": "elf",
    "instance_name": "lm_head_gemv",
    **GEMV_K2048_BACKEND,
}
