# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Checkpoint semantics the shared B8 decoder reads. Only its drafter role
runs here; the TARGET_ names are its target role's (Qwen3-8B), which the
shared modules import but this example does not build."""

# the z-lab Qwen3-Coder-30B-A3B DFlash drafter
DRAFT_ID = "z-lab/Qwen3-Coder-30B-A3B-DFlash"
DRAFT_SHA256 = "906df3d198d88c1e740519da9237d669f38d7ff63bf3c03ca59334a42e9ac677"
DRAFT_LAYERS = 8
DRAFT_GEOMETRY = "qwen3-coder-dflash-drafter"
DRAFT_CXX = "QWEN3_CODER_DRAFT"
DRAFT_RMS_EPS = 1e-6
DRAFT_FFN = 6144
# (K, M, DK_TOT_A, DH, N_ATTN_CU, PAIR_ROWS) the drafter build must see
DRAFT_GEOMETRY_CHECK = (2048, 6144, 1024, 128, 4, 1)
# the target's layers the drafter reads, and its block and mask token
TAP_LAYERS = (1, 12, 23, 34, 45)
CHECKPOINT_BLOCK_SIZE = 16
MASK_TOKEN_ID = 151669
VOCAB_SIZE = 151936
EOS_IDS = (151643, 151645)
WEIGHT_GROUP_LAYERS = 9

# the shared decoder's target role
TARGET_ID = "Qwen/Qwen3-8B"
TARGET_LAYERS = 36
TARGET_GEOMETRY = "qwen3-8b"
TARGET_CXX = "QWEN3_8B"
TARGET_RMS_EPS = 1e-6
TARGET_FFN = 12288
VOCAB_PADDED = 155648
