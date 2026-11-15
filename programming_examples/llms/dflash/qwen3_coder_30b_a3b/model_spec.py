# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Checkpoint semantics, separate from the shared physical B8 mapping. The
TARGET_ names are the shared decoder's target role (Qwen3-8B); only its
drafter role runs here."""

TARGET_LAYERS = 36
DRAFT_LAYERS = 8
TARGET_GEOMETRY = "qwen3-8b"
DRAFT_GEOMETRY = "qwen3-coder-dflash-drafter"
TARGET_CXX = "QWEN3_8B"
DRAFT_CXX = "QWEN3_CODER_DRAFT"
WEIGHT_GROUP_LAYERS = 9
TARGET_QK_NORM = True
TARGET_RMS_EPS = 1e-6
DRAFT_RMS_EPS = 1e-6
TARGET_FFN = 12288
DRAFT_FFN = 6144
VOCAB_SIZE = 151936
VOCAB_PADDED = 155648
TARGET_ID = "Qwen/Qwen3-8B"
DRAFT_ID = "z-lab/Qwen3-Coder-30B-A3B-DFlash"
TARGET_ROPE = "default theta1000000"
TAP_LAYERS = (1, 9, 17, 25, 33)
CHECKPOINT_BLOCK_SIZE = 16
MASK_TOKEN_ID = 151669
EOS_IDS = (151643, 151645)
DRAFT_SHA256 = "c702878094f38ad5843e6ac40b327720b3147ffbf669a5d9ca1974865ca6c080"
PREFILL_SCHEMA = "qwen3-8b-prefill-mapped-bf16-v1"
PREFILL_VOCAB_PADDED = 163840
CPU_PREFILL = True
CHAT_OPTIONS = {"enable_thinking": False}
# (K, M, DK_TOT_A, DH, N_ATTN_CU, PAIR_ROWS) the drafter build must see
DRAFT_GEOMETRY_CHECK = (2048, 6144, 1024, 128, 4, 1)
