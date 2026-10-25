# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Llama target and its Qwen-style DFlash drafter have different semantics."""

TARGET_LAYERS = 32
DRAFT_LAYERS = 5
TARGET_GEOMETRY = "llama-3.1-8b-dflash-target"
DRAFT_GEOMETRY = "llama-3.1-8b-dflash-drafter"
TARGET_CXX = "LLAMA_3_1_8B"
DRAFT_CXX = "QWEN3_8B"
WEIGHT_GROUP_LAYERS = 8
TARGET_QK_NORM = False
TARGET_RMS_EPS = 1e-5
DRAFT_RMS_EPS = 1e-6
TARGET_FFN = 14336
DRAFT_FFN = 12288
VOCAB_SIZE = 128256
VOCAB_PADDED = 131072
TARGET_ID = "meta-llama/Llama-3.1-8B-Instruct"
DRAFT_ID = "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"
TARGET_ROPE = "llama3 factor8 theta500000"
TAP_LAYERS = (1, 8, 15, 22, 29)
CHECKPOINT_BLOCK_SIZE = 10
MASK_TOKEN_ID = 128002
EOS_IDS = (128001, 128008, 128009)
PREFILL_SCHEMA = "llama31-8b-prefill-mapped-bf16-v1"
PREFILL_VOCAB_PADDED = 131072
CPU_PREFILL = False
CHAT_OPTIONS = {}
DRAFT_SHA256 = "738b12235328a8f3323b69e7d56bb7847c70908481fb86a209d12ca9d6cc1f3c"
