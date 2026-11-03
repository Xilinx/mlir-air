# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Checkpoint facts the host code relies on."""

TARGET_LAYERS = 42
DRAFT_LAYERS = 5
# Hidden-state boundaries the drafter reads (after target layers 5, 11, 17,
# 23 and 35); boundary 0 is the embedding.
TAP_SLOTS = (6, 12, 18, 24, 36)
MASK_TOKEN_ID = 4
BLOCK_SIZE = 8
CAPACITY = 4096
# <eos> and <end_of_turn>
EOS_IDS = (1, 106)
CPU_PREFILL = False
CHAT_OPTIONS = {"enable_thinking": False}
