# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Bind the common BF16/BFP context prepass to this model's checkpoint."""

from paths import HERE
from draft_weights import DraftWeights
from common.prepass import BFPContextPrepass as _Prepass
from common.prepass import BF16ContextReference as _Reference, pack_bfp_weights


class BF16ContextReference(_Reference):
    def __init__(self, weights=None):
        super().__init__(weights or DraftWeights())


class BFPContextPrepass(_Prepass):
    def __init__(self, artifacts, reference=None, capacity=4096):
        super().__init__(artifacts, reference or BF16ContextReference(), capacity)
