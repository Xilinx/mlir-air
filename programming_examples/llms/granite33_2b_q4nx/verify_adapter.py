# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""verify/ adapter: the production session on the fused prefill
(shared/affine_verify.py). Pointed at by --runner=granite33_2b_q4nx.verify_adapter."""

from air_examples.llms.shared.affine_verify import adapter

globals().update(adapter("granite33_2b_q4nx"))
