# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""verify/ adapter: the production session on the fused prefill
(shared/affine_verify.py). Pointed at by --runner=mistral7b_v03_q4nx.verify_adapter."""

from air_examples.llms.shared.affine_verify import adapter

globals().update(adapter("mistral7b_v03_q4nx"))
