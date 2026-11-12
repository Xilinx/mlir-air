// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Host side of Qwen3.6's fused prefill: Qwen3.5's Gated DeltaNet mixer at this
// model's head counts.
#define NK 16
#define NV 32
#include "../../qwen35_0_8b_q4nx/kernels/q35_host.c"
