// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// The alpha / beta core's kernels and the router GEMV it also runs (FLM's
// ab_proj core does both), as one object. Each source keeps its own helpers.

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "aie_kernels/aie_kernel_utils.h"

namespace q36_prep {
#include "gdn_prep.cc"
} // namespace q36_prep

namespace q36_router {
#include "router.cc"
} // namespace q36_router
