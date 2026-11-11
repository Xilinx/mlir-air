// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// The RMS and GLU cores' kernels: Qwen3.6-35B-A3B's (moe_glu.cc, ffn.cc), and
// a residual add in place. residual_add also writes a copy of the sum, which
// would need a further [D] buffer, or a buffer the core has sent and would
// then write again within the same layer.

#include "moe_glu.cc"

extern "C" {

// xbuf += x
void residual_inplace(bf16 *__restrict xbuf, bf16 *__restrict x, int n) {
  for (int i = 0; i < n; i += V)
    aie::store_v(xbuf + i,
                 aie::add(aie::load_v<V>(xbuf + i), aie::load_v<V>(x + i)));
}

} // extern "C"
