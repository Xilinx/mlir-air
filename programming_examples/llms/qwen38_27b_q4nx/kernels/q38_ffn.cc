// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// The RMS and GLU cores' kernels: Qwen3.6-35B-A3B's (moe_glu.cc, ffn.cc), and
// residual adds in place. residual_add also writes a copy of the sum, which
// would need a further [D] buffer, or a buffer the core has sent and would
// then write again within the same layer; residual_round adds a projection's
// rounds as they arrive, with no [D] buffer to assemble them in.

#include "moe_glu.cc"

extern "C" {

// x[512 j ..] += rnd: a dense phase's round j added in place, the rows its
// residual needs
void residual_round(bf16 *__restrict x, bf16 *__restrict rnd, int j) {
  x += NC * SL * j;
  for (int i = 0; i < NC * SL; i += V)
    aie::store_v(x + i,
                 aie::add(aie::load_v<V>(x + i), aie::load_v<V>(rnd + i)));
}

// xbuf += x
void residual_inplace(bf16 *__restrict xbuf, bf16 *__restrict x, int n) {
  for (int i = 0; i < n; i += V)
    aie::store_v(xbuf + i,
                 aie::add(aie::load_v<V>(xbuf + i), aie::load_v<V>(x + i)));
}

} // extern "C"
