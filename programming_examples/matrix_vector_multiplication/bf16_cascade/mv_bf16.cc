//===- mv_bf16.cc - bf16 matvec micro-kernels for 2-tile-per-col design ---===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Per-tile micro-kernels used by matvec_2tile_add.py:
//   - matvec_vectorized_bf16(a, b, c): c[0..m] += a[m,k] @ b[k]
//   - zero_vectorized_bf16(c):         c[0..m] = 0
//   - partial_plus_r_bf16(p, r, off, d): d[0..m] = p[0..m] + r[off..off+m]
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#ifndef DIM_M
#define DIM_M 8
#endif
#ifndef DIM_K
#define DIM_K 512
#endif

// matvec_vectorized: accumulates a[m,k] @ b[k] into the existing partial
// c[0..m] (caller is responsible for zeroing on the first call across K).
template <unsigned m, unsigned k, unsigned r>
void matvec_vectorized_impl(const bfloat16 *__restrict a,
                            const bfloat16 *__restrict b,
                            bfloat16 *__restrict c) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  for (unsigned row = 0; row < m; row++) {
    aie::accum<accfloat, r> acc = aie::zeros<accfloat, r>();
    const bfloat16 *a_row = a + row * k;
    for (unsigned i = 0; i < k; i += r) {
      aie::vector<bfloat16, r> a_vec = aie::load_v<r>(a_row + i);
      aie::vector<bfloat16, r> b_vec = aie::load_v<r>(b + i);
      acc = aie::mac(acc, a_vec, b_vec);
    }
    float partial = aie::reduce_add(acc.template to_vector<float>());
    c[row] = static_cast<bfloat16>(static_cast<float>(c[row]) + partial);
  }
}

template <unsigned m>
void zero_impl(bfloat16 *__restrict c) {
  for (unsigned i = 0; i < m; i++)
    c[i] = static_cast<bfloat16>(0.0f);
}

// d[i] = partial[i] + r_full[offset + i]
template <unsigned m>
void partial_plus_r_impl(const bfloat16 *__restrict partial,
                         const bfloat16 *__restrict r_full, int offset,
                         bfloat16 *__restrict d) {
  for (unsigned i = 0; i < m; i++)
    d[i] = static_cast<bfloat16>(static_cast<float>(partial[i]) +
                                 static_cast<float>(r_full[offset + i]));
}

extern "C" {

void matvec_vectorized_bf16(bfloat16 *a, bfloat16 *b, bfloat16 *c) {
  matvec_vectorized_impl<DIM_M, DIM_K, 32>(a, b, c);
}

void zero_vectorized_bf16(bfloat16 *c) { zero_impl<DIM_M>(c); }

void partial_plus_r_bf16(bfloat16 *partial, bfloat16 *r_full, int offset,
                         bfloat16 *d) {
  partial_plus_r_impl<DIM_M>(partial, r_full, offset, d);
}

} // extern "C"
