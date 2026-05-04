// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// rope_multi: apply rope to `nrows` consecutive bf16 rows of size `dim`.
// Wrapper around the upstream rope kernel from
// mlir-aie/aie_kernels/aie2p/rope.cc, called per row sharing one LUT.

#include <aie_api/aie.hpp>
#include <stdint.h>

template <typename T, int N>
static inline void rope_kernel(const T *restrict input, const T *restrict lut,
                               T *restrict output, int32_t dims) {
  for (int v = 0; v < dims; v += N) {
    ::aie::vector<T, N> x = ::aie::load_v<N>(input + v);
    ::aie::vector<T, N> cache = ::aie::load_v<N>(lut + v);

    ::aie::vector<T, N / 2> x_even = ::aie::filter_even(x, 1);
    ::aie::vector<T, N / 2> x_odd = ::aie::filter_odd(x, 1);
    ::aie::vector<T, N / 2> cos_val = ::aie::filter_even(cache, 1);
    ::aie::vector<T, N / 2> sin_val = ::aie::filter_odd(cache, 1);

    ::aie::vector<T, N / 2> even_cos = ::aie::mul(x_even, cos_val);
    ::aie::vector<T, N / 2> even_sin = ::aie::mul(x_even, sin_val);
    ::aie::vector<T, N / 2> odd_cos = ::aie::mul(x_odd, cos_val);
    ::aie::vector<T, N / 2> odd_sin = ::aie::mul(x_odd, sin_val);

    ::aie::vector<T, N / 2> output_even = ::aie::sub(even_cos, odd_sin);
    ::aie::vector<T, N / 2> output_odd = ::aie::add(even_sin, odd_cos);

    auto [low, high] = ::aie::interleave_zip(output_even, output_odd, 1);
    ::aie::vector<T, N> y = ::aie::concat(low, high);
    ::aie::store_v(output + v, y);
  }
}

extern "C" {
void rope_multi(bfloat16 *input, bfloat16 *lut, bfloat16 *output,
                int32_t nrows, int32_t dim) {
  event0();
  for (int h = 0; h < nrows; h++) {
    rope_kernel<bfloat16, 16>(input + h * dim, lut, output + h * dim, dim);
  }
  event1();
}

// Same body as rope_multi — distinct symbol so callers operating on
// L1 buffers of different MLIR shapes (e.g. [group_size, dk] vs [dk])
// can each declare an llvm.emit_c_interface external matching their
// own memref ABI.
void rope_multi_knew(bfloat16 *input, bfloat16 *lut, bfloat16 *output,
                     int32_t nrows, int32_t dim) {
  event0();
  for (int h = 0; h < nrows; h++) {
    rope_kernel<bfloat16, 16>(input + h * dim, lut, output + h * dim, dim);
  }
  event1();
}

// Patches the last row of an [nrows, dim] L1 buffer with `src`
// ([dim] L1). Used to splice K_new/V_new (cascade-delivered) into
// the last K/V chunk in the decode herd's reduction loop, in lieu
// of a subview+memcpy MLIR sequence that would re-introduce the
// per-row strided-memref type juggling rope_multi was created
// to avoid.
void patch_last_row(bfloat16 *buf, bfloat16 *src, int32_t nrows,
                    int32_t dim) {
  event0();
  bfloat16 *dst = buf + (nrows - 1) * dim;
  constexpr int N = 16;
  for (int v = 0; v < dim; v += N) {
    ::aie::vector<bfloat16, N> x = ::aie::load_v<N>(src + v);
    ::aie::store_v(dst + v, x);
  }
  event1();
}
}
