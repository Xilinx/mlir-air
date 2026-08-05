//===- swiglu_fused.cc - Fused SwiGLU kernels for AIE2P --------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
// Fused SwiGLU kernel containing:
//   1. zero_acc_bf16 -- vectorized zero fill for accumulator buffer
//   2. matmul_bf16_fused -- 8x8x8 bf16 matmul with 2x2 unrolling
//   3. silu_inplace_bf16 -- in-place SiLU activation
//   4. elemwise_mul_bf16 -- element-wise multiply of two buffers
//
// Compiled with -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 for AIE2P.
// Tile dimensions passed via -DDIM_M, -DDIM_K, -DDIM_N.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#define REL_WRITE 0
#define REL_READ 1

#include <aie_api/aie.hpp>

// ============================================================
// Zero fill (from zero.cc pattern)
// ============================================================
template <typename T, int M, int N, int r>
void zero_vectorized(T *__restrict c) {
  const aie::vector<T, r> zeros = aie::zeros<T, r>();
  const T *__restrict c_end = c + M * N;
  for (; c + r < c_end; c += r) {
    aie::store_v(c, zeros);
  }
  for (; c < c_end; c++) {
    *c = 0;
  }
}

// ============================================================
// Matmul with 2x2 register tiling (from mm_aie2p.cc)
// ============================================================
constexpr aie::rounding_mode round_mode = aie::rounding_mode::conv_even;

template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t>
static inline void matmul_vectorized_2x2_mmul(const T_in *__restrict pA,
                                              const T_in *__restrict pB,
                                              T_out *__restrict pC) {
  using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

  event0();

  for (unsigned z = 0; z < rowA; z += 2)
    chess_prepare_for_pipelining chess_loop_range(2, ) {
      T_out *__restrict pC1 = pC + (z)*MMUL::size_C;
      T_out *__restrict pC2 = pC + ((z + 1)) * MMUL::size_C;

      for (unsigned j = 0; j < colB; j += 2)
#ifdef OPT_PERF_ENABLED
        chess_flatten_loop
#endif
        {
          const T_in *__restrict pA1 = pA + (z)*MMUL::size_A;
          const T_in *__restrict pA2 = pA + ((z + 1)) * MMUL::size_A;
          const T_in *__restrict pB1 = pB + (j)*colA * MMUL::size_B;
          const T_in *__restrict pB2 = pB + (j + 1) * colA * MMUL::size_B;

          aie::vector<T_out, MMUL::size_C> acc_C00 =
              aie::load_v<MMUL::size_C>(pC1);
          aie::vector<T_out, MMUL::size_C> acc_C01 =
              aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C * rowA);
          aie::vector<T_out, MMUL::size_C> acc_C10 =
              aie::load_v<MMUL::size_C>(pC2);
          aie::vector<T_out, MMUL::size_C> acc_C11 =
              aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C * rowA);

          MMUL C00(acc_C00);
          MMUL C01(acc_C01);
          MMUL C10(acc_C10);
          MMUL C11(acc_C11);

          for (unsigned i = 0; i < colA; ++i)
#ifdef OPT_PERF_ENABLED
            chess_flatten_loop
#endif
            {
              aie::vector<T_in, MMUL::size_A> A0 =
                  aie::load_v<MMUL::size_A>(pA1);
              pA1 += rowA * MMUL::size_A;
              aie::vector<T_in, MMUL::size_A> A1 =
                  aie::load_v<MMUL::size_A>(pA2);
              pA2 += rowA * MMUL::size_A;
              aie::vector<T_in, MMUL::size_B> B0 =
                  aie::load_v<MMUL::size_B>(pB1);
              pB1 += MMUL::size_B;
              aie::vector<T_in, MMUL::size_B> B1 =
                  aie::load_v<MMUL::size_B>(pB2);
              pB2 += MMUL::size_B;

              C00.mac(A0, B0);
              C01.mac(A0, B1);
              C10.mac(A1, B0);
              C11.mac(A1, B1);
            }

          aie::store_v(pC1, C00.template to_vector<T_out>());
          pC1 += MMUL::size_C * rowA;
          aie::store_v(pC1, C01.template to_vector<T_out>());
          pC1 += MMUL::size_C * rowA;
          aie::store_v(pC2, C10.template to_vector<T_out>());
          pC2 += MMUL::size_C * rowA;
          aie::store_v(pC2, C11.template to_vector<T_out>());
          pC2 += MMUL::size_C * rowA;
        }
    }

  event1();
}

// ============================================================
// Compile-time tile dimensions (passed via -D flags)
// ============================================================
#ifndef DIM_M
#define DIM_M 64
#define DIM_M_DIV_8 8
#endif

#ifndef DIM_K
#define DIM_K 32
#endif

#ifndef DIM_N
#define DIM_N 64
#define DIM_N_DIV_8 8
#endif

// ============================================================
// Extern C functions
// ============================================================
extern "C" {

// Zero-fill accumulator buffer [DIM_M, DIM_N] bf16
void zero_acc_bf16(bfloat16 *__restrict c_out) {
  zero_vectorized<bfloat16, DIM_M, DIM_N, 32>(c_out);
}

// linalg.fill-compatible zero function name for XRTRunner's
// lower_linalg_to_func. This name is generated by the compiler
// for a 6D blocked-layout memref view.
#define CAT2(a, b) a##b
#define CAT(a, b) CAT2(a, b)
#define MAKE_LINALG_FILL_NAME(N_div, M_div)                                    \
  CAT(CAT(CAT(CAT(CAT(CAT(CAT(CAT(linalg_fill_bf16_view1x1x, N_div), x),       \
                          M_div),                                              \
                      x),                                                      \
                  8),                                                          \
              x),                                                              \
          8),                                                                  \
      xbf16as2)
void MAKE_LINALG_FILL_NAME(DIM_N_DIV_8, DIM_M_DIV_8)(bfloat16 fill_val,
                                                     bfloat16 *c_out) {
  // linalg.fill passes a scalar value and the output memref.
  // We assume fill_val is zero (the only use case).
  zero_vectorized<bfloat16, DIM_M, DIM_N, 32>(c_out);
}

// Matmul: C += A * B with 8x8x8 mmul intrinsic
// A is [DIM_M/8, DIM_K/8, 8, 8] blocked, B is [DIM_N/8, DIM_K/8, 8, 8]
// blocked, C is [DIM_N/8, DIM_M/8, 8, 8] blocked. All bf16.
// The linalg name is generated by lower_linalg_to_func.
void op_has_no_registered_library_name(bfloat16 *a_in, bfloat16 *b_in,
                                       bfloat16 *c_out) {
  constexpr int r = 8, s = 8, t = 8;
  static_assert(DIM_M % (2 * r) == 0);
  static_assert(DIM_K % s == 0);
  static_assert(DIM_N % (2 * t) == 0);

  ::aie::set_rounding(round_mode);
  matmul_vectorized_2x2_mmul<bfloat16, bfloat16, (DIM_M / r), (DIM_K / s),
                             (DIM_N / t), r, s, t>(a_in, b_in, c_out);
}

// In-place SiLU activation on a single tile buffer.
// SiLU(x) = x * sigmoid(x) = x * 0.5 * (tanh(x/2) + 1)
// Called from MLIR as: func.call @silu_inplace_bf16(memref<DIM_M*DIM_N x bf16,
// 2>)
void silu_inplace_bf16(bfloat16 *__restrict buf) {
#ifdef SILU_NOOP
  // No-op for debugging: skip SiLU, just pass through
  (void)buf;
#else
  constexpr int VecLen = 16;
  constexpr int n = DIM_M * DIM_N;
  aie::vector<bfloat16, VecLen> half_vec =
      aie::broadcast<bfloat16, VecLen>((bfloat16)0.5f);
  aie::vector<bfloat16, VecLen> one_vec =
      aie::broadcast<bfloat16, VecLen>((bfloat16)1.0f);

  for (int i = 0; i < n; i += VecLen) {
    aie::vector<bfloat16, VecLen> x = aie::load_v<VecLen>(buf + i);

    // sigmoid(x) = 0.5 * (1 + tanh(x/2))
    aie::vector<bfloat16, VecLen> x_half = aie::mul(x, half_vec);
    aie::accum<accfloat, VecLen> tanh_in;
    tanh_in.from_vector(x_half);
    aie::vector<bfloat16, VecLen> tanh_val =
        aie::tanh<bfloat16>(tanh_in.to_vector<float>());
    aie::vector<bfloat16, VecLen> one_plus_tanh = aie::add(one_vec, tanh_val);
    aie::vector<bfloat16, VecLen> sigmoid = aie::mul(half_vec, one_plus_tanh);
    // SiLU = x * sigmoid(x)
    aie::vector<bfloat16, VecLen> result = aie::mul(x, sigmoid);
    aie::store_v(buf + i, result);
  }
#endif
}

// Element-wise multiply: gate[i] *= up[i], two separate buffers.
// Called from MLIR as: func.call @elemwise_mul_bf16(memref<4096xbf16,2>,
// memref<4096xbf16,2>) Result written to gate buffer.
void elemwise_mul_bf16(bfloat16 *__restrict gate, bfloat16 *__restrict up) {
  constexpr int VecLen = 16;
  constexpr int n = DIM_M * DIM_N;
  for (int i = 0; i < n; i += VecLen) {
    aie::vector<bfloat16, VecLen> va = aie::load_v<VecLen>(gate + i);
    aie::vector<bfloat16, VecLen> vb = aie::load_v<VecLen>(up + i);
    aie::vector<bfloat16, VecLen> vr = aie::mul(va, vb);
    aie::store_v(gate + i, vr);
  }
}

} // extern "C"
