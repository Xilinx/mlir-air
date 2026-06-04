//===- mm_bf16_x_bfp16.cc - bf16 A x bfp16ebs8 B -> bf16 C kernel ---------===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Mixed-precision GEMM micro-kernel: bf16 activations × bfp16ebs8 weights,
// f32 L1 accumulator, bf16 store-out. Targets AIE2P `mac_8x8_8x8T` BFP MMUL.
// B is uint8 at the AIR boundary (no MLIR bfp16ebs8 element type) and the
// kernel reinterprets via `aie::block_vector<bfp16ebs8>`.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_M
#define DIM_M 64
#endif
#ifndef DIM_K
#define DIM_K 64
#endif
#ifndef DIM_N
#define DIM_N 64
#endif

constexpr aie::rounding_mode round_mode = aie::rounding_mode::conv_even;

// C is N-outer M-inner [n_b][m_b][r][t] (instead of the more natural M-outer)
// so the L1->L2 drain DMA fits in the AIE2P 3-dim BD step limit.
template <unsigned rowA, unsigned colA, unsigned colB, unsigned r, unsigned s,
          unsigned t>
static void matmul_vectorized_2x2_bfp16_bf16_f32(const bfloat16 *__restrict pA,
                                                 const bfp16ebs8 *__restrict pB,
                                                 float *__restrict pC) {
  const unsigned sizeA = r * s;
  const unsigned sizeB = s * t;
  const unsigned sizeC = r * t;

  for (unsigned j = 0; j < colB; j += 2) {
    float *__restrict pC1 = pC + (j * rowA) * sizeC;
    float *__restrict pC2 = pC + ((j + 1) * rowA) * sizeC;

    for (unsigned z = 0; z < rowA; z += 2) {
      const bfloat16 *__restrict pA1 = pA + (z * colA) * sizeA;
      const bfloat16 *__restrict pA2 = pA + ((z + 1) * colA) * sizeA;

      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pB1bfp16(pB);
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pB2bfp16(pB);
      pB1bfp16.seek(j * colA);
      pB2bfp16.seek((j + 1) * colA);

      aie::vector<bfloat16, sizeA> A0;
      aie::vector<bfloat16, sizeA> A1;
      aie::block_vector<bfp16ebs8, sizeB> B0;
      aie::block_vector<bfp16ebs8, sizeB> B1;

      aie::accum<accfloat, sizeC> accC00(aie::load_v<sizeC>(pC1));
      aie::accum<accfloat, sizeC> accC10(aie::load_v<sizeC>(pC1 + sizeC));
      aie::accum<accfloat, sizeC> accC01(aie::load_v<sizeC>(pC2));
      aie::accum<accfloat, sizeC> accC11(aie::load_v<sizeC>(pC2 + sizeC));

      aie::accum<accfloat, 64> accA0;
      aie::accum<accfloat, 64> accA1;

      // Indexed loads avoid VLD_x_pstm_nrm_imm_pseudo Peano backend bug at
      // small tiles; no perf impact at larger tiles.
      for (unsigned i = 0; i < colA; ++i) {
        A0 = aie::load_v<sizeA>(pA1 + i * sizeA);
        A1 = aie::load_v<sizeA>(pA2 + i * sizeA);

        accA0 = A0;
        accA1 = mul_elem_64(A1, concat(broadcast_one_to_v32bfloat16(),
                                       broadcast_one_to_v32bfloat16()));

        B0 = pB1bfp16.pop();
        B1 = pB2bfp16.pop();

        accC00 = mac_8x8_8x8T(accA0.to_vector<bfp16ebs8>(), B0, accC00);
        accC01 = mac_8x8_8x8T(accA0.to_vector<bfp16ebs8>(), B1, accC01);
        accC10 = mac_8x8_8x8T(accA1.to_vector<bfp16ebs8>(), B0, accC10);
        accC11 = mac_8x8_8x8T(accA1.to_vector<bfp16ebs8>(), B1, accC11);
      }

      aie::store_v(pC1, accC00.template to_vector<float>());
      aie::store_v(pC1 + sizeC, accC10.template to_vector<float>());
      pC1 += 2 * sizeC;
      aie::store_v(pC2, accC01.template to_vector<float>());
      aie::store_v(pC2 + sizeC, accC11.template to_vector<float>());
      pC2 += 2 * sizeC;
    }
  }
}

template <unsigned m_tile, unsigned n_tile>
static void zero_mn_f32_impl(float *__restrict c) {
  constexpr unsigned VW = 16;
  constexpr unsigned NTOT = m_tile * n_tile;
  static_assert(NTOT % VW == 0,
                "m_tile*n_tile must be a multiple of f32 vector width");
  aie::vector<float, VW> zv = aie::zeros<float, VW>();
  for (unsigned i = 0; i < NTOT; i += VW)
    aie::store_v(c + i, zv);
}

template <unsigned m_tile, unsigned n_tile>
static void f32_to_bf16_mn_impl(const float *__restrict src,
                                bfloat16 *__restrict dst) {
  constexpr unsigned VW = 16;
  constexpr unsigned NTOT = m_tile * n_tile;
  static_assert(NTOT % VW == 0, "m_tile*n_tile must be a multiple of VW");
  for (unsigned i = 0; i < NTOT; i += VW) {
    aie::vector<float, VW> v = aie::load_v<VW>(src + i);
    aie::vector<bfloat16, VW> vb;
    for (unsigned j = 0; j < VW; j++)
      vb[j] = (bfloat16)v[j];
    aie::store_v(dst + i, vb);
  }
}

extern "C" {

void matmul_bf16_x_bfp16_packed_f32(bfloat16 *pA, uint8_t *pB_bytes,
                                    float *pC) {
  constexpr int r = 8, s = 8, t = 8;
  constexpr int m = DIM_M, k = DIM_K, n = DIM_N;
  static_assert(m % (2 * r) == 0, "DIM_M must be multiple of 16 (2x mmul m)");
  static_assert(n % (2 * t) == 0, "DIM_N must be multiple of 16 (2x mmul n)");
  static_assert(k % s == 0, "DIM_K must be multiple of 8 (mmul k)");
  ::aie::set_rounding(round_mode);
  const bfp16ebs8 *pB = reinterpret_cast<const bfp16ebs8 *>(pB_bytes);
  matmul_vectorized_2x2_bfp16_bf16_f32<m / r, k / s, n / t, r, s, t>(pA, pB,
                                                                     pC);
}

void zero_vectorized_f32_mn(float *pC) { zero_mn_f32_impl<DIM_M, DIM_N>(pC); }

void f32_to_bf16_mn(float *src, bfloat16 *dst) {
  ::aie::set_rounding(round_mode);
  f32_to_bf16_mn_impl<DIM_M, DIM_N>(src, dst);
}

} // extern "C"
