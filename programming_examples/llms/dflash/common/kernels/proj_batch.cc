// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "fused_decode/kernels/proj_qmm.cc"
#include "q4k_batch.h"

static_assert(PROJ_MM_BATCH == 2 || PROJ_MM_BATCH == 8 || PROJ_MM_BATCH == 16);

#if PROJ_MM_BATCH == 2
// Two tokens are too few for an mmul: dequantize each 32x32 group once and
// MAC it into both tokens' 32-row accumulators. The block is packed as two
// 16-row halves, joined per column so each MAC covers all 32 rows.
// x: [K/8][token][8]; acc: [M/8][token][8], as the mmul path.
// ROWS: a mask of the rows to accumulate; a row outside it is left as is.
template <int M, int N, int ROWS = 3>
static inline void q4k_gemv_b2(const q4k_block_t *A, const bf16 *x,
                               float *acc) {
  static_assert(M == 32, "two 16-row halves");
  constexpr int pr = 16;
  constexpr int half = (N / 32) * 4 * pr * 4; // qs offset of rows 16..31
  const uint4 *q_lo = A->qs;
  const uint4 *q_hi = A->qs + half;

  aie::vector<float, 2 * M> c = aie::load_v<2 * M>(acc);
  aie::accum<accfloat, M> acc0, acc1;
  acc0.from_vector(aie::filter_even(c, 8));
  acc1.from_vector(aie::filter_odd(c, 8));

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_RANGE(N / 32, N / 32)
  for (int g = 0; g < N / 32; g++)
    AIE_LOOP_FLATTEN {
      aie::vector<bf16, 64> xg = aie::load_v<64>(x + 64 * g);
      aie::vector<bf16, 32> x0 = aie::filter_even(xg, 8);
      aie::vector<bf16, 32> x1 = aie::filter_odd(xg, 8);
      aie::vector<bf16, M> a_scales = aie::load_v<M>(A->scales + g * M);
      aie::vector<bf16, M> a_mins = aie::load_v<M>(A->mins + g * M);

      aie::accum<accfloat, M> t0 = aie::zeros<accfloat, M>();
      aie::accum<accfloat, M> t1 = aie::zeros<accfloat, M>();
      AIE_LOOP_UNROLL_FULL
      for (int t = 0; t < 4; t++) {
        aie::accum<accfloat, pr * 8> lo_f, hi_f;
        lo_f.from_vector(aie::to_float(aie::load_v<pr * 8>(q_lo), 0));
        hi_f.from_vector(aie::to_float(aie::load_v<pr * 8>(q_hi), 0));
        q_lo += pr * 4;
        q_hi += pr * 4;
        aie::vector<bf16, pr * 8> lo = lo_f.template to_vector<bf16>();
        aie::vector<bf16, pr * 8> hi = hi_f.template to_vector<bf16>();
        AIE_LOOP_UNROLL_FULL
        for (int i = 0; i < 8; i++) {
          aie::vector<bf16, M> col = aie::concat(lo.template extract<pr>(i),
                                                 hi.template extract<pr>(i));
          if constexpr (ROWS & 1)
            t0 = aie::mac(t0, col, x0.get(8 * t + i));
          if constexpr (ROWS & 2)
            t1 = aie::mac(t1, col, x1.get(8 * t + i));
        }
      }
      // w = q * scale + min: the min term is min * sum(x) over the group
      if constexpr (ROWS & 1) {
        acc0 = aie::mac(acc0, t0.template to_vector<bf16>(), a_scales);
        acc0 = aie::mac(acc0, a_mins, bf16(aie::reduce_add(x0)));
      }
      if constexpr (ROWS & 2) {
        acc1 = aie::mac(acc1, t1.template to_vector<bf16>(), a_scales);
        acc1 = aie::mac(acc1, a_mins, bf16(aie::reduce_add(x1)));
      }
    }
  auto [z0, z1] = aie::interleave_zip(acc0.template to_vector<float>(),
                                      acc1.template to_vector<float>(), 8);
  aie::store_v(acc, aie::concat(z0, z1));
}
#endif

extern "C" {
void proj_qmm_mm_zero(float *__restrict acc, int arm) {
  (void)arm;
  aie_round_nearest_even();
  zero_vectorized<float, PROJ_MM_BATCH * Q4NX_ROW_BLOCK_SIZE>(acc);
}

void proj_qmm_mm_acc(bf16 *__restrict x, bf16 *__restrict packed,
                     float *__restrict acc, bf16 *__restrict scratch) {
  aie_round_nearest_even();
#if PROJ_MM_BATCH == 2
  (void)scratch;
  q4k_gemv_b2<Q4NX_ROW_BLOCK_SIZE, Q4NX_COL_BLOCK_SIZE>(
      reinterpret_cast<const q4k_block_t *>(packed), x, acc);
#else
  q4k_unpack_block<Q4NX_ROW_BLOCK_SIZE, Q4NX_COL_BLOCK_SIZE>(
      reinterpret_cast<const q4k_block_t *>(packed), scratch);
#if PROJ_MM_BATCH == 16
  q4k_mmul_b16_2x2<Q4NX_ROW_BLOCK_SIZE, Q4NX_COL_BLOCK_SIZE>(x, scratch, acc);
#else
  q4k_mmul_small<Q4NX_ROW_BLOCK_SIZE, Q4NX_COL_BLOCK_SIZE, 8>(x, scratch, acc);
#endif
#endif
}

} // extern "C"

#if PROJ_MM_BATCH == 2
// Independent sequences: mask bit s set when sequence s has a token.
extern "C" void proj_qmm_mm_acc_m(bf16 *__restrict x, bf16 *__restrict packed,
                                  float *__restrict acc,
                                  bf16 *__restrict scratch, int mask) {
  (void)scratch;
  aie_round_nearest_even();
  auto *A = reinterpret_cast<const q4k_block_t *>(packed);
  constexpr int M = Q4NX_ROW_BLOCK_SIZE, N = Q4NX_COL_BLOCK_SIZE;
  if (mask == 1)
    q4k_gemv_b2<M, N, 1>(A, x, acc);
  else if (mask == 2)
    q4k_gemv_b2<M, N, 2>(A, x, acc);
  else
    q4k_gemv_b2<M, N, 3>(A, x, acc);
}
#endif

// Keep the GEMM accumulator resident, but drain one token per short packet.
// Packet routing metadata is still generated from the AIR put's dest operand.
extern "C" void proj_qmm_mm_flush_token(float *__restrict acc,
                                        bf16 *__restrict packet, int token,
                                        int unused) {
  (void)unused;
  aie_round_nearest_even();
  constexpr int width = Q4NX_ROW_BLOCK_SIZE;
  alignas(aie::vector_decl_align) float row[width];
  AIE_LOOP_UNROLL_FULL
  for (int j = 0; j < width / 8; ++j)
    aie::store_v(row + j * 8,
                 aie::load_v<8>(acc + j * PROJ_MM_BATCH * 8 + token * 8));
  copy_float_to_bf16<width>(packet + 16, row);
}
