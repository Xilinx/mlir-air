// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// Unpack and 1x4-blocked mmul primitives adapted from pr1942-port 67831a05.
// Packed Q4K blocks are addressed as raw BF16 words, never by sizeof(struct).
// Inputs: A [K/8, token, 8]; B [output/8, K/8, 8, 8].
// Output: [output/8, token, 8] FP32 accumulators.
#pragma once
#include "aie_kernel_utils.h"
#include "model_spec.h"

template <int PR>
static inline aie::vector<bf16, PR * 8>
q4k_unpack_step(const uint4 *&qs_ptr, const bf16 *scales, const bf16 *mins) {
  aie::vector<uint4, PR * 8> q = aie::load_v<PR * 8>(qs_ptr);
  qs_ptr += PR * 4;
  aie::accum<accfloat, PR * 8> acc;
  acc.from_vector(aie::to_float(q, 0));
  aie::vector<bf16, PR * 8> qb = acc.template to_vector<bf16>();

  aie::vector<bf16, PR> s16 = aie::load_v<PR>(scales);
  aie::vector<bf16, PR> m16 = aie::load_v<PR>(mins);
  aie::vector<bf16, PR * 2> s32 = aie::concat(s16, s16);
  aie::vector<bf16, PR * 2> m32 = aie::concat(m16, m16);
  aie::vector<bf16, PR * 4> s64 = aie::concat(s32, s32);
  aie::vector<bf16, PR * 4> m64 = aie::concat(m32, m32);
  aie::vector<bf16, PR * 8> sv = aie::concat(s64, s64);
  aie::vector<bf16, PR * 8> mv = aie::concat(m64, m64);

#ifdef Q4K_UNPACK_FMA
  // Optional PR #1942 arithmetic: round q*scale+min once, without the
  // intermediate BF16 product. Validate separately from scheduling changes.
  aie::accum<accfloat, PR * 8> out;
  out.from_vector(mv);
  return aie::mac(out, qb, sv).template to_vector<bf16>();
#else
  aie::vector<bf16, PR * 8> ws = aie::mul(qb, sv).template to_vector<bf16>();
  return aie::add(ws, mv);
#endif
}

template <int MROWS, int KCOL>
static inline void q4k_unpack_block(const q4k_block_t *A, bf16 *__restrict W) {
  constexpr int PR = 16;
  constexpr int CHUNK = PR * 8;
  constexpr int NSTEP = MROWS * KCOL / CHUNK;
  constexpr int NCB = KCOL / 8;
  static_assert(MROWS % 16 == 0, "unpack emits two 8-row tiles per chunk");
  static_assert(KCOL % 32 == 0, "a 32-column quant group must be whole");
  const uint4 *qs_ptr = A->qs;

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_RANGE(NSTEP, NSTEP)
#ifdef Q4K_UNPACK_UNROLL
  AIE_LOOP_UNROLL(Q4K_UNPACK_UNROLL)
#endif
  for (int i = 0; i < NSTEP; i++)
    AIE_LOOP_FLATTEN {

      const int R = i / NCB;
      const int cb = i % NCB;



      const int off = (cb / 4) * MROWS + R * PR;
      aie::vector<bf16, CHUNK> w =
          q4k_unpack_step<PR>(qs_ptr, A->scales + off, A->mins + off);

      aie::store_v(W + ((2 * R) * NCB + cb) * 64, aie::filter_even(w, 8));
      aie::store_v(W + ((2 * R + 1) * NCB + cb) * 64, aie::filter_odd(w, 8));

    }
}

template <int MROWS, int KCOL, int BATCH>
static inline void q4k_mmul_groups(const bf16 *__restrict a,
                                    const bf16 *__restrict b,
                                    float *__restrict c) {
  static_assert(BATCH % 8 == 0 && MROWS % 32 == 0 && KCOL % 8 == 0);
  using MM = aie::mmul<8, 8, 8, bf16, bf16, accauto>;
  constexpr int nk = KCOL / 8;
  // Keep four native accumulators live at a time. Each group shares the same
  // unpacked weights; activation and accumulator strides span all token rows.
  for (int group = 0; group < BATCH / 8; ++group) {
    for (int col = 0; col < MROWS / 8; col += 4) {
      const bf16 *ap = a + group * 64;
      const bf16 *b0 = b + (col + 0) * nk * 64;
      const bf16 *b1 = b + (col + 1) * nk * 64;
      const bf16 *b2 = b + (col + 2) * nk * 64;
      const bf16 *b3 = b + (col + 3) * nk * 64;
      float *cp = c + col * BATCH * 8 + group * 64;
      MM c0(aie::load_v<64>(cp));
      MM c1(aie::load_v<64>(cp + BATCH * 8));
      MM c2(aie::load_v<64>(cp + 2 * BATCH * 8));
      MM c3(aie::load_v<64>(cp + 3 * BATCH * 8));
      AIE_PREPARE_FOR_PIPELINING
      AIE_LOOP_RANGE(nk, nk)
      for (int k = 0; k < nk; ++k) {
        auto av = aie::load_v<64>(ap);
        ap += BATCH * 8;
        c0.mac(av, aie::load_v<64>(b0)); b0 += 64;
        c1.mac(av, aie::load_v<64>(b1)); b1 += 64;
        c2.mac(av, aie::load_v<64>(b2)); b2 += 64;
        c3.mac(av, aie::load_v<64>(b3)); b3 += 64;
      }
      aie::store_v(cp, c0.template to_vector<float>());
      aie::store_v(cp + BATCH * 8, c1.template to_vector<float>());
      aie::store_v(cp + 2 * BATCH * 8, c2.template to_vector<float>());
      aie::store_v(cp + 3 * BATCH * 8, c3.template to_vector<float>());
    }
  }
}

template <int MROWS, int KCOL, int BATCH>
static inline void q4k_mmul_small(const bf16 *__restrict pA,
                                  const bf16 *__restrict pB,
                                  float *__restrict pC) {
  constexpr int r = (BATCH < 8) ? 4 : 8, s = 8, t = 8;
  using MMUL = aie::mmul<r, s, t, bf16, bf16, accauto>;
  constexpr int rowA = BATCH / r;
  constexpr int colA = KCOL / s;
  constexpr int colB = MROWS / t;
  static_assert(
      BATCH == 4 || BATCH == 8,
      "q4k_mmul_small covers rowA == 1; use q4k_mmul at 16 and above");
  static_assert(rowA == 1, "rowA must be 1 here");
  static_assert(colB % 4 == 0, "MROWS must be a multiple of 32 for 1x4");

  for (unsigned j = 0; j < colB; j += 4) {
    const bf16 *__restrict pA1 = pA;
    const bf16 *__restrict pB0 = pB + (j + 0) * colA * MMUL::size_B;
    const bf16 *__restrict pB1 = pB + (j + 1) * colA * MMUL::size_B;
    const bf16 *__restrict pB2 = pB + (j + 2) * colA * MMUL::size_B;
    const bf16 *__restrict pB3 = pB + (j + 3) * colA * MMUL::size_B;
    float *__restrict q = pC + j * MMUL::size_C;


    MMUL C0(aie::load_v<MMUL::size_C>(q + 0 * MMUL::size_C));
    MMUL C1(aie::load_v<MMUL::size_C>(q + 1 * MMUL::size_C));
    MMUL C2(aie::load_v<MMUL::size_C>(q + 2 * MMUL::size_C));
    MMUL C3(aie::load_v<MMUL::size_C>(q + 3 * MMUL::size_C));

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(colA, colA)

    for (unsigned i = 0; i < colA; ++i) {
      aie::vector<bf16, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA1);
      pA1 += MMUL::size_A;
      C0.mac(A0, aie::load_v<MMUL::size_B>(pB0));
      pB0 += MMUL::size_B;
      C1.mac(A0, aie::load_v<MMUL::size_B>(pB1));
      pB1 += MMUL::size_B;
      C2.mac(A0, aie::load_v<MMUL::size_B>(pB2));
      pB2 += MMUL::size_B;
      C3.mac(A0, aie::load_v<MMUL::size_B>(pB3));
      pB3 += MMUL::size_B;
    }
    aie::store_v(q + 0 * MMUL::size_C, C0.template to_vector<float>());
    aie::store_v(q + 1 * MMUL::size_C, C1.template to_vector<float>());
    aie::store_v(q + 2 * MMUL::size_C, C2.template to_vector<float>());
    aie::store_v(q + 3 * MMUL::size_C, C3.template to_vector<float>());
  }
}

// B16 alternative: retain four accumulators as two token groups by two
// output blocks. Each weight operand serves both groups immediately.
template <int MROWS, int KCOL>
static inline void q4k_mmul_b16_2x2(const bf16 *__restrict a,
                                  const bf16 *__restrict b,
                                  float *__restrict c) {
  static_assert(MROWS % 16 == 0 && KCOL % 8 == 0);
#if defined(DFLASH_PROJECTION_NATIVE) && DFLASH_PROJECTION_NATIVE
  using MM = aie::mmul<8,8,8,bfp16ebs8,bfp16ebs8,accfloat>;
  using BV = aie::block_vector<bfp16ebs8,64>;
#else
  using MM = aie::mmul<8,8,8,bf16,bf16,accauto>;
#endif
  constexpr int nk=KCOL/8;
  for (int col=0;col<MROWS/8;col+=2) {
    float *cp=c+col*128;
    MM c00(aie::load_v<64>(cp)),c10(aie::load_v<64>(cp+64));
    MM c01(aie::load_v<64>(cp+128)),c11(aie::load_v<64>(cp+192));
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(nk,nk)
    for (int k=0;k<nk;++k) {
      auto a0=aie::load_v<64>(a+k*128);
      auto a1=aie::load_v<64>(a+k*128+64);
      auto w0=aie::load_v<64>(b+(col*nk+k)*64);
      auto w1=aie::load_v<64>(b+((col+1)*nk+k)*64);
#if defined(DFLASH_PROJECTION_NATIVE) && DFLASH_PROJECTION_NATIVE
      aie::accum<accfloat,64> aa0(a0),aa1(a1);
      auto ones=::concat(::broadcast_one_to_v32bfloat16(),::broadcast_one_to_v32bfloat16());
      aie::accum<accfloat,64> ww0=::mul_elem_64(aie::transpose(w0,8,8),ones);
      aie::accum<accfloat,64> ww1=::mul_elem_64(aie::transpose(w1,8,8),ones);
      BV av0(::to_v64bfp16ebs8(aa0)),av1(::to_v64bfp16ebs8(aa1));
      BV bv0(::to_v64bfp16ebs8(ww0)),bv1(::to_v64bfp16ebs8(ww1));
      c00.mac(av0,aie::op_transpose(bv0));c10.mac(av1,aie::op_transpose(bv0));
      c01.mac(av0,aie::op_transpose(bv1));c11.mac(av1,aie::op_transpose(bv1));
#else
      c00.mac(a0,w0);c10.mac(a1,w0);
      c01.mac(a0,w1);c11.mac(a1,w1);
#endif
    }
    aie::store_v(cp,c00.template to_vector<float>());
    aie::store_v(cp+64,c10.template to_vector<float>());
    aie::store_v(cp+128,c01.template to_vector<float>());
    aie::store_v(cp+192,c11.template to_vector<float>());
  }
}
