// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// The fused prefill GEMM's K step in bfp16, the cores' native matmul type. A
// bf16 matmul converts both operands to bfp16 at every mac; here A and the bf16
// weights arrive as bfp16 and int4 weights are dequantized straight to it.
//
// A bfp16ebs8 block is 8x8 values in 72 bytes: per row one shared exponent
// byte and 8 int8 mantissas. Layouts, per K step:
//   A    [TM/16][TK/8][2]  row blocks in pairs, rows m, groups along k
//   B    [TN/16][TK/8][2]  column blocks in pairs, rows n, groups along k
//   acc  f32 [TN/8][TM/8][8 m][8 n]
// Pairs, so one stream feeds both rows of a 2x2 block of accumulators.

#include <aie_api/aie.hpp>
#include <cstdint>

#ifndef DIM_M
#define DIM_M 32
#endif
#ifndef DIM_N
#define DIM_N 64
#endif
#ifndef DIM_K
#define DIM_K 128
#endif
#define DQ_GROUP 32

using BV = aie::block_vector<bfp16ebs8, 64>;
using ACC = aie::accum<accfloat, 64>;
using VB = aie::vector<bfloat16, 64>;
using BS = aie::block_vector_input_buffer_stream<bfp16ebs8, 64>;

static inline ACC mma(ACC c, BV a, BV b) {
  return ACC(::mac_8x8_8x8T_conf(a, b, c, 0, 0, 0));
}

// acc += A B over KB k blocks; A's row block pairs sit AS bytes apart
template <int KB, int AS>
static inline void mm(const char *a, const char *b, float *acc) {
  constexpr int MB = DIM_M / 8, NB = DIM_N / 8, BB = 2 * 72 * KB;
  for (int z = 0; z < MB; z += 2)
    for (int j = 0; j < NB; j += 2)
      chess_prepare_for_pipelining {
        float *c00 = acc + (j * MB + z) * 64, *c10 = c00 + 64;
        float *c01 = c00 + MB * 64, *c11 = c01 + 64;
        ACC C00, C01, C10, C11;
        C00.from_vector(aie::load_v<64>(c00));
        C01.from_vector(aie::load_v<64>(c01));
        C10.from_vector(aie::load_v<64>(c10));
        C11.from_vector(aie::load_v<64>(c11));
        // two streams: the cores have two block-vector load FIFOs
        BS as((const bfp16ebs8 *)(a + z / 2 * AS));
        BS bs((const bfp16ebs8 *)(b + j / 2 * BB));
#pragma clang loop unroll_count(4)
        for (int k = 0; k < KB; k++) {
          BV A0, A1, B0, B1;
          as >> A0 >> A1;
          bs >> B0 >> B1;
          C00 = mma(C00, A0, B0);
          C01 = mma(C01, A0, B1);
          C10 = mma(C10, A1, B0);
          C11 = mma(C11, A1, B1);
        }
        aie::store_v(c00, C00.template to_vector<float>());
        aie::store_v(c01, C01.template to_vector<float>());
        aie::store_v(c10, C10.template to_vector<float>());
        aie::store_v(c11, C11.template to_vector<float>());
      }
}

// lane r * 8 + c is v[r]
static inline VB rows8(aie::vector<bfloat16, 8> v) {
  aie::vector<bfloat16, 16> b = aie::concat(v, v);
  aie::vector<bfloat16, 32> c = aie::concat(b, b);
  return aie::transpose(aie::concat(c, c), 8, 8);
}

extern "C" {

void mm_bfp16(uint8_t *a, uint8_t *b, float *acc) {
  mm<DIM_K / 8, 2 * 72 * (DIM_K / 8)>((const char *)a, (const char *)b, acc);
}

// bf16 weights, packed as bfp16 on the host in two packets per K step: packet
// h holds B's blocks [TN/16][TK/16][2] of K half h
void mm_bfp16_pkt(uint8_t *a, uint8_t *b, float *acc, int32_t h) {
  constexpr int KH = DIM_K / 16;
  mm<KH, 2 * 72 * (DIM_K / 8)>((const char *)a + h * 2 * 72 * KH,
                               (const char *)b, acc);
}

// One K step of int4 weights (packing.pack_q4) -> B:
//   uint8 q[TN/8][TK/8][8 n][8 k / 2]  two nibbles per byte, low nibble first
//   bf16  scale[TK/32][TN]             per output column, per 32-wide K group
//   bf16  mins [TK/32][TN]
// w = q * scale + min. The bf16 bit pattern 0x4300 | q is exactly 128 + q,
// so w = d (128 + q) + (m - 128 d) needs no int conversion.
void dq4_bfp16(uint8_t *__restrict packed, uint8_t *__restrict out) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  constexpr unsigned LANES = 64;
  constexpr unsigned NG = DIM_K / DQ_GROUP;
  constexpr unsigned QCOL = DIM_K / 8 * 32; // q bytes of one column block
  const bfloat16 *sc = (const bfloat16 *)(packed + DIM_N * DIM_K / 2);
  const bfloat16 *mn = sc + NG * DIM_N;
  const aie::vector<uint8, 2 * LANES> bias =
      aie::broadcast<uint8, 2 * LANES>(0x43);
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> os((bfp16ebs8 *)out);
  // per column block pair and K group: four K blocks of each column block,
  // written interleaved in B's pair order
  for (unsigned i = 0; i < DIM_N / 16 * NG; i++)
    chess_prepare_for_pipelining {
      unsigned np = i / NG, g = i % NG, n0 = np * 16;
      VB s0 = rows8(aie::load_v<8>(sc + g * DIM_N + n0));
      VB m0 = rows8(aie::load_v<8>(mn + g * DIM_N + n0));
      VB s1 = rows8(aie::load_v<8>(sc + g * DIM_N + n0 + 8));
      VB m1 = rows8(aie::load_v<8>(mn + g * DIM_N + n0 + 8));
      ACC b0, b1;
      b0.from_vector(m0);
      b0 = aie::msc(b0, s0, bfloat16(128.0f));
      b1.from_vector(m1);
      b1 = aie::msc(b1, s1, bfloat16(128.0f));
      // uint4 pointers step in bytes
      const uint4 *q0 = (const uint4 *)(packed + 2 * np * QCOL + g * 128);
      const uint4 *q1 = (const uint4 *)(packed + (2 * np + 1) * QCOL + g * 128);
#pragma clang loop unroll(full)
      for (unsigned t = 0; t < DQ_GROUP / 16; t++) {
        auto z0 = aie::interleave_zip(
            aie::unpack(aie::load_v<2 * LANES>(q0 + t * LANES)), bias, 1);
        auto z1 = aie::interleave_zip(
            aie::unpack(aie::load_v<2 * LANES>(q1 + t * LANES)), bias, 1);
        os << ACC(aie::mac(b0, aie::vector_cast<bfloat16>(z0.first), s0))
                  .template to_vector<bfp16ebs8>();
        os << ACC(aie::mac(b1, aie::vector_cast<bfloat16>(z1.first), s1))
                  .template to_vector<bfp16ebs8>();
        os << ACC(aie::mac(b0, aie::vector_cast<bfloat16>(z0.second), s0))
                  .template to_vector<bfp16ebs8>();
        os << ACC(aie::mac(b1, aie::vector_cast<bfloat16>(z1.second), s1))
                  .template to_vector<bfp16ebs8>();
      }
    }
}

} // extern "C"
