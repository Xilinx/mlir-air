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

#ifdef GEMM_MXFP4
// One K step of MXFP4 weights as the gpt-oss bundle stores them: two blocks of
// 32 output columns n by 128 k, 2560 bytes each,
//   uint8 e8m0[4 k groups][32 n], 384 bytes not read here (the bias),
//   nibbles [2 stripes of 16 n][64 byte columns][16]: bytes 0..7 of a byte
//   column are k column 2c, bytes 8..15 column 2c + 1, rows 2j (low) and
//   2j + 1 (high) in byte j,
// -> B, built as bfp16 directly. A bfp16 block row is 8 int8 mantissas m and
// an exponent byte x, the values m 2^(x - 133); twice an e2m1 value is an
// integer r in [-12, 12] (the magnitude m is r below 4, 2m - 4 below 6, 4m -
// 16 above), so with x = e8m0 + 5 the block holds the MXFP4 values exactly.
// The nibbles run along n and a block row along k: an unzip by eights splits
// four byte columns [8 k][16 n] into a stripe's two halves and a transpose
// turns each round. A block's stripes are B's column block pairs in turn.
static_assert(DIM_N == 64 && DIM_K == 128, "a K step is two MXFP4 blocks");

void dqmx_bfp16(uint8_t *__restrict packed, uint8_t *__restrict out) {
  constexpr int BLK = 2560, DATA = 512, STRIPE = 16;
  constexpr uint32_t X = 0x05050505u;
  const aie::vector<int8, 128> c4 = aie::broadcast<int8, 128>(4);
  const aie::vector<int8, 128> c6 = aie::broadcast<int8, 128>(6);
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> os((bfp16ebs8 *)out);
  // one trip per 8 k of a stripe: block b, stripe s, k group g, 8 columns q
  for (int i = 0; i < 64; i++)
    chess_prepare_for_pipelining {
      const int b = i >> 5, s = (i >> 4) & 1, g = (i >> 2) & 3, q = i & 3;
      const uint8_t *blk = packed + b * BLK;
      // the stripe's 16 row exponents, 4 to a word; e8m0 <= 250
      const uint32_t *e = (const uint32_t *)(blk + g * 32 + s * STRIPE);
      const v2int32 x0 = {(int)(e[0] + X), (int)(e[1] + X)};
      const v2int32 x1 = {(int)(e[2] + X), (int)(e[3] + X)};
      const uint8_t *d =
          blk + DATA + s * (DIM_K * STRIPE / 2) + g * 256 + q * 64;
      aie::vector<int8, 128> v = aie::vector_cast<int8>(
          aie::unpack(aie::load_v<128>((const uint4 *)d)));
      aie::vector<int8, 128> m = aie::bit_and(int8(7), v);
      aie::vector<int8, 128> t4 = aie::sub(aie::max(m, c4), c4);
      aie::vector<int8, 128> t6 = aie::sub(aie::max(m, c6), c6);
      aie::vector<int8, 128> r = aie::add(aie::add(m, t4), aie::add(t6, t6));
      r = aie::select(r, aie::neg(r), aie::ge(v, int8(8)));
      auto [lo, hi] = aie::interleave_unzip(r.template extract<64>(0),
                                            r.template extract<64>(1), 8);
      v64bfp16ebs8 b0 = {aie::transpose(lo, 8, 8), x0};
      v64bfp16ebs8 b1 = {aie::transpose(hi, 8, 8), x1};
      os << aie::block_vector<bfp16ebs8, 64>(b0);
      os << aie::block_vector<bfp16ebs8, 64>(b1);
    }
}
#endif

#ifdef GEMM_Q4K
// Half a K step pair of Q4_K weights as FastFlowLM's bundle stores them, one
// block of 32 output columns n by 256 k per packet, 4736 bytes:
//   uint8 scales [8 k groups][32 n], mins the same,
//   nibbles [256 k][32 n], the low nibble the even n,
//   bf16 S [32], M [32] (stored negated),
// with w = S s_j q + M m_j, as FastFlowLM dequantizes it: the group's
// scale and min are bf16 products, and 0x4300 | q is the bf16 128 + q. b0
// holds columns 0..31 of the panel, b1 32..63; h is the K step's half of
// the blocks. The nibbles run along n, so a byte-domain transpose turns
// each 8 x 8 block into B's rows n.
static_assert(DIM_N == 64 && DIM_K == 128, "a K step is half of two blocks");

// One 16-column pair (PAR: the block's first or second 16 columns) of
// dq4k_bfp16: the pair's bytes are split off before they are unpacked.
extern "C++" {
template <int PAR>
static inline void
dq4k_pair(const uint8_t *blk, int h,
          aie::block_vector_output_buffer_stream<bfp16ebs8, 64> &os) {
  constexpr int RB = 32, NG = 8, QS = 2 * NG * RB, SM = QS + 256 * RB / 2;
  constexpr int r0 = PAR * 16;
  const aie::vector<uint8, 64> bias = aie::broadcast<uint8, 64>(0x43);
  const bfloat16 *sm = (const bfloat16 *)(blk + SM);
  aie::vector<bfloat16, 16> S = aie::load_v<16>(sm + r0);
  aie::vector<bfloat16, 16> M = aie::load_v<16>(sm + RB + r0);
  for (int gl = 0; gl < DIM_K / DQ_GROUP; gl++) {
    const int g = h * (DIM_K / DQ_GROUP) + gl;
    aie::accum<accfloat, 16> t;
    t.from_vector(aie::to_float(aie::load_v<16>(blk + g * RB + r0), 0));
    aie::vector<bfloat16, 16> sc = aie::mul(t.template to_vector<bfloat16>(), S)
                                       .template to_vector<bfloat16>();
    t.from_vector(
        aie::to_float(aie::load_v<16>(blk + NG * RB + g * RB + r0), 0));
    aie::vector<bfloat16, 16> mn = aie::mul(t.template to_vector<bfloat16>(), M)
                                       .template to_vector<bfloat16>();
    VB s0 = rows8(sc.template extract<8>(0));
    VB s1 = rows8(sc.template extract<8>(1));
    ACC c0, c1;
    c0.from_vector(rows8(mn.template extract<8>(0)));
    c0 = aie::msc(c0, s0, bfloat16(128.0f));
    c1.from_vector(rows8(mn.template extract<8>(1)));
    c1 = aie::msc(c1, s1, bfloat16(128.0f));
    // [k][16 bytes]: bytes 0-7 hold n 0-15, bytes 8-15 n 16-31
    const uint8_t *q = blk + QS + (g * DQ_GROUP) * RB / 2;
    for (int kq = 0; kq < DQ_GROUP / 8; kq++)
      chess_prepare_for_pipelining {
        aie::vector<uint8, 32> lo[2], hi[2];
#pragma clang loop unroll(full)
        for (int c = 0; c < 2; c++) {
          aie::vector<uint8, 64> raw = aie::load_v<64>(q + (kq * 2 + c) * 64);
          auto by = aie::interleave_unzip(raw.template extract<32>(0),
                                          raw.template extract<32>(1), 8);
          aie::vector<uint8, 64> mine =
              aie::unpack(aie::vector_cast<uint4>(PAR ? by.second : by.first));
          auto nb = aie::interleave_unzip(mine.template extract<32>(0),
                                          mine.template extract<32>(1), 8);
          lo[c] = nb.first;
          hi[c] = nb.second;
        }
        aie::vector<uint8, 64> tl =
            aie::transpose(aie::concat(lo[0], lo[1]), 8, 8);
        aie::vector<uint8, 64> th =
            aie::transpose(aie::concat(hi[0], hi[1]), 8, 8);
        auto zl = aie::interleave_zip(tl, bias, 1);
        auto zh = aie::interleave_zip(th, bias, 1);
        VB wl = aie::vector_cast<bfloat16>(aie::concat(zl.first, zl.second));
        VB wh = aie::vector_cast<bfloat16>(aie::concat(zh.first, zh.second));
        os << ACC(aie::mac(c0, wl, s0)).template to_vector<bfp16ebs8>();
        os << ACC(aie::mac(c1, wh, s1)).template to_vector<bfp16ebs8>();
      }
  }
}
} // extern "C++"

void dq4k_bfp16(uint8_t *__restrict b0, uint8_t *__restrict b1,
                uint8_t *__restrict out, int32_t h) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> os((bfp16ebs8 *)out);
  dq4k_pair<0>(b0, h, os);
  dq4k_pair<1>(b0, h, os);
  dq4k_pair<0>(b1, h, os);
  dq4k_pair<1>(b1, h, os);
}
#endif

} // extern "C"
