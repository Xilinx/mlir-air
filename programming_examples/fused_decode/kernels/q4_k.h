// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
#ifndef __Q4_K_H__
#define __Q4_K_H__
#include "aie_kernel_utils.h"
#include "model_spec.h"

// Two quant layouts, both produced on the host by _requant_q4k and unrelated to
// how the weight bundle on disk was encoded:
//
//   default (Llama path)     unsigned int4 + per-group min: w = scale*q + min,
//                            so c += (q*b)*scale then c += min*sum(b).
//   Q4_0 (Qwen2.5 path)      signed int4, scale-only, no min. Selected by
//                            defining Q4_0; takes 3 args, since with no +min
//                            term there is no b_col_reduce_buffer.
//
// The first layout used to be called "Q4NX" here. That name is taken: it is the
// safetensors dtype of the OLDER FastFlowLM bundle encoding, whose dequant is
// SUBTRACTIVE (w = scale*(q - zero)) -- the opposite of the additive form
// above. The additive form matches the newer "I8" bundle dtype instead. Naming
// the layout after either bundle codec invites reading the algebra backwards,
// so it is named after what it is.

// How the Q4_0 path sign-corrects a nibble when peano is the backend:
//   1  xor bias, (u ^ 8) - 8    (default)
//   0  compare/select, select(f, f - 16, f >= 8)
//   2  no sign fix at all -- NUMERICALLY WRONG, static-cost attribution only
// Defaulted here rather than at the use site because the preprocessor has no
// scope: defining it inside the function body would still leak to the rest of
// the translation unit, just less visibly. See the block above its use for what
// the modes cost and why the emulation exists at all.
#ifndef Q4_SFIX_MODE
#define Q4_SFIX_MODE 1
#endif

#ifndef Q4_0
template <int M, int N>
void _qmm_q4k_bf16(const q4k_block_t *A, const bf16 *B, float *c,
                   bf16 *b_col_reduce_buffer) {

  constexpr int pr = 16;
  constexpr int pc = 8;
  const q4k_block_t *A_q4k = (q4k_block_t *)A;
  const uint4 *qs_ptr = A_q4k->qs;
  // precompute chunk sum of B;

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_RANGE(M / pr, M / pr)
  for (int row = 0; row < M; row += pr) {
    aie::accum<accfloat, pr> c_accum;
    c_accum.from_vector(aie::load_v<pr>(c + row));
    const bf16 *it_B = B;

    uint32_t scale_min_offset = row;
    bf16 *cur_b_reduce_Buf = b_col_reduce_buffer;

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(N / 32, N / 32)
    for (int sub_chunk = 0; sub_chunk < N / 32; sub_chunk++)
      AIE_LOOP_FLATTEN {

        aie::vector<bf16, 32> b_col = aie::load_v<32>(it_B);
        it_B += 32;

        aie::vector<bf16, pr> a_scales =
            aie::load_v<pr>(A_q4k->scales + scale_min_offset);
        aie::vector<bf16, pr> a_mins =
            aie::load_v<pr>(A_q4k->mins + scale_min_offset);

        scale_min_offset += M;
        bf16 reduce_b = *(cur_b_reduce_Buf);
        cur_b_reduce_Buf++;

        aie::accum<accfloat, pr> temp_acc;

        {

          aie::accum<accfloat, pr * 8> a_cc_f32_accum_0;
          aie::accum<accfloat, pr * 8> a_cc_f32_accum_1;
          aie::accum<accfloat, pr * 8> a_cc_f32_accum_2;
          aie::accum<accfloat, pr * 8> a_cc_f32_accum_3;
          // Load all a vectors
          aie::vector<uint4, pr * 8> a_cc_0 = aie::load_v<pr * 8>(qs_ptr);
          qs_ptr += pr * 4;
          aie::vector<float, pr * 8> a_cc_f32_0 = aie::to_float(a_cc_0, 0);

          a_cc_f32_accum_0.from_vector(a_cc_f32_0);
          aie::vector<bf16, pr * 8> a_cc_bf16_0 =
              a_cc_f32_accum_0.template to_vector<bf16>();

          aie::vector<uint4, pr * 8> a_cc_1 = aie::load_v<pr * 8>(qs_ptr);
          qs_ptr += pr * 4;
          aie::vector<float, pr * 8> a_cc_f32_1 = aie::to_float(a_cc_1, 0);

          a_cc_f32_accum_1.from_vector(a_cc_f32_1);
          aie::vector<bf16, pr * 8> a_cc_bf16_1 =
              a_cc_f32_accum_1.template to_vector<bf16>();

          aie::vector<uint4, pr * 8> a_cc_2 = aie::load_v<pr * 8>(qs_ptr);
          qs_ptr += pr * 4;
          aie::vector<float, pr * 8> a_cc_f32_2 = aie::to_float(a_cc_2, 0);

          a_cc_f32_accum_2.from_vector(a_cc_f32_2);
          aie::vector<bf16, pr * 8> a_cc_bf16_2 =
              a_cc_f32_accum_2.template to_vector<bf16>();

          aie::vector<uint4, pr * 8> a_cc_3 = aie::load_v<pr * 8>(qs_ptr);
          qs_ptr += pr * 4;
          aie::vector<float, pr * 8> a_cc_f32_3 = aie::to_float(a_cc_3, 0);

          a_cc_f32_accum_3.from_vector(a_cc_f32_3);
          aie::vector<bf16, pr * 8> a_cc_bf16_3 =
              a_cc_f32_accum_3.template to_vector<bf16>();

          // Extract all columns
          aie::vector<bf16, pr> a_col_0_0 = a_cc_bf16_0.extract<pr>(0);
          aie::vector<bf16, pr> a_col_1_0 = a_cc_bf16_0.extract<pr>(1);
          aie::vector<bf16, pr> a_col_2_0 = a_cc_bf16_0.extract<pr>(2);
          aie::vector<bf16, pr> a_col_3_0 = a_cc_bf16_0.extract<pr>(3);
          aie::vector<bf16, pr> a_col_4_0 = a_cc_bf16_0.extract<pr>(4);
          aie::vector<bf16, pr> a_col_5_0 = a_cc_bf16_0.extract<pr>(5);
          aie::vector<bf16, pr> a_col_6_0 = a_cc_bf16_0.extract<pr>(6);
          aie::vector<bf16, pr> a_col_7_0 = a_cc_bf16_0.extract<pr>(7);

          aie::vector<bf16, pr> a_col_0_1 = a_cc_bf16_1.extract<pr>(0);
          aie::vector<bf16, pr> a_col_1_1 = a_cc_bf16_1.extract<pr>(1);
          aie::vector<bf16, pr> a_col_2_1 = a_cc_bf16_1.extract<pr>(2);
          aie::vector<bf16, pr> a_col_3_1 = a_cc_bf16_1.extract<pr>(3);
          aie::vector<bf16, pr> a_col_4_1 = a_cc_bf16_1.extract<pr>(4);
          aie::vector<bf16, pr> a_col_5_1 = a_cc_bf16_1.extract<pr>(5);
          aie::vector<bf16, pr> a_col_6_1 = a_cc_bf16_1.extract<pr>(6);
          aie::vector<bf16, pr> a_col_7_1 = a_cc_bf16_1.extract<pr>(7);

          aie::vector<bf16, pr> a_col_0_2 = a_cc_bf16_2.extract<pr>(0);
          aie::vector<bf16, pr> a_col_1_2 = a_cc_bf16_2.extract<pr>(1);
          aie::vector<bf16, pr> a_col_2_2 = a_cc_bf16_2.extract<pr>(2);
          aie::vector<bf16, pr> a_col_3_2 = a_cc_bf16_2.extract<pr>(3);
          aie::vector<bf16, pr> a_col_4_2 = a_cc_bf16_2.extract<pr>(4);
          aie::vector<bf16, pr> a_col_5_2 = a_cc_bf16_2.extract<pr>(5);
          aie::vector<bf16, pr> a_col_6_2 = a_cc_bf16_2.extract<pr>(6);
          aie::vector<bf16, pr> a_col_7_2 = a_cc_bf16_2.extract<pr>(7);

          aie::vector<bf16, pr> a_col_0_3 = a_cc_bf16_3.extract<pr>(0);
          aie::vector<bf16, pr> a_col_1_3 = a_cc_bf16_3.extract<pr>(1);
          aie::vector<bf16, pr> a_col_2_3 = a_cc_bf16_3.extract<pr>(2);
          aie::vector<bf16, pr> a_col_3_3 = a_cc_bf16_3.extract<pr>(3);
          aie::vector<bf16, pr> a_col_4_3 = a_cc_bf16_3.extract<pr>(4);
          aie::vector<bf16, pr> a_col_5_3 = a_cc_bf16_3.extract<pr>(5);
          aie::vector<bf16, pr> a_col_6_3 = a_cc_bf16_3.extract<pr>(6);
          aie::vector<bf16, pr> a_col_7_3 = a_cc_bf16_3.extract<pr>(7);

          // Load all b scalars
          bfloat16 scalar_0_0 = b_col.get(0);
          bfloat16 scalar_1_0 = b_col.get(1);
          bfloat16 scalar_2_0 = b_col.get(2);
          bfloat16 scalar_3_0 = b_col.get(3);
          bfloat16 scalar_4_0 = b_col.get(4);
          bfloat16 scalar_5_0 = b_col.get(5);
          bfloat16 scalar_6_0 = b_col.get(6);
          bfloat16 scalar_7_0 = b_col.get(7);

          bfloat16 scalar_0_1 = b_col.get(8);
          bfloat16 scalar_1_1 = b_col.get(9);
          bfloat16 scalar_2_1 = b_col.get(10);
          bfloat16 scalar_3_1 = b_col.get(11);
          bfloat16 scalar_4_1 = b_col.get(12);
          bfloat16 scalar_5_1 = b_col.get(13);
          bfloat16 scalar_6_1 = b_col.get(14);
          bfloat16 scalar_7_1 = b_col.get(15);

          bfloat16 scalar_0_2 = b_col.get(16);
          bfloat16 scalar_1_2 = b_col.get(17);
          bfloat16 scalar_2_2 = b_col.get(18);
          bfloat16 scalar_3_2 = b_col.get(19);
          bfloat16 scalar_4_2 = b_col.get(20);
          bfloat16 scalar_5_2 = b_col.get(21);
          bfloat16 scalar_6_2 = b_col.get(22);
          bfloat16 scalar_7_2 = b_col.get(23);

          bfloat16 scalar_0_3 = b_col.get(24);
          bfloat16 scalar_1_3 = b_col.get(25);
          bfloat16 scalar_2_3 = b_col.get(26);
          bfloat16 scalar_3_3 = b_col.get(27);
          bfloat16 scalar_4_3 = b_col.get(28);
          bfloat16 scalar_5_3 = b_col.get(29);
          bfloat16 scalar_6_3 = b_col.get(30);
          bfloat16 scalar_7_3 = b_col.get(31);

          // All MAC operations
          temp_acc = aie::mul(a_col_0_0, scalar_0_0);
          temp_acc = aie::mac(temp_acc, a_col_1_0, scalar_1_0);
          temp_acc = aie::mac(temp_acc, a_col_2_0, scalar_2_0);
          temp_acc = aie::mac(temp_acc, a_col_3_0, scalar_3_0);
          temp_acc = aie::mac(temp_acc, a_col_4_0, scalar_4_0);
          temp_acc = aie::mac(temp_acc, a_col_5_0, scalar_5_0);
          temp_acc = aie::mac(temp_acc, a_col_6_0, scalar_6_0);
          temp_acc = aie::mac(temp_acc, a_col_7_0, scalar_7_0);
          temp_acc = aie::mac(temp_acc, a_col_0_1, scalar_0_1);
          temp_acc = aie::mac(temp_acc, a_col_1_1, scalar_1_1);
          temp_acc = aie::mac(temp_acc, a_col_2_1, scalar_2_1);
          temp_acc = aie::mac(temp_acc, a_col_3_1, scalar_3_1);
          temp_acc = aie::mac(temp_acc, a_col_4_1, scalar_4_1);
          temp_acc = aie::mac(temp_acc, a_col_5_1, scalar_5_1);
          temp_acc = aie::mac(temp_acc, a_col_6_1, scalar_6_1);
          temp_acc = aie::mac(temp_acc, a_col_7_1, scalar_7_1);
          temp_acc = aie::mac(temp_acc, a_col_0_2, scalar_0_2);
          temp_acc = aie::mac(temp_acc, a_col_1_2, scalar_1_2);
          temp_acc = aie::mac(temp_acc, a_col_2_2, scalar_2_2);
          temp_acc = aie::mac(temp_acc, a_col_3_2, scalar_3_2);
          temp_acc = aie::mac(temp_acc, a_col_4_2, scalar_4_2);
          temp_acc = aie::mac(temp_acc, a_col_5_2, scalar_5_2);
          temp_acc = aie::mac(temp_acc, a_col_6_2, scalar_6_2);
          temp_acc = aie::mac(temp_acc, a_col_7_2, scalar_7_2);
          temp_acc = aie::mac(temp_acc, a_col_0_3, scalar_0_3);
          temp_acc = aie::mac(temp_acc, a_col_1_3, scalar_1_3);
          temp_acc = aie::mac(temp_acc, a_col_2_3, scalar_2_3);
          temp_acc = aie::mac(temp_acc, a_col_3_3, scalar_3_3);
          temp_acc = aie::mac(temp_acc, a_col_4_3, scalar_4_3);
          temp_acc = aie::mac(temp_acc, a_col_5_3, scalar_5_3);
          temp_acc = aie::mac(temp_acc, a_col_6_3, scalar_6_3);
          temp_acc = aie::mac(temp_acc, a_col_7_3, scalar_7_3);

        } // end of cc, 32 columns (manually unrolled);

        // c_accum = aie::mac(c_accum, c_local_accum.template to_vector<bf16>(),
        // a_scales); c_accum = aie::mac(c_accum, a_mins, (bf16)reduce_b);
        c_accum =
            aie::mac(c_accum, temp_acc.template to_vector<bf16>(), a_scales);
        c_accum = aie::mac(c_accum, a_mins, reduce_b);
        //
      } // end of sub_chunk, 256 columns;
    aie::vector<float, pr> c_accum_f32 = c_accum.template to_vector<float>();
    aie::store_v(c + row, c_accum_f32);
  }
}

#else // Q4_0: signed int4, scale-only (no per-group min), 3-arg form.
template <int M, int N>
inline void _qmm_q4k_bf16(const q4k_block_t *A, const bf16 *B, float *c) {
  static_assert(N == 256, "N must be = 256");
  static_assert(M == 32, "M must be = 32");
  constexpr int pr = 16;
  constexpr int pc = 8;
  const q4k_block_t *A_q4k = (q4k_block_t *)A;
  uint8_t *qs_ptr = (uint8_t *)(A_q4k->qs);
  float *c_ptr = const_cast<float *>(c);

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_RANGE(M / pr, M / pr)
  for (int row = 0; row < M; row += pr) {
    aie::accum<accfloat, pr> c_accum;
    c_accum.from_vector(aie::load_v<pr>(c + row));
    bf16 *it_B = const_cast<bf16 *>(B);
    bf16 *scales_ptr = const_cast<bf16 *>(A_q4k->scales + row);

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(N / 32, N / 32)
    for (int sub_chunk = 0; sub_chunk < N / 32; sub_chunk++) {
      aie::vector<bf16, 32> b_col = aie::load_v<32>(it_B);
      it_B += 32;

      aie::vector<bf16, pr> a_scales = aie::load_v<pr>(scales_ptr);
      scales_ptr += M;

      aie::accum<accfloat, pr> c_local_accum;
      c_local_accum.from_vector(aie::zeros<float, pr>());

      {
#if defined(__chess__)
        aie::vector<int4, pr * 8> a_cc_0 = aie::load_v<pr * 8>((int4 *)qs_ptr);
        qs_ptr += pr * 4;
        aie::vector<int4, pr * 8> a_cc_1 = aie::load_v<pr * 8>((int4 *)qs_ptr);
        qs_ptr += pr * 4;
        aie::vector<int4, pr * 8> a_cc_2 = aie::load_v<pr * 8>((int4 *)qs_ptr);
        qs_ptr += pr * 4;
        aie::vector<int4, pr * 8> a_cc_3 = aie::load_v<pr * 8>((int4 *)qs_ptr);
        qs_ptr += pr * 4;

        aie::vector<float, pr * 8> a_cc_f32_0 = aie::to_float(a_cc_0, 0);
        aie::vector<float, pr * 8> a_cc_f32_1 = aie::to_float(a_cc_1, 0);
        aie::vector<float, pr * 8> a_cc_f32_2 = aie::to_float(a_cc_2, 0);
        aie::vector<float, pr * 8> a_cc_f32_3 = aie::to_float(a_cc_3, 0);
#else
        // Peano's aie-api folds aie::to_float(vector<int4>) to zero (signed
        // int4 unsupported) -> the whole matmul gets DCE'd. So the nibbles are
        // loaded as uint4 and sign-corrected here, two ways:
        //
        //   Q4_SFIX_MODE 1 (default) -- XOR-BIAS. (u ^ 8) - 8 == q over the
        //     whole signed-int4 range, so one integer xor and one float
        //     subtract do the job. The xor runs on the PACKED bytes (0x88
        //     flips bit 3 of both nibbles at once) because peano's aie-api has
        //     no uint4 bit_xor -- aie::bit_xor on a vector<uint4> segfaults
        //     the frontend.
        //
        //   Q4_SFIX_MODE 0 -- the original compare/select (v >= 8 -> v - 16).
        //     Same values, but the select's operands keep four float<128>
        //     vectors and two broadcast constants live at once, which is far
        //     past the register file, and peano spills the difference.
        //
        // Measured, aie2p -O2, inner-loop body of this function:
        //
        //   mode      bundles   vlda   vst   vge   vsel
        //   0             355    226    23    32     32
        //   1             140     33     4     0      0
        //   (chess)       142      4     0     0      0
        //
        // The body runs 8x2 per 32x256 block, and the core issues ~1 bundle
        // per cycle, so those 215 bundles were 68% of the whole qwen2.5-3B
        // projection: mode 1 took its decode intercept from 81.10 ms to
        // 32.40 ms per token (measured, 36 layers, NPU2). See
        // Q4_SFIX_MODE 2 in the A/B note below for how that was attributed.
        //
        // Only this Q4_0 branch has a sign fix at all -- Q4NX nibbles are
        // unsigned -- so nothing outside the Q4_0 models is affected.
        // Q4_SFIX_MODE is defaulted at the top of this file.
        const aie::vector<float, pr * 8> k_eight =
            aie::broadcast<float, pr * 8>(8.0f);
#if Q4_SFIX_MODE == 0
        const aie::vector<float, pr * 8> k_sixteen =
            aie::broadcast<float, pr * 8>(16.0f);
#endif

#if Q4_SFIX_MODE == 0
        aie::vector<uint4, pr * 8> u_cc_0 =
            aie::load_v<pr * 8>((uint4 *)qs_ptr);
        qs_ptr += pr * 4;
        aie::vector<uint4, pr * 8> u_cc_1 =
            aie::load_v<pr * 8>((uint4 *)qs_ptr);
        qs_ptr += pr * 4;
        aie::vector<uint4, pr * 8> u_cc_2 =
            aie::load_v<pr * 8>((uint4 *)qs_ptr);
        qs_ptr += pr * 4;
        aie::vector<uint4, pr * 8> u_cc_3 =
            aie::load_v<pr * 8>((uint4 *)qs_ptr);
        qs_ptr += pr * 4;
        auto sfix = [&](aie::vector<uint4, pr * 8> uv) {
          aie::vector<float, pr * 8> f = aie::to_float(uv, 0);
          return aie::select(f, aie::sub(f, k_sixteen), aie::ge(f, k_eight));
        };
#else
        const aie::vector<uint8, pr * 4> k_88 =
            aie::broadcast<uint8, pr * 4>((uint8)0x88);
        auto ldx = [&]() {
          aie::vector<uint8, pr * 4> raw = aie::load_v<pr * 4>(qs_ptr);
          qs_ptr += pr * 4;
          return aie::bit_xor(raw, k_88).template cast_to<uint4>();
        };
        aie::vector<uint4, pr * 8> u_cc_0 = ldx();
        aie::vector<uint4, pr * 8> u_cc_1 = ldx();
        aie::vector<uint4, pr * 8> u_cc_2 = ldx();
        aie::vector<uint4, pr * 8> u_cc_3 = ldx();
#if Q4_SFIX_MODE == 2
        // A/B BOUND ONLY -- NUMERICALLY WRONG. Drops the -8 so the body's
        // static bundle count says what is left after the xor is paid for
        // (113 bundles). Never ship; objdump/bench only.
        auto sfix = [&](aie::vector<uint4, pr * 8> uv) {
          return aie::to_float(uv, 0);
        };
#else
        auto sfix = [&](aie::vector<uint4, pr * 8> uv) {
          return aie::sub(aie::to_float(uv, 0), k_eight);
        };
#endif
#endif
        aie::vector<float, pr * 8> a_cc_f32_0 = sfix(u_cc_0);
        aie::vector<float, pr * 8> a_cc_f32_1 = sfix(u_cc_1);
        aie::vector<float, pr * 8> a_cc_f32_2 = sfix(u_cc_2);
        aie::vector<float, pr * 8> a_cc_f32_3 = sfix(u_cc_3);
#endif

        aie::accum<accfloat, pr * 8> a_cc_f32_accum_0;
        a_cc_f32_accum_0.from_vector(a_cc_f32_0);
        aie::vector<bf16, pr * 8> a_cc_bf16_0 =
            a_cc_f32_accum_0.template to_vector<bf16>();
        aie::accum<accfloat, pr * 8> a_cc_f32_accum_1;
        a_cc_f32_accum_1.from_vector(a_cc_f32_1);
        aie::vector<bf16, pr * 8> a_cc_bf16_1 =
            a_cc_f32_accum_1.template to_vector<bf16>();
        aie::accum<accfloat, pr * 8> a_cc_f32_accum_2;
        a_cc_f32_accum_2.from_vector(a_cc_f32_2);
        aie::vector<bf16, pr * 8> a_cc_bf16_2 =
            a_cc_f32_accum_2.template to_vector<bf16>();
        aie::accum<accfloat, pr * 8> a_cc_f32_accum_3;
        a_cc_f32_accum_3.from_vector(a_cc_f32_3);
        aie::vector<bf16, pr * 8> a_cc_bf16_3 =
            a_cc_f32_accum_3.template to_vector<bf16>();

        aie::vector<bf16, pr> a_col_0_0 = a_cc_bf16_0.extract<pr>(0);
        aie::vector<bf16, pr> a_col_1_0 = a_cc_bf16_0.extract<pr>(1);
        aie::vector<bf16, pr> a_col_2_0 = a_cc_bf16_0.extract<pr>(2);
        aie::vector<bf16, pr> a_col_3_0 = a_cc_bf16_0.extract<pr>(3);
        aie::vector<bf16, pr> a_col_4_0 = a_cc_bf16_0.extract<pr>(4);
        aie::vector<bf16, pr> a_col_5_0 = a_cc_bf16_0.extract<pr>(5);
        aie::vector<bf16, pr> a_col_6_0 = a_cc_bf16_0.extract<pr>(6);
        aie::vector<bf16, pr> a_col_7_0 = a_cc_bf16_0.extract<pr>(7);
        aie::vector<bf16, pr> a_col_0_1 = a_cc_bf16_1.extract<pr>(0);
        aie::vector<bf16, pr> a_col_1_1 = a_cc_bf16_1.extract<pr>(1);
        aie::vector<bf16, pr> a_col_2_1 = a_cc_bf16_1.extract<pr>(2);
        aie::vector<bf16, pr> a_col_3_1 = a_cc_bf16_1.extract<pr>(3);
        aie::vector<bf16, pr> a_col_4_1 = a_cc_bf16_1.extract<pr>(4);
        aie::vector<bf16, pr> a_col_5_1 = a_cc_bf16_1.extract<pr>(5);
        aie::vector<bf16, pr> a_col_6_1 = a_cc_bf16_1.extract<pr>(6);
        aie::vector<bf16, pr> a_col_7_1 = a_cc_bf16_1.extract<pr>(7);
        aie::vector<bf16, pr> a_col_0_2 = a_cc_bf16_2.extract<pr>(0);
        aie::vector<bf16, pr> a_col_1_2 = a_cc_bf16_2.extract<pr>(1);
        aie::vector<bf16, pr> a_col_2_2 = a_cc_bf16_2.extract<pr>(2);
        aie::vector<bf16, pr> a_col_3_2 = a_cc_bf16_2.extract<pr>(3);
        aie::vector<bf16, pr> a_col_4_2 = a_cc_bf16_2.extract<pr>(4);
        aie::vector<bf16, pr> a_col_5_2 = a_cc_bf16_2.extract<pr>(5);
        aie::vector<bf16, pr> a_col_6_2 = a_cc_bf16_2.extract<pr>(6);
        aie::vector<bf16, pr> a_col_7_2 = a_cc_bf16_2.extract<pr>(7);
        aie::vector<bf16, pr> a_col_0_3 = a_cc_bf16_3.extract<pr>(0);
        aie::vector<bf16, pr> a_col_1_3 = a_cc_bf16_3.extract<pr>(1);
        aie::vector<bf16, pr> a_col_2_3 = a_cc_bf16_3.extract<pr>(2);
        aie::vector<bf16, pr> a_col_3_3 = a_cc_bf16_3.extract<pr>(3);
        aie::vector<bf16, pr> a_col_4_3 = a_cc_bf16_3.extract<pr>(4);
        aie::vector<bf16, pr> a_col_5_3 = a_cc_bf16_3.extract<pr>(5);
        aie::vector<bf16, pr> a_col_6_3 = a_cc_bf16_3.extract<pr>(6);
        aie::vector<bf16, pr> a_col_7_3 = a_cc_bf16_3.extract<pr>(7);

        bfloat16 scalar_0_0 = b_col.get(0);
        bfloat16 scalar_1_0 = b_col.get(1);
        bfloat16 scalar_2_0 = b_col.get(2);
        bfloat16 scalar_3_0 = b_col.get(3);
        bfloat16 scalar_4_0 = b_col.get(4);
        bfloat16 scalar_5_0 = b_col.get(5);
        bfloat16 scalar_6_0 = b_col.get(6);
        bfloat16 scalar_7_0 = b_col.get(7);
        bfloat16 scalar_0_1 = b_col.get(8);
        bfloat16 scalar_1_1 = b_col.get(9);
        bfloat16 scalar_2_1 = b_col.get(10);
        bfloat16 scalar_3_1 = b_col.get(11);
        bfloat16 scalar_4_1 = b_col.get(12);
        bfloat16 scalar_5_1 = b_col.get(13);
        bfloat16 scalar_6_1 = b_col.get(14);
        bfloat16 scalar_7_1 = b_col.get(15);
        bfloat16 scalar_0_2 = b_col.get(16);
        bfloat16 scalar_1_2 = b_col.get(17);
        bfloat16 scalar_2_2 = b_col.get(18);
        bfloat16 scalar_3_2 = b_col.get(19);
        bfloat16 scalar_4_2 = b_col.get(20);
        bfloat16 scalar_5_2 = b_col.get(21);
        bfloat16 scalar_6_2 = b_col.get(22);
        bfloat16 scalar_7_2 = b_col.get(23);
        bfloat16 scalar_0_3 = b_col.get(24);
        bfloat16 scalar_1_3 = b_col.get(25);
        bfloat16 scalar_2_3 = b_col.get(26);
        bfloat16 scalar_3_3 = b_col.get(27);
        bfloat16 scalar_4_3 = b_col.get(28);
        bfloat16 scalar_5_3 = b_col.get(29);
        bfloat16 scalar_6_3 = b_col.get(30);
        bfloat16 scalar_7_3 = b_col.get(31);

        c_local_accum = aie::mac(c_local_accum, a_col_0_0, scalar_0_0);
        c_local_accum = aie::mac(c_local_accum, a_col_1_0, scalar_1_0);
        c_local_accum = aie::mac(c_local_accum, a_col_2_0, scalar_2_0);
        c_local_accum = aie::mac(c_local_accum, a_col_3_0, scalar_3_0);
        c_local_accum = aie::mac(c_local_accum, a_col_4_0, scalar_4_0);
        c_local_accum = aie::mac(c_local_accum, a_col_5_0, scalar_5_0);
        c_local_accum = aie::mac(c_local_accum, a_col_6_0, scalar_6_0);
        c_local_accum = aie::mac(c_local_accum, a_col_7_0, scalar_7_0);
        c_local_accum = aie::mac(c_local_accum, a_col_0_1, scalar_0_1);
        c_local_accum = aie::mac(c_local_accum, a_col_1_1, scalar_1_1);
        c_local_accum = aie::mac(c_local_accum, a_col_2_1, scalar_2_1);
        c_local_accum = aie::mac(c_local_accum, a_col_3_1, scalar_3_1);
        c_local_accum = aie::mac(c_local_accum, a_col_4_1, scalar_4_1);
        c_local_accum = aie::mac(c_local_accum, a_col_5_1, scalar_5_1);
        c_local_accum = aie::mac(c_local_accum, a_col_6_1, scalar_6_1);
        c_local_accum = aie::mac(c_local_accum, a_col_7_1, scalar_7_1);
        c_local_accum = aie::mac(c_local_accum, a_col_0_2, scalar_0_2);
        c_local_accum = aie::mac(c_local_accum, a_col_1_2, scalar_1_2);
        c_local_accum = aie::mac(c_local_accum, a_col_2_2, scalar_2_2);
        c_local_accum = aie::mac(c_local_accum, a_col_3_2, scalar_3_2);
        c_local_accum = aie::mac(c_local_accum, a_col_4_2, scalar_4_2);
        c_local_accum = aie::mac(c_local_accum, a_col_5_2, scalar_5_2);
        c_local_accum = aie::mac(c_local_accum, a_col_6_2, scalar_6_2);
        c_local_accum = aie::mac(c_local_accum, a_col_7_2, scalar_7_2);
        c_local_accum = aie::mac(c_local_accum, a_col_0_3, scalar_0_3);
        c_local_accum = aie::mac(c_local_accum, a_col_1_3, scalar_1_3);
        c_local_accum = aie::mac(c_local_accum, a_col_2_3, scalar_2_3);
        c_local_accum = aie::mac(c_local_accum, a_col_3_3, scalar_3_3);
        c_local_accum = aie::mac(c_local_accum, a_col_4_3, scalar_4_3);
        c_local_accum = aie::mac(c_local_accum, a_col_5_3, scalar_5_3);
        c_local_accum = aie::mac(c_local_accum, a_col_6_3, scalar_6_3);
        c_local_accum = aie::mac(c_local_accum, a_col_7_3, scalar_7_3);
      } // end of cc, 32 columns (manually unrolled)
      c_accum =
          aie::mac(c_accum, c_local_accum.template to_vector<bf16>(), a_scales);
    } // end of sub_chunk, 256 columns
    aie::vector<float, pr> c_accum_f32 = c_accum.template to_vector<float>();
    aie::store_v(c_ptr, c_accum_f32);
    c_ptr += pr;
  }
}
#endif // Q4_0

#endif // __Q4_K_H__
