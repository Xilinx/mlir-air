// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
#include "aie_array_layout.h"
#include "aie_kernel_utils.h"
#include "lut_based_ops.h"

// Hot-path inlining (see attn_qk.cc): tuned functions + entry wrappers are
// always_inline'd so the .ll kernel is llvm-merged into AIR's per-block scf.for
// loop. Peano only, always built with -DDECODE_INLINE_ATTN.
// The Peano NOINLINE workaround is retired here too -- see attn_qk.cc for the
// defect history, the evidence that it no longer reproduces, and the numbers.
// Compiling with -DATTN_PEANO_NOINLINE restores it (the guard tests only
// whether the macro is defined, so the value is irrelevant and it must be a
// compile define, not a make variable).
#if defined(__chess__) || !defined(ATTN_PEANO_NOINLINE)
#define ATTN_HOT inline __attribute__((always_inline))
#else
#define ATTN_HOT __attribute__((noinline))
#endif
#define ATTN_ENTRY __attribute__((always_inline))

typedef float y_acc_dtype;
ATTN_HOT void calculate_l(bf16 *__restrict pS, float *__restrict c,
                          float *__restrict l);
template <unsigned colQ, unsigned r, unsigned s, unsigned t>
ATTN_HOT void attn_fv(bf16 *__restrict pS, bf16 *__restrict pV,
                      y_acc_dtype *__restrict pY, float *__restrict c);
template <unsigned N>
void scale_div_aie(bf16 *a, bf16 *o, float *l);
template <typename T_in, typename T_out, int N>
void passThrough_aie(T_in *restrict in0, T_out *restrict out);

constexpr int s_prod_lock = 4;
constexpr int s_cons_lock = 5;
constexpr int v_prod_lock = 2;
constexpr int v_cons_lock = 3;
constexpr int o_prod_lock = 0;
constexpr int o_cons_lock = 1;
// constexpr int l_prod_lock = 6;
constexpr int l_cons_lock = 7;

extern "C" {

void attn_kv(bf16 *__restrict s_ping, bf16 *__restrict s_pong,
             bf16 *__restrict v_ping, bf16 *__restrict v_pong,
             bf16 *__restrict o_ping, bf16 *__restrict o_pong, int *L) {
  aie_round_nearest_even();
  alignas(aie::vector_decl_align) static y_acc_dtype
      y[Q_HEADS_PADDED_PER_CU * DH] = {};
  alignas(aie::vector_decl_align) static bf16
      y_bf16[Q_HEADS_PADDED_PER_CU * DH] = {};
  alignas(aie::vector_decl_align) static float l[16] = {};
  const aie::vector<float, 16> zero = aie::broadcast<float, 16>(0);

  // Peano's CDO loader does not zero .bss; keep ping/pong selectors in .data.
  __attribute__((section(".data"))) static bool is_svc_ping = false;
  __attribute__((section(".data"))) static bool is_lo_ping = false;
  // Peano miscompiles read-modify-write on persisted .data bools across the
  // loop; keep ping/pong state in stack locals, sync to .data once at
  // entry/exit.
  bool svc = is_svc_ping;
  bool lo = is_lo_ping;
  zero_vectorized<y_acc_dtype, Q_HEADS_PADDED_PER_CU * DH>(y);
  aie::store_v(l, zero);
  _down_lock_acquire(l_cons_lock);
  AIE_LOCK_LOOP_CTR i = L[0];
  do {
    _down_lock_acquire(s_cons_lock);
    _lock_acquire(v_cons_lock);
    svc = !svc;
    bf16 *s = svc ? s_ping : s_pong;
    bf16 *v = svc ? v_ping : v_pong;
    float *c = (float *)(s + Q_HEADS_PADDED_PER_CU * 16);
    calculate_l(s, c, l);
    attn_fv<DH / 8, GQA_R, GQA_S, GQA_T>(s, v, y, c);
    _down_lock_release(s_prod_lock);
    _lock_release(v_prod_lock);
    i -= 16;
  } while (i > 0);
  passThrough_aie<y_acc_dtype, bf16, Q_HEADS_PADDED_PER_CU * DH>(y, y_bf16);
  lo = !lo;
  bf16 *o = lo ? o_ping : o_pong;
  is_svc_ping = svc;
  is_lo_ping = lo;
  _lock_acquire(o_prod_lock);
  scale_div_aie<Q_HEADS_PADDED_PER_CU * DH>(y_bf16, o, l);
  _lock_release(o_cons_lock);
}
}

template <typename T_in, typename T_out, int N>
__attribute__((noinline)) void passThrough_aie(T_in *restrict in0,
                                               T_out *restrict out) {

  const int vec_factor = 16;

  aie::vector<T_out, vec_factor> Out;

  const int F = N / vec_factor;
  for (int i = 0; i < F; i++)
    AIE_LOOP_FLATTEN {
      aie::accum<accfloat, vec_factor> c_acc_in;
      c_acc_in.from_vector(aie::load_v<vec_factor>(in0));
      aie::store_v(out, c_acc_in.template to_vector<T_out>());
      in0 += vec_factor;
      out += vec_factor;
    }
}

#if ATTN_IMPL == ATTN_IMPL_2x4x1
ATTN_HOT
void calculate_l(bf16 *__restrict pS, float *__restrict c,
                 float *__restrict l) {
  aie_round_nearest_even();
  using MMUL = aie::mmul<8, 8, 8, bf16, bf16, accfloat>;

  bf16 *__restrict pS1 = pS;

  aie::vector<bf16, 64> S_up = aie::load_v<64>(pS1);
  pS1 += 64;
  aie::vector<bf16, 64> S_down = aie::load_v<64>(pS1);

  aie::vector<bf16, 32> S10 = aie::filter_even(S_up, 8);
  aie::vector<bf16, 32> S11 = aie::filter_odd(S_up, 8);
  aie::vector<bf16, 32> S20 = aie::filter_even(S_down, 8);
  aie::vector<bf16, 32> S21 = aie::filter_odd(S_down, 8);

  aie::vector<bf16, MMUL::size_A> S0 = aie::concat(S10, S20);
  aie::vector<bf16, MMUL::size_A> S1 = aie::concat(S11, S21);
  aie::vector<bf16, MMUL::size_B> ones = aie::broadcast<bf16, MMUL::size_B>(1);

  aie::vector<bf16, MMUL::size_C> acc_C00 = aie::zeros<bf16, MMUL::size_C>();

  MMUL C00(acc_C00);

  C00.mac(S0, ones);
  C00.mac(S1, ones);

  aie::vector<float, MMUL::size_C> sumf = C00.template to_vector<float>();
  auto sum03 = aie::filter_even(sumf, 4);
  auto sum02 = aie::filter_even(sum03, 2);
  auto sum01 = aie::filter_even(sum02, 1);
  auto sum00 = aie::concat(sum01, sum01);
  aie::accum<accfloat, 16> sum;
  sum.from_vector(sum00);

  aie::vector<float, 8> c_float32 = aie::load_v<8>(c);
  aie::vector<float, 16> c_in = aie::concat(c_float32, c_float32);

  aie::vector<float, 8> l_float32 = aie::load_v<8>(l);
  aie::vector<float, 16> l_in = aie::concat(l_float32, l_float32);

  aie::accum<accfloat, 16> l_out;
  l_out = mac_elem_16_accuracy_safe(l_in, c_in, sum, 0, 0, 0);
  aie::store_v(l, l_out.template to_vector<float>());

  // bf16 sums[8] = {};
  // aie::vector<bf16, 16> s0 = aie::load_v<16>(pS);
  // aie::vector<bf16, 16> s1 = aie::load_v<16>(pS + 16);
  // aie::vector<bf16, 16> s2 = aie::load_v<16>(pS + 32);
  // aie::vector<bf16, 16> s3 = aie::load_v<16>(pS + 48);
  // aie::vector<bf16, 16> s4 = aie::load_v<16>(pS + 64);
  // aie::vector<bf16, 16> s5 = aie::load_v<16>(pS + 80);
  // aie::vector<bf16, 16> s6 = aie::load_v<16>(pS + 96);
  // aie::vector<bf16, 16> s7 = aie::load_v<16>(pS + 112);

  // sums[0] = aie::reduce_add(s0);
  // sums[1] = aie::reduce_add(s1);
  // sums[2] = aie::reduce_add(s2);
  // sums[3] = aie::reduce_add(s3);
  // sums[4] = aie::reduce_add(s4);
  // sums[5] = aie::reduce_add(s5);
  // sums[6] = aie::reduce_add(s6);
  // sums[7] = aie::reduce_add(s7);

  // aie::vector<bf16, 8> sum_bf16 = aie::load_v<8>(sums);
  // aie::vector<bf16, 16> sum_16 = aie::concat(sum_bf16, sum_bf16);
  // aie::accum<accfloat, 16> sum_float32_16;
  // sum_float32_16.from_vector(sum_16);

  // aie::vector<float, 8> c_float32 = aie::load_v<8>(c);
  // aie::vector<float, 16> c_in = aie::concat(c_float32, c_float32);
  // aie::vector<float, 8> l_float32 = aie::load_v<8>(l);
  // aie::vector<float, 16> l_in = aie::concat(l_float32, l_float32);

  // aie::accum<accfloat, 16> l_out;
  // l_out = mac_elem_16_accuracy_safe(l_in, c_in, sum_float32_16, 0, 0, 0);
  // aie::store_v(l, l_out.template to_vector<float>());
}

template <unsigned colQ, unsigned r, unsigned s, unsigned t>
ATTN_HOT void attn_fv(bf16 *__restrict pS, bf16 *__restrict pV,
                      y_acc_dtype *__restrict pY, float *__restrict c) {

  using MMUL = aie::mmul<r, s, t, bf16, bf16, accfloat>;

  // Peano: the natural j+=4 form keeps FOUR BFP16 mmul accumulators (Y00,Y01,
  // Y10,Y11) AND the 64-wide CORRECT vector live across the loop; peano spills
  // the accumulators and a cross-regfile spill/reload defect corrupts
  // iterations (the "even-head drop", same AIESpillSlotOptimization family).
  // Restructure so nothing spills: (1) apply the online-softmax correction y *=
  // CORRECT in a pure-vector pass (CORRECT is dead afterwards, freeing
  // registers); (2) mac pass with exactly ONE zero-pressure mmul accumulator
  // live at a time -- plane 0 (pV1) fully consumed to its even lanes before
  // plane 1 (pV2) to its odd lanes. Bit-identical to the chess path. Mirrors
  // ROCm/the reference_IRON#26/#25/#30.
  {
    aie::vector<bf16, 64> S_up = aie::load_v<64>(pS);
    aie::vector<bf16, 64> S_down = aie::load_v<64>(pS + 64);
    aie::vector<bf16, MMUL::size_A> S0 =
        aie::concat(aie::filter_even(S_up, 8), aie::filter_even(S_down, 8));
    aie::vector<bf16, MMUL::size_A> S1 =
        aie::concat(aie::filter_odd(S_up, 8), aie::filter_odd(S_down, 8));
    bf16 *__restrict pV1 = pV;
    bf16 *__restrict pV2 = pV + MMUL::size_B * colQ;
    // (1) correction pass (pure vector; CORRECT dies after this scope)
    {
      aie::vector<y_acc_dtype, 64> CORRECT = aie::concat(
          aie::broadcast<float, 8>(c[0]), aie::broadcast<float, 8>(c[1]),
          aie::broadcast<float, 8>(c[2]), aie::broadcast<float, 8>(c[3]),
          aie::broadcast<float, 8>(c[4]), aie::broadcast<float, 8>(c[5]),
          aie::broadcast<float, 8>(c[6]), aie::broadcast<float, 8>(c[7]));
      for (unsigned j = 0; j < colQ; j += 1) {
        y_acc_dtype *__restrict pYc = pY + j * MMUL::size_C;
        aie::store_v(pYc, aie::mul(CORRECT, aie::load_v<MMUL::size_C>(pYc))
                              .template to_vector<y_acc_dtype>());
      }
    }
    // (2) mac pass, one accumulator live at a time (plane0 even ++ plane1 odd)
    for (unsigned j = 0; j < colQ; j += 1) {
      y_acc_dtype *__restrict pYc = pY + j * MMUL::size_C;
      bf16 *__restrict pVa = pV1 + j * MMUL::size_B;
      bf16 *__restrict pVb = pV2 + j * MMUL::size_B;
      // Peano DROPS accumulator lane 0 when an MMUL is initialized directly
      // from a loaded float buffer (MMUL Y(load_v(pYc))) then mac'd (a
      // compile-time-undef codegen bug at -O1; layout-sensitive). Restructure:
      // mac from a ZERO-init accumulator (S.V only), then add the carried y
      // (yprev) as a plain vector add. Numerically identical (yprev + S.V) but
      // avoids the load-init construct that miscompiles. (bit-exact to chess.)
      aie::vector<y_acc_dtype, MMUL::size_C> yprev =
          aie::load_v<MMUL::size_C>(pYc);
      aie::vector<bf16, MMUL::size_C> zc = aie::zeros<bf16, MMUL::size_C>();
      MMUL Y0(zc);
      Y0.mac(S0, aie::load_v<MMUL::size_B>(pVa));
      Y0.mac(S1, aie::load_v<MMUL::size_B>(pVa + MMUL::size_B * colQ * 2));
      auto Yeven = aie::filter_even(
          aie::add(Y0.template to_vector<y_acc_dtype>(), yprev), 32);
      MMUL Y1(zc);
      Y1.mac(S0, aie::load_v<MMUL::size_B>(pVb));
      Y1.mac(S1, aie::load_v<MMUL::size_B>(pVb + MMUL::size_B * colQ * 2));
      auto Yodd = aie::filter_odd(
          aie::add(Y1.template to_vector<y_acc_dtype>(), yprev), 32);
      aie::store_v(pYc, aie::concat(Yeven, Yodd));
    }
  }
}

template <unsigned N>
__attribute__((noinline)) void scale_div_aie(bf16 *a, bf16 *o, float *l) {

  constexpr int vec_factor = 64;
  const int F = N / vec_factor;

  // Peano-safe scalar normalization. The vector path (8-way aie::concat of
  // per-head inv-l broadcasts + per-group extract<8>) miscompiles under Peano:
  // it corrupts the head-0/head-4 (extract index 0 and 4) 8-lane groups even
  // though the input a and l are bit-identical. This replicates the EXACT same
  // computation scalarly: within each 64-lane block, group (g,h) occupies lanes
  // [(g*4+h)*8 .. +8) and is scaled by inv-l of head (g*4+h).
  //
  // y/a is PADDED (Q_HEADS_PADDED_PER_CU=8 head slots per dim-tile, GQA pad at
  // the last slot of each group), but o is PACKED (Q_HEADS_PER_CU real heads),
  // matching the reference and the attnO gather (dim-tile stride Q_HEADS_PER_CU
  // *8). Read at the padded slot, write at the packed one. GQA pad=0 (1B) makes
  // packed == padded, so this is a no-op there.
  constexpr int o_tile = Q_HEADS_PER_CU * 8; // packed dim-tile stride in o
  bf16 invl[8];
  for (int h = 0; h < 8; h++)
    invl[h] = (bf16)getInvBf16(l[h]);
  for (int d = 0; d < F; d++) {
    bf16 *pa = a + d * vec_factor;
    bf16 *po = o + d * o_tile;
    for (int g = 0; g < KV_HEADS_PER_CU; g++) {
      for (int h = 0; h < Q_HEADS_PER_GROUP; h++) {
        int grp = g * Q_HEADS_PER_GROUP_PADDED + h; // padded slot in y
        int op = g * Q_HEADS_PER_GROUP + h;         // packed slot in o
        bf16 iv = invl[grp];
        for (int k = 0; k < 8; k++)
          po[op * 8 + k] = (bf16)((float)pa[grp * 8 + k] * (float)iv);
      }
    }
  }
}

#elif ATTN_IMPL == ATTN_IMPL_1x4x1

void calculate_l(bf16 *__restrict pS, float *__restrict c,
                 float *__restrict l) {
  aie_round_nearest_even();
  using MMUL = aie::mmul<GQA_R, GQA_S, GQA_T, bf16, bf16, accfloat>;
  bf16 *__restrict pS1 = pS;

  aie::vector<bf16, 32> S_up = aie::load_v<32>(pS1);
  pS1 += 32;

  aie::vector<bf16, 16> S10 = aie::filter_even(S_up, 8);
  aie::vector<bf16, 16> S11 = aie::filter_odd(S_up, 8);
  aie::vector<bf16, 16> zeros = aie::zeros<bf16, 16>();

  aie::vector<bf16, MMUL::size_A> S0 = aie::concat(S10, zeros);
  aie::vector<bf16, MMUL::size_A> S1 = aie::concat(S11, zeros);
  aie::vector<bf16, MMUL::size_B> ones = aie::broadcast<bf16, MMUL::size_B>(1);

  aie::vector<bf16, MMUL::size_C> acc_C00 = aie::zeros<bf16, MMUL::size_C>();

  MMUL C00(acc_C00);

  C00.mac(S0, ones);
  C00.mac(S1, ones);

  aie::vector<float, MMUL::size_C> sumf = C00.template to_vector<float>();
  auto sum03 = aie::filter_even(sumf, 4);
  auto sum02 = aie::filter_even(sum03, 2);
  auto sum01 = aie::filter_even(sum02, 1);

  float sum0 = sum01[0];
  float sum1 = sum01[1];

  l[0] = l[0] * c[0] + sum0;
  l[1] = l[1] * c[1] + sum1;
}

typedef float y_acc_dtype;
template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void attn_fv(bf16 *__restrict pS, bf16 *__restrict pV,
             y_acc_dtype *__restrict pY, float *__restrict c) {
  aie_round_nearest_even();

  using MMUL = aie::mmul<r, s, t, bf16, bf16, accfloat>;

  y_acc_dtype *__restrict pY1 = pY;
  bf16 *__restrict pS1 = pS;

  aie::vector<bf16, 32> S_up = aie::load_v<32>(pS1);
  pS1 += 32;

  aie::vector<bf16, 16> S10 = aie::filter_even(S_up, 8);
  aie::vector<bf16, 16> S11 = aie::filter_odd(S_up, 8);
  aie::vector<bf16, 16> zeros_vec = aie::zeros<bf16, 16>();

  aie::vector<bf16, MMUL::size_A> S0 = aie::concat(S10, zeros_vec);
  aie::vector<bf16, MMUL::size_A> S1 = aie::concat(S11, zeros_vec);

  aie::vector<y_acc_dtype, 8> c0 = aie::broadcast<float, 8>(c[0]);
  aie::vector<y_acc_dtype, 8> c1 = aie::broadcast<float, 8>(c[1]);
  aie::vector<y_acc_dtype, 8> c2 = aie::broadcast<float, 8>(c[2]);
  aie::vector<y_acc_dtype, 8> c3 = aie::broadcast<float, 8>(c[3]);
  aie::vector<y_acc_dtype, 32> CORRECT = aie::concat(c0, c1, c2, c3);

  bf16 *__restrict pV1 = pV;
  aie::vector<float, 16> zeros_float = aie::zeros<float, 16>();

  AIE_PREPARE_FOR_PIPELINING AIE_LOOP_RANGE(2) for (unsigned j = 0; j < colQ;
                                                    j += 4) {
    // v0
    bf16 *__restrict pV01 = pV1 + (j + 0) * MMUL::size_B;
    bf16 *__restrict pV02 = pV1 + (j + 1) * MMUL::size_B;

    aie::vector<bf16, MMUL::size_B> V00 = aie::load_v<MMUL::size_B>(pV01);
    pV01 += MMUL::size_B * colQ;
    aie::vector<bf16, MMUL::size_B> V01 = aie::load_v<MMUL::size_B>(pV02);
    pV02 += MMUL::size_B * colQ;

    aie::vector<y_acc_dtype, 16> acc_y00 = aie::load_v<16>(pY1);
    aie::vector<y_acc_dtype, 16> acc_y01 = aie::load_v<16>(pY1 + 16);
    aie::vector<y_acc_dtype, MMUL::size_C> acc_Y00 =
        aie::concat(acc_y00, zeros_float);
    aie::vector<y_acc_dtype, MMUL::size_C> acc_Y01 =
        aie::concat(acc_y01, zeros_float);

    aie::accum<accfloat, MMUL::size_C> ACC_Y00;
    aie::accum<accfloat, MMUL::size_C> ACC_Y01;
    ACC_Y00 = aie::mul(CORRECT, acc_Y00);
    ACC_Y01 = aie::mul(CORRECT, acc_Y01);

    MMUL Y00(ACC_Y00);
    MMUL Y01(ACC_Y01);

    Y00.mac(S0, V00);
    Y01.mac(S0, V01);

    V00 = aie::load_v<MMUL::size_B>(pV01);
    V01 = aie::load_v<MMUL::size_B>(pV02);

    Y00.mac(S1, V00);
    Y01.mac(S1, V01);

    auto Y0 = aie::filter_even(Y00.template to_vector<y_acc_dtype>(), 16);
    auto Y1 = aie::filter_even(Y01.template to_vector<y_acc_dtype>(), 16);

    aie::store_v(pY1, Y0);
    pY1 += 16;

    aie::store_v(pY1, Y1);
    pY1 += 16;

    bf16 *__restrict pV03 = pV1 + (j + 2) * MMUL::size_B;
    bf16 *__restrict pV04 = pV1 + (j + 3) * MMUL::size_B;

    aie::vector<bf16, MMUL::size_B> V02 = aie::load_v<MMUL::size_B>(pV03);
    pV03 += MMUL::size_B * colQ;
    aie::vector<bf16, MMUL::size_B> V03 = aie::load_v<MMUL::size_B>(pV04);
    pV04 += MMUL::size_B * colQ;

    aie::vector<y_acc_dtype, 16> acc_y02 = aie::load_v<16>(pY1);
    aie::vector<y_acc_dtype, 16> acc_y03 = aie::load_v<16>(pY1 + 16);
    aie::vector<y_acc_dtype, MMUL::size_C> acc_Y02 =
        aie::concat(acc_y02, zeros_float);
    aie::vector<y_acc_dtype, MMUL::size_C> acc_Y03 =
        aie::concat(acc_y03, zeros_float);

    aie::accum<accfloat, MMUL::size_C> ACC_Y02;
    aie::accum<accfloat, MMUL::size_C> ACC_Y03;
    ACC_Y02 = aie::mul(CORRECT, acc_Y02);
    ACC_Y03 = aie::mul(CORRECT, acc_Y03);

    MMUL Y02(ACC_Y02);
    MMUL Y03(ACC_Y03);

    Y02.mac(S0, V02);
    Y03.mac(S0, V03);

    V02 = aie::load_v<MMUL::size_B>(pV03);
    V03 = aie::load_v<MMUL::size_B>(pV04);

    Y02.mac(S1, V02);
    Y03.mac(S1, V03);

    auto Y2 = aie::filter_even(Y02.template to_vector<y_acc_dtype>(), 16);
    auto Y3 = aie::filter_even(Y03.template to_vector<y_acc_dtype>(), 16);

    aie::store_v(pY1, Y2);
    pY1 += 16;

    aie::store_v(pY1, Y3);
    pY1 += 16;
  }
}

template <unsigned N>
void scale_div_aie(bf16 *a, bf16 *o, float *l) {
  aie_round_nearest_even();

  constexpr int vec_factor = 16;
  const int F = N / vec_factor;

  bf16 *pA1 = a;
  bf16 *pO1 = o;
  bf16 l00 = (bf16)getInvBf16(l[0]);
  aie::vector<bf16, 8> L0 = aie::broadcast<bf16, 8>(l00);
  bf16 l01 = (bf16)getInvBf16(l[1]);
  aie::vector<bf16, 8> L1 = aie::broadcast<bf16, 8>(l01);

  auto L = aie::concat(L0, L1);

  for (int d = 0; d < N / vec_factor; d++) {
    aie::vector<bf16, vec_factor> A0 = aie::load_v<vec_factor>(pA1);
    aie::accum<accfloat, vec_factor> AL = aie::mul(A0, L);
    aie::vector<bf16, vec_factor> AL_bf16 = AL.template to_vector<bf16>();
    aie::store_v(pO1, AL_bf16);
    pO1 += vec_factor;
    pA1 += vec_factor;
  }
}

#elif ATTN_IMPL == ATTN_IMPL_1x8x1

#if !defined(__chess__)
__attribute__((noinline))
#endif
void calculate_l(bf16 *__restrict pS, float *__restrict c,
                 float *__restrict l) {
  aie_round_nearest_even();
  using MMUL = aie::mmul<GQA_R, GQA_S, GQA_T, bf16, bf16, accfloat>;
  bf16 *__restrict pS1 = pS;

  aie::vector<bf16, 64> S_up = aie::load_v<64>(pS1);
  pS1 += 64;
  aie::vector<bf16, 64> S_down = aie::load_v<64>(pS1);
  pS1 += 64;

  aie::vector<bf16, 32> S10 = aie::filter_even(S_up, 8);
  aie::vector<bf16, 32> S11 = aie::filter_odd(S_up, 8);
  aie::vector<bf16, 32> S20 = aie::filter_even(S_down, 8);
  aie::vector<bf16, 32> S21 = aie::filter_odd(S_down, 8);

  aie::vector<bf16, MMUL::size_A> S0 = aie::concat(S10, S20);
  aie::vector<bf16, MMUL::size_A> S1 = aie::concat(S11, S21);
  aie::vector<bf16, MMUL::size_B> ones = aie::broadcast<bf16, MMUL::size_B>(1);

  aie::vector<bf16, MMUL::size_C> acc_C00 = aie::zeros<bf16, MMUL::size_C>();

  MMUL C00(acc_C00);

  C00.mac(S0, ones);
  C00.mac(S1, ones);

  aie::vector<float, MMUL::size_C> sumf = C00.template to_vector<float>();
  auto sum03 = aie::filter_even(sumf, 4);
  auto sum02 = aie::filter_even(sum03, 2);
  auto sum01 = aie::filter_even(sum02, 1);
  auto sum00 = aie::concat(sum01, sum01);
  aie::accum<accfloat, 16> sum;
  sum.from_vector(sum00);

  aie::vector<float, 8> c_float32 = aie::load_v<8>(c);
  aie::vector<float, 16> c_in = aie::concat(c_float32, c_float32);

  aie::vector<float, 8> l_float32 = aie::load_v<8>(l);
  aie::vector<float, 16> l_in = aie::concat(l_float32, l_float32);

  aie::accum<accfloat, 16> l_out;
  l_out = mac_elem_16_accuracy_safe(l_in, c_in, sum, 0, 0, 0);
  aie::store_v(l, l_out.template to_vector<float>());
}

typedef float y_acc_dtype;
template <unsigned colQ, unsigned r, unsigned s, unsigned t>
#if !defined(__chess__)
__attribute__((noinline))
#endif
void attn_fv(bf16 *__restrict pS, bf16 *__restrict pV,
             y_acc_dtype *__restrict pY, float *__restrict c) {
  aie_round_nearest_even();

  using MMUL = aie::mmul<r, s, t, bf16, bf16, accfloat>;

  bf16 *__restrict pV1 = pV;

#if !defined(__chess__)
  // Peano: the natural j+=4 form keeps FOUR BFP16 mmul accumulators (Y00..Y03)
  // AND the 64-wide CORRECT vector live across the loop; peano spills the
  // accumulators and a cross-regfile spill/reload defect corrupts iterations
  // (bug B, same AIESpillSlotOptimization family). Restructure so nothing
  // spills: (1) apply the online-softmax correction y *= CORRECT in a
  // pure-vector pass (CORRECT is dead afterwards, freeing registers); (2) mac
  // pass: exactly ONE zero-pressure mmul accumulator seeded from the corrected
  // acc, full-store (1x8x1 has no plane split -> no radix filter).
  {
    aie::vector<bf16, 64> S_up = aie::load_v<64>(pS);
    aie::vector<bf16, 64> S_down = aie::load_v<64>(pS + 64);
    aie::vector<bf16, MMUL::size_A> S0 =
        aie::concat(aie::filter_even(S_up, 8), aie::filter_even(S_down, 8));
    aie::vector<bf16, MMUL::size_A> S1 =
        aie::concat(aie::filter_odd(S_up, 8), aie::filter_odd(S_down, 8));
    // (1) correction pass
    {
      aie::vector<y_acc_dtype, 64> CORRECT = aie::concat(
          aie::broadcast<float, 8>(c[0]), aie::broadcast<float, 8>(c[1]),
          aie::broadcast<float, 8>(c[2]), aie::broadcast<float, 8>(c[3]),
          aie::broadcast<float, 8>(c[4]), aie::broadcast<float, 8>(c[5]),
          aie::broadcast<float, 8>(c[6]), aie::broadcast<float, 8>(c[7]));
      for (unsigned j = 0; j < colQ; j += 1) {
        y_acc_dtype *__restrict pYc = pY + j * MMUL::size_C;
        aie::store_v(pYc, aie::mul(CORRECT, aie::load_v<MMUL::size_C>(pYc))
                              .template to_vector<y_acc_dtype>());
      }
    }
    // (2) mac pass, single accumulator
    for (unsigned j = 0; j < colQ; j += 1) {
      y_acc_dtype *__restrict pYc = pY + j * MMUL::size_C;
      bf16 *__restrict pVp = pV1 + j * MMUL::size_B;
      aie::vector<bf16, MMUL::size_B> Vp = aie::load_v<MMUL::size_B>(pVp);
      MMUL Y(aie::load_v<MMUL::size_C>(pYc));
      Y.mac(S0, Vp);
      Vp = aie::load_v<MMUL::size_B>(pVp + MMUL::size_B * colQ);
      Y.mac(S1, Vp);
      aie::store_v(pYc, Y.template to_vector<y_acc_dtype>());
    }
  }
#else
  y_acc_dtype *__restrict pY1 = pY;
  bf16 *__restrict pS1 = pS;

  aie::vector<bf16, 64> S_up = aie::load_v<64>(pS1);
  pS1 += 64;
  aie::vector<bf16, 64> S_down = aie::load_v<64>(pS1);
  pS1 += 64;

  aie::vector<bf16, 32> S10 = aie::filter_even(S_up, 8);
  aie::vector<bf16, 32> S11 = aie::filter_odd(S_up, 8);
  aie::vector<bf16, 32> S20 = aie::filter_even(S_down, 8);
  aie::vector<bf16, 32> S21 = aie::filter_odd(S_down, 8);

  aie::vector<bf16, MMUL::size_A> S0 = aie::concat(S10, S20);
  aie::vector<bf16, MMUL::size_A> S1 = aie::concat(S11, S21);

  aie::vector<y_acc_dtype, 8> c0 = aie::broadcast<float, 8>(c[0]);
  aie::vector<y_acc_dtype, 8> c1 = aie::broadcast<float, 8>(c[1]);
  aie::vector<y_acc_dtype, 8> c2 = aie::broadcast<float, 8>(c[2]);
  aie::vector<y_acc_dtype, 8> c3 = aie::broadcast<float, 8>(c[3]);
  aie::vector<y_acc_dtype, 8> c4 = aie::broadcast<float, 8>(c[4]);
  aie::vector<y_acc_dtype, 8> c5 = aie::broadcast<float, 8>(c[5]);
  aie::vector<y_acc_dtype, 8> c6 = aie::broadcast<float, 8>(c[6]);
  aie::vector<y_acc_dtype, 8> c7 = aie::broadcast<float, 8>(c[7]);
  aie::vector<y_acc_dtype, 64> CORRECT =
      aie::concat(c0, c1, c2, c3, c4, c5, c6, c7);

  for (unsigned j = 0; j < colQ; j += 4)
    chess_prepare_for_pipelining chess_loop_range(2, ) {
      // v0
      bf16 *__restrict pV01 = pV1 + (j + 0) * MMUL::size_B;
      bf16 *__restrict pV02 = pV1 + (j + 1) * MMUL::size_B;
      bf16 *__restrict pV03 = pV1 + (j + 2) * MMUL::size_B;
      bf16 *__restrict pV04 = pV1 + (j + 3) * MMUL::size_B;

      aie::vector<bf16, MMUL::size_B> V00 = aie::load_v<MMUL::size_B>(pV01);
      pV01 += MMUL::size_B * colQ;
      aie::vector<bf16, MMUL::size_B> V01 = aie::load_v<MMUL::size_B>(pV02);
      pV02 += MMUL::size_B * colQ;
      aie::vector<bf16, MMUL::size_B> V02 = aie::load_v<MMUL::size_B>(pV03);
      pV03 += MMUL::size_B * colQ;
      aie::vector<bf16, MMUL::size_B> V03 = aie::load_v<MMUL::size_B>(pV04);
      pV04 += MMUL::size_B * colQ;

      aie::vector<y_acc_dtype, MMUL::size_C> acc_y00 =
          aie::load_v<MMUL::size_C>(pY1);
      aie::vector<y_acc_dtype, MMUL::size_C> acc_y01 =
          aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C);
      aie::vector<y_acc_dtype, MMUL::size_C> acc_y02 =
          aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C * 2);
      aie::vector<y_acc_dtype, MMUL::size_C> acc_y03 =
          aie::load_v<MMUL::size_C>(pY1 + MMUL::size_C * 3);

      aie::accum<accfloat, MMUL::size_C> ACC_Y00;
      aie::accum<accfloat, MMUL::size_C> ACC_Y01;
      aie::accum<accfloat, MMUL::size_C> ACC_Y02;
      aie::accum<accfloat, MMUL::size_C> ACC_Y03;
      ACC_Y00 = aie::mul(CORRECT, acc_y00);
      ACC_Y01 = aie::mul(CORRECT, acc_y01);
      ACC_Y02 = aie::mul(CORRECT, acc_y02);
      ACC_Y03 = aie::mul(CORRECT, acc_y03);

      MMUL Y00(ACC_Y00);
      MMUL Y01(ACC_Y01);
      MMUL Y02(ACC_Y02);
      MMUL Y03(ACC_Y03);

      Y00.mac(S0, V00);
      Y01.mac(S0, V01);
      Y02.mac(S0, V02);
      Y03.mac(S0, V03);

      V00 = aie::load_v<MMUL::size_B>(pV01);
      V01 = aie::load_v<MMUL::size_B>(pV02);
      V02 = aie::load_v<MMUL::size_B>(pV03);
      V03 = aie::load_v<MMUL::size_B>(pV04);

      Y00.mac(S1, V00);
      Y01.mac(S1, V01);
      Y02.mac(S1, V02);
      Y03.mac(S1, V03);

      aie::store_v(pY1, Y00.template to_vector<y_acc_dtype>());
      pY1 += MMUL::size_C;
      aie::store_v(pY1, Y01.template to_vector<y_acc_dtype>());
      pY1 += MMUL::size_C;
      aie::store_v(pY1, Y02.template to_vector<y_acc_dtype>());
      pY1 += MMUL::size_C;
      aie::store_v(pY1, Y03.template to_vector<y_acc_dtype>());
      pY1 += MMUL::size_C;
    }
#endif
}

template <unsigned N>
void scale_div_aie(bf16 *a, bf16 *o, float *l) {
  aie_round_nearest_even();

  constexpr int vec_factor = 64;
  const int F = N / vec_factor;

  bf16 *pA1 = a;
  bf16 *pO1 = o;
  bf16 l00 = (bf16)getInvBf16(l[0]);
  aie::vector<bf16, 8> L0 = aie::broadcast<bf16, 8>(l00);
  bf16 l01 = (bf16)getInvBf16(l[1]);
  aie::vector<bf16, 8> L1 = aie::broadcast<bf16, 8>(l01);
  bf16 l02 = (bf16)getInvBf16(l[2]);
  aie::vector<bf16, 8> L2 = aie::broadcast<bf16, 8>(l02);
  bf16 l03 = (bf16)getInvBf16(l[3]);
  aie::vector<bf16, 8> L3 = aie::broadcast<bf16, 8>(l03);
  bf16 l04 = (bf16)getInvBf16(l[4]);
  aie::vector<bf16, 8> L4 = aie::broadcast<bf16, 8>(l04);
  bf16 l05 = (bf16)getInvBf16(l[5]);
  aie::vector<bf16, 8> L5 = aie::broadcast<bf16, 8>(l05);
  bf16 l06 = (bf16)getInvBf16(l[6]);
  aie::vector<bf16, 8> L6 = aie::broadcast<bf16, 8>(l06);
  bf16 l07 = (bf16)getInvBf16(l[7]);
  aie::vector<bf16, 8> L7 = aie::broadcast<bf16, 8>(l07);
  auto L = aie::concat(L0, L1, L2, L3, L4, L5, L6, L7);

  for (int d = 0; d < N / vec_factor; d++) {
    aie::vector<bf16, vec_factor> A0 = aie::load_v<vec_factor>(pA1);
    aie::accum<accfloat, vec_factor> AL = aie::mul(A0, L);
    aie::vector<bf16, vec_factor> AL_bf16 = AL.template to_vector<bf16>();
    for (int h = 0; h < Q_HEADS_PER_GROUP; h++) {
      aie::vector<bf16, 8> temp = AL_bf16.extract<8>(h);
      aie::store_v(pO1, temp);
      pO1 += 8;
    }
    pA1 += vec_factor;
  }
}

#endif

// ---------------------------------------------------------------------------
// Phase-1 lock-free single-shot wrapper (AIR dataflow handles sync).
// Reuses the reference's calculate_l + attn_fv + scale_div compute. Single
// 16-key block (rounds=1). s lives in this tile's L1 (written by the QK core
// via AIR shared-L1); v fed via S2MM; o drained via MM2S. L[0] = valid keys
// (<=16).
// ---------------------------------------------------------------------------
// Arg order puts the OUTPUT (o) as the LAST memref operand; s is then a
// non-last memref => AIR's shared-L1 classifier tags this call as the s
// consumer (read), pairing with the QK producer for a producer/consumer lock.
// Phase 3: RUNTIME L (RTP, by value as the last non-memref arg so o stays the
// last memref for the shared-L1 classifier). Loops rounds = ceil(L/16) blocks,
// accumulating S.V into persistent y and the softmax denominator into l, each
// rescaled by the per-block flash correction c. The masked tail keys of the
// last partial block have s=0 (from attn_qk), so they contribute nothing.
// Mirrors the reference's attn_kv do-while; L a multiple of 16 == Phase 2.
// scale_div once at end.
#define V_BLK (16 * KV_HEADS_PER_CU * DH) // 2048 bf16 per block
// Per-block score slot, PADDED to a multiple of 64 so every block stays aligned
// for the v64 loads in calculate_l / attn_fv. The logical size is 144 (128
// scores + 8 c floats); 144 is not a multiple of 64, so odd blocks would start
// misaligned -> garbage v64 loads (head0 sum corruption). Round up to 192.
#define SSZ_BLK (((Q_HEADS_PADDED_PER_CU * 16 + 16 + 63) / 64) * 64) // 192

// ---------------------------------------------------------------------------
// Phase 1 (the reference-faithful block-streamed, AIR-lowered locks): per-block
// KV. One 16-key block per call; the AIR herd loops rounds=ceil(L/16) and
// streams s/v blocks via depth-2 ping-pong channels (no in-kernel locks). The
// weighted-V accumulator y (float) and softmax denominator l are caller L1
// buffers that persist across the block loop (reset on blk==0). attn_kv_fin
// normalizes (o = y / l) after the last block. Reuses the reference's
// calculate_l + attn_fv + scale_div + passThrough verbatim.
extern "C" {
ATTN_ENTRY
void attn_kv_blk(bf16 *__restrict s_block, bf16 *__restrict v_block,
                 float *__restrict y_state, float *__restrict l_state, int blk,
                 int L) {
  aie_round_nearest_even();
  if (blk == 0) {
    zero_vectorized<y_acc_dtype, Q_HEADS_PADDED_PER_CU * DH>(y_state);
    const aie::vector<float, 16> zero = aie::broadcast<float, 16>(0);
    aie::store_v(l_state, zero);
  }
  // Block fully beyond L: skip (pairs with attn_qk_blk's skip -- s_block/c are
  // not produced for this block, so they must not be consumed). No V
  // contribution, matching the runtime-L path.
  if (L - blk * 16 <= 0)
    return;
  float *c = (float *)(s_block + Q_HEADS_PADDED_PER_CU * 16);
#ifndef SKIP_CALC_L
  calculate_l(s_block, c, l_state);
#endif
#ifndef SKIP_ATTN_FV
  attn_fv<DH / 8, GQA_R, GQA_S, GQA_T>(s_block, v_block, y_state, c);
#endif
}

void attn_kv_fin(float *__restrict y_state, float *__restrict l_state,
                 bf16 *__restrict o) {
  aie_round_nearest_even();
  alignas(aie::vector_decl_align) bf16 y_bf16[Q_HEADS_PADDED_PER_CU * DH];
  passThrough_aie<y_acc_dtype, bf16, Q_HEADS_PADDED_PER_CU * DH>(y_state,
                                                                 y_bf16);
  scale_div_aie<Q_HEADS_PADDED_PER_CU * DH>(y_bf16, o, l_state);
}

// Hybrid (LFM2) only: on a ShortConv wave this CU's slice of the mixer output
// takes the place of the attention result, so the o-gather memtile keeps ONE
// input channel and @xnorm keeps THREE producers. Four same-id producers on a
// convergent ring do not route, and a memtile is segment scope so it cannot
// pick between two sources per wave -- so the pick has to happen here, in a
// core, off the herd RTP.
//
// Overwrites `o` AFTER attn_kv_fin rather than replacing it: the block loop
// still has to run on both arms to drain the KV traffic the (wave-invariant)
// cache readback pushes every wave, and leaving that path untouched keeps the
// attention arm bit-identical.
//
// The permutation is the INVERSE of the un-interleave the @attnO put applies
// (sizes [QH, DH/8, 8], strides [8, QH*8, 1]). The mixer's output is already
// natural (q_head, dh), so writing it straight would come out scrambled by
// exactly that pattern; undoing it here is a fixed 512-element shuffle.
void conv_o_pass(bf16 *__restrict mix, bf16 *__restrict o, int cu, int arm) {
  aie_round_nearest_even();
  if (arm != 1)
    return;
  const int QH = Q_HEADS_PER_CU;
  const bf16 *__restrict src = mix + cu * (QH * DH);
  for (int h = 0; h < QH; h++)
    for (int d = 0; d < DH / 8; d++)
      for (int e = 0; e < 8; e++)
        o[h * 8 + d * QH * 8 + e] = src[h * DH + d * 8 + e];
}
}
