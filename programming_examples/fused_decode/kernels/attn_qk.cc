// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//

#include "aie_array_layout.h"
#include "aie_kernel_utils.h"
#include "lut_based_ops.h"

// Hot-path inlining: the tuned functions + entry wrappers are always_inline'd
// so the .ll kernel is llvm-merged into AIR's per-block scf.for loop (no
// per-block call ABI). Peano only, always built with -DDECODE_INLINE_ATTN (see
// the chatbot Makefile).
// Peano used to need the hot helpers (attn_fv/calculate_l/_attn_qk/update)
// NOINLINE: it spills the online-softmax attention's many live BFP16 matrix
// accumulators, and a cross-regfile spill/reload defect corrupted / deadlocked
// the inlined form, so each helper was kept a bounded-register function.
//
// That defect no longer reproduces on llvm-aie f4a72c27 (measured 2026-08-14).
// The inlined form still exercises the same path -- inlining raises ACC2048
// allocation pressure 35 -> 192 in this file and does emit cross-regfile
// spill/reload -- and is correct: `make verify` PASS 2/2 for llama32_1b_q4nx
// with NPU output bit-identical to the NOINLINE build (same divergence steps,
// token ids, top-5 sets).
//
// NOINLINE is not free. These helpers run once per 16-token KV block (~2048
// invocations per token at ctx 2048), and each call carries a prologue that
// saves/restores callee-saved accumulators. That is a per-block cost, so it
// shows up in the context SLOPE, not the intercept -- llama-3.2-1B decode:
//
//   NOINLINE       13.230 + 1.899*(ctx/1k) ms   17.03 ms @2k  58.8 tok/s
//   always_inline  13.267 + 1.278*(ctx/1k) ms   15.82 ms @2k  63.2 tok/s
//   (chess         13.050 + 1.249*(ctx/1k) ms   15.55 ms @2k  64.3 tok/s)
//
// No model is known to still need the workaround. All four users of this
// engine build inlined; accumulator pressure / stack spills per kernel are
//
//   MODEL_TYPE      attn_qk ACC2048/spills   attn_kv ACC2048/spills
//   LLAMA_3_2_1B          192 / 16                194 / 16   <- verify-gated
//   LLAMA_3_2_3B          149 /  8                193 / 16
//   GEMMA3_4B              61 /  2                234 /  8
//   QWEN2_5_3B             41 /  0                141 /  0
//
// LLAMA_3_2_1B ties for the most spills, so the gated config is the worst case
// for the defect this workaround guarded. 3B / gemma / qwen still want their
// own `make verify`. Compiling with -DATTN_PEANO_NOINLINE restores the old form
// if one bites -- the guard tests only whether the macro is DEFINED, so any
// value works and a bare make variable of that name will not reach the
// compiler; it has to be passed as a -D (e.g. via the kernel CXXFLAGS).
#if defined(__chess__) || !defined(ATTN_PEANO_NOINLINE)
#define ATTN_HOT inline __attribute__((always_inline))
#else
#define ATTN_HOT __attribute__((noinline))
#endif
#define ATTN_ENTRY __attribute__((always_inline))

ATTN_HOT aie::vector<bf16, 16> update(bf16 *m, float *c,
                                      aie::vector<bf16, 16> &out,
                                      aie::mask<16> &mask, bool &is_first);
template <unsigned colQ, unsigned r, unsigned s, unsigned t>
ATTN_HOT void _attn_qk(bf16 *__restrict pQ, bf16 *__restrict pK,
                       bf16 *__restrict pY, bf16 *__restrict m,
                       float *__restrict c, aie::mask<16> &mask,
                       bool &is_first);

// Running-max state for a new query. update_split() keeps the max as a pair:
// m_hi in the lower half, m_lo in the upper half. m_lo starts at 0 so the
// first block's comparison stays finite.
#if defined(HAS_QKV_BIAS) && ATTN_IMPL == ATTN_IMPL_1x8x1
#if ATTN_KV_PARTS > 1
#error "attn_merge reads one scaled max per head; update_split keeps a raw pair"
#endif
static inline aie::vector<bf16, 16> m_state_init() {
  return aie::concat(aie::broadcast<bf16, 8>(-0x1.FEp127f),
                     aie::zeros<bf16, 8>());
}
#else
static inline aie::vector<bf16, 16> m_state_init() {
  return aie::broadcast<bf16, 16>(-0x1.FEp127f);
}
#endif

constexpr int q_prod_lock = 0;
constexpr int q_cons_lock = 1;
constexpr int k_prod_lock = 2;
constexpr int k_cons_lock = 3;
constexpr int s_prod_lock = 4;
constexpr int s_cons_lock = 5;
// constexpr int l_prod_lock = 6;
constexpr int l_cons_lock = 7;

extern "C" {

void attn_qk(bf16 *__restrict q_ping, bf16 *__restrict q_pong,
             bf16 *__restrict k_ping, bf16 *__restrict k_pong,
             bf16 *__restrict s_ping, bf16 *__restrict s_pong, int *L) {
  aie_round_nearest_even();
  // Peano's CDO loader does not zero .bss; keep ping/pong selectors and
  // zero-init scratch in .data so they are actually zeroed at load.
  __attribute__((section(".data"))) static bool is_ql_ping = false;
  __attribute__((section(".data"))) static bool is_ksc_ping = false;
  alignas(aie::vector_decl_align) static __attribute__((section(".data")))
  bf16 m[16] = {};
  alignas(aie::vector_decl_align) static
      __attribute__((section(".data"))) float c_local[8] = {};
  // Peano: non-static so the broadcast/init runs each call (peano's non-atomic
  // static-init guard lives in unzeroed .bss and may skip the constructor).
  const aie::vector<int, 16> idx = aie::vector<int, 16>(
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

  // Peano miscompiles read-modify-write on persisted .data bools; keep
  // ping/pong state in stack locals, sync to .data once at entry/exit.
  bool ql = is_ql_ping;
  bool ksc = is_ksc_ping;
  _lock_acquire(q_cons_lock);
  ql = !ql;
  // bf16 *l = ql ? l_ping : l_pong;
  bf16 *q = ql ? q_ping : q_pong;
  // aie::store_v(l_local, zero);
  aie::store_v(m, m_state_init());
  int rounds = (L[0] + 15) / 16;
  _lock_release(l_cons_lock);
  AIE_LOCK_LOOP_CTR i = L[0];
  do {
    _lock_acquire(k_cons_lock);
    _lock_acquire(s_prod_lock);
    ksc = !ksc;
    bf16 *k = ksc ? k_ping : k_pong;
    bf16 *s = ksc ? s_ping : s_pong;
    float *c = (float *)(s + Q_HEADS_PADDED_PER_CU * 16);
    bool is_first = (i == L[0]);
    aie::mask<16> mask = aie::le(idx, (i < 16) ? i : 16);
    _attn_qk<DH / 8, GQA_R, GQA_S, GQA_T>(q, k, s, m, c_local, mask, is_first);
    aie::vector<float, 8> c_vec = aie::load_v<8>(c_local);
    aie::store_v(c, c_vec);
    _lock_release(k_prod_lock);
    _lock_release(s_cons_lock);
    i -= 16;
  } while (i > 0);
  _lock_release(q_prod_lock);
  is_ql_ping = ql;
  is_ksc_ping = ksc;
}
}

ATTN_HOT
aie::vector<bf16, 16> update(bf16 *m, float *c, aie::vector<bf16, 16> &out,
                             aie::mask<16> &mask, bool &is_first) {
  aie::vector<bf16, 16> masked_vm =
      aie::select((bf16)(-0x1.FEp127f), out, mask);
  bf16 vm = aie::reduce_max(masked_vm);
  bf16 vmax = aie::max(vm, (bf16)*m);
  aie::vector<bf16, 16> Vecsub = aie::sub(out, vmax);

  constexpr int min_clamp = -87.0f;
  constexpr int max_clamp = 88.0f;
  aie::vector<bf16, 16> min_value = aie::broadcast<bf16, 16>(min_clamp);
  aie::vector<bf16, 16> max_value = aie::broadcast<bf16, 16>(max_clamp);

  aie::vector<bf16, 16> Vec = aie::clamp(Vecsub, min_value, max_value);
  aie::accum<accfloat, 16> Outvec = getExpBf16(Vec);
  Vec = Outvec.template to_vector<bf16>(0);
  Vec = aie::select((bf16)0, Vec, mask);

  bf16 correct = aie::sub((bf16)*m, vmax);
  aie::vector<bf16, 16> correct_vec = aie::broadcast<bf16, 16>(correct);
  correct_vec = aie::clamp(correct_vec, min_value, max_value);
  aie::accum<accfloat, 16> correct_acc = getExpBf16(correct_vec);
  aie::vector<float, 16> correct_float =
      correct_acc.template to_vector<float>();

  *m = (float)vmax;
  *c = (float)correct_float.get(0);
  return Vec;
}

#if ATTN_IMPL == ATTN_IMPL_2x4x1

template <unsigned colQ, unsigned r, unsigned s, unsigned t>
ATTN_HOT void _attn_qk(bf16 *__restrict pQ, bf16 *__restrict pK,
                       bfloat16 *__restrict pY, bf16 *__restrict m,
                       float *__restrict c, aie::mask<16> &mask,
                       bool &is_first) {

  using MMUL = aie::mmul<r, s, t, bf16, bf16, accfloat>;

  bfloat16 *__restrict pY1 = pY;
  // g times K
  bf16 *__restrict pK1 = pK;
  bf16 *__restrict pK2 = pK + 1 * MMUL::size_B;

  {
    bf16 *__restrict pQ1 = pQ;
    aie::vector<bf16, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pQ1);
    pQ1 += MMUL::size_A;
    aie::vector<bf16, MMUL::size_B> B00 = aie::load_v<MMUL::size_B>(pK1);
    aie::vector<bf16, MMUL::size_B> B0 = aie::transpose(B00, 8, 8);
    pK1 += MMUL::size_B * 2;
    aie::vector<bf16, MMUL::size_B> B01 = aie::load_v<MMUL::size_B>(pK2);
    aie::vector<bf16, MMUL::size_B> B1 = aie::transpose(B01, 8, 8);
    pK2 += MMUL::size_B * 2;

    aie::vector<bf16, MMUL::size_C> acc_C00 = aie::zeros<bf16, MMUL::size_C>();
    aie::vector<bf16, MMUL::size_C> acc_C01 = aie::zeros<bf16, MMUL::size_C>();

    MMUL C00(acc_C00);
    MMUL C01(acc_C01);

    C00.mac(A0, B0);
    C01.mac(A0, B1);

    AIE_PREPARE_FOR_PIPELINING AIE_LOOP_RANGE(7) for (unsigned i = 1; i < colQ;
                                                      ++i) {
      A0 = aie::load_v<MMUL::size_A>(pQ1);
      pQ1 += MMUL::size_A;

      B00 = aie::load_v<MMUL::size_B>(pK1);
      B0 = aie::transpose(B00, 8, 8);
      pK1 += MMUL::size_B * 2;
      B01 = aie::load_v<MMUL::size_B>(pK2);
      B1 = aie::transpose(B01, 8, 8);
      pK2 += MMUL::size_B * 2;

      C00.mac(A0, B0);
      C01.mac(A0, B1);
    }
    auto mout0 = aie::interleave_zip(C00.template to_vector<bf16>(),
                                     C01.template to_vector<bf16>(), 8);

    aie::vector<bf16, 32> mout2, mout3;

    mout2 = aie::filter_even(mout0.first, 32);
    mout3 = aie::filter_odd(mout0.first, 32);

    mout2 = aie::mul(mout2, (bf16)ATTN_SCALE).template to_vector<bf16>(0);
    mout3 = aie::mul(mout3, (bf16)ATTN_SCALE).template to_vector<bf16>(0);

    aie::vector<bf16, 16> out[4];
    out[0] = aie::filter_even(mout2, 16);
    out[1] = aie::filter_odd(mout2, 16);
    out[2] = aie::filter_even(mout3, 16);
    out[3] = aie::filter_odd(mout3, 16);

    AIE_LOOP_UNROLL_FULL
    for (int h = 0; h < Q_HEADS_PER_GROUP; h++) {
      aie::vector<bf16, 16> vec = update(m + h, c + h, out[h], mask, is_first);
      aie::store_v(pY1, vec);
      pY1 += 16;
    }
    pY1 += 16 * ATTN_GROUPS_PADDING;
  }
  {
    bf16 *__restrict pQ1 = pQ;
    aie::vector<bf16, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pQ1);
    pQ1 += MMUL::size_A;
    aie::vector<bf16, MMUL::size_B> B00 = aie::load_v<MMUL::size_B>(pK1);
    aie::vector<bf16, MMUL::size_B> B0 = aie::transpose(B00, 8, 8);
    pK1 += MMUL::size_B * 2;
    aie::vector<bf16, MMUL::size_B> B01 = aie::load_v<MMUL::size_B>(pK2);
    aie::vector<bf16, MMUL::size_B> B1 = aie::transpose(B01, 8, 8);
    pK2 += MMUL::size_B * 2;

    aie::vector<bf16, MMUL::size_C> acc_C00 = aie::zeros<bf16, MMUL::size_C>();
    aie::vector<bf16, MMUL::size_C> acc_C01 = aie::zeros<bf16, MMUL::size_C>();

    MMUL C00(acc_C00);
    MMUL C01(acc_C01);

    C00.mac(A0, B0);
    C01.mac(A0, B1);

    AIE_PREPARE_FOR_PIPELINING AIE_LOOP_RANGE(7) for (unsigned i = 1; i < colQ;
                                                      ++i) {
      A0 = aie::load_v<MMUL::size_A>(pQ1);
      pQ1 += MMUL::size_A;

      B00 = aie::load_v<MMUL::size_B>(pK1);
      B0 = aie::transpose(B00, 8, 8);
      pK1 += MMUL::size_B * 2;
      B01 = aie::load_v<MMUL::size_B>(pK2);
      B1 = aie::transpose(B01, 8, 8);
      pK2 += MMUL::size_B * 2;

      C00.mac(A0, B0);
      C01.mac(A0, B1);
    }
    auto mout0 = aie::interleave_zip(C00.template to_vector<bf16>(),
                                     C01.template to_vector<bf16>(), 8);

    aie::vector<bf16, 32> mout2, mout3;
    mout2 = aie::filter_even(mout0.second, 32);
    mout3 = aie::filter_odd(mout0.second, 32);
    mout2 = aie::mul(mout2, (bf16)ATTN_SCALE).template to_vector<bf16>(0);
    mout3 = aie::mul(mout3, (bf16)ATTN_SCALE).template to_vector<bf16>(0);

    aie::vector<bf16, 16> out[4];
    out[0] = aie::filter_even(mout2, 16);
    out[1] = aie::filter_odd(mout2, 16);
    out[2] = aie::filter_even(mout3, 16);
    out[3] = aie::filter_odd(mout3, 16);
    AIE_LOOP_UNROLL_FULL
    for (int h = 0; h < Q_HEADS_PER_GROUP; h++) {
      aie::vector<bf16, 16> vec =
          update(m + 4 + h, c + 4 + h, out[h], mask, is_first);
      aie::store_v(pY1, vec);
      pY1 += 16;
    }
    pY1 += 16 * ATTN_GROUPS_PADDING;
  }
}

#elif ATTN_IMPL == ATTN_IMPL_1x4x1

template <unsigned colQ, unsigned r, unsigned s, unsigned t>
void _attn_qk(bf16 *__restrict pQ, bf16 *__restrict pK, bfloat16 *__restrict pY,
              bf16 *__restrict m, float *__restrict c, aie::mask<16> &mask,
              bool &is_first) {

  using MMUL = aie::mmul<r, s, t, bf16, bf16, accfloat>;

  bfloat16 *__restrict pY1 = pY;
  // g times K
  bf16 *__restrict pK1 = pK;
  bf16 *__restrict pK2 = pK + 1 * MMUL::size_B;
  {
    bf16 *__restrict pQ1 = pQ;
    aie::vector<bf16, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pQ1);
    pQ1 += MMUL::size_A;
    aie::vector<bf16, MMUL::size_B> B00 = aie::load_v<MMUL::size_B>(pK1);
    aie::vector<bf16, MMUL::size_B> B0 = aie::transpose(B00, 8, 8);
    pK1 += MMUL::size_B * 2;
    aie::vector<bf16, MMUL::size_B> B01 = aie::load_v<MMUL::size_B>(pK2);
    aie::vector<bf16, MMUL::size_B> B1 = aie::transpose(B01, 8, 8);
    pK2 += MMUL::size_B * 2;

    aie::vector<bf16, MMUL::size_C> acc_C00 = aie::zeros<bf16, MMUL::size_C>();
    aie::vector<bf16, MMUL::size_C> acc_C01 = aie::zeros<bf16, MMUL::size_C>();

    MMUL C00(acc_C00);
    MMUL C01(acc_C01);

    C00.mac(A0, B0);
    C01.mac(A0, B1);

    AIE_PREPARE_FOR_PIPELINING AIE_LOOP_RANGE(7) for (unsigned i = 1; i < colQ;
                                                      ++i) {
      A0 = aie::load_v<MMUL::size_A>(pQ1);
      pQ1 += MMUL::size_A;

      B00 = aie::load_v<MMUL::size_B>(pK1);
      B0 = aie::transpose(B00, 8, 8);
      pK1 += MMUL::size_B * 2;
      B01 = aie::load_v<MMUL::size_B>(pK2);
      B1 = aie::transpose(B01, 8, 8);
      pK2 += MMUL::size_B * 2;

      C00.mac(A0, B0);
      C01.mac(A0, B1);
    }
    // C is r x keys with one q head per row, so C00/C01 (key halves 0-7 and
    // 8-15) hold all GQA_R heads. interleave_zip by 8 gives .first = heads 0,1
    // and .second = heads 2,3, each as [k0-7, k8-15] pairs. This used to drop
    // .second on the floor, so heads 2 and 3 got no score at all -- their
    // softmax denominator came out zero and attn_kv then produced nothing for
    // them. Every 1x4x1 model has Q_HEADS_PADDED_PER_CU == 4, so that was half
    // the heads on both sides of the score buffer.
    static_assert(Q_HEADS_PADDED_PER_CU == 4,
                  "1x4x1 _attn_qk emits four heads per CU from one 4-row mmul");
    auto mout0 = aie::interleave_zip(C00.template to_vector<bf16>(),
                                     C01.template to_vector<bf16>(), 8);

    aie::vector<bf16, 16> heads[4];
    heads[0] = aie::filter_even(mout0.first, 16);
    heads[1] = aie::filter_odd(mout0.first, 16);
    heads[2] = aie::filter_even(mout0.second, 16);
    heads[3] = aie::filter_odd(mout0.second, 16);

    AIE_LOOP_UNROLL_FULL
    for (int h = 0; h < Q_HEADS_PADDED_PER_CU; h++) {
      heads[h] =
          aie::mul(heads[h], (bf16)ATTN_SCALE).template to_vector<bf16>(0);
      aie::store_v(pY1, update(m + h, c + h, heads[h], mask, is_first));
      pY1 += 16;
    }
  }
}

#elif ATTN_IMPL == ATTN_IMPL_1x8x1

#ifdef HAS_QKV_BIAS
// q/k projection biases (Qwen2.5) can put raw scores in the tens of thousands,
// where a bf16 step is coarser than the score differences the softmax depends
// on. So the scores are not rounded before the running max comes off: each raw
// block accumulator is split into hi (rounded to bf16) and lo (the rest), both
// go through the same shuffles into heads, and update_split() takes the offset
// from the max as (hi - max) + lo. hi - max is exact for every key within a
// factor of two of the max, which are the keys that carry weight.
//
// The running max is the pair (m_hi, m_lo), the largest hi + lo so far, so
// every exponent is at most 0 as in update(). With a max of hi alone, the top
// key's exponent would be lo * ATTN_SCALE, which grows with the bf16 step of
// the scores; at large scores the exponentials overflow the softmax sum and
// the output is NaN. m_lo sits in the upper half of the 16-slot m state, which
// one CU's heads do not use.
static_assert(Q_HEADS_PADDED_PER_CU <= 8,
              "update_split keeps m_lo in m[8 + h]");
ATTN_HOT aie::vector<bf16, 16> update_split(bf16 *m_hi, bf16 *m_lo, float *c,
                                            aie::vector<bf16, 16> &hi,
                                            aie::vector<bf16, 16> &lo,
                                            aie::mask<16> &mask) {
  aie::vector<bf16, 16> masked_vm = aie::select((bf16)(-0x1.FEp127f), hi, mask);
  bf16 vmax = aie::reduce_max(masked_vm);
  // The block's max as the pair (vmax, dmax). The top key has hi == vmax, so
  // its offset from vmax is exactly its lo.
  aie::vector<bf16, 16> d = aie::add(aie::sub(hi, vmax), lo);
  bf16 dmax = aie::reduce_max(aie::select((bf16)(-0x1.FEp127f), d, mask));

  // The block raises the running max if (vmax - m_hi) + (dmax - m_lo) > 0.
  // The first difference is exact when the two are close and the second is
  // small, so the sign can be trusted.
  bf16 rise = aie::add(aie::sub(vmax, *m_hi), aie::sub(dmax, *m_lo));
  bool raised = (__builtin_bit_cast(uint16_t, rise) & 0x8000) == 0 &&
                __builtin_bit_cast(uint16_t, rise) != 0;
  bf16 n_hi = raised ? vmax : *m_hi;
  bf16 n_lo = raised ? dmax : *m_lo;

  aie::vector<bf16, 16> Vecsub =
      aie::add(aie::sub(hi, n_hi), aie::sub(lo, n_lo));
  Vecsub = aie::mul(Vecsub, (bf16)ATTN_SCALE).template to_vector<bf16>();

  constexpr int min_clamp = -87.0f;
  constexpr int max_clamp = 88.0f;
  aie::vector<bf16, 16> min_value = aie::broadcast<bf16, 16>(min_clamp);
  aie::vector<bf16, 16> max_value = aie::broadcast<bf16, 16>(max_clamp);

  aie::vector<bf16, 16> Vec = aie::clamp(Vecsub, min_value, max_value);
  aie::accum<accfloat, 16> Outvec = getExpBf16(Vec);
  Vec = Outvec.template to_vector<bf16>(0);
  Vec = aie::select((bf16)0, Vec, mask);

  // Most blocks leave the max where it was; their correction is exactly 1,
  // which also lets attn_kv's lazy rescale skip them.
  float cf = 1.0f;
  if (raised) {
    bf16 correct = aie::add(aie::sub(*m_hi, n_hi), aie::sub(*m_lo, n_lo));
    aie::vector<bf16, 16> correct_vec =
        aie::mul(aie::broadcast<bf16, 16>(correct), (bf16)ATTN_SCALE)
            .template to_vector<bf16>();
    correct_vec = aie::clamp(correct_vec, min_value, max_value);
    aie::accum<accfloat, 16> e = getExpBf16(correct_vec);
    cf = e.to_vector<float>().get(0);
  }

  *m_hi = n_hi;
  *m_lo = n_lo;
  *c = cf;
  return Vec;
}
#endif

template <unsigned colQ, unsigned r, unsigned s, unsigned t>
ATTN_HOT void _attn_qk(bf16 *__restrict pQ, bf16 *__restrict pK,
                       bfloat16 *__restrict pY, bf16 *__restrict m,
                       float *__restrict c, aie::mask<16> &mask,
                       bool &is_first) {

  using MMUL = aie::mmul<r, s, t, bf16, bf16, accfloat>;

  bfloat16 *__restrict pY1 = pY;
  // g times K
  bf16 *__restrict pK1 = pK;
  bf16 *__restrict pK2 = pK + 1 * MMUL::size_B;
  {
    bf16 *__restrict pQ1 = pQ;
    aie::vector<bf16, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pQ1);
    pQ1 += MMUL::size_A;
    aie::vector<bf16, MMUL::size_B> B00 = aie::load_v<MMUL::size_B>(pK1);
    aie::vector<bf16, MMUL::size_B> B0 = aie::transpose(B00, 8, 8);
    pK1 += MMUL::size_B * 2;
    aie::vector<bf16, MMUL::size_B> B01 = aie::load_v<MMUL::size_B>(pK2);
    aie::vector<bf16, MMUL::size_B> B1 = aie::transpose(B01, 8, 8);
    pK2 += MMUL::size_B * 2;

    aie::vector<bf16, MMUL::size_C> acc_C00 = aie::zeros<bf16, MMUL::size_C>();
    aie::vector<bf16, MMUL::size_C> acc_C01 = aie::zeros<bf16, MMUL::size_C>();

    MMUL C00(acc_C00);
    MMUL C01(acc_C01);

    C00.mac(A0, B0);
    C01.mac(A0, B1);

    AIE_PREPARE_FOR_PIPELINING AIE_LOOP_RANGE(7) for (unsigned i = 1; i < colQ;
                                                      ++i) {
      A0 = aie::load_v<MMUL::size_A>(pQ1);
      pQ1 += MMUL::size_A;

      B00 = aie::load_v<MMUL::size_B>(pK1);
      B0 = aie::transpose(B00, 8, 8);
      pK1 += MMUL::size_B * 2;
      B01 = aie::load_v<MMUL::size_B>(pK2);
      B1 = aie::transpose(B01, 8, 8);
      pK2 += MMUL::size_B * 2;

      C00.mac(A0, B0);
      C01.mac(A0, B1);
    }
#ifdef HAS_QKV_BIAS
    auto a00 = C00.to_accum(), a01 = C01.to_accum();
    aie::vector<bf16, MMUL::size_C> h00 = a00.template to_vector<bf16>();
    aie::vector<bf16, MMUL::size_C> h01 = a01.template to_vector<bf16>();
    aie::vector<bf16, MMUL::size_C> l00 =
        aie::sub(a00, h00).template to_vector<bf16>();
    aie::vector<bf16, MMUL::size_C> l01 =
        aie::sub(a01, h01).template to_vector<bf16>();
    auto hz = aie::interleave_zip(h00, h01, 8);
    auto lz = aie::interleave_zip(l00, l01, 8);

    aie::vector<bf16, 16> hi[8], lo[8];
    for (int i = 0; i < 4; i++) {
      hi[i] = hz.first.template extract<16>(i);
      hi[4 + i] = hz.second.template extract<16>(i);
      lo[i] = lz.first.template extract<16>(i);
      lo[4 + i] = lz.second.template extract<16>(i);
    }

#else
    auto mout0 = aie::interleave_zip(C00.template to_vector<bf16>(),
                                     C01.template to_vector<bf16>(), 8);

    aie::vector<bf16, 64> mout2 =
        aie::mul(mout0.first, (bf16)ATTN_SCALE).template to_vector<bf16>(0);
    aie::vector<bf16, 64> mout3 =
        aie::mul(mout0.second, (bf16)ATTN_SCALE).template to_vector<bf16>(0);

    aie::vector<bf16, 16> out[8];

    out[0] = mout2.extract<16>(0);
    out[1] = mout2.extract<16>(1);
    out[2] = mout2.extract<16>(2);
    out[3] = mout2.extract<16>(3);
    out[4] = mout3.extract<16>(0);
    out[5] = mout3.extract<16>(1);
    out[6] = mout3.extract<16>(2);
    out[7] = mout3.extract<16>(3);
#endif

    for (int h = 0; h < Q_HEADS_PER_CU; h++) {
#ifdef HAS_QKV_BIAS
      aie::vector<bf16, 16> vec =
          update_split(m + h, m + 8 + h, c + h, hi[h], lo[h], mask);
#else
      aie::vector<bf16, 16> vec = update(m + h, c + h, out[h], mask, is_first);
#endif
      aie::store_v(pY1, vec);
      pY1 += 16;
    }
    pY1 += ATTN_GROUPS_PADDING * 16;
  }
}

#endif

// ---------------------------------------------------------------------------
// Lock-free wrapper (AIR dataflow handles sync). Reuses the reference's
// _attn_qk + update() compute. Writes scores+correction into s (which lives on
// the KV tile's L1 via AIR shared-L1). s is the LAST memref operand so AIR's
// shared-L1 classifier (last memref = producer) tags this call as the s
// producer. Phase 3c: q and k are SEPARATE buffers, both delivered by the
// memtile (k reordered from natural [key,kvh,dh], q linear), so no combined-qk
// split.
// ---------------------------------------------------------------------------
// Phase 3: RUNTIME sequence length L (RTP, passed by value as the last,
// non-memref arg so the shared-L1 "last memref = producer" convention still
// tags s). Loops rounds = ceil(L/16) 16-key blocks of flash online softmax with
// a running max m carried across blocks; the LAST block uses a partial mask
// le(idx, rem) for the L%16 tail keys (mirrors the reference's do-while: i=L;
// mask=le(idx, i<16?i:16); i-=16). L a multiple of 16 reproduces Phase 2.
#define K_BLK (16 * KV_HEADS_PER_CU * DH) // 16 keys * (2 kv * 64) = 2048 bf16
// Per-block score slot padded to a multiple of 64 (alignment for v64 loads in
// attn_kv); logical 144 (128 scores + 8 c floats) rounded up to 192.
#define SSZ_BLK (((Q_HEADS_PADDED_PER_CU * 16 + 16 + 63) / 64) * 64) // 192
// ---------------------------------------------------------------------------
// Phase 1 (the reference-faithful block-streamed, AIR-lowered locks): per-block
// QK. One 16-key block per call. The AIR herd loops rounds=ceil(L/16) and
// streams k blocks via depth-2 ping-pong channels (no in-kernel _lock_acquire).
// The running max m and the per-block correction scratch c are caller-provided
// L1 buffers that persist across the herd's block iterations (m is reset on
// the first block in reach, blk == lo/16, for a new query). Reuses the
// reference's _attn_qk + update verbatim, so the flash-attention math is
// identical to the reference's attn_qk.
extern "C" {
// NOTE arg order: s_block is the LAST memref so AIR's shared-L1 classifier tags
// this (qk) call as the s PRODUCER, pairing with the kv consumer (s non-last
// there) for a producer/consumer lock. m_state/c_state are local scratch/state.
ATTN_ENTRY
void attn_qk_blk(bf16 *__restrict q, bf16 *__restrict k_block,
                 bf16 *__restrict m_state, float *__restrict c_state,
                 bf16 *__restrict s_block, int blk, int L) {
  aie_round_nearest_even();
  const aie::vector<int, 16> idx = aie::vector<int, 16>(
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
  // The RTP packs a sliding window above L: bits 0-19 are L, bits 20-30 the
  // window in units of 16 keys (0 = full attention). See attn_window_lo().
  const int lo = attn_window_lo(L);
  L &= ATTN_RTP_L_MASK;
#if ATTN_KV_PARTS == 1
  if (blk == lo / 16)
    aie::store_v(m_state, m_state_init()); // reset running max for a new query
#endif
  int rem = L - blk * 16;
  // Block fully beyond the current KV length L (rem<=0): every key is masked,
  // so it contributes nothing -- but running _attn_qk on an all-masked block
  // feeds -inf through the online-softmax max/rescale and corrupts m/c. Skip it
  // (matches the runtime-L path, which never loops these). Lets ONE fixed
  // ATTN_MAXL build serve every L (the reference one-MAX_L masking).
  if (rem <= 0)
    return;
  // ...and the same for a block wholly before a sliding window: every key in
  // it is out of reach, so it is skipped rather than fed through as -inf.
  if ((blk + 1) * 16 <= lo)
    return;
  rem = (rem < 16) ? rem : 16;
  aie::mask<16> mask = aie::le(idx, rem); // partial mask on the last block
  // The block the window opens in: keys before `lo` are masked like the tail.
  if (lo > blk * 16)
    mask = mask & aie::gt(idx, lo - blk * 16);
  bool is_first = (blk == lo / 16);
  _attn_qk<DH / 8, GQA_R, GQA_S, GQA_T>(q, k_block, s_block, m_state, c_state,
                                        mask, is_first);
  float *c = (float *)(s_block + Q_HEADS_PADDED_PER_CU * 16);
  aie::store_v(c, aie::load_v<8>(c_state));
#if ATTN_KV_PARTS > 1
  // The running max rides behind c: the kv core needs it to merge its partial
  // sums with the other part's.
  aie::store_v(s_block + ATTN_S_M_OFF, aie::load_v<16>(m_state));
#endif
}

#if ATTN_KV_PARTS > 1
// With the context split across parts, a part's first block need not be the
// window's, so the state is cleared before the block loop instead.
ATTN_ENTRY
void attn_qk_reset(bf16 *__restrict m_state) {
  aie::store_v(m_state, m_state_init());
}
#endif
}
