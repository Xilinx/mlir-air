// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// float32 products on the bf16 MAC datapath, transcribed from FastFlowLM's
// FLM_Xclbin/Qwen3_6/qwen3_6_moe_decoding/kernels/bf16_limb_math.hpp: a float32
// operand is split into bf16 limbs (x ~= l0 + l1 [+ l2]) and each limb meets
// the other operand in one MAC.
//
//   mac_f32_bf16_low        f32 x bf16, 2 limbs (~17 mantissa bits)
//   mac_limbs3_tri          f32 x f32, 3 limbs each, the 6 products above the
//                           fp32 mantissa floor (fp32-exact)
//
// Every accum -> bf16 conversion goes 16 lanes at a time (to_bf16): Peano's
// 32-lane conversion returns the upper half twice.

#ifndef QWEN36_BF16_LIMB_HPP
#define QWEN36_BF16_LIMB_HPP

#include <utility>

#include <aie_api/aie.hpp>

namespace bf16_limb {

template <unsigned N>
inline aie::vector<bfloat16, N> to_bf16(aie::accum<accfloat, N> a) {
  if constexpr (N <= 16) {
    return a.template to_vector<bfloat16>();
  } else {
    static_assert(N == 32, "16 or 32 lanes");
    aie::vector<float, N> f = a.template to_vector<float>();
    aie::accum<accfloat, 16> lo, hi;
    lo.from_vector(f.template extract<16>(0));
    hi.from_vector(f.template extract<16>(1));
    return aie::concat(lo.template to_vector<bfloat16>(),
                       hi.template to_vector<bfloat16>());
  }
}

template <unsigned N>
inline aie::vector<bfloat16, N> ones() {
  return aie::broadcast<bfloat16, N>((bfloat16)1.0f);
}

template <unsigned N>
inline std::pair<aie::vector<bfloat16, N>, aie::vector<bfloat16, N>>
split_f32(aie::vector<float, N> x) {
  aie::accum<accfloat, N> r0;
  r0.from_vector(x);
  aie::vector<bfloat16, N> l0 = to_bf16(r0);
  aie::vector<bfloat16, N> l1 = to_bf16(aie::msc(r0, l0, ones<N>()));
  return {l0, l1};
}

template <unsigned N>
struct limbs3 {
  aie::vector<bfloat16, N> l0, l1, l2;
};

template <unsigned N>
inline limbs3<N> split_f32_3(aie::vector<float, N> x) {
  limbs3<N> out;
  aie::accum<accfloat, N> r0;
  r0.from_vector(x);
  out.l0 = to_bf16(r0);
  aie::accum<accfloat, N> r1 = aie::msc(r0, out.l0, ones<N>());
  out.l1 = to_bf16(r1);
  out.l2 = to_bf16(aie::msc(r1, out.l1, ones<N>()));
  return out;
}

// acc + a * b, a float32 in 2 limbs
template <unsigned N>
inline aie::accum<accfloat, N> mac_f32_bf16_low(aie::accum<accfloat, N> acc,
                                                aie::vector<float, N> a,
                                                aie::vector<bfloat16, N> b) {
  auto [l0, l1] = split_f32(a);
  acc = aie::mac(acc, b, l1);
  return aie::mac(acc, b, l0);
}

// The same into two accumulators, one limb each, for reduction loops; fold
// them with reduce() after the loop.
template <unsigned N>
inline void
mac_f32_bf16_low(aie::accum<accfloat, N> &acc0, aie::accum<accfloat, N> &acc1,
                 aie::vector<float, N> a, aie::vector<bfloat16, N> b) {
  auto [l0, l1] = split_f32(a);
  acc0 = aie::mac(acc0, b, l0);
  acc1 = aie::mac(acc1, b, l1);
}

template <unsigned N>
inline aie::accum<accfloat, N> reduce(aie::accum<accfloat, N> acc0,
                                      aie::accum<accfloat, N> acc1) {
  aie::accum<accfloat, N> out;
  out.from_vector(aie::add(acc1.template to_vector<float>(),
                           acc0.template to_vector<float>()));
  return out;
}

template <unsigned N>
inline aie::accum<accfloat, N>
mac_limbs_bf16(aie::accum<accfloat, N> acc, aie::vector<bfloat16, N> l0,
               aie::vector<bfloat16, N> l1, aie::vector<bfloat16, N> b) {
  acc = aie::mac(acc, b, l1);
  return aie::mac(acc, b, l0);
}

template <unsigned N>
inline aie::accum<accfloat, N> mac_limbs_bf16(aie::accum<accfloat, N> acc,
                                              limbs3<N> l,
                                              aie::vector<bfloat16, N> b) {
  acc = aie::mac(acc, b, l.l2);
  acc = aie::mac(acc, b, l.l1);
  return aie::mac(acc, b, l.l0);
}

template <unsigned N>
inline aie::accum<accfloat, N> mac_limbs3_tri(aie::accum<accfloat, N> acc,
                                              const limbs3<N> &a,
                                              const limbs3<N> &b) {
  acc = aie::mac(acc, a.l2, b.l0);
  acc = aie::mac(acc, a.l1, b.l1);
  acc = aie::mac(acc, a.l0, b.l2);
  acc = aie::mac(acc, a.l1, b.l0);
  acc = aie::mac(acc, a.l0, b.l1);
  return aie::mac(acc, a.l0, b.l0);
}

template <unsigned N>
inline aie::accum<accfloat, N> mac_limbs3_f32_tri(aie::accum<accfloat, N> acc,
                                                  const limbs3<N> &a,
                                                  aie::vector<float, N> b) {
  return mac_limbs3_tri(acc, a, split_f32_3(b));
}

template <unsigned N>
inline aie::accum<accfloat, N> msc_limbs3_tri(aie::accum<accfloat, N> acc,
                                              const limbs3<N> &a,
                                              const limbs3<N> &b) {
  acc = aie::msc(acc, a.l2, b.l0);
  acc = aie::msc(acc, a.l1, b.l1);
  acc = aie::msc(acc, a.l0, b.l2);
  acc = aie::msc(acc, a.l1, b.l0);
  acc = aie::msc(acc, a.l0, b.l1);
  return aie::msc(acc, a.l0, b.l0);
}

template <unsigned N>
inline aie::accum<accfloat, N> msc_limbs3_f32_tri(aie::accum<accfloat, N> acc,
                                                  const limbs3<N> &a,
                                                  aie::vector<float, N> b) {
  return msc_limbs3_tri(acc, a, split_f32_3(b));
}

} // namespace bf16_limb

#endif
