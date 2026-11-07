// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// One Gated DeltaNet decode step for one value head and one 32-wide chunk of
// its value dimension, transcribed from FastFlowLM's
// FLM_Xclbin/Qwen3_6/qwen3_6_moe_decoding/kernels/
// gate_delta_net_decode_128_128_128_32.cc. With the chunk's state S
// [128 keys][32 values] (float32, key-major):
//
//   u  = beta (v - alpha S^T k)
//   S' = alpha S + k u^T
//   o  = S'^T q
//
// A head's inputs arrive as one buffer of three SLOT-wide slots, as FLM's conv
// core lays them out: k then alpha (the decay, already exponentiated) as a
// float32 in the next two bf16, v then beta likewise, then q (l2-normalized
// and scaled). Every float32 x bf16 product runs as bf16 limbs.

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "aie_kernels/aie_kernel_utils.h"
#include "bf16_limb.hpp"

namespace {

constexpr int DK = 128;
constexpr int DV = 128;
constexpr int VC = 32; // a core's chunk of the value dimension
constexpr int VL = 32;
constexpr int SLOT = DV + 32;
using acc_t = aie::accum<accfloat, VL>;

} // namespace

extern "C" {

// kvq: [k, alpha | v, beta | q], 3 SLOT; s_in, s_out: [DK][VC] float32;
// out: [VC] bf16; chunk: which VC values of v.
void gdn_step(float *__restrict s_in, float *__restrict s_out,
              bfloat16 *__restrict kvq, bfloat16 *__restrict out, int chunk) {
  const bfloat16 *k = kvq, *v = kvq + SLOT, *q = kvq + 2 * SLOT;
  const float alpha = *(const float *)(k + DK);
  const float beta = *(const float *)(v + DV);

  // S^T k: S is float32, k bf16-exact; one accumulator per limb
  acc_t sk0 = aie::zeros<accfloat, VL>(), sk1 = aie::zeros<accfloat, VL>();
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_RANGE(DK, DK)
  for (int c = 0; c < DK; c++) {
    aie::vector<bfloat16, VL> kb = aie::broadcast<bfloat16, VL>(k[c]);
    bf16_limb::mac_f32_bf16_low(sk0, sk1, aie::load_v<VL>(s_in + c * VC), kb);
  }
  acc_t sk = bf16_limb::reduce(sk0, sk1);

  // u = beta v - (beta alpha) S^T k
  const auto beta_l = bf16_limb::split_f32_3(aie::broadcast<float, VL>(beta));
  const auto beta_alpha_l =
      bf16_limb::split_f32_3(aie::broadcast<float, VL>(beta * alpha));
  aie::vector<bfloat16, VL> vv = aie::load_v<VL>(v + chunk * VC);
  acc_t u = aie::zeros<accfloat, VL>();
  u = bf16_limb::mac_limbs_bf16(u, beta_l, vv);
  u = bf16_limb::msc_limbs3_f32_tri(u, beta_alpha_l,
                                    sk.template to_vector<float>());
  auto [u0, u1] = bf16_limb::split_f32(u.template to_vector<float>());

  // S' = alpha S + k u^T, and o = S'^T q with S' rounded to bf16
  const auto alpha_l = bf16_limb::split_f32_3(aie::broadcast<float, VL>(alpha));
  acc_t o = aie::zeros<accfloat, VL>();
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_RANGE(DK, DK)
  for (int c = 0; c < DK; c++) {
    aie::vector<bfloat16, VL> kb = aie::broadcast<bfloat16, VL>(k[c]);
    acc_t ns = aie::zeros<accfloat, VL>();
    ns = bf16_limb::mac_limbs_bf16(ns, u0, u1, kb);
    ns = bf16_limb::mac_limbs3_f32_tri(ns, alpha_l,
                                       aie::load_v<VL>(s_in + c * VC));
    aie::store_v(s_out + c * VC, ns.template to_vector<float>());
    o = aie::mac(o, q[c], bf16_limb::to_bf16(ns));
  }
  aie::vector<bfloat16, VL> ov = bf16_limb::to_bf16(o);
  aie::store_v(out, ov.template extract<16>(0));
  aie::store_v(out + 16, ov.template extract<16>(1));
}

// The same, with head h's outputs at out + 32 h: a core's 32 heads in one
// buffer.
void gdn_step_h(float *__restrict s_in, float *__restrict s_out,
                bfloat16 *__restrict kvq, bfloat16 *__restrict out, int chunk,
                int h) {
  gdn_step(s_in, s_out, kvq, out + h * VC, chunk);
}

} // extern "C"
