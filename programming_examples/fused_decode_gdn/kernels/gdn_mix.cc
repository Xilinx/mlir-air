// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// fused_decode_gdn's mixer tiles: the LFM2 hybrid's stage and conv tiles,
// re-purposed for Qwen3.5, and the mixer-output tile.
//
// The three mixer phases land on the stage tile in CONV_LAND-row waves;
// gdn_stage3 assembles them into the shared buffer `mix`. The conv tile takes
// `mix` once per wave (gdn_take) and sends the 16 heads' rows to the GDN herd
// and the attention gate to the mixer-output tile (gdn_out), which forwards
// one of the two to the attention kv cores:
//   Gated DeltaNet wave (arm 1): the 16 heads' [q k v | z | b a | pad] (544
//       rows each, the first 8704 rows) go to the GDN herd; the herd's
//       gated-normed output (2048) goes to the kv cores, which put it in
//       place of o.
//   attention wave (arm 2): sigmoid(gate) of the 8 heads (landing 1's first
//       2048 rows) goes to the kv cores, which multiply o by it.
// Only one call per wave may reference `mix` (AIR wraps each call in an
// acquire / release of the shared buffer against the stage tile's one).

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "gdn_math.h"

namespace {
constexpr int LAND = 3072;     // CONV_LAND: rows per mixer phase
constexpr int GDN_IN = 8704;   // 16 heads x 544
constexpr int GATE = 3072;     // attention: the gate's offset in mix
constexpr int DOUT = 2048;     // the mixer output (o-proj X)
constexpr int HEAD_ROWS = 544; // a head's rows: q k v | z | b a (hi, lo) | pad
} // namespace

extern "C" {

// lut: the layer's rope_w slab; a Gated DeltaNet layer's starts with its 16
// heads' -exp(A_log), then their dt_bias (f32), copied behind the landings
void gdn_stage3(bfloat16 *__restrict l0, bfloat16 *__restrict l1,
                bfloat16 *__restrict l2, bfloat16 *__restrict lut,
                bfloat16 *__restrict mix, int _arm) {
  (void)_arm;
  for (int i = 0; i < LAND; i += 32) {
    aie::store_v(mix + i, aie::load_v<32>(l0 + i));
    aie::store_v(mix + LAND + i, aie::load_v<32>(l1 + i));
    aie::store_v(mix + 2 * LAND + i, aie::load_v<32>(l2 + i));
  }
  for (int i = 0; i < 64; i += 32)
    aie::store_v(mix + 3 * LAND + i, aie::load_v<32>(lut + i));
}

// xo [GDN_IN]: the GDN input rows; gate [DOUT]: (attention) sigmoid(gate)
void gdn_take(bfloat16 *__restrict mix, bfloat16 *__restrict xo,
              bfloat16 *__restrict gate, int arm) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  for (int i = 0; i < GDN_IN; i += 32)
    aie::store_v(xo + i, aie::load_v<32>(mix + i));
  if (arm == 1) {
    // the 16 heads' gates at once (FP32 vector math is emulated on the cores,
    // so one pass here rather than one per head on the GDN herd): beta =
    // bf16(sigmoid(b)), decay = exp(-exp(A_log) softplus(a + dt_bias)), b and a
    // the sums of the head's rows 512 + 513 and 514 + 515; both replace those
    // rows as f32 words [beta | decay]
    using namespace gm;
    alignas(64) float bb[16], aa[16];
    for (int h = 0; h < 16; h++) {
      const bfloat16 *r = mix + h * HEAD_ROWS + 512;
      bb[h] = (float)r[0] + (float)r[1];
      aa[h] = (float)r[2] + (float)r[3];
    }
    const float *cst = (const float *)(mix + 3 * LAND);
    VF avv = aie::add(aie::load_v<16>(aa), aie::load_v<16>(cst + 16));
    VF zero = vbc(0.0f);
    VF sg = sigmoid_v(aie::load_v<16>(bb));
    VF ea = exp_v(aie::min(avv, aie::sub(zero, avv)));
    VF sp = aie::add(aie::max(avv, zero), log1p_v(ea));
    sp = aie::select(sp, avv, aie::gt(avv, vbc(20.0f)));
    VF dc = exp_v(vmul(sp, aie::load_v<16>(cst)));
    alignas(64) float sv[16], dv[16];
    aie::store_v(sv, sg);
    aie::store_v(dv, dc);
    for (int h = 0; h < 16; h++) {
      float *g = (float *)(xo + h * HEAD_ROWS + 512);
      g[0] = (float)(bfloat16)sv[h];
      g[1] = dv[h];
    }
    return;
  }
  if (arm != 2)
    return;
  for (int i = 0; i < DOUT; i += 16) {
    gm::VF g = gm::to_f(aie::load_v<16>(mix + GATE + i));
    aie::accum<accfloat, 16> a;
    a.from_vector(gm::sigmoid_v(g));
    aie::store_v(gate + i, a.template to_vector<bfloat16>());
  }
}

// y: the GDN output (arm 1) or sigmoid(gate) (arm 2)
void gdn_out(bfloat16 *__restrict g, bfloat16 *__restrict gin,
             bfloat16 *__restrict y, int arm) {
  const bfloat16 *src = arm == 2 ? g : gin;
  for (int i = 0; i < DOUT; i += 32)
    aie::store_v(y + i, aie::load_v<32>(src + i));
}

} // extern "C"
