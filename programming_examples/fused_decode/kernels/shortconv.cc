// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
///@file shortconv.cc
///@brief LFM2 Lfm2ShortConv mixer -- the ph0 egress consumer for a conv layer,
///       the analogue of rope.cc for an attention layer.
//
// Layouts follow the reference mlir-aie design (conv_bx.cc + conv_conv.cc);
// see llms/lfm2_1_2b_q4nx/docs/FLM_CONV_REFERENCE.md. The two deliberate
// deviations from it are noted at the bottom.
//
//     [B | C | X] = in_proj(x)     // 3*D, X LAST
//     BX          = B * X          // the PRE-conv signal, and the state
//     conv        = w0*BX[t-2] + w1*BX[t-1] + w2*BX[t]
//     y           = C * conv       // out_proj is the next proj phase
//
// THE STATE RIDES INSIDE THE WEIGHT BUFFER, exactly as the reference does it:
//     wbx = [ w0(D) | w1(D) | w2(D) | BX[t-2](D) | BX[t-1](D) ]   (5*D)
// Taps are TAP-MAJOR (contiguous 16-lane load per tap; HF ships channel-major
// (D,1,3) and the packer transposes) and OLDEST-FIRST. Carrying the two history
// samples in the weight stream means the conv layer needs NO separate state
// buffer, NO extra DDR argument and NO extra channel -- the host writes the
// carried state into the per-layer weight slab each token.
//
// `bx_new` is the contiguous BX copy the reference's conv_bx also emits; the
// host stores it and it becomes the next token's BX[t-1].
//
// Accumulation is aie::mul + 2x aie::mac into accfloat, NOT a mulf->addf chain
// (rejected by the aievec lowering -- see CLAUDE.md).
#include "aie_array_layout.h"
#include "aie_kernel_utils.h"
#include "typedef.hpp"

// Channel count. Defaults to the model header's CONV_DIM; overridable as
// -DSC_DIM=<n> for reduced-width bring-up builds (CONV_DIM itself is a
// constexpr in the model header and cannot be redefined from the command line).
#ifndef SC_DIM
#define SC_DIM CONV_DIM
#endif

// How many ph0 egress waves make up the assembled [B|C|X]. The reference's
// landing buffer is the ATTENTION qkv width (D + DK + DV), so its 3*D in_proj
// arrives in two waves.
#ifndef SC_WAVES
#define SC_WAVES 2
#endif

// ONE buffer holds everything the two mixer tiles share:
//     mix = [ B(L) | C(L) | X(L) | w0(L) | w1(L) | w2(L) ]      (6*L)
// The stage core fills the [B|C|X] third a wave at a time and DMAs the taps
// into the rest; the conv core reads the whole thing. Keeping it as ONE
// allocation is what lets AIR place it once and let both cores reach it as
// neighbour memory -- see shortconv_stage.
template <int L>
void pseduo_shortconv(bf16 *restrict y, bf16 *restrict st_new,
                      const bf16 *restrict mix, const bf16 *restrict st) {
  constexpr int VS = 16;
  bf16 *B = const_cast<bf16 *>(mix);
  bf16 *C = const_cast<bf16 *>(mix) + L;
  bf16 *X = const_cast<bf16 *>(mix) + 2 * L;
  bf16 *w0 = const_cast<bf16 *>(mix) + 3 * L;
  bf16 *w1 = const_cast<bf16 *>(mix) + 4 * L;
  bf16 *w2 = const_cast<bf16 *>(mix) + 5 * L;
  bf16 *s0 = const_cast<bf16 *>(st);     // BX[t-2]
  bf16 *s1 = const_cast<bf16 *>(st) + L; // BX[t-1]

  AIE_PREPARE_FOR_PIPELINING AIE_LOOP_RANGE(128) for (int i = 0; i < L;
                                                      i += VS) {
    aie::vector<bf16, VS> bv = aie::load_v<VS>(B + i);
    aie::vector<bf16, VS> xv = aie::load_v<VS>(X + i);
    aie::vector<bf16, VS> cv = aie::load_v<VS>(C + i);
    aie::vector<bf16, VS> bx = aie::mul(bv, xv).template to_vector<bf16>();

    aie::accum<accfloat, VS> acc =
        aie::mul(aie::load_v<VS>(s0 + i), aie::load_v<VS>(w0 + i));
    acc = aie::mac(acc, aie::load_v<VS>(s1 + i), aie::load_v<VS>(w1 + i));
    acc = aie::mac(acc, bx, aie::load_v<VS>(w2 + i));

    aie::vector<bf16, VS> cw = acc.template to_vector<bf16>();
    aie::store_v(y + i, aie::mul(cv, cw).template to_vector<bf16>());
    // Emit the SHIFTED state in place: [BX[t-1] | BX]. Doing the shift here
    // (rather than on the host) keeps the state a single opaque 2*D blob that
    // is read and written at the same DDR offset.
    aie::store_v(st_new + i, aie::load_v<VS>(s1 + i));
    aie::store_v(st_new + L + i, bx);
  }
}

extern "C" {

// ShortConv, one core over SC_DIM channels.
//
//   mix    [6*SC_DIM]  SHARED with the stage tile: [B|C|X|w0|w1|w2]
//   st     [2*SC_DIM]  [BX[t-2] | BX[t-1]], DMA'd onto THIS tile
//   y      [SC_DIM]    mixer output -> the out_proj phase
//   st_new [2*SC_DIM]  the SHIFTED state [BX[t-1] | BX], written straight back
//                        over this layer's state slot so the host never shifts
//
// OPERAND ORDER IS LOAD-BEARING. AIR classifies an external call's buffers by
// position -- the LAST memref operand is the one written -- and uses that to
// decide whether a buffer is a cross-core hand-off worth placing once and
// sharing, or private state to be cloned per core
// (air::herdBufferHasCrossCoreDependence). Put `mix` last and it is read as a
// per-core output, gets duplicated on both tiles, and the design deadlocks.
void shortconv_compute(bf16 *restrict mix, bf16 *restrict st, bf16 *restrict y,
                       bf16 *restrict st_new, int _arm) {
  aie_round_nearest_even();
  (void)_arm; // per-token RTP arm-gate operand (kept alive so AIR emits the arm
              // lock)
  pseduo_shortconv<SC_DIM>(y, st_new, mix, st);
}

// Stage one ph0 egress WAVE into the shared [B|C|X].
//
// in_proj is 3*D wide but the mixer's ph0 landing buffer is narrower, so the
// assembled input is built from SC_WAVES successive waves -- exactly as the
// reference does (kernel/rope.cc: memcpy, release the producer lock, acquire
// the consumer lock, memcpy the next slice).
//
// `dst` is on the NEIGHBOUR TILE. This store crosses the tile boundary as
// ordinary core writes into adjacent data memory: no DMA, no BD, no route. That
// is the point -- it is how the reference moves 3*D without spending a channel,
// and DMAing it into one tile instead is what our earlier builds deadlocked on.
// `dst` is LAST so AIR sees this core as the buffer's producer.
//
//   src  [3*SC_DIM/SC_WAVES]  one wave, on this (stage) tile
//   dst  [6*SC_DIM]           the shared mix buffer, on the conv tile
//   wave which slice of dst[0 : 3*SC_DIM] this call fills
void shortconv_stage(bf16 *restrict src, bf16 *restrict dst, int wave,
                     int _arm) {
  aie_round_nearest_even();
  (void)_arm;
  constexpr int VS = 16;
  constexpr int SLICE = 3 * SC_DIM / SC_WAVES;
  bf16 *d = dst + wave * SLICE;
  AIE_PREPARE_FOR_PIPELINING AIE_LOOP_RANGE(192) for (int i = 0; i < SLICE;
                                                      i += VS)
      aie::store_v(d + i, aie::load_v<VS>(src + i));
}

// Two waves in ONE call, from two landing buffers.
//
// The per-wave form above needs one call per wave, and AIR wraps every call in
// its own acquire/release on the shared buffer -- so the stage core signals
// SC_WAVES times per token while the conv core waits once, and the design
// hangs. The reference has the same split (its landing buffer is filled twice)
// but releases the cross-tile lock ONCE, after both copies. This is that: one
// lock cycle on `dst`, whatever the landing granularity.
void shortconv_stage2(bf16 *restrict src0, bf16 *restrict src1,
                      bf16 *restrict dst, int _arm) {
  aie_round_nearest_even();
  (void)_arm;
  constexpr int VS = 16;
  constexpr int SLICE = 3 * SC_DIM / 2;
  AIE_PREPARE_FOR_PIPELINING AIE_LOOP_RANGE(192) for (int i = 0; i < SLICE;
                                                      i += VS) {
    aie::store_v(dst + i, aie::load_v<VS>(src0 + i));
    aie::store_v(dst + SLICE + i, aie::load_v<VS>(src1 + i));
  }
}
}
