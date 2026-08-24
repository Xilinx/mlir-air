// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// NUMERIC GATE for the batched q4k matmul (q4k_mm.h). One core, one call:
// accumulate GATE_NBLK weight blocks into one Yt and hand it back.
//
// This is the check q4k_mm.h's header says is missing. The kernel uses AIE
// intrinsics and cannot run on the host, and the layout it depends on is
// derived from two independent sources (q4_k.h's packed order on one side,
// aie::mmul's pointer walk on the other) -- so nothing but a device run against
// a numpy reference actually settles it. q4k_mm_gate.py is that run.
//
// GATE_NBLK > 1 exists to exercise the ACCUMULATE path: the real engine walks
// J2 blocks along the contraction into one accumulator, and a block-stride
// mistake is invisible at NBLK=1. The blocks are contiguous in `packed` and the
// activation tile for block b sits at b*(GATE_BATCH*GATE_KCOL).
//
// Not a performance harness -- q4k_mm_bench.cc is. Nothing here is unrolled and
// the C zeroing is a scalar loop, both on purpose: the gate should be as close
// to obviously-correct as possible so that a failure indicts q4k_mm.h and not
// this file.
#include "q4k_mm.h"

#ifndef GATE_MROWS
#define GATE_MROWS 32
#endif
#ifndef GATE_KCOL
#define GATE_KCOL 256
#endif
#ifndef GATE_BATCH
#define GATE_BATCH 16
#endif
#ifndef GATE_NBLK
#define GATE_NBLK 2
#endif

// GATE_MMUL_PROBE: bypass q4k_mm.h entirely and do ONE aie::mmul<8,8,8> per
// probe, straight out of the caller's buffers. The strides in q4k_mmul are
// unambiguous -- they are written down in the source -- so when a whole-block
// result disagrees with numpy the open question is what aie::mmul means by an
// A, B and C *tile*, and that is answerable with one multiply and no strides at
// all. Probe p reads A at Xt+64p and B at packed+64p and writes C at Yt+64p.
#ifdef GATE_MMUL_PROBE
#include <aie_api/aie.hpp>
extern "C" void q4k_mm_gate(bf16 *packed, bf16 *Xt, float *Yt, bf16 *W) {
  using MMUL = aie::mmul<8, 8, 8, bf16, bf16, accauto>;
  for (int p = 0; p < GATE_MMUL_PROBE; ++p) {
    MMUL C;
    C.mul(aie::load_v<64>(Xt + 64 * p), aie::load_v<64>(packed + 64 * p));
    aie::store_v(Yt + 64 * p, C.template to_vector<float>());
  }
  // Echo B so the probe reports what actually reached L1, same as the gate.
  for (int i = 0; i < 64 * GATE_MMUL_PROBE; ++i)
    W[i] = packed[i];
}
#else

extern "C" {

void q4k_mm_gate(bf16 *packed, bf16 *Xt, float *Yt, bf16 *W) {
  // q4k_mmul loads C on entry and stores it on exit -- it accumulates. The
  // engine relies on that across blocks, so the gate has to supply the zero.
  for (int i = 0; i < GATE_BATCH * GATE_MROWS; ++i)
    Yt[i] = 0.0f;

  for (int b = 0; b < GATE_NBLK; ++b)
    // Stepped on the bf16 side, NOT as `A + b` -- see Q4K_BLOCK_BF16 in
    // q4k_mm.h. `A + b` is a 9216-byte stride and the first thing it reads as a
    // scale is a nibble, which is how this gate first failed.
    q4k_mm_block<GATE_MROWS, GATE_KCOL, GATE_BATCH>(
        (const q4k_block_t *)(packed + b * Q4K_BLOCK_BF16),
        Xt + b * (GATE_BATCH * GATE_KCOL), Yt, W);
}

} // extern "C"
#endif // GATE_MMUL_PROBE
