// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// BATCHED q4k matmul: unpack the weight block ONCE, then aie::mmul it against a
// [KCOL x BATCH] activation tile.
//
// WHY THIS EXISTS. The decode kernel (_qmm_q4k_bf16 in q4_k.h) is a GEMV: its
// activation is a vector, and it does one multiply per unpacked weight, at a
// measured 512 MACs / 140 bundles = 3.7 MAC/cycle/core. The prefill matmul
// (matrix_multiplication/bf16_in_fp32_out/mm_aie2p.cc) uses aie::mmul<8,8,8> and
// measures 9797 GFLOP/s over 32 cores = 98 MAC/cycle/core. Speculative decoding
// (DFlash) needs a batch of 16 tokens per call, and widening the GEMV loop would
// keep the slow form -- so this takes the prefill's matmul and feeds it
// unpacked q4k weights.
//
// The cost of a weight block then splits in two:
//   unpack  -- proportional to WEIGHTS,        independent of BATCH  (intercept)
//   mmul    -- proportional to WEIGHTS*BATCH,  scales with BATCH     (slope)
// Measuring that intercept and slope is the whole point; see q4k_mm_bench.cc.
//
// BUILD FLAG THAT MATTERS: AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 must be ON
// (it already is, in the Makefile's PEANO_KBASE). Measured on q4k_mmul<32,256,16>:
// 1787 bundles with it, 11088 without -- the emulation is a 3.9x SPEEDUP on the
// multiply, not an overhead. Building without it drops the batch at which this
// kernel stops being memory bound from 16.7 to 2.7, which would remove the
// reason to batch at all.
//
// L1 COST THE GEMV DOES NOT HAVE. q4k_unpack_block materialises the unpacked
// tile into W before the mmul can read it -- MROWS*KCOL*2 bytes, 16 KB at
// 32x256. That is the largest buffer on a proj core once it exists (the whole
// core is 25.5 KB without it at batch 16), and it is what sets the batch
// ceiling: 25 instead of 38 on qwen3-4b. The mmul contracts over KCOL, so the
// tile can be chunked and accumulated at the same total unpack cost -- 32x64 is
// 4 KB. See batch_l1_budget.py --scratch-rows/--scratch-cols.
//
// Neither of the two profitable folds fits at the full 256-column tile:
// 32x512 (contraction) is 57.5 KB on the core and 64x256 (rows) is the same,
// against a 54 KB budget. Both are worth having -- measured at batch 16, per
// 32x256 block: 2936 cycles for kcol 512 and 2993 for MROWS 64, against 3184
// here -- and whether either survives a smaller scratch turns on one unsettled
// question: is the fold's gain the accumulator traffic, or the scheduler's
// window over a longer straight-line body? See q4k_mm_chunked below.
//
// NUMERICALLY VALIDATED ON DEVICE. q4k_mm_gate.py runs this on one core and
// compares against numpy BIT-EXACTLY -- not to a tolerance -- at batch 16 and
// 32, one and two blocks, several seeds. Re-run it after touching anything
// here:
//
//     python3 q4k_mm_gate.py --mode exact     # the layout gate
//     python3 q4k_mm_gate.py --mode random    # + the accuracy number
//
// ONE EXCEPTION, AND IT IS EXPECTED: the gate's numpy model reproduces the
// SHIPPING rounding sequence, so `--mode exact` fails by construction under
// -DQ4K_UNPACK_FMA, which removes one of those roundings. That is the model
// being right about the old arithmetic, not the kernel being wrong. The FMA
// path is gated on dump_layer_output.py --diff (small and nonzero, +/-1 in the
// bf16 bit pattern) plus dflash_verify_gate.py instead -- see the
// Q4K_UNPACK_FMA block below. Everything else here still owes the gate an
// exact pass.
//
// Accuracy against an exact fp32 matmul of the same inputs, on realistically
// quantized weights: 1.3% rms, no bias, flat in contraction depth [measured].
//
// The gate was worth building. It caught two faults that every static check
// passed, both of which give a plausible wrong answer rather than a crash:
// q4k_mm_block handed q4k_mmul its operands in the wrong ROLES (the layout half
// of the weights-as-B swap landed, the call sites did not), and `A + b` stepped
// 9216 bytes because sizeof(q4k_block_t) is not the size of a packed block. See
// Q4K_BLOCK_BF16 above and q4k_mm_block below.
//
// Two things about the arithmetic that only a device run could establish, both
// now modelled in the gate: aie::mmul multiplies in a bfp16 block format (8
// elements share an exponent from the block max, 7 significant bits), and every
// bf16 rounding on the core rounds toward MINUS INFINITY. The second is biased,
// so what it contributes to a dot product is linear in K and proportional to
// the MEAN of the operands -- invisible on centred weights, which is what q4k's
// min/max codec produces, and 11% at K=512 on weights that are not centred.
#ifndef __Q4K_MM_H__
#define __Q4K_MM_H__
#include "aie_kernel_utils.h"
#include "model_spec.h"

// A PACKED BLOCK IS 5120 BYTES AND sizeof(q4k_block_t) IS NOT.
//
// `uint4` is byte-addressed: aie::load_v<N> reads N nibbles, but pointer
// arithmetic on uint4* counts BYTES (q4_k.h's GEMV consumes 512 nibbles per
// 32-column group and advances qs_ptr by 256 -- that only balances at 2 nibbles
// per unit). So `uint4 qs[8192]` reserves 8192 bytes for 4096 bytes of data and
// the struct measures 9216 against a packed block's 5120.
//
// It has never mattered, because every production use casts a bf16* to
// q4k_block_t* at the call boundary and steps BLOCKS on the bf16 side. It bites
// the moment new code writes `A + b`: that lands 4096 bytes into the next
// block's nibbles and reads them as scales, which is a NaN, not a small error.
// Step blocks as `(const q4k_block_t *)(w + b * Q4K_BLOCK_BF16)`.
//
// Asserted as an inequality on purpose -- if the struct is ever made exact,
// this fires and says the workaround can go.
constexpr int Q4K_BLOCK_BF16 =
    Q4NX_ROW_BLOCK_SIZE * Q4NX_COL_BLOCK_SIZE / 4 +
    2 * (Q4NX_ROW_BLOCK_SIZE * Q4NX_COL_BLOCK_SIZE / Q4NX_GROUP_SIZE);
static_assert(sizeof(q4k_block_t) != 2 * Q4K_BLOCK_BF16,
              "q4k_block_t is now exactly one packed block -- drop "
              "Q4K_BLOCK_BF16 and index the struct directly");

// Q4K_MM_FULL_UNROLL: straight-line the whole block so that STATIC bundle count
// equals DYNAMIC cycle count. Only the static-cost bench defines it -- a real
// build wants the rolled form. (Diffing two builds at different contraction
// lengths does not work here: the trip count lives in a register, so both builds
// emit the same body.)
//
// The unpack and the multiply take SEPARATE unroll controls because the unpack
// can no longer be unrolled: with the correct two-tile store the unrolled form
// crashes the Peano backend --
//
//   Register not in mBMs
//   UNREACHABLE executed at .../aie2p/AIE2PMCCodeEmitterGen.inc:1581
//   Running pass 'AIE2 Assembly Printer' on q4k_unpack_block<32, 256>
//
// -- while the ROLLED form, which is what a real build uses, compiles fine.
// So the unpack intercept is no longer measurable as an exact cycle count; only
// the multiply is. Q4K_UNPACK_NOPERM (numerically wrong, see below) restores the
// old fully-unrolled unpack for comparison against the earlier numbers.
#ifdef Q4K_MM_FULL_UNROLL
#define Q4K_MM_LOOP AIE_LOOP_UNROLL_FULL
#else
#define Q4K_MM_LOOP
#endif

// Q4K_MM_UNROLL=N: partial unroll of the CONTRACTION loop only -- the innermost
// `for i < colA` that carries the mac chains -- leaving the z and j blocking
// loops alone. The same lever as Q4K_UNPACK_UNROLL, tried for the same reason:
// with the unpack's arithmetic fixed the two halves share ONE budget (either
// one shedding 13.4 ms of core reaches the memory floor), and the mmul body
// looked like the sparser of the two -- 1.6 ops per bundle against the unpack's
// 2.7, which reads as dependency stalls that more copies in flight would fill.
//
// AT BATCH 8 IT DOES NOTHING. That reading was wrong: `vmac.f` on 8x8
// accumulators is multi-cycle, so a sparse bundle count is not a fillable
// stall. Static bundles for the contraction loop, x32 iterations, against the
// 576 the rolled form takes under FMA + unroll 8:
//
//   N=2   768   1.091        N=8   860   1.222
//   N=4   816   1.159        N=16  854   1.213
//
// and N=2, the least-bad, measured NEUTRAL on device -- 84.480 / 86.624 /
// 86.451 against a control's 85.923 / 84.514 / 86.303 over three paired
// replicates at L161. Fully overlapping.
//
// THE FIRST REPLICATE READ 84.480 AND LOOKED LIKE A 1.4 ms WIN. It was noise.
// Run-to-run spread on a single unchanged build here is 1.8 ms, wider than the
// ~1 ms build-to-build figure quoted elsewhere, so on this instrument nothing
// under about 2 ms is a result no matter how clean one run looks.
//
// KEPT ANYWAY, unset, because batch 16 uses a different multiply --
// q4k_mmul's 2x2 blocking with four live accumulators, not q4k_mmul_small's
// 1x4 -- and the sweep above says nothing about that shape. Re-sweep it there
// rather than assuming this answer transfers.
//
// BIT-EXACT by construction: unrolling changes neither the order of the macs
// nor which accumulator each lands in. Unlike Q4K_UNPACK_FMA, dump_layer_output
// must read IDENTICAL, and anything else is a bug.
//
// Unset by default -- shipping kernels stay byte-identical.
#ifdef Q4K_MM_UNROLL
#define Q4K_MM_ILOOP AIE_LOOP_UNROLL(Q4K_MM_UNROLL)
#else
#define Q4K_MM_ILOOP Q4K_MM_LOOP
#endif
//
// Q4K_UNPACK_UNROLL=N is the middle ground the paragraph above never tried: a
// PARTIAL unroll (AIE_LOOP_UNROLL(N), i.e. clang's unroll_count) rather than
// the full one that crashes the backend. This loop runs MROWS*KCOL/128 = 64
// rolled iterations with a loop-carried pointer, which is the shape an unroll
// helps -- but ONLY alongside Q4K_UNPACK_FMA, and the sign flips without it.
// N=8 is what the shipping configuration uses; see the measured table in
// q4k_unpack_block.
//
// Unset by default, so the shipping kernels are byte-identical and
// check_kernels_inert.py stays satisfied. If a value crashes Peano the way
// AIE_LOOP_UNROLL_FULL does, that is the answer for that value; try a smaller
// one rather than concluding the loop cannot be unrolled at all.
#if defined(Q4K_MM_FULL_UNROLL) && defined(Q4K_UNPACK_NOPERM)
#define Q4K_UNPACK_LOOP AIE_LOOP_UNROLL_FULL
#elif defined(Q4K_UNPACK_UNROLL)
#define Q4K_UNPACK_LOOP AIE_LOOP_UNROLL(Q4K_UNPACK_UNROLL)
#else
#define Q4K_UNPACK_LOOP
#endif

// Q4K_UNPACK_FMA: do w = q*scale + min in ONE accumulator pass.
//
// WHAT IT IS FOR. Disassembling proj_qmm_mm_acc says the unpack loop body is 31
// bundles and 68 ops per 128-lane chunk, and SEVENTEEN of those ops are
// vconv.bf16.fp32 / vconv.fp32.bf16 -- pure format shuffling with no arithmetic
// content -- plus the vadd/vsub.f magic-constant pairs that implement the bf16
// rounding. Only four ops are the stores into W. The unpack is not memory
// bound and it is not the broadcast; it is the bf16 round trip between every
// arithmetic step. aie::mul returns an accum, `ws` rounds it to bf16, aie::add
// lifts it straight back to an accum. Seeding the accumulator with `min` and
// mac-ing q*scale into it removes one whole round trip.
//
// IT CHANGES THE ANSWER, in the direction of the exact result: q*scale is no
// longer rounded to bf16 before the add, so there is one rounding rather than
// two. The final w is still bf16 and the mmul still takes it down to bfp16
// (7 significant bits, shared exponent per 8), so most of the difference is
// re-quantised away downstream -- but it is NOT bit-identical, and
// q4k_mm_gate.py --mode exact models the shipping arithmetic. Gate this one on
// dump_layer_output.py --diff (expect small and nonzero, not zero) followed by
// dflash_verify_gate.py, not on bit-exactness.
//
// IT ALSO UNBLOCKS THE UNROLL. Partial unrolling was a measured device LOSS on
// the shipping form -- see the note in q4k_unpack_block -- because the body
// already had no registers to spare. With one fewer accumulator round trip it
// pays. Static bundles per 128-lane chunk, x64 chunks per weight block:
//
//   shipping                       31.0    1984   1.000
//   FMA                            28.0    1792   0.903
//   FMA + Q4K_UNPACK_UNROLL=2      19.0    1216   0.613
//   FMA + Q4K_UNPACK_UNROLL=8      17.6    1128   0.569
//   FMA + Q4K_UNPACK_UNROLL=16     29.1    1864   0.940
//   Q4K_UNPACK_UNROLL=2 alone      38.5    2464   1.242
//   Q4K_UNPACK_UNROLL=4 alone      33.5    2144   1.081
//
// THE STATIC COUNT HAS A CONTROL, which is why the numbers above are worth
// anything. The last two rows are the two configurations that were run on
// device before this file was touched: unroll 2 alone measured +6.3 ms on a
// 34.98 ms unpack (1.18x) against a predicted 1.242x, and unroll 4 alone
// measured neutral against a predicted 1.081x. Same sign, same magnitude. It is
// still a static count and it cannot see a memory stall, so device timing is
// the verdict -- but it is a cheap and calibrated way to reject candidates.
//
// Unset by default so the shipping kernels stay byte-identical and
// check_kernels_inert.py stays satisfied. The switch itself is in
// q4k_unpack_step below.

// One 128-lane unpack step: 128 nibbles -> bf16, scaled by the per-(row,group)
// scale/min. A 128-lane chunk covers 8 columns x 16 rows and so sits inside ONE
// 32-column group, which is why a single 16-lane scale/min pair serves it.
template <int PR>
static inline aie::vector<bf16, PR * 8>
q4k_unpack_step(const uint4 *&qs_ptr, const bf16 *scales, const bf16 *mins) {
  aie::vector<uint4, PR * 8> q = aie::load_v<PR * 8>(qs_ptr);
  qs_ptr += PR * 4;
  aie::accum<accfloat, PR * 8> acc;
  acc.from_vector(aie::to_float(q, 0));
  aie::vector<bf16, PR * 8> qb = acc.template to_vector<bf16>();

  // Replicate the 16-lane scale/min across the chunk's 8 columns.
  aie::vector<bf16, PR> s16 = aie::load_v<PR>(scales);
  aie::vector<bf16, PR> m16 = aie::load_v<PR>(mins);
  aie::vector<bf16, PR * 2> s32 = aie::concat(s16, s16);
  aie::vector<bf16, PR * 2> m32 = aie::concat(m16, m16);
  aie::vector<bf16, PR * 4> s64 = aie::concat(s32, s32);
  aie::vector<bf16, PR * 4> m64 = aie::concat(m32, m32);
  aie::vector<bf16, PR * 8> sv = aie::concat(s64, s64);
  aie::vector<bf16, PR * 8> mv = aie::concat(m64, m64);

  // w = q*scale + min  (the additive form -- see the q4_k.h header comment).
#ifdef Q4K_UNPACK_FMA
  // ONE accumulator pass instead of two -- see the Q4K_UNPACK_FMA note above.
  // aie::mul returns an accum that the bf16 result type immediately rounds back
  // down, and aie::add lifts it straight back up, so the shipping form pays a
  // full fp32 -> bf16 -> fp32 round trip in the middle of what is one fused
  // operation. Seeding the accumulator with `min` and mac-ing q*scale into it
  // removes that round trip.
  aie::accum<accfloat, PR * 8> out;
  out.from_vector(mv);
  return aie::mac(out, qb, sv).template to_vector<bf16>();
#else
  aie::vector<bf16, PR * 8> ws = aie::mul(qb, sv).template to_vector<bf16>();
  return aie::add(ws, mv);
#endif
}

// The six concats above ARE redundant three times out of four -- see the
// measured note in q4k_unpack_block on why hoisting them loses anyway.

// Unpack a whole MROWS x KCOL block into the layout aie::mmul wants for A.
//
// THE PACKED ORDER, read off q4_k.h's GEMV rather than assumed. That loop walks
// `qs_ptr` linearly and slices each 128-nibble load with `.extract<16>(j)` for
// j in 0..7, naming the results a_col_<j>. So a 128-lane load is EIGHT COLUMNS
// of SIXTEEN ROWS, column-major. Its loop nest is row-block (16) outer,
// 32-column group next, then four 8-column loads. For a (row, col) weight:
//
//     packed[ R*(KCOL*16) + col*16 + rr ]      row = 16*R + rr
//
// WHICH MMUL OPERAND. A chunk is contraction-major with the weight row minor.
// aie::mmul<r,s,t> computes C[r x t] = A[r x s] * B[s x t] and takes B as
// [s][t] row-major -- [contraction][output]. That is exactly the chunk's own
// order. Taking A as the ACTIVATIONS and B as the WEIGHTS therefore needs no
// transpose of the weights at all, where the other assignment does:
//
//     Y[MROWS x BATCH] = W * X          weights are A, needs an 8x16 transpose
//     Yt[BATCH x MROWS] = Xt * Wt       weights are B, needs two filters
//
// The second is taken, and it is FORCED rather than preferred. Three reasons,
// in order of how binding they are:
//
//   1. THE SCALE BROADCAST. q4k_unpack_step gives lane l the scale s16[l % 16],
//      which is only right when the chunk's row index is l % 16 -- i.e. column-
//      major within 16 rows, which is mmul's B order. In A order 128 contiguous
//      elements are two row-major 8x8 tiles, so row(l) = 8*(l/64) + (l%64)/8,
//      and lanes 0 and 16 share an l % 16 but sit on different rows. No host
//      packing order fixes that; it is a property of the broadcast. Storing
//      scales pre-expanded would take a 32x256 block from 5120 to 36864 bytes.
//   2. aie::transpose cannot do the 8x16 anyway -- the 16-bit specialization on
//      AIE2 covers 32, 16 and 8 elements and a 128-lane bf16 vector fails to
//      instantiate.
//   3. Yt is TOKEN-major, which is the order the egress consumer wants -- it
//      removes the 2D group gather the wire analysis worked out.
//
// The cost is 10% on the multiply at batch 16 (1787 -> 1965), because rowA/colB
// flip from 4x2 to 2x4; at batch 32 both are 4x4 and it is free. That is the
// price of correctness here, not an artifact to optimise away.
//
// B tile (i, j) sits at (j*colA + i)*64 with colA = KCOL/8, i the contraction
// block and j the output-row block:
//
//     Wb[ (j*(KCOL/8) + i)*64 + s_in*8 + t_in ]   row = 8j + t_in, col = 8i + s_in
//
// A chunk covers ONE contraction block and SIXTEEN rows, so it splits into the
// two output blocks j = 2R and j = 2R+1. Within the chunk, lane = j_col*16 + rr,
// so group-of-8 index lane/8 is even exactly when rr < 8: filter_even with a
// chunk size of 8 collects rows 0..7 of every contraction step, in contraction
// order, which IS the j=2R tile. filter_odd gives j=2R+1. Two 64-lane stores,
// no shuffle network.
template <int MROWS, int KCOL>
static inline void q4k_unpack_block(const q4k_block_t *A, bf16 *__restrict W) {
  constexpr int PR = 16;        // rows per unpack step
  constexpr int CHUNK = PR * 8; // 128 lanes per step
  constexpr int NSTEP = MROWS * KCOL / CHUNK;
  constexpr int NCB = KCOL / 8; // contraction blocks (mmul's colA)
  static_assert(MROWS % 16 == 0, "unpack emits two 8-row tiles per chunk");
  static_assert(KCOL % 32 == 0, "a 32-column quant group must be whole");
  const uint4 *qs_ptr = A->qs;

  // HOW MUCH TO UNROLL DEPENDS ON Q4K_UNPACK_FMA, and the sign flips. Measured
  // on the configuration where this kernel is the critical path -- qwen3-4b
  // batch 8, mode 3 + PROJ_PP_ONLY=w, dispatch_time.py median of 25:
  //
  //                                    L=128    L=161    vs its own control
  //   Q4K_UNPACK_UNROLL=2             114.328            +6.3 ms
  //   Q4K_UNPACK_UNROLL=4             108.478            neutral
  //   Q4K_UNPACK_UNROLL=8                      105.543   -3.30 ms, BIT-EXACT
  //   Q4K_UNPACK_UNROLL=8 + FMA                 85.923   -22.92 ms
  //   (controls: 108.074 at L=128, 108.840 at L=161)
  //
  // Small unrolls lose on the shipping arithmetic because the body has no
  // registers to spare; 8 wins because by then the loop-carried scalar address
  // chain -- 21 of the body's 68 ops -- amortises across the copies. Layering
  // the FMA on top is worth another 19.6 ms, because it is what frees the
  // registers the unroll wants. Neither ordering was predictable from the
  // other; both were built.
  //
  // AND THE ONE THAT STILL LOSES: hoisting the scale/min broadcast out of the
  // 4-block group. 134.300 at L=128, +26.2 ms, and BIT-IDENTICAL output. The
  // redundancy it removes is real -- `off` is (cb/4)*MROWS + R*PR, so the four
  // contraction blocks of a 32-column group rebuild the identical 128-lane
  // scale and min vectors, two loads and six concats, three times out of four.
  // Restructured as (row half, group) outer x 4 inner with the ladder lifted,
  // the answer is bit-for-bit the same and it costs 24% MORE: a 4-iteration
  // inner loop cannot amortise a software-pipeline prologue, and that dominates
  // the arithmetic saved. Do not re-derive it.
  // AND SPLITTING THE ROW HALF OUT OF THE LOOP INDEX ALSO LOSES. Scalar
  // address arithmetic is the largest remaining category in this body -- 14.1
  // of 47.9 ops per chunk under FMA + unroll 8 -- and all of it exists to
  // rebuild `i / NCB`, `i % NCB` and the two `(2R+p)*NCB + cb` store addresses
  // that a nested `for R { for cb { } }` would make plain increments. Unlike
  // the hoist above, the inner loop would be NCB = 32 iterations, long enough
  // to amortise a prologue. Static bundles per weight block, against the 1128
  // the shipping flat loop takes at FMA + unroll 8:
  //
  //   nested, no unroll   1856   0.935 of shipping-arithmetic base
  //   nested, unroll 2    1434   0.723      <- best, and still worse than 0.569
  //   nested, unroll 4    2510   1.265
  //   nested, unroll 8    2208   1.113
  //
  // The inner body stops software-pipelining once unrolled (156, 274 and 601
  // bundles for 4, 8 and 16 copies), which no flat-loop unroll does. So those
  // scalar ops are not costing what their count suggests: they co-issue in the
  // scalar slots beside the vector work and are close to free, and buying them
  // back disturbs the vector schedule, which is not. THE FLAT LOOP WITH THE
  // LINEAR INDEX IS THE FAST SHAPE -- three restructurings have now failed to
  // beat it. What is left for this kernel is arithmetic (see Q4K_UNPACK_FMA),
  // not addressing.
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_RANGE(NSTEP, NSTEP)
  Q4K_UNPACK_LOOP
  for (int i = 0; i < NSTEP; i++)
    AIE_LOOP_FLATTEN {
      // i is row-half major, contraction block minor: i = R*NCB + cb.
      const int R = i / NCB;  // which 16-row half
      const int cb = i % NCB; // which 8-column contraction block
      // Scale/min slot: group-major, row-minor, matching q4_k.h's
      // `scale_min_offset += M` walk. The group is the 32-column group WITHIN
      // this row half -- cb/4, not i/4. (i/4 is what this used to say, and it
      // indexes past the end of the scale array on the second row half of any
      // block with MROWS > 16.)
      const int off = (cb / 4) * MROWS + R * PR;
      aie::vector<bf16, CHUNK> w =
          q4k_unpack_step<PR>(qs_ptr, A->scales + off, A->mins + off);
#ifdef Q4K_UNPACK_NOPERM
      // NUMERICALLY WRONG, cost attribution only -- same standing as
      // Q4_SFIX_MODE=2 in q4_k.h. Dumps the chunk contiguously instead of
      // splitting it into the two mmul B tiles, which is what this used to do.
      // Its only use is pricing the split: build both rolled and diff.
      aie::store_v(W + i * CHUNK, w);
#else
      aie::store_v(W + ((2 * R) * NCB + cb) * 64, aie::filter_even(w, 8));
      aie::store_v(W + ((2 * R + 1) * NCB + cb) * 64, aie::filter_odd(w, 8));
#endif
    }
}

// Yt[BATCH x MROWS] += Xt[BATCH x KCOL] * Wt[KCOL x MROWS], 2x2 register-
// blocked, same structure as matmul_vectorized_2x2_mmul in mm_aie2p.cc.
//
// pA is the ACTIVATIONS and pB the unpacked WEIGHTS -- see the q4k_unpack_block
// header for why round that way. The body below is mm_aie2p's verbatim; only
// what the three dimensions MEAN has changed, so `rowA` counts batch blocks and
// `colB` counts weight-row blocks.
//
// pA must be tile-blocked the same way mm_aie2p's A is: tile (z, i) at
// (i*rowA + z)*64, row-major [batch][contraction] inside. A plain [BATCH][KCOL]
// buffer is NOT that; the memtile has to land it blocked, which is a strided BD
// rather than compute.
// 2x2 register blocking: two A tiles and two B tiles held live across the
// contraction, four accumulators. mm_aie2p's structure verbatim.
//
// The accumulators MUST be named locals. Writing them as `MMUL C[RB][CB]` with
// unrolled index loops -- the obvious way to make the blocking a template
// parameter -- puts them on the stack instead of in registers and the frame
// then overflows AIE2's load/store displacement field:
//   immediate operand value -52032 is out of range [-32768, -64]
// So an alternative blocking has to be spelled out, as q4k_mmul_2x4 below is.
template <int MROWS, int KCOL, int BATCH>
static inline void q4k_mmul(const bf16 *__restrict pA, const bf16 *__restrict pB,
                            float *__restrict pC) {
  constexpr int r = 8, s = 8, t = 8;
  using MMUL = aie::mmul<r, s, t, bf16, bf16, accauto>;
  constexpr int rowA = BATCH / r; // batch blocks
  constexpr int colA = KCOL / s;  // contraction blocks
  constexpr int colB = MROWS / t; // weight-row blocks
  static_assert(rowA % 2 == 0, "BATCH must be a multiple of 16");
  static_assert(colB % 2 == 0, "MROWS must be a multiple of 16");

  Q4K_MM_LOOP
  for (unsigned z = 0; z < rowA; z += 2) {
    float *__restrict pC1 = pC + (z)*MMUL::size_C;
    float *__restrict pC2 = pC + ((z + 1)) * MMUL::size_C;
    Q4K_MM_LOOP
    for (unsigned j = 0; j < colB; j += 2) {
      const bf16 *__restrict pA1 = pA + (z)*MMUL::size_A;
      const bf16 *__restrict pA2 = pA + ((z + 1)) * MMUL::size_A;
      const bf16 *__restrict pB1 = pB + (j)*colA * MMUL::size_B;
      const bf16 *__restrict pB2 = pB + (j + 1) * colA * MMUL::size_B;

      MMUL C00(aie::load_v<MMUL::size_C>(pC1));
      MMUL C01(aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C * rowA));
      MMUL C10(aie::load_v<MMUL::size_C>(pC2));
      MMUL C11(aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C * rowA));

      AIE_PREPARE_FOR_PIPELINING
      AIE_LOOP_RANGE(colA, colA)
      Q4K_MM_ILOOP
      for (unsigned i = 0; i < colA; ++i) {
        aie::vector<bf16, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA1);
        pA1 += rowA * MMUL::size_A;
        aie::vector<bf16, MMUL::size_A> A1 = aie::load_v<MMUL::size_A>(pA2);
        pA2 += rowA * MMUL::size_A;
        aie::vector<bf16, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB1);
        pB1 += MMUL::size_B;
        aie::vector<bf16, MMUL::size_B> B1 = aie::load_v<MMUL::size_B>(pB2);
        pB2 += MMUL::size_B;
        C00.mac(A0, B0);
        C01.mac(A0, B1);
        C10.mac(A1, B0);
        C11.mac(A1, B1);
      }
      aie::store_v(pC1, C00.template to_vector<float>());
      pC1 += MMUL::size_C * rowA;
      aie::store_v(pC1, C01.template to_vector<float>());
      pC1 += MMUL::size_C * rowA;
      aie::store_v(pC2, C10.template to_vector<float>());
      pC2 += MMUL::size_C * rowA;
      aie::store_v(pC2, C11.template to_vector<float>());
      pC2 += MMUL::size_C * rowA;
    }
  }
}

// 2x4 register blocking: two A tiles and FOUR B tiles, eight accumulators.
//
// WHY TRY IT. At batch 16 with MROWS 32 the operand roles leave rowA=2 and
// colB=4, so 2x2 runs one z iteration and two j iterations: 2 x colA x (2 A
// loads + 2 B loads) = 4*colA loads for 4*colA macs. Covering all four B tiles
// at once is one group instead of two: colA x (2 A + 4 B) = 6*colA loads for
// 8*colA macs -- 0.75 loads per mac instead of 1.
//
// MEASURED: IT DOES NOT PAY. Eight accumulators cost more than the loads they
// save, at both batches:
//
//   batch 16   2x2 1965   2x4 2035   (+3.6%)
//   batch 32   2x2 3533   2x4 3786   (+7.2%)
//
// Kept so the idea is not re-tried. 2x2 is the right blocking, which also means
// the 10% the operand swap costs at batch 16 cannot be re-blocked away.
template <int MROWS, int KCOL, int BATCH>
static inline void q4k_mmul_2x4(const bf16 *__restrict pA,
                                const bf16 *__restrict pB,
                                float *__restrict pC) {
  constexpr int r = 8, s = 8, t = 8;
  using MMUL = aie::mmul<r, s, t, bf16, bf16, accauto>;
  constexpr int rowA = BATCH / r;
  constexpr int colA = KCOL / s;
  constexpr int colB = MROWS / t;
  static_assert(rowA % 2 == 0, "BATCH must be a multiple of 16");
  static_assert(colB % 4 == 0, "MROWS must be a multiple of 32");
  constexpr int SC = MMUL::size_C;

  Q4K_MM_LOOP
  for (unsigned z = 0; z < rowA; z += 2) {
    Q4K_MM_LOOP
    for (unsigned j = 0; j < colB; j += 4) {
      const bf16 *__restrict pA1 = pA + (z)*MMUL::size_A;
      const bf16 *__restrict pA2 = pA + (z + 1) * MMUL::size_A;
      const bf16 *__restrict pB0 = pB + (j + 0) * colA * MMUL::size_B;
      const bf16 *__restrict pB1 = pB + (j + 1) * colA * MMUL::size_B;
      const bf16 *__restrict pB2 = pB + (j + 2) * colA * MMUL::size_B;
      const bf16 *__restrict pB3 = pB + (j + 3) * colA * MMUL::size_B;
      float *__restrict q = pC + ((j)*rowA + z) * SC;

      MMUL C00(aie::load_v<SC>(q + 0 * SC * rowA));
      MMUL C10(aie::load_v<SC>(q + 0 * SC * rowA + SC));
      MMUL C01(aie::load_v<SC>(q + 1 * SC * rowA));
      MMUL C11(aie::load_v<SC>(q + 1 * SC * rowA + SC));
      MMUL C02(aie::load_v<SC>(q + 2 * SC * rowA));
      MMUL C12(aie::load_v<SC>(q + 2 * SC * rowA + SC));
      MMUL C03(aie::load_v<SC>(q + 3 * SC * rowA));
      MMUL C13(aie::load_v<SC>(q + 3 * SC * rowA + SC));

      AIE_PREPARE_FOR_PIPELINING
      AIE_LOOP_RANGE(colA, colA)
      Q4K_MM_ILOOP
      for (unsigned i = 0; i < colA; ++i) {
        aie::vector<bf16, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA1);
        pA1 += rowA * MMUL::size_A;
        aie::vector<bf16, MMUL::size_A> A1 = aie::load_v<MMUL::size_A>(pA2);
        pA2 += rowA * MMUL::size_A;
        aie::vector<bf16, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB0);
        pB0 += MMUL::size_B;
        aie::vector<bf16, MMUL::size_B> B1 = aie::load_v<MMUL::size_B>(pB1);
        pB1 += MMUL::size_B;
        aie::vector<bf16, MMUL::size_B> B2 = aie::load_v<MMUL::size_B>(pB2);
        pB2 += MMUL::size_B;
        aie::vector<bf16, MMUL::size_B> B3 = aie::load_v<MMUL::size_B>(pB3);
        pB3 += MMUL::size_B;
        C00.mac(A0, B0);
        C10.mac(A1, B0);
        C01.mac(A0, B1);
        C11.mac(A1, B1);
        C02.mac(A0, B2);
        C12.mac(A1, B2);
        C03.mac(A0, B3);
        C13.mac(A1, B3);
      }
      aie::store_v(q + 0 * SC * rowA, C00.template to_vector<float>());
      aie::store_v(q + 0 * SC * rowA + SC, C10.template to_vector<float>());
      aie::store_v(q + 1 * SC * rowA, C01.template to_vector<float>());
      aie::store_v(q + 1 * SC * rowA + SC, C11.template to_vector<float>());
      aie::store_v(q + 2 * SC * rowA, C02.template to_vector<float>());
      aie::store_v(q + 2 * SC * rowA + SC, C12.template to_vector<float>());
      aie::store_v(q + 3 * SC * rowA, C03.template to_vector<float>());
      aie::store_v(q + 3 * SC * rowA + SC, C13.template to_vector<float>());
    }
  }
}

// SMALL BATCHES. q4k_mmul static_asserts BATCH % 16 == 0, because its 2x2
// blocking steps z by 2 and rowA = BATCH/8. That was fine while the plan said
// batch 16. It is not fine now: pricing the DFlash iteration against
// max(compute, memory) puts the best block size at 3-5, not 16, and nothing in
// that sweep puts it above 6. So the batch the analysis actually recommends is
// one the kernel refuses to compile.
//
// This is that kernel. rowA is 1, so there is no z to block over and the
// blocking goes entirely into the weight rows: 1x4, four accumulators, same
// register pressure as 2x2. The A operand is loaded once and fed to all four.
//
//   BATCH 4  -> aie::mmul<4,8,8>, size_A = size_C = 32
//   BATCH 8  -> aie::mmul<8,8,8>, size_A = size_C = 64
//
// The load ratio is worse than 2x2 -- 5 loads per 4 macs against 4 per 4 -- and
// there is no way around it at rowA = 1: one A tile simply cannot feed more
// than colB weight tiles. That cost is real and bench_q4k_mm.py --batches 4,8
// measures it. What buys it back is that a small block needs far fewer tokens'
// worth of everything else.
//
// Layouts are the SAME as q4k_mmul's, with r = 4 or 8 substituted:
//   A tile (0, i) at i*size_A          (rowA = 1, so A is contraction-major)
//   B tile (i, j) at (j*colA + i)*64   (unchanged -- the unpack is untouched)
//   C tile (0, j) at j*size_C          (contiguous)
template <int MROWS, int KCOL, int BATCH>
static inline void q4k_mmul_small(const bf16 *__restrict pA,
                                  const bf16 *__restrict pB,
                                  float *__restrict pC) {
  constexpr int r = (BATCH < 8) ? 4 : 8, s = 8, t = 8;
  using MMUL = aie::mmul<r, s, t, bf16, bf16, accauto>;
  constexpr int rowA = BATCH / r;
  constexpr int colA = KCOL / s;
  constexpr int colB = MROWS / t;
  static_assert(BATCH == 4 || BATCH == 8,
                "q4k_mmul_small covers rowA == 1; use q4k_mmul at 16 and above");
  static_assert(rowA == 1, "rowA must be 1 here");
  static_assert(colB % 4 == 0, "MROWS must be a multiple of 32 for 1x4");

  Q4K_MM_LOOP
  for (unsigned j = 0; j < colB; j += 4) {
    const bf16 *__restrict pA1 = pA;
    const bf16 *__restrict pB0 = pB + (j + 0) * colA * MMUL::size_B;
    const bf16 *__restrict pB1 = pB + (j + 1) * colA * MMUL::size_B;
    const bf16 *__restrict pB2 = pB + (j + 2) * colA * MMUL::size_B;
    const bf16 *__restrict pB3 = pB + (j + 3) * colA * MMUL::size_B;
    float *__restrict q = pC + j * MMUL::size_C;

    // Named locals, not an array -- see q4k_mmul: MMUL C[4] goes to the stack
    // and overflows AIE2's load/store displacement field.
    MMUL C0(aie::load_v<MMUL::size_C>(q + 0 * MMUL::size_C));
    MMUL C1(aie::load_v<MMUL::size_C>(q + 1 * MMUL::size_C));
    MMUL C2(aie::load_v<MMUL::size_C>(q + 2 * MMUL::size_C));
    MMUL C3(aie::load_v<MMUL::size_C>(q + 3 * MMUL::size_C));

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(colA, colA)
    Q4K_MM_ILOOP
    for (unsigned i = 0; i < colA; ++i) {
      aie::vector<bf16, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA1);
      pA1 += MMUL::size_A; // rowA == 1
      C0.mac(A0, aie::load_v<MMUL::size_B>(pB0));
      pB0 += MMUL::size_B;
      C1.mac(A0, aie::load_v<MMUL::size_B>(pB1));
      pB1 += MMUL::size_B;
      C2.mac(A0, aie::load_v<MMUL::size_B>(pB2));
      pB2 += MMUL::size_B;
      C3.mac(A0, aie::load_v<MMUL::size_B>(pB3));
      pB3 += MMUL::size_B;
    }
    aie::store_v(q + 0 * MMUL::size_C, C0.template to_vector<float>());
    aie::store_v(q + 1 * MMUL::size_C, C1.template to_vector<float>());
    aie::store_v(q + 2 * MMUL::size_C, C2.template to_vector<float>());
    aie::store_v(q + 3 * MMUL::size_C, C3.template to_vector<float>());
  }
}

// Dispatch to whichever of the two the batch supports, so callers do not have
// to know where the 2x2 blocking stops working.
//
// always_inline so this leaves no section of its own. Without it clang outlines
// the dispatcher, inlines the real body INTO it, and the bench then measures a
// differently-inlined copy -- batch 16 read 2354 bundles through the dispatcher
// against 1965 called directly, a 20% swing that is an artifact of where the
// inliner stopped rather than anything about the kernel.
template <int MROWS, int KCOL, int BATCH>
[[clang::always_inline]] static inline void
q4k_mmul_any(const bf16 *__restrict pA, const bf16 *__restrict pB,
             float *__restrict pC) {
  if constexpr (BATCH < 16)
    q4k_mmul_small<MROWS, KCOL, BATCH>(pA, pB, pC);
  else
    q4k_mmul<MROWS, KCOL, BATCH>(pA, pB, pC);
}

// One weight block, batched: unpack then multiply.
//
// ACTIVATIONS FIRST. q4k_mmul's first operand is pA and pA is the activations
// -- the unpack emits the weights in mmul's B order precisely so they can go in
// the second slot (see the q4k_unpack_block header for why that assignment is
// forced). This call read `(W, B, C)` until the device gate caught it: the
// layout half of the operand swap landed, the call sites did not. The wrong
// order is not a small error, it multiplies unrelated operands and walks off
// the end of the activation buffer, because pB is indexed to colB*colA*64 and
// an activation tile only has rowA*colA*64 elements.
template <int MROWS, int KCOL, int BATCH>
static inline void q4k_mm_block(const q4k_block_t *A, const bf16 *__restrict B,
                                float *__restrict C, bf16 *__restrict W) {
  q4k_unpack_block<MROWS, KCOL>(A, W);
  q4k_mmul_any<MROWS, KCOL, BATCH>(B, W, C);
}

// FUSING THE UNPACK INTO THE MMUL IS A MEASURED LOSS. Do not re-derive it.
//
// The idea: q4k_mm_block materialises a whole MROWS x KCOL block into W before
// the mmul reads a byte of it, so every weight block makes a 16 KB round trip
// through the same L1 the mmul streams its A operand from. Consume each 64-lane
// B tile in the mac that needs it instead, and W disappears. It lines up
// exactly -- q4k_unpack_block's step i = R*NCB + cb and NCB == colA, so steps
// cb and NCB+cb are precisely the four B tiles contraction block cb needs --
// and at rowA = 1 the B-tile reuse it would trade away is already 1, so it
// looks free.
//
// IT IS NOT FREE. The four aie::mmul<8,8,8> accumulators are 16 bm registers,
// which is the ENTIRE file, and q4k_unpack_step's aie::accum<accfloat,128>
// wants the same registers. Peano spills all four accumulators to the stack and
// reloads them every iteration: 64 sp references in the loop body against 0 for
// the unfused pair. Splitting into two half-passes of two accumulators each --
// which duplicates no unpack work, only the A operand read -- still spills one
// accumulator per iteration. Static bundles per weight block, disassembling
// proj_qmm_mm_acc:
//
//   unfused   2784   (unpack 32 x 64  +  mmul 23 x 32)
//   fused     3616   (+29.9%)
//
// And the premise was wrong anyway. Of the 31 bundles in the unpack loop body
// only FOUR are the W stores, so removing them could never have been worth more
// than ~6%. The real content is format conversion: 17 of 68 ops are
// vconv.bf16.fp32 / vconv.fp32.bf16 -- see Q4K_UNPACK_FMA above, which attacks
// that instead and wins.
//
// THE "14.3 ms OF CONTENTION" READING OF THE PROBE SPLIT IS ALSO WRONG. Those
// marginal costs above the 70.990 ms memory floor (qwen3-4b batch 8, L=128,
// mode 3 + PROJ_PP_ONLY=w) are
//
//   unpack alone (PROBE=3) 20.71    mmul alone (PROBE=2) 2.11    both 37.08
//
// and 20.71 + 2.11 = 22.8 does fall 14.3 short of 37.08 -- but that is ordinary
// latency hiding, not L1 contention. Per weight block the dispatch costs
// max(DMA, core). The mmul's true core cost is 16.37 ms; run alone it hides
// almost entirely under the weight DMA and only 2.11 is exposed, and run after
// the unpack has already consumed the DMA shadow all 16.37 is exposed.
// 37.08 - 20.71 = 16.37 exactly. There is no contention term to reclaim, and
// the headroom to the floor is core time, all of it.
//
// NCHUNK blocks along the contraction, accumulated into one C, through ONE
// KCOL-wide scratch that each chunk overwrites.
//
// WHY. Widening KCOL to 512 measured 13.4% off the multiply, but it doubles the
// scratch to 32 KB and that does not fit the proj core (57.5 of 54 KB). This is
// the same total work at the same 16 KB scratch. Whether it also keeps the gain
// depends on where the gain came from, which is the open question:
//
//   if it is the C traffic     -- q4k_mmul does load_v(pC) on entry and
//                                 store_v(pC) on exit, so NCHUNK chunks pay it
//                                 NCHUNK times and this form saves nothing
//   if it is the schedule      -- the bodies are fully unrolled and adjacent, so
//                                 the compiler sees one long straight-line
//                                 region either way and this form keeps it all
//
// bench_q4k_mm.py --chunks 2 runs it. IT DOES NOT BUILD, and the reason is
// worth keeping: two inlined fully-unrolled bodies need a frame past AIE2's
// 16-bit load/store displacement field --
//
//   fatal error: error in backend: immediate operand value -33152 is out of
//                range [-32768, -64]
//   Running pass 'AIE2 Assembly Printer' on q4k_mm_chunked<32, 128, 16, 2>
//
// It fails at kcol 128 too, where the two chunks are HALF the code of the
// kcol-256 single body that compiles, so it is frame size and not code size.
// Rolling the loop compiles but makes both variants emit an identical body, so
// the static-bundle method cannot answer this at all -- it needs on-device
// timing. Do not assume the answer in the meantime: the mechanism was guessed
// once already and this is what was meant to settle it.
template <int MROWS, int KCOL, int BATCH, int NCHUNK>
static inline void q4k_mm_chunked(const q4k_block_t *A,
                                  const bf16 *__restrict B,
                                  float *__restrict C, bf16 *__restrict W) {
  Q4K_MM_LOOP
  for (int c = 0; c < NCHUNK; c++) {
    // Chunk-major B: each chunk's activation tile is contiguous, which is how
    // inX delivers it (one COL_BLOCK chunk per weight block).
    //
    // always_inline is load-bearing, not a hint. At full unroll each body is
    // ~1800 bundles, so clang's size heuristic outlines them and emits ONE
    // shared copy called NCHUNK times -- which measures the outlining decision
    // (exactly NCHUNK x the single cost, no cross-body scheduling) instead of
    // the question being asked. Forcing them into one body is the point.
    // `(const q4k_block_t *)(w + c*Q4K_BLOCK_BF16)`, not `A + c` -- the struct
    // is 9216 bytes and a packed block is 5120. See Q4K_BLOCK_BF16 above.
    [[clang::always_inline]] q4k_unpack_block<MROWS, KCOL>(
        (const q4k_block_t *)((const bf16 *)A + c * Q4K_BLOCK_BF16), W);
    // Activations first -- see q4k_mm_block.
    [[clang::always_inline]] q4k_mmul<MROWS, KCOL, BATCH>(
        B + c * KCOL * BATCH, W, C);
  }
}

#endif // __Q4K_MM_H__
