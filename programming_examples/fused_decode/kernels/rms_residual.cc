// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
#include "aie_array_layout.h"
#include "aie_kernel_utils.h"
#include "typedef.hpp"

///@brief  rms norm: y = nrom(x) * w;
///@brief                    x_buf = x
///@param y: output
///@param x_buf: copy of x
///@param x: input
///@param w: weight
/// 1 / sqrt(mean(x^2) + eps) over MODEL_DIM elements -- the whole of rmsnorm
/// except the final scale. Split out because the batched entry points below
/// need the factor WITHOUT the multiply pass: they regenerate the normalized
/// row a slice at a time rather than storing it.
static inline float rms_rsqrt(const bf16 *restrict x) {
  bf16 *it_x = const_cast<bf16 *>(x);
  const float epsilon = 1e-6;

  constexpr int vector_size = 16;
  constexpr float one_over_D = 1.0f / (float)MODEL_DIM;

  aie::accum<accfloat, vector_size> sum_squares = aie::zeros<accfloat>();
  aie::vector<float, vector_size> mx_vec;

  const int F = MODEL_DIM / vector_size;
  for (int i = 0; i < F; i++) {
    auto x_vec = aie::load_v<vector_size>(it_x);
    sum_squares = aie::mac_square(sum_squares, x_vec);
    it_x += vector_size;
  }

  mx_vec = sum_squares.template to_vector<float>();
  float sum = aie::reduce_add(mx_vec);
  sum = sum * one_over_D;
  // bf16 mean_square = aie::div(sum, bf16(D));
  // bf16 rms = aie::sqrt(mean_square + epsilon);
  // bf16 divrms = aie::div(1, (bf16)32);
  // compute 1 / sqrt(sum + epsilon), use magic
  sum = sum + epsilon;
  float x2, divrms;
  const float threehalfs = 1.5F;
  uint32_t i_u32;

  x2 = sum * 0.5F;
  divrms = sum;
  i_u32 = *(uint32_t *)&divrms;      // evil floating point bit level hacking
  i_u32 = 0x5f3759df - (i_u32 >> 1); // what the fuck?
  divrms = *(float *)&i_u32;
  divrms = divrms * (threehalfs - (x2 * divrms * divrms)); // 1st iteration
  divrms =
      divrms * (threehalfs -
                (x2 * divrms * divrms)); // 2nd iteration, this can be removed
  // bf16 divrms_bf16 = (bf16)divrms;
  return divrms;
}

void rms_norm(bf16 *restrict y, const bf16 *restrict x,
              const bf16 *restrict w) {
  bf16 *it_y = const_cast<bf16 *>(y);
  bf16 *it_x = const_cast<bf16 *>(x);
  bf16 *it_w = const_cast<bf16 *>(w);

  constexpr int vector_size = 16;
  const int F = MODEL_DIM / vector_size;
  const float divrms = rms_rsqrt(x);

  for (int i = 0; i < F; i++) {
    aie::vector<bf16, vector_size> x_vec = aie::load_v<vector_size>(it_x);
    aie::vector<bf16, vector_size> w_vec = aie::load_v<vector_size>(it_w);
    aie::vector<float, vector_size> wx_vec = aie::mul(x_vec, w_vec);
    aie::vector<bf16, vector_size> o_vec = aie::mul(wx_vec, divrms);
    aie::store_v(it_y, o_vec);
    it_x += vector_size;
    it_y += vector_size;
    it_w += vector_size;
  }
  event1();
}

///@brief residual add: y = x + x_buf, and x_buf = y
///@param y: output
///@param x: input
///@param x_buf: copy of x
void residual_add(bf16 *restrict y, const bf16 *restrict x_buf,
                  const bf16 *restrict x) {
  bf16 *it_x = const_cast<bf16 *>(x);
  bf16 *it_y = const_cast<bf16 *>(y);
  bf16 *it_x_buf = const_cast<bf16 *>(x_buf);
  constexpr int vector_size = 16;

  for (int i = 0; i < MODEL_DIM / vector_size; i++) {
    aie::vector<bf16, vector_size> x_vec = aie::load_v<vector_size>(it_x);
    aie::vector<bf16, vector_size> x_buf_vec =
        aie::load_v<vector_size>(it_x_buf);
    auto out_vec = aie::add(x_vec, x_buf_vec);
    aie::store_v(it_y, out_vec);
    aie::store_v(it_x_buf, out_vec);
    it_x += vector_size;
    it_y += vector_size;
    it_x_buf += vector_size;
  }
}

///@brief

#ifdef RMS_DELAY
// CRITICAL-PATH PROBE for the RMS CORE, the mirror of proj_qmm.cc's PROJ_DELAY.
//
// WHY THIS CORE. The projection is 16 cores wide, but rms is ONE tile, and its
// work is proportional to BATCH*K: it norms every token, regenerates every
// @xnorm refeed chunk for the projection engine, and consumes both residual
// stages. Nothing about it parallelises with the batch, so it is the natural
// candidate for a batch-8 serial bottleneck -- and unlike the projection, its
// cost has never been separated from the dispatch.
//
// Read the caveats above PROJ_DELAY in proj_qmm.cc before reading a sweep of
// this: register-only work hides in spare VLIW slots at small counts, so a
// shallow response near zero is not proof of slack. What a 1:1 response DOES
// show is that this core is on the critical path.
//
// Xorshift, not a multiply-add: s = a*s + c is affine and Peano composes N of
// them into one.
static volatile unsigned rms_probe_seed = 2463534242u;
static volatile unsigned rms_probe_sink;
static inline void rms_probe_delay() {
  unsigned s = rms_probe_seed;
  for (int i = 0; i < RMS_DELAY; i++) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
  }
  rms_probe_sink = s;
}
#endif

extern "C" {

// AIR-friendly pure-compute entry points (no in-kernel locks; AIR owns sync),
// mirroring proj_qmm.cc's split of the reference proj_main. Used by the AIR
// decode vehicle (proj_qmm_demux_fb.py) for the post-attention residual +
// RMSNorm stage.
//   rms_norm_aie:     y = rmsnorm(x) * w           (y,x,w each MODEL_DIM bf16)
//   residual_add_aie: y = x_buf + x ; x_buf = y    (MODEL_DIM bf16)
void rms_norm_aie(bf16 *restrict y, bf16 *restrict x, bf16 *restrict w,
                  int _arm) {
  aie_round_nearest_even();
  (void)_arm; // per-token RTP arm-gate operand (kept alive so AIR emits the arm
              // lock)
  rms_norm(y, x, w);
}

// 2K-weight variants: w holds TWO MODEL_DIM norm weights back-to-back; _lo uses
// the first, _hi the second (w + MODEL_DIM). Lets one packet channel carry two
// norms so the Gemma 4-norm rms tile keeps <=4 packet ids per S2MM port.
void rms_norm_lo_aie(bf16 *restrict y, bf16 *restrict x, bf16 *restrict w,
                     int _arm) {
  aie_round_nearest_even();
  (void)_arm;
  rms_norm(y, x, w);
}
void rms_norm_hi_aie(bf16 *restrict y, bf16 *restrict x, bf16 *restrict w,
                     int _arm) {
  aie_round_nearest_even();
  (void)_arm;
  rms_norm(y, x, w + MODEL_DIM);
}

// Header-bearing variant for the PACKET x-feed (STAGE_MLP>=2 convergence):
// write the AIE packet header (pkt_id) at y+14 and the rmsnorm payload at y+16,
// matching the reference rms_residual + proj_qmm.cc's header convention. The
// packet is then streamed from offset 14 (header-first) so the switchbox can
// route it.
void rms_norm_aie_hdr(bf16 *restrict y, bf16 *restrict x, bf16 *restrict w,
                      unsigned int pkt_id) {
  aie_round_nearest_even();
  *reinterpret_cast<unsigned int *>(y + 14) = pkt_id;
  rms_norm(y + 16, x, w);
}

void residual_add_aie(bf16 *restrict y, bf16 *restrict x_buf,
                      bf16 *restrict x) {
  aie_round_nearest_even();
  residual_add(y, x_buf, x);
}

// ---- DECODE_BATCH > 1 -----------------------------------------------------
// The rms core cannot hold two batches. One BATCH*MODEL_DIM bf16 buffer is
// 32 KB at batch 8 against a ~54 KB L1 budget, so the raw batch and the
// normalized batch cannot both be resident -- and the raw one has to survive,
// because it is what the post-attention residual adds into.
//
// So the normalized batch is NEVER MATERIALIZED. rms_scale_row_aie records one
// float per row (the 1/rms factor), and rms_chunk_aie regenerates whatever
// slice of the normalized batch is being streamed out at that moment, for all
// BATCH rows at once, into a small staging buffer. The big buffer stays raw
// and doubles as the residual accumulator: x -> h -> layer output.
//
// The cost is that a slice is recomputed once per re-broadcast round rather
// than once per token, which is a multiply pass over BATCH*MODEL_DIM per
// round. It buys the batch: without it the core needs two 32 KB buffers.
//
// Layout convention, shared with the descriptors in xfeed_bd.py: the big
// buffer is row-major [BATCH][MODEL_DIM]; the staging buffer is [BATCH][n],
// which is exactly the (token, element) window one @xnorm chunk carries.

/// scales[t] = 1 / sqrt(mean(x[t]^2) + eps) for row t of a [BATCH][MODEL_DIM].
/// `_arm` is the per-token RTP gate operand, kept alive for the same reason
/// rms_norm_aie keeps it: AIR emits the arm lock from the use.
void rms_scale_row_aie(float *restrict scales, bf16 *restrict x, int t,
                       int _arm) {
  (void)_arm;
  scales[t] = rms_rsqrt(x + t * MODEL_DIM);
}

// ---- band-streamed residual (RMS_BAND_STREAM) ------------------------------
// rms_rsqrt needs the WHOLE MODEL_DIM row at once to sum its squares. Once the
// raw residual is banded (one [BATCH][n] band resident at a time instead of
// [BATCH][MODEL_DIM]), no single call sees the whole row, so the sum has to be
// built up across a band loop and closed out separately. Unused unless
// fused_decode.py's band-streamed path calls them.

/// Add row t's sum-of-squares over ONE band (width n, at `off` within the
/// row) into scales[t]. `row_stride` is x's true per-token pitch -- MODEL_DIM
/// if x is still the full resident row (the transitional, not-yet-DMA-banded
/// caller), or n if x is a tightly-packed [BATCH][n] band buffer (the eventual
/// one-band-resident caller), mirroring how residual_acc_row_aie's `acc`
/// (MODEL_DIM stride) and `x` (n stride) already differ for the same reason.
/// Call once per band, in band order; `first` nonzero on the first band of a
/// row zeroes the accumulator instead of adding to whatever was there before
/// (there is no separate zero-init call). Follow with ONE
/// rms_scale_row_finalize_aie per row after the last band.
void rms_scale_row_partial_aie(float *restrict scales, bf16 *restrict x,
                               int t, int row_stride, int off, int n,
                               int first, int _arm) {
  (void)_arm;
  constexpr int vector_size = 16;
  bf16 *it_x = const_cast<bf16 *>(x) + t * row_stride + off;
  aie::accum<accfloat, vector_size> sum_squares = aie::zeros<accfloat>();
  for (int i = 0; i < n / vector_size; i++) {
    auto x_vec = aie::load_v<vector_size>(it_x);
    sum_squares = aie::mac_square(sum_squares, x_vec);
    it_x += vector_size;
  }
  aie::vector<float, vector_size> mx_vec =
      sum_squares.template to_vector<float>();
  float sum = aie::reduce_add(mx_vec);
  scales[t] = first ? sum : scales[t] + sum;
}

/// Turn row t's accumulated sum-of-squares (built up over every band by
/// rms_scale_row_partial_aie) into 1/sqrt(mean(x^2)+eps) in place. This is
/// rms_rsqrt's fast-inverse-sqrt tail, duplicated rather than shared: pulling
/// it out from under rms_rsqrt would change that function's own codegen, and
/// nothing else here is supposed to.
void rms_scale_row_finalize_aie(float *restrict scales, int t, int _arm) {
  (void)_arm;
  const float epsilon = 1e-6f;
  const float one_over_D = 1.0f / (float)MODEL_DIM;
  float sum = scales[t] * one_over_D + epsilon;
  float x2, divrms;
  const float threehalfs = 1.5f;
  uint32_t i_u32;
  x2 = sum * 0.5f;
  divrms = sum;
  i_u32 = *(uint32_t *)&divrms;      // evil floating point bit level hacking
  i_u32 = 0x5f3759df - (i_u32 >> 1); // what the fuck?
  divrms = *(float *)&i_u32;
  divrms = divrms * (threehalfs - (x2 * divrms * divrms)); // 1st iteration
  divrms =
      divrms * (threehalfs -
                (x2 * divrms * divrms)); // 2nd iteration, this can be removed
  scales[t] = divrms;
}

/// rms_scale_row_partial_aie with row_stride == off == n -- i.e. `x` is a
/// fresh, tightly-packed [batch][n] band fetch, not an offset into a
/// MODEL_DIM-strided resident row. Its own symbol (not the same one called
/// with row_stride=n, off=0) purely so the MLIR-level FuncOp declaration can
/// give it a band-sized memref type distinct from rms_scale_row_partial_aie's
/// whole-row one -- func.call requires the two to match exactly, and one
/// symbol can't have two declared signatures. Mirrors
/// rms_chunk_banded_aie/residual_acc_row_banded_aie's existing split.
void rms_scale_row_partial_banded_aie(float *restrict scales,
                                      bf16 *restrict x, int t, int n,
                                      int first, int _arm) {
  rms_scale_row_partial_aie(scales, x, t, n, 0, n, first, _arm);
}

/// One chunk of the normalized batch, for every row:
///   y[t*n + i] = x[t*MODEL_DIM + c*n + i] * w[c*n + i] * scales[t]
/// Call once per chunk per re-broadcast round; `y` is then put on @xnorm whole.
static inline void rms_chunk(bf16 *restrict y, bf16 *restrict x,
                             const bf16 *restrict w,
                             const float *restrict scales, int batch, int c,
                             int n) {
  constexpr int vector_size = 16;
  const bf16 *w_base = w + c * n;
  for (int t = 0; t < batch; t++) {
    const bf16 *it_x = x + t * MODEL_DIM + c * n;
    const bf16 *it_w = w_base;
    bf16 *it_y = y + t * n;
    const float s = scales[t];
    // n is a runtime argument, so the compiler sees an unknown trip count on a
    // loop that is always XCHUNK/16 = 32. Under Peano AIE_PREPARE_FOR_PIPELINING
    // expands to NOTHING -- it is Chess-only, see aie_kernel_utils.h -- so the
    // hints that do anything on this toolchain are the unroll and the range.
    AIE_LOOP_RANGE(8)
    AIE_LOOP_UNROLL(8)
    for (int i = 0; i < n / vector_size; i++) {
      aie::vector<bf16, vector_size> x_vec = aie::load_v<vector_size>(it_x);
      aie::vector<bf16, vector_size> w_vec = aie::load_v<vector_size>(it_w);
      aie::vector<float, vector_size> wx_vec = aie::mul(x_vec, w_vec);
      aie::vector<bf16, vector_size> o_vec = aie::mul(wx_vec, s);
      aie::store_v(it_y, o_vec);
      it_x += vector_size;
      it_w += vector_size;
      it_y += vector_size;
    }
  }
}

void rms_chunk_aie(bf16 *restrict y, bf16 *restrict x, bf16 *restrict w,
                   float *restrict scales, int batch, int c, int n) {
#ifdef RMS_DELAY
  rms_probe_delay();
#endif
// == 1, not ifdef: probe 2 below is a distinct variant and an ifdef here would
// swallow it.
#if defined(RMS_CHUNK_PROBE) && RMS_CHUNK_PROBE == 1
  // Diagnostic builds only: make row t of the X feed the CONSTANT (t+1)/8, so
  // every projection output row comes out proportional to t+1. Reading the KV
  // cache then says what row 7 of the mmul's A operand actually was -- 8x row 0
  // if the feed is right, 7x if it got the previous token's, 0 if it got
  // nothing. The flush-side labels (PROJ_FLUSH_PROBE=3) proved the descriptor
  // chain BELOW the accumulator; this is the hop above it.
  constexpr int vector_size = 16;
  for (int t = 0; t < batch; t++) {
    const auto c = aie::broadcast<bf16, vector_size>((bf16)((float)(t + 1) * 0.125f));
    bf16 *it_y = y + t * n;
    for (int i = 0; i < n / vector_size; i++)
      aie::store_v(it_y + i * vector_size, c);
  }
  (void)x;
  (void)w;
  (void)scales;
  (void)c;
#elif defined(RMS_CHUNK_PROBE) && RMS_CHUNK_PROBE == 2
  // TIMING ONLY -- delete the chunk regeneration outright and leave y as it
  // was.
  //
  // A TRUE DELETION, AND ON THIS TOOLCHAIN THAT IS THE ONLY KIND WORTH HAVING.
  // Probe 1 above looks like it should be cheaper than what it replaces: it
  // drops both loads and both multiplies and only stores a constant. Measured,
  // it is 250.1 ms against a 163.2 ms baseline -- 87 ms SLOWER -- because it is
  // written as a plain C loop and rms_chunk is written with aie:: intrinsics.
  //
  // PEANO DOES NOT AUTO-VECTORISE. Not this loop, not any loop: a bare
  //   for (int i = 0; i < 4096; i++) y[i] = 0.125f;
  // with a compile-time trip count, a __restrict pointer and plain float emits
  // ZERO vector instructions, at -O2 and at -O3 -fvectorize -fslp-vectorize
  // alike. There is no -fno-vectorize in the build flags; the aie2p backend
  // simply does not do it. Every vector instruction in these kernels exists
  // because someone wrote aie::load_v / aie::store_v / aie::mul by hand
  // (rms_chunk: 34 vector ops in 95 instructions; probe 1: 5 in 53, and none of
  // them a vector store).
  //
  // So a diagnostic that SUBSTITUTES plain C for an intrinsic kernel is not
  // measuring the kernel, it is measuring a 16x deoptimisation of it. That trap
  // has now been walked into three times -- PROJ_MM_PROBE=1's y_acc copy loop,
  // this probe 1, and the reading of both as "the arithmetic is only 2%". A
  // probe that deletes work and writes nothing cannot fall into it.
  //
  // This is the deletion counterpart of the RMS_DELAY sweep. Injection says
  // this core has no absorbing region; only deletion says how much of the
  // dispatch it is.
  (void)y;
  (void)x;
  (void)w;
  (void)scales;
  (void)batch;
  (void)c;
  (void)n;
#else
  rms_chunk(y, x, w, scales, batch, c, n);
#endif
}

// 2K-weight variants, matching rms_norm_lo_aie / rms_norm_hi_aie: one packet
// channel carries two norm weights back to back.
void rms_chunk_lo_aie(bf16 *restrict y, bf16 *restrict x, bf16 *restrict w,
                      float *restrict scales, int batch, int c, int n) {
  rms_chunk(y, x, w, scales, batch, c, n);
}
void rms_chunk_hi_aie(bf16 *restrict y, bf16 *restrict x, bf16 *restrict w,
                      float *restrict scales, int batch, int c, int n) {
  rms_chunk(y, x + 0, w + MODEL_DIM, scales, batch, c, n);
}

/// acc[t*MODEL_DIM + off + i] += x[t*n + i] -- one round of a projection's
/// output added into row t of the residual accumulator. The projection egresses
/// (round, token), so a whole round arrives as a [BATCH][n] staging buffer and
/// lands at a fixed `off` inside every row.
void residual_acc_row_aie(bf16 *restrict acc, bf16 *restrict x, int t, int off,
                          int n) {
  constexpr int vector_size = 16;
  bf16 *it_a = acc + t * MODEL_DIM + off;
  bf16 *it_x = x + t * n;
  for (int i = 0; i < n / vector_size; i++) {
    aie::vector<bf16, vector_size> a_vec = aie::load_v<vector_size>(it_a);
    aie::vector<bf16, vector_size> x_vec = aie::load_v<vector_size>(it_x);
    aie::store_v(it_a, aie::add(a_vec, x_vec));
    it_a += vector_size;
    it_x += vector_size;
  }
}

// ---- RMS_BAND_STREAM level 3 (not yet wired in) ---------------------------
// rms_chunk / residual_acc_row above take a MODEL_DIM-strided `x`/`acc` --
// correct only while the resident buffer is the whole row. Level 3 shrinks it
// to one band, fetched fresh per call, so both need an n-strided variant
// instead: the buffer passed in IS the band, already offset to start at 0, so
// no MODEL_DIM stride and no extra `c*n`/`off` term on that side.

/// rms_chunk, but `x` is a fresh [batch][n] band fetch (row_stride=n) instead
/// of an offset into a MODEL_DIM-strided resident row. `w` is unchanged: the
/// norm weight stays fully resident regardless of level, so its `c*n` offset
/// into the whole K-wide weight is still correct.
static inline void rms_chunk_banded(bf16 *restrict y, bf16 *restrict x,
                                    const bf16 *restrict w,
                                    const float *restrict scales, int batch,
                                    int c, int n) {
  constexpr int vector_size = 16;
  const bf16 *w_base = w + c * n;
  for (int t = 0; t < batch; t++) {
    const bf16 *it_x = x + t * n;
    const bf16 *it_w = w_base;
    bf16 *it_y = y + t * n;
    const float s = scales[t];
    for (int i = 0; i < n / vector_size; i++) {
      aie::vector<bf16, vector_size> x_vec = aie::load_v<vector_size>(it_x);
      aie::vector<bf16, vector_size> w_vec = aie::load_v<vector_size>(it_w);
      aie::vector<float, vector_size> wx_vec = aie::mul(x_vec, w_vec);
      aie::vector<bf16, vector_size> o_vec = aie::mul(wx_vec, s);
      aie::store_v(it_y, o_vec);
      it_x += vector_size;
      it_w += vector_size;
      it_y += vector_size;
    }
  }
}

void rms_chunk_banded_aie(bf16 *restrict y, bf16 *restrict x, bf16 *restrict w,
                          float *restrict scales, int batch, int c, int n) {
  rms_chunk_banded(y, x, w, scales, batch, c, n);
}

#ifdef RMS_ROW_OUT
/// ONE WHOLE ROW of the normalized batch: y[i] = x[t*row_stride + i] * w[i] *
/// scales[t], for i in [0, n).
///
/// This is rms_chunk with the ROW INDEX and the WEIGHT OFFSET separated, which
/// is the only reason it cannot be rms_chunk itself: rms_chunk derives both
/// from the same `c` (`it_x = x + t*MODEL_DIM + c*n` and `w_base = w + c*n`),
/// so selecting row t by passing c=t would offset the norm weight by t*n too
/// and silently normalize against the wrong weights.
///
/// WHY IT EXISTS. The batched path stages a CROSS-ROW CHUNK ([batch][XCHUNK]),
/// so consecutive sends need different chunks and the buffer is refilled --
/// i.e. recomputed -- once per output row-block. Staging ONE WHOLE ROW instead
/// is smaller (5 KB against 8 KB at batch 8) and computes each value exactly
/// once, which is what block 1 does; the cross-row interleave the projection
/// cores need then happens in the memtile's READ descriptor rather than here.
/// See docs/BZeroPlan.md.
///
/// `row_stride` is x's true per-token pitch, matching
/// rms_scale_row_partial_aie's parameter of the same name: MODEL_DIM while x is
/// the resident [batch][MODEL_DIM] residual, or n for a phase whose K differs
/// (the down projection's GLU_OUT).
///
/// Behind RMS_ROW_OUT, like proj_qmm.cc's batched entry points and for the same
/// reason: merely adding a function can move the shipping kernels' codegen, and
/// check_kernels_inert.py holds them byte-identical.
void rms_row_aie(bf16 *restrict y, bf16 *restrict x, bf16 *restrict w,
                 float *restrict scales, int t, int row_stride, int n) {
  constexpr int vector_size = 16;
  const bf16 *it_x = x + t * row_stride;
  const bf16 *it_w = w;
  bf16 *it_y = y;
  const float s = scales[t];
  AIE_LOOP_RANGE(8)
  AIE_LOOP_UNROLL(8)
  for (int i = 0; i < n / vector_size; i++) {
    aie::vector<bf16, vector_size> x_vec = aie::load_v<vector_size>(it_x);
    aie::vector<bf16, vector_size> w_vec = aie::load_v<vector_size>(it_w);
    aie::vector<float, vector_size> wx_vec = aie::mul(x_vec, w_vec);
    aie::vector<bf16, vector_size> o_vec = aie::mul(wx_vec, s);
    aie::store_v(it_y, o_vec);
    it_x += vector_size;
    it_w += vector_size;
    it_y += vector_size;
  }
}
#endif

/// residual_acc_row_aie, but `acc` is the fresh [batch][n] band fetch being
/// accumulated into (row_stride=n, offset 0 -- the caller's fetch already
/// selected the right band, so there is no separate `off` term). `x` (the
/// projection round data) is unchanged, already n-strided.
void residual_acc_row_banded_aie(bf16 *restrict acc, bf16 *restrict x, int t,
                                 int n) {
  constexpr int vector_size = 16;
  bf16 *it_a = acc + t * n;
  bf16 *it_x = x + t * n;
  for (int i = 0; i < n / vector_size; i++) {
    aie::vector<bf16, vector_size> a_vec = aie::load_v<vector_size>(it_a);
    aie::vector<bf16, vector_size> x_vec = aie::load_v<vector_size>(it_x);
    aie::store_v(it_a, aie::add(a_vec, x_vec));
    it_a += vector_size;
    it_x += vector_size;
  }
}

/// residual_acc_row_banded_aie, OUT OF PLACE: out = acc + x.
///
/// WHY A SEPARATE DESTINATION AND NOT `acc += x`. The in-place form makes the
/// band buffer both the @rmsX get destination and the @layerOut put source,
/// and air-to-aie sizes a tile-DMA BD's lock acquire/release from
/// ceil(reads/writes) over the memcpy ops naming that buffer while the core
/// side is hardwired to 1 -- so six gets against two puts emits a layerOut BD
/// that acquires 3 against a core that releases 1, and the dispatch hangs
/// with nothing written. Writing the sum somewhere else leaves the fetch
/// buffer written-only and the drain buffer read-only, and both ratios are
/// then 1:1.
///
/// It also keeps the @rmsX BD ring uniform. With one task on the channel the
/// ring is count-free and lock-driven, so its slots rotate independently of
/// which core call site is asking; every slot targeting the SAME buffer is
/// what makes that safe. Two band buffers on the ring would deliver
/// round-robin into whichever one the slot names, not the one the core is
/// about to read.
///
/// `w` is the band width (always copied); `n <= w` is how much of it the
/// projection round actually contributes (DECODE_ACC_STOP passes 0 to freeze
/// the residual while keeping every transfer, so the tail must still be a
/// copy and not left undefined).
void residual_acc_row_banded_out_aie(bf16 *restrict out, bf16 *restrict acc,
                                     bf16 *restrict x, int t, int w, int n) {
  constexpr int vector_size = 16;
  bf16 *it_o = out + t * w;
  bf16 *it_a = acc + t * w;
  bf16 *it_x = x + t * w;
  for (int i = 0; i < n / vector_size; i++) {
    aie::vector<bf16, vector_size> a_vec = aie::load_v<vector_size>(it_a);
    aie::vector<bf16, vector_size> x_vec = aie::load_v<vector_size>(it_x);
    aie::store_v(it_o, aie::add(a_vec, x_vec));
    it_o += vector_size;
    it_a += vector_size;
    it_x += vector_size;
  }
  for (int i = n / vector_size; i < w / vector_size; i++) {
    aie::store_v(it_o, aie::load_v<vector_size>(it_a));
    it_o += vector_size;
    it_a += vector_size;
  }
}

/// Copy `n` elements out of a band fetch into the resident norm weight.
///
/// RMS_W_ON_X: at RMS_BAND_STREAM>=3 the norm weights arrive on @rmsX, in the
/// same band-shaped transfer and the same buffer as every other visit, because
/// the rms core has two S2MM ports and level 3 needs three unless the weights
/// stop owning one (see RMS_W_ON_X in fused_decode.py). A band fetch is
/// BATCH*STG_W >= MODEL_DIM elements wide, so the whole K-wide weight fits in
/// one; the launch side lands it at the front with a band-shaped descriptor and
/// this lifts it out before the next fetch overwrites the buffer.
///
/// Deliberately NOT rms_copy_aie: that one is fixed at MODEL_DIM over two
/// MODEL_DIM-typed buffers, and this reads a band-typed one. func.call needs an
/// exact type match, which is the same reason the other banded leaves are
/// separate symbols from their whole-row twins.
void band_to_weight_aie(bf16 *restrict w, bf16 *restrict band, int n) {
  constexpr int vector_size = 16;
  for (int i = 0; i < n / vector_size; i++)
    aie::store_v(w + i * vector_size,
                 aie::load_v<vector_size>(band + i * vector_size));
}

/// Copy one whole band, band-typed on both sides.
///
/// Exists to RELEASE A CHANNEL BUFFER EARLY, not to move data anywhere useful.
/// The residual accumulate consumes an @outY round and an @rmsX band together,
/// so AIR keeps the @outY landing buffer acquired across the band's arrival --
/// and the band cannot arrive while the round after it is queued behind it (see
/// _RMS_STG_COPY in fused_decode.py). Copying the round out gives the get a
/// reader of its own, which is what moves the release to just after it.
///
/// Same reason band_to_weight_aie is not rms_copy_aie: func.call needs an exact
/// memref type match, and this one is band-typed on BOTH sides.
void band_copy_aie(bf16 *restrict dst, bf16 *restrict src, int n) {
  constexpr int vector_size = 16;
  for (int i = 0; i < n / vector_size; i++)
    aie::store_v(dst + i * vector_size,
                 aie::load_v<vector_size>(src + i * vector_size));
}

// MLIR-managed-lock decode (q4nx_decode_repro): pure-compute leaves that let
// the rms core's lock/control-flow live in the aie.core body (explicit
// aie.use_lock) instead of inside rms_residual()'s in-kernel
// _lock_acquire/_lock_release.
//   rms_copy_aie:         dst = src                  (step 1: x_buf = x0)
//   residual_add_aie_hdr: pkt_id@y+14; y+16 = x_buf + x ; x_buf = y+16  (step
//   3)
void rms_copy_aie(bf16 *restrict dst, bf16 *restrict src) {
  aie_round_nearest_even();
  // Vectorised, like band_copy_aie above. It was a scalar loop, and Peano does
  // not auto-vectorise (see RMS_CHUNK_PROBE=2 below), so it ran 16x slower than
  // the residual_add_aie it stands in for under DECODE_ACC_STOP -- a diagnostic
  // that costs MORE than the thing it removes reports the opposite of what it
  // is asked. Same trap as RMS_CHUNK_PROBE=1, found by census rather than by
  // being burned a fourth time.
  constexpr int vector_size = 16;
  static_assert(MODEL_DIM % vector_size == 0, "MODEL_DIM must be a whole "
                                              "number of bf16 vectors");
  for (int i = 0; i < MODEL_DIM / vector_size; i++)
    aie::store_v(dst + i * vector_size,
                 aie::load_v<vector_size>(src + i * vector_size));
}

void residual_add_aie_hdr(bf16 *restrict y, bf16 *restrict x_buf,
                          bf16 *restrict x, unsigned int pkt_id) {
  aie_round_nearest_even();
  *reinterpret_cast<unsigned int *>(y + 14) = pkt_id;
  residual_add(y + 16, x_buf, x);
}

void rms_residual(bf16 *restrict y, bf16 *restrict x_ping,
                  bf16 *restrict x_pong, bf16 *restrict w,
                  bf16 *restrict y_out_0, bf16 *restrict y_out_1,
                  bf16 *restrict x_buf, int *IS_ATTN) {
  aie_round_nearest_even();
  constexpr int w_prod_lock = 0;
  constexpr int w_cons_lock = 1;
  constexpr int y_prod_lock = 2;
  constexpr int y_cons_lock = 3;
  constexpr int x_prod_lock = 4;
  constexpr int x_cons_lock = 5;
  constexpr int rtp_available_lock = 6;
  constexpr int lm_head_out_prod_lock = 7;
  constexpr int lm_head_out_cons_lock = 8;
  uint32_t *pkt_id_ptr = reinterpret_cast<uint32_t *>(y + 14);
  bf16 *x;

  // .data (not .bss): Peano's loader does not zero .bss, so keep these
  // ping/pong selectors in a section that is initialized at load.
  __attribute__((section(".data"))) static bool is_x_ping = false;
  __attribute__((section(".data"))) static bool is_lm_head_ping = false;
  __attribute__((section(".data"))) static bool is_w_ping = false;

  _lock_acquire(rtp_available_lock);
  if (IS_ATTN[0] == 1) {
    // input later norm + residul
    _lock_acquire(x_cons_lock);
    _lock_acquire(w_cons_lock);
    *pkt_id_ptr = pkt_id_rms_to_proj;
    is_x_ping = !is_x_ping;
    x = is_x_ping ? x_ping : x_pong;
    memcpy(x_buf, x, MODEL_DIM * sizeof(bf16));
    rms_norm(y + 16, x_buf, w);
    _lock_release(x_prod_lock);
    _lock_release(w_prod_lock);
    _lock_release(y_cons_lock, QKV_REPEATS);

    _lock_acquire(x_cons_lock);

    _lock_acquire(y_prod_lock, QKV_REPEATS);
    *pkt_id_ptr = pkt_id_rms_to_proj;
    is_x_ping = !is_x_ping;
    x = is_x_ping ? x_ping : x_pong;
    residual_add(x_buf, x_buf, x);
    // pre_feed_forward
    _lock_acquire(w_cons_lock);
    rms_norm(y + 16, x_buf, w);
    _lock_release(w_prod_lock);

    _lock_release(x_prod_lock);
    _lock_release(y_cons_lock, UP_GATE_REPEATS);

    _lock_acquire(x_cons_lock);
    _lock_acquire(y_prod_lock, UP_GATE_REPEATS);
    *pkt_id_ptr = pkt_id_rms_to_it;
    is_x_ping = !is_x_ping;
    x = is_x_ping ? x_ping : x_pong;
    residual_add(y + 16, x_buf, x);
    _lock_release(x_prod_lock);
    _lock_release(y_cons_lock, 1);
    _lock_acquire(y_prod_lock, 1);
  } else {
    _lock_acquire(x_cons_lock);
    _lock_acquire(w_cons_lock);
    *pkt_id_ptr = pkt_id_rms_to_proj;
    is_x_ping = !is_x_ping;
    x = is_x_ping ? x_ping : x_pong;
    rms_norm(y + 16, x, w);
    _lock_release(x_prod_lock);
    _lock_release(w_prod_lock);

    constexpr int vocab_size_per_round = VOCAB_SIZE_PADDED / MODEL_DIM;
    constexpr int y_repeats_per_round = MODEL_DIM / M_PER_ROUND;
    for (int i = 0; i < vocab_size_per_round; i++) {
      _lock_release(y_cons_lock, y_repeats_per_round);
      _lock_acquire(lm_head_out_prod_lock, 1);
      _lock_acquire(x_cons_lock);

      is_lm_head_ping = !is_lm_head_ping;
      bf16 *lm_head_dst = is_lm_head_ping ? y_out_0 : y_out_1;
      is_x_ping = !is_x_ping;
      x = is_x_ping ? x_ping : x_pong;
      memcpy(lm_head_dst, x, MODEL_DIM * sizeof(bf16));

      _lock_release(x_prod_lock);
      _lock_release(lm_head_out_cons_lock, 1);
      _lock_acquire(y_prod_lock, y_repeats_per_round);
    }
  }
}
}
