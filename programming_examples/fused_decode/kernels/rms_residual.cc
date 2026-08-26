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

extern "C" {

// AIR-friendly pure-compute entry points (no in-kernel locks; AIR owns sync),
// mirroring proj_qmm.cc's split of the reference proj_main. Used by the AIR
// decode vehicle (proj_qmm_demux_fb.py) for the post-attention residual +
// RMSNorm stage.
//   rms_norm_aie:     y = rmsnorm(x) * w           (y,x,w each MODEL_DIM bf16)
//   residual_add_aie: y = x_buf + x ; x_buf = y    (MODEL_DIM bf16)
void rms_norm_aie(bf16 *restrict y, bf16 *restrict x, bf16 *restrict w,
                  int _arm) {
  (void)_arm; // per-token RTP arm-gate operand (kept alive so AIR emits the arm
              // lock)
  rms_norm(y, x, w);
}

// 2K-weight variants: w holds TWO MODEL_DIM norm weights back-to-back; _lo uses
// the first, _hi the second (w + MODEL_DIM). Lets one packet channel carry two
// norms so the Gemma 4-norm rms tile keeps <=4 packet ids per S2MM port.
void rms_norm_lo_aie(bf16 *restrict y, bf16 *restrict x, bf16 *restrict w,
                     int _arm) {
  (void)_arm;
  rms_norm(y, x, w);
}
void rms_norm_hi_aie(bf16 *restrict y, bf16 *restrict x, bf16 *restrict w,
                     int _arm) {
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
  *reinterpret_cast<unsigned int *>(y + 14) = pkt_id;
  rms_norm(y + 16, x, w);
}

void residual_add_aie(bf16 *restrict y, bf16 *restrict x_buf,
                      bf16 *restrict x) {
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
#ifdef RMS_CHUNK_PROBE
  // Diagnostic builds only: make row t of the X feed the CONSTANT (t+1)/8, so
  // every projection output row comes out proportional to t+1. Reading the KV
  // cache then says what row 7 of the mmul's A operand actually was -- 8x row 0
  // if the feed is right, 7x if it got the previous token's, 0 if it got
  // nothing. The flush-side labels (PROJ_FLUSH_PROBE=3) proved the descriptor
  // chain BELOW the accumulator; this is the hop above it.
  for (int t = 0; t < batch; t++)
    for (int i = 0; i < n; i++)
      y[t * n + i] = (bf16)((float)(t + 1) * 0.125f);
  (void)x;
  (void)w;
  (void)scales;
  (void)c;
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

// MLIR-managed-lock decode (q4nx_decode_repro): pure-compute leaves that let
// the rms core's lock/control-flow live in the aie.core body (explicit
// aie.use_lock) instead of inside rms_residual()'s in-kernel
// _lock_acquire/_lock_release.
//   rms_copy_aie:         dst = src                  (step 1: x_buf = x0)
//   residual_add_aie_hdr: pkt_id@y+14; y+16 = x_buf + x ; x_buf = y+16  (step
//   3)
void rms_copy_aie(bf16 *restrict dst, bf16 *restrict src) {
  for (int i = 0; i < MODEL_DIM; i++)
    dst[i] = src[i];
}

void residual_add_aie_hdr(bf16 *restrict y, bf16 *restrict x_buf,
                          bf16 *restrict x, unsigned int pkt_id) {
  *reinterpret_cast<unsigned int *>(y + 14) = pkt_id;
  residual_add(y + 16, x_buf, x);
}

void rms_residual(bf16 *restrict y, bf16 *restrict x_ping,
                  bf16 *restrict x_pong, bf16 *restrict w,
                  bf16 *restrict y_out_0, bf16 *restrict y_out_1,
                  bf16 *restrict x_buf, int *IS_ATTN) {
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
