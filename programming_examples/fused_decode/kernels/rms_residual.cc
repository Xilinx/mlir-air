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
void rms_norm(bf16 *restrict y, const bf16 *restrict x,
              const bf16 *restrict w) {
  bf16 *it_y = const_cast<bf16 *>(y);
  bf16 *it_x = const_cast<bf16 *>(x);
  bf16 *it_w = const_cast<bf16 *>(w);
  // Added to the MEAN square, so it is comparable to mean(x^2) -- which for
  // most models is O(1) at every layer and makes the exact value irrelevant.
  // It is NOT irrelevant for a model whose embeddings are small: LFM2's are
  // ~0.6 in L2 over 2048 channels, so mean(x^2) ~ 1.8e-4 and layer 0's
  // normalizer moves by several percent between 1e-6 and 1e-5. Models that
  // care set RMS_NORM_EPS in their header; the default preserves the value
  // every previously-shipped model was validated against.
  const float epsilon = RMS_NORM_EPS;

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

  it_x = const_cast<bf16 *>(x);

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

// MLIR-managed-lock decode (q4nx_decode_repro): pure-compute leaves that let
// the rms core's lock/control-flow live in the aie.core body (explicit
// aie.use_lock) instead of inside rms_residual()'s in-kernel
// _lock_acquire/_lock_release.
//   rms_copy_aie:         dst = src                  (step 1: x_buf = x0)
//   residual_add_aie_hdr: pkt_id@y+14; y+16 = x_buf + x ; x_buf = y+16  (step
//   3)
void rms_copy_aie(bf16 *restrict dst, bf16 *restrict src) {
  aie_round_nearest_even();
  for (int i = 0; i < MODEL_DIM; i++)
    dst[i] = src[i];
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
