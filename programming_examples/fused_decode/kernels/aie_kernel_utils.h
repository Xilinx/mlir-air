/*
    Copyright (C) 2014 - 2022 Xilinx, Inc. All rights reserved.
    Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.
    SPDX-License-Identifier: MIT
*/

#ifndef _AIE_KERNEL_UTILS_
#define _AIE_KERNEL_UTILS_
#include "typedef.hpp"
#if defined(__chess__)
#define AIE_LOOP_UNROLL(x) [[chess::unroll_loop(x)]]
#define AIE_LOOP_UNROLL_FULL [[chess::unroll_loop()]]
#define AIE_LOOP_NO_UNROLL [[chess::no_unroll]]
#define AIE_LOOP_MIN_ITERATION_COUNT(x) [[chess::min_loop_count(x)]]
#define AIE_LOOP_MAX_ITERATION_COUNT(x) [[chess::max_loop_count(x)]]
#define AIE_LOOP_RANGE(a, ...)                                                 \
  [[chess::min_loop_count(a)]] __VA_OPT__(                                     \
      [[chess::max_loop_count(__VA_ARGS__)]])
#define AIE_PREPARE_FOR_PIPELINING [[chess::prepare_for_pipelining]]
#define AIE_NO_PREPARE_FOR_PIPELINING [[chess::no_prepare_for_pipelining]]
#define AIE_MODULO_SCHEDULING_BUDGET_RATIO(x)                                  \
  [[chess::modulo_scheduling_budget_ratio(x)]]
#define AIE_KEEP_SW_LOOP [[chess::keep_sw_loop]]
#define AIE_PEEL_PIPELINED_LOOP(x) [[chess::peel_pipelined_loop(x)]]
#define AIE_KEEP_FREE_FOR_PIPELINING(x) [[chess::keep_free_for_pipelining(x)]]
#define AIE_ALLOCATE(x) [[chess::allocate(x)]]
#define AIE_NO_HW_LOOP [[chess::no_hw_loop]]
#define AIE_TRY_INITIATION_INTERVAL(x)
#define AIE_PREPARE_FOR_POSTPIPELINING
#define AIE_LOOP_FLATTEN chess_flatten_loop

#elif defined(__AIECC__)
#ifndef __STRINGIFY
#define __STRINGIFY(a) #a
#endif
#define AIE_LOOP_UNROLL(x) _Pragma(__STRINGIFY(clang loop unroll_count(x)))
#define AIE_LOOP_UNROLL_FULL _Pragma("clang loop unroll(full)")
#define AIE_LOOP_NO_UNROLL _Pragma("clang loop unroll(disable)")
#define AIE_LOOP_MIN_ITERATION_COUNT(x)                                        \
  _Pragma(__STRINGIFY(clang loop min_iteration_count(x)))
#define AIE_LOOP_MAX_ITERATION_COUNT(x)                                        \
  _Pragma(__STRINGIFY(clang loop max_iteration_count(x)))
#define AIE_LOOP_RANGE(a, ...)                                                 \
  AIE_LOOP_MIN_ITERATION_COUNT(a)                                              \
  __VA_OPT__(AIE_LOOP_MAX_ITERATION_COUNT(__VA_ARGS__))
#define AIE_PREPARE_FOR_PIPELINING
#define AIE_NO_PREPARE_FOR_PIPELINING
#define AIE_MODULO_SCHEDULING_BUDGET_RATIO(x)
#define AIE_KEEP_SW_LOOP
#define AIE_PEEL_PIPELINED_LOOP(x)
#define AIE_KEEP_FREE_FOR_PIPELINING(x)
#define AIE_ALLOCATE(x)
#define AIE_NO_HW_LOOP
#define AIE_TRY_INITIATION_INTERVAL(x)                                         \
  _Pragma(__STRINGIFY(clang loop pipeline_initiation_interval(x)))
#define AIE_PREPARE_FOR_POSTPIPELINING _Pragma("clang loop pipeline(disable)")
#define AIE_LOOP_FLATTEN

#else
#define AIE_LOOP_UNROLL(x)
#define AIE_LOOP_UNROLL_FULL
#define AIE_LOOP_NO_UNROLL
#define AIE_LOOP_MIN_ITERATION_COUNT(x)
#define AIE_LOOP_MAX_ITERATION_COUNT(x)
#define AIE_LOOP_RANGE(a, ...)
#define AIE_PREPARE_FOR_PIPELINING
#define AIE_NO_PREPARE_FOR_PIPELINING
#define AIE_MODULO_SCHEDULING_BUDGET_RATIO(x)
#define AIE_KEEP_SW_LOOP
#define AIE_PEEL_PIPELINED_LOOP(x)
#define AIE_KEEP_FREE_FOR_PIPELINING(x)
#define AIE_ALLOCATE(x)
#define AIE_NO_HW_LOOP
#define AIE_TRY_INITIATION_INTERVAL(x)
#define AIE_PREPARE_FOR_POSTPIPELINING
#define AIE_LOOP_FLATTEN
#endif

// Loop-counter type for loops whose body ends in a per-iteration lock release.
// Peano/llvm-aie (non-chess) sinks the final lock release past the loop
// back-edge, so it runs once on loop exit instead of every iteration and
// deadlocks any multi-iteration lock loop. A volatile counter makes the trip
// count opaque and keeps the release inside the loop. chess schedules
// correctly.
#if defined(__chess__)
#define AIE_LOCK_LOOP_CTR int
#else
#define AIE_LOCK_LOOP_CTR volatile int
#endif

// Some lock utilities
/// \brief Acquire a lock with given lock ID on this CT and number of locks
/// \param lock_id The lock ID to acquire
/// \param num_locks The number of locks to acquire
inline void _lock_acquire(int lock_id, int num_locks = 1) {
  acquire_greater_equal(lock_id + 48, num_locks);
}

/// \brief Release a lock with given lock ID on this CT and number of locks
/// \param lock_id The lock ID to release
/// \param num_locks The number of locks to release
inline void _lock_release(int lock_id, int num_locks = 1) {
  release(lock_id + 48, num_locks);
}

/// \brief Acquire a lock with given lock ID on left CT and number of locks
/// \param lock_id The lock ID to acquire
/// \param num_locks The number of locks to acquire
inline void _left_lock_acquire(int lock_id, int num_locks = 1) {
  acquire_greater_equal(lock_id + 16, num_locks);
}

/// \brief Release a lock with given lock ID on left CT and number of locks
/// \param lock_id The lock ID to release
/// \param num_locks The number of locks to release
inline void _left_lock_release(int lock_id, int num_locks = 1) {
  release(lock_id + 16, num_locks);
}

/// \brief Acquire a lock with given lock ID on up CT and number of locks
/// \param lock_id The lock ID to acquire
/// \param num_locks The number of locks to acquire
inline void _up_lock_acquire(int lock_id, int num_locks = 1) {
  acquire_greater_equal(lock_id + 32, num_locks);
}

/// \brief Release a lock with given lock ID on up CT and number of locks
/// \param lock_id The lock ID to release
/// \param num_locks The number of locks to release
inline void _up_lock_release(int lock_id, int num_locks = 1) {
  release(lock_id + 32, num_locks);
}

/// \brief Acquire a lock with given lock ID on down CT and number of locks
/// \param lock_id The lock ID to acquire
/// \param num_locks The number of locks to acquire
inline void _down_lock_acquire(int lock_id, int num_locks = 1) {
  acquire_greater_equal(lock_id, num_locks);
}

/// \brief Release a lock with given lock ID on down CT and number of locks
/// \param lock_id The lock ID to release
/// \param num_locks The number of locks to release
inline void _down_lock_release(int lock_id, int num_locks = 1) {
  release(lock_id, num_locks);
}


// ATTN_BENCH_UNROLL: straight-line attention's contraction loops so that STATIC
// bundle count equals DYNAMIC cycle count, for bench_attn.py only. A real build
// leaves them rolled.
//
// Without it the static count is one LOOP BODY, not one call, and is therefore
// identical for DH=64 and DH=128 -- the trip count lives in a register, so both
// builds emit the same code. Same trap Q4K_MM_FULL_UNROLL exists to avoid in
// q4k_mm.h; attention walks into it harder, because colQ = DH/8 is 8 or 16
// rather than 1. Lives here rather than in attn_qk.cc because attn_kv.cc is a
// separate translation unit and needs the same switch.
#ifdef ATTN_BENCH_UNROLL
#define ATTN_Q_LOOP AIE_LOOP_UNROLL_FULL
#else
#define ATTN_Q_LOOP
#endif

// ---------------------------------------------------------------------------
// Bench-only DECOMPOSITION knobs, for bench_attn_batch.py. Every one of these
// produces a NUMERICALLY WRONG kernel; none is defined by any build that runs.
// They exist so the batching argument is made against MEASURED shares instead
// of guessed ones -- the question "what fraction of attention could a batch
// amortize?" is a question about which pieces are per-token and which are per
// KV block, and the only way to price a piece is to remove it and re-count.
//
//   ATTN_BENCH_NO_TRANSPOSE   drop the aie::transpose on each K tile. Per KV
//                             block, so a batch could hoist it.
//   ATTN_BENCH_NO_UPDATE      drop the online-softmax update(). Per (token,
//                             head), so a batch can NEVER hoist it -- this is
//                             the floor the lever cannot get under.
//   ATTN_BENCH_NO_CORRECT     drop attn_fv's separate rescale pass over y.
//                             Per token; prices the y accumulator traffic.
//
// Removing work can only make the compiler's job easier, so each delta is an
// UPPER bound on what removing that piece saves. That is the direction that
// keeps the conclusion honest: an upper bound on the saving is an upper bound
// on what batching can win.
//   ATTN_BENCH_NO_KLOAD       serve every K tile from one hoisted load. Also
//                             per KV block, so also hoistable by a batch.
//
#ifdef ATTN_BENCH_NO_TRANSPOSE
#define ATTN_TRANSPOSE(v) (v)
#else
#define ATTN_TRANSPOSE(v) aie::transpose((v), 8, 8)
#endif

#ifdef ATTN_BENCH_NO_KLOAD
#define ATTN_KLOAD(p) K_hoisted
#else
#define ATTN_KLOAD(p) aie::load_v<MMUL::size_B>(p)
#endif

#ifdef ATTN_BENCH_NO_UPDATE
// Identity: keeps `out` live so the mmul and its epilogue are not dead-coded,
// while deleting the exp/max/rescale. m and c go unwritten, hence wrong.
#define ATTN_UPDATE(m, c, out, mask, is_first) (out)
#else
#define ATTN_UPDATE(m, c, out, mask, is_first)                                 \
  update((m), (c), (out), (mask), (is_first))
#endif

template <typename T, int M>
void zero_vectorized(T *__restrict c) {
  constexpr int r = 256 / (sizeof(T) * 8); // one 256 bit store unit
  static_assert((M) % r == 0);
  const aie::vector<T, r> zeros = aie::zeros<T, r>();
  const T *__restrict c_end = c + M;
  for (; c < c_end; c += r) {
    aie::store_v(c, zeros);
  }
}

template <int m>
void copy_float_to_bf16(bfloat16 *y, const float *y_acc) {
  static_assert(m % 16 == 0);
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_UNROLL_FULL
  for (int i = 0; i < m / 16; i++) {
    aie::vector<float, 16> y_vec = aie::load_v<16>(y_acc + i * 16);
    aie::accum<accfloat, 16> acc;
    acc.from_vector(y_vec);
    aie::store_v(y + i * 16, acc.template to_vector<bfloat16>());
  }
}

template <int m>
void copy_bf16_to_float(float *y, const bfloat16 *y_bf16) {
  static_assert(m % 16 == 0);
  AIE_PREPARE_FOR_PIPELINING
  for (int i = 0; i < m / 16; i++) {
    aie::vector<bfloat16, 16> y_vec = aie::load_v<16>(y_bf16 + i * 16);
    aie::accum<accfloat, 16> acc;
    acc.from_vector(y_vec);
    aie::store_v(y + i * 16, acc.template to_vector<float>());
  }
}

template <int m>
void copy_bf16_to_bf16(bfloat16 *y, const bfloat16 *y_bf16) {
  static_assert(m % 16 == 0);
  AIE_PREPARE_FOR_PIPELINING
  for (int i = 0; i < m / 16; i++) {
    aie::vector<bfloat16, 16> y_vec = aie::load_v<16>(y_bf16 + i * 16);
    aie::store_v(y + i * 16, y_vec);
  }
}

// Round-to-nearest-even for every accumulator narrowing in this core.
//
// An AIE core comes up with its rounding mode set to FLOOR (toward -inf), and
// the mode is a core register that nothing else here writes -- so a kernel that
// does not set it narrows accum->bf16 by flooring, on EVERY conversion. That is
// not a wash: it is a one-sided bias, so it does not cancel across a reduction
// the way round-to-nearest error does, and it compounds down a residual stream.
//
// Measured on LFM2-1.2B decode (16 layers, NPU2): flooring reproduced the
// device BIT-EXACTLY while round-to-nearest was 0.0152 away, and the bias grew
// from 0.015 relative error on layer 0's ShortConv state to 0.264 by layer 15,
// dragging the whole-model logit cosine to 0.876 and flipping top-1 tokens.
// Setting the mode once per kernel entry costs one register write.
//
// This is the same call the conv2d/matmul kernels in aie_kernels and
// programming_examples already make; the decode kernels were the ones missing
// it. Call it FIRST in every extern "C" entry point -- the register is per
// core, but a core runs whichever kernels its herd assigns, so relying on some
// other entry point having set it is how one path silently keeps flooring.
//
// Taken from origin/main 980a8acc (#1929); this branch carries only that
// commit's rounding half, not its opt-in finer SiLU table.
static inline void aie_round_nearest_even() {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
}

#endif
