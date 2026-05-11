//===- attn_decode_npu2.cc --------------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
// AIE2P kernels for the LLaMA-3.2-1B decode flash-attention design under
// programming_examples/llama2_mha/ (entry point attn_decode_npu2.py).
//
// Functions: simple_rms_bf16, xrms_demux_bf16, vecmat_bf16_bf16,
// linalg_fill_bf16, freq_pos_bf16_32_16, sinf_bf16_32_16, cosf_bf16_32_16,
// shuffle_apply_rope_bf16_64, attn_1_group, softmax_group, attn_2_group,
// fill_neg99_bf16, fill_zero_bf16, vector_copy_n_bf16.
//
// LLaMA GQA group_size=4: 4 Q heads share 1 K and 1 V. K/V cache rows
// are read ONCE per history position and reused across the 4 Q heads
// via attn_1_group / attn_2_group / softmax_group.
//
// AIE2P-portability notes (vs the NPU1 mha.cc reference):
// - sinf/cosf: rope_sincos-style polynomial with native n=16 vectors
//   and 32-elem padded internal buffers (24-elem outer interface)
// - shuffle_apply_rope: aie::filter_even/odd + interleave_zip (no
//   shuffle_T16_* constants on AIE2P)
// - softmax: aie::exp2-based (AIE2P has no LUT-based exp lowering)
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <type_traits>

// AIE2P native vector width for accfloat ops.
static constexpr unsigned VEC = 16;

//------------------------------------------------------------------------------
// sinf_cosf_poly_bf16: ported from rope_sincos/rope.cc. Native n=16 vectors
// for AIE2P; processes N elements via N/n iterations.
//------------------------------------------------------------------------------
// Inputs are f32 (not bf16) to preserve sin/cos input precision at large
// |ux|. Truncating ux to bf16 here would shift the input by up to 0.5
// radians at |ux| ~ 600, putting sin/cos onto a totally different point
// of their oscillation and producing 0.3+ absolute error per element.
template <unsigned N, unsigned n, bool isSine>
static void sinf_cosf_poly_bf16(const float *__restrict inputs,
                                bfloat16 *__restrict outputs) {
  constexpr float sin_poly_factors[4] = {
      -0x1.5555555555555p-3, 0x1.1111111110bb3p-7, -0x1.a01a019e83e5cp-13,
      0x1.71de3796cde01p-19};
  constexpr float cos_poly_factors[4] = {
      0x1.5555555555555p-5, -0x1.6c16c16c16967p-10, 0x1.A01A019F4EC91p-16,
      -0x1.27E4FA17F667Bp-22};
  const float twobypival = 0x1.45f306dc9c883p-1;
  const float piby2_1val = 0x1.921fb54400000p0;
  for (unsigned i = 0; i < N; i += n) {
    aie::accum<accfloat, n> one_acc;
    aie::accum<accfloat, n> poly_acs, acs0, acs2, out_acs;
    aie::accum<accfloat, n> poly_acc, acc0, acc2, out_acc, output_acc;
    aie::vector<bfloat16, n> x, r, r2, out_abs;
    aie::vector<bfloat16, n> s0, s1, s2, s3, c0, c1, c2, c3;
    aie::vector<bfloat16, n> negoneby2, one, negone;
    s0 = aie::broadcast<bfloat16, n>(sin_poly_factors[0]);
    s1 = aie::broadcast<bfloat16, n>(sin_poly_factors[1]);
    s2 = aie::broadcast<bfloat16, n>(sin_poly_factors[2]);
    s3 = aie::broadcast<bfloat16, n>(sin_poly_factors[3]);
    c0 = aie::broadcast<bfloat16, n>(cos_poly_factors[0]);
    c1 = aie::broadcast<bfloat16, n>(cos_poly_factors[1]);
    c2 = aie::broadcast<bfloat16, n>(cos_poly_factors[2]);
    c3 = aie::broadcast<bfloat16, n>(cos_poly_factors[3]);
    negoneby2 = aie::broadcast<bfloat16, n>(-0.5);
    one = aie::broadcast<bfloat16, n>(1.0);
    negone = aie::broadcast<bfloat16, n>(-1.0);
    acs0.from_vector(s0);
    acs2.from_vector(s2);
    acc0.from_vector(c0);
    acc2.from_vector(c2);
    one_acc.from_vector(one);

#define POLY_EVAL_4_VECTOR(r, s0, s1, s2, s3, c0, c1, c2, c3, outs, outc, r2)  \
  {                                                                            \
    aie::accum<accfloat, n> t1s, t2s, t1c, t2c, r2_acc;                        \
    t1s = mac(s0, s1, r);                                                      \
    t2s = mac(s2, s3, r);                                                      \
    t1c = mac(c0, c1, r);                                                      \
    t2c = mac(c2, c3, r);                                                      \
    r2_acc = mul(r, r);                                                        \
    r2 = r2_acc.template to_vector<bfloat16>();                                \
    outs = mac(t1s, r2, t2s.template to_vector<bfloat16>());                   \
    outc = mac(t1c, r2, t2c.template to_vector<bfloat16>());                   \
  }

    // Range reduction in scalar f32, per-element. Storing ux as bf16
    // anywhere in this pipeline would shift large |ux| onto a different
    // point of the sin/cos oscillation (pos=1000, freq=0.6636 -> true
    // ux=663.6, bf16 ux=664; sin(664)-sin(663.6) ~ 0.3 abs error). And
    // bf16 of the int quadrant zj for zj>~256 rounds to even (e.g. 637
    // -> 640), pushing x = ux - zj*pi/2 far outside the polynomial range
    // [-pi/4, pi/4]. Doing the whole reduction in f32 fixes both.
    alignas(aie::vector_decl_align) bfloat16 x_buf[n];
    aie::mask<n> is_odd, is_neg;
    {
      uint32_t odd_bits = 0, neg_bits = 0;
      const float twobypif = (float)twobypival;
      const float piby2f = (float)piby2_1val;
      for (unsigned j = 0; j < n; j++) {
        float ux_j = inputs[i + j];
        float zp_f32 = ux_j * twobypif + 0.5f;
        int zj = (int)zp_f32;
        x_buf[j] = (bfloat16)(ux_j - (float)zj * piby2f);
        if (isSine) {
          if (zj & 1)
            odd_bits |= (1u << j);
          if (zj & 2)
            neg_bits |= (1u << j);
        } else {
          if (!(zj & 1))
            odd_bits |= (1u << j);
          if (((zj >> 1) & 1) ^ (zj & 1))
            neg_bits |= (1u << j);
        }
      }
      is_odd = aie::mask<n>::from_uint32(odd_bits);
      is_neg = aie::mask<n>::from_uint32(neg_bits);
    }
    x = aie::load_v<n>(x_buf);
    r = aie::mul(x, x).template to_vector<bfloat16>();

    aie::vector<bfloat16, n> t_vec =
        mac(one_acc, negoneby2, r).template to_vector<bfloat16>();

    POLY_EVAL_4_VECTOR(r, acs0, s1, acs2, s3, acc0, c1, acc2, c3, poly_acs,
                       poly_acc, r2);

    aie::accum<accfloat, n> x_acc_fresh;
    x_acc_fresh.from_vector(x);
    aie::vector<bfloat16, n> x3 = aie::mul(r, x).template to_vector<bfloat16>();
    out_acs = mac(x_acc_fresh, x3, poly_acs.template to_vector<bfloat16>());
    aie::accum<accfloat, n> t_acc_fresh;
    t_acc_fresh.from_vector(t_vec);
    out_acc = mac(t_acc_fresh, r2, poly_acc.template to_vector<bfloat16>());

    out_abs = select(out_acs.template to_vector<bfloat16>(),
                     out_acc.template to_vector<bfloat16>(), is_odd);
    output_acc = aie::mul(negone, out_abs);
    bfloat16 *__restrict pOut = outputs + i;
    aie::store_v(
        pOut,
        select(out_abs, output_acc.template to_vector<bfloat16>(), is_neg));
#undef POLY_EVAL_4_VECTOR
  }
}

//------------------------------------------------------------------------------
// freq_pos: out[i] = pos / rope_base^(2i/dk). dk=64 => 32 entries (no padding;
// fits exactly one native n=16 vector pair). rope_base = 500000 matches the
// HuggingFace LLaMA-3 / LLaMA-3.2 rope_theta config.
//------------------------------------------------------------------------------
// Freq table kept in f32 (doubles size 64B -> 128B, negligible). Storing
// as bf16 + multiplying by bf16 pos rounds the product (e.g., pos=1000,
// freq[1]=0.663601 -> bf16 ux = 664.0 vs true 663.6). At |ux| > ~50 the
// 0.4 input shift puts sin/cos onto a different point of their oscillation,
// producing ~0.3 absolute error per element. Computing pos*freq in f32
// (and truncating to bf16 only when handing off to the polynomial) closes
// most of that gap.
alignas(aie::vector_decl_align) static const float freq_table_dk64[32] = {
    1.000000f, 0.663601f, 0.440367f, 0.292228f, 0.193923f, 0.128687f, 0.085397f,
    0.056670f, 0.037606f, 0.024955f, 0.016560f, 0.010990f, 0.007293f, 0.004839f,
    0.003211f, 0.002131f, 0.001414f, 0.000938f, 0.000623f, 0.000413f, 0.000274f,
    0.000182f, 0.000121f, 0.000080f, 0.000053f, 0.000035f, 0.000023f, 0.000016f,
    0.000010f, 0.000007f, 0.000005f, 0.000003f,
};

template <unsigned N, unsigned n>
static void freq_pos_bf16(int pos, const float *freq_table,
                          bfloat16 *__restrict outputs) {
  // Compute ux = pos * freq[i] in scalar f32, truncate to bf16 only for
  // storage (sin/cos polynomial input is bf16). The scalar f32 mul
  // preserves the exact pos and freq precision; bf16 truncation of the
  // product is fine at the sin/cos input scale (|ux| up to ~2000).
  const float pos_f32 = (float)pos;
  for (unsigned i = 0; i < N; ++i)
    outputs[i] = (bfloat16)(pos_f32 * freq_table[i]);
}

//------------------------------------------------------------------------------
// shuffle_apply_rope: HuggingFace LLaMA half-split RoPE. Pairs are
// (x[i], x[i + N/2]) for i in 0..N/2-1, not the original pairwise
// (x[2i], x[2i+1]). N=64 (head_dim) so half=32=2*VL with VL=16.
//------------------------------------------------------------------------------
template <unsigned N, unsigned n>
static void shuffle_apply_rope(int offset, const bfloat16 *__restrict fcr,
                               const bfloat16 *__restrict fci,
                               bfloat16 *__restrict outputs) {
  constexpr unsigned half = N / 2;
  static_assert(half % n == 0, "half must be a multiple of vector width n");
  bfloat16 *__restrict p = outputs + offset;
  for (unsigned i = 0; i < half; i += n) {
    aie::vector<bfloat16, n> x1 = aie::load_v<n>(p + i);
    aie::vector<bfloat16, n> x2 = aie::load_v<n>(p + half + i);
    aie::vector<bfloat16, n> c = aie::load_v<n>(fcr + i);
    aie::vector<bfloat16, n> s = aie::load_v<n>(fci + i);
    // out_first  = x1 * c - x2 * s
    // out_second = x1 * s + x2 * c
    aie::vector<bfloat16, n> out1 =
        msc(aie::mul(x1, c), x2, s).template to_vector<bfloat16>();
    aie::vector<bfloat16, n> out2 =
        mac(aie::mul(x1, s), x2, c).template to_vector<bfloat16>();
    aie::store_v(p + i, out1);
    aie::store_v(p + half + i, out2);
  }
}

// Pad head_size to multiple of 2*n, run vectorized, copy back.
// Used so head_size=48 works with native n=16 (need 64-elem work buffer).
template <unsigned head_size>
static void shuffle_apply_rope_padded(int offset,
                                      const bfloat16 *__restrict fcr,
                                      const bfloat16 *__restrict fci,
                                      bfloat16 *__restrict outputs) {
  static_assert(head_size % 16 == 0);
  constexpr unsigned n = VEC;
  constexpr unsigned two_n = n * 2;
  constexpr unsigned padded = ((head_size + two_n - 1) / two_n) * two_n;
  alignas(aie::vector_decl_align) bfloat16 work[padded];
  for (unsigned j = 0; j < head_size; j += 16)
    aie::store_v(work + j, aie::load_v<16>(outputs + j + offset));
  for (unsigned j = head_size; j < padded; j += 16)
    aie::store_v(work + j, aie::zeros<bfloat16, 16>());
  shuffle_apply_rope<padded, n>(0, fcr, fci, work);
  for (unsigned j = 0; j < head_size; j += 16)
    aie::store_v(outputs + j + offset, aie::load_v<16>(work + j));
}

//------------------------------------------------------------------------------
// softmax_group: replaced with inline-MLIR 3-pass softmax in
// attn_decode_npu2.py (uses linalg.fill, vector.transfer_*, math.exp,
// vector.reduction). The .cc impl is no longer needed.
//------------------------------------------------------------------------------
#ifndef SEQ_LEN
#define SEQ_LEN 128
#endif

static_assert(SEQ_LEN % 16 == 0, "SEQ_LEN must be multiple of 16");

//------------------------------------------------------------------------------
// simple_rms_bf16 helpers (templates, kept outside extern "C"). Body matches
// programming_examples/flash_attention/kernel_fusion_based/simple_rms.cc;
// inlined here so rms_herd can link from the same .o as the GQA herd.
//------------------------------------------------------------------------------
static inline float fast_rsqrt(float x) {
  union {
    float f;
    int32_t i;
  } u;
  u.f = x;
  u.i = 0x5f3759df - (u.i >> 1);
  float y = u.f;
  // Single Newton-Raphson iteration. RMSNorm tolerates ~1% rstd error
  // and the 2nd iter pulls in extra scalar __mulsf3/__divsf3 libcalls
  // that bloat the AIE2P core ELF beyond the 16 KB program memory limit
  // when rms shares a tile with vecmat + attn + softmax + RoPE.
  y = y * (1.5f - 0.5f * x * y * y);
  return y;
}

// Runtime-N rms (no templating). Per-build template instantiation for
// large N (e.g., 2048) explodes the AIE2P core's 16 KB program memory
// when fully unrolled. Runtime loops keep the function size constant
// across N at the cost of a few cycles per token.
static inline void rms_kernel_bf16(const bfloat16 *__restrict x,
                                   const bfloat16 *__restrict w,
                                   bfloat16 *__restrict y, int n_val,
                                   float eps) {
  constexpr int VL = 16;
  const int n_vecs = n_val / VL;

  aie::accum<accfloat, VL> ssq = aie::zeros<accfloat, VL>();
  for (int i = 0; i < n_vecs; i++) {
    aie::vector<bfloat16, VL> xv = aie::load_v<VL>(x + i * VL);
    ssq = aie::mac(ssq, xv, xv);
  }
  // Vector reduce avoids a scalar 16-iter sum loop (more libcall churn).
  // Multiply by precomputed 1/n instead of dividing — sidesteps __divsf3.
  aie::vector<float, VL> ssq_v = ssq.to_vector<float>();
  float total = aie::reduce_add(ssq_v);
  float inv_n = 1.0f / (float)n_val;
  float rstd = fast_rsqrt(total * inv_n + eps);

  aie::vector<bfloat16, VL> rstd_v =
      aie::broadcast<bfloat16, VL>((bfloat16)rstd);
  for (int i = 0; i < n_val; i += VL) {
    aie::vector<bfloat16, VL> xv = aie::load_v<VL>(x + i);
    aie::vector<bfloat16, VL> wv = aie::load_v<VL>(w + i);
    aie::accum<accfloat, VL> a1 = aie::mul(xv, rstd_v);
    aie::vector<bfloat16, VL> rx = a1.to_vector<bfloat16>();
    aie::accum<accfloat, VL> a2 = aie::mul(rx, wv);
    aie::store_v(y + i, a2.to_vector<bfloat16>());
  }
}

//------------------------------------------------------------------------------
// extern "C" entry points — same names/signatures as mha.cc so mha.py
// works unchanged when linked against mha_npu2.o.
//------------------------------------------------------------------------------

#ifndef DIM_N
#define DIM_N 64
#endif
#ifndef DIM_K
#define DIM_K 64
#endif
// TILE_K is the inner-k chunk size processed per vecmat call. The caller
// invokes vecmat_bf16_bf16 (DIM_K / TILE_K) times with x_offset stepping in
// TILE_K units, accumulating into c[offset..offset+DIM_N]. Defaults to DIM_K
// (single-chunk GEMV) for backward compatibility with small-K configs.
#ifndef TILE_K
#define TILE_K DIM_K
#endif
static_assert(DIM_K % TILE_K == 0, "DIM_K must be a multiple of TILE_K");
#ifndef HEAD_SIZE
#define HEAD_SIZE 64
#endif
#define HEAD_SIZE_BY_TWO (HEAD_SIZE / 2)

// DATA_MOVEMENT_ONLY (set via -DDATA_MOVEMENT_ONLY at compile time)
// short-circuits every extern-C kernel function to an immediate return.
// Locks still cycle through the DMA chain (the herd-body channel.get/put
// still acquires/releases), so per-token DMA timing is preserved while
// the actual compute drops to ~zero. Use this build to measure how much
// of the per-token latency is DMA / sync overhead vs core compute.
#ifdef DATA_MOVEMENT_ONLY
#define STUB_RET return
#else
#define STUB_RET ((void)0)
#endif

extern "C" {

// vecmat_bf16_bf16: replaced with inline-MLIR vecmat in attn_decode_npu2.py
// (handwritten broadcast-bf16+extf+f32-fma pattern, after matvec_cascade.py).
// Costs ~78us extra per token vs the chess-pipelined extern, accepted to
// remove the largest extern dependency.

// dk=64 => 32-elem buffers, native n=16 vectors. No padding needed.
void freq_pos_bf16_32_16(int pos, bfloat16 *outputs) {
  STUB_RET;
  // Backwards-compat wrapper: still emits bf16 (used by the .o-only path).
  for (unsigned i = 0; i < 32; ++i)
    outputs[i] = (bfloat16)((float)pos * freq_table_dk64[i]);
}

// f32-output variant. Used by attn_decode_npu2.py to keep ux precision
// going into sinf/cosf — bf16 truncation here would shift sin/cos input
// onto the wrong oscillation point at large pos. See sinf_cosf_poly_bf16.
void freq_pos_f32_32_16(int pos, float *outputs) {
  STUB_RET;
  const float pos_f32 = (float)pos;
  for (unsigned i = 0; i < 32; ++i)
    outputs[i] = pos_f32 * freq_table_dk64[i];
}

void sinf_bf16_32_16(const float *inputs, bfloat16 *outputs) {
  STUB_RET;
  sinf_cosf_poly_bf16<32, VEC, true>(inputs, outputs);
}

void cosf_bf16_32_16(const float *inputs, bfloat16 *outputs) {
  STUB_RET;
  sinf_cosf_poly_bf16<32, VEC, false>(inputs, outputs);
}

void shuffle_apply_rope_bf16_64(int offset, const bfloat16 *fcr,
                                const bfloat16 *fci, bfloat16 *outputs) {
  STUB_RET;
  // dk=64 = exactly 2 * native n=16. No padding wrapper needed.
  shuffle_apply_rope<HEAD_SIZE, VEC>(offset, fcr, fci, outputs);
}

// xrms_demux_bf16: split a [tile_k, tile_n] padded packet (laid out flat)
// into x_raw[k] and w_rms[k]. The padded packet contains [x_raw[0:k],
// w_rms[0:k], junk...] in row-major order. K elements each, vector-copy.
void xrms_demux_bf16(const bfloat16 *padded, bfloat16 *x_raw, bfloat16 *w_rms,
                     int32_t k) {
  STUB_RET;
  constexpr int VL = 16;
  for (int i = 0; i < k; i += VL)
    aie::store_v(x_raw + i, aie::load_v<VL>(padded + i));
  const bfloat16 *w_src = padded + k;
  for (int i = 0; i < k; i += VL)
    aie::store_v(w_rms + i, aie::load_v<VL>(w_src + i));
}

// Fused copy-and-scale for Q: pre-scales Q by 1/sqrt(DIM_N) once per token
// so the per-K-row attn_1_group skips the (bf16)(scalar * 0.125f) tail
// computation (saves 4 scalar muls + bf16 conversions per attn_1 call).
// 32-lane to match attn_1's vector width.
void copy_scale_q_bf16(const bfloat16 *src, bfloat16 *dst,
                       int32_t total_elems) {
  STUB_RET;
  aie::vector<bfloat16, 32> scale =
      aie::broadcast<bfloat16, 32>((bfloat16)0.125f);
  for (int i = 0; i < total_elems; i += 32) {
    aie::vector<bfloat16, 32> v = aie::load_v<32>(src + i);
    auto acc = aie::mul(v, scale);
    aie::store_v(dst + i, acc.template to_vector<bfloat16>());
  }
}

//------------------------------------------------------------------------------
// GQA group-aware kernels: process group_size Q heads sharing 1 K, 1 V cache.
// K and V rows are read once per history position and reused across the group.
//------------------------------------------------------------------------------
#ifndef GROUP_SIZE
#define GROUP_SIZE 4
#endif

// attn_1_group:
//   Q       : [GROUP_SIZE, DIM_N] bf16 (Q heads after RoPE)
//   k_row   : [DIM_N] bf16 (one K row from cache)
//   pos     : history position (column to write)
//   attn_out: [GROUP_SIZE, SEQ_LEN] bf16 (per-Q-head score matrix)
// Writes attn_out[g, pos] = dot(Q[g], k_row) / sqrt(DIM_N) for g in
// 0..GROUP_SIZE.
void attn_1_group(const bfloat16 *Q, const bfloat16 *k_row, int pos,
                  bfloat16 *attn_out) {
  STUB_RET;
  // 32-lane (512-bit) vectors: AIE2P registers are 512-bit so bf16<32>
  // fits natively. Halves K loads (4→2) and macs per Q head (4→2) vs
  // 16-lane. accum<accfloat,32> = 1024-bit accumulator. 4 independent
  // acc chains (one per Q head) so the compiler can interleave macs.
  static_assert(GROUP_SIZE == 4 && DIM_N == 64,
                "attn_1_group unrolled for GROUP_SIZE=4, DIM_N=64");
  aie::vector<bfloat16, 32> k0 = aie::load_v<32>(k_row + 0);
  aie::vector<bfloat16, 32> k1 = aie::load_v<32>(k_row + 32);

  // Fully unrolled per-Q-head: 4 independent acc chains for VLIW
  // interleaving, no if-else dispatch chain.
  aie::accum<accfloat, 32> a0 = aie::zeros<accfloat, 32>();
  aie::accum<accfloat, 32> a1 = aie::zeros<accfloat, 32>();
  aie::accum<accfloat, 32> a2 = aie::zeros<accfloat, 32>();
  aie::accum<accfloat, 32> a3 = aie::zeros<accfloat, 32>();

  a0 = aie::mac(a0, aie::load_v<32>(Q + 0 * DIM_N + 0), k0);
  a1 = aie::mac(a1, aie::load_v<32>(Q + 1 * DIM_N + 0), k0);
  a2 = aie::mac(a2, aie::load_v<32>(Q + 2 * DIM_N + 0), k0);
  a3 = aie::mac(a3, aie::load_v<32>(Q + 3 * DIM_N + 0), k0);
  a0 = aie::mac(a0, aie::load_v<32>(Q + 0 * DIM_N + 32), k1);
  a1 = aie::mac(a1, aie::load_v<32>(Q + 1 * DIM_N + 32), k1);
  a2 = aie::mac(a2, aie::load_v<32>(Q + 2 * DIM_N + 32), k1);
  a3 = aie::mac(a3, aie::load_v<32>(Q + 3 * DIM_N + 32), k1);

  // Q is pre-scaled by 1/sqrt(DIM_N)=0.125 in copy_scale_q_bf16 once per
  // token, so no per-call scalar multiply here.
  attn_out[0 * SEQ_LEN + pos] =
      (bfloat16)(aie::reduce_add(a0.to_vector<float>()));
  attn_out[1 * SEQ_LEN + pos] =
      (bfloat16)(aie::reduce_add(a1.to_vector<float>()));
  attn_out[2 * SEQ_LEN + pos] =
      (bfloat16)(aie::reduce_add(a2.to_vector<float>()));
  attn_out[3 * SEQ_LEN + pos] =
      (bfloat16)(aie::reduce_add(a3.to_vector<float>()));
}

// attn_2_group:
//   softmax : [GROUP_SIZE, SEQ_LEN] bf16
//   v_row   : [DIM_N] bf16 (one V row from cache)
//   pos     : history position
//   xb_out  : [GROUP_SIZE, DIM_N] bf16 (per-Q-head accumulator)
// xb_out[g, :] += softmax[g, pos] * v_row[:].
void attn_2_group(const bfloat16 *softmax, const bfloat16 *v_row, int pos,
                  bfloat16 *xb_out) {
  STUB_RET;
  // Fully unrolled across all 4 Q heads with V loads hoisted (shared across
  // the group) and 8 independent acc chains for VLIW interleaving. Old
  // for-loop reloaded v_row inside each g iter (8 redundant loads/call).
  static_assert(GROUP_SIZE == 4 && DIM_N == 64,
                "attn_2_group unrolled for GROUP_SIZE=4, DIM_N=64");
  aie::vector<bfloat16, 32> v0 = aie::load_v<32>(v_row + 0);
  aie::vector<bfloat16, 32> v1 = aie::load_v<32>(v_row + 32);

  aie::vector<bfloat16, 32> w0 =
      aie::broadcast<bfloat16, 32>(softmax[0 * SEQ_LEN + pos]);
  aie::vector<bfloat16, 32> w1 =
      aie::broadcast<bfloat16, 32>(softmax[1 * SEQ_LEN + pos]);
  aie::vector<bfloat16, 32> w2 =
      aie::broadcast<bfloat16, 32>(softmax[2 * SEQ_LEN + pos]);
  aie::vector<bfloat16, 32> w3 =
      aie::broadcast<bfloat16, 32>(softmax[3 * SEQ_LEN + pos]);

  bfloat16 *o0 = xb_out + 0 * DIM_N;
  bfloat16 *o1 = xb_out + 1 * DIM_N;
  bfloat16 *o2 = xb_out + 2 * DIM_N;
  bfloat16 *o3 = xb_out + 3 * DIM_N;

  aie::accum<accfloat, 32> a0_0, a0_1, a1_0, a1_1, a2_0, a2_1, a3_0, a3_1;
  a0_0.from_vector(aie::load_v<32>(o0 + 0));
  a1_0.from_vector(aie::load_v<32>(o1 + 0));
  a2_0.from_vector(aie::load_v<32>(o2 + 0));
  a3_0.from_vector(aie::load_v<32>(o3 + 0));
  a0_1.from_vector(aie::load_v<32>(o0 + 32));
  a1_1.from_vector(aie::load_v<32>(o1 + 32));
  a2_1.from_vector(aie::load_v<32>(o2 + 32));
  a3_1.from_vector(aie::load_v<32>(o3 + 32));

  a0_0 = mac(a0_0, w0, v0);
  a1_0 = mac(a1_0, w1, v0);
  a2_0 = mac(a2_0, w2, v0);
  a3_0 = mac(a3_0, w3, v0);
  a0_1 = mac(a0_1, w0, v1);
  a1_1 = mac(a1_1, w1, v1);
  a2_1 = mac(a2_1, w2, v1);
  a3_1 = mac(a3_1, w3, v1);

  aie::store_v(o0 + 0, a0_0.template to_vector<bfloat16>());
  aie::store_v(o1 + 0, a1_0.template to_vector<bfloat16>());
  aie::store_v(o2 + 0, a2_0.template to_vector<bfloat16>());
  aie::store_v(o3 + 0, a3_0.template to_vector<bfloat16>());
  aie::store_v(o0 + 32, a0_1.template to_vector<bfloat16>());
  aie::store_v(o1 + 32, a1_1.template to_vector<bfloat16>());
  aie::store_v(o2 + 32, a2_1.template to_vector<bfloat16>());
  aie::store_v(o3 + 32, a3_1.template to_vector<bfloat16>());
}

void simple_rms_bf16(bfloat16 *x, bfloat16 *w, bfloat16 *y, int32_t N) {
  STUB_RET;
  rms_kernel_bf16(x, w, y, N, 1e-5f);
}

} // extern "C"
