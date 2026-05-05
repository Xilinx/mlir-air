//===- mha_gqa.cc -----------------------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
// Step 2b: AIE2P mha kernels at LLaMA dims, GQA group_size=4.
// 4 Q heads share 1 K and 1 V (LLaMA GQA). K/V cache rows are read
// ONCE per history position and reused across the 4 Q heads via
// attn_1_group / attn_2_group / softmax_group kernels.
// Inherits Step 2a's dk=64 freq table and rope/sinf/cosf kernels.
//
// Differences from mha.cc:
// - sinf/cosf: use rope_sincos-style polynomial with native n=16 vectors
//   and 32-elem padded internal buffers (24-elem outer interface preserved)
// - shuffle_apply_rope: use AIE2P-friendly aie::filter_even/odd +
//   interleave_zip (no shuffle_T16_* constants)
// - softmax: aie::exp2-based (AIE2P has no LUT-based exp lowering)
// - vecmat, attn_1, attn_2, vector_copy, linalg_fill: ported as-is
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <type_traits>

// AIE2P native vector width for accfloat ops.
static constexpr unsigned VEC = 16;

//------------------------------------------------------------------------------
// vecmat: c[offset:offset+n] += a[k] @ b[k, n]. Tile sizes baked at compile.
// Same template body as mha.cc — uses portable aie_api ops.
//------------------------------------------------------------------------------
template <typename T_in, typename T_out, unsigned k, unsigned n, unsigned s,
          unsigned t>
static void vecmat_vectorized(int offset, T_in *__restrict a,
                              T_in *__restrict b, T_out *__restrict c) {
  static_assert(n % t == 0 && k % 2 == 0);
  static_assert(s == 8);
  static_assert(k % s == 0);
  static_assert(std::is_same<T_in, bfloat16>::value);

  T_out *__restrict c_ptr = c + offset;
  for (int col = 0; col < (int)n; col += t) {
    const T_in *__restrict a_ptr1 = a;
    const T_in *__restrict b_ptr1 = b;
    for (int row = 0; row < (int)k; row += 8) {
      aie::vector<T_in, 8> a_vec = aie::load_v<8>(a_ptr1);
      a_ptr1 += 8;
      aie::accum<accfloat, t> c_acc;
      c_acc.from_vector(aie::load_v<t>(c_ptr));
      for (int i = 0; i < 8; i++) {
        const aie::vector<T_in, t> b_vec = aie::load_v<t>(b_ptr1);
        b_ptr1 += n;
        const aie::vector<T_in, t> s0 = aie::broadcast<T_in, t>(a_vec[i]);
        c_acc = mac(c_acc, s0, b_vec);
      }
      aie::store_v(c_ptr, c_acc.template to_vector<T_out>());
    }
    b += t;
    c_ptr += t;
  }
}

//------------------------------------------------------------------------------
// sinf_cosf_poly_bf16: ported from rope_sincos/rope.cc. Native n=16 vectors
// for AIE2P; processes N elements via N/n iterations.
//------------------------------------------------------------------------------
template <unsigned N, unsigned n, bool isSine>
static void sinf_cosf_poly_bf16(const bfloat16 *__restrict inputs,
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
    aie::vector<bfloat16, n> ux = aie::load_v<n>(inputs + i);
    aie::accum<accfloat, n> ux_acc, x_acc, zp_acc, one_acc;
    aie::accum<accfloat, n> poly_acs, acs0, acs2, out_acs;
    aie::accum<accfloat, n> poly_acc, acc0, acc2, out_acc, output_acc;
    aie::vector<bfloat16, n> x, r, r2, out_abs;
    aie::vector<bfloat16, n> s0, s1, s2, s3, c0, c1, c2, c3;
    aie::vector<bfloat16, n> twobypi, oneby2, negoneby2, piby2_1, one, negone;
    s0 = aie::broadcast<bfloat16, n>(sin_poly_factors[0]);
    s1 = aie::broadcast<bfloat16, n>(sin_poly_factors[1]);
    s2 = aie::broadcast<bfloat16, n>(sin_poly_factors[2]);
    s3 = aie::broadcast<bfloat16, n>(sin_poly_factors[3]);
    c0 = aie::broadcast<bfloat16, n>(cos_poly_factors[0]);
    c1 = aie::broadcast<bfloat16, n>(cos_poly_factors[1]);
    c2 = aie::broadcast<bfloat16, n>(cos_poly_factors[2]);
    c3 = aie::broadcast<bfloat16, n>(cos_poly_factors[3]);
    twobypi = aie::broadcast<bfloat16, n>(twobypival);
    piby2_1 = aie::broadcast<bfloat16, n>(piby2_1val);
    oneby2 = aie::broadcast<bfloat16, n>(0.5);
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

    // Range reduction with scalar quadrant detection
    zp_acc.from_vector(oneby2);
    ux_acc.from_vector(ux);
    zp_acc = mac(zp_acc, ux, twobypi);
    aie::vector<bfloat16, n> zp_bf16 = zp_acc.template to_vector<bfloat16>();
    alignas(aie::vector_decl_align) bfloat16 zp_buf[n];
    aie::store_v(zp_buf, zp_bf16);
    alignas(aie::vector_decl_align) bfloat16 z_buf[n];
    aie::mask<n> is_odd, is_neg;
    {
      uint32_t odd_bits = 0, neg_bits = 0;
      for (unsigned j = 0; j < n; j++) {
        int zj = (int)(float)zp_buf[j];
        z_buf[j] = (bfloat16)(float)zj;
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
    aie::vector<bfloat16, n> z_rounded = aie::load_v<n>(z_buf);
    x_acc = msc(ux_acc, piby2_1, z_rounded);

    x = x_acc.template to_vector<bfloat16>();
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
// freq_pos: out[i] = pos / 10000^(2i/dk). dk=64 => 32 entries (no padding;
// fits exactly one native n=16 vector pair).
//------------------------------------------------------------------------------
alignas(aie::vector_decl_align) static const bfloat16 freq_table_dk64[32] = {
    (bfloat16)1.000000f,  (bfloat16)0.749894f,  (bfloat16)0.562341f,
    (bfloat16)0.421697f,  (bfloat16)0.316228f,  (bfloat16)0.237137f,
    (bfloat16)0.177828f,  (bfloat16)0.133352f,  (bfloat16)0.100000f,
    (bfloat16)0.074989f,  (bfloat16)0.056234f,  (bfloat16)0.042170f,
    (bfloat16)0.031623f,  (bfloat16)0.023714f,  (bfloat16)0.017783f,
    (bfloat16)0.013335f,  (bfloat16)0.010000f,  (bfloat16)0.007499f,
    (bfloat16)0.005623f,  (bfloat16)0.004217f,  (bfloat16)0.003162f,
    (bfloat16)0.002371f,  (bfloat16)0.001778f,  (bfloat16)0.001334f,
    (bfloat16)0.001000f,  (bfloat16)0.000750f,  (bfloat16)0.000562f,
    (bfloat16)0.000422f,  (bfloat16)0.000316f,  (bfloat16)0.000237f,
    (bfloat16)0.000178f,  (bfloat16)0.000133f,
};

template <unsigned N, unsigned n>
static void freq_pos_bf16(int pos, const bfloat16 *freq_table,
                          bfloat16 *__restrict outputs) {
  aie::vector<bfloat16, n> vecPos =
      aie::broadcast<bfloat16, n>(aie::to_float<bfloat16>(pos));
  for (unsigned i = 0; i < N; i += n) {
    aie::vector<bfloat16, n> v = aie::load_v<n>(freq_table + i);
    aie::store_v(outputs + i,
                 aie::mul(vecPos, v).template to_vector<bfloat16>());
  }
}

//------------------------------------------------------------------------------
// shuffle_apply_rope: RoPE rotation via filter_even/odd + interleave_zip.
// AIE2P-portable (no shuffle_T16_* constants).
//------------------------------------------------------------------------------
template <unsigned N, unsigned n>
static void shuffle_apply_rope(int offset, const bfloat16 *__restrict fcr,
                               const bfloat16 *__restrict fci,
                               bfloat16 *__restrict outputs) {
  constexpr unsigned two_n = n * 2;
  const bfloat16 *__restrict pFcr = fcr;
  const bfloat16 *__restrict pFci = fci;
  for (unsigned i = 0; i < N; i += two_n) {
    aie::vector<bfloat16, two_n> v0 = aie::load_v<two_n>(outputs + i + offset);
    aie::vector<bfloat16, n> v0Even = aie::filter_even(v0, 1);
    aie::vector<bfloat16, n> v0Odd = aie::filter_odd(v0, 1);
    aie::vector<bfloat16, n> vFcr = aie::load_v<n>(pFcr);
    pFcr += n;
    aie::vector<bfloat16, n> vFci = aie::load_v<n>(pFci);
    pFci += n;
    aie::vector<bfloat16, n> vOutEven =
        msc(aie::mul(v0Even, vFcr), v0Odd, vFci).template to_vector<bfloat16>();
    aie::vector<bfloat16, n> vOutOdd =
        mac(aie::mul(v0Even, vFci), v0Odd, vFcr).template to_vector<bfloat16>();
    auto [vOutLo, vOutHi] = aie::interleave_zip(vOutEven, vOutOdd, 1);
    bfloat16 *__restrict pOut = outputs + i + offset;
    aie::store_v(pOut, vOutLo);
    aie::store_v(pOut + n, vOutHi);
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
// softmax_bf16 using aie::exp2 (no LUT). exp(x) = exp2(x * log2(e)).
// num_elems is ignored — the caller (mha.py) pre-fills the [SEQ_LEN] input
// with -99 beyond pos+1 so we can process the entire SEQ_LEN buffer
// uniformly. Out-of-range entries contribute ~0 to the sum.
//------------------------------------------------------------------------------
#ifndef SEQ_LEN
#define SEQ_LEN 128
#endif

static_assert(SEQ_LEN % 16 == 0, "SEQ_LEN must be multiple of 16");

static void softmax_bf16_impl(const bfloat16 *__restrict in, int /*num_elems*/,
                              bfloat16 *__restrict out) {
  constexpr unsigned V = 16;
  constexpr int N = SEQ_LEN;
  // Pass 1: vector max reduction
  aie::vector<bfloat16, V> max_vec =
      aie::broadcast<bfloat16, V>((bfloat16)-1e30f);
  for (int i = 0; i < N; i += V) {
    aie::vector<bfloat16, V> v = aie::load_v<V>(in + i);
    max_vec = aie::max(max_vec, v);
  }
  bfloat16 max_val = aie::reduce_max(max_vec);

  // Pass 2: exp(x - max), accumulate sum
  const float LOG2E = 1.4426950408889634f;
  aie::vector<bfloat16, V> log2e_vec = aie::broadcast<bfloat16, V>(LOG2E);
  aie::vector<bfloat16, V> max_bcast = aie::broadcast<bfloat16, V>(max_val);
  aie::accum<accfloat, V> sum_acc = aie::zeros<accfloat, V>();
  for (int i = 0; i < N; i += V) {
    aie::vector<bfloat16, V> v = aie::load_v<V>(in + i);
    aie::vector<bfloat16, V> diff = aie::sub(v, max_bcast);
    aie::vector<bfloat16, V> e =
        aie::exp2<bfloat16>(aie::mul(diff, log2e_vec).to_vector<float>());
    aie::store_v(out + i, e);
    sum_acc = aie::add(sum_acc, e);
  }
  float total = aie::reduce_add(sum_acc.template to_vector<float>());

  // Pass 3: divide by sum
  float inv = 1.0f / total;
  aie::vector<bfloat16, V> inv_bcast = aie::broadcast<bfloat16, V>(inv);
  for (int i = 0; i < N; i += V) {
    aie::vector<bfloat16, V> v = aie::load_v<V>(out + i);
    aie::store_v(out + i,
                 aie::mul(v, inv_bcast).template to_vector<bfloat16>());
  }
}

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

  aie::vector<bfloat16, VL> rstd_v = aie::broadcast<bfloat16, VL>((bfloat16)rstd);
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

extern "C" {

void linalg_fill_bf16(int offset, bfloat16 *c) {
  bfloat16 *p = c + offset;
  aie::vector<bfloat16, 16> z = aie::zeros<bfloat16, 16>();
  for (int i = 0; i < DIM_N; i += 16)
    aie::store_v(p + i, z);
}

// Inner-k chunked GEMV. x_offset is in elements (caller passes
// chunk_idx * TILE_K). Reads a[x_offset..x_offset+TILE_K] @ b[TILE_K, DIM_N]
// and accumulates into c[offset..offset+DIM_N]. Caller must zero c (via
// linalg_fill_bf16) before the first chunk in a given GEMV head.
void vecmat_bf16_bf16(int x_offset, int offset, bfloat16 *a, bfloat16 *b,
                      bfloat16 *c) {
  vecmat_vectorized<bfloat16, bfloat16, TILE_K, DIM_N, 8, 16>(offset,
                                                              a + x_offset, b,
                                                              c);
}

// dk=64 => 32-elem buffers, native n=16 vectors. No padding needed.
void freq_pos_bf16_32_16(int pos, bfloat16 *outputs) {
  freq_pos_bf16<32, VEC>(pos, freq_table_dk64, outputs);
}

void sinf_bf16_32_16(const bfloat16 *inputs, bfloat16 *outputs) {
  sinf_cosf_poly_bf16<32, VEC, true>(inputs, outputs);
}

void cosf_bf16_32_16(const bfloat16 *inputs, bfloat16 *outputs) {
  sinf_cosf_poly_bf16<32, VEC, false>(inputs, outputs);
}

void shuffle_apply_rope_bf16_64(int offset, const bfloat16 *fcr,
                                const bfloat16 *fci, bfloat16 *outputs) {
  // dk=64 = exactly 2 * native n=16. No padding wrapper needed.
  shuffle_apply_rope<HEAD_SIZE, VEC>(offset, fcr, fci, outputs);
}

// xrms_demux_bf16: split a [tile_k, tile_n] padded packet (laid out flat)
// into x_raw[k] and w_rms[k]. The padded packet contains [x_raw[0:k],
// w_rms[0:k], junk...] in row-major order. K elements each, vector-copy.
void xrms_demux_bf16(const bfloat16 *padded, bfloat16 *x_raw, bfloat16 *w_rms,
                     int32_t k) {
  constexpr int VL = 16;
  for (int i = 0; i < k; i += VL)
    aie::store_v(x_raw + i, aie::load_v<VL>(padded + i));
  const bfloat16 *w_src = padded + k;
  for (int i = 0; i < k; i += VL)
    aie::store_v(w_rms + i, aie::load_v<VL>(w_src + i));
}

// Vectorized fills to replace per-element scf.for stores in the herd
// body — those unroll into thousands of scalar stores in the AIE2P core
// ELF, exceeding the 16 KB program memory.
void fill_neg99_bf16(bfloat16 *p, int32_t total_elems) {
  aie::vector<bfloat16, 16> v = aie::broadcast<bfloat16, 16>((bfloat16)-99);
  for (int i = 0; i < total_elems; i += 16)
    aie::store_v(p + i, v);
}

void fill_zero_bf16(bfloat16 *p, int32_t total_elems) {
  aie::vector<bfloat16, 16> v = aie::zeros<bfloat16, 16>();
  for (int i = 0; i < total_elems; i += 16)
    aie::store_v(p + i, v);
}

void vector_copy_n_bf16(const bfloat16 *src, bfloat16 *dst,
                        int32_t total_elems) {
  for (int i = 0; i < total_elems; i += 16)
    aie::store_v(dst + i, aie::load_v<16>(src + i));
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
// Writes attn_out[g, pos] = dot(Q[g], k_row) / sqrt(DIM_N) for g in 0..GROUP_SIZE.
void attn_1_group(const bfloat16 *Q, const bfloat16 *k_row, int pos,
                  bfloat16 *attn_out) {
  for (int g = 0; g < GROUP_SIZE; g++) {
    const bfloat16 *q_g = Q + g * DIM_N;
    float result = 0.0f;
    for (unsigned i = 0; i < DIM_N / 16; i++) {
      aie::vector<bfloat16, 16> qv = aie::load_v<16>(q_g + i * 16);
      aie::vector<bfloat16, 16> kv = aie::load_v<16>(k_row + i * 16);
      aie::accum<accfloat, 16> acc = aie::mul(qv, kv);
      result += aie::reduce_add(acc.to_vector<float>());
    }
    result *= 0.125f; // 1/sqrt(64)
    attn_out[g * SEQ_LEN + pos] = (bfloat16)result;
  }
}

// softmax_group: per-row softmax over [GROUP_SIZE, SEQ_LEN]. Same -inf-padding
// assumption as softmax_bf16 (caller pre-fills [pos+1..SEQ_LEN-1] with -99).
void softmax_group(const bfloat16 *attn_in, int num_elems, bfloat16 *attn_out) {
  for (int g = 0; g < GROUP_SIZE; g++) {
    softmax_bf16_impl(attn_in + g * SEQ_LEN, num_elems, attn_out + g * SEQ_LEN);
  }
}

// attn_2_group:
//   softmax : [GROUP_SIZE, SEQ_LEN] bf16
//   v_row   : [DIM_N] bf16 (one V row from cache)
//   pos     : history position
//   xb_out  : [GROUP_SIZE, DIM_N] bf16 (per-Q-head accumulator)
// xb_out[g, :] += softmax[g, pos] * v_row[:].
void attn_2_group(const bfloat16 *softmax, const bfloat16 *v_row, int pos,
                  bfloat16 *xb_out) {
  for (int g = 0; g < GROUP_SIZE; g++) {
    bfloat16 weight = softmax[g * SEQ_LEN + pos];
    aie::vector<bfloat16, 16> wB = aie::broadcast<bfloat16, 16>(weight);
    bfloat16 *out_g = xb_out + g * DIM_N;
    for (unsigned j = 0; j < DIM_N / 16; j++) {
      aie::accum<accfloat, 16> acc;
      acc.from_vector(aie::load_v<16>(out_g + j * 16));
      acc = mac(acc, wB, aie::load_v<16>(v_row + j * 16));
      aie::store_v(out_g + j * 16, acc.template to_vector<bfloat16>());
    }
  }
}

void simple_rms_bf16(bfloat16 *x, bfloat16 *w, bfloat16 *y, int32_t N) {
  rms_kernel_bf16(x, w, y, N, 1e-5f);
}

} // extern "C"
