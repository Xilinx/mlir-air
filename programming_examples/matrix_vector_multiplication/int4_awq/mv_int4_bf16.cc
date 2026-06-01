//===- mv_int4_bf16.cc - AWQ uint4 weight x bf16 activation matvec/matmul -===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Per-tile micro-kernels for the int4-AWQ GEMV and GEMM examples:
//   - matvec_int4_bf16_packed(packed, b, c):
//         c[0..m] += dequant(A)[m, k] @ b[k]
//         where dequant(A)[r, k] = (q[r, k] - z[r, g(k)]) * s_a[r, g(k)],
//         q ∈ uint4, z ∈ uint4 (stored uint8), s_a ∈ bf16, b ∈ bf16, c ∈ bf16.
//         The packed L1 BO contains Q, S, Z back-to-back per tile:
//             [ Q : m * k/2 bytes uint8 (row-major) ]
//             [ S : k/gs * m bf16                   ]
//             [ Z : k/gs * m uint8                  ]
//         Offsets are 32-byte aligned when m and gs are powers of two ≥ 16.
//   - matmul_int4_bf16_packed(packed, a, c):
//         c[0..m, 0..n] += a[0..m, 0..k] @ dequant(W)[0..k, 0..n]
//         W laid out as [n_tile, k/2] (output-major) — same packed layout
//         as the GEMV with M renamed to N. Used by the int4-AWQ GEMM prefill.
//   - zero_vectorized_bf16(c):           c[0..m] = 0
//   - zero_vectorized_bf16_mn(c):        c[0..m*n] = 0 (vectorized for GEMM)
//   - partial_plus_r_bf16(p, r, off, d): d[0..m] = p[0..m] + r[off..off+m]
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_M
#define DIM_M 8
#endif
#ifndef DIM_N
#define DIM_N 16
#endif
#ifndef DIM_K
#define DIM_K 2048
#endif
#ifndef DIM_GS
#define DIM_GS 128
#endif

// matmul_int4_bf16_packed's K is the per-call k_chunk (matches the host
// builder's TILE_K_L1), not the matvec's full K. Default to 128 so a build
// that only consumes the matvec (e.g. DIM_K=2048) doesn't instantiate the
// matmul template at a k_chunk that overflows L1 scratch.
#ifndef DIM_K_CHUNK
#define DIM_K_CHUNK 128
#endif

static_assert(DIM_K % DIM_GS == 0, "DIM_K must be a multiple of DIM_GS");
static_assert(DIM_K_CHUNK % DIM_GS == 0,
              "DIM_K_CHUNK must be a multiple of DIM_GS");

// int4-AWQ matvec inner kernel. One accfloat accumulator per output row spans
// the full K. Within each group we accumulate raw (w_bf16 × b) into a per-
// group accfloat, then fold the per-group bf16 scale via a single mac with
// the broadcast scalar — keeps the inner loop a tight load-unpack-sub-cvt-mac
// chain and avoids the per-group acc32→bf16 plumbing that an int8-mac design
// would impose.
template <unsigned m, unsigned k, unsigned gs, unsigned r>
void matvec_int4_bf16_impl(uint8_t *__restrict a_q, bfloat16 *__restrict a_s,
                           uint8_t *__restrict a_z, bfloat16 *__restrict b,
                           bfloat16 *__restrict c) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  static_assert(gs % r == 0, "group size must be multiple of inner vector r");
  constexpr unsigned NSUB = gs / r;

  for (unsigned row = 0; row < m; row++) {
    aie::accum<accfloat, r> acc;
    acc.from_vector(aie::zeros<float, r>());
    const uint8_t *__restrict aq = a_q + row * (k / 2);

    for (unsigned g = 0; g < k / gs; g++) {
      aie::vector<int8, r> zv =
          aie::broadcast<int8, r>((int8_t)a_z[g * m + row]);
      bfloat16 sa = a_s[g * m + row];

      aie::accum<accfloat, r> g_acc;
      g_acc.from_vector(aie::zeros<float, r>());

#pragma clang loop unroll(full)
      for (unsigned i = 0; i < NSUB; i++) {
        const unsigned off = (g * gs + i * r) / 2;
        aie::vector<uint8, r / 2> packed = aie::load_v<r / 2>(aq + off);
        aie::vector<int8, r> w_int8 =
            packed.template cast_to<uint4>().template unpack_sign<int8>(false);
        w_int8 = aie::sub(w_int8, zv);
        aie::vector<bfloat16, r> w_bf16 = aie::to_float<bfloat16>(w_int8, 0);
        aie::vector<bfloat16, r> b_vec = aie::load_v<r>(b + g * gs + i * r);
        g_acc = aie::mac(g_acc, w_bf16, b_vec);
      }

      aie::vector<bfloat16, r> g_bf16 = g_acc.template to_vector<bfloat16>();
      acc = aie::mac(acc, g_bf16, sa);
    }

    float s = aie::reduce_add(acc.template to_vector<float>());
    c[row] = (bfloat16)((float)c[row] + s);
  }
}

template <unsigned m>
static void zero_impl(bfloat16 *__restrict c) {
  for (unsigned i = 0; i < m; i++)
    c[i] = (bfloat16)0.0f;
}

// Vectorized zero for the GEMM C tile. Peano auto-vectorizes the scalar
// `for c[i] = 0` loop with a stride-4 store that skips every 4th element
// once the buffer is >= one full vector wide — the bug only manifests
// across repeated kernel calls that read-modify c[]. Explicit aie::store_v
// of aie::zeros avoids it.
template <unsigned m_tile, unsigned n_tile>
static void zero_mn_impl(bfloat16 *__restrict c) {
  constexpr unsigned VW = 32;
  constexpr unsigned NTOT = m_tile * n_tile;
  static_assert(NTOT % VW == 0,
                "m_tile*n_tile must be a multiple of vector width");
  aie::vector<bfloat16, VW> zv = aie::zeros<bfloat16, VW>();
  for (unsigned i = 0; i < NTOT; i += VW)
    aie::store_v(c + i, zv);
}

// int4-AWQ GEMM inner kernel using aie::mmul<8,8,8,bf16,bf16,accfloat>.
// Dequant produces UNSCALED bf16 W (just nibble unpack + zero subtract).
// Per (m_b, n_b): preload the f32 c tile, then for each group do an unscaled
// MMUL, convert to f32 vec, multiply by f32 scale (scalar, no extra bf16
// truncate), and accumulate into the c tile. Across host K-chunk calls the
// c buffer stays f32 (bf16-GEMM accum pattern). Matches mac-kernel's
// "truncate at group sum level, not per W element" precision pattern.
template <unsigned m_tile, unsigned n_tile, unsigned k_chunk, unsigned gs,
          unsigned R = 32>
void mm_int4_bf16_mmul_impl(uint8_t *__restrict a_q, bfloat16 *__restrict a_s,
                            uint8_t *__restrict a_z, bfloat16 *__restrict a,
                            float *__restrict c) {
  constexpr unsigned r = 8, s = 8, t = 8;
  constexpr unsigned MB = m_tile / r;
  constexpr unsigned NB = n_tile / t;
  constexpr unsigned KB = k_chunk / s;
  constexpr unsigned KB_PER_G = gs / s;
  constexpr unsigned NG = k_chunk / gs;
  static_assert(m_tile % r == 0, "m_tile must be multiple of 8");
  static_assert(n_tile % t == 0, "n_tile must be multiple of 8");
  static_assert(k_chunk % s == 0, "k_chunk must be multiple of 8");
  static_assert(gs % s == 0, "gs must be multiple of mmul k-tile (8)");
  static_assert(gs % R == 0, "gs must be multiple of R");

  ::aie::set_rounding(aie::rounding_mode::conv_even);

  using MMUL = aie::mmul<r, s, t, bfloat16, bfloat16, accauto>;

  alignas(32) bfloat16 a_pack[KB * MB * r * s];
  alignas(32) bfloat16 b_pack[NB * KB * s * t];

  // Pack A row-major [m_tile][k_chunk] → [KB][MB][r][s].
  for (unsigned k_b = 0; k_b < KB; k_b++) {
    for (unsigned m_b = 0; m_b < MB; m_b++) {
      bfloat16 *dst = a_pack + (k_b * MB + m_b) * (r * s);
      for (unsigned m_i = 0; m_i < r; m_i++) {
        aie::vector<bfloat16, s> v =
            aie::load_v<s>(a + (m_b * r + m_i) * k_chunk + k_b * s);
        aie::store_v(dst + m_i * s, v);
      }
    }
  }

  // Dequant W (NO scale fold) → [NB][KB][s][t].
  for (unsigned g = 0; g < NG; g++) {
    for (unsigned n = 0; n < n_tile; n++) {
      aie::vector<int8, R> zv =
          aie::broadcast<int8, R>((int8_t)a_z[g * n_tile + n]);
      unsigned n_b = n / t;
      unsigned n_i = n % t;
      const uint8_t *__restrict aq_n = a_q + n * (k_chunk / 2);

      for (unsigned i = 0; i < gs / R; i++) {
        unsigned k_base = g * gs + i * R;
        aie::vector<uint8, R / 2> pk = aie::load_v<R / 2>(aq_n + k_base / 2);
        aie::vector<int8, R> w_i8 =
            pk.template cast_to<uint4>().template unpack_sign<int8>(false);
        w_i8 = aie::sub(w_i8, zv);
        aie::vector<bfloat16, R> w_bf16 = aie::to_float<bfloat16>(w_i8, 0);

        unsigned k_b_base = k_base / s;
        bfloat16 *base = b_pack + n_b * (KB * s * t) + n_i;
#pragma clang loop unroll(full)
        for (unsigned j = 0; j < R; j++) {
          unsigned k_b = k_b_base + j / s;
          unsigned k_i = j % s;
          base[k_b * (s * t) + k_i * t] = w_bf16[j];
        }
      }
    }
  }

  // Per (m_b, n_b): preload c into scalar f32 scratch, per group MMUL +
  // scale fold + accumulate into scratch, store scratch back.
  for (unsigned m_b = 0; m_b < MB; m_b++) {
    for (unsigned n_b = 0; n_b < NB; n_b++) {
      float *cdst_base = c + (m_b * r) * n_tile + n_b * t;
      alignas(32) float c_acc_buf[r * t];
      for (unsigned m_i = 0; m_i < r; m_i++) {
        aie::vector<float, t> row = aie::load_v<t>(cdst_base + m_i * n_tile);
        aie::store_v(c_acc_buf + m_i * t, row);
      }

      for (unsigned g = 0; g < NG; g++) {
        // Per-group unscaled MMUL. Init accumulator from zeros (bf16-GEMM
        // pattern, avoids the runtime `first` flag).
        aie::vector<float, r * t> zero_init = aie::zeros<float, r * t>();
        MMUL C_g(zero_init);
#pragma clang loop unroll(full)
        for (unsigned kg = 0; kg < KB_PER_G; kg++) {
          unsigned k_b = g * KB_PER_G + kg;
          aie::vector<bfloat16, MMUL::size_A> A =
              aie::load_v<MMUL::size_A>(a_pack + (k_b * MB + m_b) * (r * s));
          aie::vector<bfloat16, MMUL::size_B> B =
              aie::load_v<MMUL::size_B>(b_pack + (n_b * KB + k_b) * (s * t));
          C_g.mac(A, B);
        }
        // Vectorized post-MMUL scale fold: convert C_g to bf16, multiply
        // by bf16 scale broadcast (mul → f32 accum, no extra truncate),
        // lift to f32 vec, accumulate row-by-row into c_acc_buf.
        aie::vector<bfloat16, r * t> c_g_bf16 =
            C_g.template to_vector<bfloat16>();
        // Build bf16 scale broadcast: (m_i, n_i) row-major, scale per n_i.
        // Load 8 scales into a vec<bf16, 8> and tile across r rows.
        alignas(32) bfloat16 scale_row_buf[t];
        for (unsigned n_i = 0; n_i < t; n_i++)
          scale_row_buf[n_i] = a_s[g * n_tile + n_b * t + n_i];
        aie::vector<bfloat16, t> scale_row = aie::load_v<t>(scale_row_buf);
        alignas(32) bfloat16 scale_tile_buf[r * t];
        for (unsigned m_i = 0; m_i < r; m_i++)
          aie::store_v(scale_tile_buf + m_i * t, scale_row);
        aie::vector<bfloat16, r * t> scale_tile =
            aie::load_v<r * t>(scale_tile_buf);
        aie::accum<accfloat, r * t> scaled_acc = aie::mul(c_g_bf16, scale_tile);
        aie::vector<float, r * t> scaled_f32 =
            scaled_acc.template to_vector<float>();
        // Accumulate into c_acc_buf row by row (vector load + add + store).
        for (unsigned m_i = 0; m_i < r; m_i++) {
          aie::vector<float, t> inc = scaled_f32.template extract<t>(m_i);
          aie::vector<float, t> old = aie::load_v<t>(c_acc_buf + m_i * t);
          aie::vector<float, t> sum = aie::add(old, inc);
          aie::store_v(c_acc_buf + m_i * t, sum);
        }
      }

      // Store accumulator back to c row-by-row.
      for (unsigned m_i = 0; m_i < r; m_i++) {
        aie::vector<float, t> row = aie::load_v<t>(c_acc_buf + m_i * t);
        aie::store_v(cdst_base + m_i * n_tile, row);
      }
    }
  }
}

// f32 zero for the GEMM C accumulator (kept in f32 across host K-chunk
// iterations to avoid bf16 truncation of partial sums).
template <unsigned m_tile, unsigned n_tile>
static void zero_mn_f32_impl(float *__restrict c) {
  constexpr unsigned VW = 16;
  constexpr unsigned NTOT = m_tile * n_tile;
  static_assert(NTOT % VW == 0,
                "m_tile*n_tile must be a multiple of f32 vector width");
  aie::vector<float, VW> zv = aie::zeros<float, VW>();
  for (unsigned i = 0; i < NTOT; i += VW)
    aie::store_v(c + i, zv);
}

// f32 → bf16 narrowing for the final L1 C drain. Run after the host's K-loop
// completes; one call per (m_tile × n_tile) tile.
template <unsigned m_tile, unsigned n_tile>
static void f32_to_bf16_mn_impl(const float *__restrict src,
                                bfloat16 *__restrict dst) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  constexpr unsigned VW = 16;
  constexpr unsigned NTOT = m_tile * n_tile;
  static_assert(NTOT % VW == 0, "m_tile*n_tile must be a multiple of VW");
  for (unsigned i = 0; i < NTOT; i += VW) {
    aie::vector<float, VW> v = aie::load_v<VW>(src + i);
    aie::vector<bfloat16, VW> vb;
    for (unsigned j = 0; j < VW; j++)
      vb[j] = (bfloat16)v[j];
    aie::store_v(dst + i, vb);
  }
}

template <unsigned m>
static void partial_plus_r_impl(const bfloat16 *__restrict partial,
                                const bfloat16 *__restrict r_full, int offset,
                                bfloat16 *__restrict d) {
  for (unsigned i = 0; i < m; i++)
    d[i] = (bfloat16)((float)partial[i] + (float)r_full[offset + i]);
}

extern "C" {

// Packed-BO entry. The single L1 input combines Q, S, Z so the compute tile
// stays within its 2-S2MM-channel budget when paired with the broadcast B.
void matvec_int4_bf16_packed(uint8_t *packed, bfloat16 *b, bfloat16 *c) {
  constexpr unsigned Q_BYTES = DIM_M * (DIM_K / 2);
  constexpr unsigned S_BYTES = (DIM_K / DIM_GS) * DIM_M * 2;
  uint8_t *a_q = packed;
  bfloat16 *a_s = reinterpret_cast<bfloat16 *>(packed + Q_BYTES);
  uint8_t *a_z = packed + Q_BYTES + S_BYTES;
  matvec_int4_bf16_impl<DIM_M, DIM_K, DIM_GS, 32>(a_q, a_s, a_z, b, c);
}

void zero_vectorized_bf16(bfloat16 *c) { zero_impl<DIM_M>(c); }

void zero_vectorized_bf16_mn(bfloat16 *c) { zero_mn_impl<DIM_M, DIM_N>(c); }

void zero_vectorized_f32_mn(float *c) { zero_mn_f32_impl<DIM_M, DIM_N>(c); }

void f32_to_bf16_mn(float *src, bfloat16 *dst) {
  f32_to_bf16_mn_impl<DIM_M, DIM_N>(src, dst);
}

// Packed-BO GEMM entry, f32 output. Same Q+S+Z packing as the GEMV
// (output-major W); c is kept in f32 across host-side K-chunk iterations so
// the per-K-chunk partial sums don't bf16-truncate. Convert to bf16 once at
// the end with f32_to_bf16_mn.
void matmul_int4_bf16_packed_f32(uint8_t *packed, bfloat16 *a, float *c) {
  constexpr unsigned Q_BYTES = DIM_N * (DIM_K_CHUNK / 2);
  constexpr unsigned S_BYTES = (DIM_K_CHUNK / DIM_GS) * DIM_N * 2;
  uint8_t *a_q = packed;
  bfloat16 *a_s = reinterpret_cast<bfloat16 *>(packed + Q_BYTES);
  uint8_t *a_z = packed + Q_BYTES + S_BYTES;
  mm_int4_bf16_mmul_impl<DIM_M, DIM_N, DIM_K_CHUNK, DIM_GS>(a_q, a_s, a_z, a,
                                                            c);
}

void partial_plus_r_bf16(bfloat16 *partial, bfloat16 *r_full, int offset,
                         bfloat16 *d) {
  partial_plus_r_impl<DIM_M>(partial, r_full, offset, d);
}

} // extern "C"
