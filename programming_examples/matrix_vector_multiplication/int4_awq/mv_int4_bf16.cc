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

static_assert(DIM_K % DIM_GS == 0, "DIM_K must be a multiple of DIM_GS");

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
// Phase 1: pack row-major A into a_pack and dequant W (with per-group bf16
// scale folded in) into b_pack — both in the operand layouts that the bf16
// GEMM kernel's MMUL loop expects (K-major A, N-major B).
// Phase 2: MMUL loop yields f32 accumulators; convert to bf16 and add to
// the existing row-major c.
template <unsigned m_tile, unsigned n_tile, unsigned k_chunk, unsigned gs,
          unsigned R = 32>
void mm_int4_bf16_mmul_impl(uint8_t *__restrict a_q, bfloat16 *__restrict a_s,
                            uint8_t *__restrict a_z, bfloat16 *__restrict a,
                            bfloat16 *__restrict c) {
  constexpr unsigned r = 8, s = 8, t = 8;
  constexpr unsigned MB = m_tile / r;
  constexpr unsigned NB = n_tile / t;
  constexpr unsigned KB = k_chunk / s;
  constexpr unsigned NG = k_chunk / gs;
  static_assert(m_tile % r == 0, "m_tile must be multiple of 8");
  static_assert(n_tile % t == 0, "n_tile must be multiple of 8");
  static_assert(k_chunk % s == 0, "k_chunk must be multiple of 8");
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

  // Dequant W (with scale fold) → [NB][KB][s][t]. One (g, n, i) iteration
  // produces R K-values for fixed n that scatter across R/s k-blocks at
  // stride t within each k-block.
  for (unsigned g = 0; g < NG; g++) {
    for (unsigned n = 0; n < n_tile; n++) {
      bfloat16 sc = a_s[g * n_tile + n];
      aie::vector<int8, R> zv =
          aie::broadcast<int8, R>((int8_t)a_z[g * n_tile + n]);
      aie::vector<bfloat16, R> sv = aie::broadcast<bfloat16, R>(sc);
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
        aie::vector<bfloat16, R> w_scaled =
            aie::mul(w_bf16, sv).template to_vector<bfloat16>();

        unsigned k_b_base = k_base / s;
        bfloat16 *base = b_pack + n_b * (KB * s * t) + n_i;
#pragma clang loop unroll(full)
        for (unsigned j = 0; j < R; j++) {
          unsigned k_b = k_b_base + j / s;
          unsigned k_i = j % s;
          base[k_b * (s * t) + k_i * t] = w_scaled[j];
        }
      }
    }
  }

  // MMUL: one accumulator per (m_b, n_b) tile, reduced across KB k-blocks.
  for (unsigned m_b = 0; m_b < MB; m_b++) {
    for (unsigned n_b = 0; n_b < NB; n_b++) {
      MMUL C;
#pragma clang loop unroll(full)
      for (unsigned k_b = 0; k_b < KB; k_b++) {
        aie::vector<bfloat16, MMUL::size_A> A =
            aie::load_v<MMUL::size_A>(a_pack + (k_b * MB + m_b) * (r * s));
        aie::vector<bfloat16, MMUL::size_B> B =
            aie::load_v<MMUL::size_B>(b_pack + (n_b * KB + k_b) * (s * t));
        if (k_b == 0)
          C.mul(A, B);
        else
          C.mac(A, B);
      }
      aie::vector<bfloat16, r * t> ctile = C.template to_vector<bfloat16>();
      for (unsigned m_i = 0; m_i < r; m_i++) {
        aie::vector<bfloat16, t> row = ctile.template extract<t>(m_i);
        bfloat16 *cdst = c + (m_b * r + m_i) * n_tile + n_b * t;
        aie::vector<bfloat16, t> c_old = aie::load_v<t>(cdst);
        aie::vector<bfloat16, t> c_new = aie::add(row, c_old);
        aie::store_v(cdst, c_new);
      }
    }
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

// Packed-BO GEMM entry. Same Q+S+Z packing as the GEMV (output-major W),
// driven by an m_tile-row activation tile a[].
void matmul_int4_bf16_packed(uint8_t *packed, bfloat16 *a, bfloat16 *c) {
  constexpr unsigned Q_BYTES = DIM_N * (DIM_K / 2);
  constexpr unsigned S_BYTES = (DIM_K / DIM_GS) * DIM_N * 2;
  uint8_t *a_q = packed;
  bfloat16 *a_s = reinterpret_cast<bfloat16 *>(packed + Q_BYTES);
  uint8_t *a_z = packed + Q_BYTES + S_BYTES;
  mm_int4_bf16_mmul_impl<DIM_M, DIM_N, DIM_K, DIM_GS>(a_q, a_s, a_z, a, c);
}

void partial_plus_r_bf16(bfloat16 *partial, bfloat16 *r_full, int offset,
                         bfloat16 *d) {
  partial_plus_r_impl<DIM_M>(partial, r_full, offset, d);
}

} // extern "C"
