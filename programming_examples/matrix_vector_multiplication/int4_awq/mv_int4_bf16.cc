//===- mv_int4_bf16.cc - AWQ uint4 weight x bf16 activation matvec --------===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Per-tile micro-kernels for the int4-AWQ GEMV examples:
//   - matvec_int4_bf16_packed(packed, b, c):
//         c[0..m] += dequant(A)[m, k] @ b[k]
//         where dequant(A)[r, k] = (q[r, k] - z[r, g(k)]) * s_a[r, g(k)],
//         q ∈ uint4, z ∈ uint4 (stored uint8), s_a ∈ bf16, b ∈ bf16, c ∈ bf16.
//         The packed L1 BO contains Q, S, Z back-to-back per tile:
//             [ Q : m * k/2 bytes uint8 (row-major) ]
//             [ S : k/gs * m bf16                   ]
//             [ Z : k/gs * m uint8                  ]
//         Offsets are 32-byte aligned when m and gs are powers of two ≥ 16.
//   - zero_vectorized_bf16(c):           c[0..m] = 0
//   - partial_plus_r_bf16(p, r, off, d): d[0..m] = p[0..m] + r[off..off+m]
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_M
#define DIM_M 8
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

void partial_plus_r_bf16(bfloat16 *partial, bfloat16 *r_full, int offset,
                         bfloat16 *d) {
  partial_plus_r_impl<DIM_M>(partial, r_full, offset, d);
}

} // extern "C"
