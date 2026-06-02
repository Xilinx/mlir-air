//===- dequant.cc - AWQ int4 -> bfloat16 dequantization kernel ------------===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Vectorized AWQ int4 -> bfloat16 dequantization.
//
// Input layout (single packed L1 BO per tile -- matches the production
// int4-AWQ GEMV/GEMM packing in matrix_vector_multiplication/int4_awq):
//   [ Q : DIM_N/2 bytes uint8 -- two int4 weights per byte (low nibble first) ]
//   [ S : DIM_N/GROUP_SIZE bf16 -- per-group scale ]
//   [ Z : DIM_N/GROUP_SIZE uint8 -- per-group zero point (int4 range) ]
//
// Output:
//   DIM_N bfloat16 -- (weight - zero) * scale, group-wise.
//
// Inner loop processes R=32 nibbles per iteration via the AIE int4 unpack
// intrinsic, broadcast zero-point subtract, int8->bf16 conversion, and a
// single per-group scalar scale multiply.

#include <aie_api/aie.hpp>
#include <cstdint>

#ifndef DIM_N
#define DIM_N 256
#endif

#ifndef GROUP_SIZE
#define GROUP_SIZE 128
#endif

static_assert(DIM_N % 2 == 0,
              "DIM_N must be even (two int4 weights packed per byte)");
static_assert(DIM_N % GROUP_SIZE == 0,
              "DIM_N must be a multiple of GROUP_SIZE");
// The vectorized inner loop processes r=32 nibbles per iteration; the
// templated impl requires gs to be a multiple of r so each group fills a
// whole number of vector iterations.
static_assert(GROUP_SIZE % 32 == 0,
              "GROUP_SIZE must be a multiple of 32 (inner vector width)");

template <unsigned n, unsigned gs, unsigned r = 32>
static void
dequant_int4_bf16_impl(uint8_t *__restrict q, bfloat16 *__restrict s,
                       uint8_t *__restrict z, bfloat16 *__restrict out) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  static_assert(gs % r == 0, "group size must be multiple of inner vector r");
  constexpr unsigned NG = n / gs;
  constexpr unsigned NSUB = gs / r;

  for (unsigned g = 0; g < NG; g++) {
    aie::vector<int8, r> zv = aie::broadcast<int8, r>((int8_t)z[g]);
    bfloat16 sv = s[g];

#pragma clang loop unroll(full)
    for (unsigned i = 0; i < NSUB; i++) {
      const unsigned base = g * gs + i * r;
      aie::vector<uint8, r / 2> pk = aie::load_v<r / 2>(q + base / 2);
      aie::vector<int8, r> w_i8 =
          pk.template cast_to<uint4>().template unpack_sign<int8>(false);
      w_i8 = aie::sub(w_i8, zv);
      aie::vector<bfloat16, r> w_bf16 = aie::to_float<bfloat16>(w_i8, 0);

      aie::accum<accfloat, r> macc;
      macc.from_vector(aie::zeros<float, r>());
      macc = aie::mac(macc, w_bf16, sv);
      aie::store_v(out + base, macc.template to_vector<bfloat16>());
    }
  }
}

extern "C" {

// Packed-BO entry. Q + S + Z are concatenated in a single L1 buffer so the
// compute tile stays within its 2-S2MM channel budget (1 S2MM for the
// packed input, 1 MM2S for the dequantized output).
void dequant_int4_bf16(uint8_t *__restrict packed,
                       bfloat16 *__restrict output) {
  constexpr unsigned Q_BYTES = DIM_N / 2;
  constexpr unsigned S_BYTES = (DIM_N / GROUP_SIZE) * 2;
  uint8_t *q = packed;
  bfloat16 *s = reinterpret_cast<bfloat16 *>(packed + Q_BYTES);
  uint8_t *z = packed + Q_BYTES + S_BYTES;
  dequant_int4_bf16_impl<DIM_N, GROUP_SIZE>(q, s, z, output);
}

} // extern "C"
