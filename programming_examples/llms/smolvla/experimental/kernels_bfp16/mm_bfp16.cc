//===- mm_bfp16.cc - bf16 x bfp16ebs8 GEMM, suffixed, + SwiGLU drain -*- C++
//-*-===//
//
// SPDX-License-Identifier: MIT
//
// matrix_multiplication/bf16_x_bfp16/mm_bf16_x_bfp16.cc with every entry point
// renamed name##SYM_SUFFIX, so several differently tiled copies (DIM_M/N/K are
// compile-time) link into one multi-launch ELF, plus f32_to_bf16_swiglu_mn for
// a Gate|Up GEMM whose tile_n block holds tile_n/2 gate then tile_n/2 up
// columns. Compile with -I pointing at matrix_multiplication/bf16_x_bfp16/.
//
//===----------------------------------------------------------------------===//

#ifndef SYM_SUFFIX
#define SYM_SUFFIX
#endif
#define SYM_CAT2(a, b) a##b
#define SYM_CAT(a, b) SYM_CAT2(a, b)
#define SYM(name) SYM_CAT(name, SYM_SUFFIX)

#define matmul_bf16_x_bfp16_packed_f32 SYM(matmul_bf16_x_bfp16_packed_f32)
#define zero_vectorized_f32_mn SYM(zero_vectorized_f32_mn)
#define f32_to_bf16_mn SYM(f32_to_bf16_mn)
#include "mm_bf16_x_bfp16.cc"

extern "C" {

// The accumulator is (DIM_N/8, DIM_M/8, 8, 8), the same N-outer layout as the
// bf16 GEMM's, so this is kernels_swiglu/mm_swiglu.cc's drain verbatim: g and u
// narrowed to bf16 under conv_even, then silu_and_mul_bf16's math under FLOOR.
void SYM(f32_to_bf16_swiglu_mn)(float *src, bfloat16 *dst) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  constexpr unsigned VW = 16, T = 8;
  constexpr unsigned NB = DIM_N / T, H = NB / 2, BE = DIM_M * T;
  static_assert(NB % 2 == 0, "tile_n must hold whole gate/up block pairs");
  static_assert(BE % VW == 0, "block must hold whole vectors");
  const aie::vector<bfloat16, VW> half_v =
      aie::broadcast<bfloat16, VW>((bfloat16)0.5f);
  const aie::vector<bfloat16, VW> one_v =
      aie::broadcast<bfloat16, VW>((bfloat16)1.0f);
  for (unsigned jb = 0; jb < H; jb++) {
    const float *pg = src + jb * BE;
    const float *pu = src + (jb + H) * BE;
    bfloat16 *pd = dst + jb * BE;
    for (unsigned e = 0; e < BE; e += VW) {
      ::aie::set_rounding(aie::rounding_mode::conv_even);
      aie::vector<bfloat16, VW> g =
          narrow_f32_to_bf16<VW>(aie::load_v<VW>(pg + e));
      aie::vector<bfloat16, VW> u =
          narrow_f32_to_bf16<VW>(aie::load_v<VW>(pu + e));
      ::aie::set_rounding(aie::rounding_mode::floor);
      aie::vector<bfloat16, VW> g_half = aie::mul(g, half_v);
      aie::accum<accfloat, VW> tanh_in;
      tanh_in.from_vector(g_half);
      aie::vector<bfloat16, VW> tanh_val =
          aie::tanh<bfloat16>(tanh_in.template to_vector<float>());
      aie::vector<bfloat16, VW> sigmoid =
          aie::mul(half_v, aie::add(one_v, tanh_val));
      aie::vector<bfloat16, VW> silu = aie::mul(g, sigmoid);
      aie::vector<bfloat16, VW> out = aie::mul(silu, u);
      aie::store_v(pd + e, out);
    }
  }
}

} // extern "C"
