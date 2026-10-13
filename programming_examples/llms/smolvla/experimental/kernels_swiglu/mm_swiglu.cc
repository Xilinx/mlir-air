//===- mm_swiglu.cc - mm_aie2p.cc + a SwiGLU drain epilogue -----*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
//
// The shared GEMM microkernel plus f32_to_bf16_swiglu_mn, the drain for
// build_module(epilogue_swiglu=True): each tile_n block of C holds tile_n/2
// gate columns then the matching tile_n/2 up columns, and the drain writes
// SiLU(gate) * up as a tile_n/2-wide bf16 tile. Compile with -I pointing at
// matrix_multiplication/bf16_in_fp32_out/.
//
//===----------------------------------------------------------------------===//

#include "mm_aie2p.cc"

extern "C" {

// C is (DIM_N/8, DIM_M/8, 8, 8): each 8-column block is DIM_M*8 contiguous
// floats, so gate block jb pairs with up block jb + NB/2 and the output block
// jb has the same layout at half the width. g and u are narrowed to bf16 with
// conv_even (as the f32_to_bf16_mn drain does), then the SiLU math is
// silu_and_mul_bf16's under FLOOR, the rounding a freshly loaded core runs that
// kernel with; together that keeps the output bit-identical to the
// GEMM-drain -> silu_and_mul launch pair it replaces.
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
