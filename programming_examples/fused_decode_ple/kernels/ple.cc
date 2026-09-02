// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Per-layer-embedding (PLE) kernels for Gemma4 -- the feature this builder is
// named for. Three stages run per decoder layer:
//
//   proj_layer_embedding  1536 -> 256   from the token EMBEDDING (not the
//                                       running hidden state), + norm, + the
//                                       token-embedding residual, + 2 scales
//   gate_layer_embedding  1536 -> 256   + GELU, from the post-FFN residual
//   per_layer_up           256 -> 1536  + norm, + residual, + layer scale
//
// Ported from FastFlowLM's decoding/kernels/{proj_layer_embedding,
// gate_layer_embedding,per_layer_up}.cc and bf16_proj.h. The arithmetic is
// theirs; the packaging is this repo's -- these are AIR-friendly pure-compute
// entry points with no in-kernel locks (AIR owns sync), following proj_qmm.cc.
//
// Every entry point here is validated against the numpy reference in
// programming_examples/llms/gemma4_e2b_q4nx/, which is itself gated per layer
// against FastFlowLM's own golden activations.
#include "aie_array_layout.h"
#include "aie_kernel_utils.h"
#include "lut_based_ops.h"

// ---------------------------------------------------------------------------
// bf16 matrix-vector, one weight block at a time.
//
// A block is BF16_PROJ_M_BLOCK (32) outputs x BF16_PROJ_K_BLOCK (256) inputs,
// laid out [in][out]: 32 contiguous outputs per input element. That is exactly
// how the Q4NX bundle already stores the PLE matrices -- inp_gate is
// (8,1536,32) = [out_tile][in][out_in_tile], and block (i,j) lands at flat
// offset (i*K/256 + j)*8192 because 1536*32 == 6*8192. So the bundle's raw
// bytes ARE the block stream in the order this kernel consumes them, and the
// builder streams them verbatim. Do not "de-tile and repack" -- that round trip
// is what the loader does for the CPU reference, and it is not wanted here.
// ---------------------------------------------------------------------------
template <int M, int K>
void mvm_blk(float *acc, bf16 *w, bf16 *x) {
  aie::vector<float, M> acc_vec = aie::load_v<M>(acc);
  aie::accum<accfloat, M> a;
  a.from_vector(acc_vec);

  constexpr int XV = 32;
  bf16 *wp = w;
  for (int i = 0; i < K / XV; i++) {
    aie::vector<bf16, XV> xv = aie::load_v<XV>(x + i * XV);
    for (int j = 0; j < XV; j++) {
      aie::vector<bf16, M> wcol = aie::load_v<M>(wp);
      a = aie::mac(a, wcol, xv[j]);
      wp += M;
    }
  }
  aie::store_v(acc, a.template to_vector<float>());
}

// RMSNorm y = x * rsqrt(mean(x^2) + eps) * w.
//
// Uses the same fast inverse square root FastFlowLM's kernels use (magic
// constant + two Newton steps) rather than this repo's rms_residual.cc path,
// so the PLE branch stays bit-comparable with the reference implementation it
// was ported from. Two Newton iterations land well inside bf16 resolution.
template <int N>
void rmsnorm(bf16 *y, bf16 *x, bf16 *w) {
  constexpr int V = 16;
  constexpr float one_over_N = 1.0f / (float)N;
  const float epsilon = 1e-6f;

  aie::accum<accfloat, V> sum_squares = aie::zeros<accfloat>();
  bf16 *it = x;
  for (int i = 0; i < N / V; i++) {
    auto xv = aie::load_v<V>(it);
    sum_squares = aie::mac_square(sum_squares, xv);
    it += V;
  }
  float sum = aie::reduce_add(sum_squares.template to_vector<float>());
  sum = sum * one_over_N + epsilon;

  float x2 = sum * 0.5f;
  float r = sum;
  uint32_t bits = *(uint32_t *)&r;
  bits = 0x5f3759df - (bits >> 1);
  r = *(float *)&bits;
  r = r * (1.5f - (x2 * r * r));
  r = r * (1.5f - (x2 * r * r));

  bf16 *ix = x;
  bf16 *iy = y;
  bf16 *iw = w;
  for (int i = 0; i < N / V; i++) {
    aie::vector<bf16, V> xv = aie::load_v<V>(ix);
    aie::vector<bf16, V> wv = aie::load_v<V>(iw);
    auto wx = aie::mul(xv, wv).template to_vector<float>();
    aie::store_v(iy, aie::mul(wx, r).template to_vector<bf16>());
    ix += V;
    iy += V;
    iw += V;
  }
}

template <int N>
void scale_inplace(bf16 *y, const bf16 s) {
  for (int i = 0; i < N; i += 16) {
    aie::vector<bf16, 16> v = aie::load_v<16>(y + i);
    aie::store_v(
        y + i,
        aie::mul(v, aie::broadcast<bf16, 16>(s)).template to_vector<bf16>());
  }
}

template <int N>
void add_inplace(bf16 *y, bf16 *b) {
  for (int i = 0; i < N; i += 16) {
    aie::vector<bf16, 16> yv = aie::load_v<16>(y + i);
    aie::vector<bf16, 16> bv = aie::load_v<16>(b + i);
    aie::store_v(y + i, aie::add(yv, bv));
  }
}

template <int N>
void mul_inplace(bf16 *y, bf16 *b) {
  for (int i = 0; i < N; i += 16) {
    aie::vector<bf16, 16> yv = aie::load_v<16>(y + i);
    aie::vector<bf16, 16> bv = aie::load_v<16>(b + i);
    aie::store_v(y + i, aie::mul(yv, bv).template to_vector<bf16>());
  }
}

extern "C" {

// ---- block-streamed bf16 projection (proj_qmm.cc zero/acc/flush pattern) ----

// Clear the 32-wide f32 accumulator before an output tile's K sweep.
void ple_zero(float *restrict acc) {
  aie::store_v(acc, aie::zeros<float, BF16_PROJ_M_BLOCK>());
}

// acc += w_block . x_slice, for one (out_tile, in_block) pair.
void ple_mac(float *restrict acc, bf16 *restrict w, bf16 *restrict x) {
  aie_round_nearest_even();
  mvm_blk<BF16_PROJ_M_BLOCK, BF16_PROJ_K_BLOCK>(acc, w, x);
}

// Write one finished output tile (32 values) as bf16.
void ple_flush(bf16 *restrict y, float *restrict acc) {
  aie::vector<float, BF16_PROJ_M_BLOCK> v = aie::load_v<BF16_PROJ_M_BLOCK>(acc);
  aie::accum<accfloat, BF16_PROJ_M_BLOCK> a;
  a.from_vector(v);
  aie::store_v(y, a.template to_vector<bf16>());
}

// ---- stage tails (everything after the matmul, per stage) ----

// proj_layer_embedding tail. x_proj holds the raw 1536->256 projection of the
// token embedding; emb is this layer's slice of the per-layer token-embedding
// table; norm_w is model.per_layer_proj_norm.weight.
//
// The two scale constants have names that are swapped relative to what they do
// (FastFlowLM's naming, kept so their kernels port unchanged). The ORDER below
// is the correct one: 1536**-0.5 first, on the model projection; 2**-0.5 last,
// after the embedding residual.
void ple_proj_tail(bf16 *restrict x_proj, bf16 *restrict emb,
                   bf16 *restrict norm_w) {
  scale_inplace<PLI_D>(x_proj, (bf16)PER_LAYER_INPUT_SCALE);
  rmsnorm<PLI_D>(x_proj, x_proj, norm_w);
  add_inplace<PLI_D>(x_proj, emb);
  scale_inplace<PLI_D>(x_proj, (bf16)PER_LAYER_MODEL_PROJECTION_SCALE);
}

// gate_layer_embedding tail: GELU over the 256-wide gate. A_FUNC is A_GELU for
// this model, so getActivationBf16 is the GELU LUT (see lut_based_ops.h).
void ple_gate_act(bf16 *restrict g, int _arm) {
  (void)_arm; // arm-gate operand, kept alive so AIR emits the arm lock
  aie_round_nearest_even();
  for (int i = 0; i < PLI_D; i += 16) {
    aie::vector<bf16, 16> v = aie::load_v<16>(g + i);
    // getActivationBf16 returns the raw v16bfloat16; bind it to a named
    // aie::vector so the implicit conversion happens before store_v.
    aie::vector<bf16, 16> act = getActivationBf16(v);
    aie::store_v(g + i, act);
  }
}

// pli *= gate, before the 256->1536 up projection.
void ple_apply_gate(bf16 *restrict pli, bf16 *restrict gate) {
  mul_inplace<PLI_D>(pli, gate);
}

// per_layer_up tail: y = (residual + rmsnorm(y, w)) * layer_output_scale.
// layer_scale is a 1-element buffer (the bundle stores it as shape (1,)).
void ple_up_tail(bf16 *restrict y, bf16 *restrict residual, bf16 *restrict w,
                 bf16 *restrict layer_scale, int _arm) {
  (void)_arm;
  aie_round_nearest_even();
  rmsnorm<MODEL_DIM>(y, y, w);
  add_inplace<MODEL_DIM>(y, residual);
  scale_inplace<MODEL_DIM>(y, *layer_scale);
}

} // extern "C"
