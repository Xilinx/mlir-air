// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
///@file lfm2-1.2b.h
///@brief Define the model parameters for the LFM2-1.2B model
///
/// LFM2-1.2B is a HYBRID conv-attention decoder: of its 16 layers, only 6 are
/// attention (indices 2, 5, 8, 10, 12, 14) and the other 10 are a gated causal
/// depthwise convolution ("ShortConv"). The two layer types share an identical
/// outer shape -- norm -> big projection -> mixer -> out_proj(2048x2048) ->
/// residual -> ffn_norm -> SwiGLU -> residual -- and differ only in the mixer
/// and the width of the big projection (QKV 3072 vs conv in_proj 6144).
///
/// The ATTENTION half is spec-identical to llama3.2-1b (same MODEL_DIM, head
/// counts, DH=64 and even the same 8192 intermediate), so it reuses that
/// topology unchanged. The two deltas are per-head QK-norm (as qwen3/gemma3)
/// and the Q4_0 codec (as qwen2.5-3b). Both already have precedent here.
#ifndef __LFM2_1_2B_H__
#define __LFM2_1_2B_H__

#if MODEL_TYPE == LFM2_1_2B
constexpr int MODEL_DIM = 2048;
constexpr int NUM_ATTN_HEADS = 32;
constexpr int NUM_KV_HEADS = 8;
constexpr int INTERMEDIATE_SIZE = 8192;
constexpr int GLU_SLICE = 1024;
constexpr int DH = 64;
constexpr int VOCAB_SIZE = 65536;
constexpr float ATTN_SCALE = 0.125f; // 1/sqrt(64)

// HF `norm_eps`. Unlike every other model here this value is load-bearing:
// LFM2's embeddings are small (||e|| ~ 0.6 over 2048 channels, so
// mean(e^2) ~ 1.8e-4), which puts eps within an order of magnitude of the term
// it guards. At the default 1e-6 layer 0's normalizer comes out 2-5% high --
// measured as a matching 2-11% scale error on the ShortConv state, and 0.89
// logits cosine on a token whose embedding is small. See verify_conv.py --seq.
#define RMS_NORM_EPS 1e-5f

#define ATTN_IMPL ATTN_IMPL_2x4x1
#define A_FUNC A_SILU

// Per-head RMSNorm on Q and K (HF: q_layernorm / k_layernorm, over DH).
#define HAS_QK_NORM

// LFM2 weights are symmetric signed-int4 (Q4_0, w = q*scale, no min term),
// the same on-device codec qwen2.5-3b uses.
#define Q4_0

// ---- ShortConv (the 10 non-attention layers) ----
// h = B * v;  h = causal_depthwise_conv1d(h, W, k=3);  y = C * h
// where [B | C | v] = in_proj(x), v LAST. Taps are oldest-first, and causality
// is expressed as a PRE-PAD rather than a mask, so prefill and decode are the
// same kernel with a different pad. The taps are bf16, NOT quantized -- only
// 2048*3 values, which the reference design also keeps full precision.
constexpr int CONV_DIM = 2048;
constexpr int CONV_L_CACHE = 3;                // kernel size
constexpr int CONV_IN_PROJ_OUT = 3 * CONV_DIM; // 6144 = B | C | v
// Layers 2, 5, 8, 10, 12, 14 are attention; the rest are ShortConv. The
// schedule is IRREGULAR (gaps of 2,2,1,1,1,0 conv layers) -- drive the layer
// loop off this list, never a modulo.
constexpr int NUM_LAYERS = 16;
constexpr int NUM_ATTN_LAYERS = 6;
constexpr int NUM_CONV_LAYERS = 10;

constexpr int Q4NX_ROW_BLOCK_SIZE = 32;
constexpr int Q4NX_COL_BLOCK_SIZE = 256;
constexpr int Q4NX_GROUP_SIZE = 32;
#endif

#endif // __LFM2_1_2B_H__
