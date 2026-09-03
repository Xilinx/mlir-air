// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
///@file gemma4-e2b.h
///@brief Define the model parameters for the gemma4-e2b (text) model
#ifndef __GEMMA4_E2B_H__
#define __GEMMA4_E2B_H__

#if MODEL_TYPE == GEMMA4_E2B
constexpr int MODEL_DIM = 1536;
constexpr int NUM_ATTN_HEADS = 8;
// TWO, not the model's one. Gemma4 is MQA with a single kv head, but this build
// REPLICATES it to each of two attention CUs (see the ATTN_IMPL note below and
// the gemma4-e2b entry in fused_decode_ple.py), and the packer tiles the k/v
// rows to match. NUM_KV_HEADS is what model_spec.h derives DK and DV from, so
// leaving it at the model's 1 makes the kernel slice
//   k = qkv[4096:4608]   v = qkv[4608:5120]
// out of a buffer laid out q(4096) | k(1024) | v(1024). The kernel's "v" is
// then the replicated SECOND COPY OF K, and unnormalized -- measured in the
// device's KV cache as two blocks both at cos 0.9992 to k and 1000x apart in
// scale.
constexpr int NUM_KV_HEADS = 2;
// DELIBERATELY NOT FLM's VALUE. FastFlowLM sets this to the narrow 6144 and
// switches the wide layers on with a DOUBLE_WIDE_MLP define; this builder
// instead pads every layer up to the wide shape (see the gemma4-e2b entry in
// fused_decode_ple.py), so the kernel only ever sees 12288. Revisit together
// with that staging decision, not on its own.
constexpr int INTERMEDIATE_SIZE = 12288;
constexpr int GLU_SLICE = 1024;
// TWO head dims in one model: full-attention layers (4,9,...,34) use 512,
// the sliding-window layers use 256. DH is the wide one for the same
// pad-to-max reason as INTERMEDIATE_SIZE; SWA_DH is what the sliding path
// actually rotates and attends over.
constexpr int DH = 512;
constexpr int SWA_DH = 256;
constexpr int SLIDING_WINDOW = 512;
constexpr int VOCAB_SIZE = 262144;
// 1.0, NOT 1/sqrt(DH): Gemma4 folds query_pre_attn_scalar into the q weights.
// Verified against FLM's own golden activations (their ATTN_SCALE is also 1.0).
constexpr float ATTN_SCALE = 1.0f;
constexpr float SWA_ATTN_SCALE = 1.0f;
// Claim the model_spec.h exemption: 1.0 is NOT 1/sqrt(DH) here and that is
// deliberate, not a value copied from a neighbouring header. Gemma4 folds
// query_pre_attn_scalar into the q weights, so the kernel must not scale again.
// Confirmed two ways: FastFlowLM's own gemma4-e2b.h also sets 1.0, and the
// numpy reference reproduces their golden activations with ATTN_SCALE=1.0.
#define ATTN_SCALE_NOT_INV_SQRT_DH

// --- per-layer embeddings (PLE) ---
// Names match FastFlowLM's gemma4-e2b.h exactly so their kernels port over
// unmodified.
constexpr int PLI_D = 256; // PLI = Per Layer Input
// THESE TWO NAMES ARE SWAPPED relative to what they do, and the swap is FLM's.
// PER_LAYER_INPUT_SCALE is 1536**-0.5 and is applied FIRST, to the model
// projection; PER_LAYER_MODEL_PROJECTION_SCALE is 2**-0.5 and is applied LAST,
// after the token-embedding residual. The ORDER in proj_layer_embedding is what
// is correct -- do not "fix" the names into each other's slots.
constexpr float PER_LAYER_INPUT_SCALE = 0.02551551815399144f;
constexpr float PER_LAYER_MODEL_PROJECTION_SCALE = 0.7071067811865476f;

// bf16 (non-q4) projection tiling, used by the three PLE kernels. One weight
// block is M_BLOCK*K_BLOCK = 8192 bf16, laid out [in][out] with 32 contiguous
// outputs per input element.
constexpr int BF16_PROJ_M_BLOCK = 32;
constexpr int BF16_PROJ_K_BLOCK = 256;

// MQA: 1 real kv head, REPLICATED to each of 2 CUs, with the 8 q heads split
// 4/4. One CU would be arithmetically enough, but N_ATTN_CU=1 is the only
// setting that selects the builder's untested CU_PER_COL==1 attention path, and
// that path wedges the decode wave on NPU2 -- see the gemma4-e2b entry in
// fused_decode_ple.py for the measurement.
#define ATTN_IMPL ATTN_IMPL_1x4x1
#define A_FUNC A_GELU
// qk-norm: RMSNorm(DH) on Q,K per head before RoPE (rope_w carries q/k norm w).
#define HAS_QK_NORM
// Gemma4 keeps Gemma3's 4-norm sandwich and adds a FIFTH norm on the per-layer
// embedding branch (post_layernorm), applied to the PLE projection before its
// residual add.
#define HAS_PER_LAYER_EMBEDDING
constexpr int NUM_LAYERS = 35;
// Layers >= this index reuse a lower layer's KV cache and carry the double-wide
// FFN. The two switches are the same bit -- see the builder's LAYER_CLASS note.
constexpr int FIRST_KV_SHARED_LAYER = 15;

constexpr int Q4NX_ROW_BLOCK_SIZE = 32;
constexpr int Q4NX_COL_BLOCK_SIZE = 256;
constexpr int Q4NX_GROUP_SIZE = 32;
#endif

#endif // __GEMMA4_E2B_H__
