///@file phi4-4b.h
///@brief Define the model parameters for the phi4-4b model
#ifndef __PHI4_4B_H__
#define __PHI4_4B_H__

#if MODEL_TYPE == PHI4_4B
constexpr int MODEL_DIM = 3072;
constexpr int NUM_ATTN_HEADS = 24;
constexpr int NUM_KV_HEADS = 8;
constexpr int INTERMEDIATE_SIZE = 8192;
constexpr int GLU_SLICE = 1024;
constexpr int DH = 128;
constexpr int VOCAB_SIZE = 200064;
constexpr float ATTN_SCALE = 0.08838834764831843f;

#define ATTN_IMPL ATTN_IMPL_2x4x1
#define A_FUNC A_SILU

constexpr int Q4NX_ROW_BLOCK_SIZE = 32;
constexpr int Q4NX_COL_BLOCK_SIZE = 256;
constexpr int Q4NX_GROUP_SIZE = 32;
#endif

#endif // __LLAMA3_2_3B_H__