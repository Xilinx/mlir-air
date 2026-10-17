// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// Compile the unchanged QK/PV primitives together so their shared LUT storage
// has one definition. The legacy lock-driven entry points are unused by AIR's
// block wrappers; give their overlapping lock constants distinct C++ names.
#include "attn_qk_batch.cc"
#define s_prod_lock pv_s_prod_lock
#define s_cons_lock pv_s_cons_lock
#define l_cons_lock pv_l_cons_lock
#include "attn_kv_batch.cc"
#undef l_cons_lock
#undef s_cons_lock
#undef s_prod_lock

extern "C" __attribute__((always_inline))
void dflash_copy_query(bf16 *out, bf16 *in) {
  for (int i=0;i<4096;i+=64)
    aie::store_v(out+i,aie::load_v<64>(in+i));
}
extern "C" __attribute__((always_inline))
void dflash_relay_key(bf16 *out, bf16 *in) {
  for (int i=0;i<8192;i+=64)
    aie::store_v(out+i,aie::load_v<64>(in+i));
}
extern "C" __attribute__((always_inline))
void dflash_relay_query(bf16 *out, bf16 *in) {
  for (int cu=0;cu<2;++cu)
    for (int row=0;row<4;++row)
      for (int d=0;d<16;++d)
        aie::store_v(out+cu*4096+row*1024+d*64,
                     aie::load_v<64>(in+row*2048+d*128+cu*64));
}

// Keep the combined frame-loop caller from inlining both large attention
// bodies together: partial causal blocks are sensitive to register pressure.
extern "C" __attribute__((noinline))
void dflash_fused_qk(bf16 *q, bf16 *k, bf16 *m, float *c, bf16 *s,
                     int block, int length, int first, int causal) {
  attn_qk_blk_tile(q,k,m,c,s,block,length,first,causal);
}
extern "C" __attribute__((noinline))
void dflash_fused_pv(bf16 *s, bf16 *v, float *y, float *l,
                     int block, int length, int causal) {
  attn_kv_blk_tile(s,v,y,l,block,length,causal);
}

// Scores are dead after final attention normalization. Reuse their 768-word
// slab for the cascade join; 512 payload words plus zero padding per frame.
// This avoids the current cascade lowering's nonzero-offset limitation and
// does not borrow a DMA-owned input buffer.
extern "C" __attribute__((always_inline))
void dflash_join_send(bf16 *stage, bf16 *output, int chunk) {
  for (int i=0;i<512;i+=64)
    aie::store_v(stage+i,aie::load_v<64>(output+4096+chunk*512+i));
  for (int i=512;i<768;i+=64)
    aie::store_v(stage+i,aie::zeros<bf16,64>());
}
extern "C" __attribute__((always_inline))
void dflash_join_receive(bf16 *output, bf16 *stage, int chunk) {
  for (int i=0;i<512;i+=64)
    aie::store_v(output+4096+chunk*512+i,aie::load_v<64>(stage+i));
}
