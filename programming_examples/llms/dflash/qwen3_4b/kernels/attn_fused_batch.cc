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

#ifdef DFLASH_ATTENTION_PROFILE
// Diagnostic only. Each core owns its globals; two query waves reset them
// independently. Unsigned low-word subtraction handles counter wraparound.
static unsigned dflash_profile[32];
static inline unsigned dflash_clock() { return (unsigned)::get_cycles(); }
extern "C" __attribute__((noinline)) void dflash_profile_begin(int wave) {
  for (int i=0;i<32;++i) dflash_profile[i]=0;
  dflash_profile[6]=wave;
  dflash_profile[7]=dflash_clock();
}
extern "C" __attribute__((noinline)) void dflash_profile_end(bf16 *scratch) {
  dflash_profile[0]=dflash_clock()-dflash_profile[7];
  unsigned *out=reinterpret_cast<unsigned *>(scratch);
  for (int i=0;i<32;++i) out[i]=dflash_profile[i];
  for (int i=64;i<768;++i) scratch[i]=(bf16)0;
}
#endif

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
// bodies together. This is also a diagnostic for register-pressure-sensitive
// code generation at partial causal blocks; arithmetic is unchanged.
extern "C" __attribute__((noinline))
void dflash_fused_qk(bf16 *q, bf16 *k, bf16 *m, float *c, bf16 *s,
                     int block, int length, int first, int causal) {
#if defined(DFLASH_ATTENTION_ABLATION) && (DFLASH_ATTENTION_ABLATION & 1)
  aie_round_nearest_even();
  // Supply diagnostic only: consume the same Q/K arrays and write every
  // output slab. These synthetic scores are NOT model attention.
  aie::vector<bf16,64> sum=aie::zeros<bf16,64>();
  for(int i=0;i<4096;i+=64) sum=aie::add(sum,aie::add(aie::load_v<64>(q+i),aie::load_v<64>(k+i)));
  aie::store_v(m,sum);
  for(int i=0;i<32;i+=16) aie::store_v(c+i,aie::broadcast<float,16>(1));
  for(int i=0;i<768;i+=64) aie::store_v(s+i,aie::broadcast<bf16,64>((bf16).0625));
  for(int row=0;row<4;++row)
    aie::store_v(reinterpret_cast<float *>(s+row*192+128),aie::broadcast<float,8>(1));
  return;
#endif
#ifdef DFLASH_ATTENTION_PROFILE
  unsigned start=dflash_clock();
#endif
  attn_qk_blk_tile(q,k,m,c,s,block,length,first,causal);
#ifdef DFLASH_ATTENTION_PROFILE
  dflash_profile[1]+=dflash_clock()-start;
  ++dflash_profile[3];
#endif
}
extern "C" __attribute__((noinline))
void dflash_fused_pv(bf16 *s, bf16 *v, float *y, float *l,
                     int block, int length, int causal) {
#if defined(DFLASH_ATTENTION_ABLATION) && (DFLASH_ATTENTION_ABLATION & 2)
  aie_round_nearest_even();
  aie::vector<bf16,64> sum=aie::zeros<bf16,64>();
  for(int i=0;i<4096;i+=64) sum=aie::add(sum,aie::load_v<64>(v+i));
  // QK does not define score padding or rows beyond the causal boundary.
  // Consume only the initialized probability elements, as production PV does.
  for(int row=0;row<4;++row)
    if(length+row*causal>block*16)
      for(int i=0;i<128;i+=64) sum=aie::add(sum,aie::load_v<64>(s+row*192+i));
  auto f=aie::mul(sum,(bf16)1).template to_vector<float>();
  for(int i=0;i<4096;i+=64) aie::store_v(y+i,f);
  for(int i=0;i<64;i+=16) aie::store_v(l+i,aie::broadcast<float,16>(1));
  return;
#endif
#ifdef DFLASH_ATTENTION_PROFILE
  unsigned start=dflash_clock();
#endif
  attn_kv_blk_tile(s,v,y,l,block,length,causal);
#ifdef DFLASH_ATTENTION_PROFILE
  dflash_profile[2]+=dflash_clock()-start;
  ++dflash_profile[4];
#endif
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
