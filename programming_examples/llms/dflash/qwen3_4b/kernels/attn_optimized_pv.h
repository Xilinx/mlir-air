// Optional PV update: retain separate FP32 multiply/add rounding, remove the
// intermediate rescaled-output store/reload. Partial masks use the old path.
#pragma once
static inline __attribute__((always_inline))
void dflash_pv_full_tile_fused_rescale(bf16 *scores,bf16 *values,
    float *output,float *denominator,int block) {
  aie_round_nearest_even();
  using MM=aie::mmul<8,8,8,bf16,bf16,accfloat>;
  constexpr int width=Q_HEADS_PADDED_PER_CU*DH;
  #pragma clang loop unroll(disable)
  for(int local=0;local<ATTN_QUERY_TILE;++local) {
    float *y=output+local*width;
    float *den=denominator+local*16;
    bf16 *s=scores+local*SSZ_BLK;
    float *c=reinterpret_cast<float *>(s+128);
    if(block==0) {
      zero_vectorized<float,width>(y);
      aie::store_v(den,aie::zeros<float,16>());
    }
    calculate_l(s,c,den);
  }
  #pragma clang loop unroll(disable)
  for(int local=0;local<ATTN_QUERY_TILE;local+=2) {
    #pragma clang loop unroll(disable)
    for(int group=0;group<2;++group) {
      auto q0=aie::load_v<64>(scores+local*SSZ_BLK+group*64);
      auto q1=aie::load_v<64>(scores+(local+1)*SSZ_BLK+group*64);
      auto first=aie::concat(aie::filter_even(q0,8),aie::filter_even(q1,8));
      auto second=aie::concat(aie::filter_odd(q0,8),aie::filter_odd(q1,8));
      float *c0=reinterpret_cast<float *>(scores+local*SSZ_BLK+128)+group*4;
      float *c1=reinterpret_cast<float *>(scores+(local+1)*SSZ_BLK+128)+group*4;
      #pragma clang loop unroll(disable)
      for(int d=0;d<DH/8;++d) {
        bf16 *v=values+(group*(DH/8)+d)*64;
        aie::vector<float,64> contribution;
        {
          MM product(aie::zeros<bf16,64>());
          product.mac(first,aie::load_v<64>(v));
          product.mac(second,aie::load_v<64>(v+KV_HEADS_PER_CU*DH*8));
          contribution=product.template to_vector<float>();
        }
        float *y0=output+local*width+d*64+group*32;
        float *y1=output+(local+1)*width+d*64+group*32;
        auto previous=aie::concat(aie::load_v<32>(y0),aie::load_v<32>(y1));
        auto correction=aie::concat(
          aie::broadcast<float,8>(c0[0]),aie::broadcast<float,8>(c0[1]),
          aie::broadcast<float,8>(c0[2]),aie::broadcast<float,8>(c0[3]),
          aie::broadcast<float,8>(c1[0]),aie::broadcast<float,8>(c1[1]),
          aie::broadcast<float,8>(c1[2]),aie::broadcast<float,8>(c1[3]));
        auto scaled=aie::mul(correction,previous).template to_vector<float>();
        auto sum=aie::add(contribution,scaled);
        aie::store_v(y0,sum.template extract<32>(0));
        aie::store_v(y1,sum.template extract<32>(1));
      }
    }
  }
}
