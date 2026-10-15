// Optional same-arithmetic correction lookup across independent query/heads.
#pragma once
static inline __attribute__((always_inline))
void dflash_qk_tile_packed_correction(bf16 *queries, bf16 *keys,
    bf16 *maximum, float *correction, bf16 *scores, int block, int length,
    int first_row, int causal) {
  aie_round_nearest_even();
  static_assert(ATTN_QUERY_TILE == 2 || ATTN_QUERY_TILE == 4);
  alignas(aie::vector_decl_align) bf16 deltas[ATTN_QUERY_TILE * 8];
  for (int i=0;i<ATTN_QUERY_TILE*8;i+=16)
    aie::store_v(deltas+i,aie::zeros<bf16,16>());
  #pragma clang loop unroll(disable)
  for (int local=0;local<ATTN_QUERY_TILE;local+=2)
    if (length+(local+1)*causal-block*16>0)
      dflash_qk_pair_raw(queries,keys,scores,first_row+local,local);
  const auto neg_inf=aie::broadcast<bf16,16>(-0x1.FEp127f);
  const aie::vector<int,16> positions(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16);
  const auto lo=aie::broadcast<bf16,16>(-87.0f);
  const auto hi=aie::broadcast<bf16,16>(88.0f);
  #pragma clang loop unroll(disable)
  for (int local=0;local<ATTN_QUERY_TILE;++local) {
    bf16 *m=maximum+local*16;
    bf16 *s=scores+local*SSZ_BLK;
    if (block==0) aie::store_v(m,neg_inf);
    int valid=length+local*causal-block*16;
    if (valid<=0) continue;
    valid=valid<16?valid:16;
    auto mask=aie::le(positions,valid);
    #pragma clang loop unroll(disable)
    for (int head=0;head<8;++head) {
      auto raw=aie::load_v<16>(s+head*16);
      auto masked=aie::select((bf16)(-0x1.FEp127f),raw,mask);
      bf16 vm=aie::reduce_max(masked);
      bf16 vmax=aie::max(vm,(bf16)m[head]);
      auto difference=aie::sub(raw,vmax);
      auto clamped=aie::clamp(difference,lo,hi);
      aie::accum<accfloat,16> exponential=getExpBf16(clamped);
      auto probability=exponential.template to_vector<bf16>(0);
      probability=aie::select((bf16)0,probability,mask);
      deltas[local*8+head]=aie::sub((bf16)m[head],vmax);
      m[head]=(float)vmax;
      aie::store_v(s+head*16,probability);
    }
  }
  alignas(aie::vector_decl_align) float corrections[ATTN_QUERY_TILE*8];
  #pragma clang loop unroll(disable)
  for (int i=0;i<ATTN_QUERY_TILE*8;i+=16) {
    auto delta=aie::clamp(aie::load_v<16>(deltas+i),lo,hi);
    aie::accum<accfloat,16> exponential=getExpBf16(delta);
    aie::store_v(corrections+i,exponential.template to_vector<float>());
  }
  #pragma clang loop unroll(disable)
  for (int local=0;local<ATTN_QUERY_TILE;++local) {
    if (length+local*causal-block*16<=0) continue;
    auto c=aie::load_v<8>(corrections+local*8);
    aie::store_v(correction+local*8,c);
    aie::store_v(reinterpret_cast<float *>(scores+local*SSZ_BLK+128),c);
  }
}
