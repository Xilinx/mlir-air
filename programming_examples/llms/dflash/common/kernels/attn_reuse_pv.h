// Two independent query pairs consume each V operand before it is discarded.
// Each pair retains two K8 MACs, a separate FP32 rescale and a separate add.
#pragma once
static inline __attribute__((always_inline)) void
dflash_pv_reuse_v(bf16 *scores, bf16 *values, float *output, float *denominator,
                  int block) {
  static_assert(ATTN_QUERY_TILE == 4);
  using MM = aie::mmul<8, 8, 8, bf16, bf16, accfloat>;
  constexpr int width = Q_HEADS_PADDED_PER_CU * DH;
  for (int local = 0; local < 4; ++local) {
    if (block == 0) {
      zero_vectorized<float, width>(output + local * width);
      aie::store_v(denominator + local * 16, aie::zeros<float, 16>());
    }
    calculate_l(scores + local * SSZ_BLK,
                reinterpret_cast<float *>(scores + local * SSZ_BLK + 128),
                denominator + local * 16);
  }
#pragma clang loop unroll(disable)
  for (int group = 0; group < 2; ++group) {
    // When no running max in the group moved, every correction is exactly 1
    // and the old output seeds the mac instead of being rescaled. The
    // corrections are FP32; compare their bits with 1.0f.
    uint32_t off_unit = 0;
    for (int local = 0; local < 4; ++local) {
      const uint32_t *c =
          reinterpret_cast<const uint32_t *>(scores + local * SSZ_BLK + 128) +
          group * 4;
      for (int h = 0; h < 4; ++h)
        off_unit |= c[h] ^ 0x3f800000u;
    }
    const bool unit = off_unit == 0;
    auto s0 = aie::load_v<64>(scores + group * 64);
    auto s1 = aie::load_v<64>(scores + SSZ_BLK + group * 64);
    auto s2 = aie::load_v<64>(scores + 2 * SSZ_BLK + group * 64);
    auto s3 = aie::load_v<64>(scores + 3 * SSZ_BLK + group * 64);
    auto a0 = aie::concat(aie::filter_even(s0, 8), aie::filter_even(s1, 8));
    auto a1 = aie::concat(aie::filter_odd(s0, 8), aie::filter_odd(s1, 8));
    auto b0 = aie::concat(aie::filter_even(s2, 8), aie::filter_even(s3, 8));
    auto b1 = aie::concat(aie::filter_odd(s2, 8), aie::filter_odd(s3, 8));
    if (unit) {
#pragma clang loop unroll(disable)
      for (int d = 0; d < DH / 8; ++d) {
        bf16 *vp = values + (group * (DH / 8) + d) * 64;
        auto v0 = aie::load_v<64>(vp);
        auto v1 = aie::load_v<64>(vp + KV_HEADS_PER_CU * DH * 8);
        float *y0 = output + d * 64 + group * 32;
        float *y1 = output + width + d * 64 + group * 32;
        float *y2 = output + 2 * width + d * 64 + group * 32;
        float *y3 = output + 3 * width + d * 64 + group * 32;
        MM pa(aie::concat(aie::load_v<32>(y0), aie::load_v<32>(y1)));
        MM pb(aie::concat(aie::load_v<32>(y2), aie::load_v<32>(y3)));
        pa.mac(a0, v0);
        pb.mac(b0, v0);
        pa.mac(a1, v1);
        pb.mac(b1, v1);
        auto ra = pa.template to_vector<float>();
        auto rb = pb.template to_vector<float>();
        aie::store_v(y0, ra.template extract<32>(0));
        aie::store_v(y1, ra.template extract<32>(1));
        aie::store_v(y2, rb.template extract<32>(0));
        aie::store_v(y3, rb.template extract<32>(1));
      }
    } else {
#pragma clang loop unroll(disable)
    for (int d = 0; d < DH / 8; ++d) {
      aie::vector<float, 64> ca, cb;
      {
        bf16 *vp = values + (group * (DH / 8) + d) * 64;
        auto v0 = aie::load_v<64>(vp);
        auto v1 = aie::load_v<64>(vp + KV_HEADS_PER_CU * DH * 8);
        MM pa(aie::zeros<bf16, 64>()), pb(aie::zeros<bf16, 64>());
        pa.mac(a0, v0);
        pb.mac(b0, v0);
        pa.mac(a1, v1);
        pb.mac(b1, v1);
        ca = pa.template to_vector<float>();
        cb = pb.template to_vector<float>();
      }
#pragma clang loop unroll(full)
      for (int pair = 0; pair < 2; ++pair) {
        const int local = pair * 2;
        float *y0 = output + local * width + d * 64 + group * 32;
        float *y1 = output + (local + 1) * width + d * 64 + group * 32;
        float *c0 = reinterpret_cast<float *>(scores + local * SSZ_BLK + 128) +
                    group * 4;
        float *c1 =
            reinterpret_cast<float *>(scores + (local + 1) * SSZ_BLK + 128) +
            group * 4;
        auto old = aie::concat(aie::load_v<32>(y0), aie::load_v<32>(y1));
        auto correction = aie::concat(
            aie::broadcast<float, 8>(c0[0]), aie::broadcast<float, 8>(c0[1]),
            aie::broadcast<float, 8>(c0[2]), aie::broadcast<float, 8>(c0[3]),
            aie::broadcast<float, 8>(c1[0]), aie::broadcast<float, 8>(c1[1]),
            aie::broadcast<float, 8>(c1[2]), aie::broadcast<float, 8>(c1[3]));
        auto scaled = aie::mul(correction, old).template to_vector<float>();
        auto sum = aie::add(pair == 0 ? ca : cb, scaled);
        aie::store_v(y0, sum.template extract<32>(0));
        aie::store_v(y1, sum.template extract<32>(1));
      }
    }
    }
  }
}
