// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// The verify pass's kernels for 8 token rows, one layer per wave: the Q4_K
// array's attention and MoE projections (attn_b8.cc, moe_b8.cc), the
// residual cores, the planner's routing and the feeder's messages. Every
// message to the array cores is an H-sized message, x arriving as three of
// them, header int32 word 0 the layer's turns. Two residual cores own 4 of
// the 8 token rows each, as f32 tiles [256 dim tiles][4 tokens][8]; their
// phases share one code path, and a unit's first K (V) block is the RoPE
// table (a dummy), so each core's DMA channels repeat once per wave.

#include "moe_b8.cc"
#define CHUNK A_CHUNK
#define PAY A_PAY
#include "attn_b8.cc"
#undef CHUNK
#undef PAY

namespace {

constexpr int NTL = 256; // dim tiles
constexpr int RW4 = 32;  // a residual tile: 4 tokens x 8

} // namespace

extern "C" {

// moe_block into the MoE sum's buffer: o_proj's row blocks
void block_o(uint8_t *__restrict x, uint8_t *__restrict w,
             float *__restrict acc, uint8_t *__restrict sp, int32_t cb,
             int32_t a, int32_t off) {
  moe_block(x, w, acc, sp, cb, a, off);
}

// h += rows 4 c .. 4 c + 3 of a round's sum over its columns. A round:
// [column][core r][4 row tiles][8 tokens][8], dim tiles t0 + 4 r ..
inline void add_round(const bf16 *rnd, float *h, int t0, int c, int ncol) {
  for (int r = 0; r < 4; r++)
    for (int j = 0; j < 4; j++) {
      aie::accum<accfloat, 32> a;
      float *ht = h + (t0 + 4 * r + j) * RW4;
      a.from_vector(aie::load_v<32>(ht));
      for (int col = 0; col < ncol; col++)
        a = aie::add(
            a, aie::load_v<32>(rnd + (col * 4 + r) * 256 + j * 64 + c * 32));
      aie::store_v(ht, a.template to_vector<float>());
    }
}

// h += MoE row group rg (the columns' partial sums)
void res_moe(bf16 *__restrict rnd, float *__restrict h, int32_t rg, int32_t c) {
  add_round(rnd, h, 16 * rg, c, 4);
}

// h += o_proj round j (one row group per column)
void res_o(bf16 *__restrict rnd, float *__restrict h, int32_t j, int32_t c) {
  for (int col = 0; col < 4; col++)
    add_round(rnd + col * 4 * 256, h, 64 * j + 16 * col, c, 1);
}

} // extern "C"

namespace {
// 2^(-j / 256) in Q27, j = 0 .. 256: a sum of 8 fits int32
constexpr int32_t EXP2_Q27[257] = {
    134217728, 133854811, 133492875, 133131918, 132771936, 132412929, 132054891,
    131697822, 131341719, 130986578, 130632398, 130279175, 129926908, 129575593,
    129225227, 128875810, 128527337, 128179806, 127833215, 127487561, 127142842,
    126799055, 126456197, 126114267, 125773261, 125433177, 125094013, 124755766,
    124418433, 124082012, 123746502, 123411898, 123078199, 122745403, 122413506,
    122082507, 121752403, 121423191, 121094869, 120767436, 120440887, 120115222,
    119790437, 119466530, 119143500, 118821342, 118500056, 118179638, 117860087,
    117541400, 117223575, 116906609, 116590500, 116275246, 115960844, 115647293,
    115334589, 115022730, 114711715, 114401541, 114092206, 113783707, 113476042,
    113169210, 112863206, 112558031, 112253680, 111950152, 111647445, 111345557,
    111044485, 110744227, 110444781, 110146144, 109848315, 109551291, 109255071,
    108959651, 108665030, 108371206, 108078177, 107785939, 107494492, 107203833,
    106913960, 106624871, 106336563, 106049035, 105762284, 105476309, 105191107,
    104906676, 104623014, 104340120, 104057990, 103776623, 103496017, 103216169,
    102937078, 102658742, 102381159, 102104326, 101828242, 101552904, 101278310,
    101004459, 100731349, 100458977, 100187342, 99916441,  99646272,  99376834,
    99108125,  98840142,  98572884,  98306348,  98040534,  97775437,  97511058,
    97247394,  96984442,  96722202,  96460670,  96199846,  95939727,  95680311,
    95421597,  95163583,  94906266,  94649644,  94393717,  94138482,  93883937,
    93630080,  93376909,  93124423,  92872620,  92621498,  92371054,  92121288,
    91872197,  91623780,  91376035,  91128959,  90882551,  90636810,  90391733,
    90147319,  89903566,  89660471,  89418035,  89176253,  88935126,  88694650,
    88454825,  88215648,  87977118,  87739232,  87501990,  87265390,  87029429,
    86794107,  86559420,  86325368,  86091949,  85859162,  85627003,  85395473,
    85164568,  84934288,  84704630,  84475594,  84247176,  84019377,  83792193,
    83565624,  83339667,  83114321,  82889585,  82665456,  82441933,  82219015,
    81996699,  81774985,  81553870,  81333353,  81113432,  80894106,  80675373,
    80457231,  80239679,  80022715,  79806339,  79590547,  79375338,  79160712,
    78946666,  78733198,  78520308,  78307994,  78096253,  77885085,  77674488,
    77464461,  77255001,  77046108,  76837780,  76630015,  76422812,  76216169,
    76010084,  75804557,  75599586,  75395169,  75191305,  74987992,  74785228,
    74583013,  74381345,  74180222,  73979643,  73779606,  73580110,  73381154,
    73182735,  72984853,  72787506,  72590693,  72394412,  72198661,  72003440,
    71808747,  71614580,  71420938,  71227820,  71035224,  70843149,  70651593,
    70460555,  70270033,  70080027,  69890535,  69701555,  69513086,  69325126,
    69137675,  68950730,  68764292,  68578357,  68392925,  68207994,  68023564,
    67839632,  67656197,  67473258,  67290815,  67108864};
constexpr int NCB_X = 8;    // x's chunks
constexpr int PER_PART = 3; // chunks per x part
} // namespace

extern "C" {

// x part p of a message into xb; cnt[0] = the layer's turns
void mx_part(uint8_t *__restrict msg, uint8_t *__restrict xb,
             int32_t *__restrict cnt, int32_t p) {
  cnt[0] = ((const int32_t *)msg)[0];
  const int n = (p + 1) * PER_PART > NCB_X ? NCB_X - p * PER_PART : PER_PART;
  const uint8_t *s = msg + HDR;
  uint8_t *d = xb + p * PER_PART * CHUNK;
  for (int i = 0; i < n * CHUNK; i += 64)
    aie::store_v(d + i, aie::load_v<64>(s + i));
}

// the plan's turns (int32 word 0)
void fd_turns(uint8_t *__restrict plan, int32_t *__restrict cnt) {
  cnt[0] = ((const int32_t *)plan)[0];
}

// x part p into each of the 4 H messages, word 0 of each header the turns
void fd_xpart(uint8_t *__restrict hm, uint8_t *__restrict xt,
              uint8_t *__restrict plan, int32_t p) {
  const int n = (p + 1) * PER_PART > NCB_X ? NCB_X - p * PER_PART : PER_PART;
  const uint8_t *s = xt + p * PER_PART * CHUNK;
  for (int m = 0; m < 4; m++) {
    uint8_t *d = hm + m * HMSG;
    ((int32_t *)d)[0] = ((const int32_t *)plan)[0];
    for (int i = 0; i < n * CHUNK; i += 64)
      aie::store_v(d + HDR + i, aie::load_v<64>(s + i));
  }
}

// ---- the router, on the planner core. x: [D / 8][8 rows][8] bf16; a block
// of 16 router columns: [2 column tiles][16 expert tiles][8 columns][8
// experts]; lg: [16 expert tiles][8 rows][8] f32

// piece q of x ([64 column tiles][8 rows][8] bf16) as bfp16 tiles into xt
void x2_tiles(bf16 *__restrict x, uint8_t *__restrict xt, int32_t q) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  OS os((bfp16ebs8 *)(xt + q * 64 * BLK));
  for (int j = 0; j < 64; j++) {
    ACC a;
    a.from_vector(aie::load_v<64>(x + j * 64));
    os << a.template to_vector<bfp16ebs8>();
  }
}

void lg_zero(float *__restrict lg) {
  for (int i = 0; i < 1024; i += 32)
    aie::store_v(lg + i, aie::zeros<float, 32>());
}

void lg_block(bf16 *__restrict x, bf16 *__restrict w, float *__restrict lg,
              int32_t cb) {
  using MM = aie::mmul<8, 8, 8, bf16, bf16, accfloat>;
  const auto a0 = aie::load_v<64>(x + 2 * cb * 64);
  const auto a1 = aie::load_v<64>(x + (2 * cb + 1) * 64);
  for (int g = 0; g < 16; g++) {
    MM m(aie::load_v<64>(lg + g * 64));
    m.mac(a0, aie::load_v<64>(w + g * 64));
    m.mac(a1, aie::load_v<64>(w + 1024 + g * 64));
    aie::store_v(lg + g * 64, m.template to_vector<float>());
  }
}

// The plan from lg: each row's top 8 (ties to the lower expert), scored by a
// softmax over them; the experts in first-use order, row by row; ids offset
// by a + 128 b, a last turn short of 4 padded with the first expert at score
// 0. Floats are emulated on the scalar unit, so the selection compares
// order-preserving int keys.
void plan_make(float *__restrict lg, uint8_t *__restrict plan, int32_t a,
               int32_t b) {
  constexpr int NE = 128, MAXU = 64;
  int32_t *pi = (int32_t *)plan;
  float *pf = (float *)plan;
#pragma clang loop unroll(disable)
  for (int i = 0; i < (80 + 8 * MAXU) * 4; i += 64)
    aie::store_v(plan + i, aie::zeros<uint8_t, 64>());
  int32_t slot[NE], ex[MAXU], key[NE];
#pragma clang loop unroll(disable)
  for (int e = 0; e < NE; e++)
    slot[e] = -1;
  int u = 0;
  constexpr float log2_e = 1.44269504088896341f;
#pragma clang loop unroll(disable)
  for (int r = 0; r < 8; r++) {
#pragma clang loop unroll(disable)
    for (int e = 0; e < NE; e++) {
      const int32_t v =
          __builtin_bit_cast(int32_t, lg[(e >> 3) * 64 + r * 8 + (e & 7)]);
      key[e] = v ^ ((v >> 31) & 0x7fffffff);
    }
    int top[8];
#pragma clang loop unroll(disable)
    for (int i = 0; i < 8; i++) {
      int32_t best = INT32_MIN;
      int bi = 0;
#pragma clang loop unroll(disable)
      for (int e = 0; e < NE; e++)
        if (key[e] > best) {
          best = key[e];
          bi = e;
        }
      top[i] = bi;
      key[bi] = INT32_MIN;
    }
    // softmax over the 8 in fixed point: 2^-x from a Q27 table, so the
    // scores are f32-exact where bf16 exp2 is not (floats are emulated)
    const float l0 = lg[(top[0] >> 3) * 64 + r * 8 + (top[0] & 7)];
    int32_t e[8], sum = 0;
#pragma clang loop unroll(disable)
    for (int i = 0; i < 8; i++) {
      const float d =
          (l0 - lg[(top[i] >> 3) * 64 + r * 8 + (top[i] & 7)]) * log2_e;
      const int32_t q = d > 30.0f ? 30 << 16 : (int32_t)(d * 65536.0f + 0.5f);
      const int n = q >> 16, j = (q >> 8) & 255, f = q & 255;
      const int32_t a = EXP2_Q27[j], b = EXP2_Q27[j + 1];
      e[i] = (a + (((b - a) * f) >> 8)) >> n;
      sum += e[i];
    }
    const float inv = 1.0f / (float)sum;
#pragma clang loop unroll(disable)
    for (int i = 0; i < 8; i++) {
      int s = slot[top[i]];
      if (s < 0) {
        s = u++;
        slot[top[i]] = s;
        ex[s] = top[i];
      }
      pf[80 + 8 * s + r] = (float)e[i] * inv;
    }
  }
  const int base = a + NE * b;
  pi[0] = (u + 3) / 4 < 2 ? 2 : (u + 3) / 4;
#pragma clang loop unroll(disable)
  for (int c = 0; c < 4; c++)
    pi[1 + c] = c;
#pragma clang loop unroll(disable)
  for (int s = 0; s < MAXU; s++)
    pi[8 + s] = base + ex[s < u ? s : 0];
}

} // extern "C"

namespace {
constexpr int HT = NTL * RW4; // a residual core's f32 words
constexpr int STG = 256;      // f32 words of a residual way-in stage
constexpr int HSTG = HT / STG;
// a message to the feeder: [plan | x as bfp16 tiles]
constexpr int PLAN_B = (80 + 8 * 64) * 4;
} // namespace

extern "C" {

// ---- residual cores

// stage k of a layer's way in, all in phase 0: h (kept only by the
// dispatch's first layer), then RMSNorm 1's and 2's weights, bf16 pairs in
// the f32 words
void res_in2(float *__restrict stg, float *__restrict h, bf16 *__restrict wn,
             int32_t k, int32_t ph, int32_t lay) {
  if (k < HSTG) {
    if (lay == 0)
      for (int i = 0; i < STG; i += 32)
        aie::store_v(h + STG * k + i, aie::load_v<32>(stg + i));
    return;
  }
  bf16 *d = wn + 2 * STG * (k - HSTG);
  const bf16 *s = (const bf16 *)stg;
  for (int i = 0; i < 2 * STG; i += 32)
    aie::store_v(d + i, aie::load_v<32>(s + i));
}

// phase ph's counts: n[0] stages in, n[1] rounds, n[2] x pieces out: (h +
// both norm weights, -, x1), (-, o_proj, x2), (-, the MoE, -)
void res_counts(int32_t *__restrict n, int32_t ph) {
  n[0] = ph == 0 ? HSTG + 8 : 0;
  // o_proj's rounds: 4 (a row group per column each); the MoE's: 16
  n[1] = ph == 0 ? 0 : (ph == 1 ? 4 : 16);
  n[2] = ph == 2 ? 0 : 4;
}

// h += round k of phase ph: o_proj's in phase 1, the MoE's in phase 2
void res_add(bf16 *__restrict rnd, float *__restrict h, int32_t k, int32_t c,
             int32_t ph) {
  if (ph == 2)
    res_moe(rnd, h, k, c);
  else
    res_o(rnd, h, k, c);
}

// xs [64 tiles][4 rows][8] bf16 = rows of RMSNorm(h) w, dim tiles 64 q ..;
// w = wn's half ph: RMSNorm 1's, then 2's
void res_xn(float *__restrict h, bf16 *__restrict wn, bf16 *__restrict xs,
            int32_t q, int32_t ph) {
  const bf16 *w = wn + 2048 * ph;
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  aie::accum<accfloat, 32> s = aie::zeros<accfloat, 32>();
  for (int i = 0; i < NTL; i++) {
    aie::accum<accfloat, 32> a;
    a.from_vector(aie::load_v<32>(h + i * RW4));
    aie::vector<bf16, 32> hi = a.template to_vector<bf16>();
    aie::vector<bf16, 32> lo =
        aie::msc(a, hi, bf16(1.0f)).template to_vector<bf16>();
    s = aie::mac(s, hi, hi);
    s = aie::mac(s, hi, lo);
    s = aie::mac(s, hi, lo);
  }
  alignas(aie::vector_decl_align) float v[32];
  aie::store_v(v, s.template to_vector<float>());
  alignas(aie::vector_decl_align) float rs[8] = {}, rl[8] = {};
  for (int t = 0; t < 4; t++) {
    float m = 0;
    for (int i = 0; i < 8; i++)
      m += v[t * 8 + i];
    float r = aie::invsqrt(m / 2048.0f + 1e-6f);
    rs[t] = r;
    rl[t] = r - (float)(bf16)r;
  }
  alignas(aie::vector_decl_align) bf16 bh[8], bl[8];
  for (int t = 0; t < 8; t++) {
    bh[t] = (bf16)rs[t];
    bl[t] = (bf16)rl[t];
  }
  const VB sh = rows8(aie::load_v<8>(bh)), sl = rows8(aie::load_v<8>(bl));
  alignas(aie::vector_decl_align) float hx[64] = {};
  alignas(aie::vector_decl_align) bf16 yt[64];
  for (int k = 0; k < NTL / 4; k++) {
    const int i = NTL / 4 * q + k;
    VB wi = cols8(aie::load_v<8>(w + i * 8));
    ACC g = aie::mul(sh, wi);
    g = aie::mac(g, sl, wi);
    VB gh, gl;
    limbs(g, gh, gl);
    aie::store_v(hx, aie::load_v<32>(h + i * RW4));
    ACC x;
    x.from_vector(aie::load_v<64>(hx));
    VB xh, xl;
    limbs(x, xh, xl);
    ACC y = aie::mul(xh, gh);
    y = aie::mac(y, xl, gh);
    y = aie::mac(y, xh, gl);
    aie::store_v(yt, y.template to_vector<bf16>());
    aie::store_v(xs + k * 32, aie::load_v<32>(yt));
  }
}

// ---- units

// a q round's heads into qs [token][8 heads][DH], raw
void qk_raw(bf16 *__restrict rnd, bf16 *__restrict qs, int32_t h) {
  for (int t = 0; t < TOK; t++)
    for (int c = 0; c < 4; c++)
      gather(rnd, c, t, qs + t * 1024 + (4 * h + c) * DH);
}

} // extern "C"

namespace {
// q normed and rotated by RW into qb. Its own frame: a call from qk_step
// to qk_blk would otherwise stack both past the core's stack
__attribute__((noinline)) void qk_rope(const bf16 *qs, const bf16 *rw,
                                       uint8_t *qb) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  OS os((bfp16ebs8 *)qb);
  alignas(aie::vector_decl_align) bf16 y[QH * DH], tile[64];
  for (int t = 0; t < TOK; t++) {
    for (int hd = 0; hd < QH; hd++) {
      bf16 *yh = y + hd * DH;
      norm_rope(qs + t * 1024 + hd * DH, rw + QN, rw + COS + 64 * t,
                rw + SIN + 64 * t, yh);
      for (int i = 0; i < DH; i += 16)
        aie::store_v(yh + i, aie::mul(aie::load_v<16>(yh + i), (bf16)SCALE)
                                 .template to_vector<bf16>());
    }
    // tile d: [8 heads][dims 8 d ..]
    for (int d = 0; d < 16; d++) {
      for (int hd = 0; hd < QH; hd++)
        aie::store_v(tile + hd * 8, aie::load_v<8>(y + hd * DH + d * 8));
      put_tile(os, aie::load_v<64>(tile));
    }
  }
}
} // namespace

extern "C" {

// block b of a unit's K stream: block 0 is RW, which norms and rotates q
// into qb; block b > 0 is key block b - 1. s: the block's scores (block 0's
// are not read)
void qk_step(bf16 *__restrict qs, bf16 *__restrict kb, uint8_t *__restrict qb,
             uint8_t *__restrict kf, bf16 *__restrict m, bf16 *__restrict s,
             int32_t b, int32_t pos, int32_t causal) {
  if (b > 0)
    qk_blk(qb, kb, kf, m, s, b - 1, pos, causal);
  else
    qk_rope(qs, kb, qb);
}

// block b of a unit's V stream; block 0 is a dummy
void pv_step(bf16 *__restrict s, bf16 *__restrict vb, uint8_t *__restrict vf,
             float *__restrict y, int32_t b, int32_t pos, int32_t causal) {
  if (b > 0)
    pv_blk(s, vb, vf, y, b - 1, pos, causal);
}

// both halves of a unit's output, one after the other
void pv_fin2(float *__restrict y, uint8_t *__restrict ot) {
  pv_fin(y, ot, 0);
  pv_fin(y, ot + QH / 2 * 16 * BLK, 1);
}

// ---- planner

// x2_tiles into a message's x
void px_tiles(bf16 *__restrict x, uint8_t *__restrict px, int32_t q) {
  x2_tiles(x, px + PLAN_B, q);
}

// ---- feeder

// fd_xpart from a message
void fd_xpart2(uint8_t *__restrict hm, uint8_t *__restrict px, int32_t p) {
  fd_xpart(hm, px + PLAN_B, px, p);
}

// ---- router

void slot_set(int32_t *__restrict sl, int32_t v) { sl[0] = v; }

} // extern "C"
