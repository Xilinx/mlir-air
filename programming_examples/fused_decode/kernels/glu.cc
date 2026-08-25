// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//

#include "aie_array_layout.h"
#include "aie_kernel_utils.h"
#include "lut_based_ops.h"

template <int L>
void pseduo_glu(bf16 *y, const bf16 *x) {
  bf16 *gate_ptr = const_cast<bf16 *>(x) + (L / 2);
  bf16 *hid_ptr = const_cast<bf16 *>(x);
  bf16 *y_ptr = y;
  for (int i = 0; i < L / 2; i += 16) {
    aie::vector<bf16, 16> gate_vec = aie::load_v<16>(gate_ptr + i);
    aie::vector<bf16, 16> hid_vec = aie::load_v<16>(hid_ptr + i);
    gate_vec = getActivationBf16(gate_vec);

    aie::vector<bf16, 16> y_vec = aie::mul(gate_vec, hid_vec);
    aie::store_v(y_ptr + i, y_vec);
  }
}

constexpr int x_prod_lock = 0;
constexpr int x_cons_lock = 1;
constexpr int y_prod_lock = 2;
constexpr int y_cons_lock = 3;
// phase 1: qkv proj, D -> (D + D // 2) to RoPE
// phase 2: o_proj, D -> D to rms_residual
// phase 3: up/gate proj, D -> D * 8 to MT_GLU
// phase 4: down proj, D * 4 -> D, to rms_residual
// phase 4 is special, do not clear y to zero
extern "C" {

// AIR-friendly pure-compute GLU (no in-kernel locks; AIR owns sync),
// proj_qmm.cc pattern. x = [up(GLU_SLICE/2) ++ gate(GLU_SLICE/2)]; y[i] =
// silu(gate[i])*up[i] (GLU_SLICE/2 outputs). GLU_SLICE=1024 -> x[1024], y[512].
void glu_aie(bf16 *restrict y, bf16 *restrict x, int _arm) {
  (void)_arm; // per-token RTP arm-gate operand (kept alive so AIR emits the arm
              // lock)
#if defined(GLU_ROW_PROBE) && GLU_ROW_PROBE == 2
  // The batch-1 half of the probe, so the two builds can be compared under it.
  for (int i = 0; i < GLU_SLICE / 2; i += 16)
    aie::store_v(y + i, aie::sub(aie::load_v<16>(x + i),
                                 aie::load_v<16>(x + GLU_SLICE / 2 + i)));
#elif defined(GLU_ROW_PROBE) && GLU_ROW_PROBE == 4
  for (int i = 0; i < GLU_SLICE / 2; i += 16) {
    aie::vector<bf16, 16> gate = aie::load_v<16>(x + GLU_SLICE / 2 + i);
    aie::vector<bf16, 16> hid = aie::load_v<16>(x + i);
    aie::vector<bf16, 16> o = aie::mul(gate, hid);
    aie::store_v(y + i, o);
  }
#elif defined(GLU_ROW_PROBE) && GLU_ROW_PROBE == 5
  // The batch-1 twin of the table probe. This is the control the whole
  // diagnosis turns on: the SAME getActivationBf16, the same core, the same
  // silicon, differing only in whether the build is batched.
  for (int i = 0; i < GLU_SLICE / 2; i += 16) {
    aie::vector<bf16, 16> gate = aie::load_v<16>(x + GLU_SLICE / 2 + i);
    gate = getActivationBf16(gate);
    aie::store_v(y + i, gate);
  }
#else
  pseduo_glu<GLU_SLICE>(y, x);
#endif
}

// DECODE_BATCH > 1, LM HEAD arm: half a slice, copied. No math at all.
//
// This exists for a LOCK reason, not an arithmetic one. On the vocab arm this
// core is a pure relay -- a slice of logits arrives on the gate-up dest and
// leaves on the second MM2S -- and the obvious spelling sends the INPUT buffer
// straight back out. That makes the input buffer both DMA-written and
// DMA-read, and AIR then sizes its lock credit by the ratio of those two counts
// (getLockValuePair), which across two arms comes out at 2 and stalls the port.
// Copying into the output buffer keeps each buffer one-directional -- input
// written only, output read only -- which is the case AIR gives credit 1
// unconditionally, and is what the decode arm already looks like.
//
// The cost is one L1-to-L1 pass over the slice per relay, against a vocab GEMV
// of 2048x512 MACs behind it.
void glu_copy_aie(bf16 *restrict y, bf16 *restrict x, int off, int n,
                  int _arm) {
  (void)_arm; // per-token RTP arm-gate operand, as in glu_aie
  const bf16 *src = x + off;
  for (int i = 0; i < n; i += 16)
    aie::store_v(y + i, aie::load_v<16>(src + i));
}

// DECODE_BATCH > 1: row t of a batched round. The gate-up projection egresses
// (round, token), so one round arrives as [BATCH][GLU_SLICE] and leaves as
// [BATCH][GLU_SLICE/2] -- the GLU itself is per token and unchanged, which is
// the whole reason this is a row index rather than a new kernel.
void glu_row_aie(bf16 *restrict y, bf16 *restrict x, int t, int _arm) {
  (void)_arm; // per-token RTP arm-gate operand, as in glu_aie
#ifdef GLU_ROW_PROBE
  // Diagnostic builds only, and each answers ONE question about the batched
  // gate-up egress that no descriptor check can:
  //   1  swap the halves     -- did the two egress rounds land [gate|up]?
  //   2  y = up - gate       -- antisymmetric and silu-free: reads BOTH halves,
  //                             so a half landing in the wrong place shows, and
  //                             a swap shows as an exact sign flip
  //   3  y = up              -- the first half alone
  //   4  y = gate*up         -- the real kernel with ONLY the LUT removed.
  //                             Probe 2 clears the plumbing, but at ~13x the
  //                             real GLU magnitude, so it cannot tell a fault
  //                             that scales with the signal from one that does
  //                             not. This has the same magnitude and the same
  //                             multiply as the real thing and no table, so it
  //                             separates getActivationBf16 from aie::mul.
  //   5  y = silu(gate)      -- the LUT alone, no multiply. If 4 is clean and
  //                             5 is not, the table is what is wrong.
  bf16 *xr = x + t * GLU_SLICE;
  bf16 *yr = y + t * (GLU_SLICE / 2);
#if GLU_ROW_PROBE == 1
  for (int i = 0; i < GLU_SLICE / 2; i += 16) {
    aie::vector<bf16, 16> up = aie::load_v<16>(xr + i);
    aie::vector<bf16, 16> gate = aie::load_v<16>(xr + GLU_SLICE / 2 + i);
    up = getActivationBf16(up);
    aie::vector<bf16, 16> o = aie::mul(up, gate);
    aie::store_v(yr + i, o);
  }
#elif GLU_ROW_PROBE == 2
  for (int i = 0; i < GLU_SLICE / 2; i += 16)
    aie::store_v(yr + i, aie::sub(aie::load_v<16>(xr + i),
                                  aie::load_v<16>(xr + GLU_SLICE / 2 + i)));
#elif GLU_ROW_PROBE == 4
  // pseduo_glu with getActivationBf16 deleted and nothing else changed.
  for (int i = 0; i < GLU_SLICE / 2; i += 16) {
    aie::vector<bf16, 16> gate = aie::load_v<16>(xr + GLU_SLICE / 2 + i);
    aie::vector<bf16, 16> hid = aie::load_v<16>(xr + i);
    aie::vector<bf16, 16> o = aie::mul(gate, hid);
    aie::store_v(yr + i, o);
  }
#elif GLU_ROW_PROBE == 5
  // The table alone. Same loads, same store, no multiply.
  for (int i = 0; i < GLU_SLICE / 2; i += 16) {
    aie::vector<bf16, 16> gate = aie::load_v<16>(xr + GLU_SLICE / 2 + i);
    gate = getActivationBf16(gate);
    aie::store_v(yr + i, gate);
  }
#else
  for (int i = 0; i < GLU_SLICE / 2; i += 16)
    aie::store_v(yr + i, aie::load_v<16>(xr + i));
#endif
#else
  pseduo_glu<GLU_SLICE>(y + t * (GLU_SLICE / 2), x + t * GLU_SLICE);
#endif
}

// Small-slice variant for the demux8 wire-up bisection (M=256 proj payload):
// x = [up(128) ++ gate(128)], y[i] = silu(gate[i])*up[i] (128 out).
void glu_aie256(bf16 *restrict y, bf16 *restrict x) { pseduo_glu<256>(y, x); }

// GLU->proj-X feedback isolation (glu_fb_min): gate/up proj output 512 =
// [up(256) ++ gate(256)] -> y[256] = silu(gate)*up. y[256] is then a legal
// proj K (multiple of COL_BLOCK=256) = the down_proj X.
void glu_aie512(bf16 *restrict y, bf16 *restrict x) { pseduo_glu<512>(y, x); }

// demux_fb MLP integration: gate/up proj cycle-test output 3072 =
// [up(1536) ++ gate(1536)] -> y[1536] = silu(gate)*up.
void glu_aie3072(bf16 *restrict y, bf16 *restrict x) { pseduo_glu<3072>(y, x); }

// demux_fb MLP increment 2: GLU out 1536 zero-padded to 2048 so the down phase
// X keeps the uniform proj K=2048 (down = W_down @ [glu(1536) ++ zeros(512)]).
void glu_aie3072_pad2048(bf16 *restrict y, bf16 *restrict x) {
  pseduo_glu<3072>(y, x);
  for (int i = 1536; i < 2048; i++)
    y[i] = (bf16)0.0f;
}

// Header-bearing variant for the PACKET x-feed (STAGE_MLP>=2 down-phase X):
// pkt_id at y+14, padded GLU payload (1536 + 512 zeros) at y+16. Streamed from
// offset 14.
void glu_aie3072_pad2048_hdr(bf16 *restrict y, bf16 *restrict x,
                             unsigned int pkt_id) {
  *reinterpret_cast<unsigned int *>(y + 14) = pkt_id;
  pseduo_glu<3072>(y + 16, x);
  for (int i = 1536; i < 2048; i++)
    y[16 + i] = (bf16)0.0f;
}

// MLP_REAL (Llama-3.2-1B real dims): gate/up proj output 16384 =
// [up(8192) ++ gate(8192)] -> y[8192] = silu(gate)*up = INTERMEDIATE. No
// padding: the down phase K = INTERMEDIATE = 8192 exactly.
void glu_aie16384(bf16 *restrict y, bf16 *restrict x) {
  pseduo_glu<16384>(y, x);
}

// Header-bearing variant for the PACKET x-feed (MLP_REAL down-phase X): pkt_id
// at y+14, GLU payload (8192) at y+16. Streamed from offset 14.
void glu_aie16384_hdr(bf16 *restrict y, bf16 *restrict x, unsigned int pkt_id) {
  *reinterpret_cast<unsigned int *>(y + 14) = pkt_id;
  pseduo_glu<16384>(y + 16, x);
}

// In-place GLU for the 16384 gate/up (MLP_REAL): write
// out[i]=silu(gate[i])*up[i] back into x[i] (the up region). Safe: iter i reads
// up[i] and gate[8192+i] before storing x[i]; the gate region [8192:16384] is
// never written while i<8192. Saves the 16KB separate output buffer -> fits L1
// with the silu activation LUT.
void glu_aie16384_inplace(bf16 *x) { pseduo_glu<16384>(x, x); }

// Header-bearing in-place GLU (MLP_REAL STAGE_MLP>=2 down-phase X packet
// x-feed). x must be sized 16400 = 16384 input + YHDR(16). Compute glu in-place
// into the up region x[0:8192] (gate x[8192:16384] consumed), then emit
// [pkt_id@+14, payload@+16] into the now-free region starting at x+8192. xFb is
// put from x+8206 (=8192+14), length HDR(2)+8192. Read x[0:8192] / write
// x[8208:16400] do not overlap.
void glu_aie16384_hdr_inplace(bf16 *x, unsigned int pkt_id) {
  pseduo_glu<16384>(x, x);
  bf16 *out = x + 8192;
  for (int i = 0; i < 8192; i += 16) {
    aie::vector<bf16, 16> v = aie::load_v<16>(x + i);
    aie::store_v(out + 16 + i, v);
  }
  *reinterpret_cast<unsigned int *>(out + 14) = pkt_id;
}

void glu(bf16 *y_ping, bf16 *y_pong, const bf16 *x_ping, const bf16 *x_pong) {
  _lock_acquire(x_cons_lock);
  _lock_acquire(y_prod_lock);

  pseduo_glu<GLU_SLICE>(y_ping, x_ping);

  _lock_release(y_cons_lock);
  _lock_release(x_prod_lock);

  _lock_acquire(x_cons_lock);
  _lock_acquire(y_prod_lock);

  pseduo_glu<GLU_SLICE>(y_pong, x_pong);

  _lock_release(y_cons_lock);
  _lock_release(x_prod_lock);
}
}
