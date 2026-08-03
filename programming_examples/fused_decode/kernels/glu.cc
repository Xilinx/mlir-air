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
  pseduo_glu<GLU_SLICE>(y, x);
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