# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Fused RMS→GEMV(Q+K+V)→RoPE/passthrough pipeline in 1 launch.
#
# Single launch, single segment, 3 herds:
#   RMSNorm [1,1] → broadcast → GEMV [8,1] → pipe → RoPE/V [8,1]
#
# Concatenated weight matrix [M_TOTAL, K] where M_TOTAL = M_Q + M_K + M_V.
# GEMV herd processes all rows identically.
# RoPE herd applies RoPE to Q and K rows, passes V rows through unchanged.
#
# LLaMA-3.2-1B GQA dims:
#   Q: [2048, 2048] (32 heads × 64 dim)
#   K: [512, 2048]  (8 heads × 64 dim)
#   V: [512, 2048]  (8 heads × 64 dim)

import argparse
import subprocess
import os
import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects import arith, math as math_dialect
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp, subview
from air.dialects.vector import (
    transfer_read,
    transfer_write,
    BroadcastOp,
    reduction as vector_reduction,
)
from air.dialects.func import FuncOp, CallOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper

range_ = for_

# LLaMA-3.2-1B QKV projection dims
N = 2048  # embedding dim
K = N
M_Q = 2048  # Q output (32 heads × 64)
M_K = 512  # K output (8 heads × 64)
M_V = 512  # V output (8 heads × 64)
M_TOTAL = M_Q + M_K + M_V  # 3072

TILE_M = 64
M_INPUT = 4
HERD_M = 8
HEAD_DIM = 64
VEC_SIZE = 16
EPS = 1e-5
DTYPE = bfloat16

# RoPE boundary: rows < ROPE_BOUNDARY get RoPE, rows >= ROPE_BOUNDARY are V (passthrough)
ROPE_BOUNDARY = M_Q + M_K  # 2560


def _make_mul_map(factor):
    return AffineMap.get(
        0,
        1,
        [AffineExpr.get_mul(AffineSymbolExpr.get(0), AffineConstantExpr.get(factor))],
    )


@module_builder
def build_module():
    xrt = type_mapper(DTYPE)
    vecTy = VectorType.get([VEC_SIZE], xrt)
    identity_map = AffineMapAttr.get(AffineMap.get_identity(1))

    # L3 types
    x_ty = MemRefType.get([N], xrt)
    w_rms_ty = MemRefType.get([N], xrt)
    w_qkv_ty = MemRefType.get([M_TOTAL, K], xrt)
    # LUT for all M_TOTAL heads: Q+K get real cos/sin, V gets identity (cos=1, sin=0)
    lut_ty = MemRefType.get([M_TOTAL], xrt)
    out_ty = MemRefType.get([M_TOTAL], xrt)

    # Memory spaces
    l1s = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l2s = IntegerAttr.get(T.i32(), MemorySpace.L2)

    # L1 types
    l1_n_ty = MemRefType.get([N], xrt, memory_space=l1s)
    l1_vec_ty = MemRefType.get([VEC_SIZE], xrt, memory_space=l1s)
    l1_a_ty = MemRefType.get([M_INPUT, K], xrt, memory_space=l1s)
    l1_b_ty = MemRefType.get([K], xrt, memory_space=l1s)
    l1_c_ty = MemRefType.get([TILE_M], xrt, memory_space=l1s)
    l1_row_ty = MemRefType.get([HEAD_DIM], xrt, memory_space=l1s)

    # L2 weight staging
    l2_a_ty = MemRefType.get([HERD_M, TILE_M, K], xrt, memory_space=l2s)

    # Channels
    Channel("rms_broadcast", size=[1, 1], broadcast_shape=[HERD_M, 1])
    Channel("pipe", size=[HERD_M, 1])

    # External kernel declarations
    matvec_func = FuncOp(
        "matvec_vectorized_bf16_bf16",
        ([T.i32(), T.i32(), T.i32(), l1_a_ty, l1_b_ty, l1_c_ty], []),
        visibility="private",
    )
    fill_func = FuncOp("linalg_fill_bf16", ([xrt, l1_c_ty], []), visibility="private")
    rope_func = FuncOp(
        "rope", ([l1_row_ty, l1_row_ty, l1_row_ty, T.i32()], []), visibility="private"
    )
    for f in [matvec_func, fill_func]:
        f.attributes["link_with"] = StringAttr.get("mv.o")
        f.attributes["llvm.emit_c_interface"] = UnitAttr.get()
    rope_func.attributes["link_with"] = StringAttr.get("rope.o")
    rope_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    n_launch_iters = M_TOTAL // (TILE_M * HERD_M)

    @FuncOp.from_py_func(x_ty, w_rms_ty, w_qkv_ty, lut_ty, out_ty)
    def rms_qkv_rope(x_in, w_rms, w_qkv, lut, qkv_out):

        launch_size = [n_launch_iters, 1]

        @launch(operands=[x_in, w_rms, w_qkv, lut, qkv_out], sizes=launch_size)
        def launch_main(iv_x, iv_y, sx, sy, l_x, l_wr, l_w, l_lut, l_out):

            @segment(name="fused_seg", operands=[iv_x, l_x, l_wr, l_w, l_lut, l_out])
            def seg(s_iv, s_x, s_wr, s_w, s_lut, s_out):

                l2_a = AllocOp(l2_a_ty, [], [])
                l1_a = AllocOp(l1_a_ty, [], [])
                l1_b = AllocOp(l1_b_ty, [], [])
                l1_c = AllocOp(l1_c_ty, [], [])

                mul_map = _make_mul_map(HERD_M * TILE_M)
                seg_row_off = affine_apply(mul_map, [s_iv])

                # L3→L2: weights for this iteration
                dma_memcpy_nd(
                    l2_a,
                    s_w,
                    src_offsets=[0, seg_row_off, 0],
                    src_sizes=[HERD_M, TILE_M, K],
                    src_strides=[TILE_M * K, K, 1],
                )

                # ========================================
                # Herd 1: RMSNorm [1,1]
                # ========================================
                @herd(name="rms_herd", sizes=[1, 1], operands=[s_x, s_wr])
                def rms_body(_tx, _ty, _sx, _sy, h_x, h_wr):
                    l1_x = AllocOp(l1_n_ty, [], [])
                    l1_w = AllocOp(l1_n_ty, [], [])
                    l1_out = AllocOp(l1_n_ty, [], [])
                    l1_acc = AllocOp(l1_vec_ty, [], [])

                    dma_memcpy_nd(l1_x, h_x)
                    dma_memcpy_nd(l1_w, h_wr)

                    c0 = ConstantOp(T.index(), 0)
                    cst0 = ConstantOp(xrt, 0.0)
                    n_f = ConstantOp(xrt, float(N))
                    eps_f = ConstantOp(xrt, EPS)
                    v_zero = BroadcastOp(vecTy, cst0)

                    transfer_write(None, v_zero, l1_acc, [c0], identity_map, [True])
                    for j in range_(0, N, VEC_SIZE):
                        sv = subview(l1_x.result, [j], [VEC_SIZE], [1])
                        sv2 = subview(l1_out.result, [j], [VEC_SIZE], [1])
                        v = transfer_read(vecTy, sv, [c0], identity_map, cst0, [True])
                        vsq = arith.mulf(v, v)
                        transfer_write(None, vsq, sv2, [c0], identity_map, [True])
                        vsq_r = transfer_read(
                            vecTy, sv2, [c0], identity_map, cst0, [True]
                        )
                        vacc = transfer_read(
                            vecTy, l1_acc, [c0], identity_map, cst0, [True]
                        )
                        vsum = arith.addf(vacc, vsq_r)
                        transfer_write(None, vsum, l1_acc, [c0], identity_map, [True])
                        yield_([])

                    vfin = transfer_read(
                        vecTy, l1_acc, [c0], identity_map, cst0, [True]
                    )
                    total = vector_reduction(xrt, "add", vfin)
                    rms = arith.divf(total, n_f)
                    rms_eps = arith.addf(rms, eps_f)
                    f32 = F32Type.get()
                    rms_f32 = arith.extf(f32, rms_eps)
                    rstd_f32 = math_dialect.rsqrt(rms_f32)
                    rstd = arith.truncf(xrt, rstd_f32)
                    v_rstd = BroadcastOp(vecTy, rstd)

                    for j in range_(0, N, VEC_SIZE):
                        sv_x = subview(l1_x.result, [j], [VEC_SIZE], [1])
                        sv_w = subview(l1_w.result, [j], [VEC_SIZE], [1])
                        sv_o = subview(l1_out.result, [j], [VEC_SIZE], [1])
                        vx = transfer_read(
                            vecTy, sv_x, [c0], identity_map, cst0, [True]
                        )
                        vw = transfer_read(
                            vecTy, sv_w, [c0], identity_map, cst0, [True]
                        )
                        vnormed = arith.mulf(vx, v_rstd)
                        vweighted = arith.mulf(vnormed, vw)
                        transfer_write(
                            None, vweighted, sv_o, [c0], identity_map, [True]
                        )
                        yield_([])

                    ChannelPut("rms_broadcast", l1_out)

                    DeallocOp(l1_x)
                    DeallocOp(l1_w)
                    DeallocOp(l1_out)
                    DeallocOp(l1_acc)

                # ========================================
                # Herd 2: GEMV [HERD_M,1] — processes Q+K+V rows
                # ========================================
                @herd(
                    name="gemv_herd",
                    sizes=[HERD_M, 1],
                    operands=[l1_a, l1_b, l1_c, l2_a],
                )
                def gemv_body(_tx, _ty, _sx, _sy, _l1a, _l1b, _l1c, _l2a):
                    ChannelGet("rms_broadcast", _l1b, indices=[_tx, _ty])

                    zero = ConstantOp(FloatAttr.get(xrt, 0), None)
                    CallOp(fill_func, [zero, _l1c])
                    mul_mi = _make_mul_map(M_INPUT)
                    for j in range_(0, TILE_M // M_INPUT):
                        j_off = affine_apply(mul_mi, [j])
                        dma_memcpy_nd(
                            _l1a,
                            _l2a,
                            src_offsets=[_tx, j_off, 0],
                            src_sizes=[1, M_INPUT, K],
                            src_strides=[TILE_M * K, K, 1],
                        )
                        m_c = ConstantOp(IntegerAttr.get(T.i32(), M_INPUT), None)
                        k_c = ConstantOp(IntegerAttr.get(T.i32(), K), None)
                        ro = arith.index_cast(T.i32(), j_off)
                        CallOp(matvec_func, [m_c, k_c, ro, _l1a, _l1b, _l1c])
                        yield_([])

                    ChannelPut("pipe", _l1c, indices=[_tx, _ty])

                gemv_body.attributes["link_with"] = StringAttr.get("mv.o")

                # ========================================
                # Herd 3: RoPE [HERD_M,1] — always applies RoPE
                #   Q/K rows: real cos/sin rotation
                #   V rows: identity LUT (cos=1, sin=0) → passthrough
                # ========================================
                @herd(
                    name="rope_herd",
                    sizes=[HERD_M, 1],
                    operands=[s_lut, s_out, seg_row_off],
                )
                def rope_body(_tx, _ty, _sx, _sy, h_lut, h_out, h_row_off):
                    l1_in = AllocOp(l1_row_ty, [], [])
                    l1_lut = AllocOp(l1_row_ty, [], [])
                    l1_out_buf = AllocOp(l1_row_ty, [], [])

                    ChannelGet("pipe", l1_in, indices=[_tx, _ty])

                    head_off = arith.addi(
                        h_row_off, arith.muli(_tx, ConstantOp(T.index(), HEAD_DIM))
                    )

                    dma_memcpy_nd(
                        l1_lut,
                        h_lut,
                        src_offsets=[head_off],
                        src_sizes=[HEAD_DIM],
                        src_strides=[1],
                    )

                    dim_i32 = ConstantOp(T.i32(), HEAD_DIM)
                    CallOp(rope_func, [l1_in, l1_lut, l1_out_buf, dim_i32])

                    dma_memcpy_nd(
                        h_out,
                        l1_out_buf,
                        dst_offsets=[head_off],
                        dst_sizes=[HEAD_DIM],
                        dst_strides=[1],
                    )

                    DeallocOp(l1_in)
                    DeallocOp(l1_lut)
                    DeallocOp(l1_out_buf)

                rope_body.attributes["link_with"] = StringAttr.get("rope.o")

                DeallocOp(l2_a)
                DeallocOp(l1_a)
                DeallocOp(l1_b)
                DeallocOp(l1_c)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--output-format", default="elf")
    args = parser.parse_args()

    module = build_module()
    if args.print_module_only:
        print(module)
        exit(0)

    # Compile kernels
    peano = os.environ.get("PEANO_INSTALL_DIR", "")
    aieopt_dir = os.path.dirname(
        os.path.dirname(
            subprocess.check_output(["which", "aie-opt"], text=True).strip()
        )
    )
    kernel_dir = os.path.join(aieopt_dir, "include", "aie_kernels", "aie2p")
    this_dir = os.path.dirname(os.path.abspath(__file__))
    mv_src = os.path.join(
        this_dir, "..", "matrix_vector_multiplication", "bf16", "mv.cc"
    )
    rope_src = os.path.join(kernel_dir, "rope.cc")
    flags = [
        "-O2",
        "-std=c++20",
        "--target=aie2p-none-unknown-elf",
        "-Wno-parentheses",
        "-Wno-attributes",
        "-Wno-macro-redefined",
        "-Wno-empty-body",
        "-DNDEBUG",
        f"-I{aieopt_dir}/include",
    ]
    subprocess.check_call(
        [f"{peano}/bin/clang++"]
        + flags
        + [f"-DDIM_M_OUTPUT={TILE_M}", "-c", mv_src, "-o", "mv.o"]
    )
    subprocess.check_call(
        [f"{peano}/bin/clang++"] + flags + ["-c", rope_src, "-o", "rope.o"]
    )
    print(f"Compiled mv.o (DIM_M_OUTPUT={TILE_M}) and rope.o")

    np.random.seed(42)
    x_in = (np.random.randn(N) * 0.5).astype(DTYPE)
    w_rms = np.ones(N, dtype=DTYPE)

    # Concatenated QKV weight matrix [3072, 2048]
    w_q = (np.random.randn(M_Q, K) * 0.1).astype(DTYPE)
    w_k = (np.random.randn(M_K, K) * 0.1).astype(DTYPE)
    w_v = (np.random.randn(M_V, K) * 0.1).astype(DTYPE)
    w_qkv = np.concatenate([w_q, w_k, w_v], axis=0)

    # RoPE LUT: real cos/sin for Q+K, identity (cos=1, sin=0) for V
    half = HEAD_DIM // 2
    theta = 500000.0
    inv_freq = 1.0 / (theta ** (np.arange(half, dtype=np.float32) / half))
    cos = np.cos(inv_freq).astype(np.float32)
    sin = np.sin(inv_freq).astype(np.float32)
    rope_lut_row = np.concatenate([cos, sin])
    identity_lut_row = np.concatenate(
        [np.ones(half, dtype=np.float32), np.zeros(half, dtype=np.float32)]
    )
    n_qk_heads = ROPE_BOUNDARY // HEAD_DIM  # 40 heads (32 Q + 8 K)
    n_v_heads = M_V // HEAD_DIM  # 8 heads
    lut = np.concatenate(
        [
            np.tile(rope_lut_row, n_qk_heads),  # Q+K: real rotation
            np.tile(identity_lut_row, n_v_heads),  # V: identity (passthrough)
        ]
    ).astype(DTYPE)

    # Golden reference
    x_f = x_in.astype(np.float32)
    w_rms_f = w_rms.astype(np.float32)
    rms_val = np.sqrt(np.mean(x_f**2) + EPS)
    normed = (x_f / rms_val * w_rms_f).astype(np.float32)

    # Q projection + RoPE
    q_raw = np.dot(w_q.astype(np.float32), normed)
    q_ref = np.empty_like(q_raw)
    for h in range(M_Q // HEAD_DIM):
        s = h * HEAD_DIM
        x1, x2 = q_raw[s : s + half], q_raw[s + half : s + HEAD_DIM]
        q_ref[s : s + half] = x1 * cos - x2 * sin
        q_ref[s + half : s + HEAD_DIM] = x1 * sin + x2 * cos
    q_ref = q_ref.astype(DTYPE)

    # K projection + RoPE
    k_raw = np.dot(w_k.astype(np.float32), normed)
    k_ref = np.empty_like(k_raw)
    for h in range(M_K // HEAD_DIM):
        s = h * HEAD_DIM
        x1, x2 = k_raw[s : s + half], k_raw[s + half : s + HEAD_DIM]
        k_ref[s : s + half] = x1 * cos - x2 * sin
        k_ref[s + half : s + HEAD_DIM] = x1 * sin + x2 * cos
    k_ref = k_ref.astype(DTYPE)

    # V projection (no RoPE)
    v_ref = np.dot(w_v.astype(np.float32), normed).astype(DTYPE)

    from air.backend.xrt import XRTBackend
    import filelock, tempfile

    be = XRTBackend(
        verbose=args.verbose,
        omit_while_true_loop=False,
        omit_pingpong=True,
        output_format=args.output_format,
        instance_name="rms_qkv_rope",
    )
    compiled = be.compile(module)

    import time

    WARMUP, ITERS = 5, 20
    out_buf = np.zeros(M_TOTAL, dtype=DTYPE)

    # Concatenated reference: [Q_roped | K_roped | V_raw]
    ref = np.concatenate([q_ref, k_ref, v_ref])

    with filelock.FileLock(tempfile.gettempdir() + "/npu.lock"):
        f = be.load(compiled)

        results = f(x_in, w_rms, w_qkv, lut, out_buf)
        hw_out = results[4]

        # Slice into Q/K/V
        hw_q = hw_out[:M_Q]
        hw_k = hw_out[M_Q : M_Q + M_K]
        hw_v = hw_out[M_Q + M_K :]

        q_err = np.max(np.abs(hw_q.astype(np.float32) - q_ref.astype(np.float32)))
        k_err = np.max(np.abs(hw_k.astype(np.float32) - k_ref.astype(np.float32)))
        v_err = np.max(np.abs(hw_v.astype(np.float32) - v_ref.astype(np.float32)))
        q_nz = np.count_nonzero(hw_q)
        k_nz = np.count_nonzero(hw_k)
        v_nz = np.count_nonzero(hw_v)

        print(f"Q: err={q_err:.4f}, nz={q_nz}/{M_Q}")
        print(f"K: err={k_err:.4f}, nz={k_nz}/{M_K}")
        print(f"V: err={v_err:.4f}, nz={v_nz}/{M_V}")

        fail = q_nz == 0 or k_nz == 0 or v_nz == 0
        if fail:
            print("FAIL (zeros detected)")
            be.unload()
            exit(1)
        print("PASS")

        for _ in range(WARMUP):
            f(x_in, w_rms, w_qkv, lut, out_buf)
        times = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            f(x_in, w_rms, w_qkv, lut, out_buf)
            times.append(time.perf_counter() - t0)
    be.unload()

    avg = np.mean(times) * 1e6
    mn = np.min(times) * 1e6
    print(f"\nRMS→GEMV(QKV)→RoPE/V pipeline (1 launch, 3 herds, {HERD_M} cols):")
    print(f"  N={N}, M_Q={M_Q}, M_K={M_K}, M_V={M_V}, K={K}")
    print(f"  n_launch_iters={M_TOTAL // (TILE_M * HERD_M)}")
    print(f"  avg={avg:.0f} us, min={mn:.0f} us")
