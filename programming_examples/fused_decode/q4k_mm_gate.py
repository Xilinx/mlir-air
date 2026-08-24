#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""NUMERIC GATE for the batched q4k matmul (kernels/q4k_mm.h), on device.

WHY THIS EXISTS. q4k_mm.h was written and benchmarked without ever producing a
number. Its correctness rests on a layout derived from two independent sources
-- the packed nibble order q4_k.h's GEMV walks, and the pointer walk
aie::mmul<8,8,8> does over its A/B/C tiles -- and a mistake in either is a
silent wrong answer, not a crash. The kernel uses AIE intrinsics so it will not
run on the host. So: one core, one block, compare against numpy.

THE COMPARISON IS ==, IN EVERY MODE. That is the whole design. A layout bug
does not perturb an answer, it permutes it, so an exact test is both easier to
pass and impossible to argue with -- no tolerance to tune, nothing to hide
behind. Getting there took modelling two things the hardware does and a datasheet
would not have told you (both measured here, see bfp16_ebs8 and bf16_rd):

  * aie::mmul multiplies in a bfp16 block format -- 8 elements share an
    exponent taken from the block max, 7 significant bits each
  * every bf16 rounding on the way, in the unpack and in both operand
    conversions, rounds toward MINUS INFINITY rather than to nearest

Both push the same direction, so whether the error cancels along a contraction
depends on the MEAN of the operands, not on their spread. The gate reports the
bias separately from pass/fail for that reason -- and it is why `random` mode
quantizes a real weight matrix instead of picking a codec at random. With an
independently drawn scale and min the dequantized weights have a large positive
mean and the measured bias is -11% at K=512; with the min/max rule real q4k
uses, the weights are centred and the same kernel measures 1.3% rms with no
bias at all. The kernel is identical in both. Only the data changed.

  --mode exact   scale=1, min=0, nibbles 0..15, small signed integer
                 activations. Everything is exactly representable end to end,
                 so the two models above cannot mask a fault: this is the
                 layout gate.
  --mode random  realistic bf16 scales/mins/activations. Same == bar, and the
                 accuracy cost against exact fp32 is real rather than modelled.
  --mode probe-* diagnostics, not gates. Identity weights and index-carrying
                 activations, so each output position reports which input it
                 read. What to reach for when exact fails.

WHAT IT DOES NOT COVER. One core, no DMA pressure, no cascade, no egress. It
settles the arithmetic and the layout, and nothing about the engine around it.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from air.ir import *
from air.dialects.air import *
from air.dialects.func import FuncOp, CallOp
from air.dialects.memref import AllocOp, DeallocOp
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import type_mapper

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from proj_qmm_pack import pack_q4k_block, ROW_BLOCK, COL_BLOCK, GROUP, BLOCK_BF16

MMUL_S = MMUL_T = 8


def mmul_r(batch):
    """The mmul's row dimension, which is not always 8.

    q4k_mmul_small uses aie::mmul<4,8,8> at batch 4 so that rowA is 1 rather
    than a half tile. That changes size_A and size_C, and therefore the host
    packing -- so the gate has to know which kernel it is testing.
    """
    return 4 if batch < 8 else 8


# ---------------------------------------------------------------- host layouts
#
# The two orders aie::mmul imposes on the buffers around it. Both are read off
# q4k_mmul's pointer walk in q4k_mm.h, not guessed:
#
#   A tile (z, i) at (i*rowA + z)*64, row-major [8 batch][8 contraction]
#   C tile (z, j) at (j*rowA + z)*64, row-major [8 batch][8 weight-row]
#
# A plain [BATCH][KCOL] activation buffer is NOT the A order -- this is the
# "Xt has to arrive tile-blocked" the wire design calls out, done here in numpy
# because there is no memtile in this test to do it with a strided BD.
def pack_A(X):
    """[BATCH, KCOL] -> mmul A tile order."""
    b, k = X.shape
    r = mmul_r(b)
    return (
        X.reshape(b // r, r, k // MMUL_S, MMUL_S)
        .transpose(2, 0, 1, 3)  # [i][z][rr][ss]
        .reshape(-1)
        .copy()
    )


def unpack_C(C, batch, mrows):
    """mmul C tile order -> [BATCH, MROWS]."""
    r = mmul_r(batch)
    return (
        C.reshape(mrows // MMUL_T, batch // r, r, MMUL_T)
        .transpose(1, 2, 0, 3)  # [z][rr][j][tt]
        .reshape(batch, mrows)
        .copy()
    )


def pack_B(W):
    """[MROWS, KCOL] -> mmul B tile order, which is what q4k_unpack_block emits.

    B tile (i, j) at (j*colA + i)*64, row-major [8 contraction][8 weight-row]:
    Wb[(j*colA + i)*64 + s*8 + t] = W[8j + t, 8i + s].
    """
    m, k = W.shape
    return (
        W.reshape(m // MMUL_T, MMUL_T, k // MMUL_S, MMUL_S)  # [j][t][i][s]
        .transpose(0, 2, 3, 1)  # [j][i][s][t]
        .reshape(-1)
        .copy()
    )


def bfp16_ebs8(v):
    """v[..., 8] -> the bfp16 block format aie::mmul actually multiplies in.

    MEASURED, not read off a datasheet: fitted against device output until it
    reproduced 512/512 elements bit-for-bit, then confirmed at other shapes.
    Each group of 8 gets a shared exponent from the group MAX and keeps 7
    significant bits -- and the residue is dropped by ROUNDING TOWARD MINUS
    INFINITY, not to nearest.

    The rounding mode is the part worth knowing. Rounding down costs each
    operand about half an ulp EVERY time, in the same direction, so the error
    it contributes to a dot product is (K/2) * ulp * mean(other operand) --
    linear in the contraction depth, and proportional to the mean rather than
    the magnitude. Centred operands make it vanish; off-centre ones make it
    grow. Measured both ways in --mode random.

    It is also why a tolerance-based comparison against a plain fp32 reference
    can read as "12% wrong" when the kernel is exactly right: the reference was
    modelling the wrong arithmetic. Model it and the comparison goes back to ==.

    Applied to BOTH operands, on groups of 8 along the contraction for A and
    along the weight-row axis for B (the API transposes B before converting).
    """
    mx = np.abs(v).max(-1, keepdims=True)
    e = np.where(mx > 0, np.floor(np.log2(np.maximum(mx, 1e-38))), 0.0)
    u = 2.0 ** (e - 6)
    return np.floor(v / u) * u


def mmul_ref(x_bo, w_bo, batch, mrows, kcol, nblk):
    """Replay q4k_mmul exactly: same tile walk, same block format, same order.

    Accumulation order is kept sequential over the contraction blocks because
    fp32 addition is not associative and the point of this reference is to be
    bit-exact, not merely close.
    """
    r = mmul_r(batch)
    rowA, colA, colB = batch // r, kcol // MMUL_S, mrows // MMUL_T
    C = np.zeros((colB, rowA, r, MMUL_T), np.float32)
    for blk in range(nblk):
        A = x_bo[blk * batch * kcol :][: batch * kcol].reshape(colA, rowA, r, 8)
        B = w_bo[blk * mrows * kcol :][: mrows * kcol].reshape(colB, colA, 8, 8)
        Aq = bfp16_ebs8(A).astype(np.float32)
        Bq = bfp16_ebs8(B.transpose(0, 1, 3, 2)).transpose(0, 1, 3, 2)
        Bq = np.ascontiguousarray(Bq, np.float32)
        for i in range(colA):
            C += np.matmul(Aq[i][None], Bq[:, i][:, None])
    return C.reshape(-1)


def bf16_rd(x):
    """f32 -> bf16 precision, ROUNDING TOWARD MINUS INFINITY.

    Which is what the AIE bf16 pipeline does, measured the same way as
    bfp16_ebs8: fitted against the device's own unpacked weights until it
    reproduced all 8192 of them. Round-to-nearest matches 4890/8192; this
    matches 8192/8192.

    Floor, not truncate-toward-zero. The multiply cannot tell them apart --
    q*scale is never negative -- but the min ADD can, and toward-zero gets
    7583/8192 there. So both roundings in q4k_unpack_step, and both operand
    conversions in the mmul, go the same way: down.

    That direction is the reason this matters beyond bookkeeping. Two biased
    roundings per weight and a third per multiply all push the same way, so the
    error does not cancel over a contraction, it sums.
    """
    xf = np.ascontiguousarray(x, np.float32)
    i = xf.view(np.int32)
    t = i & ~0xFFFF
    t = np.where(((i & 0xFFFF) != 0) & (xf < 0), t + 0x10000, t)
    return t.astype(np.int32).view(np.float32)


def dequant_ref(q, scale, mn):
    """The weight matrix the kernel reconstructs, at its own precision.

    q4k_unpack_step does bf16(bf16(q) * s) + m with s, m bf16, so the reference
    rounds at the same two points -- and in the same direction, see bf16_rd.

    The scale and min conversions are NOT bf16_rd. Those happen on the host, in
    pack_q4k_block, where numpy rounds to nearest; only the two roundings the
    core performs go toward minus infinity. Rounding all four the same way
    looks tidier and is wrong.
    """
    s = np.repeat(scale.astype(bfloat16).astype(np.float32), GROUP, axis=1)
    m = np.repeat(mn.astype(bfloat16).astype(np.float32), GROUP, axis=1)
    assert s.shape == q.shape, (s.shape, q.shape)
    return bf16_rd(bf16_rd(q.astype(np.float32) * s) + m)


# ------------------------------------------------------------------ the design
def build_module(mrows, kcol, batch, nblk):
    bf16_t = type_mapper(bfloat16)
    f32_t = T.f32()
    l1 = IntegerAttr.get(T.i32(), MemorySpace.L1)

    n_packed = nblk * BLOCK_BF16  # bf16 elements of packed weight
    n_x = nblk * batch * kcol
    n_y = batch * mrows
    n_w = mrows * kcol

    l3_packed = MemRefType.get([n_packed], bf16_t)
    l3_x = MemRefType.get([n_x], bf16_t)
    l3_y = MemRefType.get([n_y], f32_t)
    # The unpack scratch comes back too. It is not an output the engine wants,
    # but it splits the gate in half: if W is right and Y is wrong the fault is
    # in q4k_mmul, and if W is wrong it is in q4k_unpack_block. Without it a
    # failure only says "the composition is wrong".
    l3_wout = MemRefType.get([n_w], bf16_t)

    l1_packed = MemRefType.get([n_packed], bf16_t, memory_space=l1)
    l1_x = MemRefType.get([n_x], bf16_t, memory_space=l1)
    l1_y = MemRefType.get([n_y], f32_t, memory_space=l1)
    l1_w = MemRefType.get([n_w], bf16_t, memory_space=l1)

    module = Module.create()
    with InsertionPoint(module.body):
        gate = FuncOp(
            "q4k_mm_gate",
            ([l1_packed, l1_x, l1_y, l1_w], []),
            visibility="private",
        )
        gate.attributes["link_with"] = StringAttr.get("q4k_mm_gate.o")
        gate.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        @FuncOp.from_py_func(l3_packed, l3_x, l3_y, l3_wout)
        def q4k_gate(a_packed, a_x, a_y, a_w):
            @launch(operands=[a_packed, a_x, a_y, a_w])
            def launch_body(l_packed, l_x, l_y, l_w):
                @segment(name="seg", operands=[l_packed, l_x, l_y, l_w])
                def segment_body(s_packed, s_x, s_y, s_w):
                    @herd(
                        name="gate_herd",
                        sizes=[1, 1],
                        operands=[s_packed, s_x, s_y, s_w],
                        link_with="q4k_mm_gate.o",
                    )
                    def herd_body(_tx, _ty, _sx, _sy, h_packed, h_x, h_y, h_w):
                        b_packed = AllocOp(l1_packed, [], [])
                        b_x = AllocOp(l1_x, [], [])
                        b_y = AllocOp(l1_y, [], [])
                        b_w = AllocOp(l1_w, [], [])
                        dma_memcpy_nd(b_packed, h_packed)
                        dma_memcpy_nd(b_x, h_x)
                        CallOp(gate, [b_packed, b_x, b_y, b_w])
                        dma_memcpy_nd(h_y, b_y)
                        dma_memcpy_nd(h_w, b_w)
                        DeallocOp(b_packed)
                        DeallocOp(b_x)
                        DeallocOp(b_y)
                        DeallocOp(b_w)

    return module


# -------------------------------------------------------------------- the data
def make_case(mode, mrows, kcol, batch, nblk, seed):
    """-> (packed bf16 BO, Xt bf16 BO, Y_ref [batch, mrows] f32)."""
    rng = np.random.default_rng(seed)
    ng = kcol // GROUP
    Wf = np.zeros((mrows, nblk * kcol), np.float32)
    packed = []
    probe = mode.startswith("probe")
    for b in range(nblk):
        q = rng.integers(0, 16, (mrows, kcol)).astype(np.uint8)
        if probe:
            # Identity weights over the first MROWS columns, zero beyond, so
            # Y[b, r] == X[b, r] and every output position reports which input
            # it came from. Exactly representable: q is 0/1, scale 1, min 0.
            q = np.zeros((mrows, kcol), np.uint8)
            q[np.arange(mrows), np.arange(mrows)] = 1
            scale = np.ones((mrows, ng), np.float32)
            mn = np.zeros((mrows, ng), np.float32)
        elif mode == "exact":
            # Identity codec: W == q, small integers, exactly representable.
            scale = np.ones((mrows, ng), np.float32)
            mn = np.zeros((mrows, ng), np.float32)
        else:
            # Quantize an actual weight matrix rather than picking a codec at
            # random, because the bias this kernel has depends on the MEAN of
            # the dequantized weights and a random codec does not reproduce it.
            # Independent scale and min give W a large positive mean; real q4k
            # puts min at the group minimum, so W is centred wherever the
            # weights are -- near zero. Same min/max rule as _requant_q4k.
            wr = (rng.standard_normal((mrows, kcol)) * 0.02).astype(np.float32)
            wg = wr.reshape(mrows, ng, GROUP)
            mn = wg.min(2)
            scale = np.where((wg.max(2) - mn) <= 0, 1.0, (wg.max(2) - mn) / 15.0)
            q = np.clip(np.round((wg - mn[..., None]) / scale[..., None]), 0, 15)
            q = q.astype(np.uint8).reshape(mrows, kcol)
        packed.append(pack_q4k_block(q, scale, mn))
        Wf[:, b * kcol : (b + 1) * kcol] = dequant_ref(q, scale, mn)

    if probe:
        # One axis at a time, constant along the other: with identity weights
        # every output position then carries the INDEX of the input it read,
        # which is what pins the tile layout down instead of inferring it.
        rows = np.arange(batch, dtype=np.float32)[:, None] + 1
        cols = np.arange(nblk * kcol, dtype=np.float32)[None, :] % mrows + 1
        X = np.broadcast_to(
            rows if mode == "probe-batch" else cols, (batch, nblk * kcol)
        ).astype(np.float32)
    elif mode == "exact":
        # Signed, so a transposed or mis-strided read cannot pass by symmetry.
        X = rng.integers(-8, 8, (batch, nblk * kcol)).astype(np.float32)
    else:
        X = rng.standard_normal((batch, nblk * kcol)).astype(np.float32)
    Xb = X.astype(bfloat16)
    # The yardstick: what an exact fp32 matmul of the same inputs would give.
    # Not what the device should produce -- see mmul_ref -- but what the
    # device's answer should be judged against for ACCURACY.
    Y_exact = Xb.astype(np.float32) @ Wf.T

    x_bo = np.concatenate(
        [pack_A(Xb[:, b * kcol : (b + 1) * kcol]) for b in range(nblk)]
    )
    packed_bo = np.concatenate(packed).view(bfloat16)
    # Every block in mmul B order. The scratch is overwritten per block so only
    # the LAST one comes back from the device, but having them all on the host
    # is what lets a failing run be replayed in numpy without the device.
    W_bo = np.concatenate(
        [
            pack_B(Wf[:, b * kcol : (b + 1) * kcol])  # already bf16-valued
            for b in range(nblk)
        ]
    )
    # The gate's expectation: the device's own arithmetic, replayed.
    Y_ref = mmul_ref(x_bo.astype(np.float32), W_bo, batch, mrows, kcol, nblk).astype(
        np.float32
    )
    return packed_bo, x_bo, Y_ref, W_bo, Y_exact


# ------------------------------------------------------------------ diagnosis
def diagnose(Y, Y_ref, C_raw, batch, mrows):
    """When it fails, say something more useful than 'mismatch'.

    A layout bug permutes the answer rather than perturbing it, so the useful
    question is not how big the error is but whether the right values are
    present in the wrong places.
    """
    print("\n  diagnosis:")
    got, want = Y.ravel(), Y_ref.ravel()
    n_exact = int(np.count_nonzero(np.isclose(got, want, rtol=1e-6, atol=1e-6)))
    print(f"    {n_exact}/{got.size} elements match in place")
    # Is it a permutation? Same multiset of values, different order.
    if np.allclose(np.sort(got), np.sort(want), rtol=1e-5, atol=1e-4):
        print("    the VALUE MULTISET matches -> pure layout permutation,")
        print("    the arithmetic is right and an index expression is not")
    if np.allclose(Y.T if batch == mrows else Y, Y_ref, rtol=1e-3, atol=1e-3):
        print("    transposing the result matches -> A/B operand roles swapped")
    if np.count_nonzero(got) == 0:
        print("    output is ALL ZERO -> the kernel did not run, or C was")
        print("    never stored (check link_with and the .o next to the design)")
    bad = np.argmax(np.abs(got - want))
    print(
        f"    worst element {bad} (batch {bad // mrows}, row {bad % mrows}): "
        f"got {got[bad]:.6g} want {want[bad]:.6g}"
    )
    print(f"    raw C[0:8]   {np.array2string(C_raw[:8], precision=4)}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--mode",
        choices=("exact", "random", "probe-batch", "probe-row"),
        default="exact",
        help="probe-* are diagnostics, not gates: identity weights plus an "
        "index-carrying activation, so each output position reports which "
        "input it read",
    )
    ap.add_argument("--mrows", type=int, default=ROW_BLOCK)
    ap.add_argument("--kcol", type=int, default=COL_BLOCK)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument(
        "--nblk",
        type=int,
        default=2,
        help="weight blocks accumulated along the contraction; >1 is what "
        "exercises the accumulate path the engine depends on",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rtol", type=float, default=2e-2)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("-p", "--print-module-only", action="store_true")
    ap.add_argument("--device", default="npu2")
    args = ap.parse_args()

    if args.kcol != COL_BLOCK or args.mrows != ROW_BLOCK:
        # pack_q4k_block is written against the 32x256 block; anything else
        # would need the packer generalized first.
        sys.exit(f"packer is fixed at {ROW_BLOCK}x{COL_BLOCK}")
    # L1 is 64 KB and aiecc reports overflow as an opaque "pipeline failed",
    # so check it here where the message can name the buffer that did it.
    need = (
        args.nblk * BLOCK_BF16 * 2  # packed weights
        + args.nblk * args.batch * args.kcol * 2  # activations, tile-blocked
        + args.batch * args.mrows * 4  # f32 accumulator
        + args.mrows * args.kcol * 2  # unpack scratch
        + 4096  # stack
    )
    if need > 64 * 1024:
        sys.exit(
            f"L1: needs {need} B of {64 * 1024}. The activation tile is "
            f"{args.nblk * args.batch * args.kcol * 2} B at batch {args.batch} "
            f"x {args.nblk} blocks; drop --nblk or --batch."
        )
    if args.batch not in (4, 8) and args.batch % 16:
        sys.exit("batch must be 4 or 8 (q4k_mmul_small) or a multiple of 16 (q4k_mmul)")

    with Context(), Location.unknown():
        module = build_module(args.mrows, args.kcol, args.batch, args.nblk)
    if args.print_module_only:
        print(module)
        return 0

    build, obj = prepare_build()
    compile_kernel(obj, args.mrows, args.kcol, args.batch, args.nblk)
    stage(build, obj)

    packed_bo, x_bo, Y_ref, W_bo, Y_exact = make_case(
        args.mode, args.mrows, args.kcol, args.batch, args.nblk, args.seed
    )
    y_bo = np.zeros(args.batch * args.mrows, np.float32)
    w_bo = np.zeros(args.mrows * args.kcol, bfloat16)

    backend = XRTBackend(
        verbose=args.verbose,
        omit_pingpong=True,
        target_device=args.device,
        stack_size=4096,
    )
    compiled = backend.compile(module)
    fn = backend.load(compiled)
    outs = fn(packed_bo, x_bo, y_bo, w_bo)
    backend.unload()

    C_raw = np.asarray(outs[-2], np.float32).ravel()
    W_got = np.asarray(outs[-1]).ravel().astype(np.float32)
    Y = unpack_C(C_raw, args.batch, args.mrows)
    Y_exp = unpack_C(Y_ref, args.batch, args.mrows)

    # A device run costs a couple of minutes; a layout question costs seconds
    # once the raw buffers are on disk. Always save, so a failure can be picked
    # apart offline instead of by re-running.
    np.savez(
        "gate_raw.npz",
        C_raw=C_raw,
        W_got=W_got,
        W_bo=W_bo,
        Y_ref=Y_ref,
        Y_exact=Y_exact,
        x_bo=x_bo.astype(np.float32),
        mrows=args.mrows,
        kcol=args.kcol,
        batch=args.batch,
        nblk=args.nblk,
    )

    print(
        f"\nq4k_mm gate  [{args.mode}]  MROWS {args.mrows} KCOL {args.kcol} "
        f"BATCH {args.batch} NBLK {args.nblk}"
    )
    # The unpack, against the same bf16 rounding model the reference uses.
    W_ref = W_bo[(args.nblk - 1) * args.mrows * args.kcol :]
    n_eq = int(np.count_nonzero(W_got == W_ref))
    w_ulp = np.abs(W_got - W_ref) / np.maximum(np.abs(W_ref), 2.0**-30) * 2**8
    w_ok = n_eq == W_ref.size
    print(
        f"  unpack   {n_eq}/{W_ref.size} bit-identical"
        + ("" if w_ok else f", worst {w_ulp.max():.1f} bf16 ulp")
    )

    # The multiply, against a replay of the device's own arithmetic. This is
    # the gate, and it is == in every mode -- including `random`. Modelling the
    # bfp16 block format (see bfp16_ebs8) is what makes that possible; before
    # it, random mode could only be checked to a tolerance, and a tolerance
    # wide enough to pass is wide enough to hide a real fault.
    ok = np.array_equal(Y, Y_exp)
    print(
        f"  multiply bit-exact vs replayed device arithmetic: {'YES' if ok else 'NO'}"
    )

    # Separately: what the batched path costs in accuracy. Not a pass/fail --
    # the number the engine has to live with.
    err = np.abs(Y_exp - Y_exact)
    sig = float(np.sqrt((Y_exact.astype(np.float64) ** 2).mean()))
    if sig > 0:
        bias = float((Y_exp - Y_exact).mean())
        print(
            f"  vs exact fp32: rms {float(np.sqrt((err**2).mean()))/sig:.3%}"
            f", mean bias {bias/sig:+.3%} of signal"
            f"  [K={args.nblk * args.kcol}]"
        )
        if abs(bias) > 0.3 * float(np.sqrt((err**2).mean())):
            print(
                "    the error is BIASED, not noise -- bfp16 rounds toward -inf,"
                " so it grows with contraction depth, not with its square root"
            )

    if args.mode.startswith("probe"):
        axis = "batch" if args.mode == "probe-batch" else "weight-row"
        print(f"  probe: each C position should carry its {axis} index + 1")
        for t in range(min(4, C_raw.size // 64)):
            print(f"    tile {t}: {np.array2string(C_raw[t*64:t*64+8].astype(int))}")
    if not ok:
        diagnose(Y, Y_exp, C_raw, args.batch, args.mrows)
        if not w_ok:
            print("    the unpack is already wrong -- fix q4k_unpack_block first")
            n = int(np.argmax(W_got != W_ref))
            print(f"    first bad W element {n}: got {W_got[n]} want {W_ref[n]}")
    ok = ok and w_ok
    print("GATE PASS" if ok else "GATE FAIL")
    return 0 if ok else 1


def prepare_build():
    """-> (build dir, object path).

    aircc resolves `link_with` relative to its own working directory and copies
    the design into ./air_project, so the .o has to be BOTH beside the build dir
    and inside air_project, and the process has to run from the build dir. Same
    arrangement dequant_awq's Makefile makes; done here so the gate is one
    command rather than a make target.
    """
    build = HERE / "build_gate"
    (build / "air_project").mkdir(parents=True, exist_ok=True)
    return build, build / "q4k_mm_gate.o"


def stage(build, obj):
    shutil.copy(obj, build / "air_project" / obj.name)
    os.chdir(build)


def compile_kernel(obj, mrows, kcol, batch, nblk, extra=()):
    """Peano-compile q4k_mm_gate.cc with the same flags the bench uses."""
    import bench_q4k_mm as bench

    obj.parent.mkdir(parents=True, exist_ok=True)
    peano, inc = bench._peano(), bench._aie_include()
    cmd = [
        str(peano / "bin" / "clang++"),
        "-std=c++20",
        "--target=aie2p-none-unknown-elf",
        "-Wno-parentheses",
        "-Wno-attributes",
        "-Wno-macro-redefined",
        "-Wno-empty-body",
        "-Wno-deprecated-declarations",
        "-DNDEBUG",
        "-DMODEL_TYPE=LLAMA_3_2_1B",
        "-D__AIE_API_AIE_ADF_HPP__",
        "-I",
        str(inc),
        "-I",
        str(HERE / "kernels"),
        "-I",
        str(HERE / "models"),
        # The 3.9x speedup on the multiply; the bench measures with it and a
        # real build sets it, so the gate has to check the same arithmetic.
        "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        f"-DGATE_MROWS={mrows}",
        f"-DGATE_KCOL={kcol}",
        f"-DGATE_BATCH={batch}",
        f"-DGATE_NBLK={nblk}",
        *extra,
        "-O2",
        "-c",
        str(HERE / "kernels" / "q4k_mm_gate.cc"),
        "-o",
        str(obj),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"kernel compile failed:\n{r.stdout}\n{r.stderr}")


if __name__ == "__main__":
    sys.exit(main())
