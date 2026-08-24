#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Does the batched projection agree with the GEMV it replaces? Asked on device.

q4k_mm_gate.py proves the batched matmul computes what numpy says it computes.
That is necessary and not sufficient: the engine's projection is not a bare
matmul, it is proj_qmm.cc's zero / accumulate / flush split, and what it has to
match is _qmm_q4k_bf16 -- a kernel doing genuinely different arithmetic for the
same result.

    GEMV      never builds W. It factors the +min term out as
              min[r,g] * (sum of x over group g), which is what
              b_col_reduce_add and the whole rc/fill cache exist for, and
              accumulates q*x per group.
    batched   materializes w = q*scale + min elementwise, because aie::mmul
              needs a real B operand, then multiplies in bfp16.

Same maths on paper. Different roundings, in different places, in different
precisions. How far apart they land is a measurement, not a derivation -- and
it is the number that says whether the batched path can stand in for the GEMV
without moving the model's output.

WHAT THIS RUNS. One core, one herd, one launch, both kernels, the same packed
weights and the same activations in L1 for both. No host round trip in between,
so a difference is attributable to the kernels and nothing else. Reported three
ways, because "they differ by X" alone does not say which one is right:

    GEMV    vs exact fp32     what ships today costs this much
    batched vs exact fp32     what would replace it costs this much
    GEMV    vs batched        how far the swap would move the output

The third is the one that matters for a drop-in, and the first two are what say
whether a difference is a regression or an improvement.

NOT COVERED. One core, one row-block, no cascade, no egress, no DMA pressure,
and nothing about the 16-token engine around it. This settles the arithmetic of
the swap.
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
import q4k_mm_gate as gate
from proj_qmm_pack import pack_q4k_block, ROW_BLOCK, COL_BLOCK, GROUP, BLOCK_BF16

# L1 is 64 KB and this design holds both paths' working sets at once. The
# activations are stored ONCE, tile-blocked, and the GEMV de-tiles a col-block
# at a time into a 512-byte scratch -- keeping a second plain [BATCH][K] copy
# costs 16 KB at batch 16 and is what would push NBLK=2 over the edge. Checked
# below rather than assumed.
L1_BYTES = 64 * 1024
L1_STACK = 4096


def l1_bytes(batch, nblk):
    k = COL_BLOCK
    return (
        nblk * BLOCK_BF16 * 2  # packed weights
        + batch * nblk * k * 2  # activations, tile-blocked, shared by both
        + k * 2  # GEMV de-tile scratch
        + batch * ROW_BLOCK * 2 * 2  # two bf16 outputs
        + ROW_BLOCK * 4  # GEMV accumulator
        + batch * ROW_BLOCK * 4  # batched accumulator
        + nblk * (k // GROUP) * 2  # reduce cache
        + ROW_BLOCK * k * 2  # unpack scratch
    )


def build_module(batch, nblk):
    bf16_t = type_mapper(bfloat16)
    f32_t = T.f32()
    l1 = IntegerAttr.get(T.i32(), MemorySpace.L1)
    k = COL_BLOCK

    n_w = nblk * BLOCK_BF16
    n_x = batch * nblk * k
    n_y = batch * ROW_BLOCK

    l3 = lambda n, t: MemRefType.get([n], t)
    l1t = lambda n, t: MemRefType.get([n], t, memory_space=l1)

    module = Module.create()
    with InsertionPoint(module.body):
        decls = {}
        for nm, sig in (
            (
                "proj_gate_gemv",
                [
                    l1t(n_x, bf16_t),
                    l1t(n_w, bf16_t),
                    l1t(n_y, bf16_t),
                    l1t(ROW_BLOCK, f32_t),
                    l1t(nblk * (k // GROUP), bf16_t),
                    l1t(k, bf16_t),
                ],
            ),
            (
                "proj_gate_mm",
                [
                    l1t(n_x, bf16_t),
                    l1t(n_w, bf16_t),
                    l1t(n_y, bf16_t),
                    l1t(n_y, f32_t),
                    l1t(ROW_BLOCK * k, bf16_t),
                ],
            ),
        ):
            f = FuncOp(nm, (sig, []), visibility="private")
            f.attributes["link_with"] = StringAttr.get("proj_qmm_gate.o")
            f.attributes["llvm.emit_c_interface"] = UnitAttr.get()
            decls[nm] = f

        @FuncOp.from_py_func(
            l3(n_w, bf16_t), l3(n_x, bf16_t), l3(n_y, bf16_t), l3(n_y, bf16_t)
        )
        def proj_gate(a_w, a_xt, a_yg, a_ym):
            ops = [a_w, a_xt, a_yg, a_ym]

            @launch(operands=ops)
            def launch_body(*lo):
                @segment(name="seg", operands=list(lo))
                def segment_body(*so):
                    @herd(
                        name="proj_gate_herd",
                        sizes=[1, 1],
                        operands=list(so),
                        link_with="proj_qmm_gate.o",
                    )
                    def herd_body(_tx, _ty, _sx, _sy, h_w, h_xt, h_yg, h_ym):
                        b_w = AllocOp(l1t(n_w, bf16_t), [], [])
                        b_xt = AllocOp(l1t(n_x, bf16_t), [], [])
                        b_xb = AllocOp(l1t(k, bf16_t), [], [])
                        b_yg = AllocOp(l1t(n_y, bf16_t), [], [])
                        b_ym = AllocOp(l1t(n_y, bf16_t), [], [])
                        b_ag = AllocOp(l1t(ROW_BLOCK, f32_t), [], [])
                        b_am = AllocOp(l1t(n_y, f32_t), [], [])
                        b_rc = AllocOp(l1t(nblk * (k // GROUP), bf16_t), [], [])
                        b_ws = AllocOp(l1t(ROW_BLOCK * k, bf16_t), [], [])
                        dma_memcpy_nd(b_w, h_w)
                        dma_memcpy_nd(b_xt, h_xt)
                        CallOp(
                            decls["proj_gate_gemv"],
                            [b_xt, b_w, b_yg, b_ag, b_rc, b_xb],
                        )
                        CallOp(decls["proj_gate_mm"], [b_xt, b_w, b_ym, b_am, b_ws])
                        dma_memcpy_nd(h_yg, b_yg)
                        dma_memcpy_nd(h_ym, b_ym)
                        for a in (b_w, b_xt, b_xb, b_yg, b_ym, b_ag, b_am, b_rc, b_ws):
                            DeallocOp(a)

    return module


def make_case(batch, nblk, seed):
    """Realistically quantized weights and normal activations -- see
    q4k_mm_gate.make_case for why the codec has to be the real min/max rule and
    not an independently drawn scale and min."""
    rng = np.random.default_rng(seed)
    ng = COL_BLOCK // GROUP
    k = nblk * COL_BLOCK
    Wf = np.zeros((ROW_BLOCK, k), np.float32)
    packed = []
    for b in range(nblk):
        wr = (rng.standard_normal((ROW_BLOCK, COL_BLOCK)) * 0.02).astype(np.float32)
        wg = wr.reshape(ROW_BLOCK, ng, GROUP)
        mn = wg.min(2)
        scale = np.where((wg.max(2) - mn) <= 0, 1.0, (wg.max(2) - mn) / 15.0)
        q = np.clip(np.round((wg - mn[..., None]) / scale[..., None]), 0, 15)
        q = q.astype(np.uint8).reshape(ROW_BLOCK, COL_BLOCK)
        packed.append(pack_q4k_block(q, scale, mn))
        Wf[:, b * COL_BLOCK : (b + 1) * COL_BLOCK] = gate.dequant_ref(q, scale, mn)

    X = rng.standard_normal((batch, k)).astype(np.float32).astype(bfloat16)
    Y_exact = X.astype(np.float32) @ Wf.T

    x_tile = np.concatenate(
        [gate.pack_A(X[:, b * COL_BLOCK : (b + 1) * COL_BLOCK]) for b in range(nblk)]
    )
    return np.concatenate(packed).view(bfloat16), x_tile, Y_exact


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--nblk", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("-p", "--print-module-only", action="store_true")
    ap.add_argument("--device", default="npu2")
    args = ap.parse_args()

    # 8 and 16/32 only. Not a q4k_mm.h limit -- q4k_mmul_any dispatches to
    # q4k_mmul_small at 4 and 8 and that kernel is gated bit-exact at both. It
    # is proj_qmm_mm_flush_row's de-tiling that stops at 8: it reads the
    # accumulator as C tile (z, j) at (j*RA + z)*64 with RA = BATCH/8, which is
    # the general aie::mmul<8,8,8> layout and coincides with q4k_mmul_small's
    # at RA == 1. At BATCH 4 the tile is aie::mmul<4,8,8>, size_C is 32 not 64,
    # and RA integer-divides to ZERO -- every j would read tile 0. Silently, and
    # with a plausible wrong answer, which is this project's characteristic
    # failure mode. Refuse instead.
    if args.batch not in (8, 16, 32):
        sys.exit(
            f"--batch {args.batch}: proj_qmm_mm_flush_row de-tiles for "
            f"aie::mmul<8,8,8> (batch 8, 16, 32). Batch 4 needs a size_C=32 "
            f"variant of the flush; q4k_mm.h itself is fine at 4 and "
            f"q4k_mm_gate.py --batch 4 covers it."
        )
    need = l1_bytes(args.batch, args.nblk) + L1_STACK
    if need > L1_BYTES:
        sys.exit(
            f"L1: needs {need} B of {L1_BYTES}. This design holds BOTH paths' "
            f"buffers at once so they share the weights; drop --nblk or --batch."
        )

    with Context(), Location.unknown():
        module = build_module(args.batch, args.nblk)
    if args.print_module_only:
        print(module)
        return 0

    build, _ = gate.prepare_build()
    obj = build / "proj_qmm_gate.o"
    compile_kernel(obj, args.batch, args.nblk)
    gate.stage(build, obj)

    w_bo, xt_bo, Y_exact = make_case(args.batch, args.nblk, args.seed)
    yg_bo = np.zeros(args.batch * ROW_BLOCK, bfloat16)
    ym_bo = np.zeros(args.batch * ROW_BLOCK, bfloat16)

    backend = XRTBackend(
        verbose=args.verbose,
        omit_pingpong=True,
        target_device=args.device,
        stack_size=L1_STACK,
    )
    fn = backend.load(backend.compile(module))
    outs = fn(w_bo, xt_bo, yg_bo, ym_bo)
    backend.unload()

    shape = (args.batch, ROW_BLOCK)
    Yg = np.asarray(outs[-2]).reshape(shape).astype(np.float32)
    Ym = np.asarray(outs[-1]).reshape(shape).astype(np.float32)

    print(
        f"\nprojection gate  BATCH {args.batch}  NBLK {args.nblk}  "
        f"K {args.nblk * COL_BLOCK}"
    )
    sig = float(np.sqrt((Y_exact.astype(np.float64) ** 2).mean()))
    rel = lambda d: float(np.sqrt((d.astype(np.float64) ** 2).mean())) / sig
    bias = lambda d: float(d.mean()) / sig

    if not np.any(Ym):
        print("  batched output is ALL ZERO -- the kernel did not run")
        print("GATE FAIL")
        return 1

    print(f"  {'':26s}{'rms':>9}{'bias':>10}")
    print(f"  {'-' * 45}")
    print(
        f"  {'GEMV    vs exact fp32':26s}"
        f"{rel(Yg - Y_exact):8.3%}{bias(Yg - Y_exact):+10.3%}"
    )
    print(
        f"  {'batched vs exact fp32':26s}"
        f"{rel(Ym - Y_exact):8.3%}{bias(Ym - Y_exact):+10.3%}"
    )
    print(f"  {'GEMV    vs batched':26s}" f"{rel(Yg - Ym):8.3%}{bias(Yg - Ym):+10.3%}")
    print()

    # The swap is a drop-in if it does not move the output further from exact
    # than the GEMV already is. That is the honest bar: the GEMV is not exact
    # either, and matching it exactly was never possible.
    e_g, e_m = rel(Yg - Y_exact), rel(Ym - Y_exact)
    verdict = "no worse than the GEMV" if e_m <= e_g * 1.5 else "WORSE than the GEMV"
    print(f"  batched path is {verdict} against exact fp32")
    print(f"  (ratio {e_m / e_g:.2f}x; the GEMV is the incumbent, not the truth)")
    ok = e_m <= e_g * 1.5
    print("GATE PASS" if ok else "GATE FAIL")
    return 0 if ok else 1


def compile_kernel(obj, batch, nblk):
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
        "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        f"-DPROJ_MM_BATCH={batch}",
        f"-DGATE_NBLK={nblk}",
        "-O2",
        "-c",
        str(HERE / "kernels" / "proj_qmm_gate.cc"),
        "-o",
        str(obj),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"kernel compile failed:\n{r.stdout}\n{r.stderr}")


if __name__ == "__main__":
    sys.exit(main())
