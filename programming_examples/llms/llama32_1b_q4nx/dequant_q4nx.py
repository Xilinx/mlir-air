# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# On-device Q4NX weight dequant (the reference dequant.xclbin mechanism, w=q*scale+min).
# Dequantizes a packed Q4NX weight [N_out, K_in] -> bf16. Herd splits N_out rows
# across compute tiles; each tile dequantizes its rows (dequant_q4nx.cc) one row
# at a time. `transpose_out=True` writes the result as [K_in, N_out] (GEMM
# input-B layout) via a strided output DMA; False writes natural [N_out, K_in]
# (used by the standalone math check).
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from air.ir import (
    AffineConstantExpr,
    AffineExpr,
    AffineMap,
    AffineSymbolExpr,
    IntegerAttr,
    IntegerType,
    MemRefType,
    StringAttr,
    UnitAttr,
)
from air.dialects.affine import apply as affine_apply
from air.dialects.air import (
    MemorySpace,
    T,
    dma_memcpy_nd,
    herd,
    launch,
    module_builder,
    segment,
)
from air.dialects.func import CallOp, FuncOp
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper

GS = 32
KERNEL_OBJ = "dequant_q4nx.o"


def row_bytes(K, gs=GS):
    ng = K // gs
    return K // 2 + ng * 2 + ng * 2  # Q + S(bf16) + MIN(bf16)


@module_builder
def build_module(N_out, K, herd_n=4, transpose_out=True):
    assert N_out % herd_n == 0
    rpt = N_out // herd_n  # rows per tile
    rb = row_bytes(K)
    bf16 = type_mapper(bfloat16)
    u8 = IntegerType.get_signless(8)

    l3_packed = MemRefType.get([N_out, rb], u8)
    l3_out = (
        MemRefType.get([K, N_out], bf16)
        if transpose_out
        else MemRefType.get([K * N_out], bf16)
    )

    l1 = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_packed_ty = MemRefType.get([rb], u8, memory_space=l1)
    l1_row_ty = MemRefType.get([K], bf16, memory_space=l1)

    dfn = FuncOp(
        "dequant_q4nx_bf16", ([l1_packed_ty, l1_row_ty], []), visibility="private"
    )
    dfn.attributes["link_with"] = StringAttr.get(KERNEL_OBJ)
    dfn.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    @FuncOp.from_py_func(l3_packed, l3_out)
    def dequant_q4nx(arg_p, arg_o):
        @launch(operands=[arg_p, arg_o])
        def launch_body(lp, lo):
            @segment(name="seg", operands=[lp, lo])
            def seg_body(sp, so):
                @herd(
                    name="dq_herd",
                    sizes=[1, herd_n],
                    operands=[sp, so],
                    link_with=KERNEL_OBJ,
                )
                def herd_body(_tx, _ty, _sx, _sy, hp, ho):
                    l1p = AllocOp(l1_packed_ty, [], [])
                    l1r = AllocOp(l1_row_ty, [], [])
                    ty_row0 = AffineMap.get(
                        0,
                        1,
                        [
                            AffineExpr.get_mul(
                                AffineSymbolExpr.get(0), AffineConstantExpr.get(rpt)
                            )
                        ],
                    )
                    row0 = affine_apply(ty_row0, [_ty])
                    for r in for_(0, rpt):
                        rg = affine_apply(
                            AffineMap.get(
                                0,
                                2,
                                [
                                    AffineExpr.get_add(
                                        AffineSymbolExpr.get(0), AffineSymbolExpr.get(1)
                                    )
                                ],
                            ),
                            [row0, r],
                        )
                        dma_memcpy_nd(
                            l1p,
                            hp,
                            src_offsets=[rg, 0],
                            src_sizes=[1, rb],
                            src_strides=[rb, 1],
                        )
                        CallOp(dfn, [l1p, l1r])
                        if transpose_out:
                            # write dequant row rg down column rg of [K, N_out]:
                            # out[k, rg] = l1r[k]. Innermost dim size 1/stride 1
                            # (canonical strided transpose BD).
                            dma_memcpy_nd(
                                ho,
                                l1r,
                                dst_offsets=[0, rg],
                                dst_sizes=[K, 1],
                                dst_strides=[N_out, 1],
                            )
                        else:
                            # natural [N_out, K]: row rg -> [rg*K : rg*K+K]
                            off = affine_apply(
                                AffineMap.get(
                                    0,
                                    1,
                                    [
                                        AffineExpr.get_mul(
                                            AffineSymbolExpr.get(0),
                                            AffineConstantExpr.get(K),
                                        )
                                    ],
                                ),
                                [rg],
                            )
                            dma_memcpy_nd(
                                ho,
                                l1r,
                                dst_offsets=[off],
                                dst_sizes=[K],
                                dst_strides=[1],
                            )
                        yield_([])
                    DeallocOp(l1p)
                    DeallocOp(l1r)


class DequantEngine:
    """Compile-once on-device Q4NX dequant for a weight shape [N_out, K].
    run(packed) -> bf16 [N_out, K] (natural). Host transposes to [K,N] for GEMM."""

    def __init__(self, N_out, K, herd_n=4):
        import os
        import subprocess
        from air.backend.xrt import XRTBackend

        self.N_out, self.K = N_out, K
        here = Path(__file__).resolve().parent
        aieopt = os.path.dirname(
            os.path.dirname(
                subprocess.check_output(["which", "aie-opt"]).decode().strip()
            )
        )
        peano = os.environ["PEANO_INSTALL_DIR"]
        # (re)compile the kernel .o for this K (DIM_K baked in) into cwd.
        subprocess.check_call(
            [
                f"{peano}/bin/clang++",
                "-O2",
                "-std=c++20",
                "--target=aie2p-none-unknown-elf",
                "-Wno-parentheses",
                "-Wno-attributes",
                "-Wno-macro-redefined",
                "-Wno-empty-body",
                "-DNDEBUG",
                "-I",
                f"{aieopt}/include",
                "-D__AIE_API_AIE_ADF_HPP__",
                f"-DDIM_K={K}",
                f"-DDIM_GS={GS}",
                "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
                "-c",
                str(here / "dequant_q4nx.cc"),
                "-o",
                "dequant_q4nx.o",
            ]
        )
        mod = build_module(N_out, K, herd_n=herd_n, transpose_out=False)
        self.be = XRTBackend(
            output_format="elf", instance_name="dequant_q4nx", omit_pingpong=True
        )
        self.fn = self.be.load(self.be.compile(mod))

    def run(self, packed):
        out = np.zeros((self.N_out * self.K,), bfloat16)
        res = self.fn(np.ascontiguousarray(packed, np.uint8), out)
        return np.asarray(res[-1]).view(bfloat16).reshape(self.N_out, self.K)

    def close(self):
        try:
            self.be.unload()
        except Exception:
            pass


def pack_weight(q, sc, mn, gs=GS):
    """q [N_out,K] uint8 nibbles, sc/mn [N_out,K/gs] -> packed [N_out, row_bytes] u8."""
    N_out, K = q.shape
    rb = row_bytes(K, gs)
    ng = K // gs
    out = np.zeros((N_out, rb), np.uint8)
    qb = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)  # [N_out, K/2]
    scb = sc.astype(bfloat16)
    mnb = mn.astype(bfloat16)
    qn = K // 2
    for n in range(N_out):
        out[n, :qn] = qb[n]
        out[n, qn : qn + ng * 2] = scb[n].view(np.uint8)
        out[n, qn + ng * 2 : qn + ng * 4] = mnb[n].view(np.uint8)
    return out


if __name__ == "__main__":
    N_out, K = 512, 2048
    np.random.seed(0)
    q = np.random.randint(0, 16, (N_out, K), np.uint8)
    sc = np.random.uniform(0.005, 0.02, (N_out, K // GS)).astype(np.float32)
    mn = np.random.uniform(-0.1, 0.1, (N_out, K // GS)).astype(np.float32)
    ref = (q.astype(np.float32) * np.repeat(sc, GS, 1) + np.repeat(mn, GS, 1)).astype(
        bfloat16
    )
    packed = pack_weight(q, sc, mn)
    mod = build_module(N_out, K, herd_n=4, transpose_out=False)
    runner = XRTRunner(
        omit_pingpong=True, output_format="elf", instance_name="dequant_q4nx"
    )
    sys.exit(
        runner.run_test(
            mod,
            inputs=[packed],
            expected_outputs=[ref.reshape(-1)],
            rtol=0.1,
            atol=0.05,
        )
    )
