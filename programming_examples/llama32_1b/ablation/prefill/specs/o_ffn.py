# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Concrete KernelGroupSpec for the prefill o_ffn kernel-group.

Mirrors the production stitch-spec in multi_launch_builder/o_ffn_multi.py.
8 sequential launches at seq=2048, emb_dim=2048, hidden_dim=8192:

  L1  o_gemm      [8,4]  attn_out x wo -> proj
  L2  res_add     [8,1]  proj + x_residual -> res1          (2D out)
  L3  ffn_rmsnorm [8,1]  res1 x ffn_norm_w -> normed2
  L4  gate_gemm   [8,4]  normed2 x w_gate -> gate
  L5  up_gemm     [8,4]  normed2 x w_up -> up
  L6  swiglu      [8,1]  SiLU(gate) x up -> swiglu
  L7  down_gemm   [8,4]  swiglu x w_down -> down
  L8  ffn_add     [8,1]  down + res1 -> output              (1D out)

15 merged-func args (slots 0-14); static slots {1,5,7,9,12};
intermediate slots {2,4,6,8,10,11,13,14}.

Slot conventions per sub-launch standalone signatures:
  - gemm:         (A[seq,K], B[K,N], C[seq,N])          weight=1, out=2
  - add_2d_to_2d: (A[seq,d], B[seq,d], C[seq,d])        no weight, out=2
  - rmsnorm:      (x[seq,d], w[d], out[seq,d])           weight=1, out=2
  - swiglu_2d:    (gate[seq,h], up[seq,h], out[seq,h])   no weight, out=2
  - ffn_add:      (A[seq,d], B[seq,d], out[n_total])     no weight, out=2
"""

from ml_dtypes import bfloat16

from specs.kernel_group import SubLaunchSpec, BatonLink, KernelGroupSpec

# ---------------------------------------------------------------------------
# Sub-launch standalone builders
# ---------------------------------------------------------------------------


def _build_o_gemm_standalone():
    """O projection GEMM: attn_out(2048,2048) x wo(2048,2048) -> proj(2048,2048)."""
    from kernel_builder.gemm_builder import _build_gemm_module

    return _build_gemm_module(
        2048,
        2048,
        2048,
        tile_m=64,
        tile_k_l2=256,
        tile_k_l1=32,
        tile_n=64,
        herd_m=8,
        herd_n=4,
    )


def _build_res_add_standalone():
    """Residual add (2D→2D): proj + x_residual -> res1."""
    from multi_launch_builder.o_ffn_multi import _build_add_2d_to_2d

    return _build_add_2d_to_2d(2048, 2048, bfloat16)


def _build_rmsnorm_standalone():
    """FFN RMSNorm (bare herd → wrap in air.launch)."""
    from weighted_rms_norm.weighted_rms_norm import build_module as build_rms
    from kernel_builder.stitching import _wrap_ir_in_launch
    from air.ir import Module

    bare = str(build_rms(2048, 2048, bfloat16, 16, herd_x=8))
    return Module.parse(_wrap_ir_in_launch(bare))


def _build_gateup_gemm_standalone(n):
    """Gate or Up GEMM: normed2(2048,2048) x w(2048,n) -> out(2048,n)."""
    from kernel_builder.gemm_builder import _build_gemm_module

    return _build_gemm_module(
        2048,
        2048,
        n,
        tile_m=64,
        tile_k_l2=64,
        tile_k_l1=32,
        tile_n=128,
        herd_m=8,
        herd_n=4,
    )


def _build_swiglu_standalone():
    """SwiGLU activation: SiLU(gate) * up -> swiglu  (2D memref variant).

    Uses build_module_2d from kernel_builder/ffn_swiglu/silu_and_mul.py.
    Signature: (rows, cols, tile_n, np_dtype_in, herd_x=8, herd_y=1).
    Already wraps in air.launch — no _wrap_ir_in_launch needed.
    Arg slots in standalone: 0=gate, 1=up, 2=out.
    """
    from kernel_builder.ffn_swiglu.silu_and_mul import build_module_2d as build_swiglu

    return build_swiglu(2048, 8192, 4096, bfloat16, herd_x=8, herd_y=1)


def _build_down_gemm_standalone():
    """Down GEMM: swiglu(2048,8192) x w_down(8192,2048) -> down(2048,2048)."""
    from kernel_builder.gemm_builder import _build_gemm_module

    return _build_gemm_module(
        2048,
        8192,
        2048,
        tile_m=64,
        tile_k_l2=256,
        tile_k_l1=32,
        tile_n=64,
        herd_m=8,
        herd_n=4,
    )


def _build_ffn_add_standalone():
    """FFN Add (2D inputs → 1D output): down + res1 -> output[n_total].

    Replicated from the nested _build_add_2d_to_1d() in o_ffn_multi.py
    (that function is defined inline inside build_o_ffn_module and cannot
    be imported directly).

    Arg slots: 0=A (down, 2D), 1=B (res1, 2D), 2=out (1D).
    """
    from air.ir import (
        AffineConstantExpr,
        AffineExpr,
        AffineMap,
        AffineMapAttr,
        AffineSymbolExpr,
        IntegerAttr,
        IntegerType,
        MemRefType,
        VectorType,
        UnitAttr,
        StringAttr,
    )
    from air.dialects.affine import apply as affine_apply
    from air.dialects.air import launch, segment, herd, module_builder
    from air.dialects.memref import (
        collapse_shape as memref_collapse_shape,
        AllocOp,
        DeallocOp,
        subview,
    )
    from air.dialects.func import FuncOp
    from air.dialects.scf import for_, yield_
    from air.dialects import arith
    from air.dialects.vector import transfer_read, transfer_write
    from air.backend.xrt_runner import type_mapper
    from air.dialects.air import MemorySpace

    seq_len = 2048
    emb_dim = 2048
    n_total = seq_len * emb_dim
    total_tiles = 8
    chunk_size = n_total // total_tiles
    tile_n = emb_dim

    @module_builder
    def _build():
        xrt_dtype = type_mapper(bfloat16)
        l3_2d_ty = MemRefType.get([seq_len, emb_dim], xrt_dtype)
        l3_1d_ty = MemRefType.get([n_total], xrt_dtype)
        l1_space = IntegerAttr.get(IntegerType.get_signless(32), MemorySpace.L1)
        l1_ty = MemRefType.get([tile_n], xrt_dtype, memory_space=l1_space)
        vec_ty = VectorType.get([16], xrt_dtype)
        identity_map = AffineMapAttr.get(AffineMap.get_identity(1))

        @FuncOp.from_py_func(l3_2d_ty, l3_2d_ty, l3_1d_ty)
        def eltwise_add(a_2d, b_2d, out_1d):
            @launch(operands=[a_2d, b_2d, out_1d])
            def add_launch(l_a, l_b, l_out):
                a_flat = memref_collapse_shape(l3_1d_ty, l_a, [[0, 1]])
                b_flat = memref_collapse_shape(l3_1d_ty, l_b, [[0, 1]])

                @segment(name="add_seg", operands=[a_flat, b_flat, l_out])
                def add_seg(s_a, s_b, s_out):
                    offset_map = AffineMap.get(
                        0,
                        3,
                        [
                            AffineExpr.get_add(
                                AffineSymbolExpr.get(0),
                                AffineExpr.get_mul(
                                    AffineExpr.get_add(
                                        AffineExpr.get_mul(
                                            AffineSymbolExpr.get(1),
                                            AffineConstantExpr.get(1),
                                        ),
                                        AffineSymbolExpr.get(2),
                                    ),
                                    AffineConstantExpr.get(chunk_size),
                                ),
                            )
                        ],
                    )

                    @herd(
                        name="add_herd",
                        sizes=[8, 1],
                        operands=[s_a, s_b, s_out],
                    )
                    def add_body(_tx, _ty, _sx, _sy, h_a, h_b, h_out):
                        l1_a = AllocOp(l1_ty, [], [])
                        l1_b = AllocOp(l1_ty, [], [])
                        l1_out = AllocOp(l1_ty, [], [])
                        c0 = arith.ConstantOp.create_index(0)
                        cst0 = arith.ConstantOp(xrt_dtype, 0.0)
                        for loop_iv in for_(0, chunk_size, tile_n):
                            offset = affine_apply(offset_map, [loop_iv, _tx, _ty])
                            from air.dialects.air import dma_memcpy_nd

                            dma_memcpy_nd(
                                l1_a,
                                h_a,
                                src_offsets=[offset],
                                src_sizes=[tile_n],
                                src_strides=[1],
                            )
                            dma_memcpy_nd(
                                l1_b,
                                h_b,
                                src_offsets=[offset],
                                src_sizes=[tile_n],
                                src_strides=[1],
                            )
                            for j in for_(0, tile_n, 16):
                                sub_a = subview(l1_a.result, [j], [16], [1])
                                sub_b = subview(l1_b.result, [j], [16], [1])
                                sub_out = subview(l1_out.result, [j], [16], [1])
                                v_a = transfer_read(
                                    vec_ty, sub_a, [c0], identity_map, cst0, [True]
                                )
                                v_b = transfer_read(
                                    vec_ty, sub_b, [c0], identity_map, cst0, [True]
                                )
                                v_sum = arith.addf(v_a, v_b)
                                transfer_write(
                                    None, v_sum, sub_out, [c0], identity_map, [True]
                                )
                                yield_([])
                            dma_memcpy_nd(
                                h_out,
                                l1_out,
                                dst_offsets=[offset],
                                dst_sizes=[tile_n],
                                dst_strides=[1],
                            )
                            yield_([])
                        DeallocOp(l1_a)
                        DeallocOp(l1_b)
                        DeallocOp(l1_out)

    return _build()


# ---------------------------------------------------------------------------
# KernelGroupSpec
# ---------------------------------------------------------------------------

SPEC = KernelGroupSpec(
    name="o_ffn",
    sub_launches=(
        # idx=0: O GEMM — weight at slot 1 (wo), output at slot 2 (proj)
        SubLaunchSpec("o_gemm", _build_o_gemm_standalone, {}, 1, 2),
        # idx=1: Res Add — no weight, output at slot 2 (res1[2D])
        SubLaunchSpec("res_add", _build_res_add_standalone, {}, None, 2),
        # idx=2: FFN RMSNorm — weight at slot 1 (ffn_norm_w), output at slot 2 (normed2)
        SubLaunchSpec("ffn_rmsnorm", _build_rmsnorm_standalone, {}, 1, 2),
        # idx=3: Gate GEMM — weight at slot 1 (w_gate), output at slot 2 (gate)
        SubLaunchSpec("gate_gemm", _build_gateup_gemm_standalone, {"n": 8192}, 1, 2),
        # idx=4: Up GEMM — weight at slot 1 (w_up), output at slot 2 (up)
        SubLaunchSpec("up_gemm", _build_gateup_gemm_standalone, {"n": 8192}, 1, 2),
        # idx=5: SwiGLU — no weight, gate=slot0, up=slot1, output at slot 2
        SubLaunchSpec("swiglu", _build_swiglu_standalone, {}, None, 2),
        # idx=6: Down GEMM — weight at slot 1 (w_down), output at slot 2 (down)
        SubLaunchSpec("down_gemm", _build_down_gemm_standalone, {}, 1, 2),
        # idx=7: FFN Add — no weight, A=slot0 (down), B=slot1 (res1), output at slot 2
        SubLaunchSpec("ffn_add", _build_ffn_add_standalone, {}, None, 2),
    ),
    merged_arg_signature=(
        "attn_out",  # 0  activation input
        "wo",  # 1  weight (static)
        "proj",  # 2  intermediate
        "x_residual",  # 3  activation input
        "res1",  # 4  intermediate  (shared: res_add out + ffn_add B)
        "ffn_norm_w",  # 5  weight (static)
        "normed2",  # 6  intermediate
        "w_gate",  # 7  weight (static)
        "gate",  # 8  intermediate
        "w_up",  # 9  weight (static)
        "up",  # 10 intermediate
        "swiglu",  # 11 intermediate
        "w_down",  # 12 weight (static)
        "down",  # 13 intermediate
        "output",  # 14 intermediate (final 1D output)
    ),
    weight_slots=frozenset({1, 5, 7, 9, 12}),
    intermediate_slots=frozenset({2, 4, 6, 8, 10, 11, 13, 14}),
    output_slots_for_validation=(14,),
    baton_links=(
        # Stitch arg_map verified against o_ffn_multi.py lines 457-465:
        #   L1 {0:0,1:1,2:2}  L2 {0:2,1:3,2:4}  L3 {0:4,1:5,2:6}
        #   L4 {0:6,1:7,2:8}  L5 {0:6,1:9,2:10} L6 {0:8,1:10,2:11}
        #   L7 {0:11,1:12,2:13}  L8 {0:13,1:4,2:14}
        BatonLink(0, 2, 1, 0),  # o_gemm.proj (slot2) -> res_add.A (slot0)
        BatonLink(1, 2, 2, 0),  # res_add.res1 (slot2) -> ffn_rmsnorm.x (slot0)
        BatonLink(2, 2, 3, 0),  # ffn_rmsnorm.normed2 (slot2) -> gate_gemm.x (slot0)
        BatonLink(2, 2, 4, 0),  # ffn_rmsnorm.normed2 (slot2) -> up_gemm.x (slot0)
        BatonLink(3, 2, 5, 0),  # gate_gemm.gate (slot2) -> swiglu.gate (slot0)
        BatonLink(4, 2, 5, 1),  # up_gemm.up (slot2) -> swiglu.up (slot1)
        BatonLink(5, 2, 6, 0),  # swiglu.swiglu (slot2) -> down_gemm.x (slot0)
        BatonLink(6, 2, 7, 0),  # down_gemm.down (slot2) -> ffn_add.A (slot0)
        BatonLink(
            1, 2, 7, 1
        ),  # res_add.res1 (slot2) -> ffn_add.B (slot1)  [residual-of-residual]
    ),
)
