# SPDX-License-Identifier: MIT
"""bf16 A x bfp16ebs8 B GEMM slices for the backbone's stitched ELFs.

matrix_multiplication/bf16_x_bfp16's schedule, with the kernel entry points
suffixed (kernels_bfp16/mm_bfp16.cc) so several tilings share one ELF, and an
optional SwiGLU drain: with `swiglu`, B's columns are block-interleaved per
tile_n (o_ffn_fused_gu.interleave_gate_up) and C is SiLU(gate) * up, n/2 wide.

B is packed on the host by pack_b_bfp16ebs8(w, tile_n, tile_k_l1): shape
[n/tile_n, k/tile_k_l1, bfp_tile_bytes(tile_n, tile_k_l1)] uint8.
"""
import sys
import types
from pathlib import Path

from air import api as air
from air.api import ops
from air.api.types import bf16, f32, i8

_HERE = Path(__file__).resolve().parent

# programming_examples/ is published as the air_examples package rather than
# put on sys.path: every directory under it would otherwise become a
# top-level module name and shadow an installed package that shares it.
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(_HERE.parent.parent.parent)
]


def compile_mm_bfp16(tile_m, tile_n, tile_k_l1, sym_suffix, out_name):
    from air_examples.llms.shared.infra.external_kernels import (
        _PROJ_ROOT,
        _compile_kernel,
    )

    extra = [
        f"-I{_PROJ_ROOT / 'matrix_multiplication' / 'bf16_x_bfp16'}",
        f"-DDIM_M={tile_m}",
        f"-DDIM_N={tile_n}",
        f"-DDIM_K={tile_k_l1}",
        f"-DSYM_SUFFIX={sym_suffix}",
        "-Wno-macro-redefined",
    ]
    _compile_kernel(
        _HERE / "kernels_bfp16" / "mm_bfp16.cc", out_name, extra_flags=extra, force=True
    )


def bfp16_extern_syms(sym_suffix, swiglu=False):
    drain = "@f32_to_bf16_swiglu_mn" if swiglu else "@f32_to_bf16_mn"
    return {
        f"@zero_vectorized_f32_mn{sym_suffix}",
        f"@matmul_bf16_x_bfp16_packed_f32{sym_suffix}",
        drain + sym_suffix,
    }


def bfp16_weight_type(k, n, tile_n, tile_k_l1):
    from air_examples.matrix_multiplication.bf16_x_bfp16.matmul_bf16_x_bfp16 import (
        bfp_tile_bytes,
    )

    return (
        f"memref<{n // tile_n}x{k // tile_k_l1}x{bfp_tile_bytes(tile_n, tile_k_l1)}xi8>"
    )


def build_gemm_bfp16(
    m,
    k,
    n,
    tile_m,
    tile_k_l2,
    tile_k_l1,
    tile_n,
    herd_m,
    herd_n,
    sym_suffix,
    link_with,
    swiglu=False,
    cols_n=False,
):
    """Module text contract as matmul_bf16_x_bfp16.build_module: args (A [m,k]
    bf16, B packed i8, C [m, n or n/2] bf16), one air.launch. The herd's first
    axis is placed along the array columns, each with its own shim DMAs; cols_n
    puts N there (herd_n columns, each streaming its own B slice) instead of M."""
    from air_examples.matrix_multiplication.bf16_x_bfp16.matmul_bf16_x_bfp16 import (
        bfp_tile_bytes,
    )

    r, s, t = 8, 8, 8
    assert m % (tile_m * herd_m) == 0 and n % (tile_n * herd_n) == 0
    assert k % tile_k_l2 == 0 and tile_k_l2 % tile_k_l1 == 0
    assert tile_m % (2 * r) == 0 and tile_n % (2 * t) == 0 and tile_k_l1 % s == 0
    assert (
        not swiglu or tile_n % (4 * t) == 0
    ), "SwiGLU pairs whole 8-column blocks per half"

    tile_bytes = bfp_tile_bytes(tile_n, tile_k_l1)
    k_per_l2 = tile_k_l2 // tile_k_l1
    N_div, K_div = n // tile_n, k // tile_k_l1
    out_tn = tile_n // 2 if swiglu else tile_n
    l2_m, l2_n = tile_m * herd_m, out_tn * herd_n

    A = air.tensor([m, k], bf16)
    B = air.tensor([N_div, K_div, tile_bytes], i8)
    C = air.tensor([m, n // 2 if swiglu else n], bf16)

    zero_acc = air.extern(f"zero_vectorized_f32_mn{sym_suffix}", link_with=link_with)
    matmul = air.extern(
        f"matmul_bf16_x_bfp16_packed_f32{sym_suffix}", link_with=link_with
    )
    drain_fn = air.extern(
        (
            f"f32_to_bf16_swiglu_mn{sym_suffix}"
            if swiglu
            else f"f32_to_bf16_mn{sym_suffix}"
        ),
        link_with=link_with,
    )

    with air.launch(
        [range(m // tile_m // herd_m), range(N_div // herd_n)],
        name="matmul_bf16_x_bfp16",
    ) as launch:

        @launch.body
        def _(li, lj):
            with air.segment(name="matmul_seg") as seg:

                @seg.body
                def _():
                    l2_a = air.alloc(
                        [herd_m, tile_m, tile_k_l2], bf16, scope=seg.private()
                    )
                    l2_b = air.alloc(
                        [herd_n, k_per_l2, tile_bytes], i8, scope=seg.private()
                    )
                    l2_c = air.alloc(
                        [herd_m, herd_n, tile_m, out_tn], bf16, scope=seg.private()
                    )
                    hd = (herd_n, herd_m) if cols_n else (herd_m, herd_n)
                    acc = air.alloc(
                        [*hd, tile_n // t, tile_m // r, r, t], f32, scope=seg.shared()
                    )
                    drain = air.alloc(
                        [*hd, out_tn // t, tile_m // r, r, t], bf16, scope=seg.shared()
                    )

                    row, col = li * l2_m, lj * l2_n
                    n_outer = lj * herd_n

                    def herd():
                        return air.herd(
                            [range(hd[0]), range(hd[1])], name="herd_0", shape=hd
                        )

                    def mn(hx, hy):
                        return (hy, hx) if cols_n else (hx, hy)

                    with herd() as zh:

                        @zh.body
                        def _(hx, hy):
                            zero_acc(acc)

                    for k2 in air.sequential(k // tile_k_l2):
                        k_l2_off = k2 * tile_k_l2
                        k_chunk_off = k2 * k_per_l2
                        ops.load(
                            l2_a,
                            A[
                                row : row + l2_m, k_l2_off : k_l2_off + tile_k_l2
                            ].reshape(herd_m, tile_m, tile_k_l2),
                        )
                        ops.load(
                            l2_b,
                            B[
                                n_outer : n_outer + herd_n,
                                k_chunk_off : k_chunk_off + k_per_l2,
                                :,
                            ],
                        )

                        with herd() as h:

                            @h.body
                            def _(hx, hy):
                                tx, ty = mn(hx, hy)
                                l1_a = air.alloc(
                                    [1, 1, tile_m // r, tile_k_l1 // s, r, s],
                                    bf16,
                                    scope=h.private(),
                                )
                                l1_b = air.alloc([tile_bytes], i8, scope=h.private())
                                for j in air.sequential(k_per_l2):
                                    k1 = j * tile_k_l1
                                    ops.load(
                                        l1_a,
                                        l2_a[tx, :, k1 : k1 + tile_k_l1]
                                        .reshape(
                                            1, 1, tile_m // r, r, tile_k_l1 // s, s
                                        )
                                        .transpose(0, 1, 2, 4, 3, 5),
                                    )
                                    ops.load(l1_b, l2_b[ty, j, :])
                                    matmul(l1_a, l1_b, acc)

                    with herd() as dh:

                        @dh.body
                        def _(hx, hy):
                            tx, ty = mn(hx, hy)
                            drain_fn(acc, drain)
                            ops.store(
                                drain[hx, hy, :, :, :, :].transpose(0, 1, 3, 4, 2, 5),
                                l2_c[tx, ty, :, :],
                            )

                    ops.store(
                        l2_c.transpose(0, 2, 1, 3),
                        C[row : row + l2_m, col : col + l2_n],
                    )

    return launch.build(target="npu2")
