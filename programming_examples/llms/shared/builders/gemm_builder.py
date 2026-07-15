# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GEMM module builder for NPU2 BF16 matrix multiplication.

Thin llama-side adapter over the contract-split example builders in
`matrix_multiplication/bf16_in_{fp32,bf16}_out/run.py`. The direct-codegen
transform lives there (a single definition per dtype, reused via
`build_module_lowered`); this file no longer keeps its own copy.
"""

from ml_dtypes import bfloat16

# External-bf16 high-precision methods (both = f32 accumulate + single epilogue
# cast = 9.3e-3). They differ only in HOW the cast is done, which fixes tile_m:
#   - fused-cast: external GEMM (f32 scratch) + separate cast launch, tile_m=64.
#                 Faster on large shapes (M*K*N >= 4e9). Links mm_m64.o (symbols _m64).
#   - drain:      in-GEMM drain-herd cast, single launch, tile_m=32 (L1 ceiling).
#                 Better on small/thin shapes. Links mm_m32.o (symbols _m32).
# Distinct symbol suffixes + .o names let BOTH variants coexist in one fused ELF.
_FUSED_CAST_SUFFIX = "_m64"
_FUSED_CAST_OBJ = "mm_m64.o"
_FUSED_CAST_TILE_M = 64
_DRAIN_SUFFIX = "_m32"
_DRAIN_OBJ = "mm_m32.o"
_DRAIN_TILE_M = 32


def gemm_registry_config(m, k, n, output_dtype="bf16", precision="high"):
    """Full per-shape build recipe from the registry: the chosen method's spec
    (build_kwargs / suffix / launches) MERGED with the registry tile sizes. This is
    the single entry point llama builders use so tiles + method are never hardcoded.

    Returns the gemm_method_spec dict plus:
      tile_k_l2, tile_k_l1, tile_n : from the registry JSON (tile_m comes from the
                                     method spec — drain=32 / fused-cast=64)
      method                       : the registry-selected method name
    """
    from kernel_registry.registry_lookup import gemm_config

    cfg = gemm_config(m, k, n, output_dtype, precision)
    return _spec_with_tiles(cfg["method"], cfg["tile"])


def _spec_with_tiles(method, tile):
    """Merge a method's build spec with the registry tile (a named dict
    {tile_m, tile_k_l2, tile_k_l1, tile_n}). tile_m is dictated by the method
    (drain=32 / fused=64) and matches spec['tile_m'] (asserted for safety).
    """
    spec = dict(gemm_method_spec(method))
    assert (
        tile["tile_m"] == spec["tile_m"]
    ), f"registry tile_m={tile['tile_m']} != method '{method}' tile_m={spec['tile_m']}"
    spec["method"] = method
    spec["tile_k_l2"] = tile["tile_k_l2"]
    spec["tile_k_l1"] = tile["tile_k_l1"]
    spec["tile_n"] = tile["tile_n"]
    return spec


def gemm_method_spec(method):
    """Reusable per-GEMM method primitive for ELF-merged kernels. Returns a dict
    describing how to build + stitch ONE GEMM by the chosen method, so any GEMM in
    any merged ELF can independently pick drain vs fused-cast (they are two
    implementations of the same bf16-in/bf16-out high-precision GEMM, with distinct
    symbol suffixes + mm.o files so both can co-link in one ELF):

      tile_m         : the forced tile_m (drain=32, fused-cast=64)
      n_launches     : launches this GEMM contributes to the stitched func (drain=1,
                       fused-cast=2 — the GEMM launch + the cast launch)
      needs_f32_scratch : fused-cast needs one extra f32 C-scratch func arg
      sym_suffix / obj  : symbol suffix + mm.o filename for co-linking
      build_kwargs   : kwargs for _build_gemm_module (minus m,k,n,tiles,herd)
    """
    if method == "fused-cast":
        return {
            "tile_m": _FUSED_CAST_TILE_M,
            "n_launches": 2,
            "needs_f32_scratch": True,
            "sym_suffix": _FUSED_CAST_SUFFIX,
            "obj": _FUSED_CAST_OBJ,
            "build_kwargs": {
                "external_fused_cast": True,
                "sym_suffix": _FUSED_CAST_SUFFIX,
                "link_with_name": _FUSED_CAST_OBJ,
            },
        }
    if method == "drain":
        return {
            "tile_m": _DRAIN_TILE_M,
            "n_launches": 1,
            "needs_f32_scratch": False,
            "sym_suffix": _DRAIN_SUFFIX,
            "obj": _DRAIN_OBJ,
            "build_kwargs": {
                "external_bf16_out": True,
                "sym_suffix": _DRAIN_SUFFIX,
                "link_with_name": _DRAIN_OBJ,
            },
        }
    raise ValueError(f"unknown gemm method: {method!r}")


def _build_gemm_module(
    m,
    k,
    n,
    tile_m,
    tile_k_l2,
    tile_k_l1,
    tile_n,
    herd_m=8,
    herd_n=4,
    external_fused_cast=False,
    external_bf16_out=False,
    sym_suffix="",
    link_with_name="mm.o",
):
    """Build a high-precision BF16-in/BF16-out GEMM via the external mm.o microkernel.

    Two methods (both = f32 accumulate + single epilogue cast = GPU-standard 9.3e-3;
    the registry picks which per shape, see gemm_registry_config):
    - external_fused_cast=True: external GEMM writes an f32 C scratch (full tile_m=64)
      then a SEPARATE on-chip cast launch → `@gemm_cast_bf16`, 2 launches, 4 args
      (A, B, C-f32-scratch, D-bf16-out). Faster on large shapes (M*K*N>=4e9).
    - external_bf16_out=True: in-GEMM drain-herd cast, 1 launch, tile_m=32 (the
      tile_m=64 drain overflows L1). Better on small/thin shapes.

    sym_suffix / link_with_name disambiguate the mm.o variant (_m64 fused / _m32
    drain) so both can co-link in one fused ELF.
    """
    if external_fused_cast:
        from matrix_multiplication.bf16_in_bf16_out.run import build_module_gemm_cast

        return build_module_gemm_cast(
            m,
            k,
            n,
            tile_m,
            tile_k_l2,
            tile_k_l1,
            tile_n,
            herd_m,
            herd_n,
            arch="aie2p",
            sym_suffix=sym_suffix,
            link_with_name=link_with_name,
        )

    if external_bf16_out:
        from matrix_multiplication.bf16_in_bf16_out.run import (
            build_module as build_gemm_bf16_ext,
        )

        return build_gemm_bf16_ext(
            m,
            k,
            n,
            tile_m,
            tile_k_l2,
            tile_k_l1,
            tile_n,
            herd_m,
            herd_n,
            bfloat16,
            bfloat16,  # bf16 output: f32 accumulator + single drain cast
            arch="aie2p",
            emit_external_call=True,
            sym_suffix=sym_suffix,
            link_with_name=link_with_name,
        )

    raise ValueError(
        "_build_gemm_module: must set external_fused_cast=True or external_bf16_out=True "
        "(llama GEMM is always the external high-precision path; tiles+method come from "
        "the registry via gemm_registry_config)."
    )
