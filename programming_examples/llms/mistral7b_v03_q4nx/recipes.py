# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Explicit candidate recipes for unregistered Mistral prefill lengths."""


def install():
    from ..shared.builders import gemm_builder

    original = gemm_builder.gemm_registry_config
    if getattr(original, "_mistral", False):
        return

    def config(m, k, n, output_dtype="bf16", precision="high"):
        try:
            return original(m, k, n, output_dtype, precision)
        except (KeyError, ValueError):
            if (
                m not in (512, 1024, 2048, 4096)
                or (k, n)
                not in ((4096, 4096), (4096, 1024), (4096, 14336), (14336, 4096))
                or output_dtype != "bf16"
                or precision != "high"
            ):
                raise
            return gemm_builder._spec_with_tiles(
                "fused-cast",
                dict(
                    tile_m=64,
                    tile_k_l2=128 if n == 14336 else 256,
                    tile_k_l1=32,
                    tile_n=128,
                ),
            )

    config._mistral = True
    gemm_builder.gemm_registry_config = config
