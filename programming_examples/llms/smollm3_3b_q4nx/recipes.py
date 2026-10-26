# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Explicit tiles for the padded SmolLM3 FFN shapes.

The shared external GEMM uses BFP16 compute with BF16 output. Build and
runtime install the same shape selection so their scratch-buffer ABI agrees.
"""


def install():
    from air_examples.llms.shared.builders import gemm_builder

    original = gemm_builder.gemm_registry_config
    if getattr(original, "_smollm3", False):
        return

    def config(m, k, n, output_dtype="bf16", precision="high"):
        if (
            m in (512, 1024, 2048, 4096)
            and (k, n) in ((2048, 11264), (11264, 2048))
            and output_dtype == "bf16"
            and precision == "high"
        ):
            # Bound the large-N DMA stride with a small L2 K tile. The
            # linked m64 object has K_L1=32 and N=128 as in the shared path.
            return gemm_builder._spec_with_tiles(
                "fused-cast",
                dict(
                    tile_m=64,
                    tile_k_l2=64 if n == 11264 else 256,
                    tile_k_l1=32,
                    tile_n=128,
                ),
            )
        return original(m, k, n, output_dtype, precision)

    config._smollm3 = True
    gemm_builder.gemm_registry_config = config
