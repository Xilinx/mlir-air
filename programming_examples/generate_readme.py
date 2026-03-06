#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Generates programming_examples/README.md with an operator dashboard
# whose NPU1/NPU2 status indicators are derived from LIT test files.
#
# Usage:
#   python3 programming_examples/generate_readme.py
#
# Status logic per example directory (scanning *.lit files recursively):
#   - Has a .lit file with REQUIRES matching the NPU target AND no XFAIL → 🟢
#   - Has a .lit file with REQUIRES matching the NPU target AND has XFAIL → 🟡
#   - No .lit file matches the NPU target → ⚪
#
# REQUIRES tag mapping:
#   "ryzen_ai"      → matches BOTH NPU1 and NPU2
#   "ryzen_ai_npu1" → matches NPU1 only
#   "ryzen_ai_npu2" → matches NPU2 only

import os
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# ── Example registry ──────────────────────────────────────────────────
# Category, human-readable name, directory path (relative to
# programming_examples/), and datatype are manually specified here
# because they cannot be reliably auto-detected from LIT files.

EXAMPLES = [
    {
        "category": "Linear Algebra",
        "name": "Matrix Multiplication",
        "path": "matrix_multiplication",
        "datatypes": "bf16, i16, i8",
    },
    {
        "category": "Linear Algebra",
        "name": "Vector-Matrix Multiplication",
        "path": "vector_matrix_multiplication",
        "datatypes": "bf16",
    },
    {
        "category": "Linear Algebra",
        "name": "Matrix-Vector Multiplication",
        "path": "matrix_vector_multiplication/bf16",
        "datatypes": "bf16",
    },
    {
        "category": "Linear Algebra",
        "name": "AXPY",
        "path": "axpy",
        "datatypes": "bf16",
    },
    {
        "category": "Element-wise",
        "name": "Element-wise Add",
        "path": "eltwise_add",
        "datatypes": "f32",
    },
    {
        "category": "Element-wise",
        "name": "Element-wise Add (with L2)",
        "path": "eltwise_add_with_l2",
        "datatypes": "f32",
    },
    {
        "category": "Element-wise",
        "name": "Element-wise Add (bf16)",
        "path": "primitives/vector_examples/vector_add",
        "datatypes": "bf16",
    },
    {
        "category": "Element-wise",
        "name": "Element-wise Mul",
        "path": "primitives/vector_examples/vector_mul",
        "datatypes": "bf16",
    },
    {
        "category": "Activation/Math",
        "name": "SiLU",
        "path": "silu",
        "datatypes": "bf16",
    },
    {
        "category": "Activation/Math",
        "name": "GELU",
        "path": "gelu",
        "datatypes": "bf16",
    },
    {
        "category": "Activation/Math",
        "name": "Softmax",
        "path": "softmax",
        "datatypes": "bf16",
    },
    {
        "category": "Activation/Math",
        "name": "Sine / Cosine",
        "path": "sine_cosine",
        "datatypes": "bf16",
    },
    {
        "category": "Activation/Math",
        "name": "RELU",
        "path": "relu",
        "datatypes": "bf16",
    },
    {
        "category": "Activation/Math",
        "name": "Leaky RELU",
        "path": "leaky_relu",
        "datatypes": "bf16",
    },
    {
        "category": "Activation/Math",
        "name": "Sigmoid",
        "path": "sigmoid",
        "datatypes": "bf16",
    },
    {
        "category": "Activation/Math",
        "name": "Tanh",
        "path": "primitives/vector_examples/vector_tanh",
        "datatypes": "bf16",
    },
    {
        "category": "Normalization",
        "name": "Layer Normalization",
        "path": "layer_norm",
        "datatypes": "bf16",
    },
    {
        "category": "Normalization",
        "name": "RMS Normalization",
        "path": "rms_norm",
        "datatypes": "bf16",
    },
    {
        "category": "Normalization",
        "name": "Weighted RMS Normalization",
        "path": "weighted_rms_norm",
        "datatypes": "bf16",
    },
    {
        "category": "Aggregation",
        "name": "Reduction (Add)",
        "path": "primitives/vector_examples/vector_reduce_add",
        "datatypes": "bf16",
    },
    {
        "category": "Pooling",
        "name": "MaxPool",
        "path": "primitives/vector_examples/vector_reduce_max",
        "datatypes": "bf16",
    },
    {
        "category": "Pooling",
        "name": "AveragePool",
        "path": "average_pool",
        "datatypes": "bf16",
    },
    {
        "category": "LLM Kernels",
        "name": "Multi-Head Attention (LLaMA2)",
        "path": "llama2_mha",
        "datatypes": "bf16",
    },
    {
        "category": "LLM Kernels",
        "name": "SwiGLU",
        "path": "swiglu",
        "datatypes": "bf16",
    },
    {
        "category": "LLM Kernels",
        "name": "FFN SwiGLU (Decode)",
        "path": "ffn_swiglu/decode",
        "datatypes": "bf16",
    },
    {
        "category": "LLM Kernels",
        "name": "FFN SwiGLU (Prefill)",
        "path": "ffn_swiglu/prefill",
        "datatypes": "bf16",
    },
    {
        "category": "LLM Kernels",
        "name": "RoPE (LUT-based)",
        "path": "rope_lut",
        "datatypes": "bf16",
    },
    {
        "category": "LLM Kernels",
        "name": "RoPE (On-chip Sin/Cos)",
        "path": "rope_sincos",
        "datatypes": "bf16",
    },
    {
        "category": "Attention",
        "name": "Flash Attention (Dataflow)",
        "path": "flash_attention/dataflow_based",
        "datatypes": "bf16",
    },
    {
        "category": "Attention",
        "name": "Flash Attention (Kernel Fusion)",
        "path": "flash_attention/kernel_fusion_based",
        "datatypes": "bf16",
    },
    {
        "category": "Attention",
        "name": "Grouped Query Attention (GQA)",
        "path": "flash_attention/kernel_fusion_based",
        "datatypes": "bf16",
    },
    {
        "category": "Data Movement",
        "name": "Passthrough (DMA)",
        "path": "passthrough/passthrough_dma",
        "datatypes": "u8, i8, i16, u16, f32, bf16",
    },
    {
        "category": "Data Movement",
        "name": "Passthrough (Channel)",
        "path": "passthrough/passthrough_channel",
        "datatypes": "u8",
    },
    {
        "category": "Data Movement",
        "name": "Passthrough (Kernel)",
        "path": "passthrough/passthrough_kernel",
        "datatypes": "u8",
    },
    {
        "category": "Data Movement",
        "name": "Shim DMA 2D",
        "path": "shim_dma_2d",
        "datatypes": "i32",
    },
    {
        "category": "Data Movement",
        "name": "Data Transfer Transpose",
        "path": "data_transfer_transpose",
        "datatypes": "u32",
    },
    {
        "category": "Data Movement",
        "name": "Transpose (bf16)",
        "path": "data_transfer_transpose/dma_bf16",
        "datatypes": "bf16",
    },
    {
        "category": "Data Movement",
        "name": "Matrix Scalar Add",
        "path": "matrix_scalar_add",
        "datatypes": "i32",
    },
    {
        "category": "Communication",
        "name": "Channel Examples",
        "path": "channel_examples",
        "datatypes": "i32",
    },
    {
        "category": "Communication",
        "name": "Multi-Segment Examples",
        "path": "multi_segment",
        "datatypes": "i32",
    },
    {
        "category": "Communication",
        "name": "Cascade Reduction",
        "path": "cascade_reduction",
        "datatypes": "i32",
    },
    {
        "category": "Memory",
        "name": "Segment Alloc",
        "path": "segment_alloc",
        "datatypes": "i32",
    },
    {
        "category": "Spatial",
        "name": "Segment Unroll",
        "path": "segment_unroll",
        "datatypes": "i32",
    },
    {
        "category": "Dataflow",
        "name": "Herd Dataflow",
        "path": "herd_dataflow",
        "datatypes": "bf16",
    },
    {
        "category": "Control Flow",
        "name": "Conditional Branching",
        "path": "conditional_branching",
        "datatypes": "i32",
    },
    {
        "category": "CNN",
        "name": "2D Convolution",
        "path": "conv2d",
        "datatypes": "i32",
    },
    {
        "category": "CNN",
        "name": "Bottleneck",
        "path": "bottleneck",
        "datatypes": "bf16",
    },
    {
        "category": "Memory",
        "name": "Shared L1 Buffer",
        "path": "shared_l1",
        "datatypes": "bf16",
    },
    {
        "category": "Quantization",
        "name": "Dequant (AWQ int4\u2192bf16)",
        "path": "dequant_awq",
        "datatypes": "int4/bf16",
    },
    {
        "category": "Primitives",
        "name": "Scalar/Vector Operations",
        "path": "primitives",
        "datatypes": "various",
    },
]


def parse_lit_file(filepath):
    """Extract REQUIRES tags and XFAIL presence from a .lit file."""
    requires_tags = set()
    has_xfail = False
    with open(filepath, "r") as f:
        for line in f:
            m = re.search(r"//\s*REQUIRES:\s*(.+)", line)
            if m:
                tags = [t.strip() for t in m.group(1).split(",")]
                requires_tags.update(tags)
            if re.search(r"//\s*XFAIL:", line):
                has_xfail = True
    return requires_tags, has_xfail


def get_npu_status(example_dir):
    """Scan all .lit files under example_dir and determine NPU1/NPU2 status.

    Returns (npu1_status, npu2_status) where each is one of:
        "pass"  → 🟢  (has matching .lit, no XFAIL)
        "xfail" → 🟡  (has matching .lit, but XFAIL)
        "none"  → ⚪  (no matching .lit)
    """
    npu1_best = "none"
    npu2_best = "none"

    for lit_path in sorted(example_dir.rglob("*.lit")):
        tags, has_xfail = parse_lit_file(lit_path)

        # Determine which NPU targets this .lit file covers
        is_generic = "ryzen_ai" in tags
        is_npu1 = "ryzen_ai_npu1" in tags or is_generic
        is_npu2 = "ryzen_ai_npu2" in tags or is_generic

        status = "xfail" if has_xfail else "pass"

        # "pass" beats "xfail" beats "none"
        if is_npu1:
            if status == "pass" or (status == "xfail" and npu1_best == "none"):
                npu1_best = status if npu1_best != "pass" else "pass"
        if is_npu2:
            if status == "pass" or (status == "xfail" and npu2_best == "none"):
                npu2_best = status if npu2_best != "pass" else "pass"

    return npu1_best, npu2_best


STATUS_EMOJI = {
    "pass": "\U0001f7e2",  # 🟢
    "xfail": "\U0001f7e1",  # 🟡
    "none": "\u26aa",  # ⚪
}


def generate_dashboard_table(base_url=""):
    """Generate the markdown table rows for the operator dashboard."""
    rows = []
    for ex in EXAMPLES:
        example_dir = SCRIPT_DIR / ex["path"]
        if not example_dir.is_dir():
            npu1, npu2 = "none", "none"
        else:
            npu1, npu2 = get_npu_status(example_dir)

        path = ex["path"]
        link = f"{base_url}{path}/"
        row = (
            f'| {ex["category"]} '
            f'| [{ex["name"]}]({link}) '
            f'| {ex["datatypes"]} '
            f"| {STATUS_EMOJI[npu1]} "
            f"| {STATUS_EMOJI[npu2]} "
            f"| [{path}/]({link}) |"
        )
        rows.append(row)
    return rows


def generate_readme(base_url=""):
    """Generate the full README.md content."""
    table_rows = generate_dashboard_table(base_url=base_url)
    table_body = "\n".join(table_rows)

    return f"""\
<!-- This file is auto-generated by generate_readme.py. Do not edit manually. -->

# MLIR-AIR Programming Examples

These programming examples demonstrate how to leverage the AIR design flow with mlir-air Python bindings and the mlir-air intermediate representation (IR) to build applications targeting AI Engines on AMD NPUs.

## Operator Dashboard

| Category | Operation | Datatype(s) | NPU1 | NPU2 | Design Example |
|:---------|:----------|:------------|:----:|:----:|:---------------|
{table_body}

### Status Legend

- \U0001f7e2 Supported and tested
- \U0001f7e1 Work in progress
- \u26aa Not yet supported

**NPU1** = AMD Ryzen AI (Phoenix, AIE2) &nbsp;&nbsp; **NPU2** = AMD Ryzen AI (Strix, AIE2P)

## Getting Started

See the top-level [README](../README.md) for environment setup and build instructions. Once your environment is configured:

```bash
# Example: run matrix multiplication (bf16, 4x4 herd, 512x512x512)
cd matrix_multiplication/bf16
make run4x4

# Print generated MLIR without running
make print
```

Most examples with a `Makefile` support `make run` (compile and execute on hardware) and `make print` (generate MLIR only). Examples without a Makefile can be run directly with Python:

```bash
python3 run.py                    # compile and run (XRTRunner)
python3 run.py --print-module-only  # print IR only
```

## Benchmarking

The [matrix multiplication](matrix_multiplication/) examples include sweep infrastructure for measuring end-to-end latency across problem sizes:

```bash
cd matrix_multiplication/bf16
make sweep4x4    # sweep problem sizes 256-2048 with a 4x4 herd
make profile     # profile a single 1024^3 problem on hardware
```

Sweep results are saved as CSV files for analysis. See the [bf16 README](matrix_multiplication/bf16/README.md) for details on tile size configuration and architecture selection.
"""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate operator dashboard README.")
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "dashboard.md",
        help="Output file path (default: programming_examples/dashboard.md). CI uses --output for GitHub Pages.",
    )
    parser.add_argument(
        "--base-url",
        default="",
        help="Base URL prefix for example links (default: relative links)",
    )
    args = parser.parse_args()

    content = generate_readme(base_url=args.base_url)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content)
    print(f"Generated {args.output}")
