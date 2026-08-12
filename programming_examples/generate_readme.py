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

import json
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
        "name": "Matrix-Vector Multiplication (Cascade)",
        "path": "matrix_vector_multiplication/bf16_cascade",
        "datatypes": "bf16",
    },
    {
        "category": "Linear Algebra",
        "name": "Matrix-Vector Multiplication (AWQ int4)",
        "path": "matrix_vector_multiplication/int4_awq",
        "datatypes": "int4 weights / bf16 activations",
    },
    {
        "category": "Linear Algebra",
        "name": "Matrix Multiplication (AWQ int4)",
        "path": "matrix_multiplication/int4_awq",
        "datatypes": "int4 weights / bf16 activations",
    },
    {
        "category": "Linear Algebra",
        "name": "Matrix Multiplication (bf16 x bfp16ebs8)",
        "path": "matrix_multiplication/bf16_x_bfp16",
        "datatypes": "bf16 activations / bfp16ebs8 weights",
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
        "name": "Attention (Decode)",
        "path": "attention_decode",
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
        "category": "LLM Kernels",
        "name": "Fused Superkernel Decode",
        "path": "fused_decode",
        "datatypes": "Q4NX weights / bf16 activations",
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
        "name": "3D Channel with Segment Unroll",
        "path": "channel_examples/channel_3d_segment_unroll",
        "datatypes": "i32",
    },
    {
        "category": "Communication",
        "name": "Broadcast Selective Capture",
        "path": "channel_examples/broadcast_selective_capture",
        "datatypes": "i32",
    },
    {
        "category": "Communication",
        "name": "Dual-Herd Packet Switch",
        "path": "channel_examples/dual_herd_packet_switch",
        "datatypes": "bf16",
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
        "name": "Conv2d 14x14",
        "path": "conv2d_14x14",
        "datatypes": "ui8/i8",
    },
    {
        "category": "CNN",
        "name": "Bottleneck",
        "path": "bottleneck",
        "datatypes": "bf16",
    },
    {
        "category": "ML Pipeline",
        "name": "MNIST-FC (Broadcast Bias Add)",
        "path": "mnist_fc/broadcast_bias_add",
        "datatypes": "f32",
    },
    {
        "category": "ML Pipeline",
        "name": "MNIST-FC (ReLU 2D)",
        "path": "mnist_fc/relu",
        "datatypes": "f32/bf16",
    },
    {
        "category": "ML Pipeline",
        "name": "MNIST-FC (Argmax)",
        "path": "mnist_fc/argmax",
        "datatypes": "f32\u2192i32",
    },
    {
        "category": "ML Pipeline",
        "name": "MNIST-FC (Integration)",
        "path": "mnist_fc/integration",
        "datatypes": "f32",
    },
    {
        "category": "Memory",
        "name": "Shared L1 Buffer (Multi-Herd)",
        "path": "shared_l1_multi_herd",
        "datatypes": "bf16",
    },
    {
        "category": "Memory",
        "name": "Shared L1 Buffer (Single-Herd)",
        "path": "shared_l1_single_herd",
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


_VERIFY_EMOJI = {"pass": "\U0001f7e2", "fail": "\U0001f534", "skip": "⚪"}

# ── LLM model registry ────────────────────────────────────────────────
# Maps the perf.json "model" key (which matches the directory name under
# programming_examples/llms/) to the canonical HuggingFace repo id it
# implements. Specified manually — like EXAMPLES above — because the HF id
# cannot be reliably auto-detected (several models reference both base and
# instruct variants, and some load weights indirectly). Keep in sync with
# programming_examples/llms/hf_models.txt.
LLM_HF_MODELS = {
    "gemma3_4b_q4nx": "FastFlowLM/Gemma3-4B-NPU2",
    "llama32_1b": "meta-llama/Llama-3.2-1B",
    "llama32_1b_int4": "amd/Llama-3.2-1B-Instruct-awq-uint4-asym-g128-bf16-lmhead",
    "llama32_1b_q4nx": "FastFlowLM/Llama-3.2-1B-NPU2",
    "llama32_3b": "meta-llama/Llama-3.2-3B",
    "llama32_3b_q4nx": "FastFlowLM/Llama-3.2-3B-NPU2",
    "qwen25_0_5b": "Qwen/Qwen2.5-0.5B",
    "qwen25_1_5b": "Qwen/Qwen2.5-1.5B",
    "qwen25_3b": "Qwen/Qwen2.5-3B",
    # Q4_0-quantized on the host straight from an upstream bf16 checkpoint (no
    # pre-quantized Qwen bundle exists), so this points at a Qwen repo rather
    # than at a FastFlowLM NPU2 bundle like the Llama/Gemma Q4NX rows. It is the
    # Instruct variant -- the example's own default (MODEL_DEFAULT in
    # qwen25_3b_q4_prefill.py) -- not the base repo the qwen25_3b row above uses.
    "qwen25_3b_q4": "Qwen/Qwen2.5-3B-Instruct",
    "qwen3_0_6b": "Qwen/Qwen3-0.6B",
    "qwen3_1_7b": "Qwen/Qwen3-1.7B",
    "qwen3_4b": "Qwen/Qwen3-4B",
    "smollm2_1_7b": "HuggingFaceTB/SmolLM2-1.7B",
}


def _llm_model_cell(model, base_url):
    """Render the Model column: name linked to its repo implementation, with a
    trailing (HF) link to the HuggingFace model page when known."""
    if not model:
        return ""
    name_link = f"[{model}]({base_url}llms/{model}/)"
    hf_id = LLM_HF_MODELS.get(model)
    if hf_id:
        return f"{name_link} ([HF](https://huggingface.co/{hf_id}))"
    return name_link


def render_llm_benchmark(perf_path, base_url="", perf_history_link="perf-history.html"):
    """Render the nightly LLM benchmark section from a perf.json file.

    `perf_path` is the JSON produced by nightlyPerfBenchmark.yml (a list of
    per-model records). Returns an empty string when the file is absent or
    empty, so the section is simply omitted (e.g. on local runs or before the
    first nightly has published an artifact).
    """
    if not perf_path:
        return ""
    p = Path(perf_path)
    if not p.is_file():
        return ""
    try:
        recs = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return ""
    if not recs:
        return ""

    def _fmt(v):
        return "—" if v is None else v

    rows = []
    for d in sorted(recs, key=lambda r: r.get("model", "")):
        m = d.get("metrics", {})
        verify = _VERIFY_EMOJI.get(d.get("verify_status", ""), "")
        rows.append(
            f'| {_llm_model_cell(d.get("model", ""), base_url)} '
            f'| {_fmt(m.get("context_len"))} '
            f'| {_fmt(m.get("ttft_ms"))} '
            f'| {_fmt(m.get("decode_tokens_per_sec"))} '
            f"| {verify} |"
        )

    # Provenance: all rows in a run share one toolchain/date, but a partially
    # written artifact may leave some rows' fields blank. Take the date from the
    # newest timestamp present, and the toolchain/runner from the newest record
    # that actually carries toolchain info (fall back to the first record).
    prov = max(
        (r for r in recs if (r.get("toolchain") or {}).get("mlir_air_sha")),
        key=lambda r: r.get("timestamp_utc") or "",
        default=recs[0],
    )
    tc = prov.get("toolchain", {})
    date = max((r.get("timestamp_utc") or "" for r in recs), default="")[:10]
    runner = prov.get("runner", "")
    air = (tc.get("mlir_air_sha") or "")[:7]
    aie = (tc.get("mlir_aie_hash") or "")[:7]
    peano = tc.get("llvm_aie_version") or ""
    provenance = " · ".join(
        s
        for s in [
            f"Last updated {date}" if date else "",
            f"runner {runner}" if runner else "",
            f"mlir-air {air}" if air else "",
            f"mlir-aie {aie}" if aie else "",
            f"llvm-aie {peano}" if peano else "",
        ]
        if s
    )

    table = "\n".join(rows)
    return f"""\

## Nightly LLM Benchmark (NPU2)

End-to-end LLM inference performance on the AMD Ryzen AI (Krackan Point, NPU2) benchmark runner, refreshed nightly. **TTFT** is time to first token (prefill latency); **Decode** is steady-state generation throughput.

| Model | Context | TTFT (ms) | Decode (tok/s) | Verify |
|:------|--------:|----------:|---------------:|:------:|
{table}

\U0001f7e2 verify passed &nbsp; \U0001f534 verify failed &nbsp; ⚪ verify skipped &nbsp; — metric not captured (e.g. the model's profile run failed)

\U0001f4c8 [Performance history over time]({perf_history_link}) — per-nightly TTFT and decode throughput plotted per model.

_{provenance}_

"""


def _operator_dashboard(base_url=""):
    """The operator dashboard table + status legend (no LLM section)."""
    table_body = "\n".join(generate_dashboard_table(base_url=base_url))
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
"""


def _getting_started_footer():
    return """\
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
"""


def generate_readme(base_url="", llm_perf_path=None, section="full"):
    """Generate dashboard content.

    section:
      - "full"      : operator dashboard + LLM benchmark + getting-started (legacy).
      - "operators" : operator dashboard + status legend only (MkDocs Programming
                      Examples page).
      - "llm"       : a standalone "LLMs on NPU" page \u2014 intro + the live nightly
                      benchmark table (MkDocs LLMs overview). The benchmark's
                      Performance History link targets the sibling MkDocs page.
    """
    if section == "operators":
        return _operator_dashboard(base_url=base_url)

    if section == "llm":
        llm_section = render_llm_benchmark(
            llm_perf_path, base_url=base_url, perf_history_link="perf-history.md"
        )
        return f"""\
<!-- This file is auto-generated by generate_readme.py. Do not edit manually. -->

# LLMs on NPU

End-to-end decoder-only LLM inference (prefill + autoregressive decode) mapped to the AMD NPU2 in bf16 via MLIR-AIR. Model coverage and performance below are refreshed nightly by CI (correctness **verify** plus **TTFT**/**decode** capture). For per-model architecture details and source, see the [`programming_examples/llms/`]({base_url}llms/) directory.
{llm_section}"""

    # section == "full" (legacy single-page dashboard)
    llm_section = render_llm_benchmark(llm_perf_path, base_url=base_url)
    return (
        _operator_dashboard(base_url=base_url) + llm_section + _getting_started_footer()
    )


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
    parser.add_argument(
        "--llm-perf",
        default=None,
        help="Path to the nightly perf.json (LLM benchmark artifact). When "
        "absent or missing, the LLM benchmark section is omitted.",
    )
    parser.add_argument(
        "--section",
        choices=["full", "operators", "llm"],
        default="full",
        help="Which page to emit: full single-page dashboard (default), "
        "operators-only, or the LLMs-on-NPU overview.",
    )
    args = parser.parse_args()

    content = generate_readme(
        base_url=args.base_url, llm_perf_path=args.llm_perf, section=args.section
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content)
    print(f"Generated {args.output}")
