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
        "category": "Element-wise",
        "name": "Element-wise Multiply",
        "path": "eltwise_mul",
        "datatypes": "bf16, f32",
    },
    {
        "category": "Convolution",
        "name": "Conv1D (causal depthwise, k=3)",
        "path": "conv1d_depthwise",
        "datatypes": "bf16",
    },
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
        "name": "Matrix Multiplication (bfp16ebs8)",
        "path": "matrix_multiplication/bfp16",
        "datatypes": "bfp16ebs8",
    },
    {
        "category": "Linear Algebra",
        "name": "AXPY",
        "path": "axpy",
        "datatypes": "bf16, f32",
    },
    {
        "category": "Element-wise",
        "name": "Element-wise Add",
        "path": "eltwise_add",
        "datatypes": "bf16, f32",
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
        "datatypes": "bf16, f32",
    },
    {
        "category": "Activation/Math",
        "name": "Leaky RELU",
        "path": "leaky_relu",
        "datatypes": "bf16, f32",
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
        "category": "Attention",
        "name": "Flash Attention + KV Cache Prefill",
        "path": "flash_attention/kv_cache_prefill",
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
    with open(filepath, "r", encoding="utf-8") as f:
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
    "llama31_8b_q4nx": "FastFlowLM/Llama-3.1-8B-NPU2",
    # Q4_0-quantized on load from an upstream bf16 checkpoint (LiquidAI
    # publishes no pre-quantized bundle), so this points at the source repo
    # rather than at an NPU2 bundle -- and that same repo is the bf16 verify
    # reference, as in the qwen25_3b_q4 / qwen25_7b_q4nx rows.
    "lfm2_1_2b_q4nx": "LiquidAI/LFM2-1.2B",
    "llama32_1b": "meta-llama/Llama-3.2-1B",
    "llama32_1b_int4": "amd/Llama-3.2-1B-Instruct-awq-uint4-asym-g128-bf16-lmhead",
    "llama32_1b_q4nx": "FastFlowLM/Llama-3.2-1B-NPU2",
    "llama32_3b": "meta-llama/Llama-3.2-3B",
    "llama32_3b_q4nx": "FastFlowLM/Llama-3.2-3B-NPU2",
    "phi4_mini_q4nx": "FastFlowLM/Phi4-mini-Instruct-NPU2",
    "qwen25_0_5b": "Qwen/Qwen2.5-0.5B",
    "qwen25_1_5b": "Qwen/Qwen2.5-1.5B",
    "qwen25_3b": "Qwen/Qwen2.5-3B",
    # Q4_0-quantized on the host straight from an upstream bf16 checkpoint (no
    # pre-quantized Qwen bundle exists), so this points at a Qwen repo rather
    # than at a FastFlowLM NPU2 bundle like the Llama/Gemma Q4NX rows. It is the
    # Instruct variant -- the example's own default (MODEL_DEFAULT in
    # qwen25_3b_q4_prefill.py) -- not the base repo the qwen25_3b row above uses.
    "qwen25_3b_q4": "Qwen/Qwen2.5-3B-Instruct",
    # Q4NX, but FastFlowLM has not published a Qwen2.5-7B NPU2 bundle, so the
    # weights are quantized on load from the Instruct checkpoint -- which is
    # also the verify reference, as in the qwen25_3b_q4 row.
    "qwen25_7b_q4nx": "Qwen/Qwen2.5-7B-Instruct",
    "qwen3_0_6b": "Qwen/Qwen3-0.6B",
    "qwen3_1_7b": "Qwen/Qwen3-1.7B",
    "qwen3_4b": "Qwen/Qwen3-4B",
    "qwen3_4b_q4nx": "FastFlowLM/Qwen3-4B-NPU2",
    "qwen3_8b_q4nx": "FastFlowLM/Qwen3-8B-NPU2",
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


def _ctx_label(n):
    """1024 -> '1k', 131072 -> '128k'."""
    return f"{n // 1024}k" if n and n % 1024 == 0 else str(n)


def load_llm_sweep_history(path):
    """Newest point per (model, context) from sweep_history.ndjson, as curves.

    Same reason as load_llm_history: a snapshot makes publishing destructive, so
    a run that swept nothing would empty the curve table.
    """
    if not path or not Path(path).is_file():
        return []
    newest = {}
    verify = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        model, ctx = r.get("model"), r.get("context_len")
        if not model or ctx is None:
            continue
        ts = r.get("timestamp_utc") or r.get("date") or ""
        cur = newest.get((model, ctx))
        # A measured point outranks a failed one at the same context, so a bad
        # run leaves the last real number rather than punching a hole.
        better = r.get("decode_tokens_per_sec") is not None
        if cur is None or (better, ts) >= (cur[0], cur[1]):
            newest[(model, ctx)] = (better, ts, r)
        # Verify is a per-model property of the whole run, so it is tracked
        # across the model's rows rather than taken from whichever point won
        # above -- a stale-but-measured point must not drag a stale verify
        # badge along with it. Strictly newest, with no preference for a
        # non-empty value: unlike a throughput number, a correctness badge that
        # silently goes stale is worse than one that is blank, and this table
        # has no per-row date to reveal the staleness. Blank until the next
        # nightly appends the field is the intended transition.
        if model not in verify or ts >= verify[model][0]:
            verify[model] = (ts, r.get("verify_status", ""))
    curves = {}
    for (model, ctx), (_, _, r) in sorted(newest.items()):
        curves.setdefault(
            model,
            {"model": model, "points": [], "verify_status": verify[model][1]},
        )["points"].append(
            {
                "context_len": ctx,
                "decode_tokens_per_sec": r.get("decode_tokens_per_sec"),
                "ms_per_token": r.get("ms_per_token"),
                "status": r.get("status", ""),
            }
        )
    return [curves[m] for m in sorted(curves)]


def load_llm_sweeps(sweep_path):
    """Read sweep.json (list of per-model tok/s-vs-context curves), or []."""
    if not sweep_path:
        return []
    p = Path(sweep_path)
    if not p.is_file():
        return []
    try:
        recs = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    # The combined artifact is a list of per-model records. A single per-model
    # <model>.sweep.json is a dict and is an easy thing to point this at by
    # mistake; iterating one yields its keys as strings and blows up further
    # down, so reject the shape here and just drop the table.
    if not isinstance(recs, list):
        print(f"warning: {p} is not a list of sweep records; ignoring it")
        return []
    return [r for r in recs if isinstance(r, dict)]


def _sweep_cell(pt):
    """The point's tok/s, or a marker: ✗ for a failure, — otherwise."""
    tps = pt.get("decode_tokens_per_sec")
    if isinstance(tps, (int, float)):
        return f"{tps:.2f}"
    return "—" if pt.get("status", "") in ("", "ok", "expected_fail") else "✗"


def render_llm_sweep(recs, base_url=""):
    """Render decode tok/s against context length, one row per model.

    Models with a sweep are published here instead of in the single-point table:
    decode throughput is dominated by KV streaming and falls by more than 10x
    across this axis, so one number cannot represent them. Cells without a
    number carry a marker for why (see _sweep_cell), and only the markers that
    actually appear are explained below the table.
    """
    if not recs:
        return ""

    ctxs = sorted(
        {
            pt.get("context_len")
            for d in recs
            for pt in d.get("points", []) or []
            if pt.get("context_len")
        }
    )
    if not ctxs:
        return ""

    head = "| Model | " + " | ".join(_ctx_label(c) for c in ctxs) + " | Verify |"
    sep = "|:------|" + "".join("------:|" for _ in ctxs) + ":------:|"

    rows, markers = [], set()
    for d in sorted(recs, key=lambda r: r.get("model", "")):
        by_ctx = {pt.get("context_len"): pt for pt in d.get("points", []) or []}
        cells = []
        for c in ctxs:
            cell = _sweep_cell(by_ctx.get(c) or {})
            if cell in ("—", "✗"):
                markers.add(cell)
            cells.append(cell)
        verify = _VERIFY_EMOJI.get(d.get("verify_status", ""), "")
        rows.append(
            f'| {_llm_model_cell(d.get("model", ""), base_url)} | '
            + " | ".join(cells)
            + f" | {verify} |"
        )

    # Only explain the markers that are actually on the table. A legend for a
    # symbol no cell uses reads as a caveat about the numbers above it.
    legend = "\n\n".join(
        note
        for marker, note in (
            ("—", "— expected failure."),
            ("✗", "✗ unexpected failure."),
        )
        if marker in markers
    )

    return f"""

### Decode throughput vs context (tok/s)

Steady-state decode throughput at increasing KV-cache depth.

The models below reimplement AMD NPU LLM designs originally developed by the
[FastFlowLM](https://github.com/ROCm/FastFlowLM) team, using the higher-level
abstractions of the MLIR-AIR dialect.

{head}
{sep}
{chr(10).join(rows)}

{legend}
"""


def load_llm_prefill_sweep_history(path):
    """Newest point per (model, padded prefill length) from the TTFT series.

    Same merge semantics as load_llm_sweep_history -- a run that swept nothing
    must not empty the table -- against the prefill_len axis instead.
    """
    if not path or not Path(path).is_file():
        return []
    newest, verify = {}, {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        model, plen = r.get("model"), r.get("prefill_len")
        if not model or plen is None:
            continue
        ts = r.get("timestamp_utc") or r.get("date") or ""
        cur = newest.get((model, plen))
        better = r.get("ttft_ms") is not None
        if cur is None or (better, ts) >= (cur[0], cur[1]):
            newest[(model, plen)] = (better, ts, r)
        if model not in verify or ts >= verify[model][0]:
            verify[model] = (ts, r.get("verify_status", ""))
    curves = {}
    for (model, plen), (_, _, r) in sorted(newest.items()):
        curves.setdefault(
            model,
            {"model": model, "points": [], "verify_status": verify[model][1]},
        )["points"].append(
            {
                "prefill_len": plen,
                "ttft_ms": r.get("ttft_ms"),
                "status": r.get("status", ""),
            }
        )
    return [curves[m] for m in sorted(curves)]


def _prefill_sweep_cell(pt):
    """The point's TTFT in ms, or a marker: ✗ for a failure, — otherwise."""
    ttft = pt.get("ttft_ms")
    if isinstance(ttft, (int, float)):
        return f"{ttft:.0f}"
    return "—" if pt.get("status", "") in ("", "ok", "expected_fail") else "✗"


def render_llm_prefill_sweep(recs, base_url=""):
    """Render prefill TTFT against padded prefill length, one row per model.

    The x-axis is the length the prefill ELFs were BUILT for, which is what TTFT
    scales with. It is deliberately not "prompt length": these prefills pad the
    prompt to the built length and read the last real token's row, so at a fixed
    engine a 128-token prompt and a 2048-token one cost the same (measured on
    llama32_1b_q4nx: 959 ms vs 983 ms). Each column here is a separate build.
    """
    if not recs:
        return ""

    lens = sorted(
        {
            pt.get("prefill_len")
            for d in recs
            for pt in d.get("points", []) or []
            if pt.get("prefill_len")
        }
    )
    if not lens:
        return ""

    head = "| Model | " + " | ".join(_ctx_label(l) for l in lens) + " | Verify |"
    sep = "|:------|" + "".join("------:|" for _ in lens) + ":------:|"

    rows, markers = [], set()
    for d in sorted(recs, key=lambda r: r.get("model", "")):
        by_len = {pt.get("prefill_len"): pt for pt in d.get("points", []) or []}
        cells = []
        for l in lens:
            cell = _prefill_sweep_cell(by_len.get(l) or {})
            if cell in ("—", "✗"):
                markers.add(cell)
            cells.append(cell)
        verify = _VERIFY_EMOJI.get(d.get("verify_status", ""), "")
        rows.append(
            f'| {_llm_model_cell(d.get("model", ""), base_url)} | '
            + " | ".join(cells)
            + f" | {verify} |"
        )

    legend = "\n\n".join(
        note
        for marker, note in (
            # Both cases that produce a dash: a length this model was not swept
            # at (no point at all), and one listed in --expect-fail, which
            # sweep_prefill.py publishes as status "expected_fail" with the real
            # cause in `detail`. Anything else that failed is ✗.
            ("—", "— not swept at this length, or an expected failure."),
            ("✗", "✗ unexpected failure."),
        )
        if marker in markers
    )

    return f"""

### Prefill latency vs padded prefill length (TTFT, ms)

Time to first token at increasing padded prefill length. Each column is a
separate build of the prefill ELFs: the prompt is padded to the built length,
so TTFT tracks that length rather than the number of real prompt tokens.

{head}
{sep}
{chr(10).join(rows)}

{legend}
"""


def load_llm_history(path):
    """Newest row per model from the append-only history, in perf.json shape.

    The table is a MERGE, not a snapshot. Rendering it from one run's perf.json
    made every publish destructive: a run that measured nothing replaced the
    whole table with nothing. Keyed per model, a run simply contributes the
    models it measured and leaves the rest at their last known value, dated.
    """
    if not path or not Path(path).is_file():
        return []
    newest, measured = {}, {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        model = r.get("model")
        if not model:
            continue
        ts = r.get("timestamp_utc") or r.get("date") or ""
        if model not in newest or ts >= newest[model][0]:
            newest[model] = (ts, r)
        # A row with no metrics records that a run failed; it should not erase
        # the last numbers we have. Prefer the newest row that measured
        # something -- its older Measured date is the staleness signal.
        if r.get("decode_tokens_per_sec") is not None or r.get("ttft_ms") is not None:
            if model not in measured or ts >= measured[model][0]:
                measured[model] = (ts, r)
    newest.update(measured)
    return [
        {
            "model": model,
            "timestamp_utc": r.get("timestamp_utc"),
            "runner": r.get("runner", ""),
            "verify_status": r.get("verify_status", ""),
            "metrics": {
                "ttft_ms": r.get("ttft_ms"),
                "decode_tokens_per_sec": r.get("decode_tokens_per_sec"),
                "context_len": r.get("context_len"),
            },
            "toolchain": {
                "mlir_air_sha": r.get("air_sha", ""),
                "mlir_aie_hash": r.get("aie_hash", ""),
                "llvm_aie_version": r.get("peano", ""),
            },
        }
        for model, (_, r) in sorted(newest.items())
    ]


def render_llm_benchmark(
    perf_path,
    base_url="",
    perf_history_link="perf-history.html",
    sweep_recs=(),
    prefill_sweep_recs=(),
    history_path=None,
):
    """Render the nightly LLM benchmark section.

    Prefers the append-only history (newest row per model) over a single run's
    perf.json, so publishing cannot delete a model the latest run did not
    measure. perf.json is the fallback for the first run and for local use.
    """
    recs = load_llm_history(history_path)
    if not recs and perf_path and Path(perf_path).is_file():
        try:
            recs = json.loads(Path(perf_path).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            recs = []
    if not recs:
        return ""

    def _fmt(v):
        return "—" if v is None else v

    # Models with a sweep are published in the sweep table below instead; a
    # single near-zero-context point next to a curve invites reading the two as
    # comparable numbers, and they are not.
    # Only a sweep that measured something displaces the scalar row; otherwise a
    # model whose every context failed appears in neither table.
    swept = {
        s.get("model")
        for s in (sweep_recs or ())
        if any(
            p.get("decode_tokens_per_sec") is not None
            for p in s.get("points", []) or ()
        )
    }

    rows = []
    for d in sorted(recs, key=lambda r: r.get("model", "")):
        if d.get("model") in swept:
            continue
        m = d.get("metrics", {})
        verify = _VERIFY_EMOJI.get(d.get("verify_status", ""), "")
        rows.append(
            f'| {_llm_model_cell(d.get("model", ""), base_url)} '
            f'| {_fmt(m.get("context_len"))} '
            f'| {_fmt(m.get("ttft_ms"))} '
            f'| {_fmt(m.get("decode_tokens_per_sec"))} '
            f'| {(d.get("timestamp_utc") or "")[:10] or "—"} '
            f"| {verify} |"
        )

    # Provenance describes the NEWEST row; rows carry their own Measured date
    # because a merged table can hold models from different runs.
    prov = max(
        (r for r in recs if (r.get("toolchain") or {}).get("mlir_air_sha")),
        key=lambda r: r.get("timestamp_utc") or "",
        default=recs[0],
    )
    tc = prov.get("toolchain", {})
    date = max((r.get("timestamp_utc") or "" for r in recs), default="")[:10]
    air = (tc.get("mlir_air_sha") or "")[:7]
    aie = (tc.get("mlir_aie_hash") or "")[:7]
    peano = tc.get("llvm_aie_version") or ""
    provenance = " · ".join(
        s
        for s in [
            f"Last updated {date}" if date else "",
            f"mlir-air {air}" if air else "",
            f"mlir-aie {aie}" if aie else "",
            f"llvm-aie {peano}" if peano else "",
        ]
        if s
    )

    _sweep_table = render_llm_sweep(sweep_recs, base_url=base_url)
    _prefill_table = render_llm_prefill_sweep(prefill_sweep_recs, base_url=base_url)
    table = "\n".join(rows)
    return f"""\

## Nightly LLM Benchmark (NPU2)

End-to-end LLM inference performance on the AMD Ryzen AI 5 PRO 340 (Krackan Point, NPU2) benchmark runner — 2×32 GB DDR5-5600 SODIMM — refreshed nightly. **TTFT** is time to first token (prefill latency); **Decode** is steady-state generation throughput.

| Model | Context | TTFT (ms) | Decode (tok/s) | Measured | Verify |
|:------|--------:|----------:|---------------:|:---------|:------:|
{table}

Verify: \U0001f7e2 pass &nbsp; \U0001f534 fail &nbsp; ⚪ skipped. &nbsp; — not measured.

{_sweep_table}
{_prefill_table}
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


def generate_readme(
    base_url="",
    llm_perf_path=None,
    section="full",
    llm_sweep_path=None,
    llm_history_path=None,
    llm_sweep_history_path=None,
    llm_prefill_sweep_path=None,
    llm_prefill_sweep_history_path=None,
):
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
            llm_perf_path,
            base_url=base_url,
            perf_history_link="perf-history.md",
            sweep_recs=load_llm_sweep_history(llm_sweep_history_path)
            or load_llm_sweeps(llm_sweep_path),
            prefill_sweep_recs=load_llm_prefill_sweep_history(
                llm_prefill_sweep_history_path
            )
            or load_llm_sweeps(llm_prefill_sweep_path),
            history_path=llm_history_path,
        )
        return f"""\
<!-- This file is auto-generated by generate_readme.py. Do not edit manually. -->

# LLMs on NPU

End-to-end decoder-only LLM inference (prefill + autoregressive decode) mapped to the AMD NPU2 in bf16 via MLIR-AIR. Coverage and performance below are refreshed nightly. Per-model details and source are in [`programming_examples/llms/`]({base_url}llms/).
{llm_section}"""

    # section == "full" (legacy single-page dashboard)
    llm_section = render_llm_benchmark(
        llm_perf_path,
        base_url=base_url,
        sweep_recs=load_llm_sweep_history(llm_sweep_history_path)
        or load_llm_sweeps(llm_sweep_path),
        prefill_sweep_recs=load_llm_prefill_sweep_history(
            llm_prefill_sweep_history_path
        )
        or load_llm_sweeps(llm_prefill_sweep_path),
        history_path=llm_history_path,
    )
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
        "--llm-sweep",
        default=None,
        help="Path to the nightly sweep.json (decode tok/s vs context). Models "
        "present here are rendered as a curve and dropped from the single-point "
        "table. Omitted when absent.",
    )
    parser.add_argument(
        "--llm-history",
        default=None,
        help="Path to history.ndjson. Preferred over --llm-perf for the "
        "benchmark table: newest row per model, so a run that did not measure "
        "a model leaves its previous value in place instead of erasing it.",
    )
    parser.add_argument(
        "--llm-sweep-history",
        default=None,
        help="Path to sweep_history.ndjson. Preferred over --llm-sweep, same "
        "reason as --llm-history.",
    )
    parser.add_argument(
        "--llm-prefill-sweep",
        default=None,
        help="Path to the nightly prefill_sweep.json (TTFT vs padded prefill "
        "length). Rendered as its own curve table. Omitted when absent.",
    )
    parser.add_argument(
        "--llm-prefill-sweep-history",
        default=None,
        help="Path to prefill_sweep_history.ndjson. Preferred over "
        "--llm-prefill-sweep, same reason as --llm-history.",
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
        base_url=args.base_url,
        llm_perf_path=args.llm_perf,
        section=args.section,
        llm_sweep_path=args.llm_sweep,
        llm_history_path=args.llm_history,
        llm_sweep_history_path=args.llm_sweep_history,
        llm_prefill_sweep_path=args.llm_prefill_sweep,
        llm_prefill_sweep_history_path=args.llm_prefill_sweep_history,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Explicit encoding: the tables carry the verify/status emoji, and on a
    # Windows checkout the default (cp1252) raises UnicodeEncodeError on them.
    args.output.write_text(content, encoding="utf-8")
    print(f"Generated {args.output}")
