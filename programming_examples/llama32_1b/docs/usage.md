# LLAMA-3.2-1B Inference — Usage Guide

## Prerequisites

### Hardware & Toolchain
- AMD NPU2 hardware (Strix, AIE2P)
- MLIR-AIR installed with Peano compiler (`PEANO_INSTALL_DIR` set)
- Python virtualenv from the mlir-air base environment (provides `numpy`,
  `ml_dtypes`, `filelock`, `pyxrt`)

### Extra Python Packages

This example needs four packages on top of the mlir-air base. Install them
with the bundled `requirements.txt`:

```bash
cd programming_examples/llama32_1b
pip install -r requirements.txt
```

This installs:

| Package | Purpose |
|---------|---------|
| `safetensors` | Load HuggingFace weight files |
| `huggingface_hub` | Download the model on first run |
| `transformers` | `AutoTokenizer` (chat template) |
| `torch` | HuggingFace reference model (used by `make verify`) |

### HuggingFace Model Access (one-time setup)

The pipeline uses gated Meta models from HuggingFace. You need to accept
Meta's license and authenticate:

```bash
# 1. Accept the license for each model variant you plan to use (default is instruct):
#      https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct   (MODEL=instruct, default)
#      https://huggingface.co/meta-llama/Llama-3.2-1B            (MODEL=base)
#    Click "Accept" on each page (requires a HuggingFace account; approval is usually instant).

# 2. Create a read token at https://huggingface.co/settings/tokens
#    Then authenticate locally (one of these):
huggingface-cli login              # interactive, paste the token
# or:
export HF_TOKEN=<your-token>       # for non-interactive / CI

# 3. First `make run` auto-downloads weights (~2.5 GB per variant) + tokenizer.
#    Cached at ~/.cache/huggingface/hub/ for subsequent runs.
```

## Quick Start

```bash
cd programming_examples/llama32_1b

# Step 1: Compile all kernels (one-time, ~3 minutes)
make compile

# Step 2: Run inference
make run
```

This runs the full pipeline: NPU prefill (processes prompt) → NPU decode (generates tokens).

---

## Commands

### `make compile`

Compiles all NPU kernels from source and caches them to disk.

- Compiles 6 external C++ kernels (rope, silu, attention, gemv) from `.cc` source
- Compiles 5 prefill ELF kernels via MLIR-AIR/aircc pipeline
- Compiles 3 decode ELF kernels via MLIR-AIR/aircc pipeline
- Results cached in `build_peano/prefill_kernel_cache/` and `build_peano/decode_kernel_cache/`
- Only needed once — subsequent runs use cached kernels via `--run-only`

### `make run`

Runs the unified inference pipeline with default settings (instruct model,
up to 1000 decode tokens — instruct model usually stops earlier on `<|eot_id|>`).

```bash
make run                                   # default prompt + instruct model
make run N_TOKENS=50                       # cap decode at 50 tokens
make run PROMPT="How does a transformer work?"  # custom prompt
make run MODEL=base PROMPT="Once upon a time" N_TOKENS=200  # base completion model
```

**Model variants** (`MODEL=base` or `MODEL=instruct`):
- `instruct` (default): Instruction-following model. Answers questions using chat template.
  Stops generation automatically on `<|eot_id|>` token.
- `base`: Text completion model. Continues the prompt verbatim.
  Same architecture and kernels — only the weights differ.

What happens internally:
1. Loads model weights from HuggingFace cache
2. Pre-loads all weights into NPU buffer objects (one-time setup)
3. **NPU Prefill**: processes entire prompt through 16 transformer layers (~1.27s)
4. **NPU Decode**: generates tokens one at a time (~92ms each)

### `make profile`

Same as `make run` but enables the otherwise-disabled `Profiler` so the
end-to-end inference path is broken down into per-XRT-call and per-CPU-op
wall times. Production code path is identical to `make run`.

```bash
make profile
make profile N_TOKENS=30 PROMPT="Explain photosynthesis in detail."
```

After the model output, the report prints (per phase: prefill / decode):

1. **END-TO-END DATAFLOW** — architecture-aware summary in dataflow order
   (tokenize → eos_pad → embed → 16×(rms_gemms_rope + flash_attn + o_ffn +
   kv_cache_extract) → final_norm → lm_head_gemv → per-query total).
   Mirrors the SVGs in [`PROFILE.html`](detail/PROFILE.html).
2. **Wall-Time Attribution** — totals: NPU XRT vs CPU host ops vs layer-loop.
3. **Per-Layer Execution** — one row per prefill layer; aggregated avg/min/max
   per layer across tokens for decode.
4. **NPU XRT Call Breakdown** — each multi-launch ELF, wall time per call.
5. **CPU Op Breakdown** — each tracked CPU host op (embed, kv_cache_extract,
   final_rms_norm, tokenize, eos_pad, decode_attention_cpu).
6. **Fine-Grained NPU Breakdown** — each XRT call split into
   `BO Write` / `NPU Run` / `BO Read` (concept explained in PROFILE.html
   Part C).
7. **Per-Token Wall Trend** (decode only) — token 1 / middle / last wall
   + first→last drift %, so you can spot any KV-cache-growth-driven slowdown.

For reproduction commands + visual dataflow + concept walkthrough see
[`PROFILE.html`](detail/PROFILE.html).

### `make verify` (and `make verify-full`)

Top-k token-level inclusion gate against HuggingFace transformers in **bf16**
(same dtype as NPU). Greedy-decodes a pre-selected prompt set × 32 tokens; at
each step, both runners' chosen tokens must appear in the OTHER side's top-5.
Pass/fail signal for end-to-end production correctness. Mirrors vLLM's
`check_logprobs_close` method.

```bash
make verify                     # 2 prompts (fast CI gate, ~2 min)
make verify-full                # full 8-prompt sweep (~6 min)
make verify MODEL=base          # base checkpoint, continuation prompts
```

`make verify` runs the first 2 prompts from the model's prompt file and is the
default CI gate. `make verify-full` runs every prompt in the file (currently 8)
for exhaustive local validation. Token count and `k` are fixed by the gate
(32 / 5) — not user-tunable.

### `make diagnosis`

Per-layer `ffn_out` cosine + max_abs error vs HF bf16 for a single prompt.
Informational only (never fails the run); reach for it when `make verify`
flags a regression and you need to localize which layer drifted.

```bash
make diagnosis                                            # uses default PROMPT
make diagnosis PROMPT="The capital of France is"
```

See [VERIFICATION.html](detail/VERIFICATION.html) for the full design rationale,
gate criteria, and report layout.

### `make clean`

Removes all build artifacts (compiled kernels, `.o` files, temporary files).
Forces full recompilation on next `make compile`.

```bash
make clean
```

---

## Performance

| Phase | Time | Detail |
|-------|------|--------|
| **Prefill** | 1.27s wall | 2048 tokens, 16 layers, NPU |
| **Decode** | 92ms/token | 10.8 tokens/sec, NPU |
| **End-to-end (per-token)** | ~92 ms after prefill | Steady-state decode rate |

Comparison with IRON:

| | AIR (this) | IRON | Speedup |
|---|---|---|---|
| Prefill | 1.27s | 2.744s | **2.17x** |
| Decode | 92ms/tok | 370ms/tok | **4.0x** |

---

## File Structure

```
llama32_1b/
├── Makefile                        ← Build commands (this guide)
├── llama32_1b_inference.py             ← Unified pipeline: NPU prefill + NPU decode
├── llama32_1b_prefill.py               ← Prefill-only pipeline
├── llama32_1b_decode.py                ← Decode-only pipeline
├── llama32_1b_weights.py               ← Weight loading from safetensors
├── llama32_1b_cpu_helpers.py           ← Small NumPy helpers: rms_norm, attention_reference, softmax
│
├── gemm_builder.py                 ← bf16 GEMM module builder + transform IR
│
├── multi_launch_builder/           ← Multi-launch ELF builders
│   ├── rms_gemms_rope_multi.py     ← Prefill: RMS+QKV+RoPE (6 launches)
│   ├── o_ffn_multi.py              ← Prefill: O+Add+FFN (8 launches)
│   ├── lm_head_multi.py            ← Prefill: LM Head (8 launches)
│   ├── rms_gemv_rope_multi.py      ← Decode: RMS+QKV+RoPE (6 launches)
│   ├── o_gemv_ffn_multi.py         ← Decode: O+FFN (8 launches)
│   └── lm_head_gemv_multi.py       ← Decode: LM Head (8 launches)
│
│   (one level up: ../llama_kernel_builder/ — shared with llama32_1b_int4/)
│   ├── stitching.py                ← MLIR text stitching for multi-launch ELFs
│   ├── cache.py                    ← KernelCache, Profiler, prepare_air_project
│   ├── external_kernels.py         ← C++ kernel compilation (rope, silu, attn, gemv)
│   ├── ffn_swiglu/                 ← SwiGLU MLIR + .cc source
│   └── rope_halfsplit.cc           ← Half-split RoPE C source
├── build_peano/                    ← Build directory (created by make compile)
│   ├── prefill_kernel_cache/       ← Compiled prefill .elf files
│   ├── decode_kernel_cache/        ← Compiled decode .elf files
│   ├── *.o                         ← Compiled C++ kernels
│   └── air_project/                ← Temporary compilation artifacts
│
└── docs/                           ← Documentation
```

---

## Troubleshooting

**"Kernel not found in cache"**: Run `make compile` first, or remove `--run-only` flag.

**NPU lock timeout**: Another process is using the NPU. Check `lsof /tmp/npu.lock`.

**Slow first token**: The NPU enters power-save after ~10s idle. The warmup pass
handles this automatically. If running manually, ensure `prepare_runtime()` is called.

**Wrong results**: Run `make verify` to gate against HuggingFace transformers
bf16 (top-k token inclusion). If verify fails, run `make diagnosis` to
localize which layer drifted. Check that `.o` files are fresh
(`make clean` then `make compile`).
