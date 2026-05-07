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

Same as `make run` but prints per-token timing and kernel breakdown.

```bash
make profile
make profile N_TOKENS=10
```

Example output (with `N_TOKENS=10`):
```
NPU prefill done in 1.27s. First token: 12366

Decoding 10 tokens (token 1 to 10)...
  Token 1: id=13, time=92ms
  Token 2: id=1102, time=91ms
  ...
  Token 10: id=578, time=92ms

Generated 10 tokens in 0.92s
Tokens/second: 10.87
Time/token: 92ms
```

### `make verify`

Runs inference and compares every intermediate result against a CPU F32 reference.
Useful for validating correctness after kernel changes.

```bash
make verify N_TOKENS=10
```

Checks:
- Per-layer KV cache correlation (NPU vs CPU)
- Logits correlation at prediction position
- Top-1 token match

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
├── llama32_1b_reference.py             ← CPU F32 reference
│
├── kernel_builder/                 ← Shared kernel infrastructure
│   ├── stitching.py                ← MLIR text stitching for multi-launch ELFs
│   ├── gemm_builder.py             ← GEMM module builder + transform IR
│   ├── cache.py                    ← KernelCache, Profiler, prepare_air_project
│   └── external_kernels.py         ← C++ kernel compilation (rope, silu, attn, gemv)
│
├── multi_launch_builder/           ← Multi-launch ELF builders
│   ├── rms_gemms_rope_multi.py     ← Prefill: RMS+QKV+RoPE (6 launches)
│   ├── o_ffn_multi.py              ← Prefill: O+Add+FFN (8 launches)
│   ├── lm_head_multi.py            ← Prefill: LM Head (8 launches)
│   ├── rms_gemv_rope_multi.py      ← Decode: RMS+QKV+RoPE (6 launches)
│   ├── o_gemv_ffn_multi.py         ← Decode: O+FFN (8 launches)
│   └── lm_head_gemv_multi.py       ← Decode: LM Head (8 launches)
│
├── ffn_swiglu/                     ← SiLU×mul kernel
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

**Wrong results**: Run `make verify` to compare against CPU reference. Check that
`.o` files are fresh (`make clean` then `make compile`).
