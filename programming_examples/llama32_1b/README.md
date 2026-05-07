# LLAMA-3.2-1B BF16 Inference on AMD NPU2

End-to-end LLAMA-3.2-1B (1B parameter, BF16) inference running on AMD NPU2 (AIE2P) hardware via MLIR-AIR. Supports both prefill (seq_len=2048) and autoregressive decode.

## Performance

| Phase | Time | vs IRON |
|-------|------|---------|
| Prefill (2048 tokens) | 1.27s wall | **2.17x faster** |
| Decode | 92ms/token (10.8 tok/s) | **4.0x faster** |

## Prerequisites

1. **MLIR-AIR base environment** — AMD NPU2 hardware, Peano compiler, the
   project's standard env: `source utils/env_setup.sh ...`

2. **Extra Python packages** for this example (on top of the base):
   ```bash
   pip install -r requirements.txt
   ```
   This installs `safetensors`, `huggingface_hub`, `transformers`, and `torch`
   (the last is used by `make verify` for HuggingFace reference comparison).

3. **HuggingFace model access** (one-time setup):
   - Accept Meta's license for the gated models you plan to use:
     - https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct  (default; `MODEL=instruct`)
     - https://huggingface.co/meta-llama/Llama-3.2-1B           (`MODEL=base`)
   - Create a read token at https://huggingface.co/settings/tokens
   - Authenticate locally:
     ```bash
     huggingface-cli login   # or: export HF_TOKEN=<your-token>
     ```
   - Weights (~2.5 GB per model) are auto-downloaded on the first `make run`
     and cached under `~/.cache/huggingface/hub/`.

## Quick Start

```bash
# One-time: compile all kernels (~3 min, cached to disk)
make compile

# Run inference (instruct model by default; up to 1000 tokens, stops early on EOT)
make run

# Run with a custom prompt
make run PROMPT="How does photosynthesis work?"

# Run base (completion) model with a longer context
make run MODEL=base PROMPT="In 1969, the first man to walk on" N_TOKENS=200

# Run with profiling breakdown
make profile

# Run with correctness verification
make verify
```

## Documentation

| Doc | What's in it |
|-----|-------------|
| [Usage Guide](docs/usage.md) | All `make` targets, command-line options, file structure |
| [Performance Profile](docs/profile.md) | Kernel timing breakdown, BO categories, memory model |
| [Implementation Guide](docs/explain.md) | How kernels are built, compiled, and stitched together |
| [Known Issues](docs/issues.md) | BF16 precision, fixed seq_len, no sampling |

## Architecture Overview

Each transformer layer runs as 3 NPU invocations (prefill) or 3 invocations (decode):

```
Prefill (per layer):
  rms_gemms_rope  (6 launches) → flash_attn (1 launch) → o_ffn (8 launches)

Decode (per token, per layer):
  rms_gemv_rope   (6 launches) → CPU attention → o_gemv_ffn (8 launches)
```

Multiple operations are fused into single ELF binaries via multi-launch merging,
reducing XRT dispatch overhead from 10 calls/layer to 3.

## Key Files

| File | Purpose |
|------|---------|
| `llama32_1b_inference.py` | Unified prefill + decode pipeline |
| `llama32_1b_prefill.py` | Standalone prefill (with profiler report) |
| `llama32_1b_decode.py` | Standalone decode |
| `llama32_1b_weights.py` | Weight loading from HuggingFace safetensors |
| `llama32_1b_reference.py` | CPU F32 reference implementation |
| `kernel_builder/` | Shared utilities: MLIR stitching, kernel cache, external kernel compilation |
| `multi_launch_builder/` | Multi-launch ELF builders (one per fused kernel) |
| `Makefile` | Build/run/profile/verify targets |
