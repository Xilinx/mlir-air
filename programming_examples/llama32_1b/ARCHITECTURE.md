# LLAMA-3.2-1B BF16 Inference on NPU2

End-to-end LLAMA-3.2-1B BF16 inference (prefill + decode) on AMD NPU2 (AIE2P).

## Status

- **Prefill** (seq_len=2048): **1.27s wall** — **2.17x faster than IRON** (2.744s)
- **Decode**: 92ms/token (10.8 tok/s) — **4.0x faster than IRON** (370ms/token)
- 3 XRT invocations/layer for both prefill and decode; LM Head reuses the
  decode-side 8-partition GEMV ELF for the single-position last-token projection
- Default model is `instruct`; `make run` answers `"What is the capital of France?"` with `"The capital of France is Paris."`

## Model Config

16 layers, emb_dim=2048, n_heads=32, head_dim=64, n_kv_heads=8, hidden_dim=8192, vocab_size=128256, BF16, rope_base=500000.

## Quick Start

```bash
cd programming_examples/llama32_1b
make compile   # One-time kernel compilation (~3 min, cached to disk)
make run       # Run inference (instruct model; up to 1000 tokens, stops on EOT)
make chat      # Interactive REPL: prep runtime once, loop on prompts (streaming)
make profile   # Run with per-kernel timing breakdown
make verify    # Run with CPU reference verification
```

## File Structure

| File | Purpose |
|------|---------|
| `llama32_1b_inference.py` | Unified prefill + decode pipeline (`make run`) |
| `llama32_1b_prefill.py` | Standalone prefill with profiler report |
| `llama32_1b_decode.py` | Standalone decode pipeline |
| `llama32_1b_weights.py` | Weight loading from HuggingFace safetensors + RoPE LUT |
| `llama32_1b_reference.py` | CPU F32 reference implementation |
| `kernel_builder/` | Shared utilities: cache, MLIR stitching, external kernel compilation |
| `multi_launch_builder/` | Multi-launch ELF builders (one per fused kernel) |
| `Makefile` | Build/run/profile/verify targets |

## Architecture

### Per-Layer Kernel Sequence

```
Prefill (per layer, 3 XRT calls):
  rms_gemms_rope.elf (6 launches) → flash_attn.elf (1 launch) → o_ffn.elf (8 launches)

Decode (per token per layer, 3 XRT calls):
  rms_gemv_rope.elf (6 launches) → CPU attention → o_gemv_ffn.elf (8 launches)
```

8 unique kernel configs compiled once via `KernelCache` and cached to disk:
- Prefill: `rms_gemms_rope.elf`, `flash_attn.elf`, `o_ffn.elf`
- Decode: `rms_gemv_rope.elf`, `o_gemv_ffn.elf`, `lm_head_gemv.elf`

### Runtime Flow

```
prepare_runtime()          ← one-time: load weights, pre-load into per-layer BOs
  ↓
run_npu_prefill()          ← 16 layers × 3 kernel calls + Final RMSNorm + LM Head
  ↓
generate() decode loop     ← per token: 16 layers × 3 kernel calls + LM Head GEMV
```

### Key Design Patterns

- **Multi-launch ELF**: Multiple `air.launch` ops in one MLIR module → single `xrt.run()`. Intermediates flow through DDR without CPU round-trip.
- **Text-based MLIR stitching**: Extract func body, rename SSA values with prefix, remap func args, assemble combined module. See `kernel_builder/stitching.py`.
- **Per-layer BOs** (`bo_key=f"kernel_L{layer_idx}"`): Each layer gets dedicated Buffer Objects. Weights written once during pre-load, reused on every inference call.
- **`static_input_indices`**: Skip BO write for weights on non-first calls.
- **`intermediate_indices`**: Skip BO write for buffers the kernel overwrites.
- **External kernel rename**: K=8192 Down GEMV uses `mv_k8192.o` (compiled with `-D` renamed symbols) to coexist with K=2048 GEMVs in one ELF.
- **Seq-first layout**: RoPE + FlashAttention accept `(seq, heads×dim)` natively — zero host transposes.
- **Half-split RoPE kernel**: Custom `rope_halfsplit.cc` matches HuggingFace Llama's rotation convention `(d[i], d[i+32])`. LUT layout is `[cos..., sin...]` (concatenated). Replaces upstream interleaved `rope.cc`. See `docs/explain.md`.

## Documentation

| Doc | Content |
|-----|---------|
| [README.md](README.md) | Quick overview for newcomers |
| [docs/usage.md](docs/usage.md) | All make targets, CLI options, file structure |
| [docs/profile.md](docs/profile.md) | Kernel timing breakdown, BO categories, memory model |
| [docs/explain.md](docs/explain.md) | Compilation pipeline, stitching details, kernel directory map |
| [docs/issues.md](docs/issues.md) | Known issues: BF16 precision, fixed seq_len, no sampling |
