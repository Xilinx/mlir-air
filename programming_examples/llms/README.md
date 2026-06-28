# LLMs on AMD NPU2 (MLIR-AIR)

End-to-end decoder-only LLM inference (prefill + autoregressive decode) mapped
to the AMD NPU2 (AIE2P / Strix) in bf16 via MLIR-AIR. Each model is a
self-contained example that composes registry-validated leaf kernels into fused
multi-launch ELFs and gates correctness against a Hugging Face bf16 reference.

## Supported models

| Model | HF checkpoint | Layers | emb / head_dim / hidden | Attention | Family delta | Status |
|---|---|---|---|---|---|---|
| **Llama-3.2-1B** | `meta-llama/Llama-3.2-1B` | 16 | 2048 / 64 / 8192 | GQA 32Q/8KV | reference exemplar | prefill + decode |
| **Llama-3.2-1B int4-AWQ** | `amd/Llama-3.2-1B-Instruct-awq-...` | 16 | 2048 / 64 / 8192 | GQA 32Q/8KV | int4-AWQ (prefill) | prefill; decode follow-up |
| **SmolLM2-1.7B** | `HuggingFaceTB/SmolLM2-1.7B` | 24 | 2048 / 64 / 8192 | MHA 32Q/32KV | pure MHA | prefill + decode |
| **Qwen2.5-0.5B** | `Qwen/Qwen2.5-0.5B` | 24 | 896 / 64 / 4864 | GQA 14Q/2KV | QKV bias | prefill + decode |
| **Qwen2.5-1.5B** | `Qwen/Qwen2.5-1.5B` | 28 | 1536 / 128 / 8960 | GQA 12Q/2KV | QKV bias, hd=128 | prefill + decode |
| **Qwen2.5-3B** | `Qwen/Qwen2.5-3B` | 36 | 2048 / 128 / 11008 | GQA 16Q/2KV | QKV bias, hd=128 | prefill + decode |
| **Qwen3-0.6B** | `Qwen/Qwen3-0.6B` | 28 | 1024 / 128 / 3072 | GQA 16Q/8KV | QK-norm, hd=128 | prefill + decode |
| **Qwen3-1.7B** | `Qwen/Qwen3-1.7B` | 28 | 2048 / 128 / 6144 | GQA 16Q/8KV | QK-norm, hd=128 | prefill + decode |
| **Qwen3-4B** | `Qwen/Qwen3-4B` | 36 | 2560 / 128 / 9728 | GQA 32Q/8KV | QK-norm, decoupled q_dim=4096 | prefill + decode |
| **Llama-3.2-3B** | `meta-llama/Llama-3.2-3B` | 28 | 3072 / 128 / 8192 | GQA 24Q/8KV | pure Llama, hd=128 | prefill + decode |

All are decoder-only with RMSNorm + SwiGLU FFN + RoPE. The architecture axes that
shape each deployment's dataflow:
- **Attention norm/bias**: Llama (none) · Qwen2.5 (**QKV bias**, fused into the
  attention-input ELF on host-loaded bias weights) · Qwen3 (**per-head QK-norm**, fused).
- **head_dim**: 64 → seq-first FlashAttention (no host transpose); 128 → head-first
  FlashAttention + a host seq↔head transpose (BF16 DMA stride limit).
- **hidden size**: small/aligned → O+FFN fuses into one ELF; large/non-aligned
  (8960/9728/11008) → O+FFN splits (gate/up/down separate ELFs; per-launch L1/BD
  limit, not a launch-count limit).

## Performance (NPU2, seq_len=2048, bf16)

Measured end-to-end via `make run`. **Prefill (TTFT)** = time-to-first-token
(tokenize + EOS-pad + prefill + LM head); **Decode (TPS)** = steady-state decode
throughput in tokens/second.

TTFT is wall-clock (includes the host head-first-FA seq↔head transpose on the
head_dim=128 models); the NPU-kernel prefill time is lower. **ELF/layer** = NPU
dispatches per transformer layer (prefill / decode); decode adds one `lm_head_gemv`
per token.

| Model | Prefill (TTFT) | Decode (TPS) | ELF/layer (prefill / decode) |
|---|---|---|---|
| Llama-3.2-1B (bf16) | 1.21 s | 12.2 tok/s | 3 / 2 |
| SmolLM2-1.7B (bf16) | 2.02 s | 8.0 tok/s | 3 / 2 |
| Qwen2.5-0.5B | 0.99 s | 11.9 tok/s | 4 / 5 |
| Qwen3-0.6B | 1.52 s | 11.7 tok/s | 3 / 2 |
| Qwen3-1.7B | 2.08 s | 7.4 tok/s | 3 / 2 |
| Qwen2.5-1.5B | 2.43 s | 6.6 tok/s | 5 / 5 |
| Llama-3.2-3B | 3.70 s | 4.7 tok/s | 3 / 5 |
| Qwen2.5-3B | 4.24 s | 3.5 tok/s | 6 / 5 |
| Qwen3-4B | 6.06 s | 3.2 tok/s | 7 / 5 |

Measured via `make profile N_TOKENS=32` on NPU2, 2026-06. All pass `make verify`
(top-5 token-set vs HF bf16, exit 0). The lean **3-ELF prefill / 2-ELF decode**
form (Qwen3-0.6B/1.7B, matching Llama-3.2-1B) needs aligned dims + emb<2560 +
hidden÷512; bigger models split the FFN and run a 5-ELF decode (the fused
`o_gemv_ffn` cascade is capped at 4 cascade core-rows and needs hidden÷512).

### What runs on NPU vs CPU
- **NPU** (the entire compute-heavy path): all GEMM (prefill) / GEMV (decode)
  projections, RMSNorm, RoPE, QK-norm (Qwen3) / QKV-bias (Qwen2.5), FlashAttention
  (prefill), SwiGLU, and the LM-head GEMV.
- **CPU** (cheap or hardware-limited, by design): decode attention (single-token +
  KV cache — NPU launch overhead > compute), the head-first-FA seq↔head transpose
  at head_dim=128 (BF16 DMA stride limit), the M=1 decode glue (residual add /
  FFN RMSNorm / SwiGLU, ~0.13 ms/layer — dispatch > compute), and the final RMSNorm.

See each model's `ARCHITECTURE.md` for the full per-layer NPU/CPU ELF sequence.

## Correctness

Every model gates on the shared `verify/` subsystem against an HF bf16 reference
using a top-k token-set inclusion check (k=5, first-divergence over 32 tokens).
Per-layer cosine via `make diagnosis` is informational. Run `make verify` in any
model directory.

## Layout

```
llms/
├── llama32_1b/         # bf16 Llama-3.2-1B (the reference exemplar)
├── llama32_1b_int4/    # int4-AWQ variant (own quantized builders)
├── llama32_3b/         # bf16 Llama-3.2-3B (pure Llama, head_dim=128)
├── smollm2_1_7b/       # bf16 SmolLM2-1.7B (MHA)
├── qwen25_0_5b/        # Qwen2.5-0.5B  (QKV bias, head_dim=64)
├── qwen25_1_5b/        # Qwen2.5-1.5B  (QKV bias, head_dim=128)
├── qwen25_3b/          # Qwen2.5-3B    (QKV bias, head_dim=128)
├── qwen3_0_6b/         # Qwen3-0.6B    (QK-norm, head_dim=128)
├── qwen3_1_7b/         # Qwen3-1.7B    (QK-norm, head_dim=128)
├── qwen3_4b/           # Qwen3-4B      (QK-norm, decoupled q_dim=4096)
├── shared/
│   ├── infra/          # KernelCache, profiling, external-kernel compilation,
│   │                   #   backend presets, MLIR text-stitching (incl. stitch_elf)
│   └── builders/       # architecture-orthogonal multi-launch block builders
│                       #   (rms_gemms_rope, o_ffn, rms_gemv_rope, o_gemv_ffn,
│                       #   lm_head_gemv) — composed via stitch_elf
└── verify/             # shared HF-reference verification framework
```

Models compose blocks from `shared/builders/` (reused directly when the shape
contract matches) or assemble new blocks declaratively with `stitch_elf` from
`shared.infra.stitching`. The `int4` example keeps its own quantized builders
since the int4/bfp16 GEMM ABIs differ from bf16.

## Running a model

```bash
cd <model>/                 # e.g. cd smollm2_1_7b
make run                    # compile (first time) + prefill + decode a prompt
make verify                 # correctness gate vs HF bf16
make profile                # timing breakdown
```

See each model's `README.md` for model-specific flags and notes.
