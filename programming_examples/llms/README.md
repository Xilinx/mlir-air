# LLMs on AMD NPU2 (MLIR-AIR)

End-to-end decoder-only LLM inference (prefill + autoregressive decode) mapped
to the AMD NPU2 (AIE2P / Strix) in bf16 via MLIR-AIR. Each model is a
self-contained example that composes registry-validated leaf kernels into fused
multi-launch ELFs and gates correctness against a Hugging Face bf16 reference.

## Supported models

| Model | HF checkpoint | Layers | Attention | Quant | Status |
|---|---|---|---|---|---|
| **Llama-3.2-1B** | `meta-llama/Llama-3.2-1B` | 16 | GQA (32 Q / 8 KV) | bf16 | prefill + decode |
| **Llama-3.2-1B int4-AWQ** | `amd/Llama-3.2-1B-Instruct-awq-uint4-asym-g128-bf16-lmhead` | 16 | GQA (32 Q / 8 KV) | int4-AWQ (prefill) | prefill (bf16 + int4 paths); decode is a follow-up |
| **SmolLM2-1.7B** | `HuggingFaceTB/SmolLM2-1.7B` | 24 | MHA (32 Q / 32 KV) | bf16 | prefill + decode |

All share `emb_dim=2048`, `head_dim=64`, `hidden_dim=8192`, SwiGLU FFN, RMSNorm,
RoPE. The architectural deltas: Llama is GQA (`kv_dim=512`), SmolLM2 is pure MHA
(`kv_dim=2048`, `rope_base=130000`, `vocab=49152` vs Llama's `128256`).

## Performance (NPU2, seq_len=2048, bf16)

Measured end-to-end via `make run`. **Prefill (TTFT)** = time-to-first-token
(tokenize + EOS-pad + prefill + LM head); **Decode (TPS)** = steady-state decode
throughput in tokens/second.

| Model | Prefill (TTFT) | Decode (TPS) |
|---|---|---|
| Llama-3.2-1B (bf16) | 1.21 s | 12.2 tok/s |
| SmolLM2-1.7B (bf16) | 2.02 s | 8.0 tok/s |
| Llama-3.2-1B int4-AWQ | ~1.4 s (bf16 prefill path) † | — (decode is a follow-up) |

Llama-3.2-1B and SmolLM2-1.7B numbers were measured directly with the current
code. † The int4-AWQ prefill figure is from that example's deployment report
(its driver reports a correctness comparison rather than a clean TTFT); the bf16
prefill path is shared with Llama-3.2-1B, so its TTFT is in the same range.

Notes:
- SmolLM2 decode TPS is lower than Llama's largely because it has more layers
  (24 vs 16); `o_gemv_ffn` is the dominant per-token cost.
- The int4-AWQ example's `int4` prefill path runs but is kernel-bound
  (~11 s end-to-end); the `bf16` path on dequantized AWQ weights is the
  recommended prefill today (identical AWQ-quality output). int4's win is in
  decode (DMA-bandwidth-bound, halved weight footprint), shipped separately.

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
├── smollm2_1_7b/       # bf16 SmolLM2-1.7B (MHA)
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
