# LLAMA-3.2-3B Q4NX Inference on AMD NPU2 (MLIR-AIR)

A full **prefill + decode** MLIR-AIR inference for Llama-3.2-3B using **Q4NX**
weights (per-block 4-bit affine quantization, `w = q*scale + min`) on NPU2
(AIE2P). Same on-device pipeline as the sibling bf16 [`llama32_3b`](../llama32_3b)
example — only the **weight source** differs: FastFlowLM's single `model.q4nx`
bundle, dequantized Q4NX→bf16 on the host at load.

- **Prefill** (`llama32_3b_q4nx_inference.py` → reuses `llama32_3b_prefill`) —
  op-by-op bf16 GEMM / RMSNorm / RoPE / SwiGLU / head-first causal-GQA flash
  attention on the NPU with resident weight BOs; on-device 8-partition LM-head
  GEMV. Fills a per-layer KV cache.
- **Autoregressive decode** (`llama32_3b_decode`) — NPU `rms_gemv_rope` +
  O/Gate/Up/Down GEMVs, CPU attention over the KV cache, NPU LM-head GEMV.
- **Host orchestration** — embedding, final RMSNorm, argmax, chat template, EOS,
  and streaming output between tokens.

Unlike the 1B Q4NX example (which gates on the greedy first token because it has
no HF checkpoint), the 3B Q4NX weights are FastFlowLM's packing of the public
`meta-llama/Llama-3.2-3B` weights, so this example gates with the standard
`make verify` **top-k token-set inclusion vs HF transformers bf16** — the closest
apples-to-bf16 signal that AIR faithfully runs the Q4NX model.

## Model Config

28 layers, emb_dim=3072, n_heads=24, head_dim=128, n_kv_heads=8 (GQA group=3),
q_dim=3072, kv_dim=1024, hidden_dim=8192, vocab_size=128256, rope_theta=500000,
eps=1e-5, tied embeddings (lm_head = embed_tokens). Q4NX codec: 32×256 blocks,
32-column groups. Pure Llama (no QK-norm, no bias).

## Performance (NPU2, AMD Strix, 28 layers, warm)

- **Prefill / TTFT** @ 2048 ≈ **3.6 s** (head_dim=128 head-first FA transpose
  included in wall).
- **Decode** ≈ **5 tok/s** (NPU Q/K/V/RoPE + GEMVs, CPU attention). Enable turbo
  for best latency: `xrt-smi configure -d <BDF> --pmode turbo`.

## Prerequisites

1. **MLIR-AIR base environment** — AMD NPU2, Peano (`PEANO_INSTALL_DIR`),
   `source utils/env_setup.sh ...`.
2. **Extra Python packages**: `pip install -r requirements.txt` (`safetensors`,
   `huggingface_hub`, `transformers`, `torch`).
3. **HuggingFace access** (one-time):
   - The tokenizer + the `make verify` bf16 reference come from the gated
     `meta-llama/Llama-3.2-3B-Instruct` — accept Meta's license and
     `huggingface-cli login` (or `export HF_TOKEN=<token>`).

## Data

**One weight source — one HuggingFace download.** Both prefill and decode read the
single `model.q4nx` bundle:

- `MODEL_SOURCE` / `Q4NX_MODEL_SOURCE` — the Q4NX model source (default
  `FastFlowLM/Llama-3.2-3B-NPU2`). `model.q4nx` is a safetensors file with
  per-layer Q4NX projections + bf16 norms/embed + a (tied) lm_head, dequantized
  Q4NX→bf16 on the host at load. May also be a local dir/file.

`tie_word_embeddings=true`, so the LM head is the full-precision `embed_tokens`
(the bundle's separate Q4NX lm_head is ignored).

Nothing compiled is committed — `make compile` / `make compile-decode` reproduce
every ELF, xclbin, and instruction stream from source (see `.gitignore`).

## Reproducibility (decode toolchain)

`make compile-decode` merges an inline attention kernel into the core via an
**external `llvm-link`** (resolved from `PATH`) whose **LLVM major must equal
Peano's** (21 today). A newer (≥ 23) one rewrites `llvm.lifetime` to the no-size
form and Peano `opt` rejects the merged module (`Broken module found`); an older
one links quietly and *silently miscompiles* the attention kernels, so the decode
runs at full speed and emits fluent garbage. The preflight aborts on either:

```bash
which llvm-link && llvm-link --version   # must match $PEANO_INSTALL_DIR/bin/clang --version
```

Every model's templates use the same `decode_L<N>.*` filenames, so this example
builds its own into THIS directory rather than the shared `fused_decode` one
(same as `gemma3_4b_q4nx`). A 1B build can therefore never satisfy a 3B run.

## Quick Start

```bash
# One-time: compile all prefill + decode kernels (~4 min; no weights needed)
make compile

# One-time: build the fused decode templates for the 3B geometry (~15 min).
# They land in this directory as decode_L2048.* / decode_L2047.*.
make compile-decode

# Run inference (instruct model by default; streams up to 1000 tokens)
make run

# Custom prompt
make run PROMPT="How does photosynthesis work?"

# Profiling breakdown (prefill + decode kernel tables)
make profile

# Interactive chat REPL
make chat

# Top-k token-set correctness gate: NPU q4nx vs HF transformers bf16
# (2 prompts × 32 greedy tokens, k=5) — the production-readiness gate
make verify

# Full 8-prompt sweep / per-layer cosine diagnosis lens
make verify-full
make diagnosis
```

Override the NPU weight source (e.g. a local bundle):

```bash
make run MODEL_SOURCE=/path/to/Llama-3.2-3B-NPU2/model.q4nx
```

## Correctness

`make verify` greedily decodes 32 tokens on the NPU (Q4NX) and on HF transformers
(bf16) per prompt, and checks that every NPU token is in HF's top-5 set at that
position. `make verify` (2-prompt CI gate): **PASS**. `make verify-full` (8-prompt
sweep, strict k=5): **7/8** — the single miss is a benign 4-bit-quant near-miss on
the multilingual translation prompt (HF's token stays within the NPU top-5), the
expected sensitivity of a 4-bit model measured against a bf16 reference.

## How it works

**Prefill** — reuses the config-driven `llama32_3b` prefill builders
(`rms_gemms_rope` · head-first `flash_attn` · `o_ffn`) with per-layer resident
weight BOs; final RMSNorm on the host, LM head on-device as an 8-partition GEMV.
Per-layer roped-K / raw-V are captured into a KV cache.

**Decode** — `llama32_3b_decode`: NPU `rms_gemv_rope` (RMSNorm + Q/K/V + RoPE),
NPU O/Gate/Up/Down standalone GEMVs, CPU attention over the KV cache, NPU LM-head
GEMV. Host does embedding, final RMSNorm, argmax, chat templating, and EOS.

**Q4NX weights** — `llama32_3b_q4nx_weights.load_q4nx_weights` wraps the
shape-agnostic `Q4nxModel` reader (from the 1B Q4NX example) and assembles the
dequantized bf16 matrices into the `llama32_3b` `LlamaWeights` container; every
stage downstream of weight loading is the bf16 `llama32_3b` driver, unchanged.

## Key Files

| Path | Purpose |
|---|---|
| `llama32_3b_q4nx_weights.py` | `load_q4nx_weights` — dequant `model.q4nx` → bf16 into the `llama32_3b` LlamaWeights container |
| `llama32_3b_q4nx_inference.py` | Thin driver: reuses the `llama32_3b` runtime + generation loop; swaps the weight source + tokenizer |
| `verify_adapter.py` | Hooks into the shared `../verify/` subsystem; NPU q4nx vs HF bf16 gate |
| `Makefile` | compile / run / profile / chat / verify / verify-full / diagnosis / clean |

Cross-directory reuse: the sibling [`llama32_3b`](../llama32_3b) prefill + decode +
inference drivers, the `Q4nxModel` reader from
[`llama32_1b_q4nx`](../llama32_1b_q4nx), and the shared `llms/` infra + `verify/`
subsystem.
