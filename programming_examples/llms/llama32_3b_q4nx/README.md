# LLAMA-3.2-3B Q4NX Inference on AMD NPU2 (MLIR-AIR)

A full **prefill + decode** MLIR-AIR inference for Llama-3.2-3B using **Q4NX**
weights (per-block 4-bit affine quantization, `w = q*scale + min`) on NPU2
(AIE2P). The decoder layer runs **entirely on the AIE array** — attention
included, over a device-resident KV cache — the same mapping FastFlowLM uses.

- **Prefill** (`llama32_3b_q4nx_prefill.py`) — op-by-op bf16 GEMM / RMSNorm /
  RoPE / SwiGLU / head-first causal-GQA flash attention on the NPU with resident
  weight BOs; on-device 8-partition LM-head GEMV. Its per-layer roped-K / raw-V
  seed the decode's device KV cache, so a long prompt is not replayed token-wise.
- **Autoregressive decode** (`q4nx_decode_3b.py`) — one dispatch per token through
  the [`fused_decode`](../../fused_decode) superkernel at `MODEL_TYPE=LLAMA_3_2_3B`:
  proj → RoPE → flash attention over the on-device KV → O → FFN, ×28 layers, then
  the LM head. No CPU attention, no host KV.
- **Host orchestration** — embedding, argmax, chat template, EOS, and streaming
  output between tokens.

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

- **Prefill** ≈ **824 tok/s** @ 2048 ctx (head_dim=128 head-first FA transpose
  included in wall).
- **Decode** ≈ **18.3 tok/s** (54.7 ms/tok) — 28 layers + LM head on-device, one
  dispatch per token. FastFlowLM's own 3B decode measures ~19.6 tok/s on the same
  machine.

> Decode streams each proj column's weights on **both** of that column's shim
> MM2S channels (`W_DUAL_CHAN`, on by default) — see
> [fused_decode/README.md](../../fused_decode/README.md#dual-mm2s-weight-feed-w_dual_chan-on-by-default).
> The decode figure above predates that change and is therefore conservative.

Both measured at turbo: `xrt-smi configure -d <BDF> --pmode turbo`.

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
# One-time: compile the prefill ELFs (~4 min; no weights needed)
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

# Full 8-prompt sweep
make verify-full

# Prefill-only weight-integrity smoke (first token == " Paris")
make verify-paris
```

There is no `make diagnosis` here: the fused decode runs all 28 layers and the LM
head inside one dispatch, so the per-layer intermediates that lens needs are not
observable from the host. `make verify` is the PASS/FAIL gate.

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

**Decode** — `q4nx_decode_3b.FusedDecode3B` drives the `fused_decode` superkernel
built at `MODEL_TYPE=LLAMA_3_2_3B`: one dispatch appends K/V at the current slot of
the device KV cache and runs proj → RoPE → flash attention → O → FFN for all 28
layers plus the LM head. One `ATTN_MAXL=2048` template serves every context length;
the per-token instruction stream is patched on the host. Host does embedding,
argmax, chat templating, and EOS.

**Q4NX weights** — `llama32_3b_q4nx_weights.load_q4nx_weights` wraps the
shape-agnostic `Q4nxModel` reader (from the 1B Q4NX example) and assembles the
dequantized bf16 matrices into the `llama32_3b` `LlamaWeights` container; every
stage downstream of weight loading is the bf16 `llama32_3b` driver, unchanged.

## Decode staircase (opt-in, ~1.1-1.2x)

The decode template streams `ATTN_MAXL` KV positions per token regardless of the real
context length, so a 2048-window build wastes most of that readback at short context.
Building one template pair per window and dispatching each token on the smallest covering
window recovers it, with a token stream identical to the single-window baseline.

```bash
make compile-decode-windows      # once; WINDOWS="64 512 2048" to override the set
make chat ... --staircase
```

Off by default. See
[`programming_examples/fused_decode/README.md`](../../fused_decode/README.md) for the
mechanism, the measurements and the `.decode_windows` manifest guard.

## Key Files

| Path | Purpose |
|---|---|
| `llama32_3b_q4nx_weights.py` | `load_q4nx_weights` — dequant `model.q4nx` → bf16 into the `llama32_3b` LlamaWeights container |
| `llama32_3b_q4nx_prefill.py` | NPU prefill engine; hands its per-layer roped-K / raw-V to the decode KV cache |
| `q4nx_decode_3b.py` | `FusedDecode3B` — one-dispatch-per-token driver for the `fused_decode` superkernel |
| `llama32_3b_q4nx_inference.py` | Driver + generation loop: prefill → fused decode, chat template, streaming |
| `verify_adapter.py` | Hooks into the shared `../verify/` subsystem; NPU q4nx vs HF bf16 gate |
| `Makefile` | compile / compile-decode / run / profile / chat / verify / verify-full / verify-paris / clean |

Cross-directory reuse: the [`fused_decode`](../../fused_decode) superkernel, the
sibling [`llama32_3b`](../llama32_3b) prefill builders, the `Q4nxModel` reader from
[`llama32_1b_q4nx`](../llama32_1b_q4nx), and the shared `llms/` infra + `verify/`
subsystem.
