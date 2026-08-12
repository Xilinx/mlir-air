# LLAMA-3.2-1B Q4NX chatbot on AMD NPU2

A full **prefill + decode** MLIR-AIR inference for Llama-3.2-1B using **Q4NX**
weights (per-block 4-bit affine quantization, `w = q*scale + min`), exposed as an
interactive chatbot on NPU2 (AIE2P):

- **Batched prefill** (`llama32_1b_q4nx_prefill.py`) — Q4NX weights, dequant
  Q4NX→bf16, op-by-op bf16 GEMM / RMSNorm / RoPE / SwiGLU / causal-GQA flash
  attention all on the NPU with resident weight BOs, on-device LM head. Fills a
  per-layer KV cache.
- **Fused per-token decode** (the standalone [`fused_decode`](../../fused_decode)
  example) — one dispatch = 16 decoder layers + LM head, reading and appending that
  shared KV cache. Runtime KV length via host instruction patching
  (`decode_insts_gen.py`), so ONE decode xclbin serves every context length
  (recompile-free per-token decode).
- **Host orchestration** (`llama32_1b_q4nx_inference.py`) — embedding, final
  RMSNorm, argmax / stochastic sampler, chat template, EOS, and streaming output
  run on the host between tokens.

Like the sibling `llama32_1b` / `llama32_1b_int4`, it plugs into the shared driver
contract (`make run`/`profile`/`chat`/`verify`) and the `verify/` subsystem. It
consumes external Q4NX weights, so `make verify` compares the NPU q4nx run against
the HF **bf16** sibling checkpoint (`meta-llama/Llama-3.2-1B-Instruct`) — the same
NPU-vs-bf16 proxy the `llama32_3b_q4nx` example uses. A fast weight-integrity smoke
(greedy first token " Paris", id 12366) is kept as `make verify-paris`. See
[Correctness](#correctness) and [Data](#data).

## Architecture

```
  user turn ─► chat template ─► [ batched PREFILL ]──fills──► shared per-layer KV cache
                              (llama32_1b_q4nx_prefill)                  │
                                                                        ▼
  reply ◄─ detok ◄─ sampler/argmax ◄─ host embed ◄─ [ per-token fused DECODE ]
     (streamed)                                     (fused_decode, reads+appends KV)
                                                     one dispatch = 16 layers + LM head
```

The decode is ONE xclbin whose per-token context length `L` is set by patching a
few instruction words on the host (`decode_insts_gen.py`): an RTP-L + KV-append
offset patch. Two same-`ATTN_MAXL` builds (`decode_L2048` + `decode_L2047`)
calibrate the L-slope so the generator can synthesize the instruction stream for
any `L` in `[1, 2048]` — byte-identical to a native per-L build. The attention
kernel skips fully-masked blocks, so the single MAX_L=2048 template is correct for
every context length.

## Performance (NPU2, AMD Strix, 16 layers, warm)

- **Prefill TTFT** @ 2048 ≈ **0.93 s** (`TEMPORAL_CAUSAL_SKIP=1`); ~constant in
  prompt length (the prefill processes the full padded `seq_len` each call).
- **Decode**: ~**50 tok/s** end-to-end at turbo (single MAX_L=2048 template).

> Decode streams each proj column's weights on **both** of that column's shim
> MM2S channels (`W_DUAL_CHAN`, on by default) — see
> [fused_decode/README.md](../../fused_decode/README.md#dual-mm2s-weight-feed-w_dual_chan-on-by-default).
> The decode figure below predates that change and is therefore conservative.


## Prerequisites

1. **MLIR-AIR base environment** — AMD NPU2, Peano (`PEANO_INSTALL_DIR`),
   `source utils/env_setup.sh ...`.
2. **Extra Python packages**: `pip install -r requirements.txt`.
3. **HF tokenizer** for Llama-3.2-1B (chat template + detokenize). Point
   `Q4NX_TOKENIZER_DIR` at any local Llama-3.2-1B tokenizer directory (default
   `~/q4nx_data/tokenizer/Llama-3.2-1B`).

## Reproducibility (decode toolchain)

`make compile-decode` merges an inline attention kernel into the core via an
**external `llvm-link`** (resolved from `PATH`), which must come from **LLVM < 23**.
A ≥ 23 one rewrites the `llvm.lifetime` intrinsic to the no-size form, which Peano
`opt` then rejects (`Broken module found`).

So keep any such LLVM's `bin` off `PATH` for the decode build. The
`make compile-decode` preflight prints the resolved `llvm-link` and aborts on a
≥ 23 one. Verify with:

```bash
which llvm-link && llvm-link --version   # major must be < 23
```

Neither the Peano wheel nor the MLIR distro wheel provides a usable one (the former
ships no `llvm-link`, the latter's is LLVM 24) — see
[`fused_decode`](../../fused_decode) Reproducibility for how to fetch one without
root.

Nothing compiled is committed — `make compile` / `make compile-decode` reproduce
every ELF, xclbin, and instruction stream from source (see `.gitignore`).

## Data

**One weight source — one HuggingFace download.** Both prefill and decode come from
the single `model.q4nx` bundle on the Hub:

- `MODEL_SOURCE` / `Q4NX_MODEL_SOURCE` — the Q4NX model source (default
  `FastFlowLM/Llama-3.2-1B-NPU2`). `model.q4nx` is a safetensors file with per-layer
  Q4NX projections + bf16 norms/embed + a (tied) lm_head. The **prefill** downloads
  and dequantizes it directly. The **decode/chatbot** derives its q4k-cascade requant
  cache + embed/norm golden from the *same* bundle on first use (one-time pack,
  cached under `~/.cache/q4nx/`). May also be a local dir/file. For the default Hub
  repo the download **pins a revision** compatible with this loader's Q4NX codec
  (`_PINNED_Q4NX_REVISION` in `llama32_1b_q4nx_weights.py`) — the Hub bundle is
  periodically re-packed to a newer block layout that would otherwise fail to load.

`tie_word_embeddings=true`, so the LM head is the full-precision `embed_tokens` (the
bundle's separate Q4NX lm_head is ignored).

Optional overrides (skip the derivation if you pre-supply them): `Q4NX_DECODE_WEIGHTS_NPZ`
(decode q4k-cascade `.npz`) and `Q4NX_GOLDEN_DIR` (embed/final_norm f32 dir).

## Quick Start

```bash
# Build 1 — Prefill ELFs (per seq_len; no weights/NPU needed) — ~3 min
make compile CTX=2048

# Build 2 — Decode kernels + two templates (decode_L2048 + decode_L2047 slope ref).
#   ~15 min; weight-free build, but needs a pre-change llvm-link (see Reproducibility).
make compile-decode

# Run inference (NPU prefill + fused NPU decode)
make run

# With the decode-throughput / TTFT profiling summary
make profile

# Interactive multi-turn chatbot (streams tokens)
make chat

# Single Q&A turn
make ask PROMPT="What is the capital of France?"

# Correctness gate: top-k token-set inclusion, NPU q4nx vs HF bf16
make verify

# Fast weight-integrity smoke: prefill first token 12366 -> *** PARIS ***
make verify-paris
```

Direct invocation (equivalent to `make chat`):

```bash
python3 llama32_1b_q4nx_inference.py --interactive          # /clear resets, /exit quits
python3 llama32_1b_q4nx_inference.py --prompt "..."         # single turn
python3 llama32_1b_q4nx_inference.py --greedy               # deterministic (default: sampler)
python3 llama32_1b_q4nx_inference.py --seq-len 512          # faster prefill, caps context at 512
```

Options: `--temperature 0.7 --top-k 5 --top-p 0.9`, `--rep-penalty`, `--seed`,
`--system "..."`, `--n-tokens N`, `--no-eos-stop`. Enable turbo for ~50 tok/s:
`xrt-smi configure -d <BDF> --pmode turbo`.

## Correctness

`make verify` runs the shared `verify/` top-k token-set inclusion gate (k=5,
first-divergence over 32 greedy tokens across 2 prompts) comparing the NPU q4nx
prefill + fused decode against the HF **bf16** sibling checkpoint
(`meta-llama/Llama-3.2-1B-Instruct`) — driven by `verify_adapter.py`. `make
verify-full` runs the full prompt set; `make diagnosis` is the informational
per-layer lens. Since there is no bf16 checkpoint for the q4nx weights themselves,
the reference is the bf16 sibling (identical to the `llama32_3b_q4nx` approach).

`make verify-paris` is a fast weight-integrity smoke: for "The capital of France
is", prefill predicts argmax **12366 (" Paris")** and prints `*** PARIS ***` —
no decode build or HF reference required.

## How it works

**Prefill** — per block, all compute runs on the NPU through the two `llama32_1b`
multi-launch stitchers plus flash attention (`rms_gemms_rope` · `flash_attn` ·
`o_ffn`) with per-layer resident weight BOs; final RMSNorm on the host, LM head
on-device as an 8-partition GEMV. The per-layer roped-K / raw-V are captured into a
KV cache (`kv_view()`), and the prefill kernel cache is **seq_len-specific**
(`_q4nx_cache_seq<N>`) so a 2048 request never reuses a shorter-context build.

**Decode** — the [`fused_decode`](../../fused_decode) example builds one fused xclbin (16 layers + LM
head). The attention block loop reads `ceil(L/16)` KV blocks; blocks past `L` are
skipped by the kernel so the online softmax is not contaminated, and the loop is
always single-buffered (`air.disable_ping_pong`) to keep the KV ring aligned.
`decode_insts_gen.py` patches the per-token instruction words (RTP-L, KV-append
offset) so one xclbin serves every `L`. The host performs embedding, sampling
(`Sampler`: repetition/frequency penalties + temperature + top-k/top-p), chat
templating, and EOS between tokens.

## Key Files

| Path | Purpose |
|---|---|
| `llama32_1b_q4nx_weights.py` | Q4NX unpack / dequant / dims; per-layer weight loaders (host dequant cache) |
| `llama32_1b_q4nx_prefill.py` | `LlamaQ4nxPrefill` — batched prefill; per-layer KV `kv_view()` handoff; Paris gate |
| `llama32_1b_q4nx_inference.py` | Orchestrator: `Session`, `FusedDecoder`, `Sampler`, chat template, EOS, streaming, `--interactive` REPL |

The decode path (fused superkernel, host instruction patcher, decode kernels) lives in
the standalone [`fused_decode`](../../fused_decode) example, which this e2e references.

Cross-directory imports (matching `llama32_1b_int4`): the shared `llms/` infra
(kernel cache, GEMM/stitcher builders, `lm_head_gemv_multi`), the sibling `llama32_1b`
prefill driver, and the `fused_decode` example (decode templates + `decode_insts_gen` +
the `proj_qmm_pack` Q4NX block packer / dequant reference).
