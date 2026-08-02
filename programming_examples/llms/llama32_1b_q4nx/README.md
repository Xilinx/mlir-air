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

Like the sibling `llama32_1b` / `llama32_1b_int4`, but it consumes external Q4NX
weights and gates on the greedy first token (" Paris", id 12366) rather than a
`make verify` top-k comparison vs HF transformers bf16 (there is no HF checkpoint
for these weights). See [Correctness](#correctness) and [Data](#data).

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

## Files

| File | Purpose |
|---|---|
| `llama32_1b_q4nx_weights.py` | Q4NX unpack / dequant / dims; per-layer weight loaders (host dequant cache) |
| `llama32_1b_q4nx_prefill.py` | `LlamaQ4nxPrefill` — batched prefill; per-layer KV `kv_view()` handoff; Paris gate |
| `llama32_1b_q4nx_inference.py` | Orchestrator: `Session`, `FusedDecoder`, `Sampler`, chat template, EOS, streaming, `--interactive` REPL |
| `proj_qmm_pack.py` | numpy Q4NX block packer + dequant reference |
| `dequant_q4nx.{cc,py}` | optional on-device Q4NX uint4→bf16 dequant kernel + builder |
| `gen_paris_golden.py` | one-time generator for the HF golden bundle (embed / final_norm / lm_head) |

The decode path (fused superkernel, host instruction patcher, decode kernels) lives in
the standalone [`fused_decode`](../../fused_decode) example, which this e2e references.

Cross-directory imports (matching `llama32_1b_int4`): the shared `llms/` infra
(kernel cache, GEMM/stitcher builders, `lm_head_gemv_multi`), the sibling `llama32_1b`
prefill driver, and the `fused_decode` example (decode templates + `decode_insts_gen`).

## Performance (NPU2, AMD Strix, 16 layers, warm)

- **Prefill TTFT** @ 2048 ≈ **0.93 s** (`TEMPORAL_CAUSAL_SKIP=1`); ~constant in
  prompt length (the prefill processes the full padded `seq_len` each call).
- **Decode**: ~**50 tok/s** end-to-end at turbo (single MAX_L=2048 template).

## Prerequisites

1. **MLIR-AIR base environment** — AMD NPU2, Peano (`PEANO_INSTALL_DIR`),
   `source utils/env_setup.sh ...`.
2. **Extra Python packages**: `pip install -r requirements.txt`.
3. **HF tokenizer** for Llama-3.2-1B (chat template + detokenize). Point
   `Q4NX_TOKENIZER` at any local Llama-3.2-1B tokenizer directory (default
   `~/q4nx_data/tokenizer/Llama-3.2-1B`).

## Reproducibility (decode toolchain)

`make compile-decode` merges an inline attention kernel into the core via an
**external `llvm-link`** (resolved from `PATH`). That `llvm-link` **must be a
pre-`llvm.lifetime`-change LLVM** (e.g. the system `/usr/bin/llvm-link`, LLVM 20).
A newer LLVM (≥ 23) `llvm-link` rewrites the `llvm.lifetime` intrinsic to the
no-size form, which Peano `opt` (LLVM 21) then rejects (`Broken module found`).

So for the decode build, **keep any ≥23 LLVM's `bin` off `PATH`** (in particular do
not put an AOMP / mlir-distro LLVM `bin` ahead of `/usr/bin`). `make compile-decode`
runs a preflight that prints the resolved `llvm-link` and aborts early if it is
≥ 23. Verify with:

```bash
which llvm-link && llvm-link --version   # expect LLVM 20.x (or any <23)
```

Nothing compiled is committed — `make compile` / `make compile-decode` reproduce
every ELF, xclbin, and instruction stream from source (see `.gitignore`).

## Data

**Prefill (default): one HuggingFace download.** The prefill weights come from the
self-contained `model.q4nx` bundle on the Hub:

- `MODEL` / `Q4NX_MODEL` — the Q4NX model source (default
  `FastFlowLM/Llama-3.2-1B-NPU2`). The prefill downloads `model.q4nx` (a safetensors
  file: per-layer Q4NX projections + bf16 norms/embed + Q4NX lm_head) via
  `huggingface_hub` and dequantizes it on load. May also be a local dir/file. This
  is all the prefill (`make run` / `verify` / `profile`) needs.

**Decode / chatbot (legacy external weights).** The fused-decode chatbot still reads
a pre-built requant cache + golden bundle out of band (a `model.q4nx`-sourced decode
path is a follow-up):

- `PARIS_GOLDEN` — golden bundle `weights/{embed_tokens,final_norm,lm_head}.f32.bin`
  (default `/tmp/paris_golden`). Generate once: `python3 gen_paris_golden.py`.
- `PARIS_REQUANT_CACHE` — decode q4k-cascade weights `.npz` (default
  `/tmp/paris_native_w.npz`).
- `PARIS_WEIGHTS` — legacy per-layer Q4NX dumps `L{k}_proj_w.bin` (only used as a
  prefill fallback when `model.q4nx` is unavailable).

## Build

```bash
# 1. Prefill ELFs (per seq_len; no weights/NPU needed) — ~3 min
make compile CTX=2048

# 2. Decode kernels + two templates (decode_L2048 + decode_L2047 slope ref)
#    ~15 min; needs PARIS_WEIGHTS for the GEMM shapes and a pre-change llvm-link
#    (see Reproducibility).
make compile-decode
```

## Run

```bash
make chat                         # interactive multi-turn chatbot (streams tokens)
make ask PROMPT="What is the capital of France?"   # single Q&A turn
make gen                          # Paris gate: prefill+decode, first token 12366 -> *** PARIS ***
make run                          # prefill-only Paris gate
make verify                       # same prefill Paris gate (name CI expects)
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

The Paris gate is greedy first-token parity: for "The capital of France is", prefill
predicts argmax **12366 (" Paris")** and the fused decode continues coherently
(`make gen` / `make run` print `*** PARIS ***`). This replaces the HF-reference
`make verify` used by the other examples (no HF checkpoint exists for these Q4NX
weights).

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
single-buffered (`DECODE_BLOCK_SINGLEBUF=1`) to keep the KV ring aligned.
`decode_insts_gen.py` patches the per-token instruction words (RTP-L, KV-append
offset) so one xclbin serves every `L`. The host performs embedding, sampling
(`Sampler`: repetition/frequency penalties + temperature + top-k/top-p), chat
templating, and EOS between tokens.
