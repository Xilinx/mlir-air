# LLAMA-3.1-8B Q4NX Inference on AMD NPU2 (MLIR-AIR)

A full **prefill + decode** MLIR-AIR inference for Llama-3.1-8B using **Q4NX**
weights (per-block 4-bit affine quantization, `w = q*scale + min`) on NPU2
(AIE2P). The decoder layer runs **entirely on the AIE array** — attention
included, over a device-resident KV cache — the same mapping FastFlowLM uses.

- **Prefill** (`llama31_8b_q4nx_prefill.py`) — op-by-op bf16 GEMM / RMSNorm /
  RoPE / SwiGLU / head-first causal-GQA flash attention on the NPU with resident
  weight BOs; on-device 8-partition LM-head GEMV. Its per-layer roped-K / raw-V
  seed the decode's device KV cache, so a long prompt is not replayed token-wise.
  The stitcher/FA builders are dimension-driven and 8B shares 3B's `head_dim=128`
  head-first FA path, so they are reused rather than re-authored.
- **Autoregressive decode** (`q4nx_decode_8b.py`) — one dispatch per token through
  the [`fused_decode`](../../fused_decode) superkernel at `MODEL_TYPE=LLAMA_3_1_8B`:
  proj → RoPE → flash attention over the on-device KV → O → FFN, ×32 layers, then
  the LM head. No CPU attention, no host KV.
- **Host orchestration** — embedding, argmax, chat template, EOS, and streaming
  output between tokens.

NPU weights are FastFlowLM's `model.q4nx` packing of `meta-llama/Llama-3.1-8B-Instruct`
(`FastFlowLM/Llama-3.1-8B-NPU2`), so this example gates with the standard
`make verify` **top-k token-set inclusion vs HF transformers bf16**.

## Model Config

32 layers, emb_dim=4096, n_heads=32, head_dim=128, n_kv_heads=8 (GQA group=4),
q_dim=4096, kv_dim=1024, hidden_dim=14336, vocab_size=128256, rope_theta=500000
with **llama3 rope scaling factor 8.0**, eps=1e-5, **untied** embeddings
(`lm_head` ≠ `embed_tokens`). Q4NX codec: 32×256 blocks, 32-column groups.
Pure Llama (no QK-norm, no bias).

Two of those differ from the Llama-3.2 siblings and are both load-bearing:

**Untied LM head.** Llama-3.2-1B/3B set `tie_word_embeddings=true`, so their LM
head *is* the embedding matrix, and the shared requant path hard-coded that.
Llama-3.1-8B sets it false and the bundle ships a real quantized `lm_head`.
Measured `cosine(bundle lm_head, embed_tokens) = 0.017` — essentially orthogonal,
so the tied default would have produced a silently wrong model rather than a
build error. `build_requant_cache(..., tie_lm_head=False)` selects the real one.

**RoPE scaling factor 8.0.** Llama-3.1 scales by 8.0, Llama-3.2 by 32.0. This is
also exactly the curve behind FastFlowLM's shared `llama_3b_8b_rope` table:
factor 8.0 reproduces that table to 4e-05 (its print precision), 32.0 is off by
75% and unscaled by 7x. So FLM-faithful and HF-correct coincide here.

## Weight fidelity (Q4NX dequant vs HF bf16, layer 0)

`python3 llama31_8b_q4nx_weights.py` — measured on this bundle:

| proj | q | k | v | o | up | gate | down | lm_head |
|---|---|---|---|---|---|---|---|---|
| cosine | 0.99687 | 0.99585 | 0.99523 | 0.99775 | 0.99810 | 0.99860 | 0.99792 | 0.99964 |

Worst 0.99523 — the expected range for this codec.

## Measured on NPU2

| gate | result |
|---|---|
| `make verify` (2 prompts x 32 tokens, k=5, vs HF bf16) | PASS 2/2 |
| `make verify-full` (8 prompts) | PASS 8/8 |
| `make verify-paris` (prefill weight-integrity smoke) | PASS (argmax 12366) |
| `make profile N_TOKENS=128` | TTFT 1.60 s, decode 10.7 tok/s |

The Paris smoke gates on **"The capital city of France is called"**, not the bare
"The capital of France is" the 1B/3B siblings use. On the bare phrasing this model
ranks `" a"` (16.000) just above `" Paris"` (15.938) — and the HF bf16 reference
ranks them the same way, so that prompt fails a correct build. The "...is called"
phrasing puts `" Paris"` 3.4 logits clear.

## Quick start

```bash
make compile          # prefill ELFs (~4 min, one-time; no weights)
make compile-decode   # fused decode templates into this dir (~15 min; no weights)
make run              # 32 layers + LM head on-device, one dispatch per token
make verify           # top-k token-set gate: NPU q4nx vs HF bf16
```

## Build notes

- **`llvm-link` must be LLVM <23** for the inline-attn merge (`make compile-decode`
  preflights this). On a box where the AIR env script puts an LLVM 24 `llvm-link`
  first on `PATH`, prepend a compatible one, e.g.
  `export PATH=/usr/lib/llvm-20/bin:$PATH`.
- **The weights do not fit one buffer (`DECODE_WGROUP=8`).** A shim BD's byte
  offset is a `uint32` end to end, so a single BO is addressable over at most
  4 GiB. This model's 32 layer slabs plus the lm-head are **4.375 GiB**: the
  offsets wrapped past the boundary and every logit came back NaN. `DECODE_WGROUP`
  splits the weights into four 1.02 GiB layer groups plus a 0.31 GiB lm-head
  buffer, each addressed from its own base, and the host slices its BOs to match
  (read off the engine, not the environment, so the two cannot disagree). The
  build and the run must use the same value; the Makefile's `DECODE_WGROUP` and
  `q4nx_decode_8b.py`'s constant are both 8, and the template stamp is keyed on it.
- **The rms/residual core runs a trimmed stack at this size (`DECODE_STACK=8064`).**
  It holds 7 K-sized bf16 buffers; at K=4096 that is 56 KB of the 64 KB L1, so the
  engine's default 10240-byte stack does not fit and buffer allocation fails
  outright. 8064 fits in banks (bank0 = stack + 1 buffer + the 4-byte per-herd RTP)
  and still leaves >2x headroom over the measured worst frame (2112 B). Any model
  at K=4096 hits this same wall.
- **The LM-head GEMV needs a halved `tile_m`.** Its L2 A stage needs
  `herd_m*tile_m*(K+1)*2 <= 512 KiB`, which at K=4096 overflows by exactly 128 B
  at the default 8x8. `herd_m` must stay 8 (4 builds but hangs the dispatch), so
  `tile_m` halves to 4 and `mv.o` is rebuilt at that `DIM_M_OUTPUT`.
- **Vocab chunking is `VOCAB_CHUNK_I2=16`, `UNI_LM=8`.** The vocab relay drains
  whole-K blocks, so `K/PAYLOAD = 8` must divide `VOCAB_I2*PAIR_ROWS`; with
  `VOCAB_FULL_ROWBLKS = 4096` the legal set is {4,8,16} and 16 gives the fewest
  host-armed waves. Another value deadlocks the vocab wave.

## Reference

`make verify` compares against `NousResearch/Meta-Llama-3.1-8B-Instruct`, a
faithful ungated re-upload of the `meta-llama` checkpoint FastFlowLM's bundle
names as its `base_model`. Its config matches the FLM bundle's exactly, including
the 3-way `eos_token_id` list, so CI needs no license grant — the same pattern
`gemma3_4b_q4nx` uses for google's gated Gemma. Point `MODEL_CHOICES` in
`verify_adapter.py` at `meta-llama` instead if you have accepted Meta's license.
