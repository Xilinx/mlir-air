# PHI-4-MINI Q4NX Inference on AMD NPU2 (MLIR-AIR)

This implementation is an MLIR-AIR reimplementation, based on the corresponding
AMD NPU LLM design originally developed by the
[FastFlowLM](https://github.com/ROCm/FastFlowLM) team.

A full **prefill + decode** MLIR-AIR inference for `microsoft/Phi-4-mini-instruct` using
**Q4NX** weights (per-block 4-bit affine quantization, `w = q*scale + min`) on
NPU2 (AIE2P). The decoder layer runs **entirely on the AIE array** — attention
included, over a device-resident KV cache — the same mapping FastFlowLM uses.

- **Prefill** (`phi4_mini_q4nx_prefill.py`) — op-by-op bf16 GEMM / RMSNorm /
  partial RoPE (`rope_partial`) / SwiGLU / head-first causal-GQA flash attention
  on the NPU with resident weight BOs; on-device 13-partition LM-head GEMV. Its
  per-layer roped-K / raw-V seed the decode's device KV cache, so a long prompt
  is not replayed token-wise.
- **Autoregressive decode** (`q4nx_decode_phi4.py`) — one dispatch per token
  through the [`fused_decode`](../../fused_decode) superkernel at
  `MODEL_TYPE=PHI4_4B`: proj → RoPE → flash attention over the on-device KV → O →
  FFN, ×32 layers, then the LM head. No CPU attention, no host KV.
- **Host orchestration** — embedding, argmax, chat template, EOS, streaming.

NPU weights are FastFlowLM's `model.q4nx` packing
(`FastFlowLM/Phi4-mini-Instruct-NPU2`); the `make verify` reference is
`microsoft/Phi-4-mini-instruct`, which is ungated and is the same repo
FastFlowLM's own reference generator uses (`models_simple/verify_phi4.py`).

## Model Config

32 layers, emb_dim=3072, n_heads=24, head_dim=128, n_kv_heads=8 (GQA group=3),
q_dim=3072, kv_dim=1024, hidden_dim=8192, vocab_size=200064, rope_theta=10000,
eps=1e-5, tied embeddings. Q4NX codec: 32×256 blocks, 32-column groups.

Attention topology and the per-phase proj/FFN block counts are **identical to
llama3.2-3b** (`I2P=[5,3,16,3]`, `J2P=[6,6,6,16]`), so the decode reuses that
geometry unchanged. Three things are Phi-4-specific:

**Partial rotary (the real work).** `partial_rotary_factor=0.75`, so RoPE covers
only the leading **96 of 128** head dims and the trailing 32 pass through
untouched — FastFlowLM ships an entire separate kernel (`rope_phi4.cc`) for this.
Here it is a `PARTIAL_ROPE_DIM` in the model header that `model_spec.h` turns
into `ROPE_DIM`; `apply_rope` rotates `ROPE_DIM` (halves `ROPE_DIM/2` apart) and
copies the tail, and the cos/sin LUT shrinks to 96 wide so the kernel reads sin
at `rope_w + ROPE_DIM/2`. `ROPE_DIM == DH` for every other model, and the
resulting `rope.o` is **byte-identical** for llama-1B/3B, gemma3-4b and
qwen2.5-3b — verified by md5 against a pre-change build.

**LongRoPE.** The bundle carries `rope.short/long.weight[48]` factor tables
(48 = 96/2 frequencies). `short_factor` is all 1.0, so within
`original_max_position_embeddings=4096` the frequencies reduce to plain
`inv_freq(1e4, 96)` — exactly FastFlowLM's `phi4_rope` short half, which this
matches numerically. The long table (up to 47.75) only engages past 4096; the
shipped 2048 template therefore never leaves the short regime.

**Fused source tensors, split bundle.** HF's `Phi3ForCausalLM` stores fused
`self_attn.qkv_proj` [5120,3072] and `mlp.gate_up_proj` [16384,3072], and the
GGUF is fused the same way — but FastFlowLM's converter *writes the q4nx bundle
already split* into q/k/v and gate/up, so the loader needs no unfusing. Only the
cosine check below splits the HF side to compare like with like; that it lands at
0.997 confirms the converter's `[q,k,v]` / `[gate,up]` row order.

## Weight fidelity (Q4NX dequant vs HF bf16, layer 0)

`python3 phi4_mini_q4nx_weights.py` — measured:

| proj | q | k | v | o | up | gate | down |
|---|---|---|---|---|---|---|---|
| cosine | 0.99709 | 0.99670 | 0.99699 | 0.99761 | 0.99826 | 0.99821 | 0.99810 |

Worst 0.99670. `cosine(bundle lm_head, embed_tokens) = 0.99958` confirms the LM
head really is tied, so the lossless bf16 embedding is used for it rather than
the bundle's separate quantized copy.

## Quick start

```bash
make compile          # prefill ELFs (~4 min, one-time; no weights)
make compile-decode   # fused decode templates into this dir (~15 min; no weights)
make run              # prefill + 32 layers/LM head on-device, one dispatch per token
make chat             # interactive chat REPL (streaming)
make verify           # top-k token-set gate: NPU q4nx vs HF bf16
```

## Two LM-head precisions (read this)

The prefill and the decode do **not** use the same LM head, and for Phi-4 it is
visible in the gate:

- **Prefill** builds its LM head from `w.lm_head`, which for a tied model is the
  **bf16 embedding**, and runs it as an on-device 13-partition GEMV.
- **Decode** streams the LM head from the **4-bit q4k cascade** (FLM's design;
  their `decoding_layer.cpp` allocates it at 5 bits/element).

Measured: prompts routed through the prefill pass `make verify-full` **8/8**;
prompts short enough to be replayed token-by-token through the decode
(< `Q4NX_PREFILL_MIN`, default 96) score **5/8**, because the 4-bit head reorders
Phi-4's flat first-token distributions. Llama-3.2-3B has the same two-headed
design and does not show it -- its margins are wider.

`run_npu2_verify.lit` therefore sets `Q4NX_VERIFY_FORCE_PREFILL=1`: it exercises
the production prefill -> KV-handoff -> decode path (strictly more coverage than
replay), and it is the path any real prompt of >=96 tokens takes.

## Build notes

- **`llvm-link` must be LLVM <23** for the inline-attn merge. If the AIR env
  script puts an LLVM 24 `llvm-link` first, prepend a compatible one:
  `export PATH=/usr/lib/llvm-20/bin:$PATH`.
- **Vocab chunking is `VOCAB_CHUNK_I2=18`, `UNI_LM=11`.** `VOCAB_SIZE_PADDED_FULL
  = ceil(200064/3072)*3072 = 202752` → 6336 rowblocks = 198 units, so `VOCAB_I2`
  must divide 198; `K/PAYLOAD = 6` must divide `VOCAB_I2*PAIR_ROWS` (so `3 |
  VOCAB_I2`); and the tested envelope caps `2*VOCAB_I2 <= 63`. That leaves
  {3,6,9,18}, and 18 gives the fewest host-armed waves. Another value deadlocks
  the vocab wave.
- Phi-4's tokenizer is o200k-based, so the ids differ from the Llama examples
  and no BOS is prepended: `"The capital of France is"` is
  `[976, 9029, 328, 10128, 382]` and `" Paris"` is `12650`.
