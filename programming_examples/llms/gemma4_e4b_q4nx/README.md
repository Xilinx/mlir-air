# Gemma4-E4B (text) Q4NX on NPU2

FastFlowLM's 4-bit `model.q4nx` bundle for Gemma4-E4B, on the NPU end to end:
the fused prefill, then one dispatch per token for all 42 decoder layers, the
per-layer-embedding (PLE) branch and the LM head.

The layout, Makefile targets and lit tests match
[`gemma4_e2b_q4nx`](../gemma4_e2b_q4nx). The Python entry points here run that
directory's code with `GEMMA4_VARIANT=e4b` (`gemma4_e4b_variant.py`), so its
design notes apply; below is what differs.

```bash
export PEANO_INSTALL_DIR=/path/to/llvm-aie   # >= 22.0.0
make compile-decode-elf compile-fused-prefill
make run-fused             # generate on device; gated on the Paris continuation
make profile               # TTFT + decode tok/s
make verify                # the on-device decode gate, one layer per class
make help                  # everything else
```

## What differs from E2B

| | E2B | E4B |
|---|---|---|
| hidden | 1536 | 2560 |
| layers (full every) | 35 (5th) | 42 (6th) |
| kv heads | 1 (MQA) | 2 (GQA 4) |
| FFN | 6144, 12288 on KV-shared layers | 10240 on every layer |
| KV-shared layers | 15-34 | 24-41 (sliding from 22, full from 23) |

**Two kv heads, one per attention CU.** E2B replicates its single head to the
same two CUs; E4B stores both heads side by side in a KV row, and the context
split (which needs a replicated head) is off.

**One FFN width.** The decode's narrow and wide classes then differ only in
whether the layer streams k/v; the prefill builds one FFN ELF instead of two.

**The LM head of the fused prefill runs as Q4 GEMMs on the prefill device**, as
the dense models' does. E2B's runs on a GEMV device of its own, built for K=1536,
whose expanded x alone is 40 KB of L1 at K=2560.

**Smaller core stack.** The PLE gate core's K-wide buffers do not fit L1 beside
the default stack at K=2560, so the `gemma4-e4b` engine entry sets
`DECODE_STACK=8192`.

**Attention scale.** FLM's own `gemma4-e4b.h` sets `ATTN_SCALE = 1/sqrt(DH)`
where its E2B header has 1.0. This port uses 1.0 for both, the HF
config's `scaling`; the numpy reference reproduces the bundle with it.

## Gates (measured here)

Decode, per layer class on device (`make verify`), at ATTN_MAXL=16:

| layer | class | cos |
|---|---|---|
| 0 | sliding, own kv | 0.999744 |
| 5 | full, own kv | 0.999765 |
| 23 | full, own kv (source of every shared full layer) | 0.999684 |
| 24 | sliding, kv<-22 | 0.999709 |
| 29 | full, kv<-23 | 0.999497 |
| 41 | full, kv<-23 (last layer) | 0.999698 |

Prefill, whole-logit cosine vs the CPU reference: per-op 0.996893, fused
0.99946 (one chunk) and 0.99980 (three).

`make verify-topk` (top-5 token-set inclusion vs the bf16 HF checkpoint over
32 greedy tokens; `MODEL_CHOICE=google/gemma-4-E4B-it` points it at a local
copy): 2/2 PASS, diverging only at near-ties.

`make run` / `make run-fused`: greedy from `<bos> The capital of France is`
gives `[9079, 236761]` (' Paris.') and stops on `<end_of_turn>`, matching the CPU
oracle (`forward_prompt` on the growing prompt) token for token.

## Performance

`make profile` (fused prefill, 64 tokens from a 16-token chat prompt): TTFT
0.33 s, decode 15.0 tok/s; 15.1 with the xclbin templates (`DECODE_ELF=0`).

Against FastFlowLM v1.0.7 (`gemma4-it:e4b`) on the same machine:

| prompt tokens | AIR fused prefill | FLM prefill |
|---|---|---|
| 126 | 215 ms | 1265 ms |
| 354 | 511 ms | 1558 ms |
| 731 | 1092 ms | 1899 ms |
| 1368 | 2203 ms | 2753 ms |

Decode (`run_npu2_sweep.lit`, tok/s at KV depth) against FLM at its nearest
measured context:

| context | AIR | FLM |
|---|---|---|
| ~130 | 13.9 | 13.7 |
| 1024 / 1368 | 14.35 | 12.8 |
| 2048 / 2650 | 14.06 | 12.5 |
| 4096 | 13.81 | |
| 8192 | 13.05 | |
| 16384 | 11.74 | |
| 32768 | 9.77 | |

The fused prefill handles prompts
up to the length it was built for, 2048 tokens unless built with
`make compile-fused-prefill FUSED_MAX_LEN=N` (two kv heads put each head's
records `max_blocks` apart, so the build fixes the length); longer ones take the
per-op prefill.
