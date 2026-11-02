# Qwen3.5-0.8B Q4NX prefill + decode on AMD NPU2

This implementation reimplements the corresponding AMD NPU LLM design,
originally developed by the [FastFlowLM](https://github.com/ROCm/FastFlowLM)
team, using the higher-level abstractions of the MLIR-AIR dialect. The decode
fork [`fused_decode_gdn`](../../fused_decode_gdn) ports FastFlowLM's Qwen3.5
decoding layer (`qwen3_5_decoding`) onto
[`fused_decode`](../../fused_decode), the reimplementation of FastFlowLM's
generic decoding layer.

Text generation (prefill and greedy autoregressive decode, batch 1) of
Qwen3.5-0.8B with the weights FastFlowLM itself ships:
[`FastFlowLM/Qwen3.5-0.8B-NPU2`](https://huggingface.co/FastFlowLM/Qwen3.5-0.8B-NPU2),
branch `flm_q4k_high_precision` at revision
`1d16e5eaa2508889fb88eb1bfab1921a30a1a466`, file `model.q4nx` (560,985,936
bytes, sha256 `787428ffe87be5e3e5a8fcaf241844b69896ca0c74a41f2432c436739f7ba4e4`,
the file FastFlowLM 1.0.11 loads). The logical quantized values are used as
supplied; only their device layout changes. The vision encoder is out of scope.

- **prefill**: the prompt in 128-row chunks on the shared
  [fused_prefill](../shared/fused_prefill) device. Every projection GEMM, the
  softmax attention and the lm_head are ops of one xclbin; the element-wise
  work and the Gated DeltaNet scan run on the host, overlapping the device.
- **decode**: **one NPU dispatch per token = 24 layers + the full-vocabulary
  lm_head**, on [`fused_decode_gdn`](../../fused_decode_gdn). The KV cache and
  the Gated DeltaNet recurrent state stay on the device across tokens.

## Architecture

| | |
|---|---|
| layers | 24: 18 Gated DeltaNet and 6 gated full attention, in groups of 3 + 1 (layer % 4 == 3 is attention) |
| Gated DeltaNet | 16 heads, key / value dim 128, causal conv1d (kernel 4), FP32 recurrent state 16 x 128 x 128 per layer, gated RMSNorm |
| full attention | 8 query heads, 2 KV heads, head dim 256, q/k RMSNorm, partial RoPE over 64 of 256 dims (theta 1e7), sigmoid output gate |
| MLP | SwiGLU, intermediate 3584; hidden 1024 |
| head / vocab | 248320, tied to the embedding (the bundle's Q8 `lm_head`) |
| norms | zero-centred (stored as `1 + w`), eps 1e-6 |

## Weights

`q35_weights.py` reads the bundle from the Hub at the pinned revision (cached
under `HF_HOME`) or from `MODEL_SOURCE=<model.q4nx | directory holding it>`, and
checks its sha256 either way. Every quantized tensor is stored in blocks of 32
rows x 256 columns:

- **Q4K** (4736 bytes; every projection but beta / alpha and `lm_head`):
  `w = d * sc * q + dmin * m` with uint8 sub-scales `sc` and mins `m` per row
  and group of 32 columns, bf16 `d`, `dmin` per row and 256 columns.
- **Q8** (8704 bytes; the beta / alpha projections, `lm_head`):
  `w = scale * q`, bf16 scale per row and group.

The bundle has no embedding table: it is tied to `lm_head`, so the host looks
up `lm_head`'s Q8 rows rounded to bf16.

How each side takes them:

- **Decode** (`q35_fdg.pack_weights`): the Q4K blocks as stored, in the
  engine's 5120-byte slot; the kernel computes `d * sum(sc * S) + dmin *
  sum(m * X)` exactly in the bundle's values. A Q8 tensor is carried as two
  affine-Q4 rows per row (`s q = 16 s hi - 128 s + s lo`, exact); the host adds
  the lm_head's row pairs when it reads the logits.
- **Prefill** (BFP16 compute): a Q4K tensor enters the affine-Q4 GEMM with
  per-group scale and min `bf16(d * sc)`, `bf16(dmin * m)`; a Q8 tensor as two
  Q4 matrices stacked along K against `[x | x]` (exact). The bundle stores the
  attention q_proj as `[8 heads' queries | 8 heads' gates]` and the norms as
  `1 + w`; both are accounted for.

## Placement

Prefill (`shared/fused_prefill/qwen35.py`, device spec `q35_spec.py`):

| Computation | Where | Implementation |
| --- | --- | --- |
| every projection GEMM, 24 layers (attention q \| gate \| k \| v and o; DeltaNet q \| k \| v \| z, beta \| alpha, out_proj; MLP gate \| up, down) | NPU | fused_prefill GEMM ops, int4 weights dequantized by the cores, BFP16 compute; chunk pairs share one weight read |
| softmax attention of the 6 attention layers (causal, head dim 256) | NPU | the device's `attn_bfp16` op: query heads 0-3 on column 0 with KV head 0, heads 4-7 on column 7 with KV head 1 |
| lm_head (Q8) of the last row | NPU | 21 GEMM ops (two Q4 halves along K) |
| RMSNorms, residual adds, q/k norm, partial RoPE, K/V records | host | `hostops` (shared) |
| attention output gate, SiLU(gate) * up | host | `kernels/q35_host.c` |
| Gated DeltaNet: conv1d (3 rows carried across chunks) + SiLU, q/k l2-norm, gates, the delta rule on an FP32 state carried across chunks, gated RMSNorm | host | `kernels/q35_host.c` (`gdn_chunk`): independent recurrences per head and 32 value columns on AVX-512 |
| embedding lookup, final norm of the last row | host | |

The host prepares a chunk's A tiles, queues its GEMM, and takes each chunk's
output as it completes; the MLP runs one to three chunk pairs behind the mixer,
so the device's MLP GEMMs overlap the host's mixer work. The delta-rule scan is
on the host because it measured faster there than on the NPU (19 / 33 / 57 ms
per prefill at 1024 / 2048 / 4096 tokens against an NPU scan's 98 / 190 / 364
ms).

Decode (`q35_fdg.py` on [`fused_decode_gdn`](../../fused_decode_gdn), whose
README describes the design): one full ELF per capacity C (the KV cache
length), the context length passed to the device as scratchpad parameters.
Per token the host writes the embedding row and the attention layers' cos /
sin row, swaps the Gated DeltaNet herd's state buffers, dispatches once, reads
the logits and takes the argmax. After prefill the host hands the prompt's K/V
rows, conv history and DeltaNet states (bf16 hi + lo words) to the decode's
buffers once, at its first step.

Why a fork of `fused_decode` rather than a model entry: the Gated DeltaNet's
recurrent state (1 MiB per layer, read and written every token) needs its own
cores and DMA channels, which the parent engine has no room for (as with
Gemma4's per-layer embeddings and `fused_decode_ple`). FastFlowLM likewise
forks its generic decoding layer for Qwen3.5. `fused_decode` and
`fused_decode_ple` are unchanged.

## Correctness

**Reference: `Qwen/Qwen3.5-0.8B` @ `2fc06364715b967f1860aea9cf38778875588b17`,
CPU BF16.** The bundle does not name its source checkpoint. Every one of its
187 quantized tensors, dequantized, is closer to this post-trained checkpoint
than to `Qwen/Qwen3.5-0.8B-Base` @ `dc7cdfe2` (median relative L2 0.073 against
0.099); the Q8 `lm_head` is 0.0058 from the post-trained tied embedding against
0.045 from the base one; the conv1d weights are bit-identical to the
post-trained checkpoint.

- **`make verify`**, the repository's gate (`run_npu2_verify.lit`): top-5
  token-set inclusion at the first divergence over 32 greedy tokens, 2 prompts,
  against the reference through `llms/verify` and
  [`verify_adapter.py`](verify_adapter.py), on the production path at
  capacity 1024.
- **`make gate-all` + `make fullfit-all`** (`gate.py`), kept because it covers
  what `make verify` does not: the long-context configurations. It runs the
  production path at each of the four configurations (prefill bucket P /
  capacity C: 512/1024, 1024/2048, 2048/4096, 4096/8192) on the 8 prompts of
  `llms/verify/prompts/instruct.txt` and on prompts that fill the bucket
  (P tokens; one at P512, two at P1024, ...). Method: greedy 32 tokens, EOS
  ignored, complete finite full-vocabulary logits at every step, mutual top-5 at
  the first divergence against the CPU BF16 captures of `make reference` /
  `make reference-timing`.

## Results

Measured on 2026-11-01 with MLIR-AIR built from this tree's base (`8e9bfc1a`),
Peano (llvm-aie) 22.0.0.2026102001, mlir-aie 1.4.5.dev83, XRT 2.21.0, on NPU2
(Strix).

**Gate**: `make verify` PASS (2/2, first divergence at steps 0 and 28). Model
gate PASS in every configuration: 8/8 instruct prompts (first divergence per
prompt 0, 28, 21, 19, 9, 17, 22, 5) and the bucket-filling prompts (1/1, 2/2,
3/3, 4/4, no divergence); 32/32 steps captured per prompt, all logits finite.

**Timing** (`make bench-all`): per configuration a prompt of exactly P tokens
(the instruct prompts' text repeated), fresh greedy requests of 128 output
tokens with EOS ignored; at least 5 s of warm-up requests, then at least 3
requests and 10 s measured; medians. TTFT is input ids on the host to the first
token chosen on the host (embedding, prefill, final norm, lm_head, argmax);
decode is the 127 following steps (127 / decode time, including the one-time
hand-off of the prefill's state); the request is both. Mean of the medians of
two interleaved runs (four at P4096):

| Config | TTFT | Decode | Request | FastFlowLM 1.0.11: TTFT / decode |
| --- | --- | --- | --- | --- |
| P512 / C1024 | 105 ms | 45.06 tok/s | 2.93 s | |
| P1024 / C2048 | 174 ms | 44.49 tok/s | 3.03 s | 630 ms / 44.16 tok/s (976 tokens) |
| P2048 / C4096 | 340 ms | 43.54 tok/s | 3.26 s | 1031 ms / 42.91 tok/s (1943 tokens) |
| P4096 / C8192 | 719 ms | 41.75 tok/s | 3.76 s | 1840 ms / 40.38 tok/s (3877 tokens) |

FastFlowLM 1.0.11 ran on the same weights (its own copy of this bundle) with
`flm bench qwen3.5:0.8b`, prompts of the listed lengths, 3 iterations, on the
same machine on 2026-11-01. The prefill is 2.6-3.6x faster than FastFlowLM's;
decode is faster at every length.
Per decode token the host spends about 0.5 ms (the embedding row, the cos /
sin rows, the logits read with the Q8 row-pair adds); the rest is the one
dispatch (21.3 ms at P512).

Where the decode's time goes (a build whose kernels return at entry, with the
same dataflow, at P512): 18.1 ms of the 21.4
ms per token is the dataflow, about 3.5 ms exposed compute (the Gated DeltaNet
step, the projections).

## Prerequisites

- NPU2 (Strix / AIE2P) + XRT, a Peano (llvm-aie >= 22) install, and an
  `llvm-link` older than LLVM 23 on `PATH` for the decode build (see
  [`../../fused_decode/README.md`](../../fused_decode/README.md))
- `source utils/env_setup.sh` (see the repo README), `HF_HOME` for the Hub
  cache
- `pip install -r requirements.txt`

## Build and run

```bash
make compile                      # prefill device + the decode ELF of C = 1024, 2048, 4096, 8192
make compile CAPACITIES=1024      # ... or one capacity
make run C=1024 PROMPT="Introduce me what is GPU"
make verify                       # the CI gate (verify-full: every prompt)
make profile                      # TTFT + decode tok/s (one warm-up request first)
make profile-fused-prefill        # prefill time at 512 / 1024 / 2048 / 4096 tokens

# the model gate and the timing protocol (OUT defaults to ./out)
make reference reference-timing   # CPU BF16 captures (minutes)
make gate-all fullfit-all         # 4 configurations
make bench-all                    # complete-request timing, 4 configurations
```

Builds go to `BUILD` (default `./build`); NPU runs should be serialized, e.g.
`flock /tmp/mlir-air-npu.lock make ...`.

## Tests

| lit | What it checks |
| --- | --- |
| `run_npu2_compile.lit` | the fused_decode_gdn full ELF (C = 1024) compiles; no NPU, no weights |
| `run_npu2_compile_fused_prefill.lit` | the prefill device compiles; no NPU, no weights |
| `run_npu2_verify.lit` | `make verify` (PASS) |
| `run_npu2_profile.lit` | `make profile` into `qwen35_0_8b_q4nx.perf.json` |
| `run_npu2_prefill_sweep.lit` | prefill time at 512-4096 tokens (`bench/sweep_prefill.py --fused`) |
| `run_npu2_sweep.lit` | decode tok/s at contexts 2048, 8192 and 32768 (`make profile-decode` through `bench/sweep_decode_runtime.py`) |

The decode sweep does not use `bench/sweep_decode.py`, which drives
`fused_decode`'s template builds and its `bench_decode` geometry. The fork's
context is compiled into each ELF, so `make profile-decode` builds the ELF of
each context and `bench.py --decode-only` times a step at its last position
over synthetic cache rows, one process for the list. `make bench-all` times
complete requests at capacities 1024-8192. The sweep stops at 32768: the decode's
KV buffer keeps a slot for each of the 24 layers, not only the 6 attention layers,
and at a 131072 capacity it exceeds the buffer offsets the decode can address
(the dispatch times out).

## Files

| File | Purpose |
| --- | --- |
| `q35_npu.py` | Production model (`Qwen35Npu`): `prefill`, `decode`, `generate`; `make run` / `profile` |
| `q35_fdg.py` | The decode on fused_decode_gdn: weight packing, prefill hand-off, per-token dispatch (`FdgDecode`) |
| `../../fused_decode_gdn/` | The decode fork: builder, kernels, model header |
| `../shared/fused_prefill/qwen35.py` | The prefill on the fused_prefill device (`Qwen35Prefill`) |
| `q35_spec.py`, `device_build.py` | The prefill device: GEMM shapes, attention groups, build |
| `kernels/q35_host.c` | The prefill's host C: Gated DeltaNet chunk, output gate, SwiGLU |
| `q35_weights.py` | The bundle reader (Q4K, Q8), tied embedding, Q8 as Q4 halves / row pairs |
| `verify_adapter.py` | `make verify`'s adapter |
| `reference.py`, `gate.py`, `bench.py`, `bench_prompts.py` | CPU BF16 reference, model gate, complete-request timing, bucket-filling prompts |

## Known limitations

- The Q8 lm_head's two row halves are added on the host while reading the
  logits (248320 adds per token); the products are on the NPU.
- The attention result depends on what lies past the context in the last
  16-position block of the KV cache, so the hand-off zeroes the cache rows
  that earlier requests left behind. Inherited from `fused_decode`'s attention
  path; not diagnosed further.
- Prompts up to 4096 tokens (the prefill device's KV records); prompt +
  generated tokens up to the decode capacity (8192 at most).
