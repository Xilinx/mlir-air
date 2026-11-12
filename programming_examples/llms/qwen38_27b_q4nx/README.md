# Qwen3.8-27B Q4NX decode on AMD NPU2

This implementation runs the dense Qwen3.8-27B with the weights FastFlowLM ships,
using the higher-level abstractions of the MLIR-AIR dialect on the decode design
of [Qwen3.6-35B-A3B](../qwen36_35b_a3b_q4nx), FastFlowLM's tile map, with that
model's mixture of experts replaced by the dense MLP.

Text generation (greedy autoregressive decode, batch 1) of Qwen3.8-27B with
[`FastFlowLM/Qwen3.8-27B-NPU2`](https://huggingface.co/FastFlowLM/Qwen3.8-27B-NPU2)
at revision `071def594ad89b0dadd104a117722d6090a5fa26`, file `model.q4nx`
(17,097,334,824 bytes, sha256
`2f0c797f525d5bbacb9f0cde5ff2ea06a3b498a6fc55f5cccc8c97d7eaf00fa1`). The logical
quantized values are used as supplied; only their device layout changes. The
vision tower and the multi-token-prediction head are out of scope.

- **prefill** on the shared fused prefill device
  ([`shared/fused_prefill`](../shared/fused_prefill)), as Qwen3.6-35B-A3B's
  prefill runs: every projection a GEMM op reading the bundle's Q4_K blocks,
  the full-attention layers' attention on the device, the Gated DeltaNet scan,
  the norms and the GLU on the host.
- **decode**: **one NPU dispatch per token for the 64 layers**, then a second
  for the Q8 LM head. The KV cache, the conv caches and the Gated DeltaNet
  states stay on the device across tokens; the prefill hands them over, and the
  decode's LM head gives the first token from the prompt's last row.

## Architecture

| | |
|---|---|
| layers | 64: 48 Gated DeltaNet and 16 gated full attention, in groups of 3 + 1 (layer % 4 == 3 is attention) |
| Gated DeltaNet | 16 query / key heads and 48 value heads (each key head serves 3), head dim 128, causal conv1d (kernel 4), FP32 recurrent state 48 x 128 x 128 per layer, gated RMSNorm |
| full attention | 24 query heads, 4 KV heads, head dim 256, q/k RMSNorm, partial RoPE over 64 of 256 dims (theta 1e7), sigmoid output gate |
| MLP | dense SiLU GLU, intermediate 17408 |
| hidden / vocab | 5120 / 248320, untied Q8 LM head and a Q8 embedding |
| norms | zero-centred (stored as `1 + w`), eps 1e-6 |

## Weights

`qwen38_27b_q4nx_weights.py` reads the bundle from `QWEN38_FLM_DIR` or
`Q4NX_MODEL_SOURCE` (a `model.q4nx` or a directory holding it), else from the
Hub at the pinned revision, checking its sha256. Every projection is **Q4_K**
in blocks of 32 rows x 256 columns (4736 bytes, `w = d * sc * q + dmin * m`);
the LM head and the embedding are **Q8** (8704 bytes per block); the norms and
alpha / beta are bf16.

The decode takes the Q4_K blocks as stored. A layer's dense weights are one
buffer region (about 221 MB), each shim channel's share of it one run of DDR,
and a token reads about 15.5 GB of weights.

## Prefill

`fused_spec.py` configures the device and `prefill.py` drives it, a layer at
a time over the prompt's 128-token chunks. A GEMM op is at most 12288 columns
wide, so the wider projections are split: the Gated DeltaNet's q | k | v and
z, the attention's q | gate and k | v, and the MLP's up | gate (interleaved per
128 rows, as the decode reads them) in three pieces, each piece's GLU written
into its share of the down GEMM's input. Attention runs on columns 0 and 7,
12 query heads and 2 KV heads each. On the host (`kernels/q38_host.c`, with
Qwen3.5's Gated DeltaNet code at 48 value heads): alpha / beta, the causal
conv, the delta rule on an FP32 state, the gated norm, q / k norm and RoPE,
the output gate and the GLU. The prefill keeps its own copy of the weights.

## Decode placement

`decode.py`, one launch repeated once per four layers:

| Columns | Role |
| --- | --- |
| 0, 1, 6, 7 | Q4_K GEMV array: every projection, two shim weight channels per column |
| 2 | RMS norm and residual, RoPE with q/k norm, output gate (attention) or SiLU(z) and the DeltaNet norm (GDN) |
| 3 | attention: 2 units, two KV heads each in turn, 6 query heads per KV head padded to the kernels' 8, the KV cache streamed per token |
| 4 | Gated DeltaNet recurrence, 4 cores of 32 value columns, the state streamed per token |
| 5 | conv1d + SiLU + l2 norm, alpha / beta, the weight issuer, GLU |

Where Qwen3.6's design needed changing at this size:

- **L1 at hidden 5120.** The RMS core holds six [5120] buffers, Qwen3.6's held
  eleven at 2048: the projections' rows add into x as they arrive
  (`kernels/q38_ffn.cc`). The GEMV cores hold one input buffer.
- **Attention.** Each unit runs two KV heads in turn; q is broadcast to the qk
  cores and each unit sends its output to memtile 3 directly.
- **Weights.** An issuer core starts each layer's dense weights, one linear
  transfer per shim channel, a descriptor per layer of a wave.

Kernels shared with Qwen3.6 take this model's sizes as macros
(`kernel_tests/common.py`: `GDN_NV`, `LM_DIM`; `fused_decode_gdn/models/qwen3.8-27b.h`
for attention and RoPE).

## Correctness

- **CPU reference** (`reference.py`): float32 torch on the bundle's dequantized
  weights. `check_vs_hf.py` checks it layer by layer against HF transformers'
  `qwen3_5` decoder layers fed the same weights (cosine 1.0000000 on a GDN and
  an attention layer).
- **`make verify`** (`run_npu2_verify.lit`): the MLP block, a GDN layer, an
  attention layer and the LM head each against the reference; the fused
  prefill of a 300-token prompt against it (last-row logits); the prompt
  prefilled into the 64-layer decode and every greedy token compared with the
  reference's; the reference against HF; and the shared top-5 gate
  ([`verify/`](../verify), `verify_adapter.py`) against `Qwen/Qwen3.8-27B` in
  bf16.

## Results

NPU2 (Strix Halo), NPU otherwise idle.

Prefill (`run_npu2_prefill_sweep.lit`): time to first token, the prompt
prefilled then the first token from the LM head, against FastFlowLM 1.0.11 on
the same bundle and machine (`flm bench`, mean TTFT):

| prompt | TTFT | tok/s | FLM TTFT |
| --- | --- | --- | --- |
| 128 | 1.43 s | 90 | |
| 512 | 3.26 s | 157 | |
| 1024 | 6.43 s | 159 | 9.19 s |
| 2048 | 12.94 s | 158 | 16.93 s |
| 4096 | 26.28 s | 156 | 32.82 s |

Decode, the 64-layer dispatch and the LM head's timed separately:

| context | layers | LM head | per token |
| --- | --- | --- | --- |
| 6 | 268 ms | 25 ms | 293 ms (3.41 tok/s) |
| 1024 | 271 ms | 25 ms | 297 ms (3.37 tok/s) |
| 8192 | 316 ms | 25 ms | 341 ms (2.93 tok/s) |
| 32768 | 468 ms | 25 ms | 493 ms (2.03 tok/s) |
| 65536 | 671 ms | 25 ms | 696 ms (1.44 tok/s) |

At short contexts that is about 53 GB/s of weights, the NPU's DDR read rate on
this machine; every attention layer reads its whole KV cache, which the longer
contexts add. A 64k build's KV cache (4.3 GB) is past 4 GB: its rows land where
they belong. At 128k aircc fails in its wrap-and-stride canonicalization.

Against FastFlowLM 1.0.11 on the same bundle and machine (`flm bench
qwen3.8-mtp:27b -i bench_4k.json --bench-iterations 3`, mean decode tok/s; FLM
without speculation is `FLM_DISABLE_SPECULATION=1`):

| context | this decode | FLM, no speculation | FLM with its MTP drafts |
| --- | --- | --- | --- |
| 1024 | 3.38 | 0.24 | 1.19 |
| 2048 | | 0.24 | 1.65 |
| 4096 | 3.17 | 0.25 | 1.66 |

## Prerequisites

- NPU2 (Strix / AIE2P) + XRT, a Peano (llvm-aie) install
- `source utils/env_setup.sh` (see the repo README), `HF_HOME` for the Hub
- `pip install -r requirements.txt`
- About 31 GB of host memory pinned for the weights: the decode's 16 GB and the
  prefill's own 15 GB copy

## Usage

```bash
make compile                 # the decode and LM head ELFs
make compile-fused-prefill   # the fused prefill device (also built on demand)
make run                     # prefill + greedy decode of PROMPT against the CPU reference
make prefill-fused           # the fused prefill of a LONG_LEN-token prompt vs the reference
make verify                  # each stage, prefill, decode, HF check, top-5 gate
make profile                 # TTFT and decode tok/s, no reference
make profile-prefill CTX=2048   # TTFT of a CTX-token prompt
make profile-decode CONTEXTS=1024,8192   # decode tok/s per context
make kernels                 # the kernel tests
```
