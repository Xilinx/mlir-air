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

- **decode**: **one NPU dispatch per token for the 64 layers**, then a second
  for the Q8 LM head. The KV cache, the conv caches and the Gated DeltaNet
  states stay on the device across tokens.
- The prompt goes through the decode a token at a time.

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

## Placement

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
  attention layer and the LM head each against the reference; the prompt
  through the 64-layer decode and every greedy token compared with the
  reference's; the reference against HF.

## Results

NPU2 (Strix Halo), NPU otherwise idle, the 64-layer dispatch and the LM head's
timed separately:

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

## Prerequisites

- NPU2 (Strix / AIE2P) + XRT, a Peano (llvm-aie) install
- `source utils/env_setup.sh` (see the repo README), `HF_HOME` for the Hub
- `pip install -r requirements.txt`
- About 16 GB of host memory pinned for the weights

## Usage

```bash
make compile                 # the decode and LM head ELFs
make run                     # greedy decode of PROMPT against the CPU reference
make verify                  # each stage, the decode, the reference against HF
make profile                 # TTFT and decode tok/s, no reference
make profile-decode CONTEXTS=1024,8192   # decode tok/s per context
make kernels                 # the kernel tests
```
