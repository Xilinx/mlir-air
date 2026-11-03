# Phi-2 on NPU2

This example runs the prefill, all 32 transformer blocks and the complete output head on the NPU. The prefill runs on the repository's shared [fused prefill](../shared/fused_prefill/phi2.py): every projection GEMM, the attention and the prefill's output head are ops of one device on one hardware context, and inputs of any length up to 2048 tokens run in 128-row chunks. Its GEMMs dequantize the group-32 affine Q4 weights on the cores and compute in **BFP16 with FP32 accumulation**. Model weights use group-32 affine Q4; embeddings, normalization parameters and biases retain BF16. Decode uses one fused body submission plus a separate final LayerNorm/full-head submission.

LayerNorm with learned bias, parallel attention/MLP residual branches, gelu_new, projection/head biases, and 32-of-80 partial RoPE. Device head dimensions are zero-padded to 128.

The checkpoint is [`microsoft/phi-2`](https://huggingface.co/microsoft/phi-2/tree/810d367871c1d460086d9f82db8696f2e0a0fcd0) at revision `810d367871c1d460086d9f82db8696f2e0a0fcd0`. `model.json` pins the checkpoint config and shard hashes. `inputs.json` contains the fixed input token IDs, without expected outputs.

| Input length P | KV capacity C |
| --- | --- |
| 512 | 1024 |
| 1024 | 2048 |

These are the native configurations: the fixed-input gate, the CI tests and the profiles use them. One fused prefill build serves all of them and takes any input of up to 2048 tokens; a configuration's KV capacity C bounds the input plus the outputs. Batch size is one.

## Usage

See [build, run, verification and profiling commands](../shared/AFFINE_Q4_DEPLOYMENTS.md).
`make help` lists the common interface. `make compile-all`, `make verify-all` and
`make profile-all` visit this model's native configurations sequentially.
`make verify`, `make accuracy` and `make tasks` run the shared [verify subsystem](../verify/README.md)
through `verify_adapter.py`; the fixed-input gate is `make verify-native`.

## Prefill on the fused prefill

One biased LayerNorm per chunk feeds both the qkv and the fc1 GEMM. The host
adds the q/k/v, fc1, dense and fc2 biases, applies RoPE to the first 32 of each
head's 80 dims, runs GELU (tanh form) and the residual add of both branches,
and overlaps that work with the device's. The device's attention runs at head
dim 128: each head's 80 values sit at column 0 (even heads) or 16 (odd heads)
of its 128, as in the decode's KV, so the prefill's K and V rows copy into the
fused decode's KV buffer unchanged, and the dense projection reads the
attention output with its canonical Q4 weights spread over the same columns.

## Implementation and measured results

See the [architecture and source guide](../shared/AFFINE_Q4_IMPLEMENTATION.md) for the model-specific mapping and common infrastructure. [Fresh upstream verification and profiling](../shared/AFFINE_Q4_RESULTS.md) lists all native configurations as measured with the earlier per-op prefill; the accompanying JSON retains gate hashes and individual request samples.
