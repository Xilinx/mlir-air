# OLMo2-1B on NPU2

This example runs the prefill, all 16 transformer blocks and the complete output head on the NPU. The prefill runs on the repository's shared [fused prefill](../shared/fused_prefill/affine.py): every projection GEMM, the attention and the prefill's output head are ops of one device on one hardware context, and inputs of any length up to 2048 tokens run in 128-row chunks. Its GEMMs dequantize the group-32 affine Q4 weights on the cores and compute in **BFP16 with FP32 accumulation**. Embeddings, normalization parameters and biases retain BF16. Decode uses one fused submission containing the 16 blocks and Q4 head.

Post-attention/post-MLP RMSNorm and full-width Q/K normalization with multi-head attention.

The checkpoint is [`allenai/OLMo-2-0425-1B-Instruct`](https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct/tree/48d788eca847d4d7548f375ad03d3c9312f6139e) at revision `48d788eca847d4d7548f375ad03d3c9312f6139e`. `model.json` pins the checkpoint config and shard hashes. `inputs.json` contains the fixed input token IDs, without expected outputs.

| Input length P | KV capacity C |
| --- | --- |
| 512 | 1024 |
| 1024 | 2048 |
| 2048 | 4096 |

These are the native configurations: the fixed-input gate, the CI tests and the profiles use them. One fused prefill build serves all of them and takes any input of up to 2048 tokens; a configuration's KV capacity C bounds the input plus the outputs. Batch size is one.

## Usage

See [build, run, verification and profiling commands](../shared/AFFINE_Q4_DEPLOYMENTS.md).
`make help` lists the common interface. `make compile-all`, `make verify-all` and
`make profile-all` visit this model's native configurations sequentially.
`make verify`, `make accuracy` and `make tasks` run the shared [verify subsystem](../verify/README.md)
through `verify_adapter.py`; the fixed-input gate is `make verify-native`.

## Prefill on the fused prefill

The host runs the embedding lookup, the norms, RoPE, residual adds, SwiGLU and the
operand layouts, and overlaps them with the device's work on the next chunk.
After the first token is chosen, the prefill's K and V rows are copied into the
fused decode's KV buffer. OLMo2 normalizes each sublayer's output (post-attention and post-feedforward RMSNorm) before the residual add and has no norm before the sublayers, so the GEMMs read the residual stream directly; its q and k RMSNorms span all heads' columns together and run before RoPE (`Desc.post_norm`, `Desc.qk_norm_rows`).

## Implementation and measured results

See the [architecture and source guide](../shared/AFFINE_Q4_IMPLEMENTATION.md) for the model-specific mapping and common infrastructure. [Fresh upstream verification and profiling](../shared/AFFINE_Q4_RESULTS.md) lists all native configurations as measured with the earlier per-op prefill; the accompanying JSON retains gate hashes and individual request samples.
