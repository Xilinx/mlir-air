# OLMo2-1B on NPU2

This example runs full-input prefill, all 16 transformer blocks and the complete output head on the NPU. Prefill GEMMs use **BFP16 compute with FP32 accumulation**, through `AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`, matching AIR's shared external GEMM path. Model weights use group-32 affine Q4; embeddings, normalization parameters and biases retain BF16. Decode uses one fused submission containing the 16 blocks and Q4 head.

Post-attention/post-MLP RMSNorm and full-width Q/K normalization with multi-head attention.

The checkpoint is [`allenai/OLMo-2-0425-1B-Instruct`](https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct/tree/48d788eca847d4d7548f375ad03d3c9312f6139e) at revision `48d788eca847d4d7548f375ad03d3c9312f6139e`. `model.json` pins the checkpoint config and shard hashes. `inputs.json` contains the fixed input token IDs, without expected outputs.

| Input length P | KV capacity C |
| --- | --- |
| 512 | 1024 |
| 1024 | 2048 |
| 2048 | 4096 |

Only these native-context configurations are exposed. Both execution and compilation reject other lengths. Batch size is one.

## Usage

See [build, run, verification and profiling commands](../shared/AFFINE_Q4_DEPLOYMENTS.md).
`make help` lists the common interface. `make compile-all`, `make verify-all` and
`make profile-all` visit this model's native configurations sequentially.

## Implementation and measured results

See the [architecture and source guide](../shared/AFFINE_Q4_IMPLEMENTATION.md) for the model-specific mapping and common infrastructure. [Fresh upstream verification and profiling](../shared/AFFINE_Q4_RESULTS.md) lists all native configurations; the accompanying JSON retains gate hashes and individual request samples.
