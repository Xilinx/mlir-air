# Granite-3.3-2B on NPU2

This example runs full-input prefill, all 40 transformer blocks and the complete output head on the NPU. Prefill GEMMs use **BFP16 compute with FP32 accumulation**, through `AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`, matching AIR's shared external GEMM path. Model weights use group-32 affine Q4; embeddings, normalization parameters and biases retain BF16. Decode uses one fused submission containing the 40 blocks and Q4 head.

Granite residual scaling (0.22), embedding scaling (12), attention scaling and output-logit scaling (1/8).

The checkpoint is [`ibm-granite/granite-3.3-2b-instruct`](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct/tree/707f574c62054322f6b5b04b6d075f0a8f05e0f0) at revision `707f574c62054322f6b5b04b6d075f0a8f05e0f0`. `model.json` pins the checkpoint config and shard hashes. `inputs.json` contains the fixed input token IDs, without expected outputs.

| Input length P | KV capacity C |
| --- | --- |
| 512 | 1024 |
| 1024 | 2048 |
| 2048 | 4096 |
| 4096 | 8192 |

Only these native-context configurations are exposed. Both execution and compilation reject other lengths. Batch size is one.

## Usage

See [build, run, verification and profiling commands](../shared/AFFINE_Q4_DEPLOYMENTS.md).
`make help` lists the common interface. `make compile-all`, `make verify-all` and
`make profile-all` visit this model's native configurations sequentially.

## Implementation and measured results

See the [architecture and source guide](../shared/AFFINE_Q4_IMPLEMENTATION.md) for the model-specific mapping and common infrastructure. [Fresh upstream verification and profiling](../shared/AFFINE_Q4_RESULTS.md) lists all native configurations; the accompanying JSON retains gate hashes and individual request samples.
