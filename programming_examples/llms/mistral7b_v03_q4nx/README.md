# Mistral-7B-Instruct-v0.3 on NPU2

This example runs full-input prefill, all 32 transformer blocks and the complete output head on the NPU. Prefill GEMMs use **BFP16 compute with FP32 accumulation**, through `AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`, matching AIR's shared external GEMM path. Model weights use group-32 affine Q4; embeddings, normalization parameters and biases retain BF16. Decode uses one fused submission containing the 32 blocks and two full-head waves.

Grouped-query attention with 32 query heads / 8 KV heads, head dimension 128, and an independent 32768-row head.

The checkpoint is [`mistralai/Mistral-7B-Instruct-v0.3`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/tree/c170c708c41dac9275d15a8fff4eca08d52bab71) at revision `c170c708c41dac9275d15a8fff4eca08d52bab71`. `model.json` pins the checkpoint config and shard hashes. `inputs.json` contains the fixed input token IDs, without expected outputs.

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
