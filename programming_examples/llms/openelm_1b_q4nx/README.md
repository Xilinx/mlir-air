# OpenELM-1.1B on NPU2

This example runs full-input prefill, all 28 transformer blocks and the complete output head on the NPU. Prefill GEMMs use **BFP16 compute with FP32 accumulation**, through `AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`, matching AIR's shared external GEMM path. Model weights use group-32 affine Q4; embeddings, normalization parameters and biases retain BF16. Decode uses one fused body submission plus a separate full BF16 tied-head submission.

Layer-dependent attention/MLP widths, per-head Q/K normalization and an original BF16 tied embedding/head. The device schedule pads the variable widths.

The checkpoint is [`apple/OpenELM-1_1B-Instruct`](https://huggingface.co/apple/OpenELM-1_1B-Instruct/tree/effd796da2a77361d7360e45e54c7cc14bc5df2a) at revision `effd796da2a77361d7360e45e54c7cc14bc5df2a`. `model.json` pins the checkpoint config and shard hashes. `inputs.json` contains the fixed input token IDs, without expected outputs.

| Input length P | KV capacity C |
| --- | --- |
| 512 | 1024 |
| 1024 | 2048 |

Only these native-context configurations are exposed. Both execution and compilation reject other lengths. Batch size is one.

## Usage

See [build, run, verification and profiling commands](../shared/AFFINE_Q4_DEPLOYMENTS.md).
`make help` lists the common interface. `make compile-all`, `make verify-all` and
`make profile-all` visit this model's native configurations sequentially.

## Implementation and measured results

See the [architecture and source guide](../shared/AFFINE_Q4_IMPLEMENTATION.md) for the model-specific mapping and common infrastructure. [Fresh upstream verification and profiling](../shared/AFFINE_Q4_RESULTS.md) lists all native configurations; the accompanying JSON retains gate hashes and individual request samples.
