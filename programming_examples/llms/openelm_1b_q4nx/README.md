# OpenELM-1.1B on NPU2

This example runs the prefill, all 28 transformer blocks and the complete output head on the NPU. The prefill runs on the repository's shared [fused prefill](../shared/fused_prefill/affine.py): every projection GEMM, the attention and the prefill's output head are ops of one device on one hardware context, and inputs of any length up to 2048 tokens run in 128-row chunks. Its GEMMs dequantize the group-32 affine Q4 weights on the cores and compute in **BFP16 with FP32 accumulation**. Model weights use group-32 affine Q4; embeddings, normalization parameters and biases retain BF16. Decode uses one fused body submission plus a separate full BF16 tied-head submission.

Layer-dependent attention/MLP widths, per-head Q/K normalization and an original BF16 tied embedding/head. Both the prefill and the decode are built for the widest layer and zero-pad the narrower ones.

The checkpoint is [`apple/OpenELM-1_1B-Instruct`](https://huggingface.co/apple/OpenELM-1_1B-Instruct/tree/effd796da2a77361d7360e45e54c7cc14bc5df2a) at revision `effd796da2a77361d7360e45e54c7cc14bc5df2a`. `model.json` pins the checkpoint config and shard hashes. `inputs.json` contains the fixed input token IDs, without expected outputs.

| Input length P | KV capacity C |
| --- | --- |
| 512 | 1024 |
| 1024 | 2048 |

These are the native configurations: the fixed-input gate, the CI tests and the profiles use them. One fused prefill build serves all of them and takes any input of up to 2048 tokens; a configuration's KV capacity C bounds the input plus the outputs. Batch size is one.

## Usage

See [build, run, verification and profiling commands](../shared/AFFINE_Q4_DEPLOYMENTS.md).
`make help` lists the common interface. `make compile-all`, `make verify-all` and
`make profile-all` visit this model's native configurations sequentially.
`make profile-fused-prefill` times the fused prefill alone. The verify/ tools
(`make verify` on other models) need a tokenizer and an HF reference that the
OpenELM checkpoint does not ship, so `make verify` stays the fixed-input gate.

## Prefill on the fused prefill

The host runs the embedding lookup, the RMSNorms, the per-head q/k RMSNorms,
RoPE, residual adds, SwiGLU and the operand layouts, and overlaps them with the
device's work on the next chunk. After the first token is chosen, the
prefill's K and V rows are copied into the fused decode's KV buffer. The
device is the shape of the widest layer (32 q heads, 8 KV heads, FFN 8192);
`OpenELMPrefill` reads each layer's widths from the checkpoint's config.json
and zero-pads its weights up to that shape, as the decode does.

## Implementation and measured results

The decode uses OpenELM's local normalization kernels: normalize in FP32,
round to BF16, then multiply by the learned BF16 weight. Decode source lookup
gives the model's `kernels/` directory precedence over the shared kernels.

See the [architecture and source guide](../shared/AFFINE_Q4_IMPLEMENTATION.md) for the model-specific mapping and common infrastructure. [Fresh upstream verification and profiling](../shared/AFFINE_Q4_RESULTS.md) lists all native configurations as measured with the earlier per-op prefill; the accompanying JSON retains gate hashes and individual request samples.
