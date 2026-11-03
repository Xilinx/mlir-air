# Granite-3.3-2B on NPU2

This example runs the prefill, all 40 transformer blocks and the complete output head on the NPU. The prefill runs on the repository's shared [fused prefill](../shared/fused_prefill/affine.py): every projection GEMM, the attention and the prefill's output head are ops of one device on one hardware context, and inputs of any length up to 4096 tokens run in 128-row chunks. Its GEMMs dequantize the group-32 affine Q4 weights on the cores and compute in **BFP16 with FP32 accumulation**. Embeddings, normalization parameters and biases retain BF16. Decode uses one fused submission containing the 40 blocks and Q4 head. Granite's embedding multiplier (12), attention multiplier (1/64), residual multiplier (0.22) and logit scaling (1/8) come from its `config.json`.

Granite residual scaling (0.22), embedding scaling (12), attention scaling and output-logit scaling (1/8).

The checkpoint is [`ibm-granite/granite-3.3-2b-instruct`](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct/tree/707f574c62054322f6b5b04b6d075f0a8f05e0f0) at revision `707f574c62054322f6b5b04b6d075f0a8f05e0f0`. `model.json` pins the checkpoint config and shard hashes. `inputs.json` contains the fixed input token IDs, without expected outputs.

| Input length P | KV capacity C |
| --- | --- |
| 512 | 1024 |
| 1024 | 2048 |
| 2048 | 4096 |
| 4096 | 8192 |

These are the native configurations: the fixed-input gate, the CI tests and the profiles use them. One fused prefill build serves all of them and takes any input of up to 4096 tokens; a configuration's KV capacity C bounds the input plus the outputs. Batch size is one.

## Usage

See [build, run, verification and profiling commands](../shared/AFFINE_Q4_DEPLOYMENTS.md).
`make help` lists the common interface. `make compile-all`, `make verify-all` and
`make profile-all` visit this model's native configurations sequentially.
`make verify`, `make accuracy` and `make tasks` run the shared [verify subsystem](../verify/README.md)
through `verify_adapter.py`; the fixed-input gate is `make verify-native`.

## Prefill on the fused prefill

The host runs the embedding lookup, RMSNorm, RoPE, residual adds, SwiGLU and the
operand layouts, and overlaps them with the device's work on the next chunk.
After the first token is chosen, the prefill's roped K and V rows are copied
into the fused decode's KV buffer. The previous prefill composed Llama's per-op
blocks for one fixed length per configuration; it has been removed.

Measured on NPU2 (AMD Ryzen AI Max+ 395, Strix Halo), fixed full-input requests at each
native P, 128 outputs, fresh requests, at least 5 s warmup and at least three
requests totaling at least 10 s (the [profile protocol](../shared/AFFINE_Q4_IMPLEMENTATION.md)).
Both builds were measured back to back in one session; "per-op prefill" is the
previous implementation, "fused prefill" this one. Decode is the unchanged fused decode.

| Input length P | TTFT, per-op prefill (s) | TTFT, fused prefill (s) | Decode, per-op build (tok/s) | Decode, fused build (tok/s) |
| --- | ---: | ---: | ---: | ---: |
| 512 | 0.616 | 0.285 | 29.32 | 29.31 |
| 1024 | 1.218 | 0.563 | 27.32 | 27.28 |
| 2048 | 2.274 | 1.346 | 23.95 | 23.97 |
| 4096 | 4.570 | 3.786 | 19.08 | 19.23 |

Accuracy (`make accuracy`, wikitext-2, 4 chunks of 2048 tokens, P1024/C2048
configuration):

| Reference | NPU path | KLD | PPL (NPU / reference) | Same top token |
| --- | --- | ---: | ---: | ---: |
| BF16 checkpoint | prefill | 0.1250 | 9.599 / 9.019 | 85.1% |
| BF16 checkpoint | decode | 0.1629 | 9.641 / 9.019 | 82.5% |
| affine-Q4 weights, dequantized | prefill | 0.0048 | 9.599 / 9.569 | 96.9% |
| affine-Q4 weights, dequantized | decode | 0.0397 | 9.641 / 9.569 | 91.3% |

## Implementation and measured results

See the [architecture and source guide](../shared/AFFINE_Q4_IMPLEMENTATION.md) for the model-specific mapping and common infrastructure. [Fresh upstream verification and profiling](../shared/AFFINE_Q4_RESULTS.md) lists all native configurations as measured with the earlier per-op prefill; the accompanying JSON retains gate hashes and individual request samples.
