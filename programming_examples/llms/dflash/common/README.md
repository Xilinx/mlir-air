# Shared DFlash implementation

The common package is used by the Qwen3-4B and Qwen3-8B B8 examples. Each model
still builds separate target and drafter programs. Equal source files do not
make model weights, physical layouts or binaries interchangeable.

| Source | Shared responsibility | Model-specific input |
|---|---|---|
| [speculative.py](speculative.py) | Draft/verify/accept/commit loop and timing | Decoder pair, mask/EOS IDs, feature taps and block size |
| [streaming.py](streaming.py) | Incremental token-to-text output | Tokenizer |
| [decoder.py](decoder.py), [decode_dynseq.py](decode_dynseq.py) | Dispatch, hidden/logit access and KV state operations | Artifact geometry, ordered BO binding and weight provider |
| [buffers.py](buffers.py) | Bounded physical ABI and logical tensor views | 8B's grouped weights and state layout |
| [weight_io.py](weight_io.py), [q4nx_blocks.py](q4nx_blocks.py) | Packed upload and raw Q4NX block helpers | Each model's packing policy |
| [kernels/](kernels/) | Projection, attention, RMS/residual, cache, GLU and prepass arithmetic | Dimensions, row groups and explicit arithmetic flags |
| [fused_attention_graph.py](fused_attention_graph.py) | Query/KV supply, attention shards and output gathering | Layout, placement and selected attention options |
| [build_prepass.py](build_prepass.py), [prepass.py](prepass.py), [prepass_runtime.py](prepass_runtime.py) | Standalone NPU context preparation | Hidden/feature widths and learned FC/KV weights |
| [context_reference.py](context_reference.py), [prefill_reference.py](prefill_reference.py) | CPU arithmetic references and tapped prefill | Model weights and semantic parameters |
| [benchmark_runner.py](benchmark_runner.py), [paired_benchmark.py](paired_benchmark.py) | Fixed-cohort timing, aggregation and engine pairing | Explicit model loaders and validated prompt states |
| [benchmark/prompts.json](benchmark/prompts.json) | Shared messages, source indices and frozen selection order | Tokenization and KV/features are generated for each model |
| [attention_check.py](attention_check.py) | Production-linked attention arithmetic, visibility and repeatability probe | Model configuration and compiled attention object |
| [build_identity.py](build_identity.py) | Source/tool/artifact identity and stale-build rejection | Model build inputs |

The complete AIR assembly and residual/cache ownership are explicit in each
model directory. The 4B mapping uses one eight-row owner/cache pair; 8B uses two
four-row pairs. Arithmetic options preserve each selected profile. B16 and
additional models can extend these boundaries without changing the host
acceptance algorithm or pretending that the physical mappings are identical.

Each model's `paths.py` locates this package when invoked as a standalone script.
The model examples do not import another DFlash model's implementation directory.
They continue to reuse existing upstream autoregressive and prefill utilities.
