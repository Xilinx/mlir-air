# Shared DFlash implementation

Shared code for the Qwen3-4B, Qwen3-8B and Llama-3.1-8B B8 examples. Each
model builds its own target, drafter and prepass programs from these sources,
with its own weights and physical layout.

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
| [b8_graph.py](b8_graph.py), [b8_owners.py](b8_owners.py), [b8_config.py](b8_config.py) | Complete two-owner B8 AIR mapping and physical ABI for the 8B models | Model-local `model_spec.py`: role geometry, normalization, RoPE and head layout |
| [decoder_8b.py](decoder_8b.py) | 8B weight binding | Native model readers and model specification |
| [prefill_weights_8b.py](prefill_weights_8b.py), [prefill_cache.py](prefill_cache.py) | Mapped BF16 prefill weights and one streamed device weight set | Native dequantizer and model tensor shapes |
| [model.py](model.py) | `DFlashModel`, what the host code below needs from a model | Each model's `dflash_model.py` |
| [runtime.py](runtime.py), [run.py](run.py), [verify_loop.py](verify_loop.py), [profile.py](profile.py) | Pipeline loading, streaming generation and chat, loop verification and the fixed-prompt profile | The model's `DFlashModel` |
| [build_target.py](build_target.py) | Target/drafter objects, AIR compilation, ABI and manifest | The model's `target_config`, selected configurations and kernel defines for each role |
| [resources.py](resources.py) | Pinned snapshot download and offline completeness checks | Target, drafter and reference repositories and revisions |
| [check_decoder.py](check_decoder.py), [check_model.py](check_model.py), [check_prepass.py](check_prepass.py) | Decoder state, model top-5 and context prepass checks | The model's `DFlashModel`; the HF reference geometry comes from its config |

The 4B AIR assembly stays in its model directory and uses one eight-row
owner/cache pair. The two 8B models use the shared assembly above with two
four-row pairs. Arithmetic options preserve each selected profile.

Each model's `paths.py` locates this package when invoked as a standalone script.
The model's entry scripts pass its `DFlashModel`, built in `dflash_model.py`, to
the shared code.
The model examples do not import another DFlash model's implementation directory.
They continue to reuse existing upstream autoregressive and prefill utilities.
