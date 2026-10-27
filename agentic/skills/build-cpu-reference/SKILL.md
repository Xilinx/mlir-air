---
name: build-cpu-reference
description: "Prepare model configuration, weights, production CPU helpers and the task-appropriate runnable reference before NPU validation; distinguish primitive arithmetic references from full-model token references."
---

# Prepare weights and reference

A hand-written full-model FP32 oracle is not required. Read the task's [validation contract](../../references/validation-contracts.md) and [source profile](../../references/project-integration.md). The inherited AIR path uses HF BF16 for model comparison and independent primitive references inside harnesses; another arithmetic/task may require another fixed reference.

## Resolve semantics before packing

Read checkpoint config and reference implementation. Record layer/embedding/intermediate/vocabulary dimensions, attention and KV head counts, explicit head dimension, norm epsilon and ordering, attention scaling/mask, positional/RoPE variant, bias, tied embeddings, tokenizer/chat template and state semantics. Use defaults only when the actual model implementation defines them. In particular, do not infer QKV bias, Q/K norm or attention geometry from the model-family name.

Map every needed tensor by name, layer and semantic axis. Check q/k/v/o and gate/up/down projections and normalization weights for the selected architecture, plus biases, per-head norms or stateful operators where present. Inspect the safetensors index/metadata and actual shapes. Resolve tied LM-head weights according to config. Record transpose, quantization/packing and padding as transformations of those semantic axes, not only matching element totals.

For the inspected AIR profile, adapt the exemplar `llama32_1b_weights.py` only where its config/name mapping matches. Keep CPU helpers limited to operations production actually imports, such as final normalization or an explicit attention fallback; isolated primitive references stay with their harness. A production fallback must be declared in placement and performance results.

## Establish a runnable reference

Bind the checkpoint/tokenizer revision and reference implementation/precision. In the inherited profile, inspect `verify/runners/hf_runner.py` and the adapter, then use `HfRunner.prefill` with the selected canonical tokenized input and supported config/signature. Check expected output shape, token range, finite meaningful logits and complete weight/config coverage. A plausible top token is a smoke signal, not model correctness.

For a quantized or altered-arithmetic mapping, keep both roles explicit: a same-arithmetic reference can test transport/state preservation, while a HF BF16 model reference supplies the full-model token comparison. Select the applicable registry reference/tolerances and the model token contract before evaluating candidates; an unsupported contract needs explicit task definition. Weight loading and a runnable reference do not prove the NPU result.

Return resolved configuration and tensor mapping, model/tokenizer/reference identities, kept helpers/fallbacks, reference observations and selected kernel/full-model verification criteria. Missing access or unsupported architecture is a specific dependency/coverage issue. Wrong shape/name calls for remapping; unexpected logits call for checking tokenizer/template, precision and input/data identity before replacing a download. Proceed to [components](../verify-kernel/SKILL.md) when its reference prerequisites are established.
