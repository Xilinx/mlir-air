# Affine-Q4 verification and performance

All **19 native configurations** passed the fixed original-HF BF16 model gate on the migrated implementation. Every model also passed a P512 A/B/A reset-and-repeatability check. Five longer configurations outside native context are excluded.

Measured on an AMD Ryzen AI 9 HX 370, Ubuntu 24.04.3, XRT 2.21.0 / firmware 1.1.2.64. Source commit `9130011`, upstream base `6356cb2`; AIR `0.0.1.2026102604+2f080d3`, MLIR-AIE `1.4.5.dev64+g843e9c3`, Peano `22.0.0.2026101901+4f5df13d`, LLVM 18 linker. After measurement, upstream `5559460` was merged as `dddd696`; all six model sources, the fused-decode engine and the shared request/runtime helpers remain byte-identical. That upstream update removes the AWQ example and includes compiler fix `4f6ec18`. The measurements use the latest published wheel named above; they do not claim execution with a locally rebuilt compiler containing that newer fix.

| Model | P / C | Median TTFT (s) | Prefill (input tok/s) | Decode (tok/s) | Mean request (s) | First divergence (zero-based) |
|---|---|---:|---:|---:|---:|---|
| Granite-3.3-2B | 512 / 1024 | 0.6349 | 806.40 | 29.96 | 4.8739 | 3; mutual ranks 4 / 2 |
| Granite-3.3-2B | 1024 / 2048 | 1.1934 | 858.02 | 27.69 | 5.7773 | None; 32/32 agree |
| Granite-3.3-2B | 2048 / 4096 | 2.2627 | 905.12 | 24.14 | 7.5383 | None; 32/32 agree |
| Granite-3.3-2B | 4096 / 8192 | 4.6802 | 875.17 | 19.16 | 11.3152 | None; 32/32 agree |
| SmolLM3-3B | 512 / 1024 | 0.8042 | 636.65 | 24.73 | 5.9471 | None; 32/32 agree |
| SmolLM3-3B | 1024 / 2048 | 1.4654 | 698.78 | 23.12 | 6.9596 | None; 32/32 agree |
| SmolLM3-3B | 2048 / 4096 | 2.8655 | 714.71 | 20.51 | 9.0622 | None; 32/32 agree |
| SmolLM3-3B | 4096 / 8192 | 8.7329 | 469.03 | 12.61 | 18.9271 | None; 32/32 agree |
| OLMo2-1B | 512 / 1024 | 0.5171 | 990.18 | 39.68 | 3.7172 | None; 32/32 agree |
| OLMo2-1B | 1024 / 2048 | 0.8842 | 1158.11 | 30.43 | 5.0577 | 31; mutual ranks 3 / 3 |
| OLMo2-1B | 2048 / 4096 | 1.7482 | 1171.52 | 20.78 | 7.8640 | None; 32/32 agree |
| OpenELM-1.1B | 512 / 1024 | 0.8094 | 632.56 | 33.63 | 4.5876 | 19; mutual ranks 4 / 3 |
| OpenELM-1.1B | 1024 / 2048 | 1.4023 | 730.25 | 31.62 | 5.4172 | 27; mutual ranks 2 / 2 |
| Phi-2 | 512 / 1024 | 1.9878 | 257.57 | 9.84 | 14.8922 | None; 32/32 agree |
| Phi-2 | 1024 / 2048 | 3.8244 | 267.75 | 7.57 | 20.6317 | None; 32/32 agree |
| Mistral-7B-Instruct-v0.3 | 512 / 1024 | 1.3572 | 377.26 | 11.79 | 12.1332 | 4; mutual ranks 2 / 2 |
| Mistral-7B-Instruct-v0.3 | 1024 / 2048 | 2.6117 | 392.08 | 11.31 | 13.8397 | None; 32/32 agree |
| Mistral-7B-Instruct-v0.3 | 2048 / 4096 | 5.2371 | 391.06 | 10.46 | 17.3828 | None; 32/32 agree |
| Mistral-7B-Instruct-v0.3 | 4096 / 8192 | 16.0451 | 255.28 | 9.11 | 29.9947 | None; 32/32 agree |

Each request has 128 outputs, batch one, EOS ignored. Complete fresh requests warm up for ≥5 seconds, followed by ≥3 measured requests totaling ≥10 seconds. Decode throughput is pooled over the 127 decode outputs per request. TTFT ends at first-token selection; KV handoff is included in decode/request time. Build and model loading are outside request timing. These are new migration measurements, not copied historical results.

SmolLM3 P4096 showed variability in the initial three-request batch. One additional batch was collected with memory telemetry. The reported row pools all six measured requests from both batches; neither batch is discarded. The machine-readable record retains each batch separately.

The [JSON record](affine_q4_results.json) contains full-precision values, individual request durations, gate divergence/ranks and capture hashes. References are original-HF CPU BF16 captures, with checkpoint shard, exact input-ID and logit hashes rechecked before reuse. The gate requires finite logits for all vocabulary entries across 32 outputs; mutual top-5 inclusion is checked at first divergence.

Reproduce with the [command guide](AFFINE_Q4_DEPLOYMENTS.md). See the [implementation guide](AFFINE_Q4_IMPLEMENTATION.md) for precision, model differences, host/device placement and shared-code boundaries.

## Existing-example regression checks

After merging upstream `5559460`, the final CLI freshly compiled Phi-2 P512
prefill/decode/head. Existing Llama-3.2-1B Q4NX default composed prefill passed
its first-token gate and exact repeated-context logits after a fresh build.
Existing Qwen3-4B DFlash target B8/C4096, built with current upstream source
compiler `dddd696` (including the required `4f6ec18` fix), passed its state, causality and
repeatability checks at prefixes 0, 64 and 1024 after a fresh build.
The current-source compiler passes the repeat-count unit/IR tests, and all 38
six-model decode template lowerings are byte-identical to the measured wheel.
Seven affine-Q4 host tests and the existing import-path checks also passed.
These bounded checks supplement the unchanged-IR comparisons for 12 existing
decode entries/modes and two shared prefill defaults.

## Current-source CI integration

The six examples now provide 19 native-configuration verify entries and 19
profile entries in the existing serial nightly suites, plus six weight-free
P512 compile entries and a per-PR host entry. Each profile performs its own
original-BF16 gate before timing; P512 also checks A/B/A reset. Pinned offline
checkpoints and separate Transformers 5.3.0 / 4.39.3 reference environments are
provisioned by the workflows. Every P/C has a separate dashboard ID and label.
Complete and partial uploads retain captures, logs, status and raw timings.

The actual Phi-2 and OpenELM P512 lit profile entries passed locally with the
current-source compiler based on upstream `5559460`: fresh weight preparation,
compilation, CPU references, numerical/reset gates and profiling. The shared
host lit entry passed all 13 tests. These two integrations supplement the
19-point campaign above; they do not replace its measurements.

| Current-source CI configuration | Median TTFT (s) | Decode (tok/s) | Model gate / reset |
|---|---:|---:|---|
| OpenELM-1.1B, P512/C1024 | 0.8096 | 33.67 | PASS / PASS |
| Phi-2, P512/C1024 | 1.9802 | 8.62 | PASS / PASS |

The first local invocations exposed a removed lit `%T` substitution, missing
OpenELM Python files in the preseed list, rejection of HF snapshot blob symlinks,
and a Hub client requiring explicit offline resolution. These loading/CI issues
were corrected before device execution. The checkpoint conversion test covers
the blob layout and rejects incorrect shard hashes. Kernel arithmetic is unchanged.

Hosted GitHub CI remains pending approval of fork workflows by an upstream
administrator. Local lit results and hosted CI status are reported separately.
