# How the six affine-Q4 examples work

These examples extend AIR's existing LLM building blocks to six architectures.
They use the shared fused-decode engine and BFP16 prefill GEMMs. Each model keeps
its normalization, position encoding, residual structure and head semantics.
The [command guide](AFFINE_Q4_DEPLOYMENTS.md) covers checkpoint preparation,
building, verification and profiling.

## Read the code in this order

1. The model's `model.json` pins the original HF checkpoint, hashes and supported
   input lengths. `common.py` defines its logical and padded dimensions.
2. `weights.py` maps the logical affine-Q4 tensors to the device schedule. It
   calls [q4_cascade.py](q4_cascade.py) for the common 4×4 dual-channel layout.
   Q4 codes, BF16 scales and minima survive packing bit for bit; padding is zero.
3. `build.py` selects the model's decode flags and prefill kernels.
   [decode_build.py](decode_build.py) uses the upstream `decode_kernels.mk` /
   `kernel_rules.mk` flow and emits the two instruction-calibration templates.
4. `session.py` owns the loaded weights and request state. `prefill()` computes
   all prompt rows, `finish_prefill()` transfers the produced KV state to decode,
   and `decode_step()` appends one token at the declared position.
5. [deployment_cli.py](deployment_cli.py) supplies the common reference,
   correctness and timing loops. The numerical comparator comes from AIR's
   existing [verification package](../verify/README.md).

A new model directory contains its semantic differences, rather than copies of
packing, manifest I/O, cache loading, process logging or request measurement.
Granite, SmolLM3 and Mistral reuse Llama's composed prefill blocks where their
semantics match. OLMo2, OpenELM and Phi-2 have local prefill
composition for their different normalization or residual layouts.

## Architecture and mapping

| Model | Layers / hidden width | What differs | Decode submissions per output |
|---|---|---|---|
| Granite-3.3-2B | 40 / 2048 | GQA; embedding, attention, residual and logit scales | One: all blocks and Q4 head waves |
| SmolLM3-3B | 36 / 2048 | GQA; a per-layer schedule enables or bypasses RoPE | One: all blocks and Q4 head waves |
| OLMo2-1B | 16 / 2048 | MHA; full-width Q/K normalization; post-attention and post-MLP RMSNorm | One: all blocks and Q4 head waves |
| OpenELM-1.1B | 28 / 2048 | Layer-dependent attention/MLP widths; per-head Q/K normalization; tied BF16 head | One body plus one head |
| Phi-2 | 32 / 2560 | LayerNorm with bias; parallel residual branches; `gelu_new`; biased projections; RoPE on 32 of 80 head dimensions | One body plus one final-LayerNorm/head |
| Mistral-7B-Instruct-v0.3 | 32 / 4096 | GQA with 32 query / 8 KV heads, head dimension 128, independent 32768-row head | One: all blocks and two head waves |

The fused body loops over all layers on device. “One submission” describes the
host launch count; it does not imply that every layer occupies separate cores
simultaneously. OpenELM's variable widths and Phi-2's head geometry use padding
to fit the shared spatial schedule. Model-local packers document the mapping.

Prefill processes the full input through every block on the NPU. It uses a
host loop over composed kernel submissions. GEMMs dequantize the fixed logical
Q4 weights into BF16 operands, compute through the BFP16 path with FP32
accumulation, and round the output to BF16. This follows
`shared/infra/external_kernels.py::compile_gemm_mm` and its default `bfp16=True`.
The cache metadata requires `gemm_bfp16=true` before runtime.

Attention has its own precision settings. In particular, the large P4096
SmolLM3/Mistral attention variants retain their native-BF16, single-accumulator
path. Selecting BFP16 GEMM does not replace that choice. OpenELM's tied BF16
head and Phi-2's bias arithmetic also retain their original semantics.

The host performs embedding lookup, rotary table preparation, dispatch, byte
layout/transfers, full-logit readback/checks and greedy selection. Models that
use a host final single-row prefill norm report it in `session.json`. Transformer
block arithmetic and every vocabulary row of the output head run on the NPU.

## Shared boundaries

| Shared source | Responsibility |
|---|---|
| `affine_q4.py`, `prepare_affine_q4.py` | Canonical group-32 Q4, pinned checkpoint conversion and validated tensor reads |
| `q4_cascade.py` | Device block/cascade packing, exact inverse check and upstream-packer comparison |
| `deployment.py` | Model/configuration identity, build paths and manifest I/O |
| `decode_build.py` | Logged subprocesses, common kernel Makefile flow and decode templates |
| `deployment_runtime.py` | Compiled-cache loading, optional resident prefill handoff and BF16 array loading |
| `deployment_cli.py`, `deployment.mk` | Common user commands, reference capture, model gate and request timing |
| `shared/infra`, `shared/builders` | Existing AIR cache, GEMM, attention, normalization and composed prefill infrastructure |
| `../../fused_decode` | Existing AIR spatial decode engine, with model-gated semantic extensions |

Model-gated changes must preserve the IR of existing engine entries. Default
prefill builder arguments retain their previous meaning. Build outputs and
large weights belong in `BUILD_DIR`, not in source control.

## Interpreting verification and profiling

The fixed model gate compares 32 free-running greedy outputs against the
original CPU BF16 checkpoint. It retains all vocabulary logits, rejects missing
or nonfinite values, and applies mutual top-5 inclusion at the first divergent
choice. This accommodates quantization while exposing the exact divergence;
it is not full-logit equality. Native-context exclusions are not extrapolated.

`--check-reset` optionally runs A / changed-request B / A in one loaded session
and requires the two A captures to be bitwise identical. The changed request
modifies the first input token and generates two outputs, exercising a different
prefill and one decode append. This checks request isolation, not exhaustive
internal-state equivalence to HF.

A profile is batch one, 128 outputs, with fresh requests and EOS ignored. Complete
requests warm up for at least five seconds; at least three measured requests
must total ten seconds. TTFT includes first-logit readback, checks and selection.
Physical KV handoff follows that boundary and is charged to decode. Report
median TTFT, input tokens divided by median TTFT, pooled decode tokens/s and
mean complete-request time. Setup/build time is separate. `samples.json` retains
completed requests even if a later request fails.
