# How the six affine-Q4 examples work

These examples extend AIR's existing LLM building blocks to six architectures.
They use the shared fused-decode engine and BFP16 prefill GEMMs. Each model keeps
its normalization, position encoding, residual structure and head semantics.
All six prefill on the shared [fused prefill](fused_prefill/affine.py).
The [command guide](AFFINE_Q4_DEPLOYMENTS.md) covers checkpoint preparation,
building, verification and profiling.

## Read the code in this order

1. The model's `model.json` pins the original HF checkpoint, hashes and supported
   input lengths. `common.py` defines its logical and padded dimensions.
2. `weights.py` maps the logical affine-Q4 tensors to the device schedule. It
   calls [q4_cascade.py](q4_cascade.py) for the common 4×4 dual-channel layout.
   Q4 codes, BF16 scales and minima survive packing bit for bit; padding is zero.
3. `build.py` selects the model's decode flags.
   [decode_build.py](decode_build.py) uses the upstream `decode_kernels.mk` /
   `kernel_rules.mk` flow and emits the two instruction-calibration templates.
   `model.json` declares `"prefill": "fused"`, so the model has no prefill
   kernels of its own: `compile` builds the shared fused prefill once
   (`deployment_runtime.build_fused_prefill`). OpenELM's and Phi-2's decodes
   end at the last block; `head.py` builds their LM head ELF (`build_head`).
4. `session.py` owns the loaded weights and request state. `prefill()` computes
   all prompt rows, `finish_prefill()` transfers the produced KV state to decode,
   and `decode_step()` appends one token at the declared position. The first
   two come from `deployment_runtime.FusedPrefillSession`.
5. [deployment_cli.py](deployment_cli.py) supplies the common reference,
   correctness and timing loops. The numerical comparator comes from AIR's
   existing [verification package](../verify/README.md).

A new model directory contains its semantic differences, rather than copies of
packing, manifest I/O, cache loading, process logging or request measurement.

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

### Prefill

The prefill runs on the device of [shared/fused_prefill](fused_prefill/), as
the repository's other fused-prefill models do: every projection GEMM, the
softmax attention and the prefill's LM head are ops of one device on one hardware
context, and the host runs the input in 128-row chunks, so any input up to the
built length takes one build. `fused_prefill/affine.py` loads the
canonical affine-Q4 tensors into the device's int4 GEMM unchanged: the codes and
the BF16 scale and minimum of each 32-column group are the operands the cores
dequantize (`w = q * scale + min` in FP32), then the GEMM computes in BFP16 with
FP32 accumulation and rounds its output to BF16. Attention is the fused
prefill's `attn_bfp16` op. The host does the embedding lookup, RMSNorm, RoPE,
residual adds, SwiGLU, operand layouts and dispatch. Granite's embedding,
attention, residual and logit multipliers come from its `config.json`:
embedding rows and scaled sublayer outputs are rounded to BF16 as in the
checkpoint's BF16 model, the attention multiplier scales q, and the logits are
divided on the host. After the first token is chosen, the prefill's roped K and
V rows are copied into the fused decode's KV buffer.

SmolLM3's layers whose `no_rope_layers` entry in its `config.json` is 0 take no
RoPE (`DensePrefill.rope_layers`). OLMo2 normalizes each sublayer's output before
the residual add and nothing before the sublayers (`Desc.post_norm`), so its
GEMMs read the residual stream directly, and its q and k RMSNorms span all
heads' columns together, before RoPE (`Desc.qk_norm_rows`).

OpenELM's layers differ in head counts and FFN width. Its device has the
widest layer's shape (32 q heads, 8 KV heads, FFN 8192), and
`affine.OpenELMPrefill` reads each layer's widths from the checkpoint's
`config.json` and zero-pads its weights up to that shape, as its decode does;
the padding contributes nothing to any output.

Phi-2 has its own layer, [fused_prefill/phi2.py](fused_prefill/phi2.py): one
biased LayerNorm feeds both the qkv and the fc1 GEMM, and the dense and fc2
outputs add to the residual together. The host adds the projection biases,
applies RoPE to the first 32 of each head's 80 dims, runs GELU and the biased
final LayerNorm, and adds the LM head's bias. The device's attention runs at
head dim 128 (`Desc.attn_dh`), each head's 80 values at column 0 or 16 of its
128 as in the decode's KV; that placement keeps each 32-column group of the
dense projection's padded input within one group of the original, so its
canonical Q4 loads unchanged.

The fused prefill is built for each model's longest native input: 4096 tokens,
2048 for OLMo2, OpenELM and Phi-2.

Transformer block GEMMs and attention, and every vocabulary row of the output
head, run on the NPU; each session lists its host work in `session.json`.

## Shared boundaries

| Shared source | Responsibility |
|---|---|
| `affine_q4.py`, `prepare_affine_q4.py` | Canonical group-32 Q4, pinned checkpoint conversion and validated tensor reads |
| `q4_cascade.py` | Device block/cascade packing, exact inverse check and upstream-packer comparison |
| `deployment.py` | Model/configuration identity, build paths and manifest I/O |
| `decode_build.py` | Logged subprocesses, common kernel Makefile flow and decode templates |
| `deployment_runtime.py` | Compiled-cache loading, optional resident prefill handoff, BF16 array loading, and the fused prefill's build and session part |
| `fused_prefill/affine.py`, `fused_prefill/phi2.py` | Canonical affine-Q4 weights, config-driven multipliers and OpenELM's per-layer widths on the shared fused prefill; Phi-2's layer |
| `affine_verify.py` | The `verify/` adapter (gate, accuracy, tasks) over a model's session |
| `deployment_cli.py`, `deployment.mk` | Common user commands, reference capture, model gate and request timing |
| `shared/infra`, `shared/builders` | Existing AIR kernel cache, ELF stitching and LM-head GEMV builders (OpenELM's and Phi-2's decode heads) |
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
