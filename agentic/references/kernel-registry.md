# Registry reuse, miss and validated handoff

Trace the production runtime and build path before deciding how the registry applies. It supplies primitive implementation/configuration assets, contracts and measured scope; its contents remain external under the supplied AIR root. A model name or the word "fused" does not establish registry integration. See [project integration](project-integration.md) for the inspected source binding.

## Distinguish connected selection from reference material

The following trace is from AIR `6746658f74200feb957f99ad4163297f3185739c`. Paths are relative to `AIR_ROOT/programming_examples`; line numbers describe the inspected revision, so retrace callers when source changes.

| Inspected path | What actually selects or implements the work | Registry scope |
|---|---|---|
| Shared GEMM builder | `llms/shared/builders/gemm_builder.py:62` calls `kernel_registry/registry_lookup.py:38` and translates its method/tiles into a build recipe | Exact-shape BF16-input GEMM selection is connected. The lookup itself does not tune or establish a gate PASS. |
| Llama-3.2-1B Q4NX prefill | `llms/llama32_1b_q4nx/llama32_1b_q4nx_prefill.py:80` imports the BF16 prefill compiler and calls it at line 98; `llms/shared/builders/o_ffn_multi.py:245` selects GEMM recipes, while `llms/llama32_1b/llama32_1b_prefill.py:122` matches their runtime scratch plan | Registry-backed GEMMs are used within composed/stitched prefill. This connection does not imply that the same model's decode uses those recipes. |
| Llama-3.2-1B Q4NX decode | `llms/llama32_1b_q4nx/Makefile:111` delegates to `fused_decode`; `llms/llama32_1b_q4nx/llama32_1b_q4nx_inference.py:270` defines `FusedDecoder` and loads `fused_decode.py` at line 325 | The traced custom decode selects its existing mapping and linked kernels rather than the shared GEMM lookup. Registry rows are optional applicable primitive/reference assets, not configuration for this mapping. |
| Custom decode implementation and packing | `fused_decode/fused_decode.py:1570` explicitly links projection symbols to `proj_qmm.o`; `fused_decode/Makefile:231` builds its C++ kernels and templates; `fused_decode/kernels/proj_qmm.cc:84` implements the quantized projection; `llms/llama32_1b_q4nx/q4nx_requant.py:114` uses `fused_decode/proj_qmm_pack.py:60` for the cascade stream | Builder, arithmetic, block/stream order, buffers and runtime binding must agree. A standalone GEMM tile or receipt does not establish that agreement. |

For a new Q4NX model or structural change in this family, start with the closest existing fused-decode implementation and its spatial mapping, then use [megakernel change analysis](../skills/megakernel-change-analysis/SKILL.md). Reuse unchanged internal kernels and evidence whose dependencies still match. An absent registry row is not a reason to decompose the model, rebuild those kernels or block this adaptation route. Trace prefill and other dispatches separately; they may require registry-backed primitive work. This is a statement about the inspected callers, not a claim that megakernels cannot use registries.

A registry reference/contract can guide a primitive check when arithmetic, outputs and scope match. Reusing a standalone implementation inside custom decode requires explicit integration of ABI, packing, ownership/lifetimes, resources, transfer/synchronization and surrounding execution. Its isolated tile, timing or numerical PASS does not transfer automatically to the embedded operator or full model. Apply the affected kernel contract and the full-model contract separately; report missing isolated coverage without relabeling a layer diagnostic as kernel PASS. No additional global verification level is introduced.

## Match an entry for a registry-backed primitive

For the inspected GEMM path, lookup matches exact `(M, K, N)`, output dtype and precision, returning measured method/tile configuration used by the shared builder. Read the current lookup and entry before relying on them. Check input/output contract, numerical datapath, layout, target/device, toolchain/API compatibility, measurement boundary and validation coverage as well: target versions and gates are not all encoded in the lookup key.

A hit is a reusable starting implementation/configuration. Read the actual reference/tolerance contract and any `NUMFAIL` annotation: the lookup does not certify PASS. [Verification contracts](validation-contracts.md) records representative thresholds and known harness discrepancies. It is not proof of a new shape, in-context correctness, another target's performance or full-model optimality. Reuse a validation or timing receipt only with a recorded dependency/coverage match. An output conversion belongs in the measured boundary when the caller requires that output representation.

## Handle a miss without inventing automation

When the selected primitive caller requires this lookup, the existing exact-shape lookup reports a miss and requests a sweep/record update; it does not perform tuning, nearest-shape search, validation or automatic admission. Identify the missing key and select an existing supported implementation as a development starting point. A nearby shape may suggest tunables but is not a validated result. This miss procedure is scoped to that primitive route; it does not impose registry coverage on the custom fused-decode path above.

Use [kernel development](../skills/verify-kernel/SKILL.md) and [Optimization](../skills/npu-optimization/SKILL.md) to choose bounded candidates from actual tunables and resource conditions. Apply [staged checks](staged-checks.md), satisfy the applicable full-output kernel gate, then compare the required kernel boundary. Do not require a complete model before useful kernel tuning, or exhaust a search before a valid candidate can be integrated.

## Retain only supported reusable claims

Propose a row in the registry's real schema after validation and measurement. Attach implementation/source and local changes, build configuration/artifact, shape/layout/output/precision, target/toolchain, local gate/reference/coverage, timed boundary/raw samples, constraints and unresolved integration obligations. Keep admission review and shared-registry mutation within the task's authorization; during frozen evaluations keep proposed rows local.

A locally validated handoff carries the ABI, packing/weight assumptions and state/lifetime requirements along with its evidence. [Model verification](../skills/verify-model/SKILL.md) remains separate; [integration diagnosis](integration-diagnostics.md) is optional. Track any later integration result without rewriting local evidence as complete-model acceptance. New shapes, arithmetic, target/toolchain or changed dependencies require reassessing the row's applicability.

## Opt-in evidence-bound selection

`programming_examples/kernel_registry/evidence.py` supplies `validate_record` and `select_verified` for local evidence-bound primitive or megakernel records. Exact keys include workload, precision, fixed tolerances, layout, target/toolchain and timing boundary. Selection checks source/artifact/reference/report hashes and chooses the lowest mean latency among matching measured PASS records. No match remains `NO_VERIFIED_MATCH`; legacy lookup and its historical rows remain unchanged.

See [record format and local proposals](maintenance.md). Megakernel records include their fused graph and ABI; a full-model token gate alone cannot populate a kernel elementwise PASS. Record failures and UNKNOWN explicitly. Admission uses normal repository review, and shared catalogs remain frozen during controlled comparisons. The agent performs any needed tuning; lookup does not launch a sweep automatically.
