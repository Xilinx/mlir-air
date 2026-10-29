# AIR integration and task preparation

Use the installed checkout's source and documented toolchain flow. Current shared examples are under `programming_examples/llms/shared`, registry data under `programming_examples/kernel_registry`, and fused decode under `programming_examples/fused_decode`. Example modules use the `air_examples` namespace; avoid adding all of programming_examples to sys.path, which can shadow installed packages. Existing dialect builders and the newer air.api can coexist; use the interface appropriate to the seed.

## From an HF ID to a task

Run `python agentic/tools/task.py OWNER/MODEL --out /path/task.json`. Use
`--requirements requirements.json` for explicit user choices, as described in
[task input](task-input.md). Shipped model IDs use pinned model.json metadata
offline. A new ID resolves a full HF revision and config; `--config` and
`--revision` provide an offline route. If native context is absent, inspect the
original implementation and use `--native-context`. Check attention masks,
position/RoPE rules, normalization, activation, biases, tied heads, tensor shapes
and state against reference code; a family name or equal dimensions is insufficient.

The default cohort is P512/C1024, P1024/C2048, P2048/C4096 and P4096/C8192, excluding capacities beyond the task's native-context contract. Default gates capture 32 greedy outputs with EOS ignored, finite full logits and fixed mutual top-5 at the first divergence. Default timing is complete 128-output requests: at least five seconds warmup, three measured requests and ten measured seconds. A user-supplied task can select another cohort or timing protocol explicitly.

## Implement a new model

1. Inspect the pinned config, reference implementation, tensor names/shapes,
   tokenizer and state. Choose a compatible existing model or kernel seed from
   the checkout; record differences in arithmetic, layout, biases, normalization,
   position handling, attention and state. Resolve supplied quantization semantics.
2. Add a model directory under `programming_examples/llms`. For the affine-Q4
   interface, provide `model.json`, fixed `inputs.json`, model-specific packing,
   a build entry and a production adapter implementing the methods used by
   `shared/deployment_cli.py`. Read an existing model's adapter and the caller's
   actual signatures. Pin checkpoint/config/tokenizer and source shard hashes.
3. Define BF16 matrices and quantization aliases before packing. Canonical
   affine Q4 uses group32 unsigned nibbles with `w=scale*q+min`; reuse
   `shared/prepare_affine_q4.py` when its arithmetic matches. Preserve supplied
   logical Q4 bit-for-bit; device packing changes layout. Declare attention/head
   precision and other exceptions explicitly.
4. Adapt the connected shared/fused implementation and register the module in
   `shared/deployment.py`. Implement unsupported operations and task differences
   in the production path, with scoped kernel validation. Reuse shared helpers
   rather than cloning an entire model or evaluator.
5. Run the requested full-model gate on each required configuration, then the
   requested profiling protocol. The shared affine-Q4 commands cover their fixed
   32-output gate and 128-output complete-request profile. A changed task protocol
   needs a matching execution interface before it can be evaluated correctly.
6. Add the model's reproduction documentation and appropriate CI coverage, using
   the shipped examples' Make/lit and shared deployment interfaces. Preserve the
   task, source/artifact identities, captures, timings and failed attempts.

The [workflow](../WORKFLOW.md) and deployment skill guide this implementation
loop. The task tool's `ADAPTATION_REQUIRED` result identifies work to do; it is
not a reason to stop a new-model deployment at metadata preparation.

## Shared execution interfaces

| Source | Role |
|---|---|
| `llms/shared/deployment.py` | Model specifications, native configurations, input and Q4 binding, BFP16 artifact requirement |
| `llms/shared/prepare_affine_q4.py` | Checkpoint identity checks and canonical Q4 preparation |
| `llms/shared/deployment.mk` | Model commands: weights, compile, reference, verify, profile and CI |
| `llms/shared/deployment_cli.py` | Production request/capture, strict complete-capture wrapper, reference and timing |
| `llms/shared/deployment_ci.py` | Source-bound gate-before-profile run, raw evidence and performance records |
| `llms/shared/infra/external_kernels.py` | External compilation; `compile_gemm_mm` defaults to BFP16 |
| `llms/shared/builders` | Shared composition builders and registry GEMM selection |
| `fused_decode` | Fused body builder, kernels, model geometry and device weight stream |

Paths in this table are relative to programming_examples. Read the actual signatures and target Makefile. [The runnable example](../examples/q4nx-supported-model.md) shows the supported command sequence. Use a separate build directory; declare checkpoint/Q4 paths explicitly. Host tools do not establish device readiness.

## Numerical and artifact identity

For supported Q4 deployments, enable `AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16` for prefill GEMM and keep `gemm_bfp16=true` in selected prefill metadata. This concerns compute, not the logical quantization. Preserve model-specific attention and head exceptions. Bind flags, geometry, source, packed weights, compiled artifact and runtime ABI before reusing a cache. An unchanged filename is not evidence of compatibility.

Use the task's device lease. Shared affine-Q4 commands acquire `/tmp/mlir-air-npu.lock`; some infrastructure also holds internal `/tmp/npu.lock`. Inspect the actual call chain before adding flock wrappers to avoid recursive acquisition. Never reset or terminate another task to obtain the device. Record compiler/AIE/Peano/XRT/driver and device identity with results. Current fused decode's linker check is in its Makefile; preserve the resolved supported tools rather than copying historical version strings blindly.
