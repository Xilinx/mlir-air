# CI and validation

This example follows the existing LLM LIT suite discovery. `run_npu2_compile`,
`run_npu2_verify`, `run_npu2_profile` and `run_npu2_full20` join the corresponding
upstream suites. Full80 additionally requires `-D dflash_full80=1`, supplied by
`check-programming-examples-llms-full80`; it is not scheduled by default.

The existing Download LLM Weights workflow seeds all three pinned 8B snapshots,
including the BF16 reference. Run it from this branch once before offline
nightly execution. Weight caches persist under `~/.cache/mlir-air/dflash/qwen3_8b`.
Prompt messages are included in the source tree; no dataset service is needed
at runtime. The first Full20 prepares 20 NPU-prefill states, then caches them.
Optional Full80 extends the same cohort to 80 states. Prefill source/build and
state identities prevent reusing another model's KV or stale tensors. Each
source/build combination gets a separate cache generation; identical artifacts
are reusable despite changes to incidental build paths or timestamps.

Full80 permits up to three hours per LIT test, including cold preparation.
Full20 remains in the normal nightly suite; exact measured duration is recorded
with the qualification results.

Device commands serialize through `/tmp/mlir-air-npu.lock`. Do not wrap them in
`/tmp/npu.lock`, which the shared prefill cache takes internally. CPU reference
and NPU model captures run in separate processes to bound memory.

| Check | Coverage |
|---|---|
| Compiler LIT | Independent resident replay-ring allocation; ordinary packet-sharing control; complete compiler suite |
| Decoder state | Prefixes 0, 7, 16, 64, 1024, 4088; repeated complete logits/hidden/KV, finite outputs, unchanged prefix/tail, target causality, draft bidirectionality and injected KV |
| Attention primitive | Production object and shared attention graph, causal/bidirectional variants, full-output SDPA tolerance, row isolation and repeatability |
| Context prepass | BFP arithmetic reference, full output tolerances, row scales/isolation, mixed positions, repeated calls, partial chunks and multi-chunk context |
| `verify_loop.py` | Repeated actual NPU prefill; 32 fixed output tokens against a sequential oracle using matching B8 arithmetic; acceptance traces and committed KV after every round |
| Model gate | First two upstream `verify/prompts/instruct.txt` prompts, 32 tokens each without EOS stopping; separate HF BF16 reference and upstream mutual top-5 comparator |
| Profile | `Introduce me what is GPU`, cap256, actual NPU prefill, warmup excluded |
| Full20 / Full80 | Paired B1/DFlash, identical model-specific saved states, complete-loop decode timer and per-request traces |

The model comparator checks the common prefix and mutual top-5 membership at
the first divergence. It does not establish bitwise equality with BF16 or
accuracy beyond the compared prefix. The separate DFlash oracle/state checks
cover greedy acceptance and committed-cache behavior; a B1 GEMV result is not
a bitwise oracle for B8 BFP projection arithmetic.

Build and run commands are in [the README](../README.md). Each verification
invocation writes fresh JSON under `BUILD_DIR/verification/`; each paired study
retains separate engine logs and detailed request records. Use a fresh output
name when repeating a study. Fresh qualification results are recorded alongside
source/artifact identities before publishing performance claims.
