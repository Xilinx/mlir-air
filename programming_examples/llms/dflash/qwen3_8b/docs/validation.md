# CI and validation

This example follows the existing LLM LIT suite discovery. `run_npu2_compile`,
`run_npu2_verify`, `run_npu2_profile` and `run_npu2_full20` join the corresponding
upstream suites. Full80 additionally requires `-D dflash_full80=1`, supplied by
`check-programming-examples-llms-full80`; it is not scheduled by default.

| Entry | Purpose | Scheduling |
|---|---|---|
| `run_npu2_compile.lit` | Weight-free target, drafter, prepass and prefill compilation | LLM compile suite; execution tests also invoke compilation |
| `run_npu2_verify.lit` | Decoder state, DFlash loop and HF BF16 top-5 checks | Nightly verify suite |
| `run_npu2_profile.lit` | Fixed-prompt TTFT and decode throughput | Nightly profile suite |
| `run_npu2_full20.lit` | Paired B1/DFlash, four prompts per dataset | Default nightly Full20 suite |
| `run_npu2_full80.lit` | Paired B1/DFlash, sixteen prompts per dataset | Optional Full80 suite |

The shared acceptance and streaming tests keep the existing
`qwen3_4b/run_host_checks.lit` entry in the PR host suite. Both models use the
tested common implementation; 8B does not duplicate that host test entry.
The profile declares `PERF_MODEL: qwen3_8b_dflash_b8`, which lets the existing
nested-example collector associate verification, profile and paired results.
Nightly artifact patterns already include both DFlash model directories.

## Import names and offline references

The loop checker is `verify_loop.py`; a `verify.py` would collide with
`llms/verify`.
Shared upstream imports use the explicitly registered `air_examples` namespace;
the example does not put `programming_examples/` or `llms/` on `sys.path`.
`python/test/api/examples_sys_path.py` checks these rules in Python CI.

The HF reference resolves the snapshot through
`resources.prepare(reference=True, offline=True)`. This uses the same pinned
file list as cache seeding, without requiring repository README/LICENSE files
that nightly does not download. The reference weights, tokenizer, arithmetic
and comparison criteria remain unchanged.

## Resources and execution

The existing Download LLM Weights workflow seeds all three pinned 8B snapshots,
including the BF16 reference. Run it once before offline nightly execution. Weight caches persist under `~/.cache/mlir-air/dflash/qwen3_8b`.
Prompt messages are included in the source tree; no dataset service is needed
at runtime. The first Full20 prepares 20 NPU-prefill states, then caches them.
Optional Full80 extends the same cohort to 80 states. Prefill source/build and
state identities prevent reusing another model's KV or stale tensors. Each
source/build combination gets a separate cache generation; identical artifacts
are reusable despite changes to incidental build paths or timestamps.

Full20 permits up to one hour per LIT test, which covers cold preparation of
the prompt states. Full80 permits up to three hours and remains optional.

Device commands serialize through `/tmp/mlir-air-npu.lock`. Do not wrap them in
`/tmp/npu.lock`, which the shared prefill cache takes internally. CPU reference
and NPU model captures run in separate processes to bound memory.

| Check | Coverage |
|---|---|
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

NPU prefill also uses the existing BFP attention primitive. Extending a prompt
inside an incomplete eight-key exponent group can change rounding in that
group. The loop check records those differences and requires exact equality for
completed groups and aligned-tile prefixes, plus repeated identical prompts.
Both benchmark engines receive the same saved prefill states.

Build and run commands are in [the README](../README.md). Each verification
invocation writes fresh JSON under `BUILD_DIR/verification/`; each paired study
retains separate engine logs and detailed request records. Use a fresh output
name when repeating a study. Fresh qualification results are recorded alongside
source/artifact identities before publishing performance claims.

## Additional arithmetic probes

From the model example directory, these commands reproduce the standalone
attention and prepass probes in addition to `make verify`:

```sh
python tests/check_attention.py --build --source build/target --output build/attention-check
flock /tmp/mlir-air-npu.lock python tests/check_attention.py --run --output build/attention-check --result build/attention-check.json
python tests/check_attention.py --build --no-causal --source build/draft --output build/draft-attention-check
flock /tmp/mlir-air-npu.lock python tests/check_attention.py --run --no-causal --output build/draft-attention-check --result build/draft-attention-check.json
make benchmark-data PER_DATASET=4 BENCHMARK_BUNDLE=build/prompt-states
flock /tmp/mlir-air-npu.lock python tests/check_prepass.py --artifacts build/prepass --fixture build/prompt-states/inputs/gsm8k_264.npz --output build/prepass-check.json
```

The prepass fixture is produced locally by NPU prefill from the pinned checkpoint
and frozen prompt; it does not require a private development artifact.
