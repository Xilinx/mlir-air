# Nightly CI and persistent resources

The example follows the upstream LLM LIT suites. No separate host-only PR-CI
suite is added. Compile needs Peano and the upstream pre-23 `llvm-link`; device
tests additionally need NPU2 and all three cached model repositories.

The scheduled suites are `verify,profile,sweep,prefill-sweep,full20`. Full20 runs
four fixed prompts per dataset for each engine. Full80 is optional: select
`full80` in the workflow's `suites` input when the larger study is needed.

## Prepare once on the runner

Use the normal AIR environment and install this example's `requirements.txt`.
Then run:

```sh
cd programming_examples/llms/dflash/qwen3_4b
make download
make weights
```

`make download` fetches the exact revisions below, including the BF16 reference
used for verification, and the frozen Full80 input archive. It does not need the
NPU. On the self-hosted nightly runner, the existing **Download LLM Weights**
workflow performs this seeding. Run it once from the PR branch before trying
that branch's device suites; rerun after changing a revision or clearing caches.

| Repository | Pinned revision |
|---|---|
| `FastFlowLM/Qwen3-4B-NPU2` | `07651f11cac70ff5b7a81731643dee6f6e7baee1` |
| `Qwen/Qwen3-4B` | `1cfa9a7208912126459214e8b04321603b3df60c` |
| `z-lab/Qwen3-4B-DFlash-b16` | `b74e3a329c4d963783143b1e970d95b002be72bd` |

All three are public; these tests do not require `HF_TOKEN`. The shared seeding
workflow may still require that secret for other models on its list.

| Cache | Default location | Reused across nightly checkouts |
|---|---|---|
| Original model files | Hugging Face's `HF_HUB_CACHE`, normally `~/.cache/huggingface/hub` | Yes, with exact revisions |
| Packed decoder weights | `~/.cache/mlir-air/dflash/qwen3_4b` | Yes; override with `DFLASH_CACHE_DIR` |
| Full80 inputs | `~/.cache/mlir-air/dflash/benchmark-16x5-v1` | Yes; override Make's `BENCHMARK_BUNDLE` |
| Compiled device programs | LIT execution directory's `dflash-build` | Shared by this run's suites, rebuilt for a new checkout/toolchain |

Use a new `DFLASH_CACHE_DIR` when changing checkpoint pins or weight-packing
code. The fixed-revision nightly reuses the existing packed files.

The original snapshots occupy about 11.6 GiB in total, including the 7.5 GiB
BF16 reference. The dataset download is 1.12 GB and its extracted bundle about
1.28 GiB. Packed weights and compiler intermediates need additional disk space.
The dataset archive and every saved prefill state are checksum-verified; no
dataset service or dataset Python package is needed at runtime. The readable
messages, formatted prompts, token IDs and identities are in `benchmark/cases.json`.

The Makefile defaults `OMP_NUM_THREADS` and `OPENBLAS_NUM_THREADS` to eight for
CPU helpers and the HF reference. Explicit environment settings override these
defaults. Local validation uses eight; record different settings when comparing
performance across machines.

After seeding, use `HF_HUB_OFFLINE=1` and `OFFLINE=1`. Missing files cause a clear
preparation failure; a partially present HF repository is not treated as a
complete checkpoint. Model-data caches persist; compiled artifacts are checked
against source/tool identities. Use a fresh `BUILD_DIR` after changing them.

## Suites and measurement

From the configured AIR build directory:

```sh
ninja check-programming-examples-llms-compile
ninja check-programming-examples-llms-verify
ninja check-programming-examples-llms-profile
ninja check-programming-examples-llms-full20

# Optional larger study; not part of the scheduled nightly run.
ninja check-programming-examples-llms-full80
```

The standard suite filter is supplied by each Ninja target; for a DFlash-only
local run, invoke the configured `lit` on `programming_examples` with
`--filter 'llms/dflash/qwen3_4b/run_npu2_verify'` (or `compile`, `profile`, `full20`).
Full80 additionally requires `-D dflash_full80=1` when invoking LIT directly;
its dedicated Ninja target supplies that parameter. An unfiltered test run
skips Full80 unless explicitly enabled.
Do not run the device suites concurrently. Like the existing LLM LITs, these
invoke Make directly with LIT's configured Python/XRT environment. The Make
commands take the session-level `/tmp/mlir-air-npu.lock`; shared `KernelCache`
separately takes `/tmp/npu.lock` for individual dispatches. Wrapping these LLM
commands in `flock /tmp/npu.lock` would self-deadlock on that inner lock.

| Suite | Workload | Output |
|---|---|---|
| Compile | Weight-free target, drafter, NPU prepass and C2048 prefill build | Identity-checked manifests and device artifacts |
| Verify | Decoder repeats/visibility at prefixes 0, 7, 16, 64, 1024, 4088; fixed 32-token loop/oracle/committed-KV checks; first two upstream `instruct.txt` prompts, 32 tokens each, EOS stopping disabled, separate HF reference process | Detailed verification JSON and LIT pass/fail |
| Profile | `Introduce me what is GPU`, cap256, EOS enabled, warmup excluded, actual NPU prefill + complete DFlash decode | Standard `qwen3_4b_dflash_b8.perf.json` |
| Full20 (nightly) | B1 and DFlash in separate processes; first 4 prompts per dataset from the frozen seed-0 manifest; cap2048, EOS enabled, greedy/thinking off, one pass, saved CPU-prefill inputs, MT-Bench first turn | Paired overall/per-dataset throughput and speedup; acceptance, emitted/round, per-request traces |
| Full80 (optional) | Same protocol, using all 16 prompts per dataset | Separate results with the same metrics |

Both studies use the current upstream fused single-token GEMV decoder for B1, with the
same requantized target weights and saved inputs as DFlash. It is separate from
the B8 row-zero sequential correctness oracle. The baseline is measured on the
current runner; the historical 19.310 tokens/s is not inserted into new results.
All compared decoders use KV capacity 4096, accommodating the frozen prompts
plus up to 2048 output tokens. The B1 builder retains the existing Qwen3-4B
per-kernel optimization settings: non-attention objects at `-O2`, inline
attention at `-O1`.
The primary decode timer includes initial and later NPU prepass. Prefill, setup,
warmup and the prefilled first token are excluded. The paper-era timing variant
is reported separately, not used as the default speedup denominator.

Both studies preserve request logs and refuse to mix a second run into an
existing output directory. For a local repeat, choose a new output name, e.g.
`make full20 FULL20_OUTPUT=build/repeat2.full20.json` or
`make full80 FULL80_OUTPUT=build/repeat2.full80.json`. Nightly checkouts start
with a clean build directory. Keep manually compiled artifacts separate from
LIT's directory: install/bin and build/bin compiler binaries can have different
hashes even when installation only changed their library search paths.

The nightly workflow keeps Full20, Full80 and short-profile history separate.
It adds an overall paired table for each executed cohort to the job summary
and uploads `full20.json` / `full80.json`, per-dataset summaries, request traces
(including partial runs), verification reports and build manifests. A measured
speedup is reported without a machine-specific minimum performance pass
threshold. Narrow dispatches such as `verify,profile,full20` or `full80`
produce downloadable partial-run artifacts and do not publish into main's
performance history.

## Runtime budget

The October 17 standalone Full20 LIT passed in **830.53 seconds (13.8 minutes)**
with cached device builds. Its B1 process took 9.8 minutes and DFlash 3.9 minutes;
the LIT also checked resources and existing artifacts. Allow about 15 minutes
for a warm Full20 run, with a 30-minute LIT timeout. Full20's measured pooled
throughputs are 15.738 / 42.234 tokens/s (B1 / DFlash), or 2.684×. Its smaller
cohort is a separate performance series from Full80.

The October 16 HX370 run measured 37.6 minutes for the B1 process and 14.7
minutes for DFlash: **52.4 minutes combined**. Its actual Full80 LIT completed
in **53.6 minutes**, including B1 compilation and cached resource/build checks.
Target/drafter/prepass/prefill compilation was already cached for that run.
Fresh compilation of those components took 6.5 minutes; fresh packing took
52 seconds. Verify/profile LITs took 4.5/1.8 minutes, including initialization.

Budget 50–60 minutes for the optional warm Full80 study, at least an additional
seven minutes for a fully cold device build, and 90 minutes for the Full80 LIT
timeout. Downloads are seeded separately. The full nightly job limit remains
540 minutes to accommodate an explicitly requested Full80 run as well.
These measurements are local; the nightly/Krackan runner
needs its own execution. Local jobs used a 20 GiB memory cap: the largest
observed verification/profile job peak was about 18.2 GiB, and Full80 about
7.9 GiB. These are observed job peaks, not a total-system memory requirement.
