# Nightly CI and persistent resources

The example follows the upstream LLM LIT suites. No separate host-only PR-CI
suite is added. Compile needs Peano and the upstream pre-23 `llvm-link`; device
tests additionally need NPU2 and all three cached model repositories.

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
ninja check-programming-examples-llms-full80
```

The standard suite filter is supplied by each Ninja target; for a DFlash-only
local run, invoke the configured `lit` on `programming_examples` with
`--filter 'llms/dflash/qwen3_4b/run_npu2_verify'` (or `compile`, `profile`, `full80`).
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
| Full80 | B1 and DFlash in separate processes; 16 prompts from each of five datasets, cap2048, EOS enabled, greedy/thinking off, one pass, saved CPU-prefill inputs, MT-Bench first turn | Paired overall/per-dataset throughput and speedup; acceptance, emitted/round, per-request traces |

Full80's B1 uses the current upstream fused single-token GEMV decoder, with the
same requantized target weights and saved inputs as DFlash. It is separate from
the B8 row-zero sequential correctness oracle. The baseline is measured on the
current runner; the historical 19.310 tokens/s is not inserted into new results.
The primary decode timer includes initial and later NPU prepass. Prefill, setup,
warmup and the prefilled first token are excluded. The paper-era timing variant
is reported separately, not used as the default speedup denominator.

Full80 preserves its request logs and refuses to mix a second run into an
existing output directory. For a local repeat, choose a new output name, e.g.
`make full80 FULL80_OUTPUT=build/repeat2.full80.json`. Nightly checkouts start
with a clean build directory. Keep manually compiled artifacts separate from
LIT's directory: install/bin and build/bin compiler binaries can have different
hashes even when installation only changed their library search paths.

The nightly workflow keeps Full80 separate from short-profile history. It adds
an overall paired table to the job summary and uploads `full80.json`, per-dataset
summaries, request traces (including partial runs), verification reports and
build manifests. A measured speedup is reported without a machine-specific
minimum performance pass threshold. Narrow dispatches such as `verify,profile,full80`
produce downloadable partial-run artifacts and do not publish into main's
performance history.

## Runtime budget

Historical HX370 process times were 31.3 minutes for B1 and 16.3 minutes for
DFlash: about 47.6 minutes combined, excluding download, packing and compilation.
Budget 50–60 minutes for a warm paired study and 90 minutes for its LIT timeout.
The full nightly job limit increases from 480 to 540 minutes. These are planning
estimates; the new toolchain and Krackan runner need their own measurements.
