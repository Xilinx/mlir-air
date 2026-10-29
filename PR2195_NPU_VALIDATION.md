# PR #2195 NPU validation

> Audit update (2026-10-29): these hardware results remain valid, but do not establish that all review concerns are resolved. A new asymmetric-job compiler schedule fails the producer-before-cap-wait invariant; it has not been reproduced on hardware. See [PR2195_AUDIT.md](PR2195_AUDIT.md). All 14 saved artifact hashes and the tracked source diff were rechecked; 18 saved output-buffer comparisons were independently confirmed bit-identical.

The runnable validation matrix completed on 2026-10-29 in `/home/vule/workspace/mlir-air-pr2195`, before the subsequent formatting and commit request. The compiler baseline is synced PR head `ef7d91082f5936a5d96f7f11296ead442128f3d0`, built separately in `/tmp/pr2195-hw-baseline` with the same MLIR/AIE packages as the candidate.

## Result

All four runnable hardware cases below pass. Outputs are compared as raw bf16 bits, including intermediate buffers. These are correctness checks; no latency or speedup claim is made.

| Case | Inputs and repeated calls | Reference | Result |
| --- | --- | --- | --- |
| Original two-job GEMM engine, with host read-back and residual | Seeds 5, 17, 29; 5 calls per seed | Freshly compiled synced PR baseline; hidden and final output buffers | All bits identical; all outputs finite |
| Original four-layer launch loop around that engine | Seeds 5, 17, 29; 5 calls per seed | Freshly compiled synced PR baseline; every activation slot and final scratch buffer | All bits identical; all outputs finite |
| Five chained GEMM jobs sharing channels, five drains per physical output channel | Seeds 19, 37; 3 chained calls per seed | Five independent single-job NPU calls per seed | All five intermediate/final outputs bit-identical; all finite |
| Six tile drains per physical output channel, followed by read-back and a second job | Seeds 5, 17, 29; 5 calls per seed | Unsplit two-job engine from the synced PR baseline | Hidden and final outputs bit-identical; all finite |

The first two cases also pass the existing CPU bf16-reference cosine threshold of 0.999. Repeated calls restore inputs and compare output bits, checking that later calls remain correct.

The six-drain case splits the first job's four folded output transfers into six tile transfers on each of four physical output channels. Its read-back waits name all six writes. The compute device before the runtime sequence is byte-for-byte identical to the unsplit engine; compute kernels, inputs, and output contents are unchanged. This exercises six leading drains from one producing job, not a claim that the entire GEMM job has only one input DMA. The exact one-producing-feed shape remains covered by the compiler regression.

Static inspection of the emitted schedules finds a maximum of four outstanding output drains per channel in the six-drain case and two in the five-job case. Hardware completion and numerical comparisons validate those schedules; these counts are not device-trace measurements.

## Hardware exposed an additional bug

The initial compiler-only revision passed 705 compiler tests but its two-job engine ELF did not complete within the 20-second polling deadline. The historical known-good ELF and a freshly rebuilt synced PR baseline both ran correctly with the same inputs. The baseline and candidate compute device preceding the runtime sequence were identical.

The runtime-sequence difference placed the final output drains behind many second-job input tasks. Once those inputs encountered a queue-cap wait, the cores could be waiting for output space with no output receiver armed. Moving only the terminal output configure/start operations earlier in a diagnostic placed module made the output bit-identical to the reference.

The production fix now arms terminal drains early when their queue slots are available. Beyond the first queueful, it uses an earlier read-back wait on the cap predecessor, or a later task on the same channel, as proof that the predecessor has retired. It retains channel FIFO order and delays the drain when that retirement proof is unavailable. This avoids placing a blocking cap wait ahead of the producer feeds it needs.

Two compiler regressions cover this rule: `@terminal_output` and `@terminal_after_cap` in `mlir/test/Conversion/AIRRtToNpu/shim_read_back_caps.mlir`. The second fails with the saved compiler before the cap-aware extension and passes after it. Final engine ELFs were rebuilt through aircc and validated without manual runtime-sequence edits. The six-drain variant necessarily changes the AIRRt transfer tiling before the production scheduler; it does not hand-order the resulting sequence.

## Environment and evidence

- Hardware: AMD Ryzen AI MAX+ 395, device `0000:c6:00.1`, reported as `RyzenAI-npu5`; designs target `npu2` / AIE2P.
- XRT 2.21.75; NPU firmware 1.1.2.65.
- MLIR wheel `24.0.0.2026100106+bdf04ebb`; AIE wheel `1.4.5.dev83+g1631e1b`.
- Existing Peano `22.0.0.2026091501+631269`; the original prebuilt `mm_engine.o` is reused in both arms. Peano is recorded explicitly rather than described as the newer repository pin.
- All NPU runs are serialized with `flock /tmp/mlir-air-npu.lock` and use polling completion checks. No benchmark medians were collected.
- Single-job and five-job cases use `--stack-size 512`; original two-job and four-layer cases retain their original default stack configuration.

Artifacts under `build/pr2195-npu/`:

- `engine_validation.log` and `engine_results.json`: final per-input results and output hashes.
- `artifact_hashes.json`: SHA-256 identities of the ELFs, input modules, split-drain module, and kernel object.
- `drain_queue_checks.json`: per-channel task counts, static outstanding-drain maxima, and device-prefix equality check.
- `engine2/candidate_seed5.log`: the initial hardware timeout, retained as negative evidence.
- `engine2/rearm_seed5.log`: the diagnostic early-output-arm experiment.
- `terminal_regressions.json`: before/after compiler checks for terminal-drain placement.
- `compile_commands.json`: exact per-case compiler arguments and working directories.
- Per-case `compile_validation.log`, AIR/AIRRt inputs, debug IR, and per-seed run logs.

Reproduce from the checkout:

```sh
# Rebuild aircc, then rebuild the per-case input.mlir files in their own directories.
# The compiler invocation and final inputs are recorded in each case's artifacts.
flock /tmp/mlir-air-npu.lock timeout 300 \
  /home/vule/workspace/mlir-air-action-expert/programming_examples/llms/smolvla/.venv312/bin/python \
  build/pr2195-npu/validate_engines.py
```

`generate_jobs.py` generates the one-job and five-job AIR modules using the existing SmolVLA GEMM engine builder. `split_drains.py` constructs the six-drain variant from the two-job engine's AIRRt checkpoint. `run_engine.py` and `run_five_jobs.py` perform the numerical comparisons. These helpers live beside the local build artifacts; they are not part of the compiler PR's source changes.

## Merged GEMM + FlashAttention limitation

The larger pre-existing two-herd reproduction did not reach hardware execution. Both the synced PR baseline and the revised compiler fail in the available AIE backend at the same engine-core buffer placement: `buf5` needs 5120 bytes with no remaining space. The memory map reserves 1024 bytes for the stack although the core carries `stack_size = 512`; its data buffers total 64768 bytes, so this layout exceeds 64 KiB. The existing per-herd stack-override procedure and an older available backend also fail placement.

This is a shared toolchain/placement blocker, not evidence that the previously reported merged-layer hang is fixed. No correctness or performance claim is made for the merged attention layer. Multi-result-loop and contradictory-order corner cases remain compiler regression checks; the hardware cases above exercise their surrounding integration mechanisms, not every possible IR shape.

## Compiler verification

Final full `lit build/mlir/test`: **705 passed, 0 failed, 11 unsupported, 7 expected failures**, 723 tests discovered. The 160 generated six-job schedule checks also pass. `git diff --check` passes. No commits, pushes, review replies, or thread resolutions were performed for these fixes.
