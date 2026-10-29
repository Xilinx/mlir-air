> Status: reopened after the 2026-10-29 audit. A skewed five-job compiler reproducer still places a drain-cap wait before its producer's final feed. Steps 3 and 6 need follow-up; the PR is not ready to declare fully fixed. The previously passing NPU matrix remains valid, and merged attention remains blocked by shared backend placement. See [PR2195_AUDIT.md](PR2195_AUDIT.md).

> Commit preparation: user authorized committing/pushing this incomplete snapshot. Applied CI's clang-format 17.0.1 procedure, rebuilt `air-opt`/`aircc`, and reran the compiler suite: 705 passed, no failures. The known asymmetric cap failure remains open. Hardware evidence is from before the formatting-only changes.

# PR #2195 review resolution plan

PR: https://github.com/Xilinx/mlir-air/pull/2195

Workspace: `/home/vule/workspace/mlir-air-pr2195`

Branch: `claude/airrt-weave-by-job`

Starting head: `b38970143a8beae1ca58b858818fab5abe067f69`

Reviewed base: `9cfca99cbc9ea08856c899fb1319bdad7535b2c3`

Synced upstream main: `66a24f982a1e78553f682c5740d50c2216444584`

Post-sync head: `ef7d91082f5936a5d96f7f11296ead442128f3d0`

## Scope and working rules

Address all six Erwei inline comments, his review-summary test requests, and all four Copilot comments. Preserve the fixes for hoisted read-after-write dependencies and premature shim-feed waits. PR #2192 is also open; account for its changes carried by this branch.

- Work on one item at a time. Update this file with findings, decisions, changed files, and validation evidence.
- Distinguish reviewer reproductions, source-confirmed mechanisms, and independently reproduced failures.
- Establish the ordering model before choosing the final scheduler design.
- Preserve each channel's FIFO order, task-definition dominance, producer progress before waits, and in-flight limits.
- User authorized implementation, compiler tests, NPU validation, and subsequently formatting, committing, and pushing the current changes. GitHub review replies and thread resolution remain unauthorized.
- NPU correctness validation is authorized. Serialize NPU use with `flock /tmp/mlir-air-npu.lock`; use `--poll-wait` for timing.

## Progress

- [x] Read PR description, complete diff, discussion, both reviews, and all 10 inline threads.
- [x] Clone the exact PR branch into a separate workspace and verify its head.
- [x] Write this plan.
- [x] On user request, merge 27 missing upstream commits and push the PR branch to the fork (2026-10-28). No conflicts; `git diff --check HEAD^ HEAD` passed. No builds or tests run. Review reproductions must now account for this updated baseline.
- [ ] Complete steps 1–7 below. Implementation is authorized. Steps 3 and 6 reopened by the audit.

## 1. Establish the ordering contract

- [x] Trace representative IR through `air-renumber-dma`, AIR-to-AIE conversion, `air-opt-shim-dma-bds`, `air-to-std`, and `airrt-to-npu`.
- [x] Identify the exact transformations that regroup channels or clone transfers; record source locations and before/after order.
- [x] Explain the four-layer launch-loop result: when IDs repeat, when waves are assigned, and which scheduling boundaries distinguish iterations.
- [x] Cover two jobs within one launch iteration as well as jobs in different iterations.
- [x] Decide whether preserving cross-channel order upstream and using the existing round rule removes the need for program-order segmentation.
- [x] Evaluate `air.launch_wave` for iteration boundaries without assuming it distinguishes jobs inside one iteration.

Exit: document a concrete ordering invariant and selected design. If metadata remains necessary, explain the case that requires it and define duplicate-ID behavior explicitly.

Sources: [Erwei on metadata and pipeline ordering](https://github.com/Xilinx/mlir-air/pull/2195#discussion_r4385245075), [Erwei on repeated IDs](https://github.com/Xilinx/mlir-air/pull/2195#discussion_r4385245088).

## 2. Preserve channel order and handle ambiguous IDs

- [x] Add a drain-order regression reproducing Erwei's swapped destination buffers.
- [x] Add conflicting feed-order and drain-order cases, including the documented fallback if segmentation remains.
- [x] Add repeated-ID cases from cloned jobs and launch iterations.
- [x] Ensure tied IDs cannot silently claim a valid ordering and recreate the premature wait.
- [x] Implement the smallest fix consistent with step 1. Rejecting ambiguous segmentation alone must not be presented as fixing the original deadlock.

Exit: same-channel feed and drain order is preserved; repeated IDs have documented, tested behavior.

Sources: [Erwei on drain order](https://github.com/Xilinx/mlir-air/pull/2195#discussion_r4385245084), [Copilot on drain order](https://github.com/Xilinx/mlir-air/pull/2195#discussion_r4372913319).

## 3. Resolve waits that conflict with the drain cap

- [x] Reproduce six leading drains, a producing feed, and a second channel reading the last drain back without program-order attributes.
- [ ] Ensure pulling drains toward their awaiter cannot place a cap wait before the feed needed to retire it. Audit counterexample: five jobs with feed counts `[14, 1, 1, 1, 1]`.
- [x] Reproduce five chained jobs with one drain each, crossing the shared four-task cap.
- [x] Ensure each segment's drains move consistently with its feeds if segmentation remains, including segments with at most four drains.
- [ ] Check channel order, configure-before-await dominance, producer feeds before blocking waits, and caps together. Uniform/sampled cases pass; deterministic asymmetric cases fail.
- [ ] Replace the proportional producer-completion estimate with a justified dependency boundary or conservative scheduling rule; add asymmetric-job regressions.

Exit: both cap-crossing regressions produce schedules satisfying all four invariants. Structural checks alone do not establish hardware correctness.

Sources: [Copilot on producer bounds](https://github.com/Xilinx/mlir-air/pull/2195#discussion_r4372913289), [Copilot on shared caps across segments](https://github.com/Xilinx/mlir-air/pull/2195#discussion_r4372913341).

## 4. Make hoisted-loop dependency repair complete

- [x] Add a read-after-write case whose writer loop returns an async token and a non-token value.
- [x] Inspect token result conventions for both `scf.for` and `scf.parallel`; handle supported token results without assuming one result at index zero.
- [x] Inspect all `addDependencyBetweenOps` callers and determine whether token-producing ancestor loops are valid sources generally.
- [x] Remove `tokenLoopsAreSources` if no concrete incompatible caller exists; otherwise document and test that incompatibility.
- [x] Preserve dominance and avoid dependencies on enclosing or later loops.

Exit: reads depend on the actual writer loop for supported result shapes, with evidence for the final API choice.

Sources: [Erwei on the mode switch](https://github.com/Xilinx/mlir-air/pull/2195#discussion_r4385245094), [Copilot on multiple results](https://github.com/Xilinx/mlir-air/pull/2195#discussion_r4372913366).

## 5. Simplify the surviving implementation

- [x] If segmentation remains, pass feeds, drains, and layout/cap options explicitly into `flushOne` instead of mutating captured state.
- [x] If program-order metadata remains, remove `keep-program-order` and the temporary `air.keep_program_order` module attribute; pass required configuration directly.
- [x] If step 1 removes these mechanisms, delete obsolete attributes, options, branches, and tests; retain behavior coverage.
- [x] Rewrite test comments to state required behavior rather than historical implementation details.

Exit: no unnecessary mode switches or hidden scheduling inputs remain.

Sources: [Erwei on scheduler state](https://github.com/Xilinx/mlir-air/pull/2195#discussion_r4385245091), [Erwei on test wording](https://github.com/Xilinx/mlir-air/pull/2195#discussion_r4385245101).

## 6. Validate the final design

- [x] Add a pipeline regression starting before `air-renumber-dma` and reaching `airrt-to-npu`, covering the selected ordering mechanism.
- [x] Demonstrate each new bug regression fails on the relevant pre-fix revision and passes after its fix where practical.
- [x] Run focused lowering, shim-weave, cap, and dependency tests, including the existing tests from #2192 and #2195.
- [x] Run the relevant broader lit suite after focused checks pass; record tool versions, commands, counts, and failures.
- [x] Complete the runnable NPU matrix: two-job engine, four-layer launch loop, five-job chain, and six-drain read-back; all bit-identical to their NPU references. Record the shared placement blocker for merged attention separately.
- [x] Report the merged GEMM + FlashAttention runtime-sequence hang separately; do not claim it resolved without new evidence.
- [ ] After the remaining cap fix, rerun the asymmetric checks, compiler suite, and affected NPU matrix with new artifact identities.

Exit: record actual validation and limitations. Do not reuse historical PR test counts as evidence for new changes.

## 7. Prepare the review handoff

- [x] Map every review concern to its fix or evidence-backed design decision, test, and source location.
- [x] Update the proposed PR description around the final implementation and actual validation.
- [x] Identify any remaining blockers or unverified reviewer claims.
- [x] Present the diff and results for review. The user subsequently authorized formatting, committing, and pushing; leave GitHub replies and thread resolution pending authorization.

## Evidence and decisions

- Initial inspection: `renumberMemcpyIfOps` assigns static operation IDs by walking IR; it does not number dynamic executions.
- Initial inspection: `air.launch_wave` is carried to configured tasks and removed after `boundShimFeedBursts`.
- Erwei reported both original bugs and drain-order/repeated-ID failures. New compiler regressions now cover the mechanisms, including a pipeline check proving cloned operations retain repeated IDs.
- Copilot's two drain-cap ordering failures and multi-result-loop gap were independently reproduced on the synced PR baseline; the new checks pass after the changes.
- Setup performed no implementation changes or tests. Implementation, compiler validation, and subsequently NPU validation were authorized and completed for the runnable cases.

### Selected ordering design

`IsolateAsyncDmaLoopNestInSCFForPattern` partitions a mixed loop into channel-connected components. `air-opt-shim-dma-bds` invokes it before wrap/stride specialization; aircc also invokes it earlier after hoisting. A read-back token only connects the drain and reader, not every other channel feeding the producer. Preserve the mixed loop whenever it writes and reads the same host buffer (including views), rather than reconstructing issue order from static IDs. Keep ordinary independent-channel isolation unchanged. Remove program-order metadata, its option/module flag, and segmented scheduling. The existing round scheduler must treat a read-back as a barrier after every preceding feed.

The four-layer case does not prove IDs are unique: rolled loop bodies are scheduled per block, while unrolling clones IDs. Launch waves identify launches/iterations only where launch-end markers distinguish them; they cannot distinguish two jobs inside one iteration. The selected design does not depend on IDs or wave numbering. Pipeline regressions will cover both shapes.

Toolchain setup: existing local LLVM installations lack `useStrictPropertiesInAssemblyFormat`. Downloaded the repository-pinned MLIR `24.0.0.2026100106+bdf04ebb` and AIE `1.4.5.dev83+g1631e1b` wheels into `/tmp/pr2195-toolchain`, configured an isolated `build/`, and built the unchanged synced PR. Baseline executable: `/tmp/pr2195-baseline-air-opt`.

### Completed implementation and validation

- Removed `keep-program-order`, `air.program_order`, `air.keep_program_order`, and the segmented scheduling path, including its `deferCaps` / `forceLayout` state. No duplicate-ID policy or sorting fallback is needed because IDs no longer drive scheduling. Contradictory/tied legacy hints cannot override channel order.
- Preserved mixed-loop order at the isolation rewrite shared by the early launch pass and shim-BD optimization. Host buffer views/casts are recognized. The independent-channel isolation test remains enabled and passes.
- Used a running maximum so each read-back wait follows all earlier feeds, with a barrier floor that prevents later jobs' feeds on other channels from overtaking it. Drain placement retains its cap lower bound; a conflicting reader is delayed instead of pulling drains ahead of their producer feeds.
- Enabled local read-back waits in loop bodies and retired those bodies' local drains before the backedge. Loops without an actual local read-back receive no added blocking fence. Cross-block drain dependencies retain their diagnostic.
- Removed `tokenLoopsAreSources`; the existing preceding-ancestor check applies to all callers. All async-token results of `scf.for` / `scf.parallel` are accepted, including loops with a live non-token result.
- Replaced the attribute-specific tests with source-order and actual pipeline tests. Added conflicting drain/feed hints, repeated IDs after cloning, both drain-cap cases, casts, independent isolation, multi-result `scf.for`, and `scf.parallel` coverage.
- Regression proof: each of the six-leading-drain, five-chained-job, and swapped-drain-destination cases fails its checks on `/tmp/pr2195-baseline-air-opt` and passes on `build/bin/air-opt`. Feed-order coverage passes on both. The baseline also fails the new multi-result-loop, isolation, and pipeline check files. Baseline pipeline diagnostics include missing channel counterparts after isolation; this is not presented as a hardware deadlock reproduction.
- Additional schedule audit: 60 six-job chains with 1–8 feeds per job and 100 with 1–99 feeds per job. Checked that every emitted drain await follows every feed belonging to its producer job. All 160 pass. This checks host issue order, not NPU execution or queue occupancy over time.
- Final full compiler suite: **705 passed, 0 failed, 11 unsupported, 7 expected failures; 723 discovered tests**.
- `git diff --check` passed. No commits, pushes, review replies, or thread resolutions performed for these fixes. Subsequent NPU results are in PR2195_NPU_VALIDATION.md.

Commands (from this workspace):

```sh
/home/vule/workspace/mlir-air/.venv_airbuild/bin/cmake --build build --target air-opt aircc air-translate air-runner -j 8
/home/vule/workspace/mlir-air/.venv_airbuild/bin/lit -sv build/mlir/test
python3 build/pr2195-evidence/pr2195-baseline-checks.py
python3 build/pr2195-evidence/check-cap-cases.py
python3 build/pr2195-evidence/pr2195-check-chains.py
python3 build/pr2195-evidence/pr2195-check-long-chains.py
git diff --check
```

Evidence: `build/pr2195-evidence/lit-final.log`, `baseline-comparison.json`, `cap-case-comparison.json`, `generated-chains.log`, and individual baseline diagnostics. Toolchain paths are local to this session; saved baseline-comparison JSON includes the exact commands. The compiler reports LLVM 24.0.0, optimized with assertions; host compiler is Clang 18.1.3 using GCC 13's standard library. At the compiler-only stage the historical four-layer result had not been rerun. The subsequent hardware matrix includes that case; a passing run is not proof that static IDs were unique.

Review mapping and proposed PR text: [PR2195_REVIEW.md](PR2195_REVIEW.md).

Final scheduling audit also added `@later_feeds` in `shim_read_back_order.mlir`: a third channel used only after a read-back cannot overtake it or issue a queue-cap wait ahead of that reader. This case fails on the synced PR baseline and passes with the barrier floor.

The input-order contract matters: a read-back burst must reach the weave in source/job order, which the upstream isolation guard and real pipeline tests establish. Four older hand-written AIRRt fixtures (`@mid`, `@midp`, `@shared_read_back`, `@disjoint_read_back`) supplied producer B feeds after their producer's read-back wait, relying on the old weave to move them across it. Their inputs now place those feeds before that barrier (and the independent disjoint read before it); their existing output checks remain intact. Already-regrouped hand-written IR does not contain enough information to recover arbitrary job membership without an ordering annotation. The selected design avoids producing that shape for host read-back loops instead of guessing from static IDs.

### NPU validation completed (2026-10-29)

The user's hardware-validation request authorized builds and NPU runs. The initial compiler-only solution failed the original two-job hardware case despite passing lit. The compute device matched the freshly rebuilt PR baseline; the runtime placed terminal output drains too late. A diagnostic early-arm edit passed bit-identically. The production scheduler now arms terminal drains when queue capacity is available, including after read-back retirement proves a cap predecessor complete. Added `@terminal_output` and `@terminal_after_cap` compiler regressions.

Final rebuilt ELFs pass the two-job and four-layer cases for three seeds and five calls per seed, the five-job cap-crossing case for two seeds and three calls per seed, and the six-drain read-back variant for three seeds and five calls per seed. Every checked intermediate/final buffer matches the NPU reference as raw bf16 bits. The merged GEMM + FlashAttention case cannot reach execution because baseline and candidate encounter the same buffer-placement failure. It is explicitly not claimed fixed.

Detailed environment, evidence, commands, limits, and final results: [PR2195_NPU_VALIDATION.md](PR2195_NPU_VALIDATION.md). The earlier compiler-only notes in this file describe the initial validation stage, not the final hardware coverage.
