# PR #2195 review handoff

> Audit update (2026-10-29): not all issues are resolved. The proportional leading-drain producer bound still fails asymmetric five-job and six-job schedules. The earlier passing tests remain valid but insufficient. This handoff is not ready to post as a completed fix. See [PR2195_AUDIT.md](PR2195_AUDIT.md).

Changes were prepared in `/home/vule/workspace/mlir-air-pr2195`, based on synced head `ef7d91082f5936a5d96f7f11296ead442128f3d0`. The user subsequently authorized formatting, committing, and pushing this snapshot, including its documented unresolved issue. No review replies or thread resolutions are authorized.

## Design decision

Preserve cross-channel order where it is lost instead of reconstructing it from operation IDs. `IsolateAsyncDmaLoopNestInSCFForPattern`, used both by early launch isolation and by `air-opt-shim-dma-bds`, splits a mixed loop into channel-connected components. A drain-to-reader token does not connect every other channel feeding that drain's producer. Such a split can move a read-back wait ahead of the missing producer feeds.

The rewrite now leaves mixed loops with host read-backs intact, including casts/views of a shared buffer. Ordinary independent-channel loops still split. BD specialization can fold inner single-channel loops and unroll remaining channel-only loops in source order. The shim weave puts an awaiting feed after every earlier feed using a running maximum. Terminal output drains are armed early when queue capacity is available; an existing read-back can prove that a cap predecessor has retired.

Read-back bursts now have an explicit input-order requirement: producer feeds precede their read-back barrier, and later-job feeds follow it. A barrier floor prevents later channels from overtaking the reader and blocking at a queue-cap await. Four older synthetic AIRRt fixtures now express that source order; their existing output checks remain intact. Already-regrouped direct AIRRt input cannot reveal arbitrary job membership. The pipeline regression demonstrates that the revised compiler preserves the required information before the weave.

Static IDs no longer participate in scheduling. Neither `air.program_order` nor a new pass option is needed. Existing `air.launch_wave` keeps its original purpose; it is unnecessary for this fix and would not distinguish two jobs within one iteration. Four rolled launch iterations and two jobs cloned after renumbering are covered separately. The latter test verifies that cloned transfers really do share IDs.

Preserving the loop also exposes local read-back dependencies to AIR lowering. It now inserts local blocking waits and retires that block's local drains before its backedge, preventing completion tokens from leaking into the next iteration. This fence is added only for blocks with an actual local read-back. Cross-block dependencies still produce the existing diagnostic.

## Review concern mapping

| Concern | Resolution | Evidence |
| --- | --- | --- |
| [Erwei: new option/IR state](https://github.com/Xilinx/mlir-air/pull/2195#discussion_r4385245075) | Removed the option, both temporary attributes, and segmentation. Preserved order at the shared isolation rewrite. | `isolate_host_read_back.mlir`; both pipeline entry points in `shim_read_back_pipeline.mlir`. |
| [Erwei: drain order](https://github.com/Xilinx/mlir-air/pull/2195#discussion_r4385245084), [Copilot: same issue](https://github.com/Xilinx/mlir-air/pull/2195#discussion_r4372913319) | Removed sorting by static IDs. Each channel retains its emitted order. | `shim_read_back_caps.mlir`, `@drain_order`: baseline swaps destinations and fails; updated compiler passes. |
| [Erwei: tied/repeated IDs](https://github.com/Xilinx/mlir-air/pull/2195#discussion_r4385245088) | IDs cannot affect the schedule. No silent segmentation or tie policy remains. | `@cloned_jobs` includes an intermediate-stage repeated-ID check and final runtime-sequence ordering checks; `@jobs` covers four rolled launch iterations. |
| [Erwei: captured segment state](https://github.com/Xilinx/mlir-air/pull/2195#discussion_r4385245091) | Removed the segmented path, `flushOne`, `deferCaps`, and `forceLayout`. The original burst layout remains one operation. | No mutable segment configuration remains. |
| [Erwei: dependency mode switch](https://github.com/Xilinx/mlir-air/pull/2195#discussion_r4385245094) | Removed `tokenLoopsAreSources`. Token-producing preceding ancestor loops are valid sources for every caller under the existing scope/order checks. | Full compiler suite, original hoisted-loop regression, and new multi-result/parallel cases pass. No incompatible caller was found. |
| [Erwei: historical test wording](https://github.com/Xilinx/mlir-air/pull/2195#discussion_r4385245101) | Replaced the attribute-specific weave test with `shim_read_back_order.mlir`; comments describe required producer/read-back behavior. | Source-order regression passes. |
| [Copilot: producer bound versus drain cap](https://github.com/Xilinx/mlir-air/pull/2195#discussion_r4372913289) | **Partial/open.** The named regression is fixed, but the proportional producer bound can still put a cap wait ahead of required feeds. | `@cap_readback` passes; audit feed-count cases `[10,1,1,1,1,1]` and `[14,1,1,1,1]` fail the producer-before-wait check. |
| [Copilot: cap across five segments](https://github.com/Xilinx/mlir-air/pull/2195#discussion_r4372913341) | **Partial/open.** Segmentation is removed, but shared cap scheduling is not generally correct for asymmetric jobs. | `@five_jobs` and the five-job NPU engine pass; the asymmetric five-job audit case fails structurally. |
| [Copilot: multi-result loops](https://github.com/Xilinx/mlir-air/pull/2195#discussion_r4372913366) | Collect every async-token result instead of requiring one result at index zero. | `hoisted_loop_multi_result_raw_dependency.mlir`: live index plus two tokens, and a parallel writer. The multi-result case fails baseline and passes updated compiler. |

Erwei's requested conflicting feed/drain and repeated-ID tests are covered by `shim_read_back_caps.mlir` and `shim_read_back_pipeline.mlir`. Sorting fallback no longer exists; the tests instead require hints to have no effect on channel order. Pipeline coverage starts before `air-renumber-dma` and includes AIR-to-AIE, shim-BD optimization, AIR-to-std, and AIRRt-to-NPU.

## Source locations

- `mlir/lib/Transform/AIRDependencyScheduleOpt.cpp`: mixed-loop host read-back guard near line 7401.
- `mlir/lib/Conversion/AIRRtToNpuPass.cpp`: running feed-order maximum near line 3909; terminal-drain placement near line 4011 and cap/read-back delay handling below it.
- `mlir/lib/Conversion/AIRLoweringPass.cpp`: nested local read-back waits and loop retirement near line 1476.
- `mlir/lib/Util/Dependency.cpp`: token-producing ancestor-loop results near line 3369.

## Validation and limitations

Built `air-opt`, `aircc`, `air-translate`, and `air-runner` with the checkout's pinned MLIR/AIE wheels in an isolated build. Full `lit build/mlir/test`: **705 passed, 0 failed, 11 unsupported, 7 expected failures**, 723 discovered tests. `git diff --check` passes.

The saved synced-baseline compiler fails the new cap, swapped-destination, multi-result-loop, isolation, and pipeline checks. Contradictory feed-order coverage passes on both revisions. The additional `@later_feeds` regression catches overtaking by a channel used only by the next job. It fails on the baseline and passes after the fix. An additional generated-schedule audit checks every drain await against its producer's feeds for 160 asymmetric six-job chains; all pass. Commands and artifacts are recorded in `PR2195_PLAN.md` and `build/pr2195-evidence/`.

Hardware validation now passes the original two-job engine, four-layer launch loop, five-job cap-crossing chain, and six-drain read-back variant. Every checked output is bit-identical to its NPU reference across the documented inputs and repeated calls. Hardware exposed a late terminal-drain flaw in the initial compiler-only revision; the scheduler fix and both terminal-drain regressions are included. Details: [PR2195_NPU_VALIDATION.md](PR2195_NPU_VALIDATION.md). No performance claim is made. The existing proportional drain-placement model remains for bursts without explicit producer boundaries. Host buffers behind unrelated aliasing function arguments are not given a new alias analysis by this change. Merged GEMM + FlashAttention validation is blocked before execution by the same AIE buffer-placement failure on baseline and candidate; its previously reported hang is not claimed fixed.

## Proposed PR title

[AIR] Preserve shim read-back order and repair hoisted loop dependencies

## Proposed PR description (draft; cap fix still incomplete)

Chained jobs can issue a host read-back before the producer's feeds or lose the dependency on a hoisted writer loop. Preserve cross-channel order when loop isolation encounters a shared host buffer, then schedule each read-back wait after every earlier feed. Delay conflicting readers. Local read-backs in rolled loops wait before reading and retire their local drains before the next iteration. Leading-drain cap waits still need a correct producer bound for asymmetric jobs; this snapshot is not a complete fix.

Accept all async-token results from preceding writer loops, including loops with non-token results. Remove program-order metadata, the pass option/module flag, segmented scheduling, and the dependency mode switch.

Validation: full compiler lit suite, 705 passed with no failures; 11 unsupported and 7 expected failures. Added before/after regressions for both drain-cap failures, destination order, loop-result dependencies, repeated cloned IDs, and the lowering pipeline. The runnable NPU matrix also passes with bit-identical outputs. The merged attention case remains blocked by a shared backend placement failure. No performance claims.
