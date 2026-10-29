# PR #2195 follow-up audit — 2026-10-29

Not all issues are resolved. No production source was changed during this audit. The remaining leading-drain producer-bound bug needs a fix and fresh validation before claiming completion.

## Review coverage

Re-fetched the PR description, discussion, both reviews, and all inline comments through GitHub CLI/API. GraphQL confirmed all 10 review threads were retrieved with no further pages: six Erwei comments and four Copilot comments. All 10 were unresolved on GitHub. At audit time, PR head was `ef7d91082f5936a5d96f7f11296ead442128f3d0`; the implementation and tests were local, uncommitted, and unpushed. The user subsequently authorized formatting, committing, and pushing this snapshot with the remaining issue documented.

| Review concern | Audit result |
| --- | --- |
| Erwei: extra option and temporary metadata (`r4385245075`) | Addressed locally: options, metadata, and segmented scheduling removed; cross-channel order preserved upstream. |
| Erwei/Copilot: drain FIFO order (`r4385245084`, `r4372913319`) | Addressed locally: ID sorting removed; destination-order regression passes. |
| Erwei: repeated/cloned IDs (`r4385245088`) | Addressed locally: IDs do not drive scheduling; pipeline test verifies repeated IDs after cloning. |
| Erwei: captured `flushOne` state (`r4385245091`) | Addressed locally: segmented path and mutable layout switches removed. |
| Erwei: dependency mode switch (`r4385245094`) | Addressed locally: flag removed; original and multi-result/parallel dependency tests pass. |
| Erwei: historical test wording (`r4385245101`) | Addressed locally: replacement tests describe the required behavior. |
| Copilot: producer bound under drain caps (`r4372913289`) | **Partial/open:** original regression passes; asymmetric producer counterexample still fails. |
| Copilot: shared cap across five jobs (`r4372913341`) | **Partial/open:** segmentation removed and tested five-job case passes; asymmetric five-job case still fails. |
| Copilot: non-token plus token loop results (`r4372913366`) | Addressed locally: all async-token results collected; regression passes. |

Erwei's review-summary requests for conflicting feed/drain hints, repeated IDs, and an actual renumber-to-NPU pipeline are covered by the added compiler tests. These passes do not prove the remaining cap rule correct.

## Remaining correctness finding

`mlir/lib/Conversion/AIRRtToNpuPass.cpp:3989` estimates the completion round of leading drain `j` as `ceil((j + 1) * rounds / drains) - 1`. At the four-task cap, the next drain uses that estimate to place an await on its predecessor. This proportional estimate is not a bound on actual producer completion when jobs have unequal input-feed counts.

The deterministic five-job case has leading output drains D0–D4 and feed counts **[14, 1, 1, 1, 1]**. Each later job reads the previous output after an explicit wait; all first-job feeds occur before the first read-back in the input. D0 requires the first job's feeds, but the emitted cap wait on D0 precedes the final B feed at input offset 832. A producer that requires that feed cannot complete D0 at this wait. A six-job variant **[10, 1, 1, 1, 1, 1]** fails similarly at offset 576.

This is an emitted-schedule correctness counterexample, not a demonstrated NPU hang. The audit AIRRt fixture records intended producer membership in its generator/checker; it contains no executing compute-core implementation. The prior 160 sampled schedules and runnable hardware cases did not expose this skew.

Evidence under `build/pr2195-audit/`:

- `check_five_asymmetric.py`, `five_asymmetric.log`, `five_asymmetric.airrt.mlir`, `five_asymmetric.npu.mlir`.
- `check_asymmetric.py`, `asymmetric.log`, `asymmetric_min.airrt.mlir`, `asymmetric_min.npu.mlir`.
- `asymmetric_expected.check` and `asymmetric_filecheck.log`: expected producer-before-wait ordering fails FileCheck.

Reproduce the five-job check from this checkout (expected exit status 1):

```sh
python3 build/pr2195-audit/check_five_asymmetric.py
```

Reproduce the saved six-job FileCheck failure:

```sh
build/bin/air-opt build/pr2195-audit/asymmetric_min.airrt.mlir \
  -airrt-to-npu -o /tmp/pr2195-audit-asymmetric.npu.mlir
/tmp/pr2195-toolchain/extracted/mlir/bin/FileCheck \
  build/pr2195-audit/asymmetric_expected.check \
  --input-file=/tmp/pr2195-audit-asymmetric.npu.mlir
```

The next fix needs a justified producer/dependency boundary or conservative scheduling rule, rather than a global feed/drain ratio. Add both asymmetric cases as permanent regressions and rerun compiler and affected hardware validation after that fix. Steps 3 and 6 in the plan are reopened.

## Validation rechecked

- Fresh full compiler suite: **705 passed, 0 failed, 11 unsupported, 7 expected failures**, 723 total; log `/tmp/pr2195-audit-lit.log`. The new failing audit fixtures are outside that suite, explaining why it stays green.
- `git diff --check` passes.
- All 14 saved NPU artifact hashes match. The current tracked source diff exactly matches `build/pr2195-npu/source_diff.patch` used for hardware validation.
- Independently compared all 18 saved output-buffer pairs again: bit-identical. The existing two-job, four-layer loop, five-job chain, and six-drain hardware results remain valid for their recorded shapes and inputs.
- No unchanged NPU runs were repeated during this audit. No new hardware claim is made for the asymmetric case.
- The merged GEMM + FlashAttention case remains unvalidated: both baseline and candidate fail the same AIE buffer placement before execution. Its reported runtime hang is not established fixed.

Detailed earlier passing evidence: [PR2195_NPU_VALIDATION.md](PR2195_NPU_VALIDATION.md). Concern-to-test mapping and implementation details: [PR2195_REVIEW.md](PR2195_REVIEW.md).

## Formatting and commit preparation

On the subsequent commit/push request, applied the repository CI's `git-clang-format origin/main` procedure using clang-format **17.0.1** and the checked-in LLVM style. Changes consist of spacing, line wrapping, and include ordering. A second formatter check reports no changes. No Python source changed in this snapshot.

Rebuilt `air-opt` and `aircc`, then reran the full compiler suite: **705 passed, 0 failed, 11 unsupported, 7 expected failures**. Log: `/tmp/pr2195-post-format-lit.log`. `git diff --check` passes. The asymmetric five-job checker still fails as documented. This snapshot is intentionally not presented as resolving that remaining issue.

Hardware runs were not repeated for formatting. The saved hardware source patch identifies the pre-format implementation; the committed source includes the subsequent formatting changes. No review replies or thread resolutions accompany the push.
