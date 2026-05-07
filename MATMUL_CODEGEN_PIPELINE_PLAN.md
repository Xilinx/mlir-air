# C++ Matmul Codegen Pipeline — Design Plan

Replace the transform-dialect scripts that drive matmul tiling/bufferization/vectorization in mlir-air with a sequence of focused C++ MLIR passes.

**Goal**: parametric, generally-applicable, debuggable, individually testable. Eventually supersede the per-test `transform_aie2*.mlir` scripts.

## Status

| Milestone | Status |
|---|---|
| **M0** — `air-matmul-pack-and-transpose` + `air-matmul-tile-l3-to-l2-copies` | ✅ landed; build clean; `check-air-mlir` passes; **IR equivalence verified byte-identical against transform-script Phases 1+3** on test 54 launch-tile input (with-perms) and on a small synthetic input (with- and no-perms) |
| **M1** — Group B (passes 13–22) for `programming_examples/matrix_multiplication/{bf16,i8}` | ✅ landed and **hardware-validated end-to-end on NPU2** (both i8 and bf16 prog_ex matmul examples PASS via `--compile-mode=compile-and-run --arch=aie2p`). See M1 sub-status. |
| **M2** — Group A passes #2–12 for tests 53/54 (test 12 deferred — non-canonical pad+kernel.cpp flow) | ✅ landed and **hardware-validated end-to-end on NPU2** for both test 54 (BFP16 emulation, f32 in/out) and test 53 (bf16 in/out, truncf-fuse + hoist-cast-pairs). All four downstream paths still PASS (legacy 54, legacy 53, prog_ex i8, prog_ex bf16). M2d pending (transform script deletion + final doc cleanup). Profiling matrix: test 54 cpp 5.067 ms vs legacy 5.078 ms; test 53 cpp 1.766 ms vs legacy 1.731 ms — within run-to-run noise on both. |
| M3 (entire family) — automatic heuristic that derives pack / tile / vector params from matmul shape and writes the `air.matmul_codegen_config` carrier attribute | **deferred to a follow-up PR**. The carrier-attribute infrastructure (`MatmulCodegenConfig.{h,cpp}` + each consumer pass's "read from carrier attr if present, else use pass options" code path) **stays in this PR** as the external API. The pass that *populates* the attribute via heuristic (`air-matmul-set-codegen-config`) does not. Tests 37/48/53/54 cpp pipelines specify all tile/pack/vector parameters via per-pass options instead. |
| **M4a** — two-pack-level (test 37) infrastructure | ✅ landed and **hardware-validated on NPU2**. 7 new/extended passes + 2 marker-flow fixes in `tile-k-and-fuse-packs`. Test 37 cpp `air_tiled.mlir` matches legacy structurally (identical alloc set/memory spaces). Tests 37/53/54 cpp paths all PASS via `--use-cpp-pipeline` on NPU2. 390/391 lit tests pass (the 1 failure is unrelated, pre-existing). **Perf parity confirmed**: test 37 cpp 1.428ms vs legacy 1.430ms (0.1% faster); test 53 cpp 1.754ms vs legacy 1.745ms (0.5% slower); test 54 cpp 5.052ms vs legacy 5.032ms (0.4% slower) via `--profile-iters 50`; test 54 Makefile `profile` target 3-run mean cpp 3342us vs legacy 3314us (0.85% slower) — all within per-run noise (5–12%). |
| M4b | not started |
| **M5 Phase 1** — wire test 48 (latest Triton-XDNA matmul strategy) to `--use-cpp-pipeline` | ✅ landed and **hardware-validated on NPU2**. Test 48 transform_aie2p.mlir maps phase-by-phase to existing M0/M1/M2 passes (no new infrastructure needed); the test 53 cpp pipeline string was reusable verbatim. Open question on `air-hoist-cast-pairs` resolved: fixed-point converges to **structurally identical IR** vs the 4 sequential `transform.air.hoist_cast_pair` calls (same op counts, alloc shapes, nesting; only diffs are SSA renumbering and missing `prologue_herd`/`compute_herd`/`epilogue_herd` annotations — cosmetic). Perf parity confirmed: 3-run-mean of `min` times legacy 0.211ms vs cpp 0.208ms (cpp slightly faster by min, within noise). |
| M5 Phase 2 — Triton-XDNA driver.py invokes cpp pipeline directly (in Triton-XDNA repo, not mlir-air) | not started |

### M1a sub-status

Approach: extract reusable helpers. Each `transform.air.FooOp::apply` body is moved into a free function `xilinx::air::runFoo(...)` in [AIRMatmulCodegenHelpers.{h,cpp}](mlir/include/air/Transform/AIRMatmulCodegenHelpers.h); the apply() shrinks to a ~10-line stub that calls the helper, and the new C++ pass also calls it. Zero duplication, transform-script tests untouched.

| Sub-step | Pass | Status |
|---|---|---|
| **M1a-0** | All 6 passes registered in `Passes.td` / `Passes.h` / `PassDetail.h` / `Passes.cpp` / `CMakeLists.txt`; new files [AIRMatmulVectorizePasses.{h,cpp}](mlir/lib/Transform/AIRMatmulVectorizePasses.cpp), [AIRMatmulCodegenHelpers.{h,cpp}](mlir/lib/Transform/AIRMatmulCodegenHelpers.cpp) created. | ✅ |
| **M1a-1** | `air-fold-unit-extent-dims` (helper `runFoldUnitExtentDimsOnFunc`) | ✅ |
| **M1a-2** | `air-eliminate-redundant-vector-transfers` (helpers: areEquivalentIndices, areIdenticalReads, hasWritesBetweenReads, runEliminateRedundantVectorTransfers) | ✅ |
| **M1a-3** | `air-flatten-for-iter-args` (helper `runFlattenForIterArgs`) | ✅ |
| **M1a-4** | `air-hoist-loop-invariant-transfers` (helpers: dependsOnLoopIV, cloneOpAndOperands, hoistTransferPairFromLoop, runHoistLoopInvariantTransfers) | ✅ |
| **M1a-5** | `air-hoist-vector-transfer-pointers` (helper `runHoistVectorTransferPointers`; consolidated `dependsOnLoopIVForHoist` into `dependsOnLoopIV`) | ✅ |
| **M1a-6** | `air-matmul-tile-for-vectorize` (NEW pass: `scf::tileUsingSCF` + `mlir::loopUnrollByFactor`; pass options `matmul-tile-sizes`, `matmul-unroll-tile-sizes`, `matmul-unroll-factor`, `fill-tile-sizes`) | ✅ |

**M1a build clean. `check-air-mlir`: 381 pass / 7 XFail / 1 pre-existing unrelated failure (`AIRBufferize/air_transform_payload.mlir`) — unchanged from M0 baseline. AIRLinalgCodegen.cpp shrank from 5800 → 5013 lines (~800 lines moved out as helpers).** Lit smoke tests run for individual passes (`air-fold-unit-extent-dims`, `air-eliminate-redundant-vector-transfers`, `air-flatten-for-iter-args`).

### M1b sub-status

| Sub-step | Pass | Status |
|---|---|---|
| **M1b-1** | `air-vector-cast-for-emulation` (helper `runVectorTypeCastOnTarget`; pass options `target-element-type`, `input-indices`, `output-indices`) | ✅ landed; lit smoke verified |
| **M1b-2** | `air-hoist-cast-pairs` (fixed-point pass; helper `runHoistCastPair` extracted from `HoistCastPairOp::apply`) | ✅ landed |

### M1c sub-status — ✅ HARDWARE VALIDATED on NPU2

Both [i8/run.py](programming_examples/matrix_multiplication/i8/run.py) and [bf16/run.py](programming_examples/matrix_multiplication/bf16/run.py) now drive matmul codegen via the C++ pipeline (`air.passmanager.PassManager.parse(...)` invocation replacing `run_transform`). Validated end-to-end on the local NPU2 with `--direct-codegen --compile-mode=compile-and-run --arch=aie2p`:

| | i8 (i8 × i8 → i16) | bf16 (bf16 × bf16 → f32 or bf16) |
|---|---|---|
| `compile-and-run` exit | 0 (PASS!) | 0 (PASS!) |
| Pipeline | M1a + M1b passes (10 steps) | M1a + M1b + air-hoist-cast-pairs for bf16-output (11 steps) |

The pipeline IR is structurally equivalent to what the legacy transform script produces (same vector shapes, same iter_arg structure, same `memref.collapse_shape`-driven 1D access for L1 input buffers).

**Two implementation bugs found and fixed during HW validation:**

1. **Outermost vs innermost scf.for targeting**: my `air-hoist-loop-invariant-transfers` and `air-hoist-vector-transfer-pointers` initially targeted the *outermost* scf.for in each herd. The underlying helpers (`runHoistLoopInvariantTransfers`, `runHoistVectorTransferPointers`) filter by `getParentOfType<scf::ForOp>() == currentLoop` — only effective when the pass targets the *innermost* loop where the transfers actually live. Fixed by walking the herd for innermost scf.fors and calling the helper on each. *Lesson: the legacy script targets the outermost via `match + split_handle {overflow_result=1}`, but the helper's parent-check filter de-facto restricts useful work to whichever loop directly contains the transfers — so for a multi-level nested IR, the script's targeting is suboptimal/lucky.*

2. **Compute-herd-only filter**: my passes ran on every herd in the function. The fill herd (and epilogue herd) have no `vector.contract` but do have `vector.transfer_write` ops. `runHoistVectorTransferPointers` collapses the L1 buffer to 1D when called on the fill herd — which defeats the downstream `air-shrink-memref-sizes-by-access` pass (it can no longer detect per-core access slices, so the full 256KB accumulator stays on a single L1 tile instead of being split per-core). Fixed by adding a `herdHasVectorContract(herd)` filter, mirroring the legacy script's targeting of `%herd2` specifically (the compute herd).

**Hardware bench environment**: pyxrt is at `/opt/xilinx/xrt/python/`; xrt-smi at `/opt/xilinx/xrt/bin/`. Both must be on `PYTHONPATH`/`PATH` for `compile-and-run` mode to detect the NPU2 device and execute the xclbin. NPU2 hardware: AMD Ryzen AI 9 HX 370 / Strix.

**End-state (M0 + M1)**: `check-air-mlir` 381 pass / 7 XFail / 1 pre-existing unrelated failure unchanged. 10 new C++ passes registered (`air-matmul-pack-and-transpose`, `air-matmul-tile-l3-to-l2-copies`, `air-matmul-tile-for-vectorize`, `air-fold-unit-extent-dims`, `air-eliminate-redundant-vector-transfers`, `air-flatten-for-iter-args`, `air-hoist-loop-invariant-transfers`, `air-hoist-vector-transfer-pointers`, `air-vector-cast-for-emulation`, `air-hoist-cast-pairs`). 7 transform.air.* op apply()s now thin wrappers over shared helpers in [AIRMatmulCodegenHelpers.{h,cpp}](mlir/lib/Transform/AIRMatmulCodegenHelpers.cpp). **prog_ex matrix_multiplication/{bf16,i8} now drives matmul codegen via the C++ pipeline — first concrete supersession of a transform script, hardware-validated.**

### M2 sub-status (in progress)

**Scope**: Group A passes #2–12 covering tests 53/54 (canonical Phase 1–12 flow). Test 12 deferred — its transform.mlir uses pad + `linalg_promote` + `lower_linalg_to_func="kernel.o"` (non-canonical), and converting it would essentially mean rewriting the test. Test 12 may revisit later as its own sub-flow if useful.

| Sub-step | Description | Status |
|---|---|---|
| **M2a** | Extracted helpers to [AIRMatmulCodegenHelpers.h](mlir/include/air/Transform/AIRMatmulCodegenHelpers.h): `runRemoveUninitializedCopy`, `runEliminateCascadeMemcpy`, `runConvertMemrefCopyToLinalgCopy`, `runFuseIntoContainingMemref`, `containsOnlyTruncfOp`, `producesResultForOp`, `runFuseTruncfLinalg`, `runNormalizeForBounds`. Helpers live in [AIRLinalgCodegen.cpp](mlir/lib/Transform/AIRLinalgCodegen.cpp) (so they can call internal-linkage patterns/static helpers in that TU); `transform.air.{remove_uninitialized_copy, eliminate_cascade_memcpy, convert_memref_copy_to_linalg_copy, fuse_into_containing_memref, fuse_truncf_linalg, normalize_for_bounds}` apply()s shrunk to thin wrappers over them. | ✅ |
| **M2b-tail** | 3 contained passes registered + built: `air-matmul-cleanup-bufferize` (Phase 7 tail; calls `runRemoveUninitializedCopy` + `runEliminateCascadeMemcpy`), `air-matmul-fuse-pingpong-loops` (Phase 8; finds marked `copy_a_loop` / `copy_b_loop` / `k_reduction_loop` scf.fors, calls `runNormalizeForBounds` + upstream `mlir::fuseIndependentSiblingForLoops`), `air-matmul-fuse-output-truncf` (Phase 2 of test 53 / bf16-out flow; walks linalg ops looking for truncf-only consumers and calls `runFuseTruncfLinalg`). New file [AIRMatmulBufferizationPasses.{h,cpp}](mlir/lib/Transform/AIRMatmulBufferizationPasses.cpp). **`air-bufferize-one-shot` dropped — upstream `one-shot-bufferize{...}` pass already accepts the same options as a pipeline string and wrapping it adds nothing.** | ✅ |
| **M2b-bufferize** | Three `bufferizeToAllocation` wrappers landed in [AIRMatmulBufferizationPasses.cpp](mlir/lib/Transform/AIRMatmulBufferizationPasses.cpp): `air-matmul-bufferize-output-l2` (Phase 2: walks for first linalg.fill, bufferizes with `MemcpyOp::LinalgCopy` into memory_space=1), `air-matmul-bufferize-l1-output` (Phase 3 tail: finds `packed_matmul`-marked op, gets DPS-init producer (linalg.pack), bufferizes with `MemcpyOp::LinalgCopy` into memory_space=2), `air-matmul-bufferize-l1-inputs` (Phase 6a: finds `fused_lhs_l1_pack` / `fused_rhs_l1_pack`-marked ops, bufferizes with `MemcpyOp::MaterializeInDestination` into memory_space=2). | ✅ |
| **M2b-tile** | New file [AIRMatmulTilePasses.{h,cpp}](mlir/lib/Transform/AIRMatmulTilePasses.cpp). `air-matmul-tile-k-and-fuse-packs` (Phase 4: walks `packed_matmul`-marked op, captures pack_a/pack_b producers BEFORE tiling, tiles K iterator with `scf::tileUsingSCF` (LoopType::ForOp), annotates outer for with `k_reduction_loop`, then fuses each pack via `scf::tileAndFuseProducerOfSlice` and re-marks with `lhs_pack_in_k` / `rhs_pack_in_k`). `air-matmul-tile-cores` (Phase 5: walks `packed_matmul`-marked op, tiles with `scf::tileUsingSCF` (LoopType::ForallOp), annotates `compute_forall` and `matmul_compute`, then fuses the K-loop-fused packs into the forall and re-marks with `fused_lhs_l1_pack` / `fused_rhs_l1_pack`). | ✅ |
| **M2b-prologue** | `air-matmul-prologue-epilogue` landed in [AIRMatmulTilePasses.cpp](mlir/lib/Transform/AIRMatmulTilePasses.cpp). Walks for `linalg.fill`, calls `linalg::generalizeNamedOp`, annotates `init_fill`, optionally `linalg::interchangeGenericOp` with the configured perm (default `[1,0,2,3]`), then `tileAsForall` (helper wrapping `scf::tileUsingSCF` with `LoopType::ForallOp`) using `prologue-tile-sizes` (default `[8,4]`). Annotates `prologue_forall`. Same flow for `linalg.unpack` (tile by `epilogue-tile-sizes`, mark `epilogue_forall`). | ✅ |
| **M2c** | Pipeline string built directly in [test 54 run.py](test/xrt/54_matmul_padding_f32_bf16_emulation/run.py) and [test 53 run.py](test/xrt/53_matmul_padding_bf16/run.py) (both gated on `--use-cpp-pipeline`). **All 12 phases wire up correctly; both tests PASS end-to-end on NPU2** with `--compile-mode=compile-and-run` in ~60 s each. Test 54 uses the f32-in/out + BFP16-emulation flow (both `air-vector-cast-for-emulation` calls — bf16 inputs and f32 acc); test 53 uses the bf16-in/bf16-out flow (`air-matmul-fuse-output-truncf` + acc-only `air-vector-cast-for-emulation` + `air-hoist-cast-pairs`). Five integration bugs found and fixed during HW bring-up — see "Lessons from M2c". | ✅ |
| **M2d** | Delete `test/xrt/{53,54}/transform_aie2p.mlir` and update plan doc with hardware results. (Currently both flows live behind `--use-cpp-pipeline` so legacy keeps working; deletion is bookkeeping after this milestone is verified stable.) | pending |

**Current end-state (M0 + M1 + M2)**: `check-air-mlir` 381 pass / 7 XFail / 1 pre-existing unrelated failure unchanged. 19 new C++ passes registered (10 from M0/M1 + 9 from M2: cleanup-bufferize, fuse-pingpong-loops, fuse-output-truncf, bufferize-output-l2, bufferize-l1-output, bufferize-l1-inputs, tile-k-and-fuse-packs, tile-cores, prologue-epilogue). Total of 13 transform.air.* op apply()s now thin wrappers over shared helpers (7 from M1 + 6 from M2a). **Hardware validation matrix on NPU2: test 54 cpp PASS, test 54 legacy PASS, test 53 cpp PASS, test 53 legacy PASS, prog_ex i8 PASS, prog_ex bf16 PASS.**

**Cross-phase plumbing decision (re-confirmed for M2)**: each pass identifies its target by attribute marker (`copy_a_loop`, `copy_b_loop`, `k_reduction_loop`, `packed_matmul`, `lhs_pack_in_k`, `rhs_pack_in_k`, `compute_forall`, `matmul_compute`, `init_fill`, `prologue_forall`, `epilogue_forall`, `fused_lhs_l1_pack`, `fused_rhs_l1_pack`). Phase 1 / Phase 4 / Phase 5 / prologue-epilogue write markers; bufferize / fuse-pingpong / vectorize passes consume them. The marker scheme worked cleanly through the entire pipeline integration — no collisions, no missing matches.

**Lessons from M2c integration (apply to M3+)**:
1. **`fuseIndependentSiblingForLoops` is loose about positioning.** It may place the merged loop at the EARLIER of the two loops' positions. Two consequences must be handled:
   - **Dominance for in-between ops.** Allocs/casts that lie strictly between the two loops can end up below the merged loop. Fix: `hoistInterveningDeps` walks BOTH target and source bodies, finds same-block defining ops in the strict interior, and topologically hoists them above the earliest of the two.
   - **Order of unrelated structural ops.** A prologue scf.forall sitting between copy_a and k_reduction is NOT used by either loop, but if the merged loop ends up at copy_a's earlier position, the prologue suddenly sequences AFTER compute — semantically wrong. Fix: BEFORE calling the upstream fuser, `moveBefore(target)` on the source loop so the merged loop is forced to stay at target's position.
2. **Mind the pass-order assumptions baked into M1 passes.** `air-matmul-tile-for-vectorize` filtered by `getParentOfType<HerdOp>()`, requiring forall→herd to run before it. The legacy script does the opposite — tile-for-vectorize first, then forall→herd. Fix: relax the filter to ALSO accept ops carrying the `matmul_compute` / `init_fill` markers (set by M2 tile-cores / prologue-epilogue), so the M2 pipeline can keep the legacy ordering. Document filters like this prominently and prefer marker-based targeting in new passes.
3. **Bufferize ALL linalg.fills, not just the first.** The bf16-out flow (test 53) creates two linalg.fill ops: the original (f32, soon orphaned) and a new one (bf16, feeds the truncf-fused matmul). `air-matmul-bufferize-output-l2` originally bufferized only the first found, leaving the bf16 one in tensor form. After downstream `one-shot-bufferize`, the bf16 init became a fresh L3 alloc that failed the `air.segment` memory-space verifier. Fix: walk for and bufferize EVERY linalg.fill in the function.
4. **Anchor the prologue insertion at the K-reduction loop.** `air-matmul-prologue-epilogue` originally relied on the linalg.fill being textually before the matmul. Bufferization-driven IR reordering between Phase 5 and Phase 6b can flip that. Fix: find the `k_reduction_loop`-marked scf.for and `moveBefore` the fill to immediately above it before generalizing/tiling, so the resulting prologue scf.forall lands above the K loop.
5. **Pipeline-string-based pipelines work fine for the supersession use case.** The initial plan called for a `buildAIRMatmulCodegenPipeline` C++ pipeline-builder. In practice, the run.py-side string version is just as expressive, debuggable (one phase at a time via Python), and maintainable. Keeping the pipeline as a Python string until M3's heuristic config-setter pass arrives.

**Hardware-validation playbook for M2c-style integration (use for M4+):**
The integration is dominated by IR-positioning bugs that lit/equivalence checks DON'T catch. The fastest debug loop turned out to be:
1. Add a per-phase `try/except` + `pm.run` + `open(f"/tmp/{prefix}_post_phase{i:02d}.mlir","w")` wrapper around the pipeline string.
2. After a HW failure, scan the per-phase IRs with `awk` extracting marker/structural positions (`prologue_forall`, `compute_forall`, `k_reduction_loop`).
3. Diff the per-phase IR against the legacy script's `air-opt --pass-pipeline=...` output at the equivalent phase boundary.
4. Side-by-side diff the post-air-copy-to-dma IR (`--print-module-only`) of both pipelines BEFORE running aiecc — peano hangs are downstream symptoms; the structural bug is usually visible at the air-level IR.

### M3a sub-status

**Scope**: hardcoded AIE2 + AIE2P heuristic + each consumer pass reads the dict attribute. Real L1-fit solver and run.py simplification belong to M3b.

| Sub-step | Description | Status |
|---|---|---|
| **M3a-1** | Carrier attribute defined as a `DictionaryAttr` named `air.matmul_codegen_config`. Helper API in [mlir/include/air/Util/MatmulCodegenConfig.h](mlir/include/air/Util/MatmulCodegenConfig.h): `findMatmulCodegenConfig(funcOp)`, `getI64Array`, `getI64`, `getBool`, `writeMatmulCodegenConfig`, `buildMatmulCodegenConfig`. Implementation in [mlir/lib/Util/MatmulCodegenConfig.cpp](mlir/lib/Util/MatmulCodegenConfig.cpp). | ✅ |
| **M3a-2** | `air-matmul-set-codegen-config` (in [AIRMatmulTilePasses.cpp](mlir/lib/Transform/AIRMatmulTilePasses.cpp)) walks for the first linalg.matmul, classifies element types, walks for any truncf-only consumer (detects bf16-via-truncf output even when the matmul itself is f32-acc), then writes the dict. Heuristic produces: pack_sizes (AIE2 [4,8,4] / AIE2P [8,8,8]); per-operand pack-transpose perms (constant `[1,0]`/`[0,1]`); tile_l3_l2_k (preferred 64 for narrow types, 16 for f32, halved until divides K and remains a multiple of packK); tile_k_factor; tile_cores ([8,8,0] for bf16-out path, [8,4,0] for f32-out path on AIE2P, generic fallback otherwise); prologue_tile = tile_cores[0:2]; epilogue_tile derived from coreTile × packSize; vector_tile/unroll/factor/fill_vector_tile (constants matching tests 53/54); plus mode flags. | ✅ |
| **M3a-3** | Six consumer passes wired to `findMatmulCodegenConfig` with fallback to existing pass-options: `air-matmul-tile-l3-to-l2-copies`, `air-matmul-pack-and-transpose`, `air-matmul-tile-k-and-fuse-packs`, `air-matmul-tile-cores`, `air-matmul-prologue-epilogue`, `air-matmul-tile-for-vectorize`. Each reads only the keys it needs; missing keys silently fall back. | ✅ |
| **M3a-4** | `--use-codegen-config` flag added to test 53 and test 54 run.py. When set, prepends the heuristic pass and DROPS hand-tuned per-pass options from the pipeline string (passes use config-attribute values via M3a-3 wiring). Implies `--use-cpp-pipeline`. | ✅ |
| **M3a-5** | HW-validated on NPU2: test 54 M3 PASS (median 5.108 ms vs M2 cpp 5.067 ms — within run-to-run noise); test 53 M3 PASS (median 1.762 ms vs M2 cpp 1.766 ms — within noise). All six existing paths still PASS (legacy 53/54, M2 cpp 53/54, prog_ex i8/bf16). `check-air-mlir` 381 pass / 7 XFail / 1 pre-existing failure unchanged. | ✅ |

**Two integration bugs found and fixed during M3a HW bring-up**:
1. **`linalg::pack` rewrites the matmul into a fresh `linalg.generic`** that does NOT inherit the discardable attrs from the original op. The codegen config attached by set-codegen-config is dropped at `air-matmul-pack-and-transpose`. Fix: snapshot the matmul's discardable attrs before pack, re-attach them to the final packed/transposed op. Same pattern needed in `runFuseTruncfLinalg` (which also creates a fresh op via `linalg.MatmulOp::create`) — `propagateDiscardable` helper added there too.
2. **Heuristic must look through the truncf-only consumer chain** to detect bf16-output-via-truncf. The matmul's own output element type is f32 (acc) when the test feeds a (matmul + truncf) pair; checking `outTy.getElementType()` alone misclassifies test 53 as f32-out and picks the wrong tile_cores. Fix: walk the matmul's users for a truncf-only `linalg.generic` whose output is bf16 — if found, treat as bf16Out for the heuristic's tile/mode-flag selection.

**Known M3a limitations (deferred to M3b)**:
- No L1-fit solver — tile_cores are picked from a hardcoded (in_type, out_type, target) lookup table that matches tests 53/54 by construction. Other matmul shapes hit a generic fallback that may not be optimal.
- Hand-tuned options stay in the run.py pipeline string (just deselected via empty option strings when M3 is on). M3b will drop them entirely once the heuristic is solver-driven.
- `air-matmul-fuse-output-truncf` and `air-hoist-cast-pairs` always run unconditionally in the pipeline (they're idempotent on non-applicable IR). M3b could opt these in/out via the config flags.

### M4a sub-status (in progress)

**Scope**: hand-tune-only port of test 37 (two pack levels, K-peel, 4×4 herd, bf16 in/f32 out). M4b (heuristic) deferred.

| Sub-step | Description | Status |
|---|---|---|
| **M4a-1** | NEW pass `air-matmul-tile-launch-tile` ([AIRMatmulTilePasses.cpp](mlir/lib/Transform/AIRMatmulTilePasses.cpp)). Tiles linalg.matmul with `scf::tileUsingSCF` (LoopType::ForallOp), annotates the new forall with `launch_tile_forall`, then manually fuses the linalg.fill producer of the matmul's accumulator into the forall body via a custom `fuseFillIntoForallSharedOuts` helper (upstream `tileAndFuseProducerOfSlice` doesn't handle the fill→shared_outs case). Smoke-tested: 512×1024×512 matmul tiled by [256, 256] produces correct per-iter fill+matmul on 256x256 slices. | ✅ |
| **M4a-2** | EXTEND `air-matmul-pack-and-transpose`: dropped the strict rank=2 perm validation (let upstream `linalg::packTranspose` enforce well-formedness); pass also walks for `packed_matmul`-marked `linalg.generic` so the second pack level can target an already-packed op. Smoke-tested: L1-pack [0,0,0,8,8,8] on top of L2-pack [64,64,64] produces correct 9-iter linalg.generic with [4×16×8×8×8×8] LHS shape. | ✅ |
| **M4a-3** | EXTEND `air-matmul-bufferize-l1-inputs`: added `memcpy-op` option (`materialize` default, `linalg-copy` for L2 path). The same pass now serves both L1 and L2 input bufferization via `memory-space` + marker + `memcpy-op` options — no separate pass needed. | ✅ |
| **M4a-4** | EXTEND `air-matmul-tile-k-and-fuse-packs`: added `k-iter-index` option so the same pass can be invoked twice (outer K at idx 2, inner K at idx 5 for the 9-iter two-pack matmul). Plus chain-fuse: when the matmul's immediate operand pack has a grandparent pack outside the loop, fuse the grandparent too — annotated with `lhs-l2-pack-in-k-marker` / `rhs-l2-pack-in-k-marker` for the L2 bufferize step. | ✅ |
| **M4a-5** | `air-matmul-tile-cores` already pads `tile-sizes` with zeros via `buildTileSizes`, so it transparently handles the 9-iter packed matmul (`tile-sizes=1,1,0,0,0,0,0,0,0`). No change needed. | ✅ |
| **M4a-6** | NEW pass `air-hoist-static-alloc` (in [AIRMatmulBufferizationPasses.cpp](mlir/lib/Transform/AIRMatmulBufferizationPasses.cpp)). Wraps the `hoistStaticallyBoundAllocationsInFunc<memref::AllocOp>` template helper from AIRLinalgBufferize.cpp via a new exported wrapper `hoistStaticAllocsInFunc`. Required by the K-peel flow so the L1 acc alloc lives outside the K-reduction loop. | ✅ |
| **M4a-7** | `air-matmul-tile-for-vectorize` already accepts longer `matmul-tile-sizes` vectors (uses `ListOption<int64_t>` + `llvm::to_vector` preserves size). The `getNumLoops() < tile.size()` check in the walk allows 9-iter ops with 9-entry tiles. No change needed. | ✅ |
| **M4a-8** | Test 37 cpp pipeline drafted in [run.py](test/xrt/37_matmul_transform_4x4_bf16/run.py) under `--use-cpp-pipeline`. Wires all 7 passes in the right order. Two marker-flow bugs in `tile-k-and-fuse-packs` (chain-fuse) found via per-phase IR diff (`AIR_DUMP_PHASES=…`) against the legacy transform script and fixed: (1) chain-fuse to L2 grandparent missed the producer because after the L1 fuse, `innerPack.getSource()` is `tensor.extract_slice`, not the L2 pack — added a walk-through-extract_slice loop. (2) Inner K-tile left the cores-scope L1 pack marked `fused_lhs_l1_pack` while ALSO marking the new inner-K pack with the same name, so `findMarkedOp` picked the orphan and `canonicalize` then DCE'd the L1 alloc — `fuseChain` now strips the marker from the producer when re-applying it to the fused op. Result: cpp `air_tiled.mlir` allocs match legacy exactly (L1 packs at `memref<…, 2>`, L2 packs at `memref<…, 1>`). **Test 37 cpp PASSes on NPU2 hardware.** | ✅ |
| **M4a-9** | Regression: 390/391 lit tests pass (the 1 failure is a pre-existing `air_transform_payload.mlir` test, last-touched in #1447, unrelated). **Tests 53/54 cpp paths still PASS on NPU2** — no AIR-side regression. | ✅ |

**Architectural note from M4a (RESOLVED)**: The marker-lifecycle fragility predicted in the original M4a-8 attempt turned out to be the actual root cause of two distinct bugs. Both fixes are local to `tile-k-and-fuse-packs::fuseChain`: walk through `tensor.extract_slice` to find chain-fuse grandparents, and strip the L1 marker from the producer before re-applying it to the fused op. The general pattern (clear prior phase marker before re-marking) is the right discipline for any future passes that re-mark fused producers across phases.

### M3b sub-status

**Scope**: drop hand-tuned per-pass options from run.py, add L1-fit guardrail, sweep new shapes. Real derivation-driven heuristic deferred to M3c.

| Sub-step | Description | Status |
|---|---|---|
| **M3b-1** | `--use-cpp-pipeline` now implies M3 (no need to pass `--use-codegen-config` separately). The pipeline string in [test 54 run.py](test/xrt/54_matmul_padding_f32_bf16_emulation/run.py) and [test 53 run.py](test/xrt/53_matmul_padding_bf16/run.py) reduced to a list of pass NAMES with no per-pass option strings — the heuristic drives everything via the `air.matmul_codegen_config` attribute. | ✅ |
| **M3b-2** | Real L1-fit-driven derivation attempted (largest divisor of packedM/packedN ≤ herdM/herdN). Result: produced valid-in-isolation tile sizes that broke downstream codegen (test 53 hit "row index 6 out of bounds" in air-to-aie; test 54 produced wrong values via mis-aligned ACC/UNPACK pattern). The downstream pipeline (`air-collapse-herd`, `air-shrink-memref-sizes-by-access`, etc.) makes implicit assumptions about tile orientation that aren't captured by L1 budget alone. **Reverted to the M3a hardcoded lookup table** but kept the L1-fit calculation as a guardrail: after the lookup picks `(coreTile0, coreTile1)`, halve `coreTile1` then `coreTile0` until the per-core L1 footprint (`LHS + RHS + ACC`) is ≤ 64 KB. The guardrail is a no-op for tests 53/54 (their hand-tuned values fit comfortably) but protects against future shape variations. | ✅ (with deferred M3c) |
| **M3b-3** | Shape-sweep on tests 53/54 with non-default --M/--N/--K args. Results: <br>· test 53 M=128/N=128/K=128 — PASS<br>· test 53 M=500/N=500/K=784 (default) — PASS<br>· **test 53 M=256/N=256/K=512 — FAIL** (also fails under legacy transform script — pre-existing bug, not M3-introduced)<br>· test 54 M=256/N=256/K=512 — PASS<br>· test 54 M=500/N=500/K=784 (default) — PASS<br>· test 54 M=512/N=512/K=512 — PASS<br>5/6 PASS. Heuristic generalizes well across shape variations. | ✅ |

**Two implementation discoveries during M3b**:
1. **`coreTile`-derived epilogue tile mismatched M2 hand-tuned for test 54.** When I switched the epilogue tile formula from herd-based (`M/herdM, N/herdN`) to coreTile-based (`coreTile1×packM, coreTile0×packN`), test 54 broke (wrong values). Fix: use `epM = max(coreTile1×packM, M/herdM)`, `epN = N/herdN`. The `max()` handles the case where the matmul shape forces fewer compute cores than the requested herd (test 53 ends up with 8 compute cores in a 4×2 layout despite herd-m=herd-n=4 being passed).
2. **The downstream `air-collapse-herd` + `air-shrink-memref-sizes-by-access` pipeline tightly couples compute/prologue/epilogue forall shapes.** A "real" L1-fit-only derivation can produce valid-on-paper tile sizes that the downstream codegen mis-handles. M3c will need to model the collapse-herd remap (or constrain the heuristic to produce shapes the downstream pipeline tolerates) before it can replace the lookup table.

---

## 1. Scope

**In-scope inputs (C++ pipeline must cover):**
- [test/xrt/12_matmul_transform_1x4_bf16](test/xrt/12_matmul_transform_1x4_bf16) — single-pack, 1×4 herd, no L1 pack
- [test/xrt/37_matmul_transform_4x4_bf16](test/xrt/37_matmul_transform_4x4_bf16) — two-level pack [64,64,64]→[8,8,8], K-peel
- [test/xrt/53_matmul_padding_bf16](test/xrt/53_matmul_padding_bf16) — bf16-out, truncf-fuse, hoist-cast-pairs, hardware padding
- [test/xrt/54_matmul_padding_f32_bf16_emulation](test/xrt/54_matmul_padding_f32_bf16_emulation) — f32-in/out with BFP16 mmul emulation, hardware padding
- [programming_examples/matrix_multiplication/{bf16,i8,i16}](programming_examples/matrix_multiplication) — vectorize-only flow (matmul herds built via iron API)

**Out of scope:**
- test 55 (iron-built, no linalg.matmul input)
- tests 15, 17, 28, 29 — these are *targets* (already-tiled hand-written IR), not *sources*

---

## 2. Two flows

| Flow | Input IR | Used by | Pipeline coverage |
|---|---|---|---|
| **A. Linalg-input** | `linalg.matmul` over launch-tile-sized `tensor<>` | tests 12, 37, 53, 54 | Full pipeline (Group A + B) |
| **B. Iron-built** | `air.herd` already in place, packed `linalg.generic` inside | prog_ex bf16/i8/i16 | Group B only (vectorize+hoist) |

---

## 3. Padding is orthogonal

Test 53/54's padding does NOT live in the transform script. The transform script consumes a single launch-tile-sized rectangular `linalg.matmul` (`LT_M × LT_N × K_FULL` where `LT_M = HERD_M × TILE_M`). Padding lives in three downstream layers:

1. **Host-side**: allocate to launch-tile multiple, zero-fill beyond `M_actual`/`N_actual`. K is *not* padded (asserted to divide K_L2_TILE).
2. **`air-wrap-func-with-parallel{loop-bounds=…,actual-sizes=…}`** + **`air-par-to-launch{depth=0,has-air-segment=true}`**: wraps the codegen output in an outer launch grid and attaches `air.actual_sizes`.
3. **`air-split-launch-for-padding`** ([AIRSplitLaunchForPadding.cpp](mlir/lib/Transform/AIRSplitLaunchForPadding.cpp), already C++): splits launches at the boundary, rewrites L3↔L2 DMA BDs to read/write only actual rows/columns. L2 buffers always hold a full tile; the padding region's contribution is zero (zero host data).

**Codegen pipeline implication**: padding adds *zero* complexity. The pipeline only needs to verify `K_FULL % K_L2_TILE == 0` and emit a launch-tile-sized vectorized `air.herd`. Everything padding-related is downstream.

---

## 4. Configuration carrier

A new attribute interface, `#air.matmul_codegen_config`, attached to the `linalg.matmul`. Single source of truth; passes read what they need via a level index.

```mlir
#air.matmul_codegen_config<
  // Static launch-tile shape (the linalg.matmul shape itself)
  // M_FULL, N_FULL, K_FULL implicit from the linalg.matmul

  // Tile sizes per level
  // level 0 = L3→L2 copy tile (K_L2_TILE); level 1 = K-tile inside packed compute;
  // level 2 = forall over cores
  tile_sizes = [[0, 0, 16], [0, 0, 2], [8, 4, 0]],

  // Pack sizes (1 entry for tests 12/53/54; 2 entries for test 37)
  pack_sizes = [[8, 8, 8]],

  // Per-operand pack-transpose perms per pack level
  pack_transposes = [{a: {outer=[1,0]}, b: {outer=[1,0], inner=[1,0]}, c: {outer=[1,0]}}],

  // Herd shape
  herd = [4, 4],

  // Vectorization
  vector_tile = [2, 2, 1, 0, 0, 0],
  vector_unroll = [2, 2],

  // Datatypes (redundant with linalg.matmul operand types but cached for fast lookup)
  in_type = f32, acc_type = f32, out_type = f32,

  // Mode flags
  bfp16_mmul_emulation = true,        // test 54: cast inputs→bf16, acc→f32
  bf16_output_hoist_pairs = false,    // tests 53, prog_ex bf16: hoist 4 extf/truncf pairs
  fuse_output_truncf = false,         // test 53: pre-pack truncf→matmul fuse
  three_herd_prologue_epilogue = true,// tests 53/54: yes; test 12: no
  k_peel = false                      // test 37: yes
>
```

---

## 5. Pass list

### Group A: linalg-input → herd (tests 12, 37, 53, 54)

| # | Pass | Replaces (in test 54 transform script) | Upstream / existing C++ called |
|---|---|---|---|
| 1 | `air-matmul-tile-l3-to-l2-copies` | Phase 1 | `linalg::tileUsingSCF` after `convert_memref_copy_to_linalg_copy` (existing C++) |
| 2 | `air-matmul-fuse-output-truncf` (opt-in) | Phase 2 of test 53 | extract from `FuseTruncfLinalg` ([AIRLinalgCodegen.cpp:~4012](mlir/lib/Transform/AIRLinalgCodegen.cpp)) |
| 3 | `air-matmul-bufferize-output-l2` | Phase 2 promotion | `linalg::bufferizeToAllocation` (upstream) |
| 4 | `air-matmul-pack-and-transpose{pack-level=N}` | Phase 3 (and again for test 37 L2 pack) | `linalg::pack` ([Transforms.h:1379](../../llvm-project/mlir/include/mlir/Dialect/Linalg/Transforms/Transforms.h)) + `linalg::packTranspose` |
| 5 | `air-matmul-bufferize-l1-output` | Phase 3 (output_l1_pack bufferize) | `linalg::bufferizeToAllocation` |
| 6 | `air-matmul-tile-k-and-fuse-packs` | Phase 4 | `linalg::tileUsingSCF` + new fusion helper for `linalg.pack` producers |
| 7 | `air-matmul-tile-cores` | Phase 5 | `linalg::tileUsingForall` + reuse `FuseIntoContainingMemrefOp` C++ |
| 8 | `air-matmul-bufferize-l1-inputs` | Phase 6a | `linalg::bufferizeToAllocation` |
| 9 | `air-matmul-prologue-epilogue` (opt-in) | Phase 6 prologue/epilogue | `linalg::generalize` + `linalg::interchange` + `linalg::tileUsingForall` |
| 10 | `air-bufferize-one-shot` | Phase 7 | `bufferization::runOneShotBufferize` (upstream) |
| 11 | `air-matmul-cleanup-bufferize` | Phase 7 tail | reuse `RemoveUninitializedCopy` ([AIRLinalgCodegen.cpp:3034](mlir/lib/Transform/AIRLinalgCodegen.cpp)) + `EliminateCascadeMemcpy` ([AIRLinalgCodegen.cpp:3075](mlir/lib/Transform/AIRLinalgCodegen.cpp)) |
| 12 | `air-matmul-fuse-pingpong-loops` | Phase 8 | upstream SCF sibling fusion + `normalize_for_bounds` extracted from existing C++ |
| (opt) | `air-hoist-static-alloc` | (test 37 K-peel) | reuse [AIRLinalgBufferize.cpp:329](mlir/lib/Transform/AIRLinalgBufferize.cpp) |

### Group B: tile-for-vectorize → vectorize → hoist (tests 12, 37, 53, 54, prog_ex)

| # | Pass | Replaces | C++ called |
|---|---|---|---|
| 13 | `air-matmul-tile-for-vectorize` | Phase 9 | `linalg::tileUsingSCF` + `loop::unroll` |
| 14 | `air-forall-to-herd` *(Group A only)* | Phase 10 first half | reuse `ParToHerdOp::applyToOne` ([ConvertToAIRPass.cpp:2282](mlir/lib/Conversion/ConvertToAIRPass.cpp)) |
| 15 | `air-herd-vectorize` | Phase 10 vectorize | reuse `HerdVectorizeOp` ([AIRHerdVectorize.cpp](mlir/lib/Transform/AIRHerdVectorize.cpp)) |
| 16 | `air-fold-unit-extent-dims` | Phase 10 tail | reuse C++ |
| 17 | `air-eliminate-redundant-vector-transfers` | Phase 10 tail | reuse C++ |
| 18 | `air-vector-cast-for-emulation` (opt-in) | Phase 11 head | reuse `VectorTypeCast` C++. Modes: `acc-only` (53/prog_ex) or `inputs-and-acc` (54 BFP16). |
| 19 | `air-hoist-loop-invariant-transfers` | Phase 11 | reuse [AIRLinalgCodegen.cpp:2721](mlir/lib/Transform/AIRLinalgCodegen.cpp) |
| 20 | `air-flatten-for-iter-args` | Phase 12 | reuse C++ |
| 21 | `air-hoist-vector-transfer-pointers` | Phase 12 | reuse [AIRLinalgCodegen.cpp:4865](mlir/lib/Transform/AIRLinalgCodegen.cpp) |
| 22 | `air-hoist-cast-pairs` (opt-in) | Phase 12 of 53, 4× hand-unrolled in prog_ex | new pass: walks all extf/truncf pairs in innermost loop and calls existing `HoistCastPair` C++ ([AIRLinalgCodegen.cpp:5488](mlir/lib/Transform/AIRLinalgCodegen.cpp)) in a fixed-point loop |

### Cross-phase coupling: attribute markers

Today the transform script uses ~10 named markers (`copy_a_loop`, `copy_b_loop`, `k_reduction_loop`, `packed_matmul`, `compute_forall`, `matmul_compute`, `init_fill`, `prologue_forall`, `epilogue_forall`, `compute_herd`, …). The C++ pipeline keeps the attribute-marker scheme — passes write markers on ops they produce and look for markers on ops they consume. This lets each pass remain individually runnable from `air-opt`.

---

## 6. Heuristic config-setter pass

`air-matmul-set-codegen-config{target=aie2p,bfp16-emulation=true,herd-m=4,herd-n=4}` — runs once at the front and writes the `#air.matmul_codegen_config` attribute:

1. **Inner pack from device model**: `air::AIEDeviceModel(target).getMatmulInstructionSize(lhsTy, rhsTy, accTy)` → `[m1Pack, n1Pack, k1Pack]`.
   - AIE2 bf16/f32 → `[4, 8, 4]`
   - AIE2P bf16/f32 → `[8, 8, 8]`
   - AIE2P i8/i32 → `[8, 8, 8]` *(verify against device model)*
   - AIE2P f32/f32 with BFP16 emulation → `[8, 8, 8]` (bf16-equivalent, since emulation casts inputs in-register)
   - No-vector fallback → `findLargestFactor(M,4)`, etc.
2. **L1 fit solver**: `selectL1TileSizes` with `bufferDepth=1` for all (mlir-air does L2 ping-pong, not L1 — per CLAUDE.md note). Returns `[M1, N1, K1]`.
3. **L2 from array shape**: `M0 = numRows × M1` capped at L2 fit, then `findLargestFactor(M, maxL0SizeM, M1)`. Same for N0.
4. **K_L2_TILE**: `K1 × scale` where scale defaults to 2 (matches test 54's K_L2_TILE=16, k1Pack=8). Verify `K_FULL % K_L2_TILE == 0`.
5. **Mode flag derivation from element types**:
   - `out_type==bf16 && acc_type==f32` → `fuse_output_truncf=true`, `bf16_output_hoist_pairs=true`
   - `target==aie2p && bfp16_emulation && in_type==f32` → `bfp16_mmul_emulation=true` with cast (inputs→bf16, acc→f32)
   - `target==aie2p && bfp16_emulation && in_type==bf16` → `bfp16_mmul_emulation=true` with cast (acc-only→f32)
6. **Elementwise-consumer detection** (future): set `bufferDepthAcc=1` if matmul has elementwise consumer; `bufferDepthAcc=0` otherwise (accumulate in registers).

User overrides (pass options or attribute pre-attached) skip the corresponding heuristic step.

---

## 7. Pipeline-builder

```cpp
void buildAIRMatmulCodegenPipeline(OpPassManager &pm,
                                    const AIRMatmulCodegenOptions &opts);
```

Branches:
- `opts.flow == iron_built` → skip passes 1–12, run only Group B.
- `opts.num_pack_levels == 2` → insert second `air-matmul-pack-and-transpose{pack-level=1}` + bufferize before `air-matmul-tile-k-and-fuse-packs`.
- `opts.three_herds` → enable pass 9.
- `opts.bfp16_emulation` → enable pass 18.
- `opts.bf16_output` → enable passes 2 and 22.
- `opts.k_peel` → enable `air-hoist-static-alloc`.

Most options come from the `#air.matmul_codegen_config` attribute, not pass options — `buildAIRMatmulCodegenPipeline` reads it from the linalg op once and configures the inner pass list.

---

## 8. Surrounding pipeline context

```
[Triton-XDNA frontend / asm_src / handwritten kernel]
        ↓ produces: func with one launch-tile-sized linalg.matmul
[NEW: air-matmul-set-codegen-config{target=aie2p,…}]
        ↓ writes #air.matmul_codegen_config attribute
[NEW: air-matmul-codegen-pipeline]   ← THIS DOC'S SCOPE (passes 1–22)
        ↓ produces: vectorized func with air.herd inside
[existing: air-wrap-func-with-parallel{loop-bounds=…,actual-sizes=…}]
[existing: air-par-to-launch]
[existing: air-copy-to-dma]
[existing: air-split-launch-for-padding]   ← handles padding via memtile DMA BDs
[existing: rest of aircc → AIE → ELF]
```

---

## 9. Test plan

Three layers, in order of cost/confidence:

- **Lit FileCheck per pass** (cheap, every CI): `mlir/test/Transform/MatmulCodegen/<pass>.mlir`. Small synthetic input → expected output. Driven by `air-opt --air-matmul-<pass>`. Lit tests landed for `air-matmul-pack-and-transpose`, `air-matmul-tile-l3-to-l2-copies`, `air-fold-unit-extent-dims`, `air-eliminate-redundant-vector-transfers`, `air-flatten-for-iter-args` (M0/M1a/M1b).
- **IR equivalence vs the legacy transform script** (medium, no hardware): run the same input IR through (a) the new C++ passes and (b) the corresponding fragment of the legacy transform script. Diff after `-canonicalize -cse`. Goal: byte-identical or canonically equivalent. M0 used this to validate against transform-script Phases 1+3 byte-identically.
- **End-to-end on NPU2 hardware** (proves real correctness): drive a programming-example or test-xrt entry through `--compile-mode=compile-and-run --arch=aie2p`. Validates that the IR is not just *equivalent* but downstream-acceptable (passes aiecc legalization, fits L1, runs on Strix). M1 used this on prog_ex i8 + bf16 — both PASS. **See Appendix A for the env-var setup needed.**

The IR-equivalence layer is fast and cheap, but it can be misleading: my M1 IR was *similar* to legacy at first inspection, yet the hardware run revealed two real bugs (outermost-vs-innermost target, missing compute-herd filter) that lit and equivalence checks missed. **Hardware validation on NPU2 is the only ground truth — schedule it before claiming a milestone done.**

---

## 10. Sequencing (milestones)

| Milestone | Scope | Outcome |
|---|---|---|
| **M0** ✅ | Passes 4 (`pack-and-transpose`) and 1 (`tile-l3-to-l2-copies`) only, with hand-attached config attribute | Landed. Lit tests + IR-equivalence vs transform-script Phases 1+3 byte-identical. |
| **M1** ✅ | Group B (passes 13–22) | Landed. prog_ex matrix_multiplication/{bf16,i8} swapped to `--pass-pipeline=...` invocation; **hardware-validated end-to-end on NPU2**. |
| **M2** ✅ | Group A + B for tests 53, 54 (single pack level; test 12 deferred — non-canonical pad+kernel.cpp flow) | Landed and **hardware-validated on NPU2**. Both tests pass via `--use-cpp-pipeline` in run.py. Five integration bugs found and fixed (see "Lessons from M2c"). Both legacy paths still pass. |
| **M3** | `air-matmul-set-codegen-config` heuristic | Users no longer pass tile sizes in run.py. Verify equivalence with M2's hand-set parameters. |
| **M4** | Two pack levels (test 37) | Add `pack-level=0,1` to pack pass. **Delete `37/transform_aie2*.mlir`.** |
| **M5** | Triton-XDNA backend integration | Triton-XDNA points its mlir-air backend at the C++ pipeline instead of generating transform scripts. Ultimate goal — no Triton-side transform-script generation. |

**Skipped**: test 55 (iron-built padding) — outside the linalg-input domain. Revisit only if we want to converge to a single matmul flow.

### Lessons from M1 (apply to M2+)

1. **Helper functions extracted from `transform.air.*` apply()s usually filter by `getParentOfType<scf::ForOp>() == currentLoop`.** That filter only matches when the pass targets the *innermost* loop where transfers/ops live, *not* the outermost in-herd. The legacy transform scripts target the outermost via `match + split_handle{overflow_result=1}`, which works "by luck" because the script is run on a specific structurally-known IR; in a generic pass, walk for the innermost loop directly.
2. **Walk for compute-only herds.** The matmul pipeline almost always has 1 fill herd + 1 compute herd + 1 epilogue herd. Passes that materially reshape vector ops or memref accesses (e.g., collapse_shape) must skip non-compute herds, otherwise downstream `air-shrink-memref-sizes-by-access` loses the per-core access pattern and L1 buffers won't split. Use `herdHasVectorContract(herd)` as the discriminator (mirrors the script's `%herd2` targeting).
3. **Lit FileCheck and IR-equivalence diffs missed both bugs above.** The IR was structurally *similar* to legacy but the L1 buffer allocation collapsed because of a single defective access pattern. **Run NPU2 hardware validation on every milestone** — it's the only test that catches `air-shrink-memref-sizes-by-access` failures and aiecc legalization issues.

---

## 11. Files to read in detail before implementation

- [AIRLinalgCodegen.cpp:1308](mlir/lib/Transform/AIRLinalgCodegen.cpp) — `AIRLinalgCodegen` pass (existing tile/promote infrastructure to mine)
- [AIRLinalgCodegen.cpp:2721](mlir/lib/Transform/AIRLinalgCodegen.cpp) — `HoistLoopInvariantTransfersOp::apply` (extract free function)
- [AIRLinalgCodegen.cpp:4012](mlir/lib/Transform/AIRLinalgCodegen.cpp) — `FuseTruncfLinalgOp` (extract)
- [AIRLinalgCodegen.cpp:5488](mlir/lib/Transform/AIRLinalgCodegen.cpp) — `HoistCastPairOp` (extract + wrap in fixed-point pass)
- [ConvertToAIRPass.cpp:2282](mlir/lib/Conversion/ConvertToAIRPass.cpp) — `ParToHerdOp` (extract)
- [AIRSplitLaunchForPadding.cpp](mlir/lib/Transform/AIRSplitLaunchForPadding.cpp) — already C++; understand the boundary it expects from the codegen pipeline

---

## 12. Open questions

1. **Where does the config attribute come from in M0–M2?** Pass options + JSON for parity with current scripts. Heuristic lands in M3.
2. **Coexistence with `transform.air.*` ops?** Yes — they share C++ implementations. The new passes are an additional entry point; existing transform-based tests keep working until their per-test scripts are deleted in M2/M4.
3. **`bufferDepthAcc=0` vs `1`** for the L1 accumulator: today mlir-air uses register-only accumulation for pure matmul. The heuristic should detect elementwise consumers (e.g., bias add) and switch to `bufferDepthAcc=1`. Out of scope for M0–M3, on by M4.
4. **`runHoistVectorTransferPointers` latent bug**: the helper produces an invalid `memref.collapse_shape` if called on an scf.for whose body has vector.transfer_read ops on subview-derived strided memrefs. M1 dodged this by filtering to compute herds only (where transfers are on full L1 allocs, not subviews). M2's linalg-input flow may exercise the bug; revisit the helper when first triggered.

---

## Appendix A — Hardware bench environment (NPU2 / Strix)

Reproducing M1's hardware validation (or running any prog_ex / test/xrt with `--compile-mode=compile-and-run`) requires:

```bash
# XRT runtime (pyxrt + xrt-smi) — installed at /opt/xilinx/xrt:
export PATH=/opt/xilinx/xrt/bin:$PATH               # for xrt-smi (target-device auto-detect)
export PYTHONPATH=/opt/xilinx/xrt/python:$PYTHONPATH # for pyxrt (NPU device load + execute)
export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:$LD_LIBRARY_PATH

# Peano (llvm-aie) for direct codegen:
export PEANO_INSTALL_DIR=/home/strixminipc/.local/lib/python3.13/site-packages/llvm-aie

# mlir-air + mlir-aie + LLVM:
export PYTHONPATH=/home/strixminipc/new_session_2/mlir-air/install/python:/home/strixminipc/new_session_2/mlir-air/mlir-aie/install/python:$PYTHONPATH
export PATH=/home/strixminipc/new_session_2/mlir-air/install/bin:/home/strixminipc/new_session_2/mlir-air/mlir-aie/install/bin:/home/strixminipc/new_session_2/mlir-air/my_install/mlir/bin:$PATH
export LD_LIBRARY_PATH=/home/strixminipc/new_session_2/mlir-air/install/lib:/home/strixminipc/new_session_2/mlir-air/mlir-aie/install/lib:$LD_LIBRARY_PATH
```

`xrt-smi examine` must be reachable via `PATH` for `XRTBackend.compile()` to auto-detect Strix as `npu2`. `pyxrt` must be importable for `XRTBackend.load()` to push the xclbin to the device. Without `xrt-smi`, the target falls back to `npu1` and the xclbin is not generated.

NPU2 hardware verified during M1: AMD Ryzen AI 9 HX 370 (Strix), XRT 2.23.0, NPU firmware 1.1.2.64.

To reproduce M1 hardware validation:
```bash
cd programming_examples/matrix_multiplication/i8
rm -rf air_project   # caching can mask aiecc failures from prior runs
python3 run.py --direct-codegen --compile-mode=compile-and-run --arch=aie2p
# expected: PASS!  (exit=0)

cd ../bf16
rm -rf air_project
python3 run.py --direct-codegen --compile-mode=compile-and-run --arch=aie2p
# expected: PASS!  (exit=0)
```
