# Root-cause walk through mlir-aie + llvm-aie sources

For each blocker captured in `blockers/`, this document points at the
specific source location in the local mlir-aie / llvm-aie clones that
explains the failure mode, and proposes the smallest upstream fix that
would unblock inlining.

## 1. `aievec.band` has no AIEVecToLLVM lowering — sinf/cosf blocker A

**Test:** `blocker_sin_mask.py --step andi`
**Failure:** `failed to legalize operation 'aievec.band' that was explicitly marked illegal`

### Where it goes wrong

`arith.andi` on `vector<16xi32>` (which is 512-bit, the AIE2P native
register width) gets converted to `aievec.band` by mlir-aie's
`ComputeBandAndBorOpPattern` — see
[mlir-aie/lib/Dialect/AIEVec/Transforms/VectorToAIEVecConversions.cpp:3957](/home/strixminipc/mlir-aie/lib/Dialect/AIEVec/Transforms/VectorToAIEVecConversions.cpp#L3957):

```cpp
template <typename SrcOpTy, typename DstOpTy>
struct ComputeBandAndBorOpPattern : OpConversionPattern<SrcOpTy> {
  ...
  if (laneSize * elWidth != 512) return failure();
  rewriter.replaceOpWithNewOp<DstOpTy>(srcOp, ..., adaptor.getLhs(), adaptor.getRhs());
};
using ComputeBandOpPattern =
    ComputeBandAndBorOpPattern<arith::AndIOp, aievec::BandOp>;
```

Then, in the AIEVecToLLVM pass at
[mlir-aie/lib/Conversion/AIEVecToLLVM/AIEVecToLLVM.cpp:5698](/home/strixminipc/mlir-aie/lib/Conversion/AIEVecToLLVM/AIEVecToLLVM.cpp#L5698):

```cpp
target.addIllegalDialect<xilinx::aievec::AIEVecDialect, ...>();
```

every `aievec` op is marked illegal — but the AIE2P pattern list at
`AIEVecToLLVM.cpp:5631-5650` has **no** `BandOpAIE2pConversion` (or
the AIE2 equivalent in the common pattern list at
`AIEVecToLLVM.cpp:5509-5550`). Only the chess C++ emitter
[TranslateAIEVecToCpp.cpp:1763](/home/strixminipc/mlir-aie/lib/Targets/AIEVecToCpp/TranslateAIEVecToCpp.cpp#L1763)
handles `aievec.band`.

### Root cause

The conversion `arith.andi → aievec.band` is **redundant** when the
target is Peano. Peano's AIE2P legalizer already lists `G_AND` as legal
on every AIE2P vector type at
[llvm-aie/llvm/lib/Target/AIE/aie2p/AIE2PLegalizerInfo.cpp:272](/home/strixminipc/llvm-aie/llvm/lib/Target/AIE/aie2p/AIE2PLegalizerInfo.cpp#L272):

```cpp
getActionDefinitionsBuilder({G_AND, G_OR})
    .legalFor({S32, V2S16, V4S8})
    .legalFor(AIE2PVectorTypes)   // includes V16S32, V32S16, V64S8 ...
    ...
```

So `arith.andi` on `vector<16xi32>` would lower cleanly via the standard
ArithToLLVM path → `llvm.and` → AIE2P `G_AND` → native instruction. The
`aievec.band` detour was added for the chess C++ emitter and now blocks
the Peano path.

### Smallest fix

Either of these would unblock Peano:

- **(A)** Skip `ComputeBandAndBorOpPattern` (and its sibling Bor/Bxor/Bneg
  patterns) when the target backend is Peano. Add a `TargetBackend`
  guard to the conversion. Surgical, low risk.
- **(B)** Add trivial `aievec.band` → `llvm.and` (and Bor → llvm.or, Bxor →
  llvm.xor) lowering to `AIEVecToLLVM.cpp` so both backends work. Slightly
  more code but symmetric with the chess emitter.

Same fix unblocks `arith.ori`, `arith.xori`, etc. on AIE2P vectors.

## 2. `G_SITOFP <16 x s32>` → `<16 x s16>` not legalized — sinf/cosf blocker B

**Test:** `blocker_sin_mask.py --step fptosi`
**Failure:** `LLVM ERROR: unable to legalize instruction: %16:_(<16 x s16>) = G_SITOFP %15:_(<16 x s32>)`

### Where it goes wrong

The AIE2P GISel legalizer at
[llvm-aie/llvm/lib/Target/AIE/aie2p/AIE2PLegalizerInfo.cpp:218](/home/strixminipc/llvm-aie/llvm/lib/Target/AIE/aie2p/AIE2PLegalizerInfo.cpp#L218):

```cpp
getActionDefinitionsBuilder({G_SITOFP, G_UITOFP})
    .libcallForCartesianProduct({S32, S64})
    .clampScalar(1, S32, S64)
    .widenScalarToNextPow2(1)
    .clampScalar(0, S32, S64);
```

is **scalar-only** — every action (`libcallForCartesianProduct`,
`clampScalar`) only matches scalar source/dest. There is no rule (legal,
custom, or scalarize) for any vector form of SITOFP/UITOFP. AIE2 (Phoenix)
at [AIE2LegalizerInfo.cpp:187](/home/strixminipc/llvm-aie/llvm/lib/Target/AIE/AIE2LegalizerInfo.cpp#L187)
has the same scalar-only pattern.

The corresponding intrinsics at
[llvm-aie/llvm/include/llvm/IR/IntrinsicsAIE2P.td:612-614](/home/strixminipc/llvm-aie/llvm/include/llvm/IR/IntrinsicsAIE2P.td#L612):

```td
class AIE2PFX2FLT : DefaultAttrsIntrinsic<[llvm_float_ty], [llvm_i32_ty, llvm_i32_ty], ...>;
class AIE2PFLT2FX : DefaultAttrsIntrinsic<[llvm_i32_ty], [llvm_float_ty, llvm_i32_ty], ...>;
def int_aie2p_fx2flt : ...AIE2PFX2FLT;
def int_aie2p_flt2fx : ...AIE2PFLT2FX;
```

are also scalar-only — `llvm_float_ty` / `llvm_i32_ty`, no vector form.

### Root cause

AIE2P **hardware itself** does not appear to expose a vector int↔fp
conversion instruction (or at least, llvm-aie does not surface one).
Both vector `G_FPTOSI <16 x s16> → <16 x s16>` (bf16 → i16, our first
attempt) and `G_SITOFP <16 x s32> → <16 x s16>` (i32 → bf16, the
workaround) hit the same wall.

### Smallest fix

Two paths:

- **(A) Scalarize at the legalizer** — add `.scalarize(0)` to the SITOFP/
  FPTOSI rule for vector forms. Result: per-lane scalar fx2flt + insert.
  Slow (16 scalar conversions per vector op) but correct. This is the
  conservative fallback.
- **(B) Add a vector conversion expansion** — combine `arith.extf bf16→f32`
  → `aievec.ups` (already legal), then per-lane fx2flt, then build a
  vector via `aievec.broadcast_scalar` + insert. More work, still
  serializes the conversion.

Both are slow enough that the cleaner answer for sin/cos is a
**different polynomial formulation** that doesn't need integer masks at
all — for example, computing the quadrant sign via `math.floor` +
`arith.cmpf` on bf16/f32 vectors (CmpFOp + SelOp are both supported per
the conversion patterns at
[VectorToAIEVecConversions.cpp:2243-2280](/home/strixminipc/mlir-aie/lib/Dialect/AIEVec/Transforms/VectorToAIEVecConversions.cpp#L2243)).

## 3. `shuffle_apply_rope` — contextual failure, not a clean bug

**Tests:** `blocker_rope_v2.py`, `blocker_rope_v3.py` — both PASS the
exact `subf(mulf, mulf) + fma(_, _, mulf)` arithmetic in isolation.

### What I looked at

The `air-dependency` pass at
[mlir-air/mlir/lib/Transform/AIRDependency.cpp:136](/home/strixminipc/mlir-air/mlir/lib/Transform/AIRDependency.cpp#L136)
treats `func.call` as a candidate for `createAsyncExecute` (wrapping it
in an `air.execute` region with full memref read/write tracking). For
the inline-MLIR replacement, the individual `vector.transfer_write` /
`vector.transfer_read` ops fall into the "unknown op" branch at
`AIRDependency.cpp:164-202`, which still wraps them in `air.execute`
when they have memref operands. Subview unwrap correctly traces back to
the parent memref via `traceDeps` at
[AIRDependency.cpp:1029-1048](/home/strixminipc/mlir-air/mlir/lib/Transform/AIRDependency.cpp#L1029).

Read/write classification uses
[mlir-air/mlir/lib/Util/Util.cpp:838](/home/strixminipc/mlir-air/mlir/lib/Util/Util.cpp#L838):

```cpp
if (mlir::hasEffect<mlir::MemoryEffects::Write>(owner, op_operand.get()))
    return 'w';
if (mlir::hasEffect<mlir::MemoryEffects::Read>(owner, op_operand.get()))
    return 'r';
```

`vector.transfer_read`/`vector.transfer_write` declare these effects
correctly upstream, so the per-subview reads/writes ARE classified. The
dependency machinery is not obviously broken for this pattern.

### Where it likely actually breaks

I could not isolate it to a single source-level cause. Hypotheses left
to test (in priority order):

1. **`memref::CollapseShapeOp`** is excluded from `air.execute` wrapping
   ([AIRDependency.cpp:178-181](/home/strixminipc/mlir-air/mlir/lib/Transform/AIRDependency.cpp#L178)).
   `c_data` is collapsed for `copy_scale_q` at
   `attn_decode_npu2.py:929`. If the inline rope writes are tracked via
   the original 2D shape but the downstream collapse-shape read is
   tracked via the 1D flat shape, the dep tracker may not match the two
   accesses to the same root memref.
2. **NKV=8 herd cores running in parallel** — the standalone tests use
   `sizes=[1, 1]`. With 8 cores all writing/reading a per-tile L1 buffer,
   any subtle race in the inline rope's write-then-read pattern would
   manifest only with 8 tiles.
3. **Code volume / regalloc** — the surrounding kernel (vecmat + rope +
   softmax + attn_1 + attn_2) is much larger than the standalone test.
   The inline rope's mid-kernel insertion may cause regalloc spills
   that interact badly with the surrounding ops.

### Next debug step

Incrementally extend `blocker_rope_v3.py` until it starts failing:

- Add `memref.collapse_shape` of the state buffer between the rope
  iterations and the L3 writeback.
- Bump `sizes=[1, 1]` to `sizes=[NKV, 1]` with broadcast inputs (or
  per-tile staging).
- Add dummy ops (vector.fma chains on a scratch buffer) before/after the
  rope to bloat the body.

Once a minimum-extension repro fails, the diff between the passing and
failing variants pinpoints the trigger.

## 4. `attn_1_group` perf — both forms compile to identical .text

**Test:** `blocker_attn1_perf.py --variant unroll4` vs `--variant naive`
**Result:** both PASS, both produce identical 1136-byte `.text`.

### What I looked at

`mlir-air` only attaches one loop annotation when lowering scf.for to
LLVM:
[AIRToAIEPass.cpp:1771-1789](/home/strixminipc/mlir-air/mlir/lib/Conversion/AIRToAIEPass.cpp#L1771):

```cpp
auto mustProgressAttr = LLVM::LoopAnnotationAttr::get(
    ctx,
    /*disableNonforced=*/nullptr,
    ...
    /*pipeline=*/nullptr,
    ...
    /*mustProgress=*/rewriter.getBoolAttr(true),
    ...
);
fop->setAttr("loop_annotation", mustProgressAttr);
```

No `pipeline=true`, no `pipelineII=N`, no `unroll.enable/full/count`,
no `itercount.range`. So inline-MLIR loops give Peano no software-
pipelining hints.

Peano's pipeliner reads these standard LLVM loop metadata keys at
[llvm-aie/llvm/lib/Target/AIE/Utils/AIELoopUtils.cpp](/home/strixminipc/llvm-aie/llvm/lib/Target/AIE/Utils/AIELoopUtils.cpp):

- `llvm.loop.itercount.range` (min trip count for II planning)
- `llvm.loop.pipeline.initiationinterval` (explicit II)
- `llvm.loop.pipeline.disable`
- `llvm.loop.unroll.full/.enable/.count`

There is also a global override `aie-loop-min-tripcount` CL option at
[llvm-aie/llvm/lib/Target/AIE/Utils/AIELoopUtils.cpp:21-26](/home/strixminipc/llvm-aie/llvm/lib/Target/AIE/Utils/AIELoopUtils.cpp#L21).

### Why "unroll4" and "naive" came out identical

Both inline forms are written with **Python `for g in range(GROUP_SIZE)`**
loops, which emit unrolled MLIR (the loop is unrolled at IR-build
time, not at compile time). So both forms hand Peano the same fully-
unrolled sequence of 8 vector.fma ops, and Peano produces the same
schedule for both. The only effect of `unroll4` vs `naive` would be the
order of fma ops in the basic block — and Peano's scheduler is free to
reorder regardless.

### Implication for inline attn_1

The MLIR primitives for attn_1 are all available. The expected
performance is gated by Peano's basic-block scheduler doing the same
job on the inline sequence as the C++ extern. Both compile to identical
code size, but `.text size == cycles` only as a rough proxy. Need a
**timed harness** (e.g., wire `blocker_attn1_perf.py` into a
`profile-decode` style measurement) to A/B the per-token cycles.

If the inline form lags, the most likely missing knob is a loop-level
`pipeline = true` annotation; mlir-air would need to emit it through
the `LoopAnnotationAttr` builder above, optionally driven by an
attribute on the source scf.for.

## Summary table

| Blocker | Root cause | Fix surface |
|---|---|---|
| `aievec.band` illegal | mlir-aie converts `arith.andi` → `aievec.band` for chess, but AIEVecToLLVM has no `aievec.band` pattern. AIE2P backend has `G_AND` legal on all vector types. | mlir-aie 1-line guard (skip pattern for Peano) OR ~10-line `aievec.band` → `llvm.and` lowering |
| Vector `G_SITOFP` not legalized | llvm-aie AIE2P legalizer's `G_SITOFP/G_UITOFP/G_FPTOSI/G_FPTOUI` rules are scalar-only. AIE2P has no vector int↔fp intrinsic. | Add `.scalarize(0)` for vector forms in `AIE2PLegalizerInfo.cpp:218`. Slow but correct fallback. |
| rope subf+fma in attn_decode | Not reproducible in isolation. Likely contextual: collapse_shape, NKV=8 cores, or code-size pressure. | Continue narrowing in `blocker_rope_v3.py` as described above. |
| attn_1 perf | mlir-air emits no `pipeline=true` loop annotation. Both inline and naive forms compile to identical code (Python unroll already unrolled). | Wire test into timed harness; if lagging, add LoopAnnotationAttr with `pipeline=true` and `itercount.range`. |
