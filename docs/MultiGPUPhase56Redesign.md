# Multi-GPU Phase 5+6 Redesign Plan

**Status**: Infrastructure landed (custom attribute + verifier + phase 4 pass migration). Baseline e2e blocked on air-to-rocdl bugs.

**Date**: 2026-05-13 (initial), 2026-05-14 (infra landed)

**Related PRs** (in flight, will need to be reworked or closed):
- #1580 — Phase 5: `air-cross-rank-dma-to-mgpu`
- #1581 — Phase 6: `air-gpu-channel-to-mgpu`

**Landed prerequisites**:
- #1577 (Phase 2: handwritten cacheline reference)
- #1578 (Phase 3: `air-rank-to-mgpu`)
- #1579 (Phase 4: `air-symmetric-alloc-to-mgpu`)
- #1611 (handwritten allgather)
- #1613 (test directory restructure)

---

## TL;DR

The current phases 5 and 6 lower their ops to **host-side `mgpuMemcpy(peer_va)`** —
which is the *wrong target*. The handwritten/cacheline reference uses
**kernel-driven cross-rank stores via `air.translate` inside `gpu.func`**, so the
higher-level abstractions should lower to **the same kernel-driven shape**, not
to a different (host-driven) mechanism.

**Plan:**

1. **Drop phase 5** (`air-cross-rank-dma-to-mgpu`) — wrong abstraction.
   `air.dma_memcpy_nd` is a single op describing one transfer; it doesn't
   naturally express the producer/consumer asymmetry that the cacheline
   pattern needs.
2. **Redesign phase 6** (`air-gpu-channel-to-mgpu` → `air-gpu-channel-to-cacheline`).
   Pass operates on `air.channel.put`/`.get` ops **inside `air.herd` bodies**
   and expands them into cache-line atomicity primitives (`air.translate`
   + `memref.store` for puts; `scf.while` + `gpu.shuffle` + `memref.load`
   for gets). Channel symbol erased after expansion.
3. The **herd boundary stays intact** — existing `air-gpu-outlining` +
   `air-to-rocdl` handle herd → `gpu.func` conversion as a separate step.

---

## Why the current design is wrong

### What handwritten/cacheline.mlir does (the lowering target)

Two `gpu.func`s:
- `@producer`: 1 wave × 64 lanes; lanes 0..30 store payload + lane 31 stores
  flag = 1, all to **peer's slot via `air.translate`** (cross-XGMI). One vec
  store = one cache-line transaction = atomic publish.
- `@consumer`: 1 wave × 64 lanes; lanes 0..31 do `memref.load %data[lane]`,
  then `gpu.shuffle idx %v, 31, 64` broadcasts lane 31's value (the flag).
  Spin-loop until flag observed.

Cross-rank synchronization is **in-band in the cache line** — no host
roundtrips, no mgpuBarrier, just the AMDGPU coherence protocol publishing a
128-byte line atomically.

### What the current phase 5/6 lower to (wrong)

Both passes generate **host-side `mgpuMemcpy(peer_va)`** with
`mgpuBarrier()` for sync. This is:
- Strictly weaker than the kernel-driven mechanism (host roundtrips, no
  cache-line atomicity).
- Different from what the handwritten reference does — so the higher-level
  tests don't demonstrate equivalence to the reference.
- A redundant code path — there's no reason to lower to host-driven DMA
  when the cache-line kernel exists and is more efficient.

### The fix: lower to AIR-level kernel primitives, not runtime calls

| | Current (host-driven, wrong) | Redesigned (kernel-driven, right) |
|---|---|---|
| put → | `mgpuBarrier()` | `air.translate` + cooperative cache-line `memref.store` (in-band flag at lane 31) |
| get → | `mgpuBarrier()` + `mgpuMemcpy(dst, peer_va, size)` | `scf.while` spin loop with `gpu.shuffle idx %v, %c31_i32, %c64_i32` broadcasting lane 31; `memref.load` after flag observed |
| Sync mechanism | host barrier + memcpy | cache-line atomicity (no host roundtrip) |
| Output dialect | LLVM/runtime calls | AIR + memref + gpu.shuffle (still in AIR/MLIR-level dialect) |
| Pipeline composition | terminal | feeds into existing `air-gpu-outlining` + `air-to-rocdl` chain |

---

## What is `air.herd` and why it matters

Per [`docs/AIRComputeModel.md`](AIRComputeModel.md) §2.3 and the example
[`test/gpu/4k_4k_mul/air_sync.mlir`](../test/gpu/4k_4k_mul/air_sync.mlir):

```
air.launch          ← gpu.launch gridDim (block grid)
  air.segment       ← block-level scope (shared memory, gpu.barrier)
    air.herd        ← per-warp / gpu.func body
```

> *"GPU (AMD MI3xx family): A herd executes entirely within a single Compute
> Unit (CU), with PE instances mapped to individual warps."*

So **`air.herd` is the gpu.func boundary**. Its body becomes the kernel
after `air-gpu-outlining`. Channel ops inside the herd body lower to
intra-kernel data movement.

---

## Why `air.channel` (not `air.dma_memcpy_nd`)

| Property | `air.dma_memcpy_nd {src_rank/dst_rank}` | `air.channel` + put/get |
|---|---|---|
| Producer/consumer decoupling | conflated in one op | put + get at distinct sites, joined by symbol |
| Asymmetric kernel structure (handwritten/cacheline has @producer + @consumer) | poor fit — pass would have to invent the split | direct fit — put → producer-side herd body, get → consumer-side herd body |
| `channel_type` hook for the symmetric-heap variant | doesn't exist | already there (`channel_type = "gpu_symmetric_heap"`) |
| Topology in the channel symbol | no | yes — bundle dimensions in `[N]` syntax |

Cacheline lowering is fundamentally a **two-endpoint producer/consumer
pattern with explicit sync**. The handwritten reference has *two distinct
kernels*. Lowering should preserve that two-endpoint structure — which
`air.channel` does naturally and `air.dma_memcpy_nd` does not.

---

## Channel-as-topology design

Per the user's decision: the channel symbol **carries topology and
connectivity**. The bundle dimensions in `air.channel @C [N]` encode the
number of parallel "wires" available to connect source ranks to destination
ranks. Each rank's kernel `put`s or `get`s on the corresponding wire index.

### Cacheline (rank 0 → rank 1, 1 wire)

```mlir
air.channel @C [1] {channel_type = "gpu_symmetric_heap"}
...
scf.if %is_producer {                    // rank 0
  air.launch ... { air.segment ... { air.herd ... {
    air.channel.put @C[%c0] (%hd[][][]) : (memref<32xi32>)
  } } }
} else {                                  // rank 1
  air.launch ... { air.segment ... { air.herd ... {
    air.channel.get @C[%c0] (%hd[][][]) : (memref<32xi32>)
  } } }
}
```

Wire 0: producer rank 0 puts → consumer rank 1 gets. Pass synthesizes the
cache-line expansion inside each herd body.

### Allgather (W ranks, every rank publishes its slice)

```mlir
air.channel @C [%world] {channel_type = "gpu_symmetric_heap"}  // W wires
...
scf.for %wire = %c0 to %world step %c1 {
  air.launch ... { air.segment ... { air.herd ... {
    %is_my_wire = arith.cmpi eq, %wire, %rid : index
    scf.if %is_my_wire {
      air.channel.put @C[%wire] (%my_slice[][][]) : (memref<32xi32>)
    } else {
      air.channel.get @C[%wire] (%output_slot[%wire * 32 ..][][]) : (memref<32xi32>)
    }
  } } }
}
```

Pass walks each wire, pairs the put/get, infers source/dest ranks from
rank-dispatch context.

**Initial scope**: only the 1-to-1 cacheline case. All-pairs / many-to-many
deferred — the channel-bundle design supports it but the pass implementation
gets analysis-heavy.

---

## Lowering pipeline (composes with existing AIR-on-GPU passes)

```
1. air-gpu-channel-to-cacheline      [NEW phase 6 redesigned]
   - Walks air.channel @C declarations of gpu_symmetric_heap type
   - For each, finds the (put, get) pair sharing the same wire index
   - Identifies producer/consumer ranks from rank-dispatch context
   - Inside put's herd body: replaces with cooperative cache-line write
     to peer's slot (via air.translate) + flag at lane 31
   - Inside get's herd body: replaces with scf.while spin loop using
     gpu.shuffle idx, then memref.load after flag observed
   - Erases the channel symbol

2. air-rank-to-mgpu                  [Phase 3, landed]
   - Inlines air.rank, resolves rank IDs from runtime mgpuGetRank()
   - Brackets parent func with mgpuSymmetricHeapInit/Destroy

3. air-symmetric-alloc-to-mgpu       [Phase 4, landed]
   - Lowers memref.alloc {air.symmetric} → mgpuSymmetricAlloc + descriptor

4. air-translate-to-llvm             [Phase 2, landed]
   - Expands air.translate → memref descriptor rebase via heap_bases

5. air-to-rocdl + air-gpu-outlining  [existing AIR-on-GPU passes]
   - Converts air.launch/segment/herd → gpu.launch_func + gpu.func

6. Standard MLIR GPU lowering        [existing]
   - convert-gpu-to-rocdl, gpu-module-to-binary, gpu-to-llvm, etc.
```

Each pass has a focused job; phase 6 doesn't touch herd boundaries.

---

## Implementation steps

### A. Repo cleanup (no code yet)

1. Close PR #1580 (phase 5) with a comment explaining the redesign decision.
   Keep the branch around for reference until the new phase 6 lands.
2. Comment on PR #1581 (phase 6) noting it will be force-pushed with the
   redesigned implementation, or close + open a new PR — whichever the
   project convention prefers.

### B. New pass — `air-gpu-channel-to-cacheline`

Files to add/rewrite:

| File | Action |
|------|--------|
| `mlir/include/air/Conversion/AIRGpuChannelToMgpuPass.h` | rename to `AIRGpuChannelToCachelinePass.h`, update class name |
| `mlir/include/air/Conversion/GPUPasses.td` | rename `AIRGpuChannelToMgpu` def → `AIRGpuChannelToCacheline`, update description, drop runtime-decl `dependentDialects` |
| `mlir/include/air/Conversion/GPUPassDetail.h` | rename macro |
| `mlir/lib/Conversion/AIRGpuChannelToMgpuPass.cpp` | rewrite from scratch |
| `mlir/lib/Conversion/CMakeLists.txt`, `Passes.cpp` | update file name |
| `mlir/test/Conversion/AIRGpuChannelToMgpu/gpu_channel.mlir` | rewrite FileCheck cases |
| `test/gpu/multi_gpu/air_channel/cacheline.mlir` | new e2e test (full hierarchy with channel inside herd) |
| `test/gpu/multi_gpu/air_channel/Makefile` | new self-contained Makefile |

### C. Pass implementation skeleton

```cpp
struct AIRGpuChannelToCachelinePass {
  void runOnOperation() override {
    // 1. Find all air.channel ops with channel_type = "gpu_symmetric_heap"
    SmallVector<air::ChannelOp> chans;
    moduleOp.walk([&](air::ChannelOp c) {
      auto attr = c->getAttrOfType<StringAttr>("channel_type");
      if (attr && attr.getValue() == "gpu_symmetric_heap")
        chans.push_back(c);
    });

    for (auto chan : chans) {
      // 2. Find all puts/gets referencing this channel symbol
      auto puts = collectPuts(chan);
      auto gets = collectGets(chan);

      // 3. Initial scope: require exactly one put and one get
      if (puts.size() != 1 || gets.size() != 1) {
        chan.emitError("only one put and one get supported in this pass");
        return signalPassFailure();
      }

      // 4. Identify producer/consumer ranks from rank-dispatch context
      auto producerRank = inferRankFromContext(puts[0]);
      auto consumerRank = inferRankFromContext(gets[0]);

      // 5. Expand the put inside its herd body
      expandPutToCachelineWrite(puts[0], producerRank, consumerRank);

      // 6. Expand the get inside its herd body
      expandGetToCachelineSpin(gets[0]);

      // 7. Erase the channel symbol
      chan.erase();
    }
  }

  // expandPutToCachelineWrite generates (inside the herd body):
  //   %from = arith.constant <producerRank> : index
  //   %to   = arith.constant <consumerRank> : index
  //   %peer = air.translate %src, %from, %to, %bases
  //         : memref<32xi32>, memref<?xindex>
  //   %tid = gpu.thread_id x
  //   %active = arith.cmpi ult, %tid, %c32 : index
  //   scf.if %active {
  //     %is_flag = arith.cmpi eq, %tid, %c31 : index
  //     ...
  //     %val = arith.select %is_flag, %c1_i32, %payload : i32
  //     memref.store %val, %peer[%tid] : memref<32xi32>
  //   }
  // (replaces the air.channel.put op)

  // expandGetToCachelineSpin generates (inside the herd body):
  //   %tid = gpu.thread_id x
  //   %active = arith.cmpi ult, %tid, %c32 : index
  //   %final_v = scf.while (%dummy = %c0_i32) : (i32) -> i32 {
  //     %v = scf.if %active -> i32 {
  //       %loaded = memref.load %dst[%tid] : memref<32xi32>
  //       scf.yield %loaded : i32
  //     } else { scf.yield %c0_i32 : i32 }
  //     %flag, %valid = gpu.shuffle idx %v, %c31_i32, %c64_i32 : i32
  //     %not_ready = arith.cmpi ne, %flag, %c1_i32 : i32
  //     scf.condition(%not_ready) %v : i32
  //   } do { ... }
  //   scf.if %active {
  //     memref.store %final_v, %dst_local[%tid] : memref<32xi32>
  //   }
  // (replaces the air.channel.get op)
};
```

### D. E2E test shape (`test/gpu/multi_gpu/air_channel/cacheline.mlir`)

Should be a 1:1 wrap of `handwritten/cacheline.mlir` using:
- `air.rank` (Phase 3)
- `memref.alloc {air.symmetric}` (Phase 4)
- `air.channel` + `air.channel.put`/`.get` inside `air.herd` bodies (NEW Phase 6)
- Full `air.launch` / `air.segment` / `air.herd` hierarchy

After the lowering pipeline, the IR should be functionally equivalent to
`handwritten/cacheline.mlir` (same kernel structure, same cache-line
atomicity, same validation).

### E. Open implementation questions

1. **`inferRankFromContext`** — how does the pass walk up from a `put`/`get` op to find which rank dispatches into it? Heuristic: walk parent ops looking for `scf.if` whose condition is `arith.cmpi eq, %rid, %const`, then extract the const. Robust enough for the simple cacheline case; may need extension for more complex patterns.

2. **`%bases` operand for `air.translate`** — in the existing handwritten test, `%bases` is set up at host scope (mgpuGetHeapBases + memcpy + wrap_bytes). In the channel-driven design, the pass needs to either:
   - Require the user to thread `%bases` through `air.launch`/`segment`/`herd` operands, OR
   - Synthesize the heap-bases setup automatically as part of the expansion.

   Initial scope: require user to thread it through (matches the air_rank/cacheline pattern; less magic in the pass).

3. **Wire-index semantics in initial 1-to-1 scope** — for cacheline (1 wire), the wire index is just `%c0`. The pass doesn't need to do any wire-routing analysis. For allgather, this becomes a real concern; deferred.

4. **Verifier additions** — the channel verifier should reject `gpu_symmetric_heap` channels that aren't inside an `air.rank` enclosing scope (since rank IDs are needed for the lowering). May need a check for puts and gets being inside `air.herd` bodies.

5. **FileCheck unit test scope** — initial cases:
   - Basic cacheline put/get pair → cache-line expansion shape
   - Channel symbol is erased after lowering
   - Pass is a no-op for non-`gpu_symmetric_heap` channels (e.g., `npu_*`)
   - Error on multiple puts/gets per channel (initial scope restriction)
   - Error on channel outside `air.rank` context

### F. Validation plan (when GPU access available)

Same pattern as phases 3/4 verification:

1. Build mlir-air with the new pass on the GPU node
2. Run `make -C test/gpu/multi_gpu/air_channel`
3. Confirm output structurally identical to `handwritten/cacheline`:
   `cache-line message PASS (data[0]=100, flag=1)` plus the `[mlir/channel]`
   tag distinguishing the source variant
4. 3-5 stability runs to rule out flake
5. Check that the lowered IR (after the full pipeline) is byte-equivalent
   (modulo SSA naming) to what `INPUT=cacheline` produces in `handwritten/`

---

## Decisions made (locked in)

- ✅ Drop `air.dma_memcpy_nd {src_rank/dst_rank}` lowering (close PR #1580)
- ✅ Lowering target is **kernel-driven cache-line atomicity**, not host-side mgpuMemcpy
- ✅ `air.channel` is the right high-level abstraction (decouples producer/consumer, has `channel_type` hook)
- ✅ `air.herd` is the GPU kernel boundary (per AIRComputeModel.md §2.3)
- ✅ Channel topology encoded by bundle dimensions in `[N]` syntax + put/get wire indices
- ✅ Initial scope: 1-to-1 cacheline (single put + single get); all-pairs deferred
- ✅ Pass produces AIR + memref + gpu.shuffle primitives **inside the herd body** — herd boundary preserved for existing AIR-on-GPU passes to lower

## Decisions to make (when work resumes)

- Pass renaming convention (`air-gpu-channel-to-cacheline` vs `air-symmetric-channel-to-cacheline`)
- How to thread `%bases` (heap_bases memref) into the herd — kernel arg vs synthesized inside the pass
- Whether to require `%peer_rank` as an explicit kernel arg in the herd, or have the pass synthesize a constant from the rank-dispatch context
- Whether to keep the phase 5 PR open as a reference for future bulk-DMA work, or close it

---

## 2026-05-14 progress: infrastructure landed, e2e baseline blocked

### What now works

1. **`#air.symmetric_heap` custom memref memory_space attribute** ([AIROpBase.td](../mlir/include/air/Dialect/AIR/AIROpBase.td)).
   Used as `memref<32xi32, #air.symmetric_heap>`. Carried by the memref type
   through SSA (no need to trace defining ops). Plumbing wired up in
   `AIRDialect.{h,cpp}`, CMake generates `AIRAttrs.{h,cpp}.inc`.

2. **AIR herd verifier** ([AIRDialect.cpp `verifyComputeMemoryAccess`](../mlir/lib/Dialect/AIR/IR/AIRDialect.cpp))
   now skips the L1-or-better-only check for memrefs whose memory_space is
   `#air.symmetric_heap`. Same for `verifyAllocMemorySpace` (segment-level
   alloc check). Direct `memref.load`/`store` on symmetric-heap memrefs
   inside `air.herd` bodies is now legal.

3. **`AIRSymmetricAllocToMgpu` pass** ([new file](../mlir/lib/Conversion/AIRSymmetricAllocToMgpuPass.cpp))
   replaces the in-flight phase 4 PR (#1579). Dispatches on the result
   memref's memory_space (`#air.symmetric_heap`) instead of the op-attribute
   `{air.symmetric}`. FileCheck unit tests updated, all pass.

4. **`air-to-rocdl` partial fix**: 1D / N-D launch + herd dimensions are
   now handled (previously assumed 2D, would crash on `getSizeOperands()[1]`
   for 1D shapes). Pattern set is now frozen so it can be reused across
   multiple launches in the same module (previously `std::move` consumed it
   on first iteration, crashed on second).

### What's still blocked

5. **`air-to-rocdl` multi-launch use-after-free**: when the test has
   *two* `air.launch` ops in the same module (one in each branch of an
   `scf.if %is_producer { producer-launch } else { consumer-launch }`
   dispatch), the pass crashes during block destruction with
   `Cannot destroy a value that still has uses!`. The pattern's
   `deleteAirHerd` / `deleteAirSegment` helpers appear to leave dangling
   references when there are multiple launches with overlapping
   herd/segment structures.

   This is a deeper restructuring issue in `AIRToROCDLPass.cpp`
   (`runOnOperation()` loop around lines 420-443). The pass was clearly
   designed for the 4Kx4K matmul shape (single launch + single segment +
   one or two herds) and has implicit assumptions baked into the
   delete-helpers' iteration order.

   Fix probably involves: collect all launches first, convert each to
   gpu.launch, defer the deletes until after all conversions, walk the
   delete order more carefully so child uses are dropped before parent
   ops are erased.

### What this unblocks for phase 6

The infrastructure (`#air.symmetric_heap` + verifier + phase 4 pass) lets
us write IR like:

```mlir
%data = memref.alloc() : memref<32xi32, #air.symmetric_heap>
...
air.herd tile (%tx) in (%ntx = ...) args(%hd = %data) : memref<32xi32, #air.symmetric_heap> {
  // direct cross-rank loads/stores from inside the kernel — verifier OK
  air.translate %hd, %from, %to, %bases : memref<32xi32, #air.symmetric_heap>, ...
  memref.store %val, %peer[%tid] : memref<32xi32, #air.symmetric_heap>
  ...
}
```

That IR shape is exactly what phase 6's redesigned pass needs to *produce*
(by expanding `air.channel.put`/`.get` inside herd bodies). So the
infrastructure work is reusable even though the e2e baseline doesn't run
end-to-end yet.

### Recommended next steps

1. Land the infrastructure PR (this branch) — gives phase 6 a target to
   lower into. Marked draft until air-to-rocdl is fixed.
2. Open a separate PR fixing the air-to-rocdl multi-launch use-after-free.
   Independent of the channel pass work.
3. Once air-to-rocdl is fixed, the air_hierarchy/cacheline.mlir e2e baseline
   can run; only then is the phase 6 channel pass actually verifiable e2e.

### Test status

- ✅ `mlir/test/Conversion/AIRSymmetricAllocToMgpu/symmetric_alloc.mlir`: 6 cases pass
- ❌ `test/gpu/multi_gpu/air_hierarchy/cacheline.mlir`: written, lit-clean, but
   blocked on the air-to-rocdl multi-launch crash. Kept in tree as the target
   shape for phase 6's lowering.
