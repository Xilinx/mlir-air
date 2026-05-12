# `multi_gpu` — symmetric-heap multi-GPU end-to-end tests

End-to-end tests for the symmetric-heap multi-GPU stack. Each test launches
N processes — one per physical GPU — that coordinate via the symmetric heap
(XGMI peer-mapped VMem buffers).

The `mlir/test/Conversion/AIR*ToMgpu/` lit tests pin pass-level invariants
with FileCheck. The tests in this directory are the e2e counterparts: they
build through the full lowering chain and run on real hardware.

## Layout

Tests are organized by IR-abstraction level. Each subdirectory holds tests
written at one level. Lower levels (closer to LLVM dialect) are the lowering
targets that higher levels reduce to.

| Subdir | Phase | Abstraction added |
|---|---|---|
| `handwritten/` | 2 | none — raw MLIR with hand-written GPU kernels and direct `mgpuSymmetricAlloc` / `mgpuGetRank` calls. The reference target. Variants: `cacheline`, `atomic`, `allgather`. |
| `air_rank/` | 3 | `air.rank` declares the multi-process world; replaces hand-written `mgpuGetRank` / heap init/destroy plumbing. Lowered by `air-rank-to-mgpu`. Variants: `cacheline`, `allgather` — each a 1:1 wrap of the corresponding `handwritten/` test. |
| `air_alloc/` | 4 (TBD) | `memref.alloc {air.symmetric}` declares symmetric-heap allocations. Lowered by `air-symmetric-alloc-to-mgpu`. |
| `air_dma/` | 5 (TBD) | `air.dma_memcpy_nd {src_rank/dst_rank}` declares cross-rank DMAs. Lowered by `air-cross-rank-dma-to-mgpu`. |
| `air_channel/` | 6 (TBD) | `air.channel {channel_type = "gpu_symmetric_heap"}` declares cross-rank channels. Lowered by `air-gpu-channel-to-mgpu`. |

A higher-level test should produce — after running its phase's lowering pass
— IR functionally equivalent to one of the `handwritten/` references.

## Running

Each subdirectory has its own self-contained `Makefile`. There is no shared
include or sourced helper — duplication is intentional, so that each phase's
PR touches only its own subdir and there's no cross-phase coupling that can
rot.

Default invocation forks 2 processes:

    make -C test/gpu/multi_gpu/handwritten

Inside a subdirectory, common knobs:

    make -C test/gpu/multi_gpu/handwritten INPUT=cacheline   # default
    make -C test/gpu/multi_gpu/handwritten INPUT=atomic
    make -C test/gpu/multi_gpu/handwritten NUM_RANKS=4
    make -C test/gpu/multi_gpu/handwritten clean

Each `Makefile` documents its own `INPUT` choices in the header comment.

## Preconditions

Each `Makefile`'s `check-preconditions` target refuses to launch if either:

- `NUM_RANKS < 2` — the cross-rank symmetric-heap test fundamentally needs
  a peer; a single-process launch has nothing to talk to.
- Fewer physical GPUs than `NUM_RANKS` — colocating ranks on one GPU would
  silently bypass XGMI/peer-VA (transparently falling back to local memory)
  and report false-positive PASSes.

## Required environment

The Makefiles invoke `air-opt`, `mlir-opt`, and `mlir-runner` via PATH, plus dlopen `libairgpu.so` and the `libmlir_*.so` runtime libraries. There are three ways to satisfy this:

1. **Source `utils/env_setup_gpu.sh`** (recommended) — sets `PATH`, `LD_LIBRARY_PATH`, `MLIR_AIR_INSTALL_DIR`, and `LLVM_INSTALL_DIR` in one go.
2. **Pass install dirs on the make command line**:
   ```
   make MLIR_AIR_INSTALL_DIR=… LLVM_INSTALL_DIR=…
   ```
   (PATH must still contain the binaries — these vars only affect `--shared-libs` paths.)
3. **Have the binaries in `PATH` already** — the Makefile derives `LLVM_INSTALL_DIR` / `MLIR_AIR_INSTALL_DIR` from `dirname $(dirname $(command -v mlir-opt))` etc.

The `check-preconditions` target validates that the resolved `LLVM_LIB_DIR` and `AIRGPU_LIB` paths actually exist before launching, so a missing env shows a clear error rather than a `dlopen` failure deep inside `mlir-runner`.

## Why duplicated boilerplate per subdir

A shared `_common.mk` or `_common.sh` would let one phase's edit silently
break another phase's tests. The boilerplate is small (~30 lines of
preconditions + driver per Makefile) and stable — phases differ in their
compile pipeline, not in the multi-process driver. Duplication is the
cheaper failure mode.
