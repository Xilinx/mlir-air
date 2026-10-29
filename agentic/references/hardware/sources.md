# Versioned source catalog

Repository-relative paths and exact revisions identify the source declarations
behind the hardware table. Recheck the named symbols when using a newer
toolchain; source declarations and measured runtime behavior have distinct roles.

| Identity | Repository and revision | Role |
|---|---|---|
| **A** | `Xilinx/mlir-aie`, `7e00b57955e108fe9d8e9419f5828a0c7e650858` | Primary source-model profile for the resource table |
| **R** | `Xilinx/aie-rt`, `8849e208bdcc533b20a0ed3f95c1ce961dee9c3a`, submodule at A:`third_party/aie-rt` | Primary AIE2P runtime declarations and DMA encoder |
| **AIR** | `Xilinx/mlir-air`, `6746658f74200feb957f99ad4163297f3185739c` | AIR checkout in the 2026-09-19 audit; its `programming_examples/fused_decode/Makefile` selects `aie2p-none-unknown-elf` |
| **A-old** | `Xilinx/mlir-aie`, `b4ddf3de12c92bcb633827d09ea7b814e997ce16` | Earlier source revision corroborating the main resource totals |
| **R-old** | `Xilinx/aie-rt`, `a8b0667133ea2851ce27793a1796c5968226d9af` | Runtime under A-old's XDNA/XRT tree; corroborates program/data memory and resource counts |

Source review date: **2026-09-19**. A:`include/aie/Dialect/AIE/IR/AIETargetModel.h`
had SHA256 `7115a3f3bfedb75ecf4e2ddb34896ebca5b9a748f9e612ae8c04c0abe879a6b8`.
These versioned declarations establish the resource profile. For an actual
deployment, record the selected compiler binaries, installed target model and
device configuration separately, as described in the
[current AIR source profile](../../profiles/README.md).

## S1 Target model

A:`include/aie/Dialect/AIE/IR/AIETargetModel.h`:

- `BaseNPU2TargetModel`, `NPU2TargetModel`, `VirtualizedNPU2TargetModel`: array dimensions, tile types and partition selection.
- `AIE2TargetModel::getLocalMemorySize`, `getMemTileSize`, `getNumBanks`, `getNumLocks`, `getNumBDs`, `isBdChannelAccessible`: local capacity, compiler bank count, lock/BD counts and BD parity pools.
- `getAddressGenGranularity`, `getBDMaxDims`, `getDmaBdMaxLen`, `getDmaBdStepBits`, `getDmaBdWrapBits`: units and descriptor field limits.
- `getMemSouthBaseAddress`, `getMemWestBaseAddress`, `getMemNorthBaseAddress`, `getMemEastBaseAddress`: core load/store windows.
- `getTileType` interface documentation: compute, MemTile and shim roles.

A:`lib/Dialect/AIE/IR/AIETargetModel.cpp`, `BaseNPU2TargetModel::getTargetArch`, binds NPU2 to `AIEArch::AIE2p`. A:`docs/Devices.md` corroborates the device and partition declarations. Derived totals and bank sizes in the main table are arithmetic on these declarations, not independently measured facts.

## S2 Connectivity and address views

A:`lib/Dialect/AIE/IR/AIETargetModel.cpp`:

- `AIE2TargetModel::getNumSourceSwitchboxConnections` / `getNumDestSwitchboxConnections`: compute/MemTile DMA endpoints, stream bundles and edge restrictions.
- `getNumSourceShimMuxConnections` / `getNumDestShimMuxConnections`: shim DMA endpoints, distinct from shim switchbox links.
- `getMemWest`, `getMemEast`, `getMemNorth`, `getMemSouth`, `isLegalMemAffinity`: neighbor affinity; local memory is the core's east view; no bottom-core southward access into MemTile.
- `AIETargetModel::getMemLocalBaseAddress`, `getLockLocalBaseIndex`: per-owner address/lock translation for MemTile west/local/east views.

These model declarations do not prove that arbitrary endpoint pairs can be routed simultaneously, or that every operation uses every modeled address view.

## S3 AIE2P driver declarations

R:`driver/src/global/xaie2pgbl_reginit.c`:

- `Aie2PCoreMod`: `ProgMemSize` and `DataMemSize`.
- `Aie2PTileMemMod`, `Aie2PMemTileMemMod`: owned data capacity.
- `Aie2PTileDmaMod`, `Aie2PMemTileDmaMod`, `Aie2PShimDmaMod`: `NumBds`, `NumChannels` (per S2MM/MM2S direction), DMA-addressable `NumLocks`.
- `Aie2PTileLockMod`, `Aie2PMemTileLockMod`, `Aie2PShimNocLockMod`: local `NumLocks`. Use these for physical local lock ownership.
- `Aie2PTileDmaProp`, `Aie2PMemTileDmaProp`, `Aie2PShimDmaProp`: `AddrAlignMask`, `AddrAlignShift`, `AddrMax`, step/wrap properties.

R:`driver/src/dma/xaie_dma.c`, `XAie_DmaSetAddrLen`: byte-count API, alignment and address-plus-length guard, address shifting and word-length conversion. R:`driver/src/dma/xaie_dma_aieml.c`, `_XAieMl_MemTileDmaCheckBdChValidity`: MemTile BD/channel parity rule, referenced by the AIE2P DMA module.

## S4 Allocation and stack

A:`include/aie/Dialect/AIE/IR/AIETargetModel.h`, `AIETargetModel::getDefaultCoreStackSize`; A:`include/aie/Dialect/AIE/IR/AIEOps.td`, `CoreOp` `stack_size` and `getEffectiveStackSize` documentation: default/override reservation and explicit warning that it is not checked against actual frames.

A:`lib/Dialect/AIE/Transforms/AIEAssignBuffers.cpp`, `basicAllocation`, `simpleBankAwareAllocation`, `fillBankLimits`: use target capacity/bank count, reserve stack and check assigned extents. A:`lib/Targets/AIETargetBCF.cpp`: linker-side stack and buffer/neighbor reservations. Compiler allocation bank groups are the level established here; physical microbanks, ports and achieved bandwidth are outside this evidence.

## S5 Descriptor allocation and validation

A:`lib/Dialect/AIE/Transforms/AIEAssignBufferDescriptorIDs.cpp`, `BdIdGenerator`, `AIEAssignBufferDescriptorIDsPass`: per-tile assignment and channel accessibility, rather than per-channel independent pools.

A:`lib/Dialect/AIE/Transforms/AIEObjectFifoStatefulTransform.cpp`, `getDMAChannelIndex`: paths requiring adjacent-MemTile access restricted to channels below `getMaxChannelNumForAdjacentMemTile`.

A:`lib/Dialect/AIE/IR/AIEDialect.cpp`, `DMABDOp::verify`: word-unit length bound, dimension count, stride/index bounds and separate iteration checks. These are checks supported by this representation/version, not an exhaustive proof for another lowering.

## Recheck against a new checkout

Use read-only source inspection in the task's permitted source trees: record `git rev-parse HEAD`, inspect the named files/symbols with `rg -n`, check selected-file changes with `git status --short -- <files>`, and hash the relevant installed/source files. Read the chosen build configuration and actual candidate's device declaration. No runtime import, Make target or compiler execution is needed for this source comparison.

If a source changed, compare the relevant symbol and its callers before copying a number. Record disagreement or unsupported ownership/addressing as `UNKNOWN`; do not fill gaps with values from another AIE generation. A later compile, route inspection or device observation must be reported separately at its own evidence level.
