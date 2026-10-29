# DMA addressing: units, scope and finite limits

This page uses the NPU2/AIE2P versions in the [main reference](README.md). Its numbers are source declarations and arithmetic derivations. They are not tested transfer configurations. The [source catalog](sources.md) names every underlying file and symbol.

## Keep three address spaces separate

An allocation offset within owned tile SRAM, the address encoded into a DMA BD, and a register address used to program that BD are different values. A core's load/store neighbor windows are a fourth view. Never validate one merely by comparing it with another's capacity.

| Quantity | Compute tile DMA | MemTile DMA | Shim NOC DMA | Source and interpretation |
|---|---|---|---|---|
| `AddrAlignMask` | `0x3` | `0x3` | `0x3` | All require a **4-byte-aligned starting address** in the driver path. This is not the complete buffer/vector allocation alignment policy. [S3](sources.md#s3-aie2p-driver-declarations) |
| `AddrAlignShift` | **2** | **2** | **0** | `XAie_DmaSetAddrLen` shifts the byte address before encoding: core/MemTile address fields use **4-byte** units, while the shim field retains byte addressing. [S3](sources.md#s3-aie2p-driver-declarations) |
| Driver `AddrMax` bound | `0x20000` (**128 KiB**) | `0x180000` (**1536 KiB**) | `0x1000000000000` (**2^48 bytes**) | `XAie_DmaSetAddrLen` rejects unaligned addresses or **Addr + Len > AddrMax**. These are aperture/validation bounds, not allocatable local memory sizes. [S3](sources.md#s3-aie2p-driver-declarations) |
| Owned local data SRAM | **64 KiB** | **512 KiB** | No tile payload SRAM budget exposed | The address aperture cannot replace ownership, BO bounds, partition or neighbor-access checks. [S1](sources.md#s1-target-model), [S3](sources.md#s3-aie2p-driver-declarations) |

The MemTile model maps west/local/east memory windows to byte bases **0, 0x80000, 0x100000**, each spanning **512 KiB**; their union explains the **1536 KiB** encoded aperture. At an edge, an absent neighbor does not become usable. The allocation's local offset and the encoded address must include the correct owner view. [S2](sources.md#s2-connectivity-and-address-views)

The core's load/store windows are south **0x40000**, west **0x50000**, north **0x60000**, and local/east **0x70000**, with **64 KiB** per memory view where a legal neighbor exists. These core load/store address windows are not the core DMA aperture. This reference does not resolve ownership/reachability for every address in the driver's **128 KiB** core-DMA aperture; check the lowering and selected owner instead of assuming twice the local SRAM. [S1](sources.md#s1-target-model), [S2](sources.md#s2-connectivity-and-address-views), [S3](sources.md#s3-aie2p-driver-declarations)

Shim addresses refer to the actual external allocation/binding. A **48-bit** address aperture says nothing about the size of one BO, host argument encoding, IOMMU mapping, segment-relative offsets or host API length. Trace the actual BO plus offset, length and packed-data provenance. Equal BO sizes can still bind the wrong data. [S3](sources.md#s3-aie2p-driver-declarations)

## A payload that fits SRAM may still need multiple BDs

The current target model declares the following fields. Transfer length is in **32-bit words (4 bytes)**, whereas higher-level dimensions may be in elements; always follow the lowering's conversion. [S1](sources.md#s1-target-model), [S5](sources.md#s5-descriptor-allocation-and-validation)

| Descriptor property | Compute | MemTile | Shim NOC |
|---|---|---|---|
| Maximum ordinary ND dimensions in one BD | **3** | **4** | **3** |
| Buffer-length field width | **14 bits** | **17 bits** | **32 bits** |
| Maximum field value, in 32-bit words | **2^14 − 1** | **2^17 − 1** | **2^32 − 1** |
| Step/stride field width | **13 bits** | **17 bits** | **20 bits** |
| Wrap field width | **8 bits** | **10 bits** | **10 bits** |

These are the model's `getBDMaxDims`, `getDmaBdMaxLen`, `getDmaBdStepBits` and `getDmaBdWrapBits` declarations. Ordinary ND dimensions exclude the separately represented iteration/repeat mechanism. A width is a raw encoding fact: API stride/wrap conventions may use biased values, and broadcast/disabled dimensions have special rules. Inspect `DMABDOp::verify`, the actual operation and the runtime encoder before deriving an allowed element stride. [S1](sources.md#s1-target-model), [S5](sources.md#s5-descriptor-allocation-and-validation)

For example, **65536 bytes** fit the compute tile's nominal local SRAM, but the declared length-field maximum is **(2^14 − 1) × 4 = 65532 bytes**. A single full-SRAM-length BD would exceed that field under this model, independently of stack or allocation pressure. A supported split/chained transfer changes the BD and protocol budget. This is a source-derived example, not a compiled test. The driver's `Len` argument is a **u32 byte count**, so the shim's wider field capacity is not a promise that this particular API accepts a transfer of that size. [S1](sources.md#s1-target-model), [S3](sources.md#s3-aie2p-driver-declarations), [S5](sources.md#s5-descriptor-allocation-and-validation)

For a strided transfer, check every addressed extent and tail against the correct allocation, not just `Addr + payload bytes`; gaps can make the accessed span larger than the transferred payload. The driver bound shown above is one validation rule, not a proof of strided bounds or semantic element order.

## Count descriptor and synchronization ownership

A BD ID names a slot in the tile's descriptor pool. Core and shim each expose **16** total slots. MemTile exposes **48**, with only **24** reachable by any given channel parity: even channels use **0–23**, odd channels **24–47**. The compiler's `BdIdGenerator` tracks assigned IDs per tile and checks channel accessibility. RX and TX on the same tile must be included in this accounting. [S1](sources.md#s1-target-model), [S5](sources.md#s5-descriptor-allocation-and-validation)

Repeated execution of a descriptor does not inherently require a new slot per repetition, but distinct resident stages, tails, chains or runtime tasks may require additional slots. Reprogramming/reusing a BD requires proof that the previous use has completed. Neither loop count nor source channel count alone determines descriptor demand.

For locks, distinguish local object ID, encoded neighbor-view ID, initial semaphore value and acquire/release amount. MemTile's local **64** locks and DMA-addressable **192** IDs have different scopes. The driver and target model support semaphore operations; having enough IDs says nothing about matched credits, wait cycles, or the last reader releasing a buffer. [S1](sources.md#s1-target-model), [S2](sources.md#s2-connectivity-and-address-views), [S3](sources.md#s3-aie2p-driver-declarations)

When the source format or lowering differs from these declarations, retain `UNKNOWN` for the unverified relationship and inspect the selected backend or emitted BD. Do not treat an unrelated parser's success as addressing validation.
