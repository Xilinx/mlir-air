# Upstream compatibility

The example uses the normal AIR build and XRTBackend flows. The qualification
base is upstream `7e9ec2b2c19e34907cf1cb54c516be73d1ef96c1`, with its dependency pins:

| Dependency | Version |
|---|---|
| MLIR-AIE | `1.4.5.dev51+ge040370` |
| Peano | `22.0.0.2026101201+83ba9ea9` |
| Host LLVM/MLIR | `24.0.0.2026100106+bdf04ebb` |
| Inline-attention linker | Pre-23 `llvm-link`, as required by upstream fused decode |

The B8 mapping requires FIFO ordering for re-fed transfers, provided by
[AIR PR #2152](https://github.com/Xilinx/mlir-air/pull/2152). O and down are
resident in the same MemTile and feed one projection stream. Each O buffer must
be sent eight times before its down buffer is sent eight times. Alternating
one descriptor per buffer stalls when down production waits for all O replays.
Allocating separate channels avoids that stall for dependent producers but does
not establish FIFO ordering when both buffers can supply data independently.

The compiler constructs an ordered circular BD chain in
`AIRToAIEPass.cpp` and `AIRToAIESchedulingUtils.cpp`, keeping the existing lock
credits and DMA allocation. This replaces the initial separate-channel workaround.
Tests are `mlir/test/Conversion/AIRToAIE/refeed_chain_order.mlir`,
`mlir/test/CppUnitTests/circular_chain_bds.cpp`, and both modes of
`test/xrt/56_refeed_chain_order`. Repeated descriptors must fit the MemTile BD
pools; this is checked during compilation. Build/install AIR with #2152 before
compiling this example. See [qualification](qualification.md) for the exact
compiler revision and executed checks.

Two small upstream Python interfaces support clean reuse:

- `fused_decode.py` accepts `DECODE_PAIR_ROWS=1` for this projection packet
  layout; defaults preserve the original B1 geometry and weight footprint.
- `Qwen3Q4nxPrefill` accepts an optional cache instance. The DFlash adapter
  supplies bounded weight-buffer reuse without replacing module globals.

The native Qwen3-8B B1 loader also keeps BF16 embeddings and releases its packed
host-weight temporary after upload, avoiding an unused FP32 vocabulary-head
expansion. The native packer writes the same NPZ members phase by phase instead
of accumulating a whole-model array and a concatenated copy. It publishes the
cache atomically; packing arithmetic, phase order and B1 dispatch are unchanged.

The B8 physical ABI is emitted as AIR subviews. O/down replay is handled by the
compiler. The build does not rewrite generated routes, locks, instructions or
binary argument metadata. The B1 comparator uses upstream's retained xclbin/template
path, also used by the 4B DFlash comparator. This supports the installed XRT
2.21 binding: it lacks `run.get_ctrl_scratchpad_bo()`, required by upstream's
newer default ELF scratchpad path. An ELF build completed but could not run
with that binding. No XRT replacement or scratchpad emulation is required.
Decoder manifests retain the configuration, ABI, source snapshot,
compiler/source hashes and artifact hashes; stale builds are rejected.

For benchmark interpretation, the supplied B1 xclbin uses the fixed-capacity
readback convention documented in `fused_decode/decode_insts_gen.py`.
Upstream also has runtime-round xclbin support using an additional readback
calibration artifact, as well as the newer ELF path. This study does not enable
those modes; the reported denominator is the supplied C4096 template runner.
