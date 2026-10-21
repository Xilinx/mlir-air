# Upstream compatibility

The example uses the normal AIR build and XRTBackend flows. The qualification
base is upstream `7e9ec2b2c19e34907cf1cb54c516be73d1ef96c1`, with its dependency pins:

| Dependency | Version |
|---|---|
| MLIR-AIE | `1.4.5.dev51+ge040370` |
| Peano | `22.0.0.2026101201+83ba9ea9` |
| Host LLVM/MLIR | `24.0.0.2026100106+bdf04ebb` |
| Inline-attention linker | Pre-23 `llvm-link`, as required by upstream fused decode |

There is one additional compiler fix in this branch. Independent resident
MemTile replay buffers must retain separate MM2S drain rings when they share a
logical packet channel. Previously the allocator could collapse O and down
onto an alternating ring, although down production depends on all eight O
replays finishing. That build compiled but its first NPU call failed.

The fix lives in `mlir/lib/Conversion/AIRToAIESchedulingUtils.cpp`, with
`mlir/test/Conversion/AIRToAIE/memtile_independent_refeed.mlir` covering independent
replays and ordinary shared packet buffers. Channel exhaustion emits a diagnostic.
The complete compiler LIT suite passed: 657 tests, 7 expected failures and
7 unsupported tests. Build/install AIR with this fix before compiling 8B.

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
with that binding. No XRT replacement or scratchpad emulation is required. Decoder manifests retain the configuration, ABI, source snapshot,
compiler/source hashes and artifact hashes; stale builds are rejected.
