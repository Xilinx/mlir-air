# Upstream compatibility

The example uses the normal AIR build and XRTBackend flows and the
dependency versions pinned under `utils/`.

The B8 mapping keeps O and down resident in the same MemTile, feeding one
projection stream. Each O buffer is sent eight times before its down buffer is
sent eight times, and down is only produced after the O sends are consumed.
AIR emits that order on one circular DMA chain for re-fed transfers that share
a flow; earlier AIR builds alternate the two buffers and stall.

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
path, also used by the 4B DFlash comparator. That path runs with XRT Python
bindings that lack `run.get_ctrl_scratchpad_bo()`, which upstream's ELF
scratchpad path requires.
Decoder manifests retain the configuration, ABI, source snapshot,
compiler/source hashes and artifact hashes; stale builds are rejected.

For benchmark interpretation, the supplied B1 xclbin uses the fixed-capacity
readback convention documented in `fused_decode/decode_insts_gen.py`.
Upstream also has runtime-round xclbin support using an additional readback
calibration artifact, as well as the newer ELF path. This study does not enable
those modes; the reported denominator is the supplied C4096 template runner.

## B16 compiler requirements

The B16 DDR-replay mapping uses the existing AIR passes with mixed-feed DMA
scheduling and mixed-memory packet-producer allocation fixes. Build/install
that AIR revision before compiling B16. The [shared B16 design](../../common/docs/b16-ddr-replay.md)
explains the source/compiler boundary, former generated-IR repairs, structural
trigger conditions and regression coverage. The optional dead-switch cleanup
is recorded separately. Dependency packages follow the normal repository setup;
Make does not install a replacement compiler.
