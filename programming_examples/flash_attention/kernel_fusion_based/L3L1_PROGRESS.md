# Flash Attention on NPU2 — Progress

## Summary

Working multi-head flash attention implementation using **memtile-relayed dataflow**
with **selective Q capture**, **k-major K sub-tile layout**, **cascade-after online
softmax merge** across 4 cascade stages, and **segment unroll for 2-head parallelism**
on NPU2 (Strix, AIE2P).

Features: MHA, GQA, MQA, causal masking, arbitrary sequence lengths up to 32k+.

**Key files**: `attn.py` (replaces original), `attn.cc` (k-major kernel), `attn_l3l1.py` (copy)

**Draft PR**: [#1466](https://github.com/Xilinx/mlir-air/pull/1466)

## Architecture

- **4x4 herd x 2 segment unroll**: 2 heads processed in parallel across 8 columns
  - Head 0: columns 0-3, Head 1: columns 4-7
  - Each head uses 4 columns (Q tiles) x 4 rows (cascade stages) = 16 compute tiles
  - Total: 32 compute tiles active simultaneously
- **Multi-head scaling**: `num_heads_per_unroll=2`, additional heads via launch iteration
  - Launch `sizes=[num_lq_iters, num_head_groups]` where `num_head_groups = num_heads // 2`
  - Cores run persistently (`omit_while_true_loop=False`), no PDI reload between iterations
- **QK distribution**: Per-stage memtile relay (`QKIn_s` L3->memtile, `QK2L1_s` memtile->L1)
  - Sub-tile layout applied at memtile relay put
  - Circuit-switched `aie.flow` multicast to all 4 column tiles per stage
  - Eliminates BD exhaustion at large LK (tested up to 32k)
- **Q distribution**: Selective capture pattern via `scf.if(tx == qt)`
- **K distribution**: k-major sub-tile layout, per-stage via memtile
- **V distribution**: Per-stage via memtile with 3D head dimension
- **GQA**: `--num-kv-heads` for grouped query attention (floor-div head mapping)
- **Causal masking**: `--causal` flag with L1 counter buffer for block index tracking
- **Cascade**: After chunk loop (cascade-after pattern)
  - Online softmax correction across 4 cascade stages
  - Three-tier: last stage PUTs, middle stages GET+merge+PUT, first stage GET+merge+div+output
- **Output gather**: `Gp2L2[4,1]` per-segment, then `GpOut[2]` to L3

## Key Design Decisions

### Why memtile relay for QK (not L3-to-L1 direct)?
L3-to-L1 direct QK causes BD exhaustion at LK>8192 (shim DMA splits K BDs when
chunks_per_stage exceeds ~32). Memtile relay absorbs the per-chunk iteration: shim
sends all data in 1-2 large BDs, memtile loops through chunk-by-chunk. Same pattern
as V distribution. Peak throughput increased from ~2746 to ~3010 GFLOPS.

### Why selective Q capture?
With separate Q and K channels, the compiler allocates different lock pairs for each,
causing deadlock when Q and K share the same tile S2MM:0 port. Selective capture uses
a single channel for both Q and K, so one lock pair, no deadlock.

### Why k-major K sub-tile layout?
The sub-tile DMA relay sends blocks in column-major order (dk blocks outer, lkp blocks
inner). The matmul template's B pointer must match: inside the `transpose_b=true` block,
the load address uses k-major indexing `pB + (i * colB + j) * size_B` instead of
n-major `pB + (j * colA + i) * size_B`.

### Why cascade-after (not cascade-before)?
Cascade-before creates circular backpressure: tile waiting for cascade can't consume
broadcast data, which blocks the broadcast for all tiles. Cascade-after: all tiles
consume data independently first, then cascade.

### Why segment unroll for multi-head?
Segment unroll duplicates the 4x4 herd across 8 columns, processing 2 heads in parallel.
The compiler handles channel specialization, core outlining, and shim DMA allocation
automatically. Additional heads iterate via the launch dimension with persistent cores.

## Performance Results

### Multi-Head (2x 4x4 herd via segment unroll, 8 columns)

| Heads | LQ | LK | Mode | Correlation | Avg Time | Peak GFLOPS |
|-------|------|------|------|-------------|----------|-------------|
| 2 | 512 | 512 | MHA | 0.995 | -- | -- |
| 4 | 512 | 512 | MHA | 0.996 | -- | -- |
| 4 | 512 | 512 | GQA 4Q/2KV | 0.997 | -- | -- |
| 4 | 512 | 512 | MQA 4Q/1KV | 0.996 | -- | -- |
| 12 | 512 | 512 | GQA 12Q/4KV | 0.995 | -- | -- |
| 2 | 512 | 512 | Causal | 0.998 | -- | -- |
| 4 | 512 | 512 | Causal | 0.998 | -- | -- |
| 12 | 8192 | 8192 | MHA | -- | 70.7 ms | **3010** |
| 12 | 4096 | 4096 | Causal | -- | 20.8 ms | **2642** |
| 2 | 32768 | 32768 | MHA | 0.666 | -- | -- |

### Sequence Length Scaling

| LK | Correlation | KV chunks/stage |
|----|-------------|-----------------|
| 512 | 0.995 | 2 |
| 4096 | 0.977 | 16 |
| 8192 | 0.944 | 32 |
| 16384 | 0.867 | 64 |
| 32768 | 0.666 | 128 |

Correlation degrades at larger LK due to bf16 cascade merge accumulation.

## dk Scaling (WIP)

dk scaling (dk=128, 256) tiles the key dimension via an inner loop in the herd body.
The matmul kernel is compiled with `-Ddk=64 -DDK_TOTAL=128` so the matmul processes
64-column inner products and the softmax uses `sqrt(128)` for scaling.

### Approach: Q re-reception with memtile staging

For each K chunk, the herd receives `num_dk_iters` Q and K dk_tile-wide column slices,
accumulating partial Q@K^T scores into G before softmax.

**Status**: Compilation succeeds for dk=128 (PR #1467 fixed domination error in
`air-isolate-async-dma-loop-nests`). Runtime deadlocks due to DMA ordering mismatch
between shim BD delivery and memtile relay consumption.

**Root cause**: The shim BD sends K data in chunks that don't align with the memtile
relay's per-tile consumption granularity. Two approaches attempted:
1. **Per-tile K puts**: Individual 4096-element puts per chunk x dk_iter. Compiles
   and produces correct BDs, but deadlocks at runtime.
2. **Memtile dk staging**: Full dk-width L2 buffer with dk_tile column extraction.
   The `air-fuse-channels` pass eliminates QKIn channels, causing "out of channels".

**Debug IR files** (for dk=128 deadlock version):
- `dk128_deadlock_input_to_air_to_aie.mlir` — input to failing pass
- `dk128_deadlock_npu.air.mlir` — full compiled output
- `dk64_input_to_air_to_aie.mlir` — working dk=64 reference
- `air_project/debug_ir/` — all 40 pass-by-pass IR files

## Compiler Fixes Required

| PR | Fix | Status |
|----|-----|--------|
| [#1458](https://github.com/Xilinx/mlir-air/pull/1458) | 3D channel broadcast specialization | Landed |
| [#1459](https://github.com/Xilinx/mlir-air/pull/1459) | `scf.if`/`affine.if` in dependency canonicalization | Landed |
| [#1460](https://github.com/Xilinx/mlir-air/pull/1460) | NumPy broadcasting rule validation | Landed |
| [#1461](https://github.com/Xilinx/mlir-air/pull/1461) | Shim DMA linkage with segment unroll | Landed |
| [#1462](https://github.com/Xilinx/mlir-air/pull/1462) | Fold redundant `scf.if` from segment unroll | Landed |
| [#1463](https://github.com/Xilinx/mlir-air/pull/1463) | metadataArray sorting for 2D channels | Landed |
| [#1465](https://github.com/Xilinx/mlir-air/pull/1465) | Disable expensive checks in transform interpreter | Landed |
| [#1467](https://github.com/Xilinx/mlir-air/pull/1467) | Fix dependency canonicalize dominance with scf.if | Landed |
| [#1468](https://github.com/Xilinx/mlir-air/pull/1468) | Fix shim DMA allocation interleaving | Open |

## How to Run

### Using Makefile

```bash
cd programming_examples/flash_attention/kernel_fusion_based

# Default: 2 heads, LQ=LK=512
make run PEANO_INSTALL_DIR=$PEANO_INSTALL_DIR

# GQA
make run PEANO_INSTALL_DIR=$PEANO_INSTALL_DIR NUM_HEADS=12 NUM_KV_HEADS=6

# Causal
make run PEANO_INSTALL_DIR=$PEANO_INSTALL_DIR EXTRA_PY_FLAGS="--causal"

# Large sequence
make run PEANO_INSTALL_DIR=$PEANO_INSTALL_DIR LK=16384 LQ=16384 LQP=256 NUM_HEADS=2

# Profile
make profile PEANO_INSTALL_DIR=$PEANO_INSTALL_DIR LK=8192 LQ=8192 NUM_HEADS=12
```

### Recompile Kernel

```bash
PEANO=$PEANO_INSTALL_DIR
$PEANO/bin/clang-19 -O2 -std=c++20 --target=aie2p-none-unknown-elf \
    -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
    -DNDEBUG -I $MLIR_AIE_INSTALL_DIR/include \
    -DBIT_WIDTH=8 -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 -DROUND_CONV_EVEN \
    -Dlqp=64 -Ddk=64 -Dlkp=64 -Ddv=64 \
    -c attn.cc -o build_peano/attn.o
cp build_peano/attn.o attn.o
```

For dk=128 kernel: add `-DDK_TOTAL=128` to the compilation flags.

## Known Limitations

1. **Correlation degrades at large LK**: bf16 cascade merge accumulation loses precision
   over many chunks. Fix: use f32 accumulators in cascade merge kernels.
2. **num_heads must be even**: Segment unroll processes 2 heads at a time.
3. **dk scaling blocked**: dk=128 compiles but deadlocks at runtime due to DMA ordering
   mismatch. Requires compiler support for memtile dk staging or fix to per-tile K BD
   delivery alignment.
4. **attn.o must match dk**: The kernel binary bakes in `constexpr_sqrt_dk`. A dk=128
   kernel binary (`-DDK_TOTAL=128`) will produce wrong results if used with dk=64 data.
   Always recompile when changing dk.
