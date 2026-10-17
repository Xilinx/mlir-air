# Matrix Multiplication with a Ragged K (bf16)

`A[M,K] x B[K,N] -> C[M,N]`, bf16 in, f32 out, where K need not be a multiple of
the K step. It is written the way a Triton GEMM is:

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k in range(0, tl.cdiv(K, BLOCK_K)):
    a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
    acc = tl.dot(a, b, acc)
```

| Triton | here |
|---|---|
| one program per C block | one launch point per `TILE_M x TILE_N` block |
| K loop | a column of cores, one `BLOCK_K` step per core |
| `acc` carried across iterations | `acc` passed down the column over the cascade |
| masked load, `other=0` | memtile DMA `pad_after` on the L2 to L1 transfer |
| `tl.dot` | `air.api.ops.dot` on blocked tiles, vectorized by direct codegen |

With the defaults, K=960 and `BLOCK_K`=256: three cores take full steps and the
last takes 192 columns of A and rows of B, padded to 256 with zeros.

NPU2 (Strix) only, and Peano only.

## Available Make Targets

- `make run` - Compile and run on NPU (128x128x960)
- `make run_k1024` - K a whole number of steps, so nothing is padded
- `make run_3step` - three steps (K=704)
- `make run_maxpad` - the longest pad the memtile adds here (K=904)
- `make run1x1` - a single launch point
- `make profile` - Run on hardware and report latency + GFLOPs
- `make sweep` - Measure end-to-end latencies across K
- `make print` - Print the generated MLIR without running

`COMPILE_MODE=compile-only` builds without XRT; `compile-and-xclbin` stops
after the xclbin.

## Configuration

```bash
M, N, K    # problem size (default: 128, 128, 960)
TILE_M     # M size of a C tile (default: 32)
TILE_N     # N size of a C tile (default: 32)
BLOCK_K    # K step, one per core (default: 256)
```

| constraint | why |
|---|---|
| `2 <= ceil(K / BLOCK_K) <= 4` | one step per core of a column |
| `K % 8 == 0` | the pad is counted in 8-deep K blocks of the matmul |
| last step pads at most 15 blocks | limit of the memtile padding field it uses |
| `TILE_M`, `TILE_N`, `BLOCK_K` multiples of 8 | the aie2p 8x8x8 bf16 matmul |
