# Matrix Multiplication (BF16)

Matrix multiplication with bf16 datatype, supporting variable tile sizes for M and N dimensions, and for K at L2 and L1 cache levels.

## Available Make Targets

- `make run4x4` - Compile and run the design on NPU with a 4×4 herd (512×512×512 matrix)
- `make run3x3` - Compile and run with a 3×3 herd (576×576×576 matrix)
- `make run2x4` - Compile and run with a 2×4 herd (512×512×512 matrix)
- `make run2x2` - Compile and run with a 2×2 herd (512×512×512 matrix)
- `make runner` - Simulate the design on air-runner to find expected latency. Generates "simulation_trace.json" that can be visualized with Perfetto UI.
- `make profile` - Run the design on hardware and measure average latency
- `make sweep4x4` - Measure end-to-end latencies across a range of problem shapes (256-2048)
- `make print` - Print the generated MLIR without running

## Tile Size Configuration

Tile sizes can be customized by setting the following parameters:

```bash
TILE_M      # M dimension tile size (default: 128)
TILE_K_L2   # K dimension L2 tile size (default: 64)
TILE_K_L1   # K dimension L1 tile size (default: 32)
TILE_N      # N dimension tile size (default: 64)
```

**Example:**
```bash
make run4x4 TILE_M=64 TILE_K_L2=128 TILE_K_L1=32 TILE_N=64
```
- Recommended configuration for `run3x3` (576×576×576): `TILE_M=64 TILE_K_L2=288 TILE_K_L1=48`

## Target Architecture Selection

The AIE target can be selected using the `AIE_TARGET` parameter:

```bash
make run4x4                  # Uses aie2 target (default)
make run4x4 AIE_TARGET=aie2p # Uses aie2p target
```
