# Quick Start Guide

This guide will help you get started with MLIR-AIR by walking through actual examples in the project.

## Prerequisites

Before starting, ensure you have:
- Built MLIR-AIR following the [building instructions](building.md)
- Set up your environment with `utils/env_setup.sh` (see building instructions for details)
- Access to a supported AIE device (NPU)

## Understanding MLIR-AIR Abstractions

MLIR-AIR provides several key abstractions for programming AIE devices:

### Core Abstractions

1. **`air.herd`** - Defines a group of compute tiles working in parallel
   - Represents spatial parallelism across AIE cores
   - Each worker in the herd executes the same kernel

2. **`air.segment`** (optional) - Reserves resources including:
   - L2 scratchpad memory space (memtiles)
   - A contiguous region of compute tiles
   - Multiple herds can exist within one segment

3. **`air.launch`** (optional) - Dispatches work to the accelerator
   - Can dispatch multiple iterations/invocations
   - Top-level construct for host-side control

### Data Movement Operations

1. **`air.dma_memcpy_nd`** - N-dimensional DMA transfers
   - Direct, explicit data movement
   - Supports both synchronous and asynchronous scheduling
   - Commonly used for simple patterns

2. **`air.channel.put/get`** - Channel-based communication
   - Decoupled producer/consumer
   - Supports both synchronous and asynchronous scheduling
   - Better for complex dataflow patterns

## Example 1: Simple Element-wise Addition

Location: `programming_examples/eltwise_add/`

This example demonstrates a minimal AIR program using only `air.herd`.

### Key Features
- Direct L3 (DDR) to L1 (core memory) transfers
- Uses `air.dma_memcpy_nd` for data movement
- Tiled computation across multiple cores
- Runs on real NPU hardware

### Code Structure

```python
@herd(name="herd_0", sizes=[1, num_tiles], operands=[arg0, arg1, arg2])
def herd_body(_tx, _ty, _sx, _sy, _l3_a, _l3_b, _l3_c):
    # Allocate L1 memory
    l1_a_data = AllocOp(l1MemrefTy, [], [])
    l1_b_data = AllocOp(l1MemrefTy, [], [])
    l1_out_data = AllocOp(l1MemrefTy, [], [])
    
    for _l_ivx in range_(0, n, tile_n * num_tiles):
        # Copy from DDR (L3) to L1
        dma_memcpy_nd(l1_a_data, _l3_a, 
                      src_offsets=[offset], 
                      src_sizes=[tile_n], 
                      src_strides=[1])
        dma_memcpy_nd(l1_b_data, _l3_b, ...)
        
        # Compute: C = A + B
        for i in range_(tile_n):
            val_a = load(l1_a_data, [i])
            val_b = load(l1_b_data, [i])
            val_out = arith.addf(val_a, val_b)
            store(val_out, l1_out_data, [i])
        
        # Copy result back to DDR (L3)
        dma_memcpy_nd(_l3_c, l1_out_data, ...)
```

### Memory Hierarchy

```
DDR (L3) ←→ L1 Memory (AIE Core)
    ↕              ↕
   DMA      Computation
```

### Running the Example

```bash
cd programming_examples/eltwise_add
make              # Compile and run on NPU
make print        # View generated MLIR
python eltwise_add.py -p  # Print module only
```

## Example 2: Complete Pipeline with All Abstractions

Location: `programming_examples/herd_dataflow/`

This comprehensive example demonstrates all three abstractions: `air.launch`, `air.segment`, and `air.herd`.

### Key Features
- Three-stage pipeline with three herds
- Uses L2 scratchpad memory (memtiles) via `air.segment`
- Channel-based communication between stages (herds)
- Both Python and MLIR implementations
- Runs on real NPU hardware

### Architecture

```
air.launch (grid size based on problem size)
    ↓
air.segment (reserves tiles and L2 scratchpad memory)
    ↓
    ├─ air.herd #1 (Row 1: Vector Add, vectorized kernel lowered using MLIR's `vector` and `memref` dialect operations)
    ├─ air.herd #2 (Row 2: Copy, implemented via `scf.for` loop around scalar `memref.load` and `store` operations)
    └─ air.herd #3 (Row 3: External kernel "add 3", showing function calls to prebuilt external kernels)
```

### Data Flow

```
DDR (L3)
   ↓ air.channel
L2 Memory (Segment)
   ↓ air.channel
L1 Memory (Herd 1) → Compute → L1 out
   ↓ air.channel
L1 Memory (Herd 2) → Compute → L1 out
   ↓ air.channel
L1 Memory (Herd 3) → Compute → L1 out
   ↓ air.channel
L2 Memory (Segment)
   ↓ air.channel
DDR (L3)
```

### Running the Example

```bash
cd programming_examples/herd_dataflow

# Run on hardware (Python source)
make run M_SIZE=512 N_SIZE=1024 AIE_TARGET=aie2 # or aie2p if running on aie2p architecture

# Just print the MLIR
make print
```

### Key Parameters
- `M_SIZE` - Must be multiple of 64
- `N_SIZE` - Must be multiple of 256 (for 4 columns)
- See [README](../programming_examples/herd_dataflow/README.md) for details

## Example 3: Production Matrix Multiplication

Location: `programming_examples/matrix_multiplication/bf16/`

Real-world matrix multiplication with optimizations.

### Key Features
- Multiple herd size configurations (2×2, 2×4, 3×3, 4×4)
- Configurable tile sizes
- External vectorized kernels
- Hardware profiling support
- Performance sweeps

### Running the Example

```bash
cd programming_examples/matrix_multiplication/bf16

# Run with 4×4 herd
make run4x4

# Run with different tile sizes and AIE targets
make run4x4 TILE_M=64 TILE_K_L2=128 TILE_K_L1=32 TILE_N=64 AIE_TARGET=aie2

# Profile on hardware
make profile

# Sweep across problem sizes
make sweep4x4

# Software simulation (event-driven simulator `air-runner`, not cycle-accurate)
make runner
```

See the [Matrix Multiplication README](../programming_examples/matrix_multiplication/bf16/README.md) for full details.

## Understanding the Build Flow

All examples follow a similar compilation pipeline:

```
Python/MLIR Source
        ↓
   AIR Dialect
        ↓
[AIR Transformation Passes]
        ↓
   AIE Dialect  
        ↓
[AIE Transformation Passes]
        ↓
Hardware Binary + Host Code
```

### Viewing Intermediate Representations

```bash
make print              # See final MLIR before lowering
make print_verbose      # See all intermediate stages (if available)
```

## Choosing Your Starting Example

| Example | Complexity | Best For | Simulator Support |
|---------|-----------|----------|-------------------|
| `eltwise_add` | ★☆☆ | Understanding `air.herd` and DMA | No |
| `eltwise_add_with_l2` | ★★☆ | Adding `air.segment` and L2 | No |
| `herd_dataflow` | ★★★ | Complete pipeline, all abstractions | No |
| `matrix_multiplication` | ★★★ | Production code, optimization | Yes (air-runner) |

## Additional Documentation

- [Examples Index](ExamplesIndex.md) - Complete list of all examples
- [GEMM Case Study](GEMMCaseStudy.md) - Deep dive into compilation
- [Performance Guidelines](PerformanceGuidelines.md) - Optimization tips
- [Troubleshooting](Troubleshooting.md) - Common issues and solutions

## Getting Help

- Review example READMEs in each directory
- Check the [MLIR-AIR GitHub Issues](https://github.com/Xilinx/mlir-air/issues)
- Read the [AIR Async Concurrency](AIRAsyncConcurrency.md) guide for AIR's asynchronous concurrency model
- Read the [GEMM Case Study](GEMMCaseStudy.md) for optimization and conversion passes in MLIR-AIR
