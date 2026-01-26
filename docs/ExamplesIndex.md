# MLIR-AIR Examples Index

This document provides a comprehensive categorized index of all code examples in the MLIR-AIR project.

## Quick Reference by Category

- [Beginner Examples](#beginner-examples) - Start here
- [Data Movement Patterns](#data-movement-patterns) - Channel and DMA examples
- [Matrix Operations](#matrix-operations) - GEMM, matrix-vector, etc.
- [Convolution Operations](#convolution-operations) - 2D convolution variants
- [Advanced Examples](#advanced-examples) - Multi-segment, pipelines, ML ops
- [Triton Integration](#triton-integration) - Triton frontend examples
- [Test Suite](#test-suite-testxrt) - Comprehensive test examples

---

## Beginner Examples

### Element-wise Add (`programming_examples/eltwise_add/`)
**Complexity:** ★☆☆

Simplest complete example showing element-wise vector addition.

**Key Features:**
- Uses only `air.herd` (no segment or launch)
- Direct L3→L1 data movement
- `air.dma_memcpy_nd` for transfers
- Tiled across multiple cores

---

### Element-wise Add with L2 (`programming_examples/eltwise_add_with_l2/`)
**Complexity:** ★★☆

Extends basic eltwise_add with L2 memory via `air.segment`.

**Key Features:**
- Introduces `air.segment` for L2 memory
- Three-level memory hierarchy (L3→L2→L1)
- Still uses `air.dma_memcpy_nd`

---

## Data Movement Patterns

### Passthrough Examples (`programming_examples/passthrough/`)
**Complexity:** ★☆☆

Three variants demonstrating different data movement approaches.

**Variants:**
1. **DMA Passthrough** - Using `air.dma_memcpy_nd`
2. **Channel Passthrough** - Using `air.channel.put/get`
3. **External Function** - Vectorized memcopy via external kernel

---

### Channel Examples (`programming_examples/channel_examples/`)
**Complexity:** ★★☆

Collection demonstrating `air.channel` abstraction.

#### Herd-to-Herd (`channel_examples/herd_to_herd/`)
Channel communication between herds (single and multi-segment).

#### Channel Size (`channel_examples/channel_size/`)
Using channel bundles with size parameters and indexing.

#### Hierarchical (`channel_examples/hierarchical/`)
Data flow: Launch→Segment→Herd and back.

#### Broadcast (`channel_examples/broadcast/`)
Broadcasting data to multiple workers (multi-herd and single-herd).

#### Worker-to-Self (`channel_examples/worker_to_self/`) - **WIP**
Worker shuffling data through self-channels.

#### Worker-to-Worker (`channel_examples/worker_to_worker/`) - **WIP**
Inter-worker tile data trading.

---

### Shim DMA 2D (`programming_examples/shim_dma_2d/`)
**Complexity:** ★★☆

Demonstrates 2D data movement using shim DMA operations (interfacing with DDR).

**Key Features:**
- Shows different invocation methods, calling different infrastructures for compilation, building, running and testing

---

### Data Transfer Transpose (`programming_examples/data_transfer_transpose/`)
**Complexity:** ★★☆

Matrix transpose using channels vs DMA.

**Subdirectories:**
- `channel/` - Channel-based transpose
- `dma/` - DMA-based transpose

---

## Matrix Operations

### Matrix Multiplication (`programming_examples/matrix_multiplication/`)
**Complexity:** ★★★

Optimized matrix multiplication with multiple data types and optimizations.

#### BF16 Variant (`matrix_multiplication/bf16/`)

**Key Features:**
- Variable tile sizes (M, N, K at L2 and L1)
- Multiple herd configurations: 2×2, 2×4, 3×3, 4×4
- Configurable: TILE_M, TILE_K_L2, TILE_K_L1, TILE_N
- External vectorized kernels (`mm.cc`, `mm_aie2p.cc`)
- Both aie2 and aie2p targets

#### I8 Variant (`matrix_multiplication/i8/`)
Similar to bf16 but for int8 data type.

#### I16 Variant (`matrix_multiplication/i16/`)
Similar to bf16 but for int16 data type.

---

### Matrix Scalar Add (`programming_examples/matrix_scalar_add/`)
**Complexity:** ★★☆

Divides 2D matrix into tiles and adds scalar to each element.

**Key Features:**
- Five different implementations (some experimental)
- Introduces tiling concepts
- Fundamental `air.launch`, `air.herd`, `air.channel` concepts

---

### Vector-Matrix Multiplication (`programming_examples/vector_matrix_multiplication/`)
**Complexity:** ★★★

Optimized vector-matrix operations.

---

## Convolution Operations

All convolution examples are in `test/xrt/` directory:

### 2D Convolution i32 (`test/xrt/13_conv2d_i32/`)
Basic 2D convolution with int32 data type.

### 2D Convolution i8 with External Vec (`test/xrt/14_conv2d_i8_extern_vec/`)
2D convolution using external vectorized kernel.

### Depthwise 2D Convolution i32 (`test/xrt/21_conv2d_depthwise_i32/`)
Depthwise convolution variant.

### Stride-2 2D Convolution i32 (`test/xrt/22_conv2d_stride2_i32/`)
Convolution with stride=2.

---

## Advanced Examples

### Herd Dataflow (`programming_examples/herd_dataflow/`)
**Complexity:** ★★★

Comprehensive tutorial demonstrating complete pipeline with all abstractions.

**Key Features:**
- Three-herd pipeline on 4-column AIE array
- Both MLIR (`air.mlir`) and Python (`run.py`) implementations
- External C++ kernel integration (`extern_func.cc`)
- Demonstrates all three abstractions: `air.launch`, `air.segment`, `air.herd`

**Documentation:** Extensive [README](../programming_examples/herd_dataflow/README.md)

---

### Multi-Segment (`programming_examples/multi_segment/`) - **WIP**
Examples using multiple segments.

**Status:** Work in progress

---

### Segment Allocation (`programming_examples/segment_alloc/`)
**Complexity:** ★★☆

Demonstrates segment L2 memory allocation accessed from herds.

**Key Concept:** Workers cannot allocate L2; segments must

---

### Conditional Branching (`programming_examples/conditional_branching/`)
**Complexity:** ★★☆

Control flow examples with if-else on runtime-parameters in AIE cores.

---

### LLaMA2 Multi-Head Attention (`programming_examples/llama2_mha/`)
**Complexity:** ★★★

Multi-head attention implementation for LLaMA2, on a single AIE core.

---

### LLaMA2 RoPE (`programming_examples/llama2_rope/`)
**Complexity:** ★★★

Rotary Position Embedding for LLaMA2.

---

### Softmax (`programming_examples/softmax/`)
**Complexity:** ★★★

Softmax operation implementation.

---

### Sine/Cosine (`programming_examples/sine_cosine/`)
**Complexity:** ★★☆

Trigonometric functions on AIE.

---

## Triton Integration

MLIR-AIR can be used as a backend for Triton. These examples in `test/xrt/` show integration:

### Triton Vector Add (`test/xrt/40_triton_vec_add/`)
Basic vector addition via Triton frontend.

### Triton Matrix Multiplication Series
- `32_triton_matmul/` - Basic
- `33_triton_matmul_ver2/` - Variant 2
- `39_triton_matmul_ver3_vectorized/` - Vectorized variant

### Triton Block Pointer Eltwise Mul (`test/xrt/31_triton_blk_ptr_eltwise_mul/`)
Element-wise multiplication using Triton block pointers.

### Triton Softmax
- `41_triton_softmax/` - Basic softmax
- `42_triton_softmax_bf16/` - BF16 variant

---

## Test Suite (`test/xrt/`)

Comprehensive test suite with 42 examples demonstrating various features:

### Basic Operations
- `01_air_to_npu` - Basic scalar i32 matrix multiplication
- `02_mul_shim_1x1` - Multiplication via shim
- `03_mul_L1L2_1x1` - Mul using L1/L2
- `06_add_shim_bf16` - BF16 addition via shim
- `30_mul_rtp_1x1` - Runtime parameter multiplication

### GEMM Variants (Matrix Multiplication)
- `04_gemm_w_pack` - Data transfers lowered from `pack` operations
- `08_gemm_extern_vec` - External vectorized GEMM kernel
- `09_gemm_extern_vec_4x4` - 4×4 variant
- `10_gemm_peeling_extern_vec` - For-loop peeling
- `11_gemm_bias_fusion` - GEMM with bias fusion
- `15_gemm_peeling_extern_vec_4x4_bf16` - BF16 4×4, for-loop peeling
- `16_gemm_peeling_extern_vec_4x4_bf16_packet` - Packet-switched routings at shim DMAs
- `17_gemm_8x16_transform_vec_4x4` - Tiling driven from transform dialect
- `27_gemm_peeling_extern_vec_4x4_i32` - I32 4×4, for-loop peeling
- `28_gemm_loop_nest_bf16` - Loop nest optimization
- `29_gemm_4_level_tiling_extern_vec_4x4_bf16` - 4-level tiling

### Transform Dialect Examples
- `12_matmul_transform_1x4_bf16`
- `18_matmul_8x16_shim_transform_bf16`
- `19_matmul_8x16_core_transform_bf16`
- `37_matmul_transform_4x4_bf16`

### Batch Operations
- `20_batch_matmul_i32` - Batch matmul (i32)
- `25_batch_matmul_bf16` - Batch matmul (bf16)

### Control & Configuration
- `05_extern_func` - External function calls
- `07_extern_linalg` - External linalg ops
- `23_ctrlpkt_config` - Control packet config
- `24_ctrlpkt_config_2gemms_4x4` - Config for 2 GEMMs

### Vector & Matrix Operations  
- `26_vecmat_i8` - Vector-matrix (i8)
- `36_cascade_vecmat_i32` - Cascade vector-matrix
- `38_cascade_vecmat_transform_2x4_i32` - Transform-based cascade

### Cascade & Reduction
- `34_cascade_vecadd` - Cascade vector addition
- `35_herd_reduce` - Herd reduction

---

## Finding the Right Example

### By Goal

| Goal | Recommended Example |
|------|-------------------|
| Learn basics | `eltwise_add` |
| Understand L2 memory | `eltwise_add_with_l2` |
| Learn channels | `channel_examples/` |
| Build pipeline | `herd_dataflow` |
| Optimize matmul | `matrix_multiplication/bf16` |
| Add convolution | `test/xrt/14_conv2d_i8_extern_vec` |
| Use external kernels | `herd_dataflow` or `test/xrt/08_gemm_extern_vec` |
| ML workloads | `llama2_mha`, `softmax` |
| Triton integration | `test/xrt/40_triton_vec_add` |

### By Complexity

| Level | Examples |
|-------|----------|
| Beginner (★☆☆) | `eltwise_add`, `passthrough`, `channel_examples/broadcast` |
| Intermediate (★★☆) | `eltwise_add_with_l2`, `matrix_scalar_add`, most channel examples |
| Advanced (★★★) | `herd_dataflow`, `matrix_multiplication`, `llama2_mha` |

### By Feature

| Feature | Example |
|---------|---------|
| `air.herd` only | `eltwise_add` |
| + `air.segment` | `eltwise_add_with_l2` |
| + `air.launch` | `herd_dataflow` |
| `air.dma_memcpy_nd` | `eltwise_add`, `passthrough/dma` |
| `air.channel.put/get` | `herd_dataflow`, `channel_examples/*` |
| External kernels | `herd_dataflow`, `matrix_multiplication` |
| Transform dialect | `test/xrt/37_matmul_transform_4x4_bf16` |

---

## Contributing Examples

When adding new examples, please:
1. Follow existing directory structure
2. Include a README.md with description and usage
3. Add Makefile with standard targets
4. Include lit test files for CI
5. Update this index
