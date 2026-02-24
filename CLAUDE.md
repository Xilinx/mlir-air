# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

MLIR-AIR is a compiler infrastructure for mapping AI workloads onto AMD NPUs (Neural Processing Units) and Versal AI Engine arrays. It provides high-level abstractions through MLIR dialects to program spatial computing architectures, manage data movement, and generate optimized code for AI accelerators. The project includes:

- **AIR Dialect**: High-level asynchronous IR for spatial computing with operations like `air.launch`, `air.segment`, `air.herd`, and `air.dma_memcpy_nd`
- **AIRRt Dialect**: Runtime-level operations for device configuration and DMA management
- **Compiler Infrastructure**: Transformation passes, scheduling optimization, and conversion to AIE dialect
- **Tools**: `air-opt`, `aircc.py`, `air-runner`, `air-translate`
- **Python API**: High-level Python bindings for kernel development
- **Runtime Libraries**: Host-side (airhost) and CPU simulation (aircpu) runtimes

Reference: Wang et al. "From Loop Nests to Silicon: Mapping AI Workloads onto AMD NPUs with MLIR-AIR" (arXiv:2510.14871)

## Environment Setup

### Quick Start with Prebuilt Wheels (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/Xilinx/mlir-air.git
cd mlir-air

# 2. Install system dependencies
sudo apt-get install -y ninja-build clang lld unzip

# 3. Set up Python virtual environment
source utils/setup_python_packages.sh

# 4. Build with wheels (software-only)
./utils/build-mlir-air-using-wheels.sh [build_dir] [install_dir]

# OR build with XRT support (for hardware execution)
./utils/build-mlir-air-using-wheels.sh --xrt-dir <xrt_path> [build_dir] [install_dir]

# 5. Set up environment
source utils/env_setup.sh [install_dir] \
    $(python3 -m pip show mlir_aie | grep Location | awk '{print $2}')/mlir_aie \
    $(python3 -m pip show llvm-aie | grep Location | awk '{print $2}')/llvm-aie \
    my_install/mlir

# 6. If using XRT, also run:
source [xrt_dir]/setup.sh
```

### Prerequisites

- **Python 3.10+**
- **gcc >= 11**
- **ninja-build, clang, lld**
- **XRT** (optional, required for hardware execution)

## Build Commands

### Build with Prebuilt Wheels

```bash
# Software-only build
./utils/build-mlir-air-using-wheels.sh

# With XRT support
./utils/build-mlir-air-using-wheels.sh --xrt-dir /opt/xilinx/xrt
```

### Manual Build from Source

```bash
# Clone and build LLVM
./utils/clone-llvm.sh
./utils/build-llvm-local.sh llvm

# Clone and build libXAIE
./utils/github-clone-build-libxaie.sh

# Clone and build MLIR-AIE
./utils/clone-mlir-aie.sh
./utils/build-mlir-aie-local.sh llvm mlir-aie/cmake/modulesXilinx aienginev2/install mlir-aie

# Build MLIR-AIR with XRT
./utils/build-mlir-air-xrt.sh llvm mlir-aie/cmake/modulesXilinx mlir-aie aienginev2/install /opt/xilinx/xrt
```

### CMake Build (Advanced)

```bash
mkdir build && cd build
cmake .. -G Ninja \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_DIR=${LLVM_DIR}/lib/cmake/llvm \
    -DMLIR_DIR=${LLVM_DIR}/lib/cmake/mlir \
    -DAIE_DIR=${AIE_DIR}/lib/cmake/aie \
    -DXRT_LIB_DIR=${XRT_DIR}/lib \
    -DXRT_INCLUDE_DIR=${XRT_DIR}/include \
    -DCMAKE_BUILD_TYPE=Release
ninja
ninja install
```

Key CMake options:
- `AIR_RUNTIME_TARGETS`: Target runtime architecture (x86_64, aarch64)
- `ENABLE_RUN_XRT_TESTS`: Enable XRT integration tests
- `BUILD_SHARED_LIBS`: Build shared libraries

## Testing

### Running Tests

```bash
cd build
ninja install

# Run C++ unit tests
ninja check-air-cpp

# Run MLIR dialect/pass tests
ninja check-air-mlir

# Run Python tests
ninja check-air-python

# Run XRT/hardware tests (requires XRT)
lit -sv --time-tests --timeout 600 -j5 test/xrt

# Run a single XRT test
lit -sv test/xrt/01_air_to_npu

# Run all e2e tests with Peano
ninja check-air-e2e-peano
```

### Test Categories

- `mlir/test/Conversion/` - Conversion pass tests (AIRToAIE, AIRRtToNpu, etc.)
- `mlir/test/Transform/` - Transform pass tests
- `mlir/test/Dialect/` - Dialect operation tests
- `test/xrt/` - XRT hardware integration tests (50+ tests)
- `test/airhost/` - AIR host runtime tests

### Test File Format

Tests use LIT with FileCheck:
```mlir
// RUN: air-opt %s -air-to-aie | FileCheck %s
// CHECK: aie.tile
```

## Code Architecture

### Key Directories

- **`mlir/`**: Core MLIR dialect and passes
  - `mlir/include/air/Dialect/AIR/`: AIR dialect TableGen definitions
  - `mlir/include/air/Dialect/AIRRt/`: AIRRt dialect definitions
  - `mlir/include/air/Conversion/`: Conversion pass declarations
  - `mlir/include/air/Transform/`: Transform pass declarations
  - `mlir/lib/Dialect/AIR/`: AIR dialect implementation
  - `mlir/lib/Dialect/AIRRt/`: AIRRt dialect implementation
  - `mlir/lib/Conversion/`: Conversion pass implementations
  - `mlir/lib/Transform/`: Transform pass implementations
  - `mlir/test/`: LIT tests for dialects and passes

- **`tools/`**: Executable tools
  - `tools/air-opt/`: MLIR optimizer with AIR passes
  - `tools/aircc/`: Compiler driver wrapper
  - `tools/air-runner/`: Simulation/execution tool
  - `tools/air-translate/`: IR translation tool

- **`python/`**: Python bindings and APIs
  - `python/air/compiler/aircc/`: Python compiler driver implementation
  - `python/air/dialects/`: Python dialect bindings
  - `python/air/backend/`: Backend implementations (xrt, cpu)

- **`runtime_lib/`**: Runtime libraries
  - `runtime_lib/airhost/`: Host-side runtime (memory, queues, DMA)
  - `runtime_lib/aircpu/`: CPU simulation backend

- **`programming_examples/`**: Example programs
  - Matrix multiplication (bf16, i16, i8)
  - Softmax, attention, flash attention
  - LLaMA2 MHA and RoPE
  - Element-wise operations
  - Convolution

- **`test/`**: Integration tests
  - `test/xrt/`: XRT hardware tests
  - `test/airhost/`: Host runtime tests

### MLIR Dialects

| Dialect | Purpose |
|---------|---------|
| **AIR** | High-level async IR for spatial computing: tiles, herds, DMAs, channels |
| **AIRRt** | Runtime operations: metadata, DMA operations, herd/segment loading |

### AIR Dialect Operations

| Operation | Description |
|-----------|-------------|
| `air.launch` | Top-level kernel launch with iteration space |
| `air.segment` | Device segment (maps to AIE columns) |
| `air.herd` | 2D array of compute tiles |
| `air.dma_memcpy_nd` | N-dimensional DMA transfer |
| `air.channel` | Named data channel for communication |
| `air.channel.put` | Put data on channel (producer) |
| `air.channel.get` | Get data from channel (consumer) |
| `air.execute` | Asynchronous execution region |
| `air.wait_all` | Synchronization barrier |

### AIRRt Dialect Operations

| Operation | Description |
|-----------|-------------|
| `airrt.module_metadata` | Module-level metadata |
| `airrt.segment_metadata` | Per-segment metadata |
| `airrt.herd_metadata` | Per-herd metadata |
| `airrt.dma_memcpy_nd` | Runtime DMA operation |
| `airrt.herd_load` | Load/initialize a herd |
| `airrt.segment_load` | Load/initialize a segment |

### Compilation Pipeline

```
Python/MLIR Input
        ↓
Loop Optimization Passes:
  - affine-loop-opt (tiling, data copy)
  - air-automatic-tiling
  - air-regularize-loop
  - air-loop-permutation
        ↓
Parallel to AIR Conversion:
  - air-par-to-launch
  - air-par-to-segment
  - air-par-to-herd
  - air-copy-to-dma
        ↓
Linalg Lowering:
  - air-linalg-codegen
  - air-linalg-bufferize
  - air-lower-linalg-tensors
        ↓
Dependency Analysis & Scheduling:
  - air-dependency
  - air-dependency-canonicalize
  - air-dependency-schedule-opt
  - air-broadcast-detection
  - air-ping-pong-transform
        ↓
DMA & Channel Optimization:
  - air-dma-to-channel
  - air-fuse-channels
        ↓
Herd Placement:
  - air-place-herds
        ↓
AIR to AIE Conversion:
  - air-to-aie (generates AIE tiles, cores, DMAs, flows)
        ↓
Runtime Lowering:
  - air-to-std (AIR to AIRRt)
  - airrt-to-npu (for Ryzen AI NPUs)
  - airrt-to-llvm (for host code)
        ↓
Output: .xclbin + .insts.bin (or .elf for npu2)
```

### Key Transformation Passes

Located in `mlir/include/air/Transform/Passes.td`:

| Pass | Description |
|------|-------------|
| `air-dependency` | Build async dependency graph |
| `air-dependency-canonicalize` | Optimize dependency graph |
| `air-dependency-schedule-opt` | Scheduling optimizations (ping-pong, broadcast, BD opt) |
| `air-place-herds` | Place herds on device grid |
| `air-dma-to-channel` | Convert DMA ops to channels |
| `air-ping-pong-transform` | Apply ping-pong buffering |
| `air-broadcast-detection` | Detect broadcast patterns |
| `air-fuse-channels` | Fuse multiple channels |
| `air-linalg-codegen` | Tiling strategies for linalg ops |
| `air-automatic-tiling` | Automatic loop tiling |
| `air-loop-permutation` | Loop reordering |

### Key Conversion Passes

Located in `mlir/include/air/Conversion/Passes.td`:

| Pass | Description |
|------|-------------|
| `air-par-to-herd` | Convert parallel loops to `air.herd` |
| `air-par-to-launch` | Convert parallel loops to `air.launch` |
| `air-par-to-segment` | Convert parallel loops to `air.segment` |
| `air-copy-to-dma` | Convert memcpy to `air.dma_memcpy_nd` |
| `air-to-aie` | Convert AIR to AIE dialect |
| `air-to-std` | Lower AIR to standard dialects |
| `airrt-to-npu` | Convert AIRRt to NPU instructions |
| `airrt-to-llvm` | Convert AIRRt to LLVM dialect |

## Working with MLIR Files

### Applying Transformation Passes

```bash
# Run single pass
air-opt input.mlir -air-dependency -o output.mlir

# Chain multiple passes
air-opt input.mlir \
    -air-dependency \
    -air-dma-to-channel \
    -canonicalize \
    -cse \
    -o output.mlir

# AIR to AIE conversion
air-opt input.mlir -air-to-aie=device=npu1 -o output.mlir
```

### Key Pass Options

```bash
# air-to-aie options
-air-to-aie{device=npu1 row-offset=2 col-offset=0 emit-while-loop=false}

# air-place-herds options
-air-place-herds{num-rows=4 num-cols=4 row-anchor=2 col-anchor=0}

# air-dependency-canonicalize options
-air-dependency-canonicalize{dump-graph=true output-dir=/tmp/graphs}
```

### Using air-translate

```bash
# Generate JSON metadata
air-translate --airrt-generate-json input.mlir

# Translate to LLVM IR
air-translate --mlir-to-llvmir input.mlir -o output.ll
```

## Compiler Toolchain Components

### aircc.py - Compiler Driver

Main compilation driver that orchestrates the full pipeline:

```bash
# Basic compilation
aircc.py input.mlir -o output.xclbin --device=npu1

# With verbose output
aircc.py input.mlir -o output.xclbin -v --device=npu1

# Debug IR mode (saves IR after each pass)
aircc.py input.mlir -o output.xclbin --debug-ir --tmpdir=./debug
```

Key options:
- `--device=<target>`: Target device (npu1, npu2, xcvc1902)
- `--tmpdir=<dir>`: Temporary directory for intermediate files
- `--output-format=<fmt>`: Output format (xclbin, elf, txn, none)
- `--debug-ir`: Save IR after each pass for debugging
- `-v`: Verbose output
- `--trace-size=<bytes>`: Enable hardware tracing

### air-opt - MLIR Optimizer

Standard MLIR optimization driver with AIR passes:

```bash
air-opt --help  # List all available passes
air-opt input.mlir -pass-pipeline='builtin.module(...)' -o output.mlir
```

### air-runner - Simulation/Execution

Execute AIR programs in simulation:

```bash
air-runner input.mlir -f graph -m arch.json -g herd
```

Options:
- `-f <function>`: Top-level function name (default: "graph")
- `-m <json>`: JSON model file for architecture simulation
- `-g <granularity>`: Simulation level (herd or core)

## Python API

### Module Building with Decorators

```python
from air.ir import *
from air.dialects.air import *
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.func import FuncOp

@module_builder
def build_module():
    memref_type = MemRefType.get([16, 16], T.i32())

    @FuncOp.from_py_func(memref_type, memref_type)
    def my_kernel(input, output):

        @launch(operands=[input, output])
        def launch_body(a, b):

            @segment(name="seg", operands=[a, b])
            def segment_body(arg0, arg1):

                @herd(name="herd", sizes=[2, 2], operands=[arg0, arg1])
                def herd_body(tx, ty, sx, sy, a, b):
                    # Allocate L1 tile memory
                    mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
                    tile_type = MemRefType.get([8, 8], T.i32(), memory_space=mem_space)
                    tile = AllocOp(tile_type, [], [])

                    # DMA transfer
                    dma_memcpy_nd(tile, a,
                                  src_offsets=[0, 0],
                                  src_sizes=[8, 8],
                                  src_strides=[16, 1])

                    DeallocOp(tile)
```

### Key Python Classes

Located in `python/air/dialects/_air_ops_ext.py`:
- `Launch` - Create `air.launch` operations
- `Segment` - Create `air.segment` operations
- `Herd` - Create `air.herd` operations
- `Channel` - Create `air.channel` operations
- `ChannelPut` / `ChannelGet` - Channel data movement
- `dma_memcpy_nd()` - DMA transfer function

### XRT Runner

```python
from air.backend.xrt_runner import XRTRunner

mlir_module = build_module()
runner = XRTRunner(
    verbose=True,
    output_format="xclbin",
    instance_name="my_kernel"
)

exit_code = runner.run_test(
    mlir_module,
    inputs=[input_array],
    expected_outputs=[expected_output]
)
```

## Common Development Workflows

### Adding a New Transformation Pass

1. **Declare in TableGen** (`mlir/include/air/Transform/Passes.td`):
```tablegen
def AIRMyNewPass : Pass<"air-my-new-pass", "ModuleOp"> {
  let summary = "Description of the pass";
  let constructor = "xilinx::air::createAIRMyNewPass()";
  let options = [
    Option<"clOption", "option-name", "int", /*default=*/"0", "Description">
  ];
}
```

2. **Implement the pass** (`mlir/lib/Transform/AIRMyNewPass.cpp`)

3. **Add to CMakeLists.txt** (`mlir/lib/Transform/CMakeLists.txt`)

4. **Write tests** (`mlir/test/Transform/AIRMyNewPass/test.mlir`)

### Adding a New Conversion Pass

1. Declare in `mlir/include/air/Conversion/Passes.td`
2. Implement in `mlir/lib/Conversion/`
3. Register in conversion registration
4. Write conversion tests

### Debugging Compilation Issues

```bash
# Enable verbose output
aircc.py input.mlir -v --tmpdir=./debug -o output.xclbin

# Debug IR mode - saves IR after every pass
aircc.py input.mlir --debug-ir --tmpdir=./debug -o output.xclbin
# Check ./debug/debug_ir/ for pass_XXX_after_*.mlir files
# Check ./debug/debug_ir/pass.log for pass sequence

# Use air-opt with IR printing
air-opt input.mlir -mlir-print-ir-after-all -pass-pipeline='...'

# Examine intermediate files
cat ./debug/placed.air.mlir      # After placement
cat ./debug/aie.air.mlir         # After AIR-to-AIE
```

## Code Formatting

The project follows LLVM coding standards. **Always run formatters before committing:**

```bash
# Format C++ files
clang-format -i <file.cpp>

# Format all modified C++ files
git diff --name-only --diff-filter=d | grep -E '\.(cpp|h)$' | xargs -r clang-format -i

# Format Python files
black <file.py>

# Format all modified Python files
git diff --name-only --diff-filter=d | grep -E '\.py$' | xargs -r black
```

## Important Notes

### Memory Spaces

- **L3 (Memory Space 0)**: External DDR memory
- **L2 (Memory Space 1)**: Shared memory tiles (MemTile)
- **L1 (Memory Space 2)**: Local tile memory

### Device Targets

- `npu1` / `npu1_1col` / `npu1_2col` / `npu1_3col` / `npu1_4col` - Ryzen AI NPU (Phoenix)
- `npu2` / `npu2_4col` - Ryzen AI NPU (Strix)
- `xcvc1902` - Versal VC1902

### Common Pitfalls

1. **Memory space annotations**: L1 buffers must be allocated with `memory_space=2`
2. **DMA dimension alignment**: Sizes/strides must be compatible with hardware buffer descriptors
3. **Channel put/get pairing**: Every `air.channel.put` needs a matching `air.channel.get`
4. **Herd placement constraints**: Check device-specific tile availability
5. **N-buffer rotation DMA BD chains**: When using N-buffer rotation patterns (e.g., 4-buffer sliding window for convolutions with `omit_pingpong`), ensure `getRepeatCounts()` in `AIRToAIESchedulingUtils.cpp` correctly detects the rotation pattern. The `detectNBufferRotation()` function checks if multiple channel operations targeting different buffers belong to the same rotation pattern, ensuring circular BD chains are generated instead of terminated sequences.
6. **Golden reference scaling vs hardware output**: AIE convolution kernels (e.g., `conv2dk1_skip_i8`) output raw quantized `uint8` values. Do NOT apply quantization scale factors (e.g., `* inp_scale4`) inside the golden reference before casting to `uint8`. The mlir-aie reference `test.py` applies the scale AFTER reading from hardware (`out.numpy() * inp_scale4`), not inside the golden calculation. Applying it inside causes the golden to be half the hardware value (2x error).
7. **Programming example Makefiles**: Assume the user has already sourced `utils/env_setup.sh` (which sets `PYTHONPATH`, `PATH`, `PEANO_INSTALL_DIR`, etc.). Do NOT set `PYTHONPATH` or `PATH` in Makefiles -- use `AIEOPT_DIR = $(shell realpath $(dir $(shell which aie-opt))/..)` for include paths (relies on `aie-opt` being in `PATH`). Kernel sources from mlir-aie are at `utils/mlir-aie/aie_kernels/aie2/`.

### Useful Debug Flags

- `aircc.py --debug-ir` - Emit IR after each pass
- `air-opt -mlir-print-ir-after-all` - Print IR after all passes
- `-air-dependency-canonicalize{dump-graph=true}` - Dump dependency graphs as DOT files
- `make debug` in programming_examples directories - Dumps all intermediate IRs to `build_*/air_project/debug_ir/`
  - Files are named `pass_XXX_after_<pass-name>.mlir` in execution order
  - `pass.log` shows the pass sequence applied
  - The `pass_051*` level IR is comparable to mlir-aie objectfifo-level IR (`output.mlir`)

### Environment Variables

- `PEANO_INSTALL_DIR`: Path to Peano compiler installation
- `XRT_DIR`: Path to XRT installation
- `AIE_DIR`: Path to mlir-aie installation
