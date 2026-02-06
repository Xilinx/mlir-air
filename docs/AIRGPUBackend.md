# Build MLIR-AIR for GPU Target

This guide provides instructions for building MLIR-AIR for GPU targets without AIE dependencies. Tested on MI300X.

## Prerequisites

- ROCm installation (tested with ROCm 6.x)
- CMake 3.20+
- Ninja build system
- Python 3.8+

### Cluster Access (Optional)

If using the OCI MI300X cluster:

```bash
salloc -p amd-arad -N 1 --gres=gpu:2 -t 0-1
srun --pty $SHELL
bash
```

## Quick Build (Recommended)

This is the fastest way to build MLIR-AIR for GPU targets.

```bash
# Clone the repository
git clone https://github.com/Xilinx/mlir-air.git
cd mlir-air

# Setup Python environment
source utils/setup_python_packages.sh

# Build LLVM
./utils/clone-llvm.sh
./utils/build-llvm-local.sh llvm

# Build MLIR-AIR for GPU (without AIE)
./utils/build-mlir-air-gpu.sh llvm

# Setup environment
source utils/env_setup_gpu.sh install
```

## Build Script Details

The `build-mlir-air-gpu.sh` script builds MLIR-AIR with:
- `-DAIR_ENABLE_AIE=OFF` - Disables AIE backend dependency
- `-DAIR_ENABLE_GPU=ON` - Enables GPU/ROCDL passes
- AIR core dialect and transformations

## Manual Build

For more control over the build process:

```bash
# Clone and setup
git clone https://github.com/Xilinx/mlir-air.git
cd mlir-air
source utils/setup_python_packages.sh

# Build LLVM
./utils/clone-llvm.sh
./utils/build-llvm-local.sh llvm

# Configure MLIR-AIR for GPU
mkdir -p build_gpu && cd build_gpu
cmake .. \
    -GNinja \
    -DMLIR_DIR=$(pwd)/../llvm/install/lib/cmake/mlir \
    -DLLVM_DIR=$(pwd)/../llvm/install/lib/cmake/llvm \
    -DAIR_ENABLE_AIE=OFF \
    -DAIR_ENABLE_GPU=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$(pwd)/install

# Build and install
ninja install
```

## Available Passes

With GPU-only build, the following passes are available:

| Pass | Description |
|------|-------------|
| `air-to-rocdl` | Lower AIR dialect to GPU/ROCDL dialect |
| `air-gpu-outlining` | Outline GPU kernels into gpu.module |
| `air-to-async` | Lower AIR dialect to async dialect |
| `convert-to-air` | Convert operations to AIR dialect |

AIE-specific passes (e.g., `air-to-aie`) are registered but will emit an error if invoked, indicating that AIE support is required.

## GPU Compilation with aircc.py

The `aircc.py` compiler supports GPU targets.

### Quick Start

```bash
# Setup environment first
source utils/env_setup_gpu.sh install

# Compile the 4k x 4k matrix multiplication example for MI300X
aircc.py --target gpu --gpu-arch gfx942 -o output.mlir test/gpu/4k_4k_mul/air_sync.mlir

# With verbose output to see compilation steps
aircc.py --target gpu --gpu-arch gfx942 -v -o output.mlir test/gpu/4k_4k_mul/air_sync.mlir

# Keep intermediate files for debugging
aircc.py --target gpu --gpu-arch gfx942 -v --tmpdir /tmp/mytest -o output.mlir test/gpu/4k_4k_mul/air_sync.mlir
```

### aircc.py Options for GPU

| Option | Default | Description |
|--------|---------|-------------|
| `--target` | `aie` | Target backend: `aie` or `gpu` |
| `--gpu-arch` | `gfx942` | GPU architecture |
| `--gpu-runtime` | `HIP` | GPU runtime: `HIP` or `OpenCL` |
| `-o <file>` | stdout | Output file |
| `--tmpdir <dir>` | auto | Directory for intermediate files |
| `-v` | off | Show compilation steps |

### Supported GPU Architectures

| Architecture | GPU |
|--------------|-----|
| `gfx942` | MI300X, MI300A |
| `gfx90a` | MI200 series |
| `gfx908` | MI100 |
| `gfx1100` | RDNA3 (RX 7900) |

### Compilation Pipeline

The aircc.py GPU pipeline runs the following passes:

1. **AIR to ROCDL** (`air-opt -air-to-rocdl`)
   - Converts `air.launch`, `air.segment`, `air.herd` → `gpu.launch`
   - Converts `air.dma_memcpy_nd` → memory operations

2. **GPU Kernel Outlining** (`air-opt -air-gpu-outlining`)
   - Outlines GPU kernels into `gpu.module`
   - Adds `gpu.container_module` and `gpu.kernel` attributes

3. **LLVM Lowering** (`mlir-opt`)
   - Lowers affine, scf, cf dialects
   - Runs `gpu-kernel-outlining`

4. **ROCDL Binary Generation** (`mlir-opt`)
   - Attaches ROCDL target with `rocdl-attach-target`
   - Converts GPU dialect to ROCDL
   - Generates embedded GPU binary with `gpu-module-to-binary`
   - Final lowering to LLVM dialect

## GPU Test Examples

### 4k x 4k Matrix Multiplication

The `test/gpu/4k_4k_mul/` directory contains a matrix multiplication example.

```bash
# Setup environment
source utils/env_setup_gpu.sh install

# Compile the example
aircc.py --target gpu --gpu-arch gfx942 -v \
    --tmpdir /tmp/matmul \
    -o /tmp/matmul/output.mlir \
    test/gpu/4k_4k_mul/air_sync.mlir

# View the generated output (contains gpu.binary)
head -50 /tmp/matmul/output.mlir

# View intermediate files
ls /tmp/matmul/
```

### Output Structure

After compilation, the output MLIR contains:
- `gpu.binary` with embedded AMDGPU ELF binary
- `gpu.launch_func` calls to invoke the kernel
- Host-side LLVM code for memory management and kernel launch

### Running the Compiled Output

Use `mlir-runner` with the ROCm runtime library to execute the compiled MLIR:

```bash
# Setup environment
source utils/env_setup_gpu.sh install

# Run the compiled output
mlir-runner --entry-point-result=void \
    --shared-libs=$LLVM_INSTALL_DIR/lib/libmlir_rocm_runtime.so \
    output.mlir
```

For debugging with ISA output:

```bash
mlir-runner --debug-only=serialize-to-isa \
    --entry-point-result=void \
    --shared-libs=$LLVM_INSTALL_DIR/lib/libmlir_rocm_runtime.so \
    output.mlir
```

Full example (compile and run):

```bash
# Setup environment
source utils/env_setup_gpu.sh install

# Compile
aircc.py --target gpu --gpu-arch gfx942 \
    -o /tmp/output.mlir \
    test/gpu/4k_4k_mul/air_sync.mlir

# Run
mlir-runner --entry-point-result=void \
    --shared-libs=$LLVM_INSTALL_DIR/lib/libmlir_rocm_runtime.so \
    /tmp/output.mlir
```

## Manual Compilation Steps

For debugging or customization, you can run the passes manually:

### Step 1: AIR to ROCDL

```bash
air-opt test/gpu/4k_4k_mul/air_sync.mlir \
    -air-to-rocdl \
    -o step1_rocdl.mlir
```

### Step 2: GPU Kernel Outlining

```bash
air-opt step1_rocdl.mlir \
    -air-gpu-outlining \
    -o step2_outlined.mlir
```

### Step 3: LLVM Lowering

```bash
mlir-opt step2_outlined.mlir \
    --pass-pipeline="builtin.module(func.func(lower-affine,convert-linalg-to-loops,convert-scf-to-cf),gpu-kernel-outlining)" \
    -o step3_gpu.mlir
```

### Step 4: ROCDL Binary Generation

```bash
mlir-opt step3_gpu.mlir \
    --pass-pipeline="builtin.module(rocdl-attach-target{chip=gfx942 O=3},gpu.module(convert-gpu-to-rocdl{chipset=gfx942 runtime=HIP},reconcile-unrealized-casts),gpu-module-to-binary,func.func(gpu-async-region),gpu-to-llvm,convert-to-llvm,reconcile-unrealized-casts)" \
    -o step4_final.mlir
```

### Step 5: Run with mlir-runner

```bash
mlir-runner --entry-point-result=void \
    --shared-libs=$LLVM_INSTALL_DIR/lib/libmlir_rocm_runtime.so \
    step4_final.mlir
```

## Environment Setup

To reactivate the environment from a new terminal:

```bash
cd mlir-air
source utils/env_setup_gpu.sh install
```

Or manually:

```bash
export PATH=/path/to/mlir-air/install/bin:/path/to/llvm/install/bin:$PATH
export PYTHONPATH=/path/to/mlir-air/python:$PYTHONPATH
```

## Troubleshooting

### Missing Python Bindings

If you see:
```
ModuleNotFoundError: No module named 'air'
```

Make sure to source the environment setup:
```bash
source utils/env_setup_gpu.sh install
```

Or add the Python path manually:
```bash
export PYTHONPATH=/path/to/mlir-air/python:$PYTHONPATH
```

### AIE Pass Errors

If you see errors like:
```
error: AIRToAIE pass requires AIE support. Rebuild with -DAIR_ENABLE_AIE=ON
```

This is expected behavior. The GPU-only build does not include AIE backend support. Use `--target gpu` with aircc.py for GPU targets.

### Missing air-opt or mlir-opt

Ensure the tools are in your PATH:
```bash
which air-opt mlir-opt
```

If not found, source the environment:
```bash
source utils/env_setup_gpu.sh install
```

### ROCm Runtime Not Found

Ensure ROCm is installed and the runtime library path is correct:
```bash
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
```

### Named Barrier Error

If you see:
```
error: cannot evaluate equated symbol 'air_kernel_0.num_named_barrier'
```

This is fixed by using `wave64=true` in the rocdl-attach-target pass, which is the default in aircc.py.

## Building with Both AIE and GPU Support

If you need both GPU and AIE backends:

```bash
cmake .. \
    -GNinja \
    -DMLIR_DIR=/path/to/llvm/install/lib/cmake/mlir \
    -DLLVM_DIR=/path/to/llvm/install/lib/cmake/llvm \
    -DAIE_DIR=/path/to/mlir-aie/install/lib/cmake/aie \
    -DAIR_ENABLE_AIE=ON \
    -DAIR_ENABLE_GPU=ON \
    -DCMAKE_BUILD_TYPE=Release
```

## Backend Options Summary

| Option | Default | Description |
|--------|---------|-------------|
| `AIR_ENABLE_AIE` | ON | Enable AIE backend (requires mlir-aie) |
| `AIR_ENABLE_GPU` | ON | Enable GPU backend (ROCDL/HIP) |

Build configurations:
- **GPU-only**: `-DAIR_ENABLE_AIE=OFF -DAIR_ENABLE_GPU=ON`
- **AIE-only**: `-DAIR_ENABLE_AIE=ON -DAIR_ENABLE_GPU=OFF`
- **Both**: `-DAIR_ENABLE_AIE=ON -DAIR_ENABLE_GPU=ON` (requires mlir-aie)
