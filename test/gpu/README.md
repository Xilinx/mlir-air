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
./utils/build-mlir-air-gpu.sh llvm/install build_gpu install_gpu

# Setup environment
source sandbox/bin/activate
source utils/env_setup.sh build_gpu llvm/install
```

## Build Script Details

The `build-mlir-air-gpu.sh` script builds MLIR-AIR with:
- `-DAIR_ENABLE_AIE=OFF` - Disables AIE backend dependency
- GPU/ROCDL passes enabled
- AIR core dialect and transformations

**Usage:**
```bash
./utils/build-mlir-air-gpu.sh <llvm_install_dir> [build_dir] [install_dir]

# Examples:
./utils/build-mlir-air-gpu.sh llvm/install                      # Uses default build/ and install/
./utils/build-mlir-air-gpu.sh llvm/install build_gpu install_gpu  # Custom directories
```

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
    -DCMAKE_BUILD_TYPE=Release

# Build
ninja
```

## Available Passes

With GPU-only build, the following passes are available:

| Pass | Description |
|------|-------------|
| `air-to-rocdl` | Lower AIR dialect to GPU/ROCDL dialect |
| `air-gpu-outlining` | Outline GPU kernels from gpu.launch operations |
| `air-to-async` | Lower AIR dialect to async dialect |
| `convert-to-air` | Convert operations to AIR dialect |

AIE-specific passes (e.g., `air-to-aie`) are registered but will emit an error if invoked, indicating that AIE support is required.

## Run GPU Tests

GPU tests are located under `test/gpu/`.

### Test 1: Verify Passes are Available

```bash
# Check GPU passes are registered
air-opt --help | grep -E "air-to-rocdl|air-gpu-outlining"
```

Expected output:
```
--air-gpu-outlining    - Outline GPU kernels from gpu.launch operations
--air-to-rocdl         - Lower AIR dialect to GPU/ROCDL dialect for AMD GPUs
```

### Test 2: Run Simple AIR to ROCDL Lowering

```bash
cd test/gpu

# Lower AIR dialect to GPU dialect
air-opt simple_test.mlir --air-to-rocdl -o output_gpu.mlir

# View the output
cat output_gpu.mlir
```

### Test 3: Full GPU Pipeline (MI300X)

```bash
cd test/gpu

# Step 1: Lower AIR to GPU
air-opt simple_test.mlir --air-to-rocdl -o step1_gpu.mlir

# Step 2: Outline GPU kernels
air-opt step1_gpu.mlir --air-gpu-outlining -o step2_outlined.mlir

# Step 3: Lower to LLVM (using mlir-opt from LLVM)
mlir-opt step2_outlined.mlir \
    --pass-pipeline="builtin.module(func.func(lower-affine, convert-linalg-to-loops, convert-scf-to-cf), gpu-kernel-outlining)" \
    -o step3_lowered.mlir

# Step 4: Target MI300X (gfx942) and generate binary
mlir-opt step3_lowered.mlir \
    --pass-pipeline="builtin.module(rocdl-attach-target{chip=gfx942 O=3}, gpu.module(convert-gpu-to-rocdl{chipset=gfx942 runtime=HIP}, reconcile-unrealized-casts), gpu-module-to-binary, func.func(gpu-async-region), gpu-to-llvm, convert-to-llvm, reconcile-unrealized-casts)" \
    -o step4_llvm.mlir

# Step 5: Run with ROCm runtime (requires GPU hardware)
mlir-cpu-runner \
    --entry-point-result=void \
    --shared-libs=$LLVM_INSTALL/lib/libmlir_rocm_runtime.so \
    step4_llvm.mlir
```

### Test 4: Verify AIE Stubs Work

```bash
# This should produce an error message (expected behavior)
echo 'module {}' | air-opt --air-to-aie
```

Expected output:
```
error: AIRToAIE pass requires AIE support. Rebuild with -DAIR_ENABLE_AIE=ON
```

## Environment Setup

To reactivate the environment from a new terminal:

```bash
cd mlir-air
source sandbox/bin/activate
source utils/env_setup.sh build_gpu llvm/install
```

## Troubleshooting

### Missing OpenSSL

If you encounter OpenSSL errors during CMake or LLVM build:

```bash
# Install OpenSSL locally
wget https://github.com/openssl/openssl/releases/download/openssl-3.5.0/openssl-3.5.0.tar.gz
tar zxvf openssl-3.5.0.tar.gz
cd openssl-3.5.0
./config --prefix=$HOME/openssl --openssldir=$HOME/openssl no-ssl2
make && make install

# Add to environment
export PATH=$HOME/openssl/bin:$PATH
export LD_LIBRARY_PATH=$HOME/openssl/lib:$LD_LIBRARY_PATH
export OPENSSL_ROOT_DIR=$HOME/openssl
```

### AIE Pass Errors

If you see errors like:
```
error: AIRToAIE pass requires AIE support. Rebuild with -DAIR_ENABLE_AIE=ON
```

This is expected behavior. The GPU-only build does not include AIE backend support. Use `air-to-rocdl` instead of `air-to-aie` for GPU targets.

### ROCm Runtime Not Found

Ensure ROCm is installed and the runtime library path is correct:
```bash
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
```

## Building with AIE Support

If you need both GPU and AIE backends, build with AIE enabled:

```bash
# Build LLVM (same as GPU build)
./utils/clone-llvm.sh
./utils/build-llvm-local.sh llvm

# Clone and build mlir-aie
./utils/clone-mlir-aie.sh
./utils/build-mlir-aie-local.sh llvm/install mlir-aie/cmake/modulesXilinx NONE mlir-aie

# Build mlir-air with AIE support
mkdir -p build_aie && cd build_aie
cmake .. \
    -GNinja \
    -DMLIR_DIR=$(pwd)/../llvm/install/lib/cmake/mlir \
    -DLLVM_DIR=$(pwd)/../llvm/install/lib/cmake/llvm \
    -DAIE_DIR=$(pwd)/../mlir-aie/install/lib/cmake/aie \
    -DAIR_ENABLE_AIE=ON \
    -DCMAKE_BUILD_TYPE=Release

ninja
```
