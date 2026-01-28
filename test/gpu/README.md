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

This is the fastest way to build MLIR-AIR for GPU targets using pre-built LLVM wheels.

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
source sandbox/bin/activate
source utils/env_setup.sh build llvm/install
```

## Build Script Details

The `build-mlir-air-gpu.sh` script builds MLIR-AIR with:
- `-DAIR_ENABLE_AIE=OFF` - Disables AIE backend dependency
- GPU/ROCDL passes enabled
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
mkdir -p build && cd build
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
| `air-to-rocdl` | Lower AIR dialect to ROCDL dialect |
| `air-to-async` | Lower AIR dialect to async dialect |
| `air-gpu-outlining` | Outline GPU kernels |
| `convert-to-air` | Convert operations to AIR dialect |

AIE-specific passes (e.g., `air-to-aie`) are registered but will emit an error if invoked, indicating that AIE support is required.

## Run GPU Tests

GPU tests are located under `test/gpu/`.

### Example: Running a basic GPU test

```bash
# From the mlir-air directory
cd test/gpu

# Lower AIR to ROCDL
air-opt air_sync.mlir -air-to-rocdl -o output_rocdl.mlir

# Outline GPU kernels
air-opt output_rocdl.mlir -air-gpu-outlining -o output_outlined.mlir

# Continue with LLVM's GPU lowering pipeline
mlir-opt output_outlined.mlir \
    --pass-pipeline="builtin.module(func.func(lower-affine, convert-linalg-to-loops, convert-scf-to-cf), gpu-kernel-outlining)" \
    -o output_gpu.mlir

# Target MI300X (gfx942)
mlir-opt output_gpu.mlir \
    --pass-pipeline="builtin.module(rocdl-attach-target{chip=gfx942 O=3}, gpu.module(convert-gpu-to-rocdl{chipset=gfx942 runtime=HIP}, reconcile-unrealized-casts), gpu-module-to-binary, func.func(gpu-async-region), gpu-to-llvm, convert-to-llvm, reconcile-unrealized-casts)" \
    -o output_llvm.mlir

# Run with ROCm runtime
mlir-cpu-runner \
    --entry-point-result=void \
    --shared-libs=$LLVM_INSTALL/lib/libmlir_rocm_runtime.so \
    output_llvm.mlir
```

## Environment Setup

To reactivate the environment from a new terminal:

```bash
cd mlir-air
source sandbox/bin/activate
source utils/env_setup.sh build llvm/install
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
# Follow the full build instructions in the main README
./utils/clone-mlir-aie.sh
./utils/build-mlir-aie-local.sh llvm mlir-aie/cmake/modulesXilinx aienginev2/install mlir-aie
./utils/build-mlir-air.sh llvm mlir-aie/cmake/modulesXilinx mlir-aie aienginev2/install
```
