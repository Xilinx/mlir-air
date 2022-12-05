# Building AIR Tools and Runtime

## Prerequisites

**mlir-air** is built and tested on Ubuntu 20.04 with the following packages installed:
```
  Xilinx Vitis 2021.2
  clang/llvm 10
  python 3.8.x
  ninja
  cmake
```
Building mlir-air requires several other open source packages:
  - [mlir](https://github.com/llvm/llvm-project/tree/main/mlir)
  - [mlir-aie](https://github.com/Xilinx/mlir-aie)
  - [Xilinx cmakeModules](https://github.com/Xilinx/cmakeModules)
  - [Xilinx embeddedsw](https://github.com/Xilinx/embeddedsw)

## Building external projects on X86

This mlir-air repository should already be cloned locally. 

First, clone and build LLVM, with the ability to target AArch64 as a cross-compiler, and with MLIR enabled. In addition, we make some common build optimizations to use a linker ('lld' or 'gold') other than 'ld' (which tends to be quite slow on large link jobs) and to link against libLLVM.so and libClang so. You may find that other options are also useful. Note that due to changing MLIR APIs, only a particular revision is expected to work.

To clone llvm and cmakeModules, see utils/clone-llvm.sh for the correct commithash. We point LLVM and subsequent tools to a common installation directory. 

```
cd utils
./clone-llvm.sh
./build-llvm-local.sh llvm build ../install
```

Next, clone and build MLIR-AIE with absolute paths to the sysroot, llvm, and cmakeModules repositories. Again, we use a common installation directory.

```
git clone https://github.com/Xilinx/cmakeModules.git
./clone-mlir-aie.sh
./build-mlir-aie-local.sh $SYSROOT /full/path/to/mlir-air/utils/llvm /full/path/to/mlir-air/utils/cmakeModules mlir-aie build ../install
```

The MLIR-AIE tools will be able to generate binaries targetting AIEngines.

Finally, build the MLIR-AIR tools for your desired use case: 

- Building on x86
- Building on x86 for deployment on a PCIe platform with AIEs
- Building on x86 for deployment on an embedded platform with AIEs
- Building on ARM for deployment on an embedded platform with AIEs

## Building on x86

```
./build-mlir-air.sh $SYSROOT /full/path/to/mlir-air/utils/llvm /full/path/to/mlir-air/utils/cmakeModules /full/path/to/mlir-air/utils/mlir-aie ../../mlir-air build utils/install
```

## Building on x86 with runtime for PCIe 

```
./build-mlir-air-pcie.sh /full/path/to/mlir-air/utils/llvm /full/path/to/mlir-air/utils/cmakeModules /full/path/to/mlir-air/utils/mlir-aie ../../mlir-air build utils/install
```

## Environment setup

```
export PATH=/path/to/install/bin:${PATH}
export PYTHONPATH=/path/to/install/python:${PYTHONPATH}
export LD_LIBRARY_PATH=/path/to/install/lib:/opt/xaiengine/lib:${LD_LIBRARY_PATH}
```

## Building a Sysroot

## Compiling Runtime for AArch64 (partial cross-compile)

## Compiling Tools and Runtime for AArch64 (full cross-compile)


-----

<p align="center">Copyright&copy; 2022 Advanced Micro Devices, Inc.</p>
