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

## Building dependencies on X86

```
cd utils
git clone https://github.com/Xilinx/cmakeModules.git
./clone-llvm.sh
./build-llvm-local.sh llvm build ../install
./clone-mlir-aie.sh
./build-mlir-aie-local.sh $SYSROOT /full/path/to/mlir-air/utils/llvm /full/path/to/mlir-air/utils/cmakeModules mlir-aie build ../install
```
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
