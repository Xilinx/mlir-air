# Getting Started and Running on VCK5000

The first step required for running on the VCK5000 is building the hardware platform and loading it on your device. We provide a platform repository titled [ROCm-air-platforms](https://github.com/Xilinx/ROCm-air-platforms) which contains instructions on how to compile and load the platform, driver, and firmware on your VCK5000. Then, the AIR tools must be installed by following the [Building external projects on X86](#building-external-projects-on-X86) and [Environment setup](#environment-setup) instructions. Then, the [AIR PCIe driver](https://github.com/Xilinx/ROCm-air-platforms/tree/main/driver) from [ROCm-air-platforms](https://github.com/Xilinx/ROCm-air-platforms) must be compiled and installed. Now you are ready to run tests using AIR on the VCK5000! For example, go to [Test 13](test/13_mb_add_one) and run `make` followed by `sudo ./test.elf` to run the application.


## Prerequisites

**mlir-air** is built and tested on Ubuntu 20.04 with the following packages installed:
```
  cmake 3.20.6
  clang/llvm 10+
  lld
  python 3.8.x
  ninja 1.10.0
  Xilinx Vitis 2021.2 
  libelf
```

In addition the following packages maybe useful: 

```

  Xilinx Vivado 2021.2/2022.1
  Xilinx Vitis 2021.2/2022.1
  Xilinx aienginev2 library from https://github.com/Xilinx/embeddedsw

```
Vivado and Vitis are used to build the platforms for available Xilinx cards with AIEs (VCK5000). Vitis also contains the AIE single core compiler xchesscc. The aienginev2 library is a driver backend used to configure the AIE array and used in building the AIR runtime. The aietools single core compiler and aienginev2 library are required to build binaries that run on AIE hardware.

NOTE: using the Vitis recommended settings64.sh script to set up your environement can cause tool conflicts. Setup your environment in the following order for aietools and Vitis:

```
export PATH=$PATH:<Vitis_install_path>/Vitis/2022.2/aietools/bin:<Vitis_install_path>/Vitis/2022.2/bin
```

NOTE: that you do not need the above additional Xilinx packages to make use of the AIR compiler passes. 

Building mlir-air requires several other open source packages:
  - [mlir](https://github.com/llvm/llvm-project/tree/main/mlir)
  - [mlir-aie](https://github.com/Xilinx/mlir-aie)
  - [Xilinx cmakeModules](https://github.com/Xilinx/cmakeModules). Note that this is installed as a submodule of mlir-aie.

In order to build and run on PCIe cards, you first have to build and install the aienginev2 library. We chose to install the library in /opt/xaiengine but it is not required for the tools to be installed there. Just ensure that when building mlir-aie and mlir-air, that you point to the directory in which the aienginev2 library was installed.

```
git clone https://github.com/stephenneuendorffer/aie-rt
cd aie-rt
git checkout phoenix_v2023.2
cd driver/src
make -f Makefile.Linux CFLAGS="-D__AIEAMDAIR__"
sudo cp -r ../include /opt/aiengine/
sudo cp libxaiengine.so* /opt/xaiengine/lib/
export LD_LIBRARY_PATH=/opt/xaiengine/lib:${LD_LIBRARY_PATH}
```

## Building external projects on X86

The mlir-air repository should be cloned locally before beginning. 

First, run utils/setup_python_packages.sh to setup the prerequisite python packages. This script creates and installs the python packages listed in utils/requirements.txt in a virtual python environment called 'sandbox'.

```
source utils/setup_python_packages.sh
```

Next, clone and build LLVM, with MLIR enabled. In addition, we make some common build optimizations to use a linker ('lld' or 'gold') other than 'ld' (which tends to be quite slow on large link jobs) and to link against libLLVM.so and libClang so. You may find that other options are also useful. Note that due to changing MLIR APIs, only a particular revision is expected to work.

Run the following to clone and build llvm:

```
cd mlir-air
./utils/clone-llvm.sh
./utils/build-llvm-local.sh llvm
```

Next, clone and build MLIR-AIE with paths to llvm, aienginev2, and cmakeModules repositories. Note that in the following commands, we assume that the aienginev2 library is installed in /opt/xaiengine as directed in the `Prerequisites` section. If the aienginev2 library was installed elsewhere, be sure that the 4th argument to build mlir-aie points to that location. 

```
./utils/clone-mlir-aie.sh
./utils/build-mlir-aie-local.sh llvm mlir-aie/cmake/modulesXilinx /opt/xaiengine mlir-aie
```

Next, we must install our experimental ROCm runtime (ROCr) which allows us to communicate with the AIEs. The [ROCm-air-platforms](https://github.com/Xilinx/ROCm-air-platforms) repository contains documentation on how to install ROCr. Run the following script to clone the ROCm-air-platforms repository:

```
./utils/clone-rocm-air-platforms.sh
```

Use the following command to build the AIR tools to compile on x86 for PCIe cards (VCK5000). Make sure that ${ROCM\_ROOT} is pointing to your local ROCm install:

```
./utils/build-mlir-air-pcie.sh llvm/ mlir-aie/cmake/modulesXilinx/ mlir-aie/ /opt/xaiengine ${ROCM_ROOT}/lib/cmake/hsa-runtime64/ ${ROCM_ROOT}/lib/cmake/hsakmt/
```

The PCIe AIR runtime requires the use of the [AIR PCIe kernel driver](https://github.com/Xilinx/ROCm-air-platforms/tree/main/driver). The driver directory in the [ROCm-air-platforms](https://github.com/Xilinx/ROCm-air-platforms) repository contains documentation on how to compile and load the AIR PCIe kernel driver. 

## Environment setup

Set up your environment to use the tools you just built with the following commands:

```
export PATH=/path/to/mlir-air/install/bin:${PATH}
export PYTHONPATH=/path/to/mlir-air/install/python:${PYTHONPATH}
export LD_LIBRARY_PATH=/path/to/install/mlir-air/lib:/opt/xaiengine/lib:${LD_LIBRARY_PATH}
export PATH=/path/to/mlir-air/install-pcie/bin:${PATH}
```

-----

<p align="center">Copyright&copy; 2022 Advanced Micro Devices, Inc.</p>
