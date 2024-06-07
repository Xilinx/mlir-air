# Getting Started and Running on Linux Ryzen™ AI

## Environment

The [MLIR-AIE](https://github.com/Xilinx/mlir-aie) repo maintains instructions on how to install dependencies and configure your environment. Follow the instructions [here](https://github.com/Xilinx/mlir-aie/blob/main/docs/buildHostLin.md). It is not necessary to follow the final steps for cloning/building/running MLIR-AIE itself.

## Prerequisites

Building MLIR-AIR several other open source packages:
  - [mlir](https://github.com/llvm/llvm-project/tree/main/mlir)
  - [mlir-aie](https://github.com/Xilinx/mlir-aie)
  - [Xilinx cmakeModules](https://github.com/Xilinx/cmakeModules). Note that this is installed as a submodule of MLIR-AIE
  - [libXAIE](https://github.com/jnider/aie-rt.git)

These prerequisitives can be installed with some helpful scripts found in the ```utils``` directory in the process described below.

First, clone the MLIR-AIR repo:
```bash
git clone https://github.com/Xilinx/mlir-air.git
cd mlir-air
```

Next, run ```utils/setup_python_packages.sh``` to setup the prerequisite python packages. This script creates and installs the python packages listed in [utils/requirements.txt](utils/requirements.txt) in a virtual python environment called ```sandbox```.

```bash
source utils/setup_python_packages.sh
```

Next, clone and build LLVM, with MLIR enabled. In addition, we make some common build optimizations to use a linker ('lld' or 'gold') other than 'ld' (which tends to be quite slow on large link jobs) and to link against libLLVM.so and libClang so. You may find that other options are also useful. Note that due to changing MLIR APIs, only a particular revision is expected to work.

Run the following to clone and build llvm:

```bash
./utils/clone-llvm.sh
./utils/build-llvm-local.sh llvm
```

Next, clone and build the aienginev2 module.
```bash
./utils/github-clone-build-libxaie.sh
```

Next, clone and build MLIR-AIE with paths to llvm, aienginev2, and cmakeModules repositories.
```bash
./utils/clone-mlir-aie.sh
./utils/build-mlir-aie-local.sh llvm mlir-aie/cmake/modulesXilinx aienginev2 mlir-aie
```

After this step, you are ready to build MLIR-AIR!

## Building

To build MLIR-AIR provide the paths to llvm, cmakeMoudles, and xrt (here, we assume it is installed in ```/opt/xilinx/xrt```):
```bash
./utils/build-mlir-air-xrt.sh llvm mlir-aie/cmake/modulesXilinx mlir-aie aienginev2 /opt/xilinx/xrt
```

## Environment

To setup your environment after building:
```bash
source utils/env_setup.sh install-xrt/ mlir-aie/install/ llvm/install/
```

Note that if you are starting a new enviroment (say be creating a new terminal sometime after building), restore your environment with:
```bash
source utils/env_setup.sh install-xrt/ mlir-aie/install/ llvm/install/
source sandbox/bin/activate
```

## Testing

Some tests for MLIR-AIR are provided. Run them as demonstrated below:

```bash
cd mlir-air/build-xrt
ninja install
ninja check-air-cpp
ninja check-air-mlir
ninja check-air-python
 
# These are the ones in test/xrt, and this is about equivalent to `ninja check-air-e2e` if you set the LIT_OPS env var appropriately
lit -sv --time-tests --show-unsupported --show-excluded  --timeout 600 -j5 test/xrt
 
# Run an individual test
lit -sv test/xrt/01_air_to_npu
```

-----

<p align="center">Copyright&copy; 2022 Advanced Micro Devices, Inc.</p>
