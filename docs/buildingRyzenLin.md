# Getting Started and Running on Linux Ryzen™ AI

## Quick Start: Build with Prebuilt Wheels

### Prerequisites

- **Python 3.10+** (required by the wheels)
- **gcc >= 11**
- **pip** (Python package manager)
- **XRT** installed (you must provide the path to your XRT installation)
- (Optional but recommended) A Python virtual environment

### Steps

1. **Clone the MLIR-AIR repository:**
   ```bash
   git clone https://github.com/Xilinx/mlir-air.git
   cd mlir-air
   ```

2. **(Optional) Set up a Python virtual environment:**
   ```bash
   python3 -m venv sandbox
   source sandbox/bin/activate
   ```

3. **Run the build script:**
   ```bash
   utils/build-mlir-air-using-wheels.sh <xrt_dir> [build_dir] [install_dir]
   ```
   - `<xrt_dir>`: Path to your XRT installation (required)
   - `[build_dir]`: Build directory (optional, default: `build`)
   - `[install_dir]`: Install directory (optional, default: `install`)

   The script will:
   - Download and unpack a prebuilt LLVM wheel (which includes MLIR)
   - pip install [`llvm-aie`](https://github.com/Xilinx/llvm-aie), which is used as a backend to generate AIE binaries
   - Install `mlir-aie` dependencies from wheels
   - Clone required CMake modules
   - Configure and build MLIR-AIR using CMake and Ninja

4. **Environment Setup:**
   The script sets up environment variables for MLIR-AIE, Python, and libraries.  
   If you start a new terminal, you may need to re-source your Python environment and set variables as needed.

5. **Testing:**
   After building, you can run tests as follows:
   ```bash
   cd <build_dir>   # default is 'build'
   ninja install
   ninja check-air-cpp
   ninja check-air-mlir
   ninja check-air-python

   # Run LIT tests (set -DLLVM_EXTERNAL_LIT if needed)
   lit -sv --time-tests --show-unsupported --show-excluded --timeout 600 -j5 test/xrt

   # Run an individual test
   lit -sv test/xrt/01_air_to_npu

   # Run all xrt tests on device (may take a long time)
   ninja check-air-e2e-peano
   ```

### Notes

- The script expects Python 3.10+ and gcc >= 11.
- For LIT tests, you may need to set `-DLLVM_EXTERNAL_LIT` to the path of your `lit` executable.
- The script installs dependencies using pip and downloads wheels from the official release pages.
- For advanced troubleshooting or custom builds, see the legacy instructions below.

---

## Manual Build (Advanced/Legacy)

The following instructions describe the manual, source-based build process. This is generally not required unless you need to build from source for development or debugging.

### Environment

The [MLIR-AIE](https://github.com/Xilinx/mlir-aie) repo maintains instructions on how to install dependencies and configure your environment. Follow the instructions [here](https://github.com/Xilinx/mlir-aie/blob/main/docs/buildHostLin.md). It is not necessary to follow the final steps for cloning/building/running MLIR-AIE itself.

### Prerequisites

Building MLIR-AIR requires several other open source packages:
  - [mlir](https://github.com/llvm/llvm-project/tree/main/mlir)
  - [mlir-aie](https://github.com/Xilinx/mlir-aie)
  - [Xilinx cmakeModules](https://github.com/Xilinx/cmakeModules). Note that this is installed as a submodule of MLIR-AIE
  - [libXAIE](https://github.com/jnider/aie-rt.git)

These prerequisites can be installed with some helpful scripts found in the `utils` directory in the process described below.

First, clone the MLIR-AIR repo:
```bash
git clone https://github.com/Xilinx/mlir-air.git
cd mlir-air
```

Next, run `utils/setup_python_packages.sh` to setup the prerequisite python packages. This script creates and installs the python packages listed in [utils/requirements.txt](https://github.com/Xilinx/mlir-air/blob/main/utils/requirements.txt) in a virtual python environment called `sandbox`.

```bash
source utils/setup_python_packages.sh
```

Next, clone and build LLVM, with MLIR enabled. In addition, we make some common build optimizations to use a linker ('lld' or 'gold') other than 'ld' (which tends to be quite slow on large link jobs) and to link against libLLVM.so and libClang.so. You may find that other options are also useful. Note that due to changing MLIR APIs, only a particular revision is expected to work.

Run the following to clone and build llvm:

```bash
./utils/clone-llvm.sh
./utils/build-llvm-local.sh llvm
```

Next, clone and build the aienginev2 module. The installed files should be generated under `aienginev2/install`.
```bash
./utils/github-clone-build-libxaie.sh
```

Next, clone and build MLIR-AIE with paths to llvm, aienginev2, and cmakeModules repositories.
MLIR-AIE requires some dependent packages to be installed.
For details on the MLIR-AIE prerequisites, please refer to the MLIR-AIE [repository](https://github.com/Xilinx/mlir-aie?tab=readme-ov-file#prerequisites).
Once the prerequisites are set up, run the following commands to build MLIR-AIE.
```bash
./utils/clone-mlir-aie.sh
./utils/build-mlir-aie-local.sh llvm mlir-aie/cmake/modulesXilinx aienginev2/install mlir-aie
```

After this step, you are ready to build MLIR-AIR!

### Building

To build MLIR-AIR provide the paths to llvm, cmakeModules, and xrt (here, we assume it is installed in `/opt/xilinx/xrt`):
```bash
./utils/build-mlir-air-xrt.sh llvm mlir-aie/cmake/modulesXilinx mlir-aie aienginev2/install /opt/xilinx/xrt
```

### Environment

To setup your environment after building:
```bash
source utils/env_setup.sh install-xrt/ mlir-aie/install/ llvm/install/
```

Note that if you are starting a new environment (e.g., by creating a new terminal sometime after building), restore your environment with:
```bash
source utils/env_setup.sh install-xrt/ mlir-aie/install/ llvm/install/
source sandbox/bin/activate
```

### Testing

Some tests for MLIR-AIR are provided. Run them as demonstrated below:

```bash
cd mlir-air/build-xrt
ninja install
ninja check-air-cpp
ninja check-air-mlir
ninja check-air-python

# These are the ones in test/xrt, and this is about equivalent to `ninja check-air-e2e` if you set the LIT_OPS env var appropriately
lit -sv --time-tests --show-unsupported --show-excluded --timeout 600 -j5 test/xrt

# Run an individual test
lit -sv test/xrt/01_air_to_npu

# Run all xrt tests on device. Takes a long time.
ninja check-air-e2e
```

-----

<p align="center">Copyright&copy; 2022 Advanced Micro Devices, Inc.</p>
