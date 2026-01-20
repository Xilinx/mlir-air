# Getting Started and Running on Linux Ryzen™ AI

## Quick Start: Build with Prebuilt Wheels

### Prerequisites

- **Python 3.10+** (required by the wheels)
- **gcc >= 11**
- **pip** (Python package manager)
- **XRT** (optional, required only for running on hardware)

### Steps

1. **Clone the MLIR-AIR repository:**
   ```bash
   git clone https://github.com/Xilinx/mlir-air.git
   cd mlir-air
   ```

2. **Install the following packages needed for MLIR-AIR:**
   ```bash
   sudo apt-get install -y ninja-build clang lld unzip
   ```

3. **Set up a Python virtual environment with the prerequisite python packages :**
   ```bash
   source utils/setup_python_packages.sh
   ```

4. **Run the build script:**
   
   **Without XRT (software-only build):**
   ```bash
   ./utils/build-mlir-air-using-wheels.sh [build_dir] [install_dir]
   ```
   
   **With XRT (for hardware execution):**
   ```bash
   ./utils/build-mlir-air-using-wheels.sh --xrt-dir <xrt_path> [build_dir] [install_dir]
   ```
   
   Parameters:
   - `--xrt-dir <xrt_path>`: Path to your XRT installation (optional, only needed for hardware execution)
   - `[build_dir]`: Build directory (optional, default: `build`)
   - `[install_dir]`: Install directory (optional, default: `install`)

   The script will:
   - Download and unpack a prebuilt LLVM wheel (which includes MLIR)
   - pip install [`llvm-aie`](https://github.com/Xilinx/llvm-aie), which is used as a backend to generate AIE binaries
   - Install `mlir-aie` dependencies from wheels
   - Clone required CMake modules
   - Configure and build MLIR-AIR using CMake and Ninja
   - Optionally configure XRT support if `--xrt-dir` is provided

5. **Environment Setup:**
   To setup your environment after building:
   ```bash
   source utils/env_setup.sh [install_dir] $(python3 -m pip show mlir_aie | grep Location | awk '{print $2}')/mlir_aie $(python3 -m pip show llvm-aie | grep Location | awk '{print $2}')/llvm-aie my_install/mlir
   ```
   This command automatically detects the installation directories of the `mlir-aie` and `llvm-aie` Python packages, and sets up environment variables for MLIR-AIR, MLIR-AIE, PEANO (llvm-aie compiler), Python, and MLIR libraries.
   
   **If you built with XRT support**, also run:
   ```bash
   source [xrt_dir]/setup.sh
   ```
   This sets up the PATHs for XRT.
   
   If you start a new terminal, you may need to re-source the above setup scripts as needed.

6. **Testing:**
   After building, you can run tests as follows:
   ```bash
   cd <build_dir>   # default is 'build'
   ninja install
   ninja check-air-cpp
   ninja check-air-mlir
   ninja check-air-python
   ```
   
   **If you built with XRT support**, you can also run XRT/hardware tests:
   ```bash
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

## Running a Quick Example

After building MLIR-AIR, you can try the i8 matrix multiplication example to verify your setup and understand the different compilation workflows. Matmul shapes are configurable in the Makefile.

### Example 1: Hardware-Free Compilation (No XRT Required)

This mode is useful for **cross-compilation** or **development without hardware access**. It generates intermediate compilation artifacts without requiring XRT to be installed:

```bash
cd programming_examples/matrix_multiplication/i8
make run4x4 COMPILE_MODE=compile-only
```

**Expected output:** `Compilation completed successfully!`

**What this does:**
- Compiles AIR dialect code through the full compilation pipeline
- Generates intermediate MLIR files and NPU instructions
- Does **not** generate xclbin (no `xclbinutil` needed)
- Does **not** require XRT or hardware

**When to use:**
- Building on a system without XRT installed
- Cross-compiling for deployment on another system
- CI/CD pipelines
- Early development and testing

### Example 2: Full Workflow with Hardware (Default)

If you have XRT and Ryzen AI hardware available, run the complete workflow:

```bash
cd programming_examples/matrix_multiplication/i8
make run4x4
```

**Expected output:** `PASS!`

**What this does:**
- Compiles the AIR code
- Generates xclbin and instruction files
- Loads and executes on NPU hardware
- Validates results against expected outputs

This is the default mode (`COMPILE_MODE=compile-and-run`) for users with hardware.

### Example 3: Advanced - Profiling with Custom Host Code

For specialized workflows like profiling with custom test executables:

```bash
make profile
```

**What this does:**
- Uses `compile-and-xclbin` mode to generate xclbin and instructions
- Runs a custom C++ test executable (not xrt_runner) for detailed profiling
- Useful for performance measurement and custom host integration

The `sweep4x4` target similarly uses `compile-and-xclbin` to benchmark across multiple problem sizes with a custom test harness.

### Additional Examples

**Different herd configurations:**
```bash
make run2x2 COMPILE_MODE=compile-only  # 2x2 herd
make run8x4 COMPILE_MODE=compile-only  # 8x4 herd
```

**Different architectures:**
```bash
make run4x4 AIE_TARGET=aie2p COMPILE_MODE=compile-only  # For NPU2/Strix
make run4x4 AIE_TARGET=aie2                              # For NPU1/Phoenix (default)
```

**View generated MLIR:**
```bash
make print  # Display MLIR module without compiling
```

**Clean build artifacts:**
```bash
make clean
```

### Other Data Types

The same patterns work for other matrix multiplication examples:
- `programming_examples/matrix_multiplication/bf16/` - bfloat16 matrix multiply
- `programming_examples/matrix_multiplication/i16/` - int16 matrix multiply

### Exploring More Examples

Explore `programming_examples/` for many more examples including:
- Element-wise operations
- Softmax
- Sine/cosine
- Llama 2-style multi-head attention
- Flash attention
- Vector instruction micro-benchmark
- And more

Most examples follow similar Makefile patterns with `COMPILE_MODE` and `AIE_TARGET` support.

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
# If you have llvm-aie (PEANO) installed via pip:
source utils/env_setup.sh install-xrt/ mlir-aie/install/ $(python3 -m pip show llvm-aie | grep Location | awk '{print $2}')/llvm-aie llvm/install/

# Or if you built llvm-aie from source:
source utils/env_setup.sh install-xrt/ mlir-aie/install/ /path/to/llvm-aie/install llvm/install/
```

Note that if you are starting a new environment (e.g., by creating a new terminal sometime after building), restore your environment with:
```bash
# If you have llvm-aie (PEANO) installed via pip:
source utils/env_setup.sh install-xrt/ mlir-aie/install/ $(python3 -m pip show llvm-aie | grep Location | awk '{print $2}')/llvm-aie llvm/install/
source sandbox/bin/activate

# Or if you built llvm-aie from source:
source utils/env_setup.sh install-xrt/ mlir-aie/install/ /path/to/llvm-aie/install llvm/install/
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
