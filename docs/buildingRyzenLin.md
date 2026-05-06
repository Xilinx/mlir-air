# Getting Started and Running on Linux Ryzen™ AI

## Install Prebuilt Wheels (Recommended)

The fastest way to get MLIR-AIR is to install the prebuilt wheel — no source build, no CMake, no LLVM clone. Use this path unless you need to modify MLIR-AIR itself or run on an unsupported Python/platform.

### Prerequisites

- **Python 3.10–3.14**
- **pip**
- **XRT** (optional, required only for running on hardware) — see [mlir-aie's XRT install instructions](https://github.com/Xilinx/mlir-aie#install-the-xdna-driver-and-xrt)

> Windows wheels are also published. This guide covers Linux only; Windows users will need to translate the `source`/`export` commands to PowerShell equivalents.

### Steps

AIR supports multiple backends — AIE (NPU / Versal AI Engines), [GPU](buildingGPU.md), and [VCK5000](buildingVCK5000.md). Backend dependencies are exposed as pip **extras** so you only install what you need. For the AIE backend on Ryzen™ AI, use the `[aie]` extra; pip reads the pinned `mlir_aie==<version>` and `llvm-aie` from the wheel's metadata and resolves them from the additional `--find-links` pages.

1. **Create a virtual environment:**
   ```bash
   python3 -m venv airenv
   source airenv/bin/activate
   pip install --upgrade pip
   ```

2. **Install MLIR-AIR with the AIE backend:**
   ```bash
   pip install 'mlir_air[aie]' \
     -f https://github.com/Xilinx/mlir-air/releases/expanded_assets/latest-air-wheels \
     -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-3 \
     -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
   ```

   The `[aie]` extra pulls `mlir_aie` (pinned to the exact version this AIR wheel was tested against) and `llvm-aie` (the Peano backend compiler — nightly). The pinned `mlir_aie` version is shown on the [release page](https://github.com/Xilinx/mlir-air/releases/tag/latest-air-wheels).

   To install AIR core only (e.g. when bringing your own backend), drop the `[aie]` extra and the last two `-f` flags.

3. **Set up environment variables** (paths derived from `pip show`):
   ```bash
   export MLIR_AIR_INSTALL_DIR=$(python3 -m pip show mlir_air | grep ^Location: | awk '{print $2}')/mlir_air
   export MLIR_AIE_INSTALL_DIR=$(python3 -m pip show mlir_aie | grep ^Location: | awk '{print $2}')/mlir_aie
   export PEANO_INSTALL_DIR=$(python3 -m pip show llvm-aie | grep ^Location: | awk '{print $2}')/llvm-aie

   export PATH=$MLIR_AIR_INSTALL_DIR/bin:$MLIR_AIE_INSTALL_DIR/bin:$PATH
   export PYTHONPATH=$MLIR_AIR_INSTALL_DIR/python:$MLIR_AIE_INSTALL_DIR/python:$PYTHONPATH
   export LD_LIBRARY_PATH=$MLIR_AIR_INSTALL_DIR/lib:$MLIR_AIE_INSTALL_DIR/lib:$LD_LIBRARY_PATH

   # Only if you have XRT installed:
   source /opt/xilinx/xrt/setup.sh
   ```

4. **Verify the install:**
   ```bash
   air-opt --help
   ```

You can now jump to [Running a Quick Example](#running-a-quick-example) below.

### Choosing a Wheel Release

We release **both RTTI and no-RTTI variants** so downstream projects can match their LLVM build configuration without having to rebuild MLIR-AIR from source.

| Tag | When to use |
|-----|-------------|
| [`latest-air-wheels`](https://github.com/Xilinx/mlir-air/releases/tag/latest-air-wheels) | Default. RTTI enabled. Use this for standalone development. The install command in step 2 above targets this tag. |
| [`latest-air-wheels-no-rtti`](https://github.com/Xilinx/mlir-air/releases/tag/latest-air-wheels-no-rtti) | For integrating MLIR-AIR into a downstream project whose LLVM is built with `-DLLVM_ENABLE_RTTI=OFF`. Use the no-RTTI install snippet below. |
| [`v*.*.*`](https://github.com/Xilinx/mlir-air/releases) (e.g. `v1.0.0`) | Pinned tagged release, for reproducible builds. Replace the find-links URL with `.../expanded_assets/v<ver>` and the package spec with `mlir_air==<ver>`. |

For the no-RTTI variant, point both find-links at the no-RTTI pages so pip resolves the matching `mlir_aie`:

```bash
pip install 'mlir_air[aie]' \
  -f https://github.com/Xilinx/mlir-air/releases/expanded_assets/latest-air-wheels-no-rtti \
  -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-no-rtti \
  -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
```

When mixing wheels into a downstream build, all three (`mlir`, `mlir_aie`, `mlir_air`) must agree on RTTI.

If the wheel install path doesn't fit your needs, see the source-build instructions below.

---

## Build From Source Using Prebuilt Dependencies

This path builds MLIR-AIR from source but installs LLVM, MLIR-AIE, and Peano from prebuilt wheels — useful when you're hacking on MLIR-AIR itself but don't want to also rebuild its dependencies.

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
