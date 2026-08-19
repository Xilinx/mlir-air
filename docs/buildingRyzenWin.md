# Getting Started and Running on Windows Ryzen™ AI

This guide covers the **native Windows** setup for MLIR-AIR. It supports compiling AIR designs and running them on a Ryzen™ AI NPU entirely within Windows, without a POSIX environment. Users who prefer POSIX-style development can instead run WSL2 and follow the [Linux guide](buildingRyzenLin.md).

Use an **x64 Native Tools Command Prompt for Visual Studio** (`cmd.exe`) for the commands in this guide. It provides MSVC, the linker, and the Windows SDK in one configured environment. Visual Studio and Visual Studio Build Tools both install that prompt and add a Start menu shortcut. PowerShell is also supported; the equivalent commands are collected in [Addendum A](#addendum-a-powershell).

MLIR-AIR builds on [MLIR-AIE](https://github.com/Xilinx/mlir-aie), and the two share the same Windows host requirements. Where this guide and [mlir-aie's native Windows guide](https://github.com/Xilinx/mlir-aie/blob/main/docs/buildHostWinNative.md) overlap, they are intended to agree; sections 1–3 below are the common host setup.

> **Python note:** The Windows XRT SDK supplies `pyxrt` bindings for **CPython 3.13**. Use Python 3.13. Another Python version requires an XRT distribution with matching bindings. The whole stack will build and then fail at the final `import pyxrt` if the minor version does not match.

## Contents

1. [Install the Windows development environment](#1-install-the-windows-development-environment)
2. [Update and verify the NPU driver](#2-update-and-verify-the-npu-driver)
3. [Install the Windows XRT SDK](#3-install-the-windows-xrt-sdk)
4. [Install prebuilt wheels (recommended)](#4-install-prebuilt-wheels-recommended)
5. [Running a quick example](#5-running-a-quick-example)
6. [Build from source](#6-build-from-source)
7. [Addendum A: PowerShell](#addendum-a-powershell)
8. [Addendum B: Windows differences and limitations](#addendum-b-windows-differences-and-limitations)

## 1. Install the Windows development environment

- A Windows 11 system with a supported Ryzen™ AI / XDNA™ NPU.
- **Visual Studio 2026** (preferred) or **Visual Studio 2022** — the full IDE or the matching **Build Tools** package. Only needed for the source build in section 6; the wheel path in section 4 does not require a compiler.
- **Python 3.13** (see the note above).
- **Git for Windows**.

### 1.1 Visual Studio components

Select **Desktop development with C++**, and confirm these individual components:

- MSVC x64/x86 build tools
- Windows SDK
- C++ CMake tools for Windows
- Git for Windows, unless Git is installed separately

### 1.2 Install the tools

```bat
REM Choose one: the full IDE or the matching Build Tools package
winget install -e --id Microsoft.VisualStudio.Community
REM winget install -e --id Microsoft.VisualStudio.BuildTools

winget install -e --id Python.Python.3.13
winget install -e --id Git.Git
```

CMake and Ninja do **not** need a system-wide install; both are pulled into the Python environment in sections 4 and 6.

## 2. Update and verify the NPU driver

Install the latest Ryzen™ AI / XDNA™ driver, then verify the NPU is visible:

```bat
"C:\Windows\System32\AMD\xrt-smi.exe" examine
```

NPU driver **32.0.20101.3760** (XRT **2.21.0**) is the minimum supported on Windows.

Two things worth knowing when checking a driver version:

- AMD publishes two version series that are not comparable — `32.0.203.x` (Ryzen AI direct) and `32.0.201xx.xxxx` (WHQL/OEM). **Compare the XRT version instead**; it is a single monotonic series and is what actually matters here.
- The driver version reported by `xrt-smi` can be confirmed independently through PnP, which is useful when an installer claims success but changed nothing:

```powershell
Get-CimInstance Win32_PnPSignedDriver |
  Where-Object { $_.DeviceName -match 'NPU' } |
  Select-Object DeviceName, DriverVersion, DriverDate
```

## 3. Install the Windows XRT SDK

The XRT SDK provides the headers, import libraries, tools, and `pyxrt` bindings. Pair the SDK with the **driver's** XRT version, so install the driver first.

```text
https://github.com/Xilinx/XRT/releases/download/2.21.75/xrt_windows_sdk.zip
```

Extract the archive so that its `xrt_sdk\xrt` directory becomes:

```text
C:\Xilinx\XRT
```

`C:\Xilinx\XRT\python\pyxrt.pyd` should now exist. `C:\Xilinx\XRT` is the canonical location; a different path works as long as the environment variables in the following sections point at it.

To confirm which CPython ABI the bindings were built against, read the binary rather than trusting a label:

```bat
python -c "import re,pathlib;print(re.search(rb'python(\d)(\d{2})\.dll',pathlib.Path(r'C:\Xilinx\XRT\python\pyxrt.pyd').read_bytes(),re.I).groups())"
```

## 4. Install prebuilt wheels (recommended)

This is the fastest path and needs no compiler, no CMake, and no LLVM clone. Use it unless you intend to modify MLIR-AIR itself.

MLIR-AIR publishes **Windows AMD64** wheels. Backend dependencies are exposed as pip **extras**; for the AIE backend use the `[aie]` extra, which pins the exact `mlir_aie` version this AIR wheel was tested against and pulls `llvm-aie` (the Peano backend compiler).

1. **Create a virtual environment** with Python 3.13:

   ```bat
   python -m venv airenv
   airenv\Scripts\activate.bat
   python -m pip install --upgrade pip
   ```

2. **Install MLIR-AIR with the AIE backend:**

   ```bat
   pip install "mlir_air[aie]" ^
     -f https://github.com/Xilinx/mlir-air/releases/expanded_assets/latest-air-wheels ^
     -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-4 ^
     -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
   ```

3. **Set up the environment.** Save this as `air_env.cmd` next to your venv and `call` it in each new prompt:

   ```bat
   @echo off
   call "%~dp0airenv\Scripts\activate.bat" || exit /b 1
   set "SP=%VIRTUAL_ENV%\Lib\site-packages"
   set "MLIR_AIR_INSTALL_DIR=%SP%\mlir_air"
   set "MLIR_AIE_INSTALL_DIR=%SP%\mlir_aie"
   set "PEANO_INSTALL_DIR=%SP%\llvm-aie"
   set "XRT_ROOT=C:\Xilinx\XRT"
   set "PATH=%MLIR_AIR_INSTALL_DIR%\bin;%MLIR_AIE_INSTALL_DIR%\bin;%XRT_ROOT%;%XRT_ROOT%\lib;C:\Windows\System32\AMD;%PATH%"
   set "PYTHONPATH=%MLIR_AIR_INSTALL_DIR%\python;%MLIR_AIE_INSTALL_DIR%\python;%XRT_ROOT%\python"
   ```

   > **`C:\Windows\System32\AMD` on `PATH` is not optional.** MLIR-AIR selects the target device by running `xrt-smi examine` and matching the reported NPU model. If `xrt-smi` cannot be found the lookup fails silently and falls back to **npu1**, so an npu2 part (Strix, Strix Halo, Krackan) is compiled for the wrong target. Alternatively, pass `target_device="npu2"` explicitly to `XRTBackend` / `XRTRunner`.

4. **Verify the install:**

   ```bat
   call air_env.cmd
   air-opt --version
   python -c "import air, pyxrt; print('ok')"
   ```

### Choosing a wheel release

Both RTTI and no-RTTI variants are published so downstream projects can match their LLVM build configuration.

| Tag | When to use |
|-----|-------------|
| [`latest-air-wheels`](https://github.com/Xilinx/mlir-air/releases/tag/latest-air-wheels) | Default. RTTI enabled. Use this for standalone development; the command above targets this tag. |
| [`latest-air-wheels-no-rtti`](https://github.com/Xilinx/mlir-air/releases/tag/latest-air-wheels-no-rtti) | For integrating into a downstream project whose LLVM is built with `-DLLVM_ENABLE_RTTI=OFF`. Point both find-links at the no-RTTI pages so pip resolves the matching `mlir_aie`. |
| [`v*.*.*`](https://github.com/Xilinx/mlir-air/releases) | Pinned tagged release, for reproducible builds. |

When mixing wheels, all three (`mlir`, `mlir_aie`, `mlir_air`) must agree on RTTI.

## 5. Running a quick example

The element-wise add example stages data L3 → L2 → L1 across a two-core herd and checks the result against a NumPy reference. It exercises the whole toolchain: the Python builder, `aircc`, Peano, xclbin packaging, and XRT dispatch.

```bat
call air_env.cmd
cd programming_examples\eltwise_add_with_l2
mkdir build_peano
cd build_peano
python ..\eltwise_add.py --output-format xclbin
```

A final `PASS!` confirms the toolchain, XRT installation, and NPU are working together.

On npu2 parts you can also exercise the full-ELF output path, which is npu2-only:

```bat
cd programming_examples\matrix_scalar_add\single_core_dma
mkdir build_peano
cd build_peano
python ..\single_core_dma.py --output-format elf
```

> **Most examples ship a `Makefile`, and Windows has no `make`.** The Makefiles are thin wrappers — the `run` target is a `mkdir` plus the `python` invocation shown above, so it is easy to run by hand. `make` from MSYS2 or Chocolatey also works if you prefer. See [Addendum B](#addendum-b-windows-differences-and-limitations).

Explore `programming_examples\` for many more designs: GEMM, element-wise operations, softmax, RMSNorm, RoPE, FlashAttention, and end-to-end LLM examples.

## 6. Build from source

Build from source only to modify MLIR-AIR itself. LLVM/MLIR and MLIR-AIE come from prebuilt wheels, so this compiles MLIR-AIR alone — a few minutes on a modern laptop, not an LLVM-sized build.

### 6.1 Prerequisites

Sections 1–3 above, plus a **native Windows MLIR-AIE setup**, which supplies three things the build needs: the `ironenv` Python environment, the `cmake\modulesXilinx` CMake modules, and the `mlir_aie` package that `AIE_DIR` points at. Follow [mlir-aie's native Windows guide](https://github.com/Xilinx/mlir-aie/blob/main/docs/buildHostWinNative.md), which reduces to:

```bat
git clone --recurse-submodules https://github.com/Xilinx/mlir-aie.git C:\dev\mlir-aie
cd C:\dev\mlir-aie
python utils\iron_setup.py --dev
call .\iron_env.cmd
```

`--dev` installs the pinned CMake, Ninja, and lit into `ironenv`. Reusing that environment keeps MLIR-AIR's tool versions aligned with MLIR-AIE's.

If you would rather not clone MLIR-AIE, you need the CMake modules and the `mlir_aie` wheel separately:

```bat
git clone --depth 1 https://github.com/Xilinx/cmakeModules.git C:\dev\cmakeModules
pip install mlir_aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-4
pip install cmake ninja lit nanobind numpy
```

### 6.2 Stage LLVM/MLIR

MLIR-AIR consumes LLVM/MLIR as a prebuilt "distro wheel" rather than building it. Download the exact pinned version and unpack it:

```bat
cd C:\dev\mlir-air

REM Print the pinned version. `bash` ships with Git for Windows; the value is
REM also derivable from the `commithash` and `DATETIME` lines at the top of the
REM script, as 24.0.0.<DATETIME>+<first 8 chars of commithash>.
bash utils\clone-llvm.sh --get-wheel-version
REM e.g. 24.0.0.2026080106+56bcc187

mkdir my_install
cd my_install
pip download mlir==<version printed above> -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/mlir-distro
python -c "import zipfile,glob;zipfile.ZipFile(glob.glob('mlir-*.whl')[0]).extractall('.')"
```

If you already staged this wheel for a MLIR-AIE source build, check whether the version matches (`utils\clone-llvm.sh --get-wheel-version` in each repo) and reuse that tree rather than downloading 1 GB again.

See [Addendum B](#addendum-b-windows-differences-and-limitations) for two post-extraction fixups that may be needed.

### 6.3 Configure and build

**Do not use `utils\build-mlir-air-using-wheels.sh` or `utils\build-mlir-air-xrt.sh` on Windows.** Both auto-detect a linker with `if [ -x "$(command -v lld)" ]; then CMAKE_ARGS="$CMAKE_ARGS -DLLVM_USE_LINKER=lld"`. Peano and the Visual Studio Clang components both put `lld` on `PATH`, LLVM then probes the GCC/Clang-style `-fuse-ld=lld` flag, and MSVC rejects it with a configure-time fatal error. Invoke CMake directly instead, as MLIR-AIR's own Windows CI does.

Save this as `configure.cmd` and adjust the paths:

```bat
@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul 2>&1
call "C:\dev\mlir-aie\iron_env.cmd" || exit /b 1

set "AIR_SRC=C:/dev/mlir-air"
set "MLIR_DISTRO=C:/dev/mlir-air/my_install/mlir"
set "AIE_PKG=C:/dev/mlir-aie/ironenv/Lib/site-packages/mlir_aie"

if not exist "%AIR_SRC%/build" mkdir "%AIR_SRC%\build"
cd /d "%AIR_SRC%\build"

cmake "%AIR_SRC%" ^
  -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_MODULE_PATH=C:/dev/mlir-aie/cmake/modulesXilinx ^
  -DMLIR_DIR=%MLIR_DISTRO%/lib/cmake/mlir ^
  -DLLVM_DIR=%MLIR_DISTRO%/lib/cmake/llvm ^
  -DAIE_DIR=%AIE_PKG%/lib/cmake/aie ^
  -DPEANO_INSTALL_DIR=C:/dev/mlir-aie/ironenv/Lib/site-packages/llvm-aie ^
  -DXRT_ROOT=C:/Xilinx/XRT ^
  -DLLVM_EXTERNAL_LIT=C:/dev/mlir-aie/ironenv/Scripts/lit.exe ^
  -DPython3_EXECUTABLE=C:/dev/mlir-aie/ironenv/Scripts/python.exe ^
  -DCMAKE_INSTALL_PREFIX=%AIR_SRC%/install
```

Every path handed to CMake uses forward slashes. Three configure lines are worth checking before building:

```text
-- Found xrt_coreutil
-- Using AIEConfig.cmake in: .../mlir_aie/lib/cmake/aie
-- Skipping e2e tests on Windows (runtime libraries not built)
```

The third is expected — see [Addendum B](#addendum-b-windows-differences-and-limitations). Reports that Vitis, `xchesscc`, AIETools, and LibXAIE were not found are also expected and harmless: Windows uses Peano exclusively.

Then build and install:

```bat
cd C:\dev\mlir-air\build
ninja install
```

### 6.4 Set up the environment

Layer the freshly built install on top of the MLIR-AIE environment. Save as `air_env.cmd`:

```bat
@echo off
call "C:\dev\mlir-aie\iron_env.cmd" || exit /b 1
set "MLIR_AIR_INSTALL_DIR=C:\dev\mlir-air\install"
set "PATH=%MLIR_AIR_INSTALL_DIR%\bin;%PATH%"
set "PYTHONPATH=%MLIR_AIR_INSTALL_DIR%\python;%PYTHONPATH%"
```

`iron_env.cmd` already adds XRT, `pyxrt` on `PYTHONPATH`, and `C:\Windows\System32\AMD`, so device detection works. Section 5's examples now run against your build.

> Because `iron_env.cmd` points `MLIR_AIE_INSTALL_DIR` at the **wheel**, `PYTHONPATH` order decides which MLIR-AIR you get. Prepending your install directory, as above, ensures a source build shadows any `mlir_air` wheel that happens to be installed in the same environment.

### 6.5 Testing

```bat
cd C:\dev\mlir-air\build
ninja check-air-cpp
ninja check-air-mlir
ninja check-air-python
```

`check-air-cpp` and `check-air-python` pass on Windows. `check-air-mlir` currently has known failures confined to the `Util/Runner` and `Util/Channel` suites; see [Addendum B](#addendum-b-windows-differences-and-limitations).

The hardware end-to-end suites (`check-air-e2e*`) are **not available on Windows** — they depend on the host runtime libraries, which are not built. Run designs under `programming_examples\` instead.

---

<a id="addendum-a-powershell"></a>
<details>
<summary><strong>Addendum A: PowerShell</strong></summary>

Visual Studio installs a **Developer PowerShell for VS** shortcut providing the same compiler environment as the Native Tools `cmd.exe` prompt. PowerShell can also be started from an existing Native Tools prompt with `pwsh`, inheriting the compiler environment.

The toolchain is identical; only shell syntax differs. `cmd.exe` uses `%NAME%` and `call`; PowerShell uses `$env:NAME` and `&`, dot-sourcing (`. .\script.ps1`) to keep changes in the current scope.

**Wheel install (section 4):**

```powershell
python -m venv airenv
.\airenv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install "mlir_air[aie]" `
  -f https://github.com/Xilinx/mlir-air/releases/expanded_assets/latest-air-wheels `
  -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-4 `
  -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
```

**Environment (section 4 step 3), as `air_env.ps1` — dot-source it, the leading dot is required:**

```powershell
. .\airenv\Scripts\Activate.ps1
$SP = Join-Path $env:VIRTUAL_ENV 'Lib\site-packages'
$env:MLIR_AIR_INSTALL_DIR = Join-Path $SP 'mlir_air'
$env:MLIR_AIE_INSTALL_DIR = Join-Path $SP 'mlir_aie'
$env:PEANO_INSTALL_DIR    = Join-Path $SP 'llvm-aie'
$env:XRT_ROOT             = 'C:\Xilinx\XRT'
$env:PATH = "$env:MLIR_AIR_INSTALL_DIR\bin;$env:MLIR_AIE_INSTALL_DIR\bin;" +
            "$env:XRT_ROOT;$env:XRT_ROOT\lib;C:\Windows\System32\AMD;$env:PATH"
$env:PYTHONPATH = "$env:MLIR_AIR_INSTALL_DIR\python;$env:MLIR_AIE_INSTALL_DIR\python;$env:XRT_ROOT\python"
```

**Source build (section 6):** dot-source MLIR-AIE's `iron_env.ps1` instead of `iron_env.cmd`, and use backticks for line continuation in the `cmake` invocation:

```powershell
. C:\dev\mlir-aie\iron_env.ps1
cmake C:/dev/mlir-air `
  -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_MODULE_PATH=C:/dev/mlir-aie/cmake/modulesXilinx `
  -DMLIR_DIR=C:/dev/mlir-air/my_install/mlir/lib/cmake/mlir `
  -DLLVM_DIR=C:/dev/mlir-air/my_install/mlir/lib/cmake/llvm `
  -DAIE_DIR=C:/dev/mlir-aie/ironenv/Lib/site-packages/mlir_aie/lib/cmake/aie `
  -DPEANO_INSTALL_DIR=C:/dev/mlir-aie/ironenv/Lib/site-packages/llvm-aie `
  -DXRT_ROOT=C:/Xilinx/XRT `
  -DLLVM_EXTERNAL_LIT=C:/dev/mlir-aie/ironenv/Scripts/lit.exe `
  -DPython3_EXECUTABLE=C:/dev/mlir-aie/ironenv/Scripts/python.exe `
  -DCMAKE_INSTALL_PREFIX=C:/dev/mlir-air/install
```

The MSVC environment must still be present — start from a Developer PowerShell, or run `VsDevCmd.bat` in a `cmd.exe` prompt and launch `pwsh` from it.

### Conda or Miniforge Python

A dedicated Conda or Miniforge environment works with the SDK when it uses Python 3.13. Activate it before running MLIR-AIE's `iron_setup.py`, which uses the active interpreter when creating `ironenv`.

</details>

<a id="addendum-b-windows-differences-and-limitations"></a>
<details>
<summary><strong>Addendum B: Windows differences and limitations</strong></summary>

### Not built on Windows

`runtime_lib` (the `airhost` / `aircpu` host runtimes) depends on POSIX APIs — `dlopen`, `mmap`, `ioctl` — and on the Linux `amdair` kernel driver, so CMake skips it on Windows. Consequently:

- The `check-air-e2e*` targets and the `test/xrt` suite do not exist in a Windows build. `programming_examples\` is the end-to-end path.
- `Util/Channel` tests in `check-air-mlir` fail because they link a host executable against that runtime.

### Peano only, no Chess

Vitis and `xchesscc` are unavailable on Windows, so Peano is the only core compiler. CMake reporting `Unable to find xchesscc` / `Could NOT find Vitis` / `Could NOT find AIETools` is expected. Chess-gated LIT tests report *unsupported* rather than failing, and examples should be run with `PEANO_INSTALL_DIR` set.

### `check-air-mlir` failures

`Util/Runner` (the `air-runner` performance simulator) currently produces different output on Windows than the tests expect, and a handful of its tests use the `|&` shell operator, which lit's internal shell does not parse on Windows. These are open issues, not setup problems — the rest of the suite passes.

### No `make`

Most `programming_examples` ship Makefiles, and Windows has no `make`. The recipes are thin: `run` is a `mkdir` plus a `python` invocation. Read the Makefile and run the `python` line directly, or install `make` from MSYS2 or Chocolatey. Examples that compile an AIE kernel first (the GEMM examples, for instance) additionally invoke Peano's `clang++` with `--target=aie2p-none-unknown-elf` — that command is also in the Makefile and can be run as-is.

LIT tests driven by Makefiles are skipped automatically on Windows.

### Wrong device auto-detected

`XRTBackend` runs `xrt-smi examine` and matches the reported model name against its device table. If `xrt-smi` is not on `PATH`, detection fails **silently** and falls back to `npu1`. Keep `C:\Windows\System32\AMD` on `PATH`, or pass `target_device="npu2"` explicitly. A design compiled for the wrong target typically shows up as a runtime timeout rather than a clear error.

### Staged LLVM wheel fixups

Two problems can appear after extracting the MLIR distro wheel (section 6.2):

- **Hardcoded DIA SDK path.** `LLVMExports.cmake` inside the wheel can carry an absolute `diaguids.lib` path from the machine that built it. If CMake fails importing LLVM targets over a missing `diaguids.lib`, MLIR-AIE ships a fixup that rewrites it to your Visual Studio installation:

  ```bat
  python -c "import sys;sys.path.insert(0,r'C:\dev\mlir-aie\utils\mlir_aie_wheels\scripts');from pathlib import Path;import download_mlir;download_mlir._fixup_llvm_diaguids(Path(r'C:\dev\mlir-air\my_install\mlir'))"
  ```

- **Future-dated timestamps.** Files extracted from these wheels can carry mtimes in the future, which makes Ninja re-run configuration indefinitely. The symptom is a hang, not an error. Stamp them into the past:

  ```bat
  python -c "import os;[os.utime(os.path.join(r,f),(315600000,315600000)) for r,_,fs in os.walk(r'C:\dev\mlir-air\my_install\mlir') for f in fs]"
  ```

Both are avoided by reusing a tree that a MLIR-AIE build already staged and patched.

### Endpoint security

Endpoint protection has been observed quarantining individual executables out of the extracted LLVM wheel, which surfaces later as a confusing CMake error. A quick assertion after extraction:

```bat
if not exist my_install\mlir\bin\llvm-debuginfod.exe echo POSSIBLE AV QUARANTINE
```

</details>

-----

<p align="center">Copyright&copy; 2026 Advanced Micro Devices, Inc.</p>
