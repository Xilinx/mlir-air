# Getting Started and Running on Windows Ryzen™ AI

This guide covers the **native Windows** setup for MLIR-AIR. It supports compiling AIR designs and running them on a Ryzen™ AI NPU entirely within Windows, without a POSIX environment. Users who prefer POSIX-style development can instead run WSL2 and follow the [Linux guide](buildingRyzenLin.md).

Use an **x64 Native Tools Command Prompt for Visual Studio** (`cmd.exe`) for the commands in this guide. It provides MSVC, the linker, and the Windows SDK in one configured environment. Visual Studio and Visual Studio Build Tools both install that prompt and add a Start menu shortcut. PowerShell is also supported; the equivalent commands are collected in [section 7](#7-powershell).

MLIR-AIR builds on [MLIR-AIE](https://github.com/Xilinx/mlir-aie), and the two share the same Windows host requirements. Where this guide and [mlir-aie's native Windows guide](https://github.com/Xilinx/mlir-aie/blob/main/docs/buildHostWinNative.md) overlap, they are intended to agree; sections 1–3 below are the common host setup.

> **Python note:** `pyxrt` is a compiled extension module, so your Python has to be the exact CPython minor version the XRT SDK built it against — **3.13** at the time of writing. Any other minor version installs and builds the whole stack, then fails at `import pyxrt`. [Section 3](#3-install-the-windows-xrt-sdk) reads the required version straight out of `pyxrt.pyd`; that check is authoritative if a newer SDK has moved on and this note has not.

## 1. Install the Windows development environment

- A Windows 11 system with a supported Ryzen™ AI / XDNA™ NPU.
- **Visual Studio 2026** (preferred) or **Visual Studio 2022** — the full IDE or the matching **Build Tools** package. Only needed for the source build in section 6; the wheel path in section 4 does not require a compiler.
- **Python**, at the minor version the note above names.
- **Git for Windows**.
- **GNU make**, only if you intend to run the example test suites in
  [section 6.5](#65-testing) — most of those tests shell out to it. Running an
  individual example does not need it.

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

REM The package id carries the minor version; match the Python note above
winget install -e --id Python.Python.3.13
winget install -e --id Git.Git

REM Only needed to run the example test suites (section 6.5)
winget install -e --id ezwinports.make
```

`winget` installs Visual Studio without any workload, so open the Visual Studio
Installer afterwards and add the components in
[section 1.1](#11-visual-studio-components) — otherwise there is no C++ compiler
and [section 6.3](#63-configure-and-build) stops with a message saying so.

`ezwinports.make` is a standalone GNU make with no other dependencies; it picks
up Git's `sh.exe` as its shell, which is what the Makefile recipes expect. `make`
from MSYS2 or Chocolatey works equally well.

CMake and Ninja do **not** need a system-wide install; both are pulled into the Python environment in sections 4 and 6.

A dedicated Conda or Miniforge environment works too, as long as it is on that same minor version. Activate it before running MLIR-AIE's `iron_setup.py` in [section 6.1](#61-prerequisites), which creates `ironenv` from the active interpreter.

## 2. Update and verify the NPU driver

Install the latest Ryzen™ AI / XDNA™ driver, then verify the NPU is visible:

```bat
"C:\Windows\System32\AMD\xrt-smi.exe" examine
```

MLIR-AIR adds no driver requirement of its own; the supported floor is whatever
[mlir-aie's native Windows guide](https://github.com/Xilinx/mlir-aie/blob/main/docs/buildHostWinNative.md)
states, since that is the host setup both repositories share.

## 3. Install the Windows XRT SDK

The XRT SDK provides the headers, import libraries, tools, and `pyxrt` bindings. Its version has to match the **driver's** XRT version, which `xrt-smi examine` reports — so install the driver first, then pick the SDK to match.

Download that release's `xrt_windows_sdk.zip` from the [XRT releases page](https://github.com/Xilinx/XRT/releases) and extract it so that its `xrt_sdk\xrt` directory becomes:

```text
C:\Xilinx\XRT
```

`C:\Xilinx\XRT\python\pyxrt.pyd` should now exist. `C:\Xilinx\XRT` is the canonical location; a different path works as long as the environment variables in the following sections point at it.

Read the CPython ABI out of the bindings rather than trusting a label, and check it against the Python you installed in section 1:

```bat
python -c "import re,pathlib;print(re.search(rb'python(\d)(\d{2})\.dll',pathlib.Path(r'C:\Xilinx\XRT\python\pyxrt.pyd').read_bytes(),re.I).groups())"
python -c "import sys;print(sys.version_info[:2])"
```

Both must print the same major and minor. If they differ, this SDK wants a different Python than the one you have; install that version and use it for the virtual environment in section 4. This check, not the note at the top of the guide, is the authority.

## 4. Install prebuilt wheels (recommended)

This is the fastest path and needs no compiler, no CMake, and no LLVM clone. Use it unless you intend to modify MLIR-AIR itself.

MLIR-AIR publishes **Windows AMD64** wheels. Backend dependencies are exposed as pip **extras**; for the AIE backend use the `[aie]` extra, which pins the exact `mlir_aie` version this AIR wheel was tested against and pulls `llvm-aie` (the Peano backend compiler).

1. **Create a virtual environment** with that Python:

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

   > `C:\Windows\System32\AMD` is on `PATH` so that MLIR-AIR can find `xrt-smi` and select the target device from the NPU model it reports. Passing `target_device="npu1"` or `target_device="npu2"` to `XRTBackend` / `XRTRunner` selects it explicitly instead.

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

Both NPU generations are supported. What differs between them:

| | npu1 (Phoenix) | npu2 (Strix, Strix Halo, Krackan) |
|---|---|---|
| Core architecture | `aie2` (default) | `aie2p` |
| `num_device_cols` | 0 (whole device) or 1–3 | 0 (whole device) or 1–7 |
| `--output-format xclbin` / `pdi` | yes | yes |
| `--output-format elf` (full ELF) | no | yes |

Full ELF needs an `aiebu-asm` configuration that targets AIE2P. Asking for it on npu1 is rejected at compile time rather than failing on the device:

```text
output_format='elf' is not supported for npu1 target. ELF output format is
only supported on npu2 and later devices.
```

On npu2 parts you can exercise that path:

```bat
cd programming_examples\matrix_scalar_add\single_core_dma
mkdir build_peano
cd build_peano
python ..\single_core_dma.py --output-format elf
```

Most examples also ship a `Makefile` whose `run` target does the same thing, so
`make run` works too once GNU make is installed ([section 1.2](#12-install-the-tools)).

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
pip install -r C:\dev\mlir-air\utils\requirements_dev.txt
```

### 6.2 Stage LLVM/MLIR

MLIR-AIR consumes LLVM/MLIR as a prebuilt "distro wheel" rather than building it. Download the exact pinned version and unpack it:

The pinned version lives in `utils\clone-llvm.sh`, as a `WHEEL_VERSION` built from the `commithash` and `DATETIME` lines above it. Read it without a shell:

```bat
cd C:\dev\mlir-air
python -c "import re;s=open('utils/clone-llvm.sh').read();g=lambda p:re.search(p,s).group(1);print(g(r'WHEEL_VERSION=(\S+)').replace('$DATETIME',g(r'DATETIME=(\d+)')).replace('${commithash:0:8}',g(r'commithash=(\w+)')[:8]))"
REM e.g. 24.0.0.2026080106+56bcc187
```

Then download and unpack it:

```bat
mkdir my_install
cd my_install
pip download mlir==<version printed above> -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/mlir-distro
python -c "import zipfile,glob;zipfile.ZipFile(glob.glob('mlir-*.whl')[0]).extractall('.')"
cd ..
python utils\fixup-llvm-diaguids.py my_install\mlir
```

That last step repoints one absolute path the wheel carries from the machine
that built it — `diaguids.lib`, the DIA SDK import library
`LLVMDebugInfoPDB` links against. Skipping it surfaces much later, as a link
failure naming a Visual Studio directory you do not have. The script is
idempotent, so re-running it after a re-extract is harmless.

(If you do have Git Bash handy, `bash utils\clone-llvm.sh --get-wheel-version` prints the same string.)

If you already staged this wheel for a MLIR-AIE source build, check whether the version matches (`utils\clone-llvm.sh --get-wheel-version` in each repo) and reuse that tree rather than downloading 1 GB again.

### 6.3 Configure and build

Invoke CMake directly, as MLIR-AIR's own Windows CI does. Save this as
`configure.cmd` and adjust the paths. `vswhere` finds Visual Studio whatever its
edition, version or location: `-products *` is needed because the default filter
covers Community/Professional/Enterprise but silently omits Build Tools, and
`-requires` rejects an install that has no C++ toolset, which is what a bare
`winget install` leaves you with.

```bat
@echo off
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSPATH=%%i"
if not defined VSPATH (
  echo No Visual Studio with the C++ toolset found -- see section 1.1.
  exit /b 1
)
call "%VSPATH%\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul 2>&1 || exit /b 1
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
  -DENABLE_RUN_XRT_TESTS=ON ^
  -DCMAKE_INSTALL_PREFIX=%AIR_SRC%/install
```

`-DENABLE_RUN_XRT_TESTS=ON` defaults to OFF and is what lets the test suites in
[section 6.5](#65-testing) dispatch to the NPU. Every path handed to CMake uses
forward slashes.

Three configure lines are worth checking before building:

```text
-- Found xrt_coreutil
-- Using AIEConfig.cmake in: .../mlir_aie/lib/cmake/aie
-- Skipping e2e tests on Windows (runtime libraries not built)
```

The third is expected: the `airhost` / `aircpu` host runtimes need POSIX APIs
(`dlopen`, `mmap`, `ioctl`) and the Linux `amdair` kernel driver, so CMake skips
them, and with them the `test/` tree. Reports that Vitis, `xchesscc`, AIETools,
and LibXAIE were not found are also expected and harmless: Windows uses Peano
exclusively, and Chess-gated tests report *unsupported* rather than failing.

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

The compile-only suites need nothing beyond the build environment:

```bat
cd C:\dev\mlir-air\build
ninja check-air-cpp
ninja check-air-mlir
ninja check-air-python
```

#### Running the examples on hardware

`check-programming-examples-peano` dispatches to the NPU, so it needs the
**runtime** environment of [section 6.4](#64-set-up-the-environment) rather than
the build environment, plus GNU make on `PATH` — most example tests shell out to
it. A wrapper that layers both:

```bat
@echo off
call C:\dev\mlir-aie\iron_env.cmd || exit /b 1
set "MLIR_AIR_INSTALL_DIR=C:\dev\mlir-air\install"
set "PATH=%MLIR_AIR_INSTALL_DIR%\bin;%PATH%"

REM GNU make, plus Git's usr\bin for the sh.exe make uses as its shell. Both
REM install per-user or machine-wide depending on how they were installed, so
REM probe rather than assume; add your own directories if either came from
REM MSYS2 or Chocolatey instead.
if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links\make.exe" set "PATH=%LOCALAPPDATA%\Microsoft\WinGet\Links;%PATH%"
if exist "%ProgramFiles%\Git\usr\bin\sh.exe" set "PATH=%ProgramFiles%\Git\usr\bin;%PATH%"
if exist "%LOCALAPPDATA%\Programs\Git\usr\bin\sh.exe" set "PATH=%LOCALAPPDATA%\Programs\Git\usr\bin;%PATH%"

where make >nul 2>&1 || (echo GNU make not found -- see section 1.2. & exit /b 1)
where sh   >nul 2>&1 || (echo sh.exe not found; is Git for Windows installed? & exit /b 1)

cd /d C:\dev\mlir-air\build
ninja check-programming-examples-peano
```

`iron_env.cmd` supplies XRT — which provides `aiebu-asm.exe`, needed to package a
full ELF — and `C:\Windows\System32\AMD` for `xrt-smi.exe`. The full suite takes
roughly 20 minutes on a Krackan Point NPU2.

Use `lit` directly, in the same environment, to run one example:

```bat
lit -sv -j1 --filter "eltwise_add_with_l2.*peano" C:/dev/mlir-air/build/programming_examples
```

#### The xrt end-to-end suites

`check-air-e2e*` and `check-air-runner` do not exist in a Windows build: the
`test/` tree is skipped along with the host runtimes it needs
([section 6.3](#63-configure-and-build)). Even with that guard lifted, much of
`test/xrt` builds a Linux host binary (`g++-13`, boost, `-luuid -lrt`) or wraps
execution in `flock`. Use `programming_examples\` as the end-to-end path on
Windows.

## 7. PowerShell

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

-----

<p align="center">Copyright&copy; 2026 Advanced Micro Devices, Inc.</p>
