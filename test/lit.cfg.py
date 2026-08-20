# ./test/lit.cfg.py -*- Python -*-
#
# Copyright (C) 2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# -*- Python -*-

import os
import platform
import re
import subprocess
import tempfile
import shutil
import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "AIR_TEST"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.environment["PYTHONPATH"] = os.pathsep.join(
    [
        os.path.join(config.air_obj_root, "python"),
        os.path.join(config.aie_obj_root, "python"),
        os.path.join(config.xrt_dir, "python"),
    ]
)

# os.environ['PYTHONPATH']
print("Running with PYTHONPATH", config.environment["PYTHONPATH"])

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".lit"]

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = []

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.air_obj_root, "test")
air_runtime_lib = os.path.join(
    config.air_obj_root, "runtime_lib", config.runtime_test_target
)

config.substitutions.append(("%PYTHON", config.python_executable))
config.substitutions.append(("%CLANG", "clang++ -fuse-ld=lld -DLIBXAIENGINEV2"))
config.substitutions.append(("%LIBXAIE_DIR%", config.libxaie_dir))
config.substitutions.append(
    (
        "%AIE_RUNTIME_DIR%",
        os.path.join(config.aie_obj_root, "runtime_lib", config.runtime_test_target),
    )
)
config.substitutions.append(("%aietools", config.vitis_aietools_dir))

test_lib_path = os.path.join(
    config.aie_obj_root, "runtime_lib", config.runtime_test_target, "test_lib"
)
config.substitutions.append(
    (
        "%test_utils_flags",
        "-lboost_program_options -lboost_filesystem "
        + f"-I{test_lib_path}/include -L{test_lib_path}/lib -ltest_utils",
    )
)

# for xchesscc_wrapper
llvm_config.with_environment("AIETOOLS", config.vitis_aietools_dir)

if config.hsa_found:
    # Getting the path to the ROCm directory. hsa-runtime64 points to the cmake
    # directory so need to go up three directories
    rocm_root = os.path.join(config.hsa_dir, "..", "..", "..")
    print("Found ROCm:", rocm_root)
    config.substitutions.append(("%HSA_DIR%", "{}".format(rocm_root)))
    config.substitutions.append(
        (
            "%airhost_libs%",
            " -I"
            + air_runtime_lib
            + "/airhost/include"
            + " -L"
            + air_runtime_lib
            + "/airhost -Wl,--whole-archive -lairhost"
            + " -Wl,-R{}/lib -Wl,-rpath,{}/lib -Wl,--whole-archive".format(
                config.libxaie_dir, rocm_root
            )
            + " -Wl,--no-whole-archive -lpthread -lstdc++ -lsysfs -ldl -lrt -lelf",
        )
    )
    if config.enable_run_airhost_tests:
        config.substitutions.append(("%run_on_board", "flock /tmp/vck5000.lock"))
    else:
        print("Skipping execution of airhost tests (ENABLE_RUN_AIRHOST_TESTS=OFF)")
        config.substitutions.append(("%run_on_board", "echo"))
else:
    print("ROCm not found")
    config.excludes.append("airhost")


run_on_npu1 = "echo"
run_on_npu2 = "echo"
xrt_flags = ""

# XRT
if config.xrt_lib_dir and config.enable_run_xrt_tests:
    print("xrt found at", os.path.dirname(config.xrt_lib_dir))
    xrt_flags = "-I{} -L{} -luuid -lxrt_coreutil".format(
        config.xrt_include_dir, config.xrt_lib_dir
    )
    config.available_features.add("xrt")

    try:
        xrtsmi = os.path.join(config.xrt_bin_dir, "xrt-smi")
        # Windows ships xrt-smi.exe with the NPU driver (System32\AMD, on PATH)
        # rather than under the XRT SDK, and it needs the .exe suffix. which()
        # covers both; on Linux the path above already resolves and is used.
        xrtsmi = shutil.which(xrtsmi) or shutil.which("xrt-smi") or xrtsmi
        result = subprocess.run(
            [xrtsmi, "examine"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        out = result.stdout.decode("utf-8")
        out_lc = out.lower()
        # Case-insensitive substring match against known NPU model strings,
        # kept in sync with programming_examples/lit.cfg.py and the XRT
        # backend's NPU_MODELS. The strict regex + exact-match list this
        # replaces predated Strix Halo and Krackan, so those parts fell to the
        # "unknown model" branch: `ryzen_ai_npu2` was never added and every
        # test gated on it reported UNSUPPORTED on hardware that can run it.
        npu2_models = ["npu4", "strix", "npu5", "strix halo", "npu6", "krackan"]
        npu1_models = ["npu1", "phoenix"]
        # Serializing through flock and run_on_npu.sh is POSIX-only: flock is
        # util-linux, and the script is bash that sources
        # /opt/xilinx/xrt/setup.sh. Now that xrt-smi is found on Windows too,
        # reusing it there would let the suite configure and then fail to launch
        # every test carrying this substitution. Run the command directly
        # instead -- the environment is already set up by whoever invoked lit.
        if os.name == "nt":
            run_on_npu = ""
        else:
            run_on_npu = (
                f"flock /tmp/npu.lock {config.air_src_root}/utils/run_on_npu.sh"
            )
        if any(k in out_lc for k in npu2_models):
            config.available_features.add("ryzen_ai")
            config.available_features.add("ryzen_ai_npu2")
            # Resolved once here so the ~230 test processes in a suite run do not
            # each re-run xrt-smi examine. That probe has a 10 s timeout and
            # intermittently exceeds it under load; on timeout the backend falls
            # back to npu1 and silently builds a 4-column design for an 8-column part.
            config.environment["AIR_TARGET_DEVICE"] = "npu2"
            run_on_npu2 = run_on_npu
            print("Running tests on NPU2 with command line: ", run_on_npu2 or "(none)")
        elif any(k in out_lc for k in npu1_models):
            config.available_features.add("ryzen_ai")
            config.available_features.add("ryzen_ai_npu1")
            config.environment["AIR_TARGET_DEVICE"] = "npu1"
            run_on_npu1 = run_on_npu
            print("Running tests on NPU1 with command line: ", run_on_npu1 or "(none)")
        else:
            # No recognized model: dump xrt-smi output so the cause (a format
            # change, or a driver error) is visible instead of silently
            # skipping every NPU test.
            print("WARNING: xrt-smi did not report a recognized NPU model.")
            print("xrt-smi returncode:", result.returncode)
            print("xrt-smi examine stdout:\n" + out)
            print("xrt-smi examine stderr:\n" + result.stderr.decode("utf-8"))
    except Exception as e:
        print(f"Failed to run xrt-smi: {e}")
else:
    print("xrt not found or xrt tests disabled")
    config.excludes.append("xrt")

config.substitutions.append(("%run_on_npu1%", run_on_npu1))
config.substitutions.append(("%run_on_npu2%", run_on_npu2))
config.substitutions.append(("%xrt_flags", xrt_flags))
config.substitutions.append(("%XRT_DIR", config.xrt_dir))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

llvm_config.use_default_substitutions()

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.air_obj_root, "test")
config.aie_tools_dir = os.path.join(config.aie_obj_root, "bin")
config.aie_python_tools_dir = os.path.join(config.aie_obj_root, "python/aie/utils")
config.air_tools_dir = os.path.join(config.air_obj_root, "bin")

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.peano_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.aie_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.aie_python_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.air_tools_dir, append_path=True)

config.substitutions.append(("%LLVM_TOOLS_DIR", config.llvm_tools_dir))

tool_dirs = [config.aie_tools_dir, config.aie_python_tools_dir, config.llvm_tools_dir]

# Test if Peano is available
try:
    result = subprocess.run(
        [os.path.join(config.peano_tools_dir, "llc"), "-mtriple=aie", "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if re.search("Xilinx AI Engine", result.stdout.decode("utf-8")) is not None:
        config.available_features.add("peano")
        config.substitutions.append(
            ("%PEANO_INSTALL_DIR", os.path.dirname(config.peano_tools_dir))
        )
        print("Peano found: " + os.path.join(config.peano_tools_dir, "llc"))
        peano_flags = (
            "-O2 -std=c++20 -DNDEBUG -mllvm -aie-disable-fold-imm"
            " -D__AIE_API_AIE_ADF_HPP__ -I{}".format(
                os.path.join(config.aie_obj_root, "include")
            )
        )
        config.substitutions.append(("%peano_flags", peano_flags))
    else:
        print("Peano not detected at expected path:", config.peano_tools_dir)
except Exception:
    print("Peano check failed.")

# Test if Chess is available
if not config.enable_chess_tests:
    print("Chess tests disabled.")
else:
    print("Looking for Chess...")

    chess_path = shutil.which("xchesscc")
    if chess_path:
        print("Chess found: " + chess_path)
        config.available_features.add("chess")
        lm_license_file = os.getenv("LM_LICENSE_FILE")
        xilinxd_license_file = os.getenv("XILINXD_LICENSE_FILE")

        if lm_license_file:
            llvm_config.with_environment("LM_LICENSE_FILE", lm_license_file)
        if xilinxd_license_file:
            llvm_config.with_environment("XILINXD_LICENSE_FILE", xilinxd_license_file)

        # Optionally validate license
        validate_chess = False
        if validate_chess:
            result = subprocess.run(
                ["xchesscc", "+v"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if len(result.stderr.decode("utf-8")) == 0:
                config.available_features.add("valid_xchess_license")
        else:
            if lm_license_file or xilinxd_license_file:
                config.available_features.add("valid_xchess_license")
            else:
                print("WARNING: Chess license environment variables not found.")

    elif os.getenv("XILINXD_LICENSE_FILE") is not None:
        print("Chess license found")
        llvm_config.with_environment(
            "XILINXD_LICENSE_FILE", os.getenv("XILINXD_LICENSE_FILE")
        )
    else:
        print("Chess not found")

tool_dirs = [
    config.peano_tools_dir,
    config.aie_tools_dir,
    config.aie_python_tools_dir,
    config.air_tools_dir,
    config.llvm_tools_dir,
]
tools = [
    "aie-opt",
    "aie-translate",
    # mlir-aie v1.4.0 replaced the aiecc.py Python wrapper with a native C++
    # aiecc binary. Register aiecc.py before aiecc so the longer token wins,
    # and alias it to the binary so existing tests keep working unchanged.
    ToolSubst("aiecc.py", FindTool("aiecc")),
    "aiecc",
    "aircc",
    "air-opt",
    "ld.lld",
    "llc",
    "llvm-objdump",
    "mlir-translate",
    "opt",
]

llvm_config.add_tool_substitutions(tools, tool_dirs)
