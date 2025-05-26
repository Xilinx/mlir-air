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
import shutil
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "AIR_PROGRAMMING_EXAMPLES"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.environment["PYTHONPATH"] = "{}:{}:{}".format(
    os.path.join(config.air_obj_root, "python"),
    os.path.join(config.aie_obj_root, "python"),
    os.path.join(config.xrt_dir, "python"),
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
config.test_exec_root = os.path.join(config.air_obj_root, "programming_examples")
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

# for xchesscc_wrapper
llvm_config.with_environment("AIETOOLS", config.vitis_aietools_dir)

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
        result = subprocess.run(
            [xrtsmi, "examine"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        result = result.stdout.decode("utf-8").split("\n")
        # Older format is "|[0000:41:00.1]  ||RyzenAI-npu1  |"
        # Newer format is "|[0000:41:00.1]  |NPU Phoenix  |"
        p = re.compile(r"[\|]?(\[.+:.+:.+\]).+\|(RyzenAI-(npu\d)|NPU (\w+))\W*\|")
        for l in result:
            m = p.match(l)
            if not m:
                continue
            print("Found Ryzen AI device:", m.group(1))
            model = "unknown"
            if m.group(3):
                model = str(m.group(3))
            if m.group(4):
                model = str(m.group(4))
            print(f"\tmodel: '{model}'")
            config.available_features.add("ryzen_ai")
            run_on_npu = (
                f"flock /tmp/npu.lock {config.air_src_root}/utils/run_on_npu.sh"
            )
            if model in ["npu1", "Phoenix"]:
                run_on_npu1 = run_on_npu
                config.available_features.add("ryzen_ai_npu1")
                print("Running tests on NPU1 with command line: ", run_on_npu)
            elif model in ["npu4", "Strix"]:
                run_on_npu2 = run_on_npu
                config.available_features.add("ryzen_ai_npu2")
                print("Running tests on NPU4 with command line: ", run_on_2npu)
            else:
                print("WARNING: xrt-smi reported unknown NPU model '{model}'.")
            break
    except:
        print("Failed to run xrt-smi")
        pass
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
config.air_tools_dir = os.path.join(config.air_obj_root, "bin")

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.peano_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.aie_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.air_tools_dir, append_path=True)

config.substitutions.append(("%LLVM_TOOLS_DIR", config.llvm_tools_dir))

tool_dirs = [config.aie_tools_dir, config.llvm_tools_dir]

# Test to see if this mlir-air installation is built for peano. If not, then look for Chess in system.
try:
    result = subprocess.run(
        [os.path.join(config.peano_tools_dir, "llc"), "-mtriple=aie", "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if re.search("Xilinx AI Engine", result.stdout.decode("utf-8")) is not None:
        config.available_features.add("peano")
        config.substitutions.append(("%PEANO_INSTALL_DIR", config.peano_tools_dir))
        print("Peano found: " + os.path.join(config.peano_tools_dir, "llc"))
        tool_dirs.append(os.path.join(config.peano_tools_dir, "bin"))
        # Peano compiler flags
        peano_flags = (
            "-O2 -std=c++20 --target=aie2-none-unknown-elf -DNDEBUG -I{}".format(
                os.path.join(config.aie_obj_root, "include")
            )
        )
        config.substitutions.append(("%peano_flags", peano_flags))
    else:
        print("Peano not found, but expected at ", config.peano_tools_dir)
except Exception as e:
    print("Peano not found.")

    if not config.enable_chess_tests:
        print("Chess tests disabled")
    else:
        print("Looking for Chess...")

        result = shutil.which("xchesscc")
        if result != None:
            print("Chess found: " + result)
            config.available_features.add("chess")
            config.available_features.add("valid_xchess_license")
            lm_license_file = os.getenv("LM_LICENSE_FILE")
            if lm_license_file != None:
                llvm_config.with_environment("LM_LICENSE_FILE", lm_license_file)
            xilinxd_license_file = os.getenv("XILINXD_LICENSE_FILE")
            if xilinxd_license_file != None:
                llvm_config.with_environment(
                    "XILINXD_LICENSE_FILE", xilinxd_license_file
                )

            # test if LM_LICENSE_FILE valid
            validate_chess = False
            if validate_chess:
                import subprocess

                result = subprocess.run(
                    ["xchesscc", "+v"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                validLMLicense = len(result.stderr.decode("utf-8")) == 0
            else:
                validLMLicense = lm_license_file or xilinxd_license_file

            if not lm_license_file and not xilinxd_license_file:
                print(
                    "WARNING: no valid xchess license that is required by some of the lit tests"
                )
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
    config.air_tools_dir,
    config.llvm_tools_dir,
]
tools = [
    "aie-opt",
    "aie-translate",
    "aiecc.py",
    "aircc.py",
    "air-opt",
    "ld.lld",
    "llc",
    "llvm-objdump",
    "mlir-translate",
    "opt",
]

llvm_config.add_tool_substitutions(tools, tool_dirs)
