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

# for xchesscc_wrapper
llvm_config.with_environment("AIETOOLS", config.vitis_aietools_dir)

# XRT
if config.xrt_lib_dir and config.enable_run_xrt_tests:
    print("xrt found at", os.path.dirname(config.xrt_lib_dir))
    xrt_flags = "-I{} -L{} -luuid -lxrt_coreutil".format(
        config.xrt_include_dir, config.xrt_lib_dir
    )
    config.available_features.add("xrt")

    run_on_npu = "echo"
    try:
        xbutil = os.path.join(config.xrt_bin_dir, "xbutil")
        result = subprocess.run(
            [xbutil, "examine"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        result = result.stdout.decode("utf-8").split("\n")
        # Starting with Linux 6.8 the format is like "[0000:66:00.1]  :  RyzenAI-npu1"
        # Starting with Linux 6.10 the format is like "|[0000:41:00.1]  ||RyzenAI-npu1  |"
        p = re.compile(r"[\|]?(\[.+:.+:.+\]).+(Phoenix|RyzenAI-(npu\d))")
        for l in result:
            m = p.match(l)
            if m:
                print("Found Ryzen AI device:", m.group(1))
                if len(m.groups()) == 3:
                    print("\tmodel:", m.group(3))
                config.available_features.add("ryzen_ai")
                run_on_npu = (
                    f"flock /tmp/npu.lock {config.air_src_root}/utils/run_on_npu.sh"
                )
    except:
        print("Failed to run xbutil")
        pass
    config.substitutions.append(("%run_on_npu", run_on_npu))
    config.substitutions.append(("%xrt_flags", xrt_flags))
    config.substitutions.append(("%XRT_DIR", config.xrt_dir))
else:
    print("xrt not found or xrt tests disabled")
    config.excludes.append("xrt")

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
            llvm_config.with_environment("XILINXD_LICENSE_FILE", xilinxd_license_file)

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
