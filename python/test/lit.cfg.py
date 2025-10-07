# ./python/test/lit.cfg.py -*- Python -*-

# Copyright (C) 2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# -*- Python -*-

import os
import sys
import importlib.util
import subprocess
import re

import lit.formats

from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "AIRPYTHON"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.environment["PYTHONPATH"] = "{}:{}:{}".format(
    os.path.join(config.air_obj_root, "python"),
    os.path.join(config.aie_obj_root, "python"),
    os.path.join(config.xrt_dir, "python"),
)

try:
    import torch_mlir

    config.available_features.add("torch_mlir")
except:
    print("torch_mlir not found")
    pass


spensor_name = "spensor"
# If spensor is already imported or can be imported
if spensor_name in sys.modules or importlib.util.find_spec(spensor_name):
    print(spensor_name + " found")
    config.available_features.add("spensor")

print("Running with PYTHONPATH", config.environment["PYTHONPATH"])

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".py"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.air_obj_root, "python", "test")
air_runtime_lib = os.path.join(config.air_obj_root, "runtime_lib")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))
config.substitutions.append(("%PYTHON", config.python_executable))

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = []

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
            run_on_npu = f"{config.air_src_root}/utils/run_on_npu.sh"
            if model in ["npu1", "Phoenix"]:
                run_on_npu1 = run_on_npu
                config.available_features.add("ryzen_ai_npu1")
                print("Running tests on NPU1 with command line: ", run_on_npu1)
            elif model in ["npu4", "Strix"]:
                run_on_npu2 = run_on_npu
                config.available_features.add("ryzen_ai_npu2")
                print("Running tests on NPU4 with command line: ", run_on_npu2)
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

config.excludes.append("lit.cfg.py")

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

config.aie_tools_dir = os.path.join(config.aie_obj_root, "bin")
config.air_tools_dir = os.path.join(config.air_obj_root, "bin")

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.aie_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.air_tools_dir, append_path=True)

tool_dirs = [config.aie_tools_dir, config.air_tools_dir, config.llvm_tools_dir]
tools = [
    "aie-opt",
    "aie-translate",
    "aiecc.py",
    "aircc.py",
    "air-opt",
    "clang",
    "clang++",
    "ld.lld",
    "llc",
    "llvm-objdump",
    "mlir-translate",
    "opt",
]

llvm_config.add_tool_substitutions(tools, tool_dirs)
