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
        out = result.stdout.decode("utf-8")
        out_lc = out.lower()
        # Case-insensitive substring match against known NPU model strings,
        # aligned with mlir-aie's hostruntime.py and the AIR XRT backend
        # (NPU_MODELS). Robust to xrt-smi table-format changes and covers newer
        # parts (Strix Halo, Krackan) that the old strict regex missed.
        npu2_models = ["npu4", "strix", "npu5", "strix halo", "npu6", "krackan"]
        npu1_models = ["npu1", "phoenix"]
        run_on_npu = f"flock /tmp/npu.lock {config.air_src_root}/utils/run_on_npu.sh"
        if any(k in out_lc for k in npu2_models):
            config.available_features.add("ryzen_ai")
            config.available_features.add("ryzen_ai_npu2")
            run_on_npu2 = run_on_npu
            print("Running tests on NPU2 with command line: ", run_on_npu2)
        elif any(k in out_lc for k in npu1_models):
            config.available_features.add("ryzen_ai")
            config.available_features.add("ryzen_ai_npu1")
            run_on_npu1 = run_on_npu
            print("Running tests on NPU1 with command line: ", run_on_npu)
        else:
            # No recognized model: dump xrt-smi output so the cause (format
            # change, or a driver error such as an mmap/memlock failure) is
            # visible instead of silently skipping every NPU test.
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

# Tests that download Hugging Face Hub gated models (e.g. meta-llama/*) need
# HF_TOKEN to be set. Mark `hf_token` as available only when the env var is
# present so REQUIRES: hf_token tests skip cleanly on machines without it.
if os.environ.get("HF_TOKEN"):
    config.available_features.add("hf_token")
    llvm_config.with_environment("HF_TOKEN", os.environ["HF_TOKEN"])
    print("HF_TOKEN found in environment; hf_token feature enabled.")
else:
    print("HF_TOKEN not set; hf_token feature disabled.")

# Forward HF Hub download tuning if the host set it (e.g. the perf runner sets
# HF_HUB_DISABLE_XET=1 because the hf_xet backend stalls there). Propagated only
# when present, so it's a no-op on hosts that don't set it.
if os.environ.get("HF_HUB_DISABLE_XET"):
    llvm_config.with_environment("HF_HUB_DISABLE_XET", os.environ["HF_HUB_DISABLE_XET"])

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
        peano_flags = "-O2 -std=c++20 -DNDEBUG -I{}".format(
            os.path.join(config.aie_obj_root, "include")
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
    config.air_tools_dir,
    config.llvm_tools_dir,
]
tools = [
    "aie-opt",
    "aie-translate",
    "aiecc.py",
    "aircc",
    "air-opt",
    "ld.lld",
    "llc",
    "llvm-objdump",
    "mlir-translate",
    "opt",
]

llvm_config.add_tool_substitutions(tools, tool_dirs)
