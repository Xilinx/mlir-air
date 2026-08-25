# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import os
import platform
import re
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "AIRMLIR"

# The internal shell, unconditionally. `not llvm_config.use_lit_shell` was the
# old LLVM idiom for "external shell except on Windows", and lit 23 removed the
# external shell: ShTest(execute_external=True) now raises at config-parse time,
# so the suite fails before a single test runs.
#
# Migrating rather than setting force_execute_external=True, which is an
# explicit stay of execution that LLVM-24 removes anyway. The one incompatibility
# in this repo was a bare `VAR=value` command prefix, which bash treats as an
# environment assignment and lit's internal shell treats as a command name; see
# the %ld_lib_path substitution in mlir/test/lit.cfg.py.
config.test_format = lit.formats.ShTest()

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.air_obj_root, "test")
air_runtime_lib = os.path.join(
    config.air_obj_root, "runtime_lib", config.runtime_test_target
)

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))
config.substitutions.append(("%CLANG", "clang"))
config.substitutions.append(("%LLC", config.llvm_tools_dir + "/llc"))
config.substitutions.append(("%OPT", config.llvm_tools_dir + "/opt"))
config.substitutions.append(
    ("%airhost_inc", "-I" + air_runtime_lib + "/airhost/include")
)
config.substitutions.append(
    ("%aircpu_lib", "-L" + air_runtime_lib + "/aircpu -laircpu")
)
config.substitutions.append(
    ("%mlir_async_lib", "-L" + config.llvm_obj_root + "/lib -lmlir_async_runtime")
)
# Used as a command prefix: `// RUN: %ld_lib_path %t.test.exe`. The leading
# `env` is load-bearing. A bare `VAR=value` prefix is a shell feature -- bash
# reads it as an environment assignment for the following command -- and lit's
# internal shell does not implement it, taking `LD_LIBRARY_PATH=...` as the name
# of the program to run and failing with "command not found". It does implement
# the `env` builtin, and `env` is also a real binary, so this spelling works
# under either shell.
config.substitutions.append(
    (
        "%ld_lib_path",
        "env LD_LIBRARY_PATH="
        + air_runtime_lib
        + "/aircpu:"
        + config.llvm_obj_root
        + "/lib",
    )
)

# Tests that link a host executable against the aircpu runtime can only run
# where that runtime was built. It needs POSIX APIs, so the top-level
# CMakeLists skips runtime_lib on Windows; gate on the artifact rather than on
# the platform, which also covers a POSIX build with the runtimes disabled.
if os.path.isdir(os.path.join(air_runtime_lib, "aircpu")):
    config.available_features.add("aircpu")

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ["Inputs", "Examples", "CMakeLists.txt", "README.txt", "LICENSE.txt"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.air_obj_root, "test")
config.air_tools_dir = os.path.join(config.air_obj_root, "bin")

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.aie_tools_dir, append_path=True)

tool_dirs = [config.air_tools_dir, config.aie_tools_dir, config.llvm_tools_dir]
tools = ["air-opt", "air-translate", "air-runner", "aie-opt", "aircc"]

llvm_config.add_tool_substitutions(tools, tool_dirs)

if config.air_enable_gpu:
    config.available_features.add("gpu")
