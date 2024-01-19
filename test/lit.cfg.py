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

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'AIR_TEST'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.environment['PYTHONPATH'] \
    = "{}:{}:{}".format(os.path.join(config.air_obj_root, "python"),
                     os.path.join(config.aie_obj_root, "python"),
                     os.path.join(config.xrt_dir, "python"))

#os.environ['PYTHONPATH']
print("Running with PYTHONPATH",config.environment['PYTHONPATH'])

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.lit']

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = []

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.air_obj_root, 'test')
air_runtime_lib = os.path.join(config.air_obj_root, "runtime_lib", config.test_arch)

config.substitutions.append(('%PYTHON', config.python_executable))
config.substitutions.append(('%CLANG', "clang++ -fuse-ld=lld -DLIBXAIENGINEV2"))
config.substitutions.append(('%LIBXAIE_DIR%', config.libxaie_dir))
config.substitutions.append(('%AIE_RUNTIME_DIR%', os.path.join(config.aie_obj_root, "runtime_lib", config.test_arch)))

if config.hsa_found:
    # Getting the path to the ROCm directory. hsa-runtime64 points to the cmake
    # directory so need to go up three directories
    rocm_root = os.path.join(config.hsa_dir, "..", "..", "..")
    print("Found ROCm:", rocm_root)
    config.substitutions.append(('%HSA_DIR%', "{}".format(rocm_root)))
    config.substitutions.append(('%airhost_libs%',
                                 " -I" + air_runtime_lib + "/airhost/include" +
                                 " -L" + air_runtime_lib + "/airhost -Wl,--whole-archive -lairhost" +
                                 " -Wl,-R{}/lib -Wl,-rpath,{}/lib -Wl,--whole-archive" +
                                 " -Wl,--no-whole-archive -lpthread -lstdc++ -lsysfs -ldl -lrt -lelf".format(config.libxaie_dir, rocm_root)))
    if config.enable_run_airhost_tests:
        config.substitutions.append(('%run_on_board', "sudo"))
    else:
        print("Skipping execution of airhost tests (ENABLE_RUN_AIRHOST_TESTS=OFF)")
        config.substitutions.append(('%run_on_board', "echo"))
else:
    print("ROCm not found")
    config.excludes.append('airhost')


# XRT
if config.xrt_lib_dir:
    print("xrt found at", os.path.dirname(config.xrt_lib_dir))
    xrt_flags = "-I{} -L{} -luuid -lxrt_coreutil".format(
        config.xrt_include_dir, config.xrt_lib_dir
    )
    config.available_features.add("xrt")

    run_on_ipu = "echo"
    try:
        xbutil = os.path.join(config.xrt_bin_dir, "xbutil")
        result = subprocess.run(
            [xbutil, "examine"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        result = result.stdout.decode("utf-8").split("\n")
        p = re.compile("\[.+:.+:.+\].+Phoenix.+Yes")
        for l in result:
            m = p.match(l)
            if m:
                print("Found Ryzen AI device:", m.group().split()[0])
                config.available_features.add("ryzen_ai")
                if config.enable_run_xrt_tests:
                    run_on_ipu = "flock /tmp/ipu.lock /opt/xilinx/run_on_ipu.sh"
                else:
                    print("Skipping execution of Ryzen AI tests (ENABLE_RUN_XRT_TESTS=OFF)")
                break
    except:
        print("Failed to run xbutil")
        pass
    config.substitutions.append(("%run_on_ipu", run_on_ipu))
    config.substitutions.append(("%xrt_flags", xrt_flags))
    config.substitutions.append(("%XRT_DIR", config.xrt_dir))
else:
    print("xrt not found")
    config.excludes.append('xrt')

llvm_config.with_system_environment(
    ['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

llvm_config.use_default_substitutions()

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.air_obj_root, 'test')
config.aie_tools_dir = os.path.join(config.aie_obj_root, 'bin')
config.air_tools_dir = os.path.join(config.air_obj_root, 'bin')

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)
llvm_config.with_environment('PATH', config.peano_tools_dir, append_path=True)
llvm_config.with_environment('PATH', config.aie_tools_dir, append_path=True)
llvm_config.with_environment('PATH', config.air_tools_dir, append_path=True)

# test if LM_LICENSE_FILE valid
if config.enable_chess_tests:
    import shutil
    result = None
    if config.vitis_root:
        result = shutil.which("xchesscc")

    import subprocess
    if result != None:
        result = subprocess.run(['xchesscc','+v'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        validLMLicense = (len(result.stderr.decode('utf-8')) == 0)
    else:
        validLMLicense = False

    if validLMLicense:
        config.available_features.add('valid_xchess_license')
        lm_license_file = os.getenv('LM_LICENSE_FILE')
        if(lm_license_file != None):
            llvm_config.with_environment('LM_LICENSE_FILE', lm_license_file)
        xilinxd_license_file = os.getenv('XILINXD_LICENSE_FILE')
        if(xilinxd_license_file != None):
            llvm_config.with_environment('XILINXD_LICENSE_FILE', xilinxd_license_file)
    else:
        print("WARNING: no valid xchess license that is required by some of the lit tests")


if config.vitis_root:
  llvm_config.with_environment('CARDANO', config.vitis_aietools_dir)
  llvm_config.with_environment('VITIS', config.vitis_root)

tool_dirs = [config.peano_tools_dir, config.aie_tools_dir, config.air_tools_dir, config.llvm_tools_dir]
tools = [
    'aie-opt',
    'aie-translate',
    'aiecc.py',
    'aircc.py',
    'air-opt',
    'clang',
    'clang++',
    'ld.lld',
    'llc',
    'llvm-objdump',
    'mlir-translate',
    'opt',
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

if config.enable_run_airhost_tests:
    lit_config.parallelism_groups["board"] = 1
    config.parallelism_group = "board"
