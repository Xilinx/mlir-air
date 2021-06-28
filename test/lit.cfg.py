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
    = "{}:{}:{}:{}:{}:{}".format(
                            os.path.join(config.air_src_root, "python"),
                            os.path.join(config.air_obj_root),
                            os.path.join(config.air_obj_root, "../npcomp/python"),
                            os.path.join(config.air_obj_root, "../peano/python"),
                            os.path.join(config.air_obj_root, "/python"),
                            os.path.join(config.air_obj_root, "../air")
                          )
#os.environ['PYTHONPATH']
print("PATH",config.environment['PYTHONPATH'])

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.lit']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.air_obj_root, 'test')

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))
config.substitutions.append(('%PYTHON', config.python_executable))
config.substitutions.append(('%VITIS_SYSROOT%', config.vitis_sysroot))
config.substitutions.append(('%aie_runtime_lib%', os.path.join(config.aie_obj_root, "runtime_lib")))
config.substitutions.append(('%air_runtime_lib%', os.path.join(config.aie_obj_root, "runtime_lib")))

if(config.enable_board_tests):
    config.substitutions.append(('%run_on_board', "echo %T >> /home/xilinx/testlog | sync | sudo"))
else:
    config.substitutions.append(('%run_on_board', "echo"))

llvm_config.with_system_environment(
    ['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ['Inputs', 'Examples', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt', 'lit.cfg.py']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.air_obj_root, 'test')
config.aie_tools_dir = os.path.join(config.aie_obj_root, 'bin')
config.air_tools_dir = os.path.join(config.air_obj_root, 'bin')

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)

tool_dirs = [config.aie_tools_dir, config.llvm_tools_dir]
tools = [
    'aie-opt',
    'aie-translate',
    'aiecc.py',
    'ld.lld',
    'llc',
    'llvm-objdump',
    'opt',
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

lit_config.parallelism_groups["board"] = 1
config.parallelism_group = "board"
