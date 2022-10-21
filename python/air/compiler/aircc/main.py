# ./python/air/compiler/aircc/main.py -*- Python -*-

# Copyright (C) 2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""
aircc - AIR compiler driver for MLIR tools
"""

import os
import platform
import sys
import subprocess
from joblib import Parallel, delayed
import tempfile

from air.mlir.passmanager import PassManager
from air.mlir.ir import Module

import air.compiler.aircc.cl_arguments as cl_arguments

def emit_wrapper(herd_name="partition", include_name="aie.inc"):
    s = """// generated by aircc, do not edit
#include "stdio.h"
#include "assert.h"
#include "air_host.h"

namespace air {
namespace partitions {
"""
    s = s + f'namespace {herd_name} {{\n'
    s = s + f'#include "{include_name}"'
    s = s + """
}
}
}
"""
    s = s + f'using namespace air::partitions::{herd_name};'
    s = s + """
extern "C" {
"""
    s = s + f'air_rt_aie_functions_t __airrt_{herd_name}_aie_functions {{'
    s = s + """
  .configure_cores = &mlir_aie_configure_cores,
  .configure_switchboxes = &mlir_aie_configure_switchboxes,
  .initialize_locks = &mlir_aie_initialize_locks,
  .configure_dmas = &mlir_aie_configure_dmas,
  .start_cores = &mlir_aie_start_cores
};
}
"""
    return s

def do_call(command):
    global opts
    if opts.verbose:
        print(" ".join(command))
    ret = subprocess.call(command)
    if ret != 0:
        print("Error encountered while running: " + " ".join(command))
        sys.exit(1)

def do_run(command):
    global opts
    if opts.verbose:
        print(" ".join(command))
    ret = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    return ret

def run_passes(pass_pipeline, mlir_module, opts, outputfile=None):
  if opts.verbose:
    print("Running:", pass_pipeline)
  PassManager.parse(pass_pipeline).run(mlir_module)
  if outputfile:
    with open(outputfile, 'w') as g:
      g.write(str(mlir_module))

def run(mlir_module, args):
  global opts
  opts = cl_arguments.parse_args(args)
  if opts.tmpdir:
    tmpdirname = opts.tmpdir
    try:
      os.mkdir(tmpdirname)
    except FileExistsError:
      pass
    if opts.verbose:
      print('created temporary directory', tmpdirname)

  with mlir_module.context as ctx:
    m = Module.parse(str(mlir_module))
    air_to_aie_pass = 'air-to-aie{emit-while-loop=false'
    air_to_aie_pass = air_to_aie_pass + f' row-offset={opts.row_offset} col-offset={opts.col_offset}'
    air_to_aie_pass = air_to_aie_pass + f' output-prefix={opts.tmpdir}/' + '}'

    pass_pipeline = ','.join([
      'air-pipeline-to-affine{lowering-type=getput}',
      'canonicalize', 'cse',
      'func.func(air-renumber-dma)',
      'func.func(convert-linalg-to-loops)',
      air_to_aie_pass
    ])
    run_passes(pass_pipeline, Module.parse(str(m)), opts)

    air_to_airrt_pass = 'air-to-aie{emit-while-loop=true'
    air_to_airrt_pass = air_to_airrt_pass + f' row-offset={opts.row_offset} col-offset={opts.col_offset}'
    air_to_airrt_pass = air_to_airrt_pass + f' output-prefix={opts.tmpdir}/' + '}'

    _,air_mlir_filename = os.path.split(opts.air_mlir_file)
    air_mlir_filename = "torch.mlir"

    # lower the airrt control program to llvm dialect

    aie_ctrl_airrt = opts.tmpdir+'/airrt.'+air_mlir_filename
    pass_pipeline = ','.join([
      'air-pipeline-to-affine{lowering-type=getput}',
      'canonicalize', 'cse',
      'func.func(air-renumber-dma)',
      air_to_airrt_pass,
      'convert-vector-to-llvm',
      'convert-math-to-llvm',
      'lower-affine',
      'air-to-std',
      'air-lower-linalg-tensors',
      'canonicalize', 'cse'
    ])
    run_passes(pass_pipeline, mlir_module, opts, aie_ctrl_airrt)

    aie_ctrl = opts.tmpdir+'/aie_ctrl.'+air_mlir_filename
    pass_pipeline = ','.join([
      'airrt-to-llvm',
      'func-bufferize',
      'func.func(finalizing-bufferize)'
    ])
    run_passes(pass_pipeline, mlir_module, opts, aie_ctrl)

    aie_ctrl_refback = opts.tmpdir+'/refback.'+air_mlir_filename
    pass_pipeline = ','.join([
      'air-pipeline-to-affine{lowering-type=getput}',
      'canonicalize', 'cse',
      'func.func(air-renumber-dma)',
      'convert-vector-to-llvm',
      'convert-math-to-llvm',
      'lower-affine',
      'air-to-std',
      'air-lower-linalg-tensors',
      'canonicalize', 'cse',
      'airrt-to-llvm',
      'canonicalize','cse'
    ])
    run_passes(pass_pipeline, Module.parse(str(m)), opts, aie_ctrl_refback)

    aie_ctrl_llvm = opts.tmpdir+'/llvm.'+air_mlir_filename
    pass_pipeline = ','.join([
      #'air-return-elimination',
      'lower-affine',
      'convert-scf-to-cf',
      'convert-memref-to-llvm',
      'convert-func-to-llvm',
      'convert-cf-to-llvm',
      'canonicalize','cse'
    ])
    run_passes(pass_pipeline, mlir_module, opts, aie_ctrl_llvm)

    # compile the llvm dialect into a .o object file

    aie_ctrl_llvm_ir = opts.tmpdir+'/'+air_mlir_filename+'.ll'
    do_call(['aie-translate', '--mlir-to-llvmir', aie_ctrl_llvm, '-o', aie_ctrl_llvm_ir])

    aie_ctrl_llvm_opt_bc = opts.tmpdir+'/'+air_mlir_filename+'.opt.bc'
    do_call(['opt', '-O3', aie_ctrl_llvm_ir, '-o', aie_ctrl_llvm_opt_bc])

    aie_ctrl_llvm_opt_ir = opts.tmpdir+'/'+air_mlir_filename+'.opt.ll'
    do_call(['llvm-dis', aie_ctrl_llvm_opt_bc, '-o', aie_ctrl_llvm_opt_ir])

    aie_ctrl_obj = opts.tmpdir+'/'+air_mlir_filename+'.o'
    do_call(['clang', '-Wno-override-module', '-fPIC'] +
            (['--target', opts.host_target] if opts.host_target else []) +
            ['-c', aie_ctrl_llvm_opt_ir, '-o', aie_ctrl_obj])

    # make aie elf files and host .o files for each herd in the program

    t = do_run(['air-translate', '--airrt-generate-json', aie_ctrl_airrt])
    module_meta = eval(t.stdout)
    herds = [module_meta[herd]["sym_name"] for herd in module_meta]
    obj_files = [aie_ctrl_obj]
    for herd in herds:
      if opts.verbose:
        print ("Compiling partition:", herd)

      # build the elf files for the herd

      herd_file = opts.tmpdir+'/aie.'+herd+'.mlir'
      aiecc_file = opts.tmpdir+'/aiecc.'+herd+'.mlir'
      aiecc_dir = opts.tmpdir+'/'+herd
      do_call(['air-opt', herd_file, '-air-lower-linalg-tensors', '--lower-affine', '-cse', '-o', aiecc_file])
      if 'x86_64' in platform.uname()[5]:
        aiecc_target = "x86_64-amd-linux-gnu"
      else:
        aiecc_target = "aarch64-linux-gnu"
      do_call(['aiecc.py'] +
              (['-v'] if opts.verbose else []) +
              (['--sysroot', opts.sysroot] if opts.sysroot else ['--sysroot=/']) +
              ['--host-target', opts.host_target if opts.host_target else aiecc_target] +
              ['--tmpdir', aiecc_dir] +
              ['--pathfinder', '--aie-generate-xaiev2'] +
              ['--no-xbridge', '--no-xchesscc', aiecc_file])

      inc_file = opts.tmpdir+'/'+air_mlir_filename+'.'+herd+'.inc'
      cpp_file = opts.tmpdir+'/'+air_mlir_filename+'.'+herd+'.cpp'
      obj_file = opts.tmpdir+'/'+air_mlir_filename+'.'+herd+'.o'

      # compile the libxaie configuration functions generated by aie-translate

      do_call(['cp',aiecc_dir+'/aie_inc.cpp',inc_file])

      with open(cpp_file, 'w') as f:
        f.write(emit_wrapper(herd, inc_file))

      cmd = [opts.cc, '-std=c++11', '-g']
      cmd += ['--sysroot=%s' % opts.sysroot] if opts.sysroot else []
      cmd += ['--target=%s' % opts.host_target] if opts.host_target else []
      cmd += ['-I.', f'-I{opts.sysroot}/opt/xaienginev2/include']
      thispath = os.path.dirname(os.path.realpath(__file__))
      cmd += [f'-I{thispath}/../../../../runtime_lib/airhost/include']
      cmd += [f'-I{thispath}/../../../../runtime_lib']
      cmd += ['-DLIBXAIENGINEV2']
      cmd += ['-DAIE_LIBXAIE_ENABLE', '-fPIC', '-c']
      cmd += ['-o', obj_file, cpp_file]
      do_call(cmd)

      obj_files.append(obj_file)

    # combine the host side .o files generated above into a single library

    lib_file = opts.tmpdir+'/'+opts.air_mlir_file+('.so' if opts.shared else '.a')
    if opts.shared:
      cmd = ['clang', '-shared']
      cmd += ['--sysroot', opts.sysroot] if opts.sysroot else []
      cmd += ['--target', opts.host_target] if opts.host_target else []
      cmd += ['-fuse-ld=lld', '-o', lib_file] + obj_files
    else:
      cmd = ['llvm-ar', 'rc', lib_file] + obj_files
    do_call(cmd)

    if opts.output_file:
      do_call(['cp', lib_file, opts.output_file])

def run_flow(opts):
    thispath = os.path.dirname(os.path.realpath(__file__))
    air_to_aie_pass = '-air-to-aie=emit-while-loop=false'
    air_to_aie_pass = air_to_aie_pass + f' row-offset={opts.row_offset} col-offset={opts.col_offset}'
    air_to_aie_pass = air_to_aie_pass + f' output-prefix={opts.tmpdir}/'
    
    do_call(['air-opt', opts.air_mlir_file,
             '-air-pipeline-to-affine=lowering-type=getput', '-canonicalize', '-cse',
             '-air-renumber-dma', air_to_aie_pass, '-o', '/dev/null'])

    air_to_airrt_pass = '-air-to-aie=emit-while-loop=false'
    air_to_airrt_pass = air_to_airrt_pass + f' row-offset={opts.row_offset} col-offset={opts.col_offset}'
    air_to_airrt_pass = air_to_airrt_pass + f' output-prefix={opts.tmpdir}/'

    _,air_mlir_filename = os.path.split(opts.air_mlir_file)
    aie_ctrl_airrt = opts.tmpdir+'/airrt.'+air_mlir_filename
    do_call(['air-opt', opts.air_mlir_file,
            '-air-pipeline-to-affine=lowering-type=getput', '-canonicalize', '-cse',
            '-air-renumber-dma', air_to_airrt_pass,
            '-convert-vector-to-llvm', '-convert-math-to-llvm', '--lower-affine', '-air-to-std',
            '-air-lower-linalg-tensors', '-canonicalize', '-cse',
            '-o', aie_ctrl_airrt])

    aie_ctrl = opts.tmpdir+'/aie_ctrl.'+air_mlir_filename
    do_call(['air-opt', aie_ctrl_airrt,
            '-airrt-to-llvm', '-func-bufferize', '-finalizing-bufferize',
            '-o', aie_ctrl])

    aie_ctrl_llvm = opts.tmpdir+'/llvm.'+air_mlir_filename
    do_call(['air-opt', aie_ctrl,
            '-air-return-elimination','--lower-affine','--convert-scf-to-cf',
            '--convert-memref-to-llvm',
            '--convert-func-to-llvm',
            '--convert-cf-to-llvm',
            '--canonicalize', '--cse',
            '-o', aie_ctrl_llvm])

    aie_ctrl_llvm_ir = opts.tmpdir+'/'+air_mlir_filename+'.ll'
    do_call(['aie-translate', '--mlir-to-llvmir', aie_ctrl_llvm, '-o', aie_ctrl_llvm_ir])

    aie_ctrl_llvm_opt_bc = opts.tmpdir+'/'+air_mlir_filename+'.opt.bc'
    do_call(['opt', '-O3', aie_ctrl_llvm_ir, '-o', aie_ctrl_llvm_opt_bc])

    aie_ctrl_llvm_opt_ir = opts.tmpdir+'/'+air_mlir_filename+'.opt.ll'
    do_call(['llvm-dis', aie_ctrl_llvm_opt_bc, '-o', aie_ctrl_llvm_opt_ir])

    aie_ctrl_obj = opts.tmpdir+'/'+air_mlir_filename+'.o'
    do_call(['clang', '-Wno-override-module', '-fPIC'] +
            (['--target', opts.host_target] if opts.host_target else []) +
            ['-c', aie_ctrl_llvm_opt_ir, '-o', aie_ctrl_obj])

    t = do_run(['air-translate', '--airrt-generate-json', aie_ctrl_airrt])

    module_meta = eval(t.stdout)
    herds = [module_meta[herd]["sym_name"] for herd in module_meta]
    print ("Compiling partitions:", herds)
    obj_files = [aie_ctrl_obj]
    for herd in herds:
      herd_file = opts.tmpdir+'/aie.'+herd+'.mlir'
      aiecc_file = opts.tmpdir+'/aiecc.'+herd+'.mlir'
      aiecc_dir = opts.tmpdir+'/'+herd
      do_call(['air-opt', herd_file, '-air-lower-linalg-tensors', '--lower-affine', '-cse', '-o', aiecc_file])
      if 'x86_64' in platform.uname()[5]:
        aiecc_target = "x86_64-amd-linux-gnu"
      else:
        aiecc_target = "aarch64-linux-gnu"
      do_call(['aiecc.py'] +
              (['-v'] if opts.verbose else []) +
              (['--sysroot', opts.sysroot] if opts.sysroot else ['--sysroot=/']) +
              ['--host-target', opts.host_target if opts.host_target else aiecc_target] +
              ['--tmpdir', aiecc_dir] +
              ['--pathfinder', '--aie-generate-xaiev2'] +
              ['--no-xbridge', '--no-xchesscc', aiecc_file])

      inc_file = opts.tmpdir+'/'+air_mlir_filename+'.'+herd+'.inc'
      cpp_file = opts.tmpdir+'/'+air_mlir_filename+'.'+herd+'.cpp'
      obj_file = opts.tmpdir+'/'+air_mlir_filename+'.'+herd+'.o'

      do_call(['cp',aiecc_dir+'/aie_inc.cpp',inc_file])

      with open(cpp_file, 'w') as f:
        f.write(emit_wrapper(herd, inc_file))

      cmd = [opts.cc, '-std=c++11', '-g']
      cmd += ['--sysroot=%s' % opts.sysroot] if opts.sysroot else []
      cmd += ['--target=%s' % opts.host_target] if opts.host_target else []
      cmd += ['-I.', f'-I{opts.sysroot}/opt/xaienginev2/include']
      cmd += [f'-I{thispath}/../../../../runtime_lib/airhost/include']
      cmd += [f'-I{thispath}/../../../../runtime_lib']
      cmd += [f'-I{thispath}/../../../../../aie/runtime_lib']
      cmd += ['-DLIBXAIENGINEV2']
      cmd += ['-DAIE_LIBXAIE_ENABLE', '-fPIC', '-c']
      cmd += ['-o', obj_file, cpp_file]
      do_call(cmd)

      obj_files.append(obj_file)

    lib_file = opts.air_mlir_file+('.so' if opts.shared else '.a')
    if opts.shared:
      cmd = ['clang', '-shared']
      cmd += ['--sysroot', opts.sysroot] if opts.sysroot else []
      cmd += ['--target', opts.host_target] if opts.host_target else []
      cmd += ['-fuse-ld=lld', '-o', lib_file] + obj_files
    else:
      cmd = ['llvm-ar', 'rc', lib_file] + obj_files
    do_call(cmd)

    if opts.output_file:
      do_call(['mv', lib_file, opts.output_file])


def main():
    global opts
    opts = cl_arguments.parse_args()
    is_windows = platform.system() == 'Windows'

    if opts.verbose:
        sys.stderr.write('\ncompiling %s\n' % opts.air_mlir_file)

    if opts.tmpdir:
      tmpdirname = opts.tmpdir
      try:
        os.mkdir(tmpdirname)
      except FileExistsError:
        pass
      if opts.verbose:
        print('created temporary directory', tmpdirname)

      run_flow(opts)
    else:
      with tempfile.TemporaryDirectory() as tmpdirname:
        run_flow(opts)
