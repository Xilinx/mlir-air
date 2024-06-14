# ./python/air/backend/xrt_backend.py -*- Python -*-
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import air.ir
import air.passmanager

from .abc import AirBackend

import air.compiler.util
import air.compiler.aircc.main as aircc

import numpy as np
import pyxrt as xrt


class XRTBackend(AirBackend):
    """Main entry-point for the xrt based AIR backend.

    Args:
      verbose: verbose
      xclbin: xclbin filename to use
      kernel: kernel name to use
      insts: instruction filename to use
    """

    def __init__(
        self,
        verbose=False,
        xclbin="air.xclbin",
        kernel="MLIR_AIE",
        insts="air.insts.txt",
        experimental_passes=False,
    ):
        super().__init__()
        self.opts_xclbin = xclbin
        self.opts_kernel = kernel
        self.opts_insts = insts
        self.verbose = verbose
        self.experimental_passes = False

    def __del__(self):
        self.unload()

    def compile(self, air_module: air.ir.Module, pipeline=None):
        """Compiles an AIR module for the NPU / XRT Runtime with aircc.

        The module is expected to be AIR dialect IR. Unless 'pipeline' is
        specified, the the input IR is passed directly to aircc. If 'pipeline'
        is specified, it is passed to aircc as the 'pipeline' command line options.

        Args:
          air_module: The MLIR module consisting of funcs in the AIR dialect.
          pipeline: aircc optimization pipeline to use.
          verbose: verbose
        Returns:
          An opaque, backend specific compiled artifact object that can be
          passed to `load`.
        """

        with air.ir.Context():

            if self.verbose:
                print("AIR Module:")
                print(air_module)

            aircc_options = [
                "--device",
                "npu1_4col",
                "air.mlir",
                "-xchesscc",
                "-xbridge",
                "-o",
                self.opts_xclbin,
            ]

            if self.verbose:
                aircc_options = aircc_options + ["-v"]

            if self.experimental_passes:
                aircc_options += ["--experimental-passes"]

            aircc.run(air_module, aircc_options)

        return air_module

    def load(self, module):
        """Load a compiled artifact into the air runtime.

        Returns: A callable that can be used to invoke the loaded module.
            The callable takes a list of numpy arrays. Each numpy array is
            assumed to be an input/output tensor. The callable also returns a
            list of numpy arrays, one for each tensor."""

        # create the device, xclbin and context
        self.device = xrt.device(0)
        self.xclbin = xrt.xclbin(self.opts_xclbin)
        self.device.register_xclbin(self.xclbin)
        self.context = xrt.hw_context(self.device, self.xclbin.get_uuid())

        # find and load the kernel
        kernels = self.xclbin.get_kernels()
        try:
            xkernel = [k for k in kernels if self.opts_kernel in k.get_name()][0]
        except:
            print(f"Kernel '{self.opts_kernel}' not found in '{self.opts_xclbin}'")
            exit(-1)
        self.kernel = xrt.kernel(self.context, xkernel.get_name())

        # load the instructions as a numpy array
        with open(self.opts_insts, "r") as f:
            instr_text = f.read().split("\n")
            instr_text = [l for l in instr_text if l != ""]
            self.instr_v = np.array([int(i, 16) for i in instr_text], dtype=np.uint32)

        self.bo_instr = xrt.bo(
            self.device,
            len(self.instr_v) * 4,
            xrt.bo.cacheable,
            self.kernel.group_id(1),
        )
        self.bo_instr.write(self.instr_v, 0)

        # 1) create and sync the buffers
        # 2) invoke the kernel
        # 3) sync the buffers
        # 4) return the contents of the buffers
        def invoker(*args):

            # limit arg length to 5
            if len(args) > 5:
                raise ValueError("Too many arguments")
            sizes_in_bytes = [a.size * a.itemsize for a in args]
            bos = [
                xrt.bo(self.device, s, xrt.bo.host_only, self.kernel.group_id(i + 3))
                for i, s in enumerate(sizes_in_bytes)
            ]

            self.bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            for i, a in enumerate(args):
                bos[i].write(a, 0)
                bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

            h = self.kernel(3, self.bo_instr, len(self.instr_v), *bos)
            h.wait()

            for i, a in enumerate(args):
                bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            return tuple(
                [
                    bos[i].read(s, 0).view(args[i].dtype)
                    for i, s in enumerate(sizes_in_bytes)
                ]
            )

        return invoker

    def compile_and_load(self, module):
        """Compile and load a module in one step."""
        c = self.compile(module)
        return self.load(c)

    def unload(self):
        """Unload any loaded module and shutdown the air runtime."""
        self.kernel = None
        self.context = None
        self.xclbin = None
        self.device = None
        self.bo_instr = None
        self.instr_v = None
