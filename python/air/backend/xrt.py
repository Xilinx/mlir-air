# ./python/air/backend/xrt_backend.py -*- Python -*-
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import air.ir
import air.passmanager

from .abc import AirBackend, AirBackendError

import air.compiler.util
import air.compiler.aircc.main as aircc

import numpy as np
import pyxrt as xrt
import os

from ml_dtypes import bfloat16


class XRTCompileArtifact:
    """A class encompassing information on the artifacts produced by compilation for the NPU/XRT"""

    def __init__(
        self,
        xclbin,
        kernel,
        insts,
    ):
        """
        Constructor for an XRTCompileArtifact

        Args:
            xclbin: xclbin file name/path
            kernel: kernel name
            insts: instruction file name/path
        """
        self.xclbin = xclbin
        self.kernel = kernel
        self.insts = insts


class XRTBackend(AirBackend):
    """Main entry-point for the xrt based AIR backend."""

    def __init__(
        self,
        verbose=False,
        experimental_passes=False,
        omit_while_true_loop=False,
        omit_pingpong=False,
    ):
        """Constructor for XRTBackend

        Args:
            verbose: verbose output
            experimental_passes: configure aircc to run additional experimental passes
            omit_while_true_loop: configure aircc to omit the while true loop it traditionally emits.
            omit_pingpong: configure aircc to omit the generation of ping-pong buffering.
        """
        super().__init__()
        self.verbose = verbose
        self.experimental_passes = experimental_passes
        self.omit_while_true_loop = omit_while_true_loop
        self.omit_pingpong = omit_pingpong
        self.currently_loaded = False

    def __del__(self):
        self.unload()

    def compile(
        self,
        air_module: air.ir.Module,
        xclbin="air.xclbin",
        kernel="MLIR_AIE",
        insts="air.insts.txt",
    ):
        """Compiles an AIR module for the NPU / XRT Runtime with aircc.

        The module is expected to be AIR dialect IR. The input IR is passed directly to aircc.

        Args:
            air_module: The MLIR module consisting of funcs in the AIR dialect.
            xclbin: xclbin filename to use
            kernel: kernel name to use
            insts: instruction filename to use
        Returns:
            An XRTCompileArtifact object
        """
        if self.currently_loaded:
            raise AirBackendError(
                "Cannot use XRTBackend to compile while the artifact is currently loaded. Call unload() first."
            )

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
                xclbin,
                "-i",
                insts,
            ]

            if self.verbose:
                aircc_options = aircc_options + ["-v"]

            if self.experimental_passes:
                aircc_options += ["--experimental-passes"]

            if self.omit_while_true_loop:
                aircc_options += ["--omit-while-true-loop"]

            if self.omit_pingpong:
                aircc_options += ["--omit-ping-pong-transform"]

            aircc.run(air_module, aircc_options)

        return XRTCompileArtifact(xclbin, kernel, insts)

    def load(self, artifact: XRTCompileArtifact):
        """Load a compiled artifact into the air runtime.

        Args:
            artifact: The result of calling compile with XRTBackend on an MLIR-AIR module.

        Returns: A callable that can be used to invoke the loaded module.
            The callable takes a list of numpy arrays. Each numpy array is
            assumed to be an input/output tensor. The callable also returns a
            list of numpy arrays, one for each tensor.
        """
        if self.currently_loaded:
            raise AirBackendError(
                "Cannot use XRTBackend to compile while the artifact is currently loaded. Call unload() first."
            )

        if not os.path.isfile(artifact.xclbin):
            raise AirBackendError(
                f"Cannot load XRTCompileArtifact because {artifact.xclbin} xclbin file does not exist"
            )
        if not os.path.isfile(artifact.insts):
            raise AirBackendError(
                f"Cannot load XRTCompileArtifact because {artifact.insts} insts file does not exist"
            )

        # create the device, xclbin and context
        self.device = xrt.device(0)
        self.xclbin = xrt.xclbin(artifact.xclbin)
        self.device.register_xclbin(self.xclbin)
        self.context = xrt.hw_context(self.device, self.xclbin.get_uuid())

        # find and load the kernel
        kernels = self.xclbin.get_kernels()
        try:
            xkernel = [k for k in kernels if artifact.kernel in k.get_name()][0]
        except:
            raise AirBackendError(
                f"Kernel '{artifact.kernel}' not found in '{self.xclbin}'"
            )
        self.kernel = xrt.kernel(self.context, xkernel.get_name())

        # load the instructions as a numpy array
        with open(artifact.insts, "r") as f:
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
                if a.dtype == bfloat16:
                    # store bfloat16 in binary as int16
                    a = a.view(np.int16)
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
        """
        Compile and load a module in one step.

        Args:
            air_module: The MLIR module consisting of funcs in the AIR dialect.

        Returns: A callable that can be used to invoke the loaded module.
            The callable takes a list of numpy arrays. Each numpy array is
            assumed to be an input/output tensor. The callable also returns a
            list of numpy arrays, one for each tensor.
        """
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
        self.currently_loaded = False
