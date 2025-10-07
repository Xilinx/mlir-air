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
        verbose: bool = False,
        omit_while_true_loop: bool = False,
        omit_pingpong: bool = False,
        lower_linalg_to_func: str = None,
        air_loop_fusion: bool = False,
        runtime_loop_tiling_sizes: list[int] = [4, 4],
        omit_auto_broadcast: bool = False,
        channel_multiplexing: list[str] = [],
        use_lock_race_condition_fix: bool = False,
        trace_offset: int = 0,
        trace_size: int = 0,
        output_format: str = "xclbin",
        kernel_name: str = "",
        instance_name: str = "",
        kernel_id: str = "",
        xclbin_input: str = "",
    ):
        """Constructor for XRTBackend

        Args:
            verbose: verbose output
            omit_while_true_loop: configure aircc to omit the while true loop it traditionally emits.
            omit_pingpong: configure aircc to omit the generation of ping-pong buffering.
            lower_linalg_to_func: configure aircc to lower linalg.generic to function calls, or loops.
            air_loop_fusion: configure aircc to add air-loop-fusion experimental pass.
            runtime_loop_tiling_sizes: configure aircc to add extra runtime loop tiling using the experimental affine-loop-opt pass.
            omit_auto_broadcast: configure aircc to omit the detection and lowering of broadcast data movements.
            channel_multiplexing: configure aircc to perform air channel multiplexing on specified memroy spaces.
            use_lock_race_condition_fix: configure aircc to enable a fix for lock race condition which protects against race condition.
            trace_offset: configure aircc to stream out profiling traces at outputs, starting from the specified offset.
            trace_size: configure aircc to stream out profiling traces at outputs, with specified trace data size.
            output_format: configure aircc to produce output binary in to one of the following formats: [xclbin, txn].
            kernel_name: configure aircc to package the kernel with the specified name.
            instance_name: configure aircc to package the kernel with specified instance name in xclbin metadata.
            kernel_id: configure aircc to package the kernel with specified kernel id in xclbin file.
            xclbin_input: configure aircc to package the kernel into an existing xclbin with specified xclbin file name.
        """
        super().__init__()
        self.verbose = verbose
        self.omit_while_true_loop = omit_while_true_loop
        self.omit_pingpong = omit_pingpong
        self.lower_linalg_to_func = lower_linalg_to_func
        self.air_loop_fusion = air_loop_fusion
        self.runtime_loop_tiling_sizes = runtime_loop_tiling_sizes
        self.omit_auto_broadcast = omit_auto_broadcast
        self.channel_multiplexing = channel_multiplexing
        self.use_lock_race_condition_fix = use_lock_race_condition_fix
        self.trace_offset = trace_offset
        self.trace_size = trace_size
        self.currently_loaded = False
        self.output_format = output_format
        self.kernel_name = kernel_name
        self.instance_name = instance_name
        self.kernel_id = kernel_id
        self.xclbin_input = xclbin_input

    def __del__(self):
        self.unload()

    def compile(
        self,
        air_module: air.ir.Module,
        xclbin="air.xclbin",
        kernel="MLIR_AIE",
        insts="air.insts.bin",
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

        # Try to get xrt.
        target_device = "npu1"
        try:
            import subprocess
            import re

            xrtsmi = "/opt/xilinx/xrt/bin/xrt-smi"
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
                if self.verbose:
                    print("Found Ryzen AI device:", m.group(1))
                model = "unknown"
                if m.group(3):
                    model = str(m.group(3))
                if m.group(4):
                    model = str(m.group(4))
                if self.verbose:
                    print(f"\tmodel: '{model}'")
                if model in ["npu1", "Phoenix"]:
                    target_device = "npu1"
                elif model in ["npu4", "Strix"]:
                    target_device = "npu2_4col"
                else:
                    print("WARNING: xrt-smi reported unknown NPU model '{model}'.")
                break
        except Exception as e:
            print("Failed to run xrt-smi")
            print(e)

        import os, site, glob

        # Try to get peano package dir from environment variable, fallback to site-packages
        peano_package_dir = os.environ.get("PEANO_INSTALL_DIR", "")

        if peano_package_dir and os.path.isdir(peano_package_dir):
            print(
                "XRTBackend: llvm-aie package detected via PEANO_INSTALL_DIR:",
                peano_package_dir,
            )

        with air.ir.Context():

            if self.verbose:
                print("AIR Module:")
                print(air_module)

            aircc_options = [
                "--device",
                target_device,
                "air.mlir",
                "-o",
                xclbin,
                "-i",
                insts,
            ]

            aircc_options += ["--air-runtime-loop-tiling-sizes"]
            for s in self.runtime_loop_tiling_sizes:
                aircc_options += [str(s)]

            if self.verbose:
                aircc_options = aircc_options + ["-v"]

            if self.omit_while_true_loop:
                aircc_options += ["--omit-while-true-loop"]

            if self.omit_pingpong:
                aircc_options += ["--omit-ping-pong-transform"]

            if self.lower_linalg_to_func:
                aircc_options += ["--lower-linalg-to-func"]
                aircc_options += [self.lower_linalg_to_func]

            if self.air_loop_fusion:
                aircc_options += ["--air-loop-fusion"]

            if self.omit_auto_broadcast:
                aircc_options += ["--omit-auto-broadcast"]

            if len(self.channel_multiplexing) != 0:
                aircc_options += ["--air-channel-multiplexing"]
                aircc_options += self.channel_multiplexing
            
            if self.use_lock_race_condition_fix:
                aircc_options += ["--use-lock-race-condition-fix"]

            if self.trace_size != 0:
                aircc_options += ["-trace-size"]
                aircc_options += [str(self.trace_size)]
                aircc_options += ["-trace-offset"]
                aircc_options += [str(self.trace_offset)]

            if self.output_format != "":
                aircc_options += ["--output-format"]
                aircc_options += [self.output_format]
            if self.kernel_name != "":
                aircc_options += ["--xclbin-kernel-name"]
                aircc_options += [self.kernel_name]
            if self.instance_name != "":
                aircc_options += ["--xclbin-instance-name"]
                aircc_options += [self.instance_name]
            if self.kernel_id != "":
                aircc_options += ["--xclbin-kernel-id"]
                aircc_options += [self.kernel_id]
            if self.xclbin_input != "":
                aircc_options += ["--xclbin-input"]
                aircc_options += [self.xclbin_input]
            if peano_package_dir != "":
                aircc_options += ["--peano"]
                aircc_options += [peano_package_dir]
                aircc_options += ["--no-xchesscc"]
                aircc_options += ["--no-xbridge"]
            else:
                aircc_options += ["--xchesscc"]
                aircc_options += ["--xbridge"]

            aircc.run(air_module, aircc_options)

        return XRTCompileArtifact(xclbin, kernel, insts)

    def compile_from_torch_mlir(
        self,
        imported_module,
        pipeline=None,
        verbose=False,
    ):
        import torch_mlir
        import torch_mlir.passmanager

        if type(imported_module) is torch_mlir.ir.Module:
            with imported_module.operation.context:
                pm = torch_mlir.passmanager.PassManager.parse(
                    "builtin.module(refback-mlprogram-bufferize)"
                )
                pm.run(imported_module.operation)

        with air.ir.Context():
            linalg_module = air.ir.Module.parse(str(imported_module))
            pm = air.passmanager.PassManager.parse(
                air.compiler.util.LINALG_TENSOR_TO_MEMREF_PIPELINE
            )
            if verbose:
                print(
                    "Running MLIR pass pipeline: ",
                    air.compiler.util.LINALG_TENSOR_TO_MEMREF_PIPELINE,
                )
            pm.run(linalg_module.operation)

            if verbose:
                print("Linalg Module:")
                print(linalg_module)

            DEFAULT_PIPELINE = (
                "builtin.module("
                + ",".join(
                    [
                        "buffer-results-to-out-params",
                        "air-linalg-codegen",
                        "air-par-to-herd{depth=-1}",
                        "air-par-to-launch{has-air-segment=true}",
                        "air-copy-to-dma",
                        "canonicalize",
                        "cse",
                    ]
                )
                + ")"
            )
            if pipeline is None:
                pipeline = DEFAULT_PIPELINE

            if callable(pipeline):
                air_module = pipeline(linalg_module)
            else:
                pm = air.passmanager.PassManager.parse(pipeline)
                pm.run(linalg_module.operation)
                air_module = linalg_module

            if verbose:
                print("Air Module:")
                print(air_module)

        return self.compile(air_module)

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
        with open(artifact.insts, "rb") as f:
            instr_data = f.read()
            self.instr_v = np.frombuffer(instr_data, dtype=np.uint32)

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
