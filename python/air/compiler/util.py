# ./python/air/compiler/util.py -*- Python -*-

# Copyright (C) 2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import air.ir
import air.passmanager
import air._mlir_libs._air
import air._mlir_libs._air.runner as runner
from air._mlir_libs._air import run_transform as run_transform

import json
import pathlib
import tempfile

__all__ = ["CostModel", "LINALG_TENSOR_TO_MEMREF_PIPELINE", "run_transform"]

LINALG_TENSOR_TO_MEMREF_PIPELINE = (
    "builtin.module("
    + ",".join(
        [
            # Bufferize.
            "one-shot-bufferize{copy-before-write bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map}",
            "cse",
        ]
    )
    + ")"
)


def _convert_module(module):
    if not isinstance(module, air.ir.Module):
        air_module = air.ir.Module.parse(str(module), air.ir.Context())
    else:
        air_module = module
    return air_module


class CostModel:
    def __init__(self):
        pass

    def op_stats(self, module):
        """Return operation count information as JSON"""
        air_module = _convert_module(module)
        # Close the handle before the pass writes to the path and before the
        # unlink. Only POSIX lets you unlink a file that is still open; on
        # Windows both the write and the unlink fail with a sharing violation.
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            name = tmpfile.name
        try:
            with air_module.context:
                pipeline = f"builtin.module(air-linalg-op-stats{{outputfile={name}}})"
                pm = air.passmanager.PassManager.parse(pipeline)
                pm.run(air_module.operation)
            with open(name) as f:
                stats = f.read()
        finally:
            # missing_ok: if the pass raised before writing the file, a
            # FileNotFoundError here would replace the failure worth reporting.
            pathlib.Path(name).unlink(missing_ok=True)
        return stats


class Runner:
    def __init__(
        self,
        json_model,
        trace_filename=None,
        sim_granularity="herd",
        launch_iterations="all",
        verbose=False,
    ):
        self.json_model = json_model
        self.trace_filename = trace_filename
        self.sim_granularity = sim_granularity
        self.launch_iterations = launch_iterations
        self.verbose = verbose

    def run(self, module, function):
        air_module = _convert_module(module)

        trace_tmpfile = None
        trace_filename = self.trace_filename
        if trace_filename is None:
            # Close it immediately: the runner below writes to this path, and
            # Windows refuses both that write and the later unlink while our
            # own handle is still open.
            trace_tmpfile = tempfile.NamedTemporaryFile(delete=False)
            trace_tmpfile.close()
            trace_filename = trace_tmpfile.name

        # the json model can be:
        #  1. json in string form
        #  2. json in python object form
        #  3. the name of a file containing (1)
        json_model = self.json_model
        if type(json_model) == str:
            if ".json" in json_model:
                with open(json_model) as f:
                    json_model = json.loads(f.read())
            else:
                json_model = json.loads(json_model)

        json_tmpfile = tempfile.NamedTemporaryFile(delete=False)
        json_tmpfile.write(str.encode(json.dumps(json_model)))
        json_tmpfile.close()

        return_trace = None
        try:
            runner.run(
                air_module,
                json_tmpfile.name,
                trace_filename,
                function,
                self.sim_granularity,
                self.launch_iterations,
                self.verbose,
            )

            # return the trace if the user didn't provide an output filename
            if trace_tmpfile:
                with open(trace_tmpfile.name) as f:
                    return_trace = f.read()
        finally:
            # Clean up on the failure paths too, and missing_ok throughout: a
            # cleanup error must not replace the failure worth reporting.
            pathlib.Path(json_tmpfile.name).unlink(missing_ok=True)
            if trace_tmpfile:
                pathlib.Path(trace_tmpfile.name).unlink(missing_ok=True)

        return return_trace
