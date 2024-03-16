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
import tempfile
import os

__all__ = [
    "CostModel",
    "LINALG_TENSOR_TO_MEMREF_PIPELINE",
    "run_transform"
]

LINALG_TENSOR_TO_MEMREF_PIPELINE = "builtin.module(" + ",".join([
    # Bufferize.
    "func.func(scf-bufferize)",
    "func.func(linalg-bufferize)", "cse",
    "func-bufferize",
    "arith-bufferize",
    "func.func(tensor-bufferize)",
    "func.func(finalizing-bufferize)",
    "canonicalize",
    "cse"
]) + ")"

def _convert_module(module):
    if not isinstance(module, air.ir.Module):
        air_module = air.ir.Module.parse(str(module),air.ir.Context())
    else:
        air_module = module
    return air_module

class CostModel:
    def __init__(self):
        pass

    def op_stats(self, module):
        """Return operation count information as JSON"""
        air_module = _convert_module(module)
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            name = tmpfile.name
            with air_module.context:
                pipeline = f"builtin.module(air-linalg-op-stats{{outputfile={name}}})"
                pm = air.passmanager.PassManager.parse(pipeline)
                pm.run(air_module.operation)
            stats = open(name).read()
            os.unlink(name)
        return stats

class Runner:
    def __init__(self, json_model, trace_filename=None, sim_granularity="herd", verbose=False):
        self.json_model = json_model
        self.trace_filename = trace_filename
        self.sim_granularity = sim_granularity
        self.verbose = verbose

    def run(self, module, function):
        air_module = _convert_module(module)

        trace_tmpfile = None
        trace_filename = self.trace_filename
        if trace_filename is None:
            trace_tmpfile = tempfile.NamedTemporaryFile(delete=False)
            trace_filename = trace_tmpfile.name
        
        # the json model can be:
        #  1. json in string form
        #  2. json in python object form
        #  3. the name of a file containing (1)
        json_model = self.json_model
        if type(json_model) == str:
            if '.json' in json_model:
                with open(json_model) as f:
                    json_model = json.loads(f.read())
            else:
                json_model = json.loads(json_model)

        json_tmpfile = tempfile.NamedTemporaryFile(delete=False)
        json_tmpfile.write(str.encode(json.dumps(json_model)))
        json_tmpfile.close()

        runner.run(air_module, json_tmpfile.name, trace_filename, function, self.sim_granularity, self.verbose)

        os.unlink(json_tmpfile.name)

        # return the trace and remove the temporary file
        # if the user didn't provide an output filename
        return_trace = None
        if trace_tmpfile:
            return_trace = open(trace_tmpfile.name).read()
            os.unlink(trace_tmpfile.name)

        return return_trace

