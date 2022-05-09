
import air.mlir.ir
import air.mlir.passmanager
import air.mlir.all_passes_registration
import air.mlir._mlir_libs._airMlir
import air.mlir._mlir_libs._airMlir.runner as runner

import tempfile
import os

__all__ = [
    "CostModel",
    "LINALG_TENSOR_TO_MEMREF_PIPELINE"
]

LINALG_TENSOR_TO_MEMREF_PIPELINE = ",".join([
    # Bufferize.
    "func.func(scf-bufferize)",
    "func.func(linalg-bufferize)",
    "func-bufferize",
    "arith-bufferize",
    "func.func(tensor-bufferize)",
    "func.func(finalizing-bufferize)",
    "canonicalize",
    "cse"
])

def _convert_module(module):
    if not isinstance(module, air.mlir.ir.Module):
        air_module = air.mlir.ir.Module.parse(str(module),air.mlir.ir.Context())
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
                pipeline = f"air-linalg-op-stats{{outputfile={name}}}"
                pm = air.mlir.passmanager.PassManager.parse(pipeline)
                pm.run(air_module)
            stats = open(name).read()
            os.unlink(name)
        return stats


class Runner:
    def __init__(self, json_filename, trace_filename=None, verbose=False):
        self.json_filename = json_filename
        self.trace_filename = trace_filename
        self.verbose = verbose

    def run(self, module, function):
        air_module = _convert_module(module)
        tmpfile = None
        trace_filename = self.trace_filename
        if trace_filename is None:
            tmpfile = tempfile.NamedTemporaryFile(delete=False)
            trace_filename = tmpfile.name
        runner.run(air_module, self.json_filename, trace_filename, function, self.verbose)
        if tmpfile:
            trace = open(trace_filename).read()
            os.unlink(trace_filename)
            return trace
        return None
