# ./python/air/backend/xrt_runner.py -*- Python -*-
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import numpy as np
from .xrt import XRTBackend
import filelock
from typing import List


class XRTRunner:
    def __init__(
        self,
        verbose: bool = False,
        experimental_passes: bool = True,
        omit_while_true_loop: bool = True,
    ):
        self.verbose = verbose
        self.experimental_passes = experimental_passes
        self.omit_while_true_loop = omit_while_true_loop

    def run_test(
        self,
        mlir_module: np.ndarray,
        inputs: List[np.ndarray],
        expected_outputs: List[np.ndarray],
    ):
        if self.verbose:
            print("Running module: ")
            print(mlir_module)

        backend = XRTBackend(
            verbose=self.verbose,
            experimental_passes=self.experimental_passes,
            omit_while_true_loop=self.omit_while_true_loop,
        )

        # run the module - slots are input/output for now, assume non-overlapping inputs/outputs
        inputs += [np.empty((0, 0)) for _o in expected_outputs]
        with filelock.FileLock("/tmp/npu.lock"):
            module_function = backend.compile_and_load(mlir_module)
            actual_outputs = module_function(*inputs)

        backend.unload()

        # Remove input slots from the received outputs
        actual_outputs = actual_outputs[len(inputs) :]

        if self._check_outputs(actual_outputs, expected_outputs):
            print("PASS!")
            return_code = 0
        else:
            print("failed.")
            return_code = -1
        return return_code

    def _check_outputs(
        self, actual_outputs: List[np.ndarray], expected_outputs: List[np.ndarray]
    ):
        assert (
            len(actual_outputs) == len(expected_outputs),
            "Number of actual outputs does not equal number of expected outputs",
        )

        for i, (actual, expected) in enumerate(zip(actual_outputs, expected_outputs)):

            # TODO: may need to reshape??
            assert (
                actual.size() == expected.size(),
                f"Actual output size {actual.size()} does not meet expected output size {expected.size()}",
            )

            if not np.ndarray.array_equal(actual, expected):
                print(f"ERROR: Output {i} does not meet expected output.")
                print("Expected: ")
                if len(expected.size()) == 2:
                    print(np.asmatrix(expected))
                else:
                    print(expected)
                print("Actual: ")
                if len(actual.size()) == 2:
                    print(np.asmatrix(actual))
                else:
                    print(actual)
                return False
        return True
