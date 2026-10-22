# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Stream one prefill weight set and bind shared builders to NPU2."""

import json
import shutil
from pathlib import Path
import sys
import types
from paths import EXAMPLES

sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(EXAMPLES)
]
from air_examples.llms.shared.infra.cache import KernelCache


class NPU2Cache(KernelCache):
    """Bind this example's shared prefill builders to the actual NPU2 target."""

    @staticmethod
    def options(options):
        if options.get("target_device", "npu2") != "npu2":
            raise ValueError("8B prefill requires the NPU2 target")
        return dict(options, target_device="npu2")

    def compile_and_cache(
        self, name, mlir_module, backend_kwargs, output_binary_name="air"
    ):
        options = self.options(backend_kwargs)
        saved = self.cache_dir / "builds" / name
        saved.mkdir(parents=True, exist_ok=False)
        (saved / "input.mlir").write_text(str(mlir_module))
        (saved / "backend.json").write_text(json.dumps(options, indent=2) + "\n")
        try:
            return super().compile_and_cache(
                name, mlir_module, options, output_binary_name
            )
        finally:
            if Path("air_project").is_dir():
                shutil.copytree("air_project", saved / "air_project")

    def load_and_run(self, name, backend_kwargs, *inputs, **kwargs):
        # Layer weights change even when shapes match. Keep one BO set per
        # kernel and explicitly upload weights on every call, rather than
        # keeping 32 resident copies. Device-overwritten scratch keeps its
        # original initialization semantics in the shared cache implementation.
        kwargs["bo_key"] = name + "_streamed"
        kwargs["static_input_indices"] = set()
        return super().load_and_run(
            name, self.options(backend_kwargs), *inputs, **kwargs
        )
