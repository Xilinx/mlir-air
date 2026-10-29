# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Inspect supported AIR interfaces without importing builders or opening devices."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INTERFACES = {
    "programming_examples/llms/shared/deployment.py": [
        "spec",
        "configuration",
        "require_bfp16",
    ],
    "programming_examples/llms/shared/deployment_cli.py": [
        "compare_capture",
        "request",
    ],
    "programming_examples/llms/shared/deployment_ci.py": ["performance_record", "main"],
    "programming_examples/llms/shared/prepare_affine_q4.py": ["prepare"],
    "programming_examples/llms/shared/infra/external_kernels.py": ["compile_gemm_mm"],
    "programming_examples/kernel_registry/registry_lookup.py": ["gemm_config"],
    "programming_examples/kernel_registry/evidence.py": [
        "select_verified",
        "validate_record",
    ],
}


def inspect(root):
    rows = []
    for relative, names in INTERFACES.items():
        path = root / relative
        row = dict(path=relative, required_symbols=names, status="MISSING")
        if path.is_file():
            data = path.read_bytes()
            try:
                tree = ast.parse(data)
                found = {
                    n.name
                    for n in tree.body
                    if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                }
                row.update(
                    status="PRESENT" if set(names) <= found else "INTERFACE_CHANGED",
                    missing_symbols=sorted(set(names) - found),
                    sha256=hashlib.sha256(data).hexdigest(),
                )
            except SyntaxError:
                row["status"] = "UNPARSEABLE"
        rows.append(row)
    return dict(
        status=(
            "INTERFACES_PRESENT"
            if all(r["status"] == "PRESENT" for r in rows)
            else "INTEGRATION_REVIEW_REQUIRED"
        ),
        interfaces=rows,
        scope="Source symbol presence only; signatures, toolchain/runtime compatibility and numerical validity need their applicable checks.",
        device_runs=0,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--air-root", type=Path, default=ROOT)
    a = p.parse_args()
    result = inspect(a.air_root)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "INTERFACES_PRESENT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
