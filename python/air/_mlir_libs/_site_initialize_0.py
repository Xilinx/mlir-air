# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Register AIR dialects and transform dialect extension on context init.
# The MLIR Python _site_initialize mechanism discovers _site_initialize_N
# modules and calls their register_dialects(registry) function and/or
# context_init_hook(context) after construction.
#
# We use context_init_hook instead of register_dialects because the _air
# nanobind extension references mlir.ir.Type (via nanobind_adaptors)
# which is not yet initialized when _site_initialize runs — importing
# _air at that point triggers a circular import.  The hook runs after
# Context construction when _air can be safely imported.


def context_init_hook(context):
    from ._air import register_dialect
    from ._mlir.ir import DialectRegistry

    registry = DialectRegistry()
    register_dialect(registry)
    context.append_dialect_registry(registry)
