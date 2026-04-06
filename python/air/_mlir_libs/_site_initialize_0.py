# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Register AIR dialects and transform dialect extension on context init.
# The MLIR Python _site_initialize mechanism discovers _site_initialize_N
# modules and calls their register_dialects(registry) function and/or
# context_init_hook(context) after construction.
#
# We use context_init_hook because the _air nanobind extension cannot be
# imported during _site_initialize (circular import: _air links
# AirAggregateCAPI which requires MLIR Python types not yet initialized).
# The hook runs after the Context is constructed, when _air can be loaded.


def context_init_hook(context):
    from ._air import register_dialect
    from ._mlir.ir import DialectRegistry

    registry = DialectRegistry()
    register_dialect(registry)
    context.append_dialect_registry(registry)
