# ./python/air/dialects/air.py -*- Python -*-

# Copyright (C) 2020-2022, Xilinx Inc.
# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from ._air_enum_gen import *
from ._air_ops_gen import *
from ._air_ops_ext import *
from ._air_transform_ops_gen import *
from ._air_transform_ops_ext import *
from .._mlir_libs import get_dialect_registry
from .._mlir_libs._air import *

register_dialect(get_dialect_registry())
