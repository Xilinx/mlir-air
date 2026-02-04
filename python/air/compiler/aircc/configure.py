#
# SPDX-License-Identifier: MIT
#
# (c) Copyright 2023 Advanced Micro Devices Inc.

import os

# Default configuration for GPU-only builds
# These are overridden by the cmake-generated version during install
air_link_with_xchesscc = False
air_compile_with_xchesscc = False
libxaie_path = ""
rocm_path = ""

def install_path():
    path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(path, '..', '..', '..', '..')
    return os.path.realpath(path)
