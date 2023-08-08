#!/usr/bin/env bash

##===- utils/github-clone-build-elfutils.sh ------------------*- Script -*-===##
#
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

##===----------------------------------------------------------------------===##
#
# This script checks out and builds libelf.
# It only build the necessary libraries to minimize build time.
#
# This script is intended to be called from the github workflows.
#
# Depends on: autoconf, flex, bison, gawk
##===----------------------------------------------------------------------===##

INSTALL_DIR=elfutils
HASH="airbin"

if [[ ! -d $INSTALL_DIR ]]; then
  git clone --branch $HASH --depth 1 https://github.com/jnider/elfutils.git $INSTALL_DIR
fi

cd $INSTALL_DIR
autoreconf -v -f -i
./configure --program-prefix="air-" --disable-debuginfod --disable-libdebuginfod --enable-maintainer-mode


# build libeu.a, required for libelf.so
make -C lib

# build libelf.a, libelf_pic.a and libelf.so
make -C libelf
