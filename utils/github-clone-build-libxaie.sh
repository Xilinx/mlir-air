#!/usr/bin/env bash

##===- utils/github-clone-build-libxaie.sh ------------------*- Script -*-===##
#
# Copyright (C) 2022, Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

##===----------------------------------------------------------------------===##
#
# This script checks out and builds libxaiev2.
#
# This script is intended to be called from the github workflows.
#
##===----------------------------------------------------------------------===##

LIBXAIE_DIR="aienginev2"
INSTALL_DIR="install"

HASH="xlnx_rel_v2021.2"

git clone --branch $HASH --depth 1 https://github.com/Xilinx/embeddedsw.git $LIBXAIE_DIR

mkdir -p $LIBXAIE_DIR/$INSTALL_DIR/lib

pushd $LIBXAIE_DIR/XilinxProcessorIPLib/drivers/aienginev2/src/
make -f Makefile.Linux CFLAGS="-D__AIELINUX__"
popd

cp -v $LIBXAIE_DIR/XilinxProcessorIPLib/drivers/aienginev2/src/*.so* $LIBXAIE_DIR/$INSTALL_DIR/lib
cp -vr $LIBXAIE_DIR/XilinxProcessorIPLib/drivers/aienginev2/include $LIBXAIE_DIR/$INSTALL_DIR
