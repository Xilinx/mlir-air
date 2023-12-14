#!/usr/bin/env bash

##===- utils/github-clone-build-libxaie.sh ------------------*- Script -*-===##
#
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

##===----------------------------------------------------------------------===##
#
# This script checks out and builds libxaiev2.
#
# This script is intended to be called from the github workflows.
#
##===----------------------------------------------------------------------===##

LIBXAIE_DIR="aienginev2"
INSTALL_DIR="install"

HASH="joel-aie"

git clone --branch $HASH --depth 1 https://github.com/jnider/aie-rt.git $LIBXAIE_DIR

mkdir -p $LIBXAIE_DIR/$INSTALL_DIR/lib

pushd $LIBXAIE_DIR/driver/src/
make -f Makefile.Linux CFLAGS="-D__AIELINUX__ -D__AIEAMDAIR__"
popd

cp -v $LIBXAIE_DIR/driver/src/*.so* $LIBXAIE_DIR/$INSTALL_DIR/lib
cp -vr $LIBXAIE_DIR/driver/include $LIBXAIE_DIR/$INSTALL_DIR
