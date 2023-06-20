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

HASH="vck5000"

git clone --branch $HASH --depth 1 https://github.com/jnider/aie-rt.git $LIBXAIE_DIR

mkdir -p $LIBXAIE_DIR/$INSTALL_DIR/lib

pushd $LIBXAIE_DIR/drivers/src/
make -f Makefile.Linux CFLAGS="-D__AIELINUX__ -D__AIESYSFS__ -D__AIEAMDAIR__"
popd

cp -v $LIBXAIE_DIR/drivers/src/*.so* $LIBXAIE_DIR/$INSTALL_DIR/lib
cp -vr $LIBXAIE_DIR/drivers/include $LIBXAIE_DIR/$INSTALL_DIR
