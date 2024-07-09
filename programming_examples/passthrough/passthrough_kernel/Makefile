# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

targetname := $(shell basename ${srcdir})

include ${srcdir}/../../makefile-common

targetname = passThroughKernel
VPATH := ${MLIR_AIE_DIR}/aie_kernels/generic

all: run

build/passThrough.cc.o: ${VPATH}/passThrough.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -DBIT_WIDTH=8 -c $< -o ${@F}

run: build/passThrough.cc.o
	mkdir -p build
	cd build && ${powershell} python3 ${srcdir}/run.py

clean:
	rm -rf ${srcdir}/build ${srcdir}/__pycache__
