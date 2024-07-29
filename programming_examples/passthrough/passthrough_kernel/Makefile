# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

targetname := $(shell basename ${srcdir})

include ${srcdir}/../../makefile-common

# mlir-aie dir should be one up from the install dir, if it is defined
# if it is not defined, assume it's been cloned in ../mlir-aie
ifeq (${MLIR_AIE_INSTALL_DIR},)
	MLIR_AIE_DIR := $(shell realpath $(srcdir)/../../../mlir-aie)
else
	MLIR_AIE_DIR := $(shell realpath ${MLIR_AIE_INSTALL_DIR}/../)
endif

targetname = passThroughKernel
VPATH := ${MLIR_AIE_DIR}/aie_kernels/generic

all: run

print:
	${powershell} python3 ${srcdir}/passthrough_kernel.py -p

${srcdir}/build/passThrough.cc.o: ${VPATH}/passThrough.cc
	mkdir -p ${srcdir}/build
	cd ${srcdir}/build && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -DBIT_WIDTH=8 -c $< -o ${@F}

run: ${srcdir}/build/passThrough.cc.o
	mkdir -p ${srcdir}/build
	cd ${srcdir}/build && ${powershell} python3 ${srcdir}/passthrough_kernel.py

clean:
	rm -rf ${srcdir}/build ${srcdir}/__pycache__