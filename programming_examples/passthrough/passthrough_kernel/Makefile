# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

targetname := $(shell basename ${srcdir})

include ${srcdir}/../../makefile-common

targetname = passThroughKernel

all: run

print:
	${powershell} python3 ${srcdir}/passthrough_kernel.py -p

${srcdir}/build/passThrough.cc.o: ${srcdir}/passThrough.cc
	mkdir -p ${srcdir}/build
	cd ${srcdir}/build && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -DBIT_WIDTH=8 -c $< -o ${@F}

run: ${srcdir}/build/passThrough.cc.o
	mkdir -p ${srcdir}/build
	cd ${srcdir}/build && ${powershell} python3 ${srcdir}/passthrough_kernel.py

clean:
	rm -rf ${srcdir}/build ${srcdir}/__pycache__