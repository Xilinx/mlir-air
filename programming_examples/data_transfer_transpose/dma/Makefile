# (c) Copyright 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

targetname := $(shell basename ${srcdir})

M ?= 64
K ?= 32

all: run_int run_float

print:
	${powershell} python3 ${srcdir}/transpose.py -p

run_int:
	mkdir -p build
	cd build && ${powershell} python3 ${srcdir}/transpose.py -m ${M} -k ${K} -t uint32

run_float:
	mkdir -p build
	cd build && ${powershell} python3 ${srcdir}/transpose.py -m ${M} -k ${K} -t float32

clean:
	rm -rf build ${srcdir}/__pycache__