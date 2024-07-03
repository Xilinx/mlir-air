# (c) Copyright 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

targetname := $(shell basename ${srcdir})

M ?= 64
K ?= 32

run:
	mkdir -p build
	cd build && ${powershell} python3 ${srcdir}/run.py -m ${M} -k ${K}

clean:
	rm -rf build __pycache__