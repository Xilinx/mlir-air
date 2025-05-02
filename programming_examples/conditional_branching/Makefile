# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

targetname := $(shell basename ${srcdir})

all: run0 run1

print:
	${powershell} python3 ${srcdir}/single_core.py -p

run0:
	mkdir -p build
	cd build && ${powershell} python3 ${srcdir}/single_core.py --param 0

run1:
	mkdir -p build
	cd build && ${powershell} python3 ${srcdir}/single_core.py --param 1

clean:
	rm -rf build __pycache__
