# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

targetname := $(shell basename ${srcdir})

all: run

print:
	${powershell} python3 ${srcdir}/single_core.py -p

run:
	mkdir -p build
	cd build && ${powershell} python3 ${srcdir}/single_core.py

clean:
	rm -rf build __pycache__
