# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

targetname := $(shell basename ${srcdir})

all: run

print:
	${powershell} python3 ${srcdir}/single_core.py -p

run:
	mkdir -p build
	cd build && ${powershell} xchesscc_wrapper aie2 -c ${srcdir}/vm.cc -o ${srcdir}/vm.o
	${powershell} python3 ${srcdir}/single_core.py

clean:
	rm -rf build __pycache__
