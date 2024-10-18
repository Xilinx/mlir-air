# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

targetname := $(shell basename ${srcdir})

all: run

print:
	${powershell} python3 ${srcdir}/single_core_dma.py -p

run:
	mkdir -p build
	cd build && ${powershell} python3 ${srcdir}/single_core_dma.py

clean:
	rm -rf build ${srcdir}/__pycache__
