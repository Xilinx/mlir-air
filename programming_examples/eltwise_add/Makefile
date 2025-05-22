# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

targetname := $(shell basename ${srcdir})

all: run

print:
	${powershell} python3 ${srcdir}/eltwise_add.py -p

run:
	mkdir -p build
	cd build && ${powershell} python3 ${srcdir}/eltwise_add.py --compiler_backend chess

clean:
	rm -rf build __pycache__
