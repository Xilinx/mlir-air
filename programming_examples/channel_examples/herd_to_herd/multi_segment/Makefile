# (c) Copyright 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

targetname := $(shell basename ${srcdir})

all: run

print:
	${powershell} python3 ${srcdir}/herd_to_herd.py -p

run:
	mkdir -p build
	cd build && ${powershell} python3 ${srcdir}/herd_to_herd.py

clean:
	rm -rf build __pycache__
