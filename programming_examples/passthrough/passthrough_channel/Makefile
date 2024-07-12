# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

targetname := $(shell basename ${srcdir})

all: run

run:
	mkdir -p build
	cd build && ${powershell} python3 ${srcdir}/run.py

clean:
	rm -rf ${srcdir}/build ${srcdir}/__pycache__
