# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

targetname := $(shell basename ${srcdir})

run:
	mkdir -p build
	cd build && ${powershell} python3 ${srcdir}/run.py

clean:
	rm -rf build __pycache__
