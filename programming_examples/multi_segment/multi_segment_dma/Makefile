# (c) Copyright 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

targetname := $(shell basename ${srcdir})

all: run

print:
	${powershell} python3 ${srcdir}/multi_segment.py -p

run:
	mkdir -p ${srcdir}/build
	cd ${srcdir}/build && ${powershell} python3 ${srcdir}/multi_segment.py

clean:
	rm -rf ${srcdir}/build ${srcdir}/__pycache__