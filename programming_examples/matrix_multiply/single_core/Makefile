##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

all: pyworkflow

targetname := $(shell basename ${srcdir})

pyworkflow:
	mkdir -p build
	cd build && ${powershell} python3 ${srcdir}/run.py 

clean:
	rm -rf build
