##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

targetname := $(shell basename ${srcdir})

channel:
	mkdir -p build
	cd build && ${powershell} python3 ${srcdir}/run.py channel

launch:
	mkdir -p build
	cd build && ${powershell} python3 ${srcdir}/run.py launch

herd:
	mkdir -p build
	cd build && ${powershell} python3 ${srcdir}/run.py herd

segment:
	mkdir -p build
	cd build && ${powershell} python3 ${srcdir}/run.py segment


clean:
	rm -rf build
