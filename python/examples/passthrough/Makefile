##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

all: build/final.xclbin

targetname = passthrough

build/air.mlir: ${srcdir}/2d_shim_dma.py
	mkdir -p ${@D}
	python3 $< > $@

build/final.xclbin: build/air.mlir
	mkdir -p ${@D}
	cd build && aircc.py -xbridge -o ${@F} --tmpdir tmp --device npu1_1col --host-target x86_64 $(<:%=../%)


${targetname}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake -E env CXXFLAGS="-std=c++23 -ggdb" cmake ${srcdir} -D CMAKE_C_COMPILER=gcc-13 -D CMAKE_CXX_COMPILER=g++-13 -DTARGET_NAME=${targetname} 
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif

run: ${targetname}.exe build/final.xclbin build/final.insts.txt 
	${powershell} ./$< -x build/final.xclbin -i build/final.insts.txt -k MLIR_AIE

clean:
	rm -rf build _build tmp ${targetname}.exe
