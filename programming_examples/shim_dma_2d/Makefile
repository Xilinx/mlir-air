# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

all: build/final.xclbin

# Determine build dir based on whether PEANO_INSTALL_DIR is set
ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

build/air.mlir: ${srcdir}/${targetname}.py
	mkdir -p ${@D}
	python3 $< > $@

build/final.xclbin: build/air.mlir
	mkdir -p ${@D}
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && aircc.py -xbridge -o ${@F} --tmpdir tmp --device npu1_1col --host-target x86_64 $(<:%=../%)

build/final.py.xclbin: build/air.mlir
	mkdir -p ${@D}
	PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR) cd $(BUILD_DIR) && python3 build.py $(<:%=../%)

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

run: ${targetname}.exe build/final.xclbin build/final.insts.bin 
	${powershell} ./$< -x build/final.xclbin -i build/final.insts.bin -k MLIR_AIE

run_py: build/final.xclbin build/final.insts.bin
	${powershell} python3 ${srcdir}/test.py build/final.xclbin build/final.insts.bin

pyworkflow:
	mkdir -p pybuild
	cd pybuild && ${powershell} python3 ${srcdir}/run.py

clean:
	rm -rf build _build pybuild tmp ${targetname}.exe