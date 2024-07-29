# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

all: ${srcdir}/build/final.xclbin

targetname := $(shell basename ${srcdir})

${srcdir}/build/air.mlir: ${srcdir}/${targetname}.py
	mkdir -p ${@D}
	python3 $< > $@

${srcdir}/build/final.xclbin: ${srcdir}/build/air.mlir
	mkdir -p ${@D}
	cd ${srcdir}/build && aircc.py -xbridge -o ${@F} --tmpdir tmp --device npu1_1col --host-target x86_64 --experimental-passes ${srcdir}/build/air.mlir

${srcdir}/build/final.py.xclbin: ${srcdir}/build/air.mlir
	mkdir -p ${@D}
	cd ${srcdir}/build && python3 build.py ${srcdir}/build/air.mlir

${srcdir}/${targetname}.exe: ${srcdir}/test.cpp
	rm -rf ${srcdir}/_build
	mkdir -p ${srcdir}/_build
	cd ${srcdir}/_build && ${powershell} cmake -E env CXXFLAGS="-std=c++23 -ggdb" cmake ${srcdir} -D CMAKE_C_COMPILER=gcc-13 -D CMAKE_CXX_COMPILER=g++-13 -DTARGET_NAME=${targetname} 
	cd ${srcdir}/_build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp ${srcdir}/_build/${targetname}.exe $@
else
	cp ${srcdir}/_build/${targetname} $@ 
endif

run: ${srcdir}/${targetname}.exe ${srcdir}/build/final.xclbin ${srcdir}/build/final.insts.txt 
	${powershell} $< -x build/final.xclbin -i build/final.insts.txt -k MLIR_AIE

run_py: ${srcdir}/build/final.xclbin ${srcdir}/build/final.insts.txt 
	${powershell} python3 ${srcdir}/test.py ${srcdir}/build/final.xclbin ${srcdir}/build/final.insts.txt 

pyworkflow:
	mkdir -p ${srcdir}/pybuild
	cd ${srcdir}/pybuild && ${powershell} python3 ${srcdir}/run.py 

clean:
	rm -rf ${srcdir}/build ${srcdir}/_build ${srcdir}/pybuild tmp ${srcdir}/${targetname}.exe
