srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

targetname := $(shell basename ${srcdir})

include ${srcdir}/../makefile-common

targetname = passThroughKernel
VPATH := ${srcdir}/../../mlir-aie/aie_kernels/generic
data_size = 4096
PASSTHROUGH_SIZE = ${data_size}

all: run

build/passThrough.cc.o: passThrough.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -DBIT_WIDTH=8 -c $< -o ${@F}

run: build/passThrough.cc.o
	mkdir -p build
	cd build && ${powershell} python3 ${srcdir}/run.py

clean:
	rm -rf build __pycache__
