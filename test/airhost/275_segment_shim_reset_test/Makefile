# Copyright (C) 2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

ACDC_AIE = $(dir $(shell which aie-opt))/..
ACDC_AIR = $(dir $(shell which air-opt))/..

all: test.elf

test.elf: aie.mlir test.cpp
	aiecc.py -v --host-target=x86_64-amd-linux-gnu --sysroot= $< \
						-I/opt/xaiengine/include \
						-I$(ACDC_AIR)/runtime_lib/airhost/include \
						-I$(ACDC_AIE)/runtime_lib/x86_64/test_lib/include \
            -I${ROCM_ROOT}/include \
						-L/opt/xaiengine/lib \
						-L$(ACDC_AIR)/runtime_lib/airhost \
						test.cpp \
						$(ACDC_AIE)/runtime_lib/x86_64/test_lib/src/test_library.cpp \
						../../elfutils/libelf/libelf.a \
            ${ROCM_ROOT}/lib/libhsa-runtime64.so.1.9.0 \
						-Wl,-R/opt/xaiengine/lib \
            -Wl,-rpath,${ROCM_ROOT}/lib \
						-Wl,--whole-archive -lairhost -Wl,--no-whole-archive -lstdc++ -ldl -o $@

clean::
	rm -rf acdc_project *.elf core* aie.mlir.prj
