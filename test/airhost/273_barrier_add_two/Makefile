# Copyright (C) 2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

ACDC_AIE = $(dir $(shell which aie-opt))/..
ACDC_AIR = $(dir $(shell which air-opt))/..

all: test.elf

test.elf: aie.mlir test.cpp
	aiecc.py -v --host-target=x86_64-amd-linux-gnu --sysroot= $< -I/opt/xaiengine/include -L/opt/xaiengine/lib -Wl,-R/opt/xaiengine/lib \
						-I$(ACDC_AIR)/runtime_lib/airhost/include -I$(ACDC_AIE)/runtime_lib/x86_64/test_lib/include $(ACDC_AIE)/runtime_lib/x86_64/test_lib/src/test_library.cpp \
						-L$(ACDC_AIR)/runtime_lib/airhost test.cpp -Wl,--whole-archive -lairhost -Wl,--no-whole-archive -lstdc++ -ldl -o $@ 

clean::
	rm -rf acdc_project *.elf
