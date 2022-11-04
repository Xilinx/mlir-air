# Copyright (C) 2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

.PHONY: all
all: test run

test.elf:  
	aiecc.py --sysroot= aie.mlir -I${ACDC}/runtime_lib/airhost/include -I${ACDC}/runtime_lib -I${ACDC}/runtime_lib/test_library.cpp -L${ACDC}/runtime_lib/airhost test.cpp -Wl,--whole-archive -lairhost -Wl,--no-whole-archive -lstdc++ -o test.elf

.PHONY: test
test: test.elf  

run: test.elf
	sudo ./test.elf

clean::
	rm -rf acdc_project *.elf aie_inc.cpp input*.mlir
	rm -rf test.exe

