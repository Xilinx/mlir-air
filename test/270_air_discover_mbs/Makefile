# Copyright (C) 2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

.PHONY: all
all: test run

test.exe:  
	clang++ test.cpp -fPIC -ldl -I${ACDC}/runtime_lib/airhost/include -I/opt/xaienginev2/include -I${ACDC}/runtime_lib ${ACDC}/runtime_lib/test_library.cpp -L${ACDC}/runtime_lib/airhost -L/opt/xaienginev2/lib -lxaiengine -Wl,--whole-archive -lairhost -Wl,--no-whole-archive -o test.exe

.PHONY: test
test: test.exe 

run: test.exe
	sudo ./test.exe

clean::
	rm -rf test.exe

