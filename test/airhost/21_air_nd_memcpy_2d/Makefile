# Copyright (C) 2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

CC=clang
ACDC_AIE = $(dir $(shell which aie-opt))/..
ACDC_AIR = $(dir $(shell which air-opt))/..

LDFLAGS = -fuse-ld=lld -rdynamic \
	  -DLIBXAIENGINEV2 \
		-lxaiengine \
		-Wl,--whole-archive -lairhost -Wl,--no-whole-archive \
		-lstdc++ \
		-lm \
		-Wl,--no-as-needed -ldl \
    -Wl,-rpath,${ROCM_ROOT}/lib ${ROCM_ROOT}/lib/libhsa-runtime64.so.1.9.0 \
		-lelf \

CFLAGS += -g -I/opt/xaiengine/include
CFLAGS += -I${ROCM_ROOT}/include
CFLAGS += -std=c++17 -I$(ACDC_AIR)/runtime_lib/airhost/include -I$(ACDC_AIE)/runtime_lib/x86_64/test_lib/include \
          -DAIR_LIBXAIE_ENABLE
LDFLAGS += -L/opt/xaiengine/lib -Wl,-R/opt/xaiengine/lib
LDFLAGS += -L$(ACDC_AIR)/runtime_lib/airhost
LDFLAGS += -L/lib/x86_64-linux-gnu/

all: herd.exe

%.exe: 2d_nd.air.a test.cpp
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(ACDC_AIE)/runtime_lib/x86_64/test_lib/src/test_library.cpp test.cpp -Wl,--whole-archive 2d_nd.air.a -Wl,--no-whole-archive

2d_nd.air.a: air.mlir
	aircc.py -v -xchesscc -xbridge --sysroot= --host-target=x86_64-amd-linux-gnu -row-offset=4 -col-offset=5 $< -o $@

clean::
	rm -rf air_project *.elf *.exe *.o *.a segment*
