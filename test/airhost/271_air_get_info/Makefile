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
		../../elfutils/libelf/libelf.a \

CFLAGS += -g -I/opt/xaiengine/include
CFLAGS += -I${ROCM_ROOT}/include
CFLAGS += -std=c++17 -I$(ACDC_AIR)/runtime_lib/airhost/include -I$(ACDC_AIE)/runtime_lib/x86_64/test_lib/include \
          -DAIR_LIBXAIE_ENABLE
LDFLAGS += -L/opt/xaiengine/lib -Wl,-R/opt/xaiengine/lib
LDFLAGS += -L$(ACDC_AIR)/runtime_lib/airhost

all: test.exe

%.exe: test.cpp
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(ACDC_AIE)/runtime_lib/x86_64/test_lib/src/test_library.cpp test.cpp -Wl,--whole-archive -Wl,--no-whole-archive


clean::
	rm -rf *.exe 
