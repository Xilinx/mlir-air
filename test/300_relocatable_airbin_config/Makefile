# Copyright (C) 2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

ACDC_AIE = $(dir $(shell which aie-opt))/..
ACDC_AIR = $(dir $(shell which air-opt))/..

.PHONY: all
all: test

LDFLAGS = -fuse-ld=lld -rdynamic \
    -DLIBXAIENGINEV2 \
    -lxaiengine \
    -Wl,--whole-archive -lairhost -Wl,--no-whole-archive \
    -lstdc++ \
    -lm \
    -Wl,--no-as-needed -ldl

CFLAGS += -g -I/opt/xaiengine/include
CFLAGS += -std=c++17 -I$(ACDC_AIR)/runtime_lib/airhost/include -I$(ACDC_AIR)/runtime_lib \
          -DAIR_LIBXAIE_ENABLE
LDFLAGS += -L/opt/xaiengine/lib -Wl,-R/opt/xaiengine/lib
LDFLAGS += -L$(ACDC_AIR)/runtime_lib/airhost

test.exe: test.cpp
	clang++ $< ${CFLAGS} ${LDFLAGS} -o $@

.PHONY: test
test: test.exe

run: test.exe
	sudo ./test.exe

clean::
	rm -rf test.exe

