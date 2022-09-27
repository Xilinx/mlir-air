# Copyright (C) 2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

AIE_SRC_DIR = acdc_project
ARM_SRC_DIR = .
ARM_OBJ_DIR = .

AIE_OPT = aie-opt
AIE_XLATE = aie-translate
AIR_OPT = air-opt
ATEN_OPT = aten-opt
AIECC = aiecc.py

AIR_INSTALL_PATH = $(dir $(shell which air-opt))/..
AIE_MLIR_PATH = ../../../../aie

LDFLAGS = -fuse-ld=lld \
		-rdynamic \
		-lxaiengine \
		-Wl,--whole-archive -lairhost -Wl,--no-whole-archive \
		-lstdc++ \
		-ldl

uname_p := $(shell uname -p)
ifeq ($(uname_p), aarch64)
	CC = clang
	CFLAGS += -g -I/opt/xaienginev2/include
	LDFLAGS += -L/opt/xaienginev2/lib
else
	SYSROOT = ../../../../DockerArm/sysroot
	CC = clang
	CFLAGS += --target=aarch64-linux-gnu --sysroot=$(SYSROOT) -g 
	CFLAGS += -I$(SYSROOT)/opt/xaienginev2/include
	LDFLAGS += --target=aarch64-linux-gnu --sysroot=$(SYSROOT) -L$(SYSROOT)/opt/xaienginev2/lib
endif

CFLAGS += -std=c++17 \
		-I$(AIR_INSTALL_PATH)/runtime_lib/airhost/include \
		-I${AIE_MLIR_PATH}/runtime_lib \
		-DAIR_LIBXAIE_ENABLE \
		-DLIBXAIENGINEV2

LDFLAGS += -L$(AIR_INSTALL_PATH)/runtime_lib/airhost

default: all

test_library.o: ${AIE_MLIR_PATH}/runtime_lib/test_library.cpp
	$(CC) $^ $(CFLAGS) -c -o $@
