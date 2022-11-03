# Copyright (C) 2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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
